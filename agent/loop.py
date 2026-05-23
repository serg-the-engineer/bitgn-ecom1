from __future__ import annotations

import time
from typing import Any

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from connectrpc.errors import ConnectError

from authorization_guard import validate_authorization_action
from completion_guard import validate_completion
from observability import Recorder, short_id, utc_now
from skill_classifier import render_skill_context, route_skills
from structured_llm import resolve_provider, run_structured_llm
from tool_contract_guard import command_path, command_tool_name, validate_tool_contract

from .formatting import format_result
from .models import AgentRunResult, NextStep, ReportTaskCompletion, Req_Exec, Req_Read, Req_Tree
from .prompt import CLI_BLUE, CLI_CLR, CLI_GREEN, CLI_RED, CLI_YELLOW, SYSTEM_PROMPT
from .runtime import dispatch


def run_agent(
    model: str,
    harness_url: str,
    task_text: str,
    *,
    recorder: Recorder | None = None,
    local_run_id: str | None = None,
    local_trial_id: str | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
    analytics_context: str | None = None,
    classification: Any | None = None,
) -> AgentRunResult:
    provider = resolve_provider()
    vm = EcomRuntimeClientSync(harness_url)
    log = [{"role": "system", "content": SYSTEM_PROMPT}]
    trial_dir = None
    message_index = 0
    tool_history: list[dict[str, Any]] = []
    skill_route = route_skills(task_text=task_text, classification=classification)

    if recorder and task_id and attempt_id:
        attempt_no = int(attempt_id.rsplit("_", 1)[-1]) if attempt_id.rsplit("_", 1)[-1].isdigit() else 1
        trial_dir = recorder.trial_dir(task_id, attempt_no)
        recorder.record_message(
            local_trial_id=local_trial_id or "",
            trial_dir=trial_dir,
            message_index=message_index,
            role="system",
            content=SYSTEM_PROMPT,
        )
        message_index += 1

    def add_message(
        message: dict[str, Any],
        *,
        step_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        nonlocal message_index
        log.append(message)
        if recorder and trial_dir and local_trial_id:
            recorder.record_message(
                local_trial_id=local_trial_id,
                trial_dir=trial_dir,
                message_index=message_index,
                role=message.get("role", ""),
                content=str(message.get("content") or ""),
                step_id=step_id,
                tool_call_id=tool_call_id,
                tool_calls=message.get("tool_calls"),
            )
        message_index += 1

    try:
        _bootstrap_agent(
            vm=vm,
            recorder=recorder,
            local_trial_id=local_trial_id,
            trial_dir=trial_dir,
            task_id=task_id,
            attempt_id=attempt_id,
            add_message=add_message,
        )
    except Exception as exc:
        return _exception_result(exc, recorder, local_trial_id, steps=0)

    if analytics_context:
        add_message(
            {
                "role": "user",
                "content": f"Historical analytics context for this task:\n{analytics_context}",
            }
        )

    skill_context = render_skill_context(skill_route)
    if skill_context:
        add_message(
            {
                "role": "user",
                "content": "Reusable workflow skills for this task:\n" + skill_context,
            }
        )
        _record_skill_route(
            recorder=recorder,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            skill_route=skill_route,
        )

    add_message({"role": "user", "content": task_text})

    for index in range(30):
        step_id = f"step_{index + 1:03d}"
        step_event = None
        started = time.time()
        try:
            if recorder:
                step_event = recorder.start_event(
                    "agent.step.started",
                    step_id,
                    local_trial_id=local_trial_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    step_id=step_id,
                    input_json={"message_count": len(log)},
                )
            job = run_structured_llm(
                provider=provider,
                model=model,
                messages_or_prompt=log,
                response_model=NextStep,
                purpose="agent_next_step",
                recorder=recorder,
                context={
                    "local_run_id": local_run_id,
                    "local_trial_id": local_trial_id,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "step_id": step_id,
                    "trial_dir": trial_dir,
                },
            )
        except Exception as exc:
            _finish_step(
                recorder,
                step_event,
                step_id,
                status="error",
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                error_json={"type": exc.__class__.__name__, "message": str(exc)},
            )
            return _exception_result(exc, recorder, local_trial_id, steps=index)

        elapsed_ms = int((time.time() - started) * 1000)
        print(
            f"Next step_{index + 1} via {provider}... "
            f"{job.plan_remaining_steps_brief[0]} ({elapsed_ms} ms)\n"
            f"  decision: {job.decision_summary}\n"
            f"  {job.function}"
        )

        add_message(_assistant_message(job, step_id), step_id=step_id)

        guard_feedback = _guard_feedback(
            job=job,
            task_text=task_text,
            tool_history=tool_history,
            task_family=skill_route.task_family,
        )
        if guard_feedback:
            _record_guard_block(
                recorder=recorder,
                step_event=step_event,
                step_id=step_id,
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                job=job,
                guard_feedback=guard_feedback,
            )
            add_message(
                {"role": "tool", "content": guard_feedback, "tool_call_id": step_id},
                step_id=step_id,
                tool_call_id=step_id,
            )
            continue

        try:
            result = dispatch(
                vm,
                job.function,
                recorder=recorder,
                local_trial_id=local_trial_id,
                trial_dir=trial_dir,
                task_id=task_id,
                attempt_id=attempt_id,
                step_id=step_id,
            )
            txt = format_result(job.function, result)
            _append_tool_history(tool_history, step_id, job.function, "ok", txt)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ConnectError as exc:
            txt = str(exc.message)
            _append_tool_history(tool_history, step_id, job.function, "error", txt)
            print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")
            if isinstance(job.function, ReportTaskCompletion):
                _finish_step(
                    recorder,
                    step_event,
                    step_id,
                    status="error",
                    local_trial_id=local_trial_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    error_json={"type": "ConnectError", "code": str(exc.code), "message": exc.message},
                )
                return AgentRunResult(
                    steps=index + 1,
                    stopped_reason="exception",
                    exception={"type": "ConnectError", "code": str(exc.code), "message": exc.message},
                    counters=_trial_counters(recorder, local_trial_id),
                )
        except Exception as exc:
            _finish_step(
                recorder,
                step_event,
                step_id,
                status="error",
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                error_json={"type": exc.__class__.__name__, "message": str(exc)},
            )
            return _exception_result(exc, recorder, local_trial_id, steps=index + 1)

        if isinstance(job.function, ReportTaskCompletion):
            return _completion_result(
                recorder=recorder,
                step_event=step_event,
                step_id=step_id,
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                job=job,
                steps=index + 1,
            )

        add_message({"role": "tool", "content": txt, "tool_call_id": step_id}, step_id=step_id, tool_call_id=step_id)
        _finish_step(
            recorder,
            step_event,
            step_id,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            output_json={
                "task_completed": job.task_completed,
                "current_state": job.current_state,
                "decision_summary": job.decision_summary,
                "uncertainty_flags": job.uncertainty_flags,
                "function": job.function.model_dump(),
            },
        )

    if recorder:
        recorder.record_event(
            event_id=short_id("evt"),
            event_type="agent.step.finished",
            name="step_limit",
            status="error",
            ts_start=utc_now(),
            ts_end=utc_now(),
            duration_ms=0,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            error_json={"type": "StepLimit", "message": "Agent reached the 30 step limit"},
        )
    return AgentRunResult(
        steps=30,
        stopped_reason="step_limit",
        exception={"type": "StepLimit", "message": "Agent reached the 30 step limit"},
        counters=_trial_counters(recorder, local_trial_id),
    )


def _bootstrap_agent(
    *,
    vm: EcomRuntimeClientSync,
    recorder: Recorder | None,
    local_trial_id: str | None,
    trial_dir,
    task_id: str | None,
    attempt_id: str | None,
    add_message,
) -> None:
    boot_event = None
    must = [
        Req_Tree(level=2, tool="tree", root="/"),
        Req_Read(path="/AGENTS.MD", tool="read"),
        Req_Exec(path="/bin/date", tool="exec"),
        Req_Exec(path="/bin/id", tool="exec"),
    ]
    try:
        if recorder:
            boot_event = recorder.start_event(
                "agent.bootstrap.started",
                "agent.bootstrap",
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                input_json={"tool_count": len(must)},
            )
        for idx, cmd in enumerate(must, start=1):
            boot_step_id = f"bootstrap_{idx:03d}"
            result = dispatch(
                vm,
                cmd,
                recorder=recorder,
                local_trial_id=local_trial_id,
                trial_dir=trial_dir,
                task_id=task_id,
                attempt_id=attempt_id,
                step_id=boot_step_id,
            )
            formatted = format_result(cmd, result)
            print(f"{CLI_GREEN}AUTO{CLI_CLR}: {formatted}")
            add_message({"role": "user", "content": formatted}, step_id=boot_step_id)
        _finish_bootstrap(
            recorder,
            boot_event,
            local_trial_id,
            task_id,
            attempt_id,
            tool_count=len(must),
        )
    except Exception as exc:
        _finish_bootstrap(
            recorder,
            boot_event,
            local_trial_id,
            task_id,
            attempt_id,
            status="error",
            tool_count=len(must),
            error_json={"type": exc.__class__.__name__, "message": str(exc)},
        )
        raise


def _finish_bootstrap(
    recorder: Recorder | None,
    boot_event: str | None,
    local_trial_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    *,
    status: str = "ok",
    tool_count: int | None = None,
    error_json: dict[str, str] | None = None,
) -> None:
    if not recorder:
        return
    recorder.finish_event(
        boot_event,
        "agent.bootstrap.finished",
        "agent.bootstrap",
        status=status,
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        output_json=None if error_json else {"status": status, "tool_count": tool_count},
        error_json=error_json,
    )


def _record_skill_route(
    *,
    recorder: Recorder | None,
    local_trial_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    skill_route,
) -> None:
    if not recorder:
        return
    recorder.record_event(
        event_id=short_id("evt"),
        event_type="skill.route.selected",
        name=skill_route.primary_skill,
        status="ok",
        ts_start=utc_now(),
        ts_end=utc_now(),
        duration_ms=0,
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        output_json={
            "primary_skill": skill_route.primary_skill,
            "skill_ids": skill_route.skill_ids,
            "task_family": skill_route.task_family,
            "risk_flags": skill_route.risk_flags,
        },
    )


def _assistant_message(job: NextStep, step_id: str) -> dict[str, Any]:
    content = "\n".join(
        item
        for item in (
            job.plan_remaining_steps_brief[0],
            f"Decision: {job.decision_summary}" if job.decision_summary else "",
            f"Uncertainty: {', '.join(job.uncertainty_flags)}" if job.uncertainty_flags else "",
        )
        if item
    )
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "type": "function",
                "id": step_id,
                "function": {
                    "name": job.function.__class__.__name__,
                    "arguments": job.function.model_dump_json(),
                },
            }
        ],
    }


def _guard_feedback(
    *,
    job: NextStep,
    task_text: str,
    tool_history: list[dict[str, Any]],
    task_family: str | None,
) -> str:
    decisions = [
        validate_tool_contract(job.function),
        validate_authorization_action(
            command=job.function,
            task_text=task_text,
            tool_history=tool_history,
        ),
        validate_completion(
            command=job.function,
            task_text=task_text,
            tool_history=tool_history,
            task_family=task_family,
        ),
    ]
    return "\n\n".join(decision.feedback for decision in decisions if not decision.allowed and decision.feedback)


def _record_guard_block(
    *,
    recorder: Recorder | None,
    step_event: str | None,
    step_id: str,
    local_trial_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    job: NextStep,
    guard_feedback: str,
) -> None:
    print(f"{CLI_YELLOW}GUARD{CLI_CLR}: {guard_feedback}")
    if not recorder:
        return
    recorder.record_event(
        event_id=short_id("evt"),
        event_type="guard.blocked",
        name=command_tool_name(job.function),
        status="skipped",
        ts_start=utc_now(),
        ts_end=utc_now(),
        duration_ms=0,
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        step_id=step_id,
        input_json=job.function.model_dump(),
        output_json={"feedback": guard_feedback},
    )
    _finish_step(
        recorder,
        step_event,
        step_id,
        status="skipped",
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        output_json={
            "task_completed": False,
            "current_state": job.current_state,
            "decision_summary": job.decision_summary,
            "guard_feedback": guard_feedback,
            "function": job.function.model_dump(),
        },
    )


def _append_tool_history(
    history: list[dict[str, Any]],
    step_id: str,
    command,
    status: str,
    output_preview: str,
) -> None:
    history.append(
        {
            "step_id": step_id,
            "tool": command_tool_name(command),
            "path": command_path(command),
            "input": command.model_dump(),
            "status": status,
            "output_preview": output_preview,
        }
    )


def _completion_result(
    *,
    recorder: Recorder | None,
    step_event: str | None,
    step_id: str,
    local_trial_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    job: NextStep,
    steps: int,
) -> AgentRunResult:
    function = job.function
    if not isinstance(function, ReportTaskCompletion):
        raise TypeError("completion result requires ReportTaskCompletion")

    if recorder:
        recorder.record_event(
            event_id=short_id("evt"),
            event_type="answer.submitted",
            name="report_completion",
            status="ok" if function.outcome == "OUTCOME_OK" else "error",
            ts_start=utc_now(),
            ts_end=utc_now(),
            duration_ms=0,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            output_json=function.model_dump(),
        )
    status = CLI_GREEN if function.outcome == "OUTCOME_OK" else CLI_YELLOW
    print(f"{status}agent {function.outcome}{CLI_CLR}. Summary:")
    for item in function.completed_steps_laconic:
        print(f"- {item}")
    print(f"\n{CLI_BLUE}AGENT SUMMARY: {function.message}{CLI_CLR}")
    for ref in function.grounding_refs:
        print(f"- {CLI_BLUE}{ref}{CLI_CLR}")
    _finish_step(
        recorder,
        step_event,
        step_id,
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        output_json={
            "task_completed": job.task_completed,
            "outcome": function.outcome,
            "current_state": job.current_state,
            "decision_summary": job.decision_summary,
            "uncertainty_flags": job.uncertainty_flags,
        },
    )
    return AgentRunResult(
        final_outcome=function.outcome,
        final_message=function.message,
        completed_steps_laconic=function.completed_steps_laconic,
        grounding_refs=function.grounding_refs,
        steps=steps,
        stopped_reason="reported_completion",
        counters=_trial_counters(recorder, local_trial_id),
    )


def _finish_step(
    recorder: Recorder | None,
    step_event: str | None,
    step_id: str,
    *,
    status: str = "ok",
    local_trial_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    output_json: dict[str, Any] | None = None,
    error_json: dict[str, Any] | None = None,
) -> None:
    if not recorder:
        return
    recorder.finish_event(
        step_event,
        "agent.step.finished",
        step_id,
        status=status,
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        step_id=step_id,
        output_json=output_json,
        error_json=error_json,
    )


def _exception_result(
    exc: Exception,
    recorder: Recorder | None,
    local_trial_id: str | None,
    *,
    steps: int,
) -> AgentRunResult:
    return AgentRunResult(
        steps=steps,
        stopped_reason="exception",
        exception={"type": exc.__class__.__name__, "message": str(exc)},
        counters=_trial_counters(recorder, local_trial_id),
    )


def _trial_counters(recorder: Recorder | None, local_trial_id: str | None) -> dict:
    if recorder and local_trial_id:
        return recorder.store.trial_counters(local_trial_id)
    return {}
