from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    EvalPolicy,
    GetBenchmarkRequest,
    StartRunRequest,
    StartTrialRequest,
    StatusRequest,
    SubmitRunRequest,
)
from connectrpc.errors import ConnectError
from pydantic import BaseModel, Field

from agent import AgentRunResult, run_agent
from analyst import (
    TaskClassification,
    analytics_context_for_task,
    analyze_failure,
    classify_task,
    should_analyze_failure,
)
from analytics_store import json_dumps
from observability import (
    Recorder,
    file_hash,
    git_snapshot,
    protobuf_to_dict,
    python_version,
    safe_env_snapshot,
    safe_slug,
    short_id,
    stable_hash,
    utc_now,
)
from reports import generate_reports
from structured_llm import resolve_provider


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

BITGN_URL = (
    os.getenv("BITGN_HOST")
    or os.getenv("BENCHMARK_HOST")
    or "https://api.bitgn.com"
)
BITGN_API_KEY = os.getenv("BITGN_API_KEY") or ""
BENCH_ID = os.getenv("BENCH_ID") or os.getenv("BENCHMARK_ID") or "bitgn/ecom1-dev"
MODEL_ID = os.getenv("MODEL_ID") or "gpt-5.5"

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"


class _TeeStream:
    _bitgn_obs_tee = True

    def __init__(self, primary: Any, secondary: Any) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())


class BenchmarkRunResult(BaseModel):
    local_run_id: str | None = None
    bitgn_run_id: str | None = None
    benchmark_id: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    total_score: float | None = None
    task_count: int = 0
    regressions: list[str] = Field(default_factory=list)


def run_benchmark(
    *,
    task_filter: list[str],
    monitor: bool,
    classify_tasks: bool,
    classify_only: bool,
    analyze_failures: bool,
    use_analytics_context: bool,
    rerun_history_on_success: bool,
    regression_mode: bool = False,
    active_fix_id: str | None = None,
    active_hypothesis_id: str | None = None,
    run_name: str | None = None,
    obs_dir: str | Path = ".bitgn_obs",
) -> BenchmarkRunResult:
    if classify_only:
        monitor = True
        classify_tasks = True
    if monitor:
        analyze_failures = analyze_failures
        use_analytics_context = use_analytics_context

    recorder = Recorder(obs_dir) if monitor else None
    original_stdout = original_stderr = None
    stdout_file = stderr_file = None
    if recorder:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        stdout_file = (recorder.run_dir / "stdout.txt").open("a", encoding="utf-8")
        stderr_file = (recorder.run_dir / "stderr.txt").open("a", encoding="utf-8")
        sys.stdout = _TeeStream(original_stdout, stdout_file)
        sys.stderr = _TeeStream(original_stderr, stderr_file)

    def restore_stdio() -> None:
        nonlocal stdout_file, stderr_file
        if original_stdout is not None:
            sys.stdout = original_stdout
        if original_stderr is not None:
            sys.stderr = original_stderr
        for handle in (stdout_file, stderr_file):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()
        stdout_file = None
        stderr_file = None

    provider = resolve_provider()
    local_run_id = recorder.local_run_id if recorder else None
    result = BenchmarkRunResult(local_run_id=local_run_id, benchmark_id=BENCH_ID)
    previous_best = recorder.store.previous_best_scores() if recorder else {}
    started_at = utc_now()

    if recorder:
        snap = git_snapshot()
        command = " ".join(sys.argv)
        recorder.store.insert_or_replace(
            "runs",
            {
                "local_run_id": recorder.local_run_id,
                "bitgn_run_id": None,
                "benchmark_id": BENCH_ID,
                "model_id": MODEL_ID,
                "provider": provider,
                "started_at": started_at,
                "ended_at": None,
                "status": "started",
                "command": command,
                "git_sha": snap["git_sha"],
                "git_dirty": snap["git_dirty"],
                "git_diff_hash": snap["git_diff_hash"],
                "py_version": python_version(),
                "uv_lock_hash": file_hash("uv.lock"),
                "env_json": json_dumps(safe_env_snapshot()),
                "total_score": None,
                "task_count": 0,
                "notes": run_name,
            },
        )
        recorder.write_json(
            recorder.run_dir / "run.json",
            {
                "local_run_id": recorder.local_run_id,
                "benchmark_id": BENCH_ID,
                "model_id": MODEL_ID,
                "provider": provider,
                "started_at": started_at,
                "command": command,
                "regression_mode": regression_mode,
                "active_fix_id": active_fix_id,
                "active_hypothesis_id": active_hypothesis_id,
            },
        )
        (recorder.run_dir / "stdout.txt").touch()
        (recorder.run_dir / "stderr.txt").touch()
        run_event = recorder.start_event(
            "run.started",
            run_name or "ECOM Python Sample",
            input_json={
                "benchmark_id": BENCH_ID,
                "task_filter": task_filter,
                "classify_tasks": classify_tasks,
                "classify_only": classify_only,
                "regression_mode": regression_mode,
            },
        )
    else:
        run_event = None

    client = HarnessServiceClientSync(BITGN_URL)
    scores: list[tuple[str, float]] = []
    run_response = None
    try:
        status_res = _harness_call(
            recorder,
            "client.status",
            StatusRequest(),
            lambda: client.status(StatusRequest()),
        )
        print("Connecting to BitGN", status_res)
        bench = _harness_call(
            recorder,
            "client.get_benchmark",
            GetBenchmarkRequest(benchmark_id=BENCH_ID),
            lambda: client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID)),
        )
        print(
            f"{EvalPolicy.Name(bench.policy)} benchmark: {bench.benchmark_id} "
            f"with {len(bench.tasks)} tasks.\n{CLI_GREEN}{bench.description}{CLI_CLR}"
        )

        if classify_only:
            _classify_only_real_trials(
                recorder=recorder,
                client=client,
                bench=bench,
                task_filter=task_filter,
                model=MODEL_ID,
                run_name=run_name,
                result=result,
            )
            if recorder:
                _finish_run(recorder, run_event, "classified", result)
                generate_reports(obs_dir=obs_dir, local_run_id=recorder.local_run_id, recorder=recorder)
                restore_stdio()
                recorder.close()
            return result

        start_run_req = StartRunRequest(
            name=run_name or "ECOM Python Sample",
            benchmark_id=BENCH_ID,
            api_key=BITGN_API_KEY,
        )
        run_response = _harness_call(
            recorder,
            "client.start_run",
            start_run_req,
            lambda: client.start_run(start_run_req),
        )
        result.bitgn_run_id = run_response.run_id
        if recorder:
            recorder.store.update(
                "runs",
                "local_run_id",
                recorder.local_run_id,
                {"bitgn_run_id": run_response.run_id, "benchmark_id": run_response.benchmark_id},
            )

        try:
            for trial_id, task_hint in _planned_trials(run_response.trial_ids, bench.tasks):
                if task_filter and task_hint and task_hint not in task_filter:
                    continue
                trial = _harness_call(
                    recorder,
                    "client.start_trial",
                    StartTrialRequest(trial_id=trial_id),
                    lambda trial_id=trial_id: client.start_trial(StartTrialRequest(trial_id=trial_id)),
                )
                if task_filter and trial.task_id not in task_filter:
                    end_req = EndTrialRequest(trial_id=trial.trial_id)
                    _harness_call(
                        recorder,
                        "client.end_trial",
                        end_req,
                        lambda end_req=end_req: client.end_trial(end_req),
                        task_id=trial.task_id,
                    )
                    continue

                trial_result = _run_trial(
                    recorder=recorder,
                    client=client,
                    trial=trial,
                    task_filter=task_filter,
                    classify_tasks=classify_tasks,
                    analyze_failures=analyze_failures,
                    use_analytics_context=use_analytics_context,
                    active_fix_id=active_fix_id,
                    active_hypothesis_id=active_hypothesis_id,
                    provider=provider,
                )
                if trial_result["score_available"]:
                    score = float(trial_result["score"])
                    scores.append((trial.task_id, score))
                    result.scores[trial.task_id] = score
        finally:
            if run_response is not None:
                _harness_call(
                    recorder,
                    "client.submit_run",
                    SubmitRunRequest(run_id=run_response.run_id, force=True),
                    lambda: client.submit_run(SubmitRunRequest(run_id=run_response.run_id, force=True)),
                )
    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
        if recorder:
            _finish_run(
                recorder,
                run_event,
                "error",
                result,
                error_json={"type": "ConnectError", "code": str(exc.code), "message": exc.message},
            )
            restore_stdio()
        raise
    except KeyboardInterrupt:
        print(f"{CLI_RED}Interrupted{CLI_CLR}")
        if recorder:
            _finish_run(recorder, run_event, "interrupted", result)
            restore_stdio()
        raise
    except Exception as exc:
        if recorder:
            _finish_run(
                recorder,
                run_event,
                "error",
                result,
                error_json={"type": exc.__class__.__name__, "message": str(exc)},
            )
            restore_stdio()
        raise

    if scores:
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"{task_id}: {style}{score:0.2f}{CLI_CLR}")
        total = sum(score for _, score in scores) / len(scores)
        result.total_score = total
        result.task_count = len(scores)
        print(f"FINAL: {total * 100.0:0.2f}%")

    if recorder:
        _finish_run(recorder, run_event, "ok", result)
        generate_reports(obs_dir=obs_dir, local_run_id=recorder.local_run_id, recorder=recorder)
        _maybe_run_regression_suite(
            recorder=recorder,
            current_result=result,
            previous_best=previous_best,
            monitor=monitor,
            active_fix_id=active_fix_id,
            active_hypothesis_id=active_hypothesis_id,
            rerun_history_on_success=rerun_history_on_success,
            regression_mode=regression_mode,
            obs_dir=obs_dir,
            run_name=run_name,
        )
        restore_stdio()
        recorder.close()
    return result


def _run_trial(
    *,
    recorder: Recorder | None,
    client: HarnessServiceClientSync,
    trial: Any,
    task_filter: list[str],
    classify_tasks: bool,
    analyze_failures: bool,
    use_analytics_context: bool,
    active_fix_id: str | None,
    active_hypothesis_id: str | None,
    provider: str,
) -> dict[str, Any]:
    task_id = trial.task_id
    print(f"{'=' * 30} Starting task: {task_id} {'=' * 30}")
    print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")
    local_trial_id = None
    attempt_no = 1
    attempt_id = "attempt_001"
    trial_dir = None
    trial_event = None
    classification_id = None
    classification: TaskClassification | None = None
    agent_result = AgentRunResult(stopped_reason="exception", exception={"type": "NotRun"})

    if recorder:
        attempt_no = recorder.store.next_attempt_no(task_id)
        attempt_id = f"attempt_{attempt_no:03d}"
        local_trial_id = f"{recorder.local_run_id}__{safe_slug(task_id)}__attempt_{attempt_no}"
        trial_dir = recorder.trial_dir(task_id, attempt_no)
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_event = recorder.start_event(
            "trial.started",
            task_id,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            input_json={
                "bitgn_trial_id": trial.trial_id,
                "task_id": task_id,
                "instruction_hash": stable_hash(trial.instruction),
                "harness_url": trial.harness_url,
            },
        )
        recorder.write_json(
            trial_dir / "trial.json",
            {
                "local_trial_id": local_trial_id,
                "local_run_id": recorder.local_run_id,
                "bitgn_trial_id": trial.trial_id,
                "task_id": task_id,
                "attempt_no": attempt_no,
                "attempt_id": attempt_id,
                "instruction": trial.instruction,
                "harness_url_hash": stable_hash(trial.harness_url),
                "started_at": utc_now(),
            },
        )
        recorder.store.insert_or_replace(
            "trials",
            {
                "local_trial_id": local_trial_id,
                "local_run_id": recorder.local_run_id,
                "bitgn_trial_id": trial.trial_id,
                "task_id": task_id,
                "attempt_no": attempt_no,
                "attempt_id": attempt_id,
                "instruction": trial.instruction,
                "instruction_hash": stable_hash(trial.instruction),
                "harness_url_hash": stable_hash(trial.harness_url),
                "started_at": utc_now(),
                "ended_at": None,
                "status": "started",
                "agent_outcome": None,
                "score_available": 0,
                "score": None,
                "score_detail_json": json_dumps([]),
                "exception_json": None,
                "classification_id": None,
                "failure_analysis_id": None,
                "active_fix_id": active_fix_id,
                "active_hypothesis_id": active_hypothesis_id,
            },
        )

    if recorder and classify_tasks:
        try:
            classification_id, classification = classify_task(
                recorder=recorder,
                model=MODEL_ID,
                task_id=task_id,
                instruction=trial.instruction,
                local_trial_id=local_trial_id,
                attempt_id=attempt_id,
                trial_dir=trial_dir,
            )
            recorder.store.update(
                "trials",
                "local_trial_id",
                local_trial_id,
                {"classification_id": classification_id},
            )
        except Exception as exc:
            print(f"{CLI_RED}classification failed for {task_id}: {exc}{CLI_CLR}")
            recorder.record_event(
                event_id=short_id("evt"),
                event_type="task.classification.finished",
                name="task_classification",
                status="error",
                ts_start=utc_now(),
                ts_end=utc_now(),
                duration_ms=0,
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                error_json={"type": exc.__class__.__name__, "message": str(exc)},
            )

    analytics_context = None
    classification_for_agent: Any | None = classification
    if recorder and use_analytics_context:
        try:
            analytics_context = analytics_context_for_task(recorder.store, task_id=task_id)
            if classification_for_agent is None:
                row = recorder.store.latest_classification(task_id)
                classification_for_agent = dict(row) if row else None
        except Exception as exc:
            print(f"{CLI_RED}analytics context failed for {task_id}: {exc}{CLI_CLR}")
            recorder.record_event(
                event_id=short_id("evt"),
                event_type="agent.step.finished",
                name="analytics_context",
                status="error",
                ts_start=utc_now(),
                ts_end=utc_now(),
                duration_ms=0,
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                error_json={"type": exc.__class__.__name__, "message": str(exc)},
            )

    try:
        agent_result = run_agent(
            MODEL_ID,
            trial.harness_url,
            trial.instruction,
            recorder=recorder,
            local_run_id=recorder.local_run_id if recorder else None,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            analytics_context=analytics_context,
            classification=classification_for_agent,
        )
    except Exception as exc:
        print(f"{CLI_RED}agent failed for {task_id}: {exc}{CLI_CLR}")
        agent_result = AgentRunResult(
            steps=0,
            stopped_reason="exception",
            exception={"type": exc.__class__.__name__, "message": str(exc)},
        )

    end_req = EndTrialRequest(trial_id=trial.trial_id)
    end = _harness_call(
        recorder,
        "client.end_trial",
        end_req,
        lambda: client.end_trial(end_req),
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
    )
    score_detail = list(end.score_detail)
    score_json = {
        "trial_id": end.trial_id,
        "score_available": end.score_available,
        "score": float(end.score) if end.score_available else None,
        "score_detail": score_detail,
    }
    if end.score_available:
        style = CLI_GREEN if end.score == 1 else CLI_RED
        explain = textwrap.indent("\n".join(score_detail), "  ")
        print(f"\n{style}Score: {end.score:0.2f}\n{explain}\n{CLI_CLR}")
    else:
        print(f"\n{CLI_BLUE}Score: not available{CLI_CLR}\n")

    if recorder and trial_dir and local_trial_id:
        score_artifact = recorder.write_json(trial_dir / "score.json", score_json)
        recorder.record_event(
            event_id=short_id("evt"),
            event_type="score.received",
            name=task_id,
            status="ok" if end.score_available else "skipped",
            ts_start=utc_now(),
            ts_end=utc_now(),
            duration_ms=0,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            output_json=score_json,
            artifact_paths_json=[score_artifact],
        )
        counters = recorder.store.trial_counters(local_trial_id)
        counters.update(
            {
                "duration_ms": None,
                "final_outcome": agent_result.final_outcome,
                "score": score_json["score"],
            }
        )
        recorder.store.update(
            "trials",
            "local_trial_id",
            local_trial_id,
            {
                "ended_at": utc_now(),
                "status": "finished",
                "agent_outcome": agent_result.final_outcome,
                "score_available": int(bool(end.score_available)),
                "score": float(end.score) if end.score_available else None,
                "score_detail_json": json_dumps(score_detail),
                "exception_json": json_dumps(agent_result.exception),
                "active_fix_id": active_fix_id,
                "active_hypothesis_id": active_hypothesis_id,
            },
        )
        if analyze_failures and should_analyze_failure(
            agent_result=agent_result,
            score_available=bool(end.score_available),
            score=float(end.score) if end.score_available else None,
        ):
            if classification is None:
                row = recorder.store.latest_classification(task_id)
                classification_payload = dict(row) if row else None
            else:
                classification_payload = classification
            try:
                analyze_failure(
                    recorder=recorder,
                    model=MODEL_ID,
                    local_trial_id=local_trial_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    instruction=trial.instruction,
                    classification=classification_payload,
                    score_json=score_json,
                    agent_result=agent_result,
                    trial_dir=trial_dir,
                )
            except Exception as exc:
                print(f"{CLI_RED}failure analysis failed for {task_id}: {exc}{CLI_CLR}")
                recorder.record_event(
                    event_id=short_id("evt"),
                    event_type="failure.analysis.finished",
                    name="failure_analysis",
                    status="error",
                    ts_start=utc_now(),
                    ts_end=utc_now(),
                    duration_ms=0,
                    local_trial_id=local_trial_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    error_json={"type": exc.__class__.__name__, "message": str(exc)},
                )
        recorder.write_json(
            trial_dir / "trial.json",
            {
                "local_trial_id": local_trial_id,
                "local_run_id": recorder.local_run_id,
                "bitgn_trial_id": trial.trial_id,
                "task_id": task_id,
                "attempt_no": attempt_no,
                "attempt_id": attempt_id,
                "instruction": trial.instruction,
                "harness_url_hash": stable_hash(trial.harness_url),
                "agent_result": agent_result.model_dump(),
                "score": score_json,
                "counters": recorder.store.trial_counters(local_trial_id),
                "ended_at": utc_now(),
            },
        )
        recorder.finish_event(
            trial_event,
            "trial.finished",
            task_id,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            output_json={"agent_result": agent_result.model_dump(), "score": score_json},
            counts_json=recorder.store.trial_counters(local_trial_id),
        )
    return score_json


def _harness_call(
    recorder: Recorder | None,
    name: str,
    request: Any,
    fn: Callable[[], Any],
    *,
    local_trial_id: str | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
) -> Any:
    started = None
    if recorder:
        started = recorder.start_event(
            "harness.call.started",
            name,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            input_json=protobuf_to_dict(request),
            tags_json={"harness_url_hash": stable_hash(BITGN_URL)},
        )
    try:
        response = fn()
    except ConnectError as exc:
        if recorder:
            recorder.finish_event(
                started,
                "harness.call.finished",
                name,
                status="error",
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                error_json={"type": "ConnectError", "code": str(exc.code), "message": exc.message},
            )
        raise
    except Exception as exc:
        if recorder:
            recorder.finish_event(
                started,
                "harness.call.finished",
                name,
                status="error",
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                error_json={"type": exc.__class__.__name__, "message": str(exc)},
            )
        raise
    if recorder:
        artifact = recorder.write_json(
            recorder.run_dir / "artifacts" / f"harness_{safe_slug(name)}_{short_id('call')}.json",
            {"request": protobuf_to_dict(request), "response": protobuf_to_dict(response)},
        )
        recorder.finish_event(
            started,
            "harness.call.finished",
            name,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            output_json=protobuf_to_dict(response),
            artifact_paths_json=[artifact],
        )
    return response


def _classify_only_real_trials(
    *,
    recorder: Recorder | None,
    client: HarnessServiceClientSync,
    bench: Any,
    task_filter: list[str],
    model: str,
    run_name: str | None,
    result: BenchmarkRunResult,
) -> None:
    if recorder is None:
        return
    start_run_req = StartRunRequest(
        name=run_name or "ECOM Python Sample classify-only",
        benchmark_id=BENCH_ID,
        api_key=BITGN_API_KEY,
    )
    run_response = _harness_call(
        recorder,
        "client.start_run",
        start_run_req,
        lambda: client.start_run(start_run_req),
    )
    result.bitgn_run_id = run_response.run_id
    recorder.store.update(
        "runs",
        "local_run_id",
        recorder.local_run_id,
        {"bitgn_run_id": run_response.run_id, "benchmark_id": run_response.benchmark_id},
    )
    try:
        for trial_id, task_hint in _planned_trials(run_response.trial_ids, bench.tasks):
            if task_filter and task_hint and task_hint not in task_filter:
                continue
            trial = None
            try:
                trial = _harness_call(
                    recorder,
                    "client.start_trial",
                    StartTrialRequest(trial_id=trial_id),
                    lambda trial_id=trial_id: client.start_trial(StartTrialRequest(trial_id=trial_id)),
                )
                if task_filter and trial.task_id not in task_filter:
                    continue
                classify_task(
                    recorder=recorder,
                    model=model,
                    task_id=trial.task_id,
                    instruction=trial.instruction,
                )
            except Exception as exc:
                task_id = getattr(trial, "task_id", task_hint or trial_id)
                print(f"{CLI_RED}classification failed for {task_id}: {exc}{CLI_CLR}")
                recorder.record_event(
                    event_id=short_id("evt"),
                    event_type="task.classification.finished",
                    name="task_classification",
                    status="error",
                    ts_start=utc_now(),
                    ts_end=utc_now(),
                    duration_ms=0,
                    task_id=task_id,
                    error_json={"type": exc.__class__.__name__, "message": str(exc)},
                )
            finally:
                if trial is not None:
                    end_req = EndTrialRequest(trial_id=trial.trial_id)
                    _harness_call(
                        recorder,
                        "client.end_trial",
                        end_req,
                        lambda end_req=end_req: client.end_trial(end_req),
                        task_id=trial.task_id,
                    )
    finally:
        _harness_call(
            recorder,
            "client.submit_run",
            SubmitRunRequest(run_id=run_response.run_id, force=True),
            lambda: client.submit_run(SubmitRunRequest(run_id=run_response.run_id, force=True)),
        )


def _planned_trials(trial_ids: list[str], tasks: Any) -> list[tuple[str, str | None]]:
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) == len(trial_ids):
        return list(zip(trial_ids, task_ids))
    return [(trial_id, None) for trial_id in trial_ids]


def _finish_run(
    recorder: Recorder,
    run_event: str | None,
    status: str,
    result: BenchmarkRunResult,
    *,
    error_json: dict[str, Any] | None = None,
) -> None:
    counters = recorder.store.run_counters(recorder.local_run_id)
    total_score = result.total_score
    if total_score is None and counters["mean_score"] is not None:
        total_score = float(counters["mean_score"])
        result.total_score = total_score
    result.task_count = counters["tasks"]
    ended_at = utc_now()
    recorder.store.update(
        "runs",
        "local_run_id",
        recorder.local_run_id,
        {
            "ended_at": ended_at,
            "status": status,
            "total_score": total_score,
            "task_count": counters["tasks"],
        },
    )
    recorder.write_json(
        recorder.run_dir / "run.json",
        {
            "local_run_id": recorder.local_run_id,
            "bitgn_run_id": result.bitgn_run_id,
            "benchmark_id": result.benchmark_id,
            "status": status,
            "total_score": total_score,
            "task_count": counters["tasks"],
            "counters": counters,
            "ended_at": ended_at,
        },
    )
    summary = _run_summary_markdown(result, counters)
    (recorder.run_dir / "summary.md").write_text(summary)
    recorder.finish_event(
        run_event,
        "run.finished",
        "benchmark_run",
        status="error" if status == "error" else "ok",
        output_json=result.model_dump(),
        error_json=error_json,
        counts_json=counters,
    )


def _maybe_run_regression_suite(
    *,
    recorder: Recorder,
    current_result: BenchmarkRunResult,
    previous_best: dict[str, float],
    monitor: bool,
    active_fix_id: str | None,
    active_hypothesis_id: str | None,
    rerun_history_on_success: bool,
    regression_mode: bool,
    obs_dir: str | Path,
    run_name: str | None,
) -> None:
    if regression_mode or not rerun_history_on_success or not active_hypothesis_id:
        return
    improved_tasks = [
        task_id
        for task_id, score in current_result.scores.items()
        if score >= 1.0 and previous_best.get(task_id, 0.0) < 1.0
    ]
    if not improved_tasks:
        return
    task_ids = recorder.store.known_task_ids()
    if not task_ids:
        return
    regression_run_id = short_id("regrun")
    created_at = utc_now()
    recorder.store.insert(
        "regression_runs",
        {
            "regression_run_id": regression_run_id,
            "created_at": created_at,
            "trigger_change_id": active_fix_id,
            "trigger_hypothesis_id": active_hypothesis_id,
            "trigger_task_id": improved_tasks[0],
            "local_run_id": None,
            "task_ids_json": json_dumps(task_ids),
            "status": "started",
            "summary_json": json_dumps({}),
        },
    )
    recorder.record_event(
        event_id=short_id("evt"),
        event_type="regression_suite.started",
        name=regression_run_id,
        status="started",
        ts_start=created_at,
        ts_end=None,
        duration_ms=None,
        output_json={"task_ids": task_ids, "trigger_hypothesis_id": active_hypothesis_id},
    )
    suite = run_benchmark(
        task_filter=task_ids,
        monitor=monitor,
        classify_tasks=False,
        classify_only=False,
        analyze_failures=False,
        use_analytics_context=True,
        rerun_history_on_success=False,
        regression_mode=True,
        active_fix_id=active_fix_id,
        active_hypothesis_id=active_hypothesis_id,
        run_name=f"Regression suite for {active_hypothesis_id}" if run_name is None else f"{run_name} regression suite",
        obs_dir=obs_dir,
    )
    regressions: list[str] = []
    for task_id, best in previous_best.items():
        current = suite.scores.get(task_id)
        if best >= 1.0 and current is not None and current < 1.0:
            regression_id = short_id("reg")
            regressions.append(regression_id)
            history_rows = recorder.store.rows(
                """
                SELECT score FROM trials
                WHERE task_id=? AND local_run_id != ? AND score_available=1
                """,
                (task_id, suite.local_run_id),
            )
            history_scores = [float(row["score"]) for row in history_rows if row["score"] is not None]
            severity = (
                "suspected_flaky"
                if any(score >= 1.0 for score in history_scores)
                and any(score < 1.0 for score in history_scores)
                else "failed_previous_pass"
            )
            recorder.store.insert(
                "regressions",
                {
                    "regression_id": regression_id,
                    "regression_run_id": regression_run_id,
                    "task_id": task_id,
                    "previous_best_score": best,
                    "current_score": current,
                    "severity": severity,
                    "detected_at": utc_now(),
                    "suspected_cause_change_id": active_fix_id,
                    "suspected_cause_hypothesis_id": active_hypothesis_id,
                    "notes": f"Task previously had best score {best:0.3f}; regression suite score is {current:0.3f}.",
                },
            )
            recorder.record_event(
                event_id=short_id("evt"),
                event_type="regression.detected",
                name=regression_id,
                status="error",
                ts_start=utc_now(),
                ts_end=utc_now(),
                duration_ms=0,
                task_id=task_id,
                output_json={"previous_best_score": best, "current_score": current},
            )
    final_status = "supported_with_regressions" if regressions else "supported"
    recorder.store.update(
        "hypotheses",
        "hypothesis_id",
        active_hypothesis_id,
        {"status": final_status, "updated_at": utc_now()},
    )
    if active_fix_id and regressions:
        recorder.store.update("changes", "change_id", active_fix_id, {"status": "caused_regression"})
    recorder.store.update(
        "regression_runs",
        "regression_run_id",
        regression_run_id,
        {
            "local_run_id": suite.local_run_id,
            "status": "finished",
            "summary_json": json_dumps({"regressions": regressions, "scores": suite.scores}),
        },
    )
    recorder.record_event(
        event_id=short_id("evt"),
        event_type="regression_suite.finished",
        name=regression_run_id,
        status="error" if regressions else "ok",
        ts_start=utc_now(),
        ts_end=utc_now(),
        duration_ms=0,
        output_json={"regressions": regressions, "hypothesis_status": final_status},
    )
    event_type = "hypothesis.supported" if not regressions else "hypothesis.refuted"
    recorder.record_event(
        event_id=short_id("evt"),
        event_type=event_type,
        name=active_hypothesis_id,
        status="ok" if not regressions else "error",
        ts_start=utc_now(),
        ts_end=utc_now(),
        duration_ms=0,
        output_json={"status": final_status, "regressions": regressions},
    )
    current_result.regressions = regressions


def _run_summary_markdown(result: BenchmarkRunResult, counters: dict[str, Any]) -> str:
    lines = [
        "# BitGN ECOM Run Summary",
        "",
        f"Run: {result.local_run_id or '-'}",
        f"Benchmark: {result.benchmark_id or '-'}",
        f"BitGN run: {result.bitgn_run_id or '-'}",
        f"Score: {'-' if result.total_score is None else f'{result.total_score:0.3f}'}",
        "",
        "## Counters",
    ]
    for key, value in counters.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BitGN ECOM Python sample")
    parser.add_argument("tasks", nargs="*", help="optional task ids, e.g. t01 t04")
    parser.add_argument("--monitor", action="store_true", help="enable local SQLite/JSONL observability")
    parser.add_argument("--classify-tasks", action="store_true", help="classify selected tasks before running")
    parser.add_argument("--classify-only", action="store_true", help="classify selected tasks without running the agent")
    parser.add_argument("--use-analytics-context", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--analyze-failures", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rerun-history-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--active-fix-id")
    parser.add_argument("--active-hypothesis-id")
    parser.add_argument("--run-name")
    parser.add_argument("--obs-dir", default=".bitgn_obs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    monitor = bool(args.monitor or args.classify_tasks or args.classify_only)
    analyze_failures = args.analyze_failures if args.analyze_failures is not None else monitor
    use_context = args.use_analytics_context if args.use_analytics_context is not None else monitor
    try:
        run_benchmark(
            task_filter=args.tasks,
            monitor=monitor,
            classify_tasks=args.classify_tasks,
            classify_only=args.classify_only,
            analyze_failures=analyze_failures,
            use_analytics_context=use_context,
            rerun_history_on_success=args.rerun_history_on_success,
            active_fix_id=args.active_fix_id,
            active_hypothesis_id=args.active_hypothesis_id,
            run_name=args.run_name,
            obs_dir=args.obs_dir,
        )
    except ConnectError:
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
