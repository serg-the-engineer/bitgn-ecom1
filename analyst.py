from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from analytics_store import AnalyticsStore, json_dumps, json_loads
from generalization_guard import generalize_runtime_guidance
from observability import Recorder, preview, short_id, stable_hash, utc_now
from structured_llm import resolve_provider, run_structured_llm


class TaskClassification(BaseModel):
    task_family: Literal[
        "catalog_search_and_identification",
        "catalog_data_update",
        "inventory_or_availability_update",
        "pricing_or_discount_update",
        "order_customer_refund_return",
        "data_quality_reconciliation",
        "reporting_or_analysis",
        "security_or_policy_refusal",
        "workspace_file_operation",
        "ambiguous_or_clarification",
        "unsupported_request",
        "other",
    ]
    task_subtype: str
    objective_summary: str
    main_difficulty: str
    expected_action_type: str
    likely_tools: list[str]
    risk_flags: list[str]
    reusable_lessons: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class FailureAnalysis(BaseModel):
    task_family: str
    failure_mode: Literal[
        "wrong_target_identification",
        "insufficient_verification",
        "bad_sql_or_tool_usage",
        "missed_instruction_constraint",
        "premature_completion",
        "security_policy_error",
        "runtime_or_connect_error",
        "llm_schema_or_parse_error",
        "timeout_or_step_budget",
        "overbroad_mutation",
        "underbroad_mutation",
        "unknown",
    ]
    root_cause_summary: str
    evidence: list[str]
    missed_observations: list[str]
    bad_tool_calls: list[str]
    proposed_hypotheses: list[str]
    recommended_single_change: str
    regression_risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


def classify_task(
    *,
    recorder: Recorder,
    model: str,
    task_id: str,
    instruction: str,
    local_trial_id: str | None = None,
    attempt_id: str | None = None,
    trial_dir: Path | None = None,
) -> tuple[str, TaskClassification]:
    provider = resolve_provider()
    start_event = recorder.start_event(
        "task.classification.started",
        "task_classification",
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        input_json={"task_id": task_id, "instruction_hash": stable_hash(instruction)},
    )
    prompt = f"""
You classify BitGN ECOM benchmark tasks for future agent improvement.

Return JSON matching the schema.
Use a general reusable task family, not an overly specific one.
Summarize what must be done, what makes it difficult, likely tools, and reusable lessons.
Do not solve the task. Do not invent facts not present in the instruction.

Task id: {task_id}
Instruction:
{instruction}
""".strip()
    try:
        classification = run_structured_llm(
            provider=provider,
            model=model,
            messages_or_prompt=prompt,
            response_model=TaskClassification,
            purpose="task_classification",
            recorder=recorder,
            context={
                "local_trial_id": local_trial_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "trial_dir": trial_dir,
            },
        )
    except Exception as exc:
        recorder.finish_event(
            start_event,
            "task.classification.finished",
            "task_classification",
            status="error",
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            error_json={"type": exc.__class__.__name__, "message": str(exc)},
        )
        raise
    classification_id = short_id("class")
    created_at = utc_now()
    raw = classification.model_dump()
    artifact_path = None
    if trial_dir:
        artifact_path = recorder.write_json(trial_dir / "classification.json", raw)
    else:
        artifact_path = recorder.write_json(
            recorder.run_dir / "artifacts" / f"classification_{task_id}.json",
            raw,
        )
    recorder.store.insert(
        "task_classifications",
        {
            "classification_id": classification_id,
            "task_id": task_id,
            "instruction_hash": stable_hash(instruction),
            "created_at": created_at,
            "model": model,
            "provider": provider,
            "task_family": classification.task_family,
            "task_subtype": classification.task_subtype,
            "objective_summary": classification.objective_summary,
            "main_difficulty": classification.main_difficulty,
            "expected_action_type": classification.expected_action_type,
            "likely_tools_json": json_dumps(classification.likely_tools),
            "risk_flags_json": json_dumps(classification.risk_flags),
            "reusable_lessons_json": json_dumps(classification.reusable_lessons),
            "confidence": classification.confidence,
            "raw_json": json_dumps(raw),
        },
    )
    recorder.finish_event(
        start_event,
        "task.classification.finished",
        "task_classification",
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        output_json={"classification_id": classification_id, **raw},
        artifact_paths_json=[artifact_path],
    )
    return classification_id, classification


def analyze_failure(
    *,
    recorder: Recorder,
    model: str,
    local_trial_id: str,
    task_id: str,
    attempt_id: str,
    instruction: str,
    classification: dict[str, Any] | TaskClassification | None,
    score_json: dict[str, Any],
    agent_result: Any,
    trial_dir: Path,
) -> tuple[str, FailureAnalysis, list[str]]:
    provider = resolve_provider()
    start_event = recorder.start_event(
        "failure.analysis.started",
        "failure_analysis",
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        input_json={"score": score_json, "agent_result": _model_dump(agent_result)},
    )
    transcript_summary = transcript_preview(trial_dir)
    events = event_summary(recorder.store, local_trial_id)
    prompt = f"""
You are the analyst for a BitGN ECOM benchmark agent.

Analyze why this trial failed. Use the trace and score details.
Return JSON matching the schema.

Rules:
- Do not propose multiple major fixes as one change.
- Formulate fixes as hypotheses.
- Prefer reusable task-family lessons over one-off facts.
- Identify regression risks.
- Mark uncertainty explicitly.

Task classification:
{json.dumps(_model_dump(classification), ensure_ascii=False, indent=2)}

Instruction:
{instruction}

Final score and score detail:
{json.dumps(score_json, ensure_ascii=False, indent=2)}

Agent result:
{json.dumps(_model_dump(agent_result), ensure_ascii=False, indent=2)}

Agent transcript summary:
{transcript_summary}

Tool/LLM event summary:
{json.dumps(events, ensure_ascii=False, indent=2)}

Full artifact paths are available in the observability directory.
""".strip()
    try:
        analysis = run_structured_llm(
            provider=provider,
            model=model,
            messages_or_prompt=prompt,
            response_model=FailureAnalysis,
            purpose="failure_analysis",
            recorder=recorder,
            context={
                "local_trial_id": local_trial_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "trial_dir": trial_dir,
            },
        )
    except Exception as exc:
        recorder.finish_event(
            start_event,
            "failure.analysis.finished",
            "failure_analysis",
            status="error",
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            error_json={"type": exc.__class__.__name__, "message": str(exc)},
        )
        raise
    failure_analysis_id = short_id("fail")
    raw = analysis.model_dump()
    artifact_path = recorder.write_json(trial_dir / "failure_analysis.json", raw)
    recorder.store.insert(
        "failure_analyses",
        {
            "failure_analysis_id": failure_analysis_id,
            "local_trial_id": local_trial_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "created_at": utc_now(),
            "model": model,
            "provider": provider,
            "failure_mode": analysis.failure_mode,
            "root_cause_summary": analysis.root_cause_summary,
            "evidence_json": json_dumps(analysis.evidence),
            "missed_observations_json": json_dumps(analysis.missed_observations),
            "bad_tool_calls_json": json_dumps(analysis.bad_tool_calls),
            "proposed_hypotheses_json": json_dumps(analysis.proposed_hypotheses),
            "recommended_single_change": analysis.recommended_single_change,
            "raw_json": json_dumps(raw),
        },
    )
    hypothesis_ids = create_hypotheses(
        recorder=recorder,
        task_id=task_id,
        task_family=analysis.task_family,
        failure_analysis_id=failure_analysis_id,
        analysis=analysis,
        attempt_id=attempt_id,
        local_trial_id=local_trial_id,
    )
    recorder.store.update(
        "trials",
        "local_trial_id",
        local_trial_id,
        {
            "failure_analysis_id": failure_analysis_id,
            "active_hypothesis_id": hypothesis_ids[0] if hypothesis_ids else None,
        },
    )
    recorder.finish_event(
        start_event,
        "failure.analysis.finished",
        "failure_analysis",
        local_trial_id=local_trial_id,
        task_id=task_id,
        attempt_id=attempt_id,
        output_json={"failure_analysis_id": failure_analysis_id, **raw, "hypotheses": hypothesis_ids},
        artifact_paths_json=[artifact_path],
    )
    return failure_analysis_id, analysis, hypothesis_ids


def create_hypotheses(
    *,
    recorder: Recorder,
    task_id: str,
    task_family: str,
    failure_analysis_id: str,
    analysis: FailureAnalysis,
    attempt_id: str,
    local_trial_id: str,
) -> list[str]:
    statements = list(analysis.proposed_hypotheses)
    if analysis.recommended_single_change and analysis.recommended_single_change not in statements:
        statements.insert(0, analysis.recommended_single_change)
    hypothesis_ids: list[str] = []
    for idx, statement in enumerate(statements[:4]):
        statement = generalize_runtime_guidance(statement)
        hypothesis_id = short_id("hyp")
        scope = "recommended" if idx == 0 else "candidate"
        expected_effect = generalize_runtime_guidance(analysis.recommended_single_change) if idx == 0 else ""
        recorder.store.insert(
            "hypotheses",
            {
                "hypothesis_id": hypothesis_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "task_id": task_id,
                "task_family": task_family,
                "source_failure_analysis_id": failure_analysis_id,
                "statement": statement,
                "expected_effect": expected_effect,
                "scope": scope,
                "status": "proposed",
                "evidence_for_json": json_dumps(analysis.evidence if idx == 0 else []),
                "evidence_against_json": json_dumps([]),
                "attempts_json": json_dumps([attempt_id]),
                "superseded_by": None,
            },
        )
        recorder.record_event(
            event_id=short_id("evt"),
            event_type="hypothesis.created",
            name=hypothesis_id,
            status="ok",
            ts_start=utc_now(),
            ts_end=utc_now(),
            duration_ms=0,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            output_json={
                "hypothesis_id": hypothesis_id,
                "statement": statement,
                "scope": scope,
                "source_failure_analysis_id": failure_analysis_id,
            },
        )
        hypothesis_ids.append(hypothesis_id)
    return hypothesis_ids


def analytics_context_for_task(
    store: AnalyticsStore,
    *,
    task_id: str,
    max_chars: int = 2200,
) -> str | None:
    classification = store.latest_classification(task_id)
    family = classification["task_family"] if classification else None
    lines: list[str] = []
    if classification:
        lines.append(f"- Classified family: {classification['task_family']}.")
        if classification["main_difficulty"]:
            lines.append(f"- Main difficulty: {generalize_runtime_guidance(classification['main_difficulty'])}")
        lessons = json_loads(classification["reusable_lessons_json"], [])
        if lessons:
            lines.append(f"- Reusable lessons: {generalize_runtime_guidance('; '.join(lessons[:3]))}")
    if family:
        hyp_rows = store.rows(
            """
            SELECT * FROM hypotheses
            WHERE task_family=? AND status IN (
              'supported', 'refuted', 'partially_supported_with_regressions',
              'supported_with_regressions', 'proposed', 'applied', 'testing'
            )
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 8
            """,
            (family,),
        )
        if hyp_rows:
            lines.append("- Similar hypotheses:")
            for row in hyp_rows:
                status = row["status"]
                lines.append(f"  - Prior family hypothesis was {status}: {generalize_runtime_guidance(row['statement'])}")
    regression_rows = store.rows(
        """
        SELECT r.*, c.title AS change_title
        FROM regressions r
        LEFT JOIN changes c ON c.change_id = r.suspected_cause_change_id
        WHERE r.task_id=? OR r.suspected_cause_hypothesis_id IN (
          SELECT hypothesis_id FROM hypotheses WHERE task_family=?
        )
        ORDER BY r.detected_at DESC
        LIMIT 5
        """,
        (task_id, family or ""),
    )
    if regression_rows:
        lines.append("- Regression constraints:")
        for row in regression_rows:
            lines.append(
                "  - A prior related change caused a regression: "
                f"{generalize_runtime_guidance(row['notes'] or row['change_title'] or 'suspected change')}"
            )
    if classification:
        risks = json_loads(classification["risk_flags_json"], [])
        if risks:
            lines.append(f"- Recommended caution: {generalize_runtime_guidance('; '.join(risks[:4]))}")
    text = "\n".join(lines).strip()
    if not text:
        return None
    return text[:max_chars]


def transcript_preview(trial_dir: Path, max_chars: int = 5000) -> str:
    path = trial_dir / "transcript.jsonl"
    if not path.exists():
        return ""
    rows = []
    for line in path.read_text().splitlines()[-30:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = item.get("content") or ""
        rows.append(f"{item.get('message_index')}: {item.get('role')}: {preview(content, 600)}")
    return "\n".join(rows)[-max_chars:]


def event_summary(store: AnalyticsStore, local_trial_id: str) -> dict[str, Any]:
    tool_rows = store.rows(
        """
        SELECT tool_name, status, COUNT(*) AS n
        FROM tool_calls
        WHERE local_trial_id=?
        GROUP BY tool_name, status
        ORDER BY tool_name, status
        """,
        (local_trial_id,),
    )
    llm_rows = store.rows(
        """
        SELECT purpose, status, COUNT(*) AS n, COALESCE(SUM(approximate_tokens), 0) AS tokens
        FROM llm_calls
        WHERE local_trial_id=?
        GROUP BY purpose, status
        ORDER BY purpose, status
        """,
        (local_trial_id,),
    )
    event_rows = store.rows(
        """
        SELECT event_type, status, COUNT(*) AS n
        FROM events
        WHERE local_trial_id=?
        GROUP BY event_type, status
        ORDER BY event_type, status
        """,
        (local_trial_id,),
    )
    return {
        "tool_calls": [dict(row) for row in tool_rows],
        "llm_calls": [dict(row) for row in llm_rows],
        "events": [dict(row) for row in event_rows],
    }


def should_analyze_failure(
    *,
    agent_result: Any,
    score_available: bool,
    score: float | None,
) -> bool:
    if score_available and score is not None and score < 1.0:
        return True
    result = _model_dump(agent_result)
    if result.get("exception"):
        return True
    if result.get("stopped_reason") in {"step_limit", "exception"}:
        return True
    final_outcome = result.get("final_outcome")
    return bool(final_outcome and final_outcome != "OUTCOME_OK")


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return str(value)
