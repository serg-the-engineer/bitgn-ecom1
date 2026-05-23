from __future__ import annotations

from pathlib import Path
from typing import Any

from analytics_store import AnalyticsStore, json_loads
from observability import Recorder, short_id, utc_now


def generate_reports(
    *,
    obs_dir: str | Path = ".bitgn_obs",
    local_run_id: str | None = None,
    recorder: Recorder | None = None,
) -> dict[str, str]:
    store = recorder.store if recorder else AnalyticsStore(Path(obs_dir) / "obs.db")
    reports_dir = Path(obs_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    run = _latest_run(store, local_run_id)
    run_id = run["local_run_id"] if run else local_run_id
    outputs = {
        "latest": reports_dir / "latest.md",
        "task_summary": reports_dir / "task_summary.md",
        "failures": reports_dir / "failures.md",
        "hypotheses": reports_dir / "hypotheses.md",
        "regressions": reports_dir / "regressions.md",
    }
    outputs["latest"].write_text(_latest_report(store, run))
    outputs["task_summary"].write_text(_task_summary_report(store))
    outputs["failures"].write_text(_failures_report(store))
    outputs["hypotheses"].write_text(_hypotheses_report(store))
    outputs["regressions"].write_text(_regressions_report(store))
    rel = {key: str(path) for key, path in outputs.items()}
    if recorder:
        recorder.record_event(
            event_id=short_id("evt"),
            event_type="report.generated",
            name="analytics_reports",
            status="ok",
            ts_start=utc_now(),
            ts_end=utc_now(),
            duration_ms=0,
            output_json={"local_run_id": run_id, "reports": rel},
            artifact_paths_json=[recorder.rel(path) or str(path) for path in outputs.values()],
        )
    if recorder is None:
        store.close()
    return rel


def summary_text(obs_dir: str | Path = ".bitgn_obs") -> str:
    store = AnalyticsStore(Path(obs_dir) / "obs.db")
    run = _latest_run(store, None)
    if not run:
        store.close()
        return "No observability runs found."
    counters = store.run_counters(run["local_run_id"])
    lines = [
        f"Run: {run['local_run_id']}",
        f"Benchmark: {run['benchmark_id'] or '-'}",
        f"Model/provider: {run['model_id'] or '-'} / {run['provider'] or '-'}",
        f"Status: {run['status'] or '-'}",
        f"Score: {_fmt(run['total_score'])}",
        f"Tasks: {counters['tasks']} passed={counters['passed']} failed={counters['failed']} unknown={counters['unknown']}",
        f"LLM calls: {counters['llm_calls']}",
        f"Tool calls: {counters['tool_calls_total']}",
    ]
    store.close()
    return "\n".join(lines)


def tasks_text(obs_dir: str | Path = ".bitgn_obs") -> str:
    store = AnalyticsStore(Path(obs_dir) / "obs.db")
    rows = _task_summary_rows(store)
    if not rows:
        store.close()
        return "No task classifications found."
    lines = ["task_id | family | objective | main difficulty", "--- | --- | --- | ---"]
    for row in rows:
        lines.append(
            f"{row['task_id']} | {row['task_family'] or '-'} | "
            f"{_one(row['objective_summary'])} | {_one(row['main_difficulty'])}"
        )
    store.close()
    return "\n".join(lines)


def hypotheses_text(obs_dir: str | Path = ".bitgn_obs") -> str:
    store = AnalyticsStore(Path(obs_dir) / "obs.db")
    rows = store.rows(
        """
        SELECT hypothesis_id, task_id, task_family, status, statement
        FROM hypotheses
        ORDER BY created_at DESC
        LIMIT 50
        """
    )
    if not rows:
        store.close()
        return "No hypotheses found."
    lines = ["hypothesis_id | task/family | status | statement", "--- | --- | --- | ---"]
    for row in rows:
        lines.append(
            f"{row['hypothesis_id']} | {row['task_id'] or row['task_family'] or '-'} | "
            f"{row['status'] or '-'} | {_one(row['statement'])}"
        )
    store.close()
    return "\n".join(lines)


def _latest_run(store: AnalyticsStore, local_run_id: str | None) -> Any:
    if local_run_id:
        return store.row("SELECT * FROM runs WHERE local_run_id=?", (local_run_id,))
    return store.row("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1")


def _latest_report(store: AnalyticsStore, run: Any) -> str:
    if not run:
        return "# Latest BitGN ECOM Run\n\nNo runs found.\n"
    run_id = run["local_run_id"]
    task_rows = store.rows(
        """
        SELECT t.task_id, t.score, t.agent_outcome, t.attempt_no, c.task_family,
               c.main_difficulty
        FROM trials t
        LEFT JOIN task_classifications c ON c.classification_id = t.classification_id
        WHERE t.local_run_id=?
        ORDER BY t.task_id, t.attempt_no
        """,
        (run_id,),
    )
    failure_rows = store.rows(
        """
        SELECT f.task_id, f.failure_mode, f.root_cause_summary,
               f.recommended_single_change, h.hypothesis_id
        FROM failure_analyses f
        LEFT JOIN hypotheses h ON h.source_failure_analysis_id = f.failure_analysis_id
        WHERE f.local_trial_id IN (SELECT local_trial_id FROM trials WHERE local_run_id=?)
        GROUP BY f.failure_analysis_id
        ORDER BY f.created_at DESC
        """,
        (run_id,),
    )
    tool_rows = store.rows(
        """
        SELECT tool_name, COUNT(*) AS n, SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors
        FROM tool_calls
        WHERE local_run_id=?
        GROUP BY tool_name
        ORDER BY n DESC
        """,
        (run_id,),
    )
    llm_rows = store.rows(
        """
        SELECT purpose, COUNT(*) AS n, COALESCE(SUM(approximate_tokens), 0) AS tokens,
               SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors
        FROM llm_calls
        WHERE local_run_id=?
        GROUP BY purpose
        ORDER BY n DESC
        """,
        (run_id,),
    )
    reg_rows = store.rows(
        """
        SELECT r.*
        FROM regressions r
        JOIN regression_runs rr ON rr.regression_run_id = r.regression_run_id
        WHERE rr.local_run_id=?
        ORDER BY r.detected_at DESC
        """,
        (run_id,),
    )
    lines = [
        "# Latest BitGN ECOM Run",
        "",
        f"Run: {run_id}",
        f"Benchmark: {run['benchmark_id'] or '-'}",
        f"Model/provider: {run['model_id'] or '-'} / {run['provider'] or '-'}",
        f"Git sha/diff hash: {run['git_sha'] or '-'} / {run['git_diff_hash'] or '-'}",
        f"Score: {_fmt(run['total_score'])}",
        "",
        "## Tasks",
        "| task_id | class | score | outcome | attempts | main difficulty |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task_id']} | {row['task_family'] or '-'} | {_fmt(row['score'])} | "
            f"{row['agent_outcome'] or '-'} | {row['attempt_no']} | {_cell(row['main_difficulty'])} |"
        )
    lines.extend(["", "## Failures", "| task_id | failure_mode | root cause | recommended single change | hypothesis |", "| --- | --- | --- | --- | --- |"])
    for row in failure_rows:
        lines.append(
            f"| {row['task_id']} | {row['failure_mode']} | {_cell(row['root_cause_summary'])} | "
            f"{_cell(row['recommended_single_change'])} | {row['hypothesis_id'] or '-'} |"
        )
    lines.extend(["", "## Tool Usage", "| tool | count | errors |", "| --- | ---: | ---: |"])
    for row in tool_rows:
        lines.append(f"| {row['tool_name']} | {row['n']} | {row['errors'] or 0} |")
    lines.extend(["", "## LLM Usage", "| purpose | calls | approx tokens | errors |", "| --- | ---: | ---: | ---: |"])
    for row in llm_rows:
        lines.append(f"| {row['purpose']} | {row['n']} | {row['tokens']} | {row['errors'] or 0} |")
    lines.extend(["", "## Regressions"])
    if reg_rows:
        lines.extend(["| regression_id | task_id | previous best | current | severity | notes |", "| --- | --- | ---: | ---: | --- | --- |"])
        for row in reg_rows:
            lines.append(
                f"| {row['regression_id']} | {row['task_id']} | {_fmt(row['previous_best_score'])} | "
                f"{_fmt(row['current_score'])} | {row['severity']} | {_cell(row['notes'])} |"
            )
    else:
        lines.append("No regressions detected for this run.")
    lines.append("")
    return "\n".join(lines)


def _task_summary_report(store: AnalyticsStore) -> str:
    rows = _task_summary_rows(store)
    lines = [
        "# BitGN ECOM Task Summary",
        "",
        "| task_id | task_family | subtype | objective | main difficulty | reusable lessons |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lessons = "; ".join(json_loads(row["reusable_lessons_json"], [])[:4])
        lines.append(
            f"| {row['task_id']} | {row['task_family'] or '-'} | {_cell(row['task_subtype'])} | "
            f"{_cell(row['objective_summary'])} | {_cell(row['main_difficulty'])} | {_cell(lessons)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _failures_report(store: AnalyticsStore) -> str:
    rows = store.rows(
        """
        SELECT f.*, h.hypothesis_id
        FROM failure_analyses f
        LEFT JOIN hypotheses h ON h.source_failure_analysis_id = f.failure_analysis_id
        GROUP BY f.failure_analysis_id
        ORDER BY f.created_at DESC
        """
    )
    lines = ["# BitGN ECOM Failures", "", "| task_id | failure_mode | root cause | recommended single change | hypothesis |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['failure_mode']} | {_cell(row['root_cause_summary'])} | "
            f"{_cell(row['recommended_single_change'])} | {row['hypothesis_id'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _hypotheses_report(store: AnalyticsStore) -> str:
    rows = store.rows("SELECT * FROM hypotheses ORDER BY created_at DESC")
    lines = ["# BitGN ECOM Hypotheses", "", "| hypothesis_id | task/task_family | statement | status | evidence |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        evidence = "; ".join(json_loads(row["evidence_for_json"], [])[:3])
        lines.append(
            f"| {row['hypothesis_id']} | {row['task_id'] or row['task_family'] or '-'} | "
            f"{_cell(row['statement'])} | {row['status'] or '-'} | {_cell(evidence)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _regressions_report(store: AnalyticsStore) -> str:
    rows = store.rows("SELECT * FROM regressions ORDER BY detected_at DESC")
    lines = ["# BitGN ECOM Regressions", "", "| regression_id | task_id | previous best | current | suspected change | notes |", "| --- | --- | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['regression_id']} | {row['task_id']} | {_fmt(row['previous_best_score'])} | "
            f"{_fmt(row['current_score'])} | {row['suspected_cause_change_id'] or '-'} | {_cell(row['notes'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _task_summary_rows(store: AnalyticsStore) -> list[Any]:
    return store.rows(
        """
        SELECT tc.*
        FROM task_classifications tc
        JOIN (
            SELECT task_id, MAX(created_at) AS created_at
            FROM task_classifications
            GROUP BY task_id
        ) latest ON latest.task_id = tc.task_id AND latest.created_at = tc.created_at
        ORDER BY tc.task_id
        """
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:0.3f}"
    return str(value)


def _cell(value: Any) -> str:
    return _one(value).replace("|", "\\|")


def _one(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text or "-"
