from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from analytics_store import AnalyticsStore, json_dumps, json_loads
from generalization_guard import scan_paths
from observability import Recorder, git_snapshot, stable_hash, utc_now
from reports import generate_reports, hypotheses_text, summary_text, tasks_text
from skill_classifier import render_skill_context, route_skills
from skill_registry import SKILLS_DIR, list_skills


def main() -> None:
    parser = argparse.ArgumentParser(description="BitGN ECOM local analytics CLI")
    parser.add_argument("--obs-dir", default=".bitgn_obs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary")
    sub.add_parser("tasks")
    task_parser = sub.add_parser("task")
    task_parser.add_argument("task_id")
    sub.add_parser("failures")
    sub.add_parser("hypotheses")
    sub.add_parser("regressions")
    sub.add_parser("report")
    sub.add_parser("skills")
    sub.add_parser("evolution")

    route_parser = sub.add_parser("route-skills")
    route_parser.add_argument("--task-text", required=True)
    route_parser.add_argument("--family")

    check_generalization = sub.add_parser("check-generalization")
    check_generalization.add_argument("paths", nargs="*")
    check_generalization.add_argument("--no-record", action="store_true")

    snapshot_skills = sub.add_parser("snapshot-skills")
    snapshot_skills.add_argument("--title", default="runtime skills snapshot")

    new_fix = sub.add_parser("new-fix")
    new_fix.add_argument("--task", required=True)
    new_fix.add_argument("--hypothesis", required=True)
    new_fix.add_argument("--title", required=True)
    new_fix.add_argument("--force", action="store_true")

    finish_fix = sub.add_parser("finish-fix")
    finish_fix.add_argument("--change", required=True)
    finish_fix.add_argument("--status", required=True)
    finish_fix.add_argument("--summary", required=True)

    args = parser.parse_args()
    obs_dir = Path(args.obs_dir)

    if args.command == "summary":
        print(summary_text(obs_dir))
    elif args.command == "tasks":
        print(tasks_text(obs_dir))
    elif args.command == "task":
        print(task_detail(obs_dir, args.task_id))
    elif args.command == "failures":
        print(failures_text(obs_dir))
    elif args.command == "hypotheses":
        print(hypotheses_text(obs_dir))
    elif args.command == "regressions":
        print(regressions_text(obs_dir))
    elif args.command == "report":
        outputs = generate_reports(obs_dir=obs_dir)
        print("Generated reports:")
        for path in outputs.values():
            print(f"- {path}")
    elif args.command == "skills":
        print(skills_text())
    elif args.command == "route-skills":
        classification = {"task_family": args.family, "risk_flags_json": "[]"} if args.family else None
        route = route_skills(task_text=args.task_text, classification=classification)
        print(render_skill_context(route))
    elif args.command == "check-generalization":
        print(check_generalization_command(obs_dir, args.paths, record=not args.no_record))
    elif args.command == "snapshot-skills":
        print(snapshot_skills_command(obs_dir, args.title))
    elif args.command == "evolution":
        print(evolution_text(obs_dir))
    elif args.command == "new-fix":
        print(new_fix_command(obs_dir, args.task, args.hypothesis, args.title, force=args.force))
    elif args.command == "finish-fix":
        print(finish_fix_command(obs_dir, args.change, args.status, args.summary))


def task_detail(obs_dir: Path, task_id: str) -> str:
    store = AnalyticsStore(obs_dir / "obs.db")
    classification = store.latest_classification(task_id)
    trials = store.rows(
        """
        SELECT local_run_id, attempt_no, status, agent_outcome, score_available, score, active_hypothesis_id
        FROM trials
        WHERE task_id=?
        ORDER BY started_at DESC
        LIMIT 20
        """,
        (task_id,),
    )
    failures = store.rows(
        """
        SELECT failure_analysis_id, failure_mode, root_cause_summary, recommended_single_change
        FROM failure_analyses
        WHERE task_id=?
        ORDER BY created_at DESC
        LIMIT 10
        """,
        (task_id,),
    )
    hypotheses = store.rows(
        """
        SELECT hypothesis_id, status, statement
        FROM hypotheses
        WHERE task_id=?
        ORDER BY created_at DESC
        LIMIT 20
        """,
        (task_id,),
    )
    lines = [f"# Task {task_id}", ""]
    if classification:
        lines.extend(
            [
                f"Family: {classification['task_family']}",
                f"Subtype: {classification['task_subtype']}",
                f"Objective: {classification['objective_summary']}",
                f"Main difficulty: {classification['main_difficulty']}",
                f"Likely tools: {', '.join(json_loads(classification['likely_tools_json'], []))}",
                "",
            ]
        )
    if trials:
        lines.extend(["## Trials", "run | attempt | status | outcome | score | hypothesis", "--- | ---: | --- | --- | ---: | ---"])
        for row in trials:
            score = "-" if not row["score_available"] else f"{row['score']:0.3f}"
            lines.append(
                f"{row['local_run_id']} | {row['attempt_no']} | {row['status'] or '-'} | "
                f"{row['agent_outcome'] or '-'} | {score} | {row['active_hypothesis_id'] or '-'}"
            )
        lines.append("")
    if failures:
        lines.extend(["## Failures", "id | mode | root cause | recommended change", "--- | --- | --- | ---"])
        for row in failures:
            lines.append(
                f"{row['failure_analysis_id']} | {row['failure_mode']} | "
                f"{_one(row['root_cause_summary'])} | {_one(row['recommended_single_change'])}"
            )
        lines.append("")
    if hypotheses:
        lines.extend(["## Hypotheses", "id | status | statement", "--- | --- | ---"])
        for row in hypotheses:
            lines.append(f"{row['hypothesis_id']} | {row['status']} | {_one(row['statement'])}")
    store.close()
    return "\n".join(lines).strip() or f"No data for task {task_id}."


def failures_text(obs_dir: Path) -> str:
    store = AnalyticsStore(obs_dir / "obs.db")
    rows = store.rows(
        """
        SELECT task_id, failure_mode, root_cause_summary, recommended_single_change
        FROM failure_analyses
        ORDER BY created_at DESC
        LIMIT 50
        """
    )
    if not rows:
        store.close()
        return "No failures found."
    lines = ["task_id | failure_mode | root cause | recommended change", "--- | --- | --- | ---"]
    for row in rows:
        lines.append(
            f"{row['task_id']} | {row['failure_mode']} | "
            f"{_one(row['root_cause_summary'])} | {_one(row['recommended_single_change'])}"
        )
    store.close()
    return "\n".join(lines)


def regressions_text(obs_dir: Path) -> str:
    store = AnalyticsStore(obs_dir / "obs.db")
    rows = store.rows(
        """
        SELECT regression_id, task_id, previous_best_score, current_score, severity, notes
        FROM regressions
        ORDER BY detected_at DESC
        LIMIT 50
        """
    )
    if not rows:
        store.close()
        return "No regressions found."
    lines = ["regression_id | task_id | previous best | current | severity | notes", "--- | --- | ---: | ---: | --- | ---"]
    for row in rows:
        lines.append(
            f"{row['regression_id']} | {row['task_id']} | {row['previous_best_score']:0.3f} | "
            f"{row['current_score']:0.3f} | {row['severity']} | {_one(row['notes'])}"
        )
    store.close()
    return "\n".join(lines)


def new_fix_command(
    obs_dir: Path,
    task_id: str,
    hypothesis_id: str,
    title: str,
    *,
    force: bool = False,
) -> str:
    store = AnalyticsStore(obs_dir / "obs.db")
    active = store.active_changes()
    if active and not force:
        active_ids = ", ".join(row["change_id"] for row in active)
        store.close()
        raise SystemExit(f"Active fix already exists: {active_ids}. Pass --force to override.")
    hyp = store.row("SELECT * FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,))
    if hyp is None:
        store.close()
        raise SystemExit(f"Unknown hypothesis: {hypothesis_id}")
    failure = None
    if hyp["source_failure_analysis_id"]:
        failure = store.row(
            "SELECT * FROM failure_analyses WHERE failure_analysis_id=?",
            (hyp["source_failure_analysis_id"],),
        )
    fix_token = uuid.uuid4().hex[:8]
    change_id = f"fix_{fix_token}"
    snap = git_snapshot()
    created_at = utc_now()
    workitems_dir = obs_dir / "workitems"
    workitems_dir.mkdir(parents=True, exist_ok=True)
    workitem = workitems_dir / f"fix_{fix_token}.md"
    raw_failure = json_loads(failure["raw_json"], {}) if failure else {}
    regression_risks = "; ".join(raw_failure.get("regression_risks", [])) if failure else "-"
    workitem.write_text(
        f"""# Fix {change_id}

Task: {task_id}
Attempt: -
Hypothesis: {hypothesis_id}

## Hypothesis
{hyp['statement']}

## Expected effect
{hyp['expected_effect'] or '-'}

## Recommended single change
{failure['recommended_single_change'] if failure else hyp['statement']}

## Regression risks
{regression_risks or '-'}

## Implementation constraint
Make only one major change. Avoid bundling unrelated prompt, tool, and orchestration changes.

## Generalization constraint
Runtime skills, rules, and guards must not include task-specific ids, dates,
runtime evidence paths, local host paths, exact command arguments, or copied
task-answer facts. Run:

```sh
uv run python analytics_cli.py check-generalization
```
"""
    )
    store.insert(
        "changes",
        {
            "change_id": change_id,
            "created_at": created_at,
            "finished_at": None,
            "task_id": task_id,
            "attempt_id": None,
            "hypothesis_id": hypothesis_id,
            "failure_analysis_id": hyp["source_failure_analysis_id"],
            "title": title,
            "description": str(workitem),
            "major_change_count": 1,
            "files_touched_json": json_dumps([]),
            "git_sha_before": snap["git_sha"],
            "git_sha_after": None,
            "git_diff_hash_before": snap["git_diff_hash"],
            "git_diff_hash_after": None,
            "diff_artifact_path": None,
            "status": "started",
            "result_summary": None,
        },
    )
    store.update(
        "hypotheses",
        "hypothesis_id",
        hypothesis_id,
        {"status": "applied", "updated_at": created_at},
    )
    store.close()
    recorder = Recorder(obs_dir)
    recorder.record_event(
        event_id=f"evt_{uuid.uuid4().hex[:12]}",
        event_type="change.started",
        name=change_id,
        status="ok",
        ts_start=created_at,
        ts_end=created_at,
        duration_ms=0,
        task_id=task_id,
        output_json={"change_id": change_id, "hypothesis_id": hypothesis_id, "workitem": str(workitem)},
    )
    recorder.record_event(
        event_id=f"evt_{uuid.uuid4().hex[:12]}",
        event_type="hypothesis.applied",
        name=hypothesis_id,
        status="ok",
        ts_start=created_at,
        ts_end=created_at,
        duration_ms=0,
        task_id=task_id,
        output_json={"change_id": change_id, "hypothesis_id": hypothesis_id},
    )
    recorder.close()
    return f"Created {change_id}\nWorkitem: {workitem}"


def finish_fix_command(obs_dir: Path, change_id: str, status: str, summary: str) -> str:
    store = AnalyticsStore(obs_dir / "obs.db")
    change = store.row("SELECT * FROM changes WHERE change_id=?", (change_id,))
    if change is None:
        store.close()
        raise SystemExit(f"Unknown change: {change_id}")
    snap = git_snapshot()
    files = _changed_files()
    guard_paths = files or [str(path) for path in _default_generalization_paths()]
    guard_text = check_generalization_command(obs_dir, guard_paths, record=True)
    if "FAILED" in guard_text:
        store.close()
        raise SystemExit(guard_text)
    diff = _full_diff()
    diff_path = obs_dir / "workitems" / f"{change_id}_diff.patch"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(diff)
    finished_at = utc_now()
    store.update(
        "changes",
        "change_id",
        change_id,
        {
            "finished_at": finished_at,
            "files_touched_json": json_dumps(files),
            "git_sha_after": snap["git_sha"],
            "git_diff_hash_after": stable_hash(diff),
            "diff_artifact_path": str(diff_path),
            "status": status,
            "result_summary": summary,
        },
    )
    hypothesis_id = change["hypothesis_id"]
    if hypothesis_id:
        store.update(
            "hypotheses",
            "hypothesis_id",
            hypothesis_id,
            {"status": status, "updated_at": finished_at},
        )
    store.close()
    recorder = Recorder(obs_dir)
    recorder.record_event(
        event_id=f"evt_{uuid.uuid4().hex[:12]}",
        event_type="change.finished",
        name=change_id,
        status="ok",
        ts_start=finished_at,
        ts_end=finished_at,
        duration_ms=0,
        task_id=change["task_id"],
        output_json={
            "change_id": change_id,
            "status": status,
            "summary": summary,
            "files_touched": files,
            "diff_artifact_path": str(diff_path),
        },
    )
    recorder.close()
    return f"Finished {change_id} as {status}\nDiff artifact: {diff_path}\n{guard_text}"


def skills_text() -> str:
    lines = ["skill_id | title | description | path", "--- | --- | --- | ---"]
    for skill in list_skills():
        lines.append(f"{skill.skill_id} | {skill.title} | {skill.description} | {skill.path}")
    return "\n".join(lines)


def check_generalization_command(
    obs_dir: Path,
    paths: list[str],
    *,
    record: bool,
) -> str:
    target_paths = [Path(path) for path in paths] if paths else _default_generalization_paths()
    reports = scan_paths(target_paths)
    if record:
        _record_generalization_reports(obs_dir, reports)
    failed = {path: report for path, report in reports.items() if not report.ok}
    if not failed:
        return "Generalization guard: OK"
    lines = ["Generalization guard: FAILED", ""]
    for path, report in failed.items():
        lines.append(f"## {path}")
        for finding in report.findings:
            loc = f":{finding.line_no}" if finding.line_no else ""
            lines.append(f"- [{finding.severity}] {finding.kind}{loc}: `{finding.atom}` - {finding.message}")
        lines.append("")
    return "\n".join(lines).strip()


def snapshot_skills_command(obs_dir: Path, title: str) -> str:
    guard_text = check_generalization_command(obs_dir, [str(SKILLS_DIR)], record=True)
    if "FAILED" in guard_text:
        raise SystemExit(guard_text)
    snapshot_id = f"skills_{uuid.uuid4().hex[:8]}"
    created_at = utc_now()
    root = obs_dir / "evolution" / "snapshots" / snapshot_id
    root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(SKILLS_DIR.glob("*.md")):
        dst = root / "skills" / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied.append(str(dst))
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "title": title,
        "paths": copied,
        "skills_hash": stable_hash("\n".join(Path(path).read_text(encoding="utf-8") for path in sorted(SKILLS_DIR.glob("*.md")))),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store = AnalyticsStore(obs_dir / "obs.db")
    store.insert(
        "skill_snapshots",
        {
            "snapshot_id": snapshot_id,
            "created_at": created_at,
            "title": title,
            "skills_hash": manifest["skills_hash"],
            "paths_json": json_dumps(copied),
            "artifact_path": str(root),
            "status": "active_snapshot",
        },
    )
    store.close()
    return f"Created {snapshot_id}\nSnapshot: {root}\n{guard_text}"


def evolution_text(obs_dir: Path) -> str:
    store = AnalyticsStore(obs_dir / "obs.db")
    active = store.active_changes()
    snapshots = store.rows(
        "SELECT snapshot_id, created_at, title, skills_hash, status FROM skill_snapshots ORDER BY created_at DESC LIMIT 10"
    )
    findings = store.rows(
        "SELECT path, kind, atom, message, created_at FROM generalization_findings WHERE status='open' ORDER BY created_at DESC LIMIT 20"
    )
    lines = [
        "# Evolution Loop",
        "",
        "Unit: one hypothesis -> one major change -> targeted rerun -> risk cluster -> full monitor run.",
        "",
        "Generalization rule: runtime rules and skills may only contain reusable workflow policy, never task ids, object ids, dates, exact evidence paths, or copied command sequences.",
        "",
        "## Active Changes",
    ]
    if active:
        for row in active:
            lines.append(f"- {row['change_id']}: {row['title']} ({row['status']})")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Skill Snapshots")
    if snapshots:
        for row in snapshots:
            lines.append(f"- {row['snapshot_id']} {row['created_at']} {row['status']}: {row['title']} hash={row['skills_hash'][:12]}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Open Generalization Findings")
    if findings:
        for row in findings:
            lines.append(f"- {row['path']}: {row['kind']} `{row['atom']}` - {_one(row['message'])}")
    else:
        lines.append("- none")
    store.close()
    return "\n".join(lines)


def _git_text(args: list[str], *, ok_returncodes: tuple[int, ...] = (0,)) -> str | None:
    try:
        proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False, timeout=30)
    except Exception:
        return None
    if proc.returncode not in ok_returncodes:
        return None
    return proc.stdout


def _git_lines(args: list[str]) -> list[str]:
    text = _git_text(args)
    if not text:
        return []
    return [line for line in text.splitlines() if line.strip()]


def _changed_files() -> list[str]:
    files: set[str] = set()
    for args in (
        ["diff", "--name-only"],
        ["diff", "--name-only", "--cached"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        files.update(_git_lines(args))
    return sorted(files)


def _full_diff() -> str:
    parts = []
    for args in (
        ["diff", "--no-ext-diff", "--binary"],
        ["diff", "--cached", "--no-ext-diff", "--binary"],
    ):
        text = _git_text(args)
        if text:
            parts.append(text.rstrip())
    for path in _git_lines(["ls-files", "--others", "--exclude-standard"]):
        if not Path(path).is_file():
            continue
        text = _git_text(
            ["diff", "--no-index", "--", "/dev/null", path],
            ok_returncodes=(0, 1),
        )
        if text:
            parts.append(text.rstrip())
    return "\n\n".join(parts) + ("\n" if parts else "")


def _default_generalization_paths() -> list[Path]:
    names = [
        "agent",
        "skill_classifier.py",
        "skill_registry.py",
        "completion_guard.py",
        "tool_contract_guard.py",
        "authorization_guard.py",
        "generalization_guard.py",
        "skills",
    ]
    return [Path(name) for name in names if Path(name).exists()]


def _record_generalization_reports(obs_dir: Path, reports: dict[str, Any]) -> None:
    store = AnalyticsStore(obs_dir / "obs.db")
    now = utc_now()
    for path, report in reports.items():
        for finding in report.findings:
            store.insert(
                "generalization_findings",
                {
                    "finding_id": f"gen_{uuid.uuid4().hex[:12]}",
                    "created_at": now,
                    "source": "analytics_cli.check-generalization",
                    "path": path,
                    "severity": finding.severity,
                    "kind": finding.kind,
                    "atom": finding.atom,
                    "message": finding.message,
                    "line_no": finding.line_no,
                    "status": "open",
                },
            )
    store.close()


def _one(value: Any, limit: int = 140) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text or "-"


if __name__ == "__main__":
    main()
