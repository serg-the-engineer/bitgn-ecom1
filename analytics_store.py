from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class AnalyticsStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta(version INTEGER NOT NULL);

            CREATE TABLE IF NOT EXISTS runs(
              local_run_id TEXT PRIMARY KEY,
              bitgn_run_id TEXT,
              benchmark_id TEXT,
              model_id TEXT,
              provider TEXT,
              started_at TEXT,
              ended_at TEXT,
              status TEXT,
              command TEXT,
              git_sha TEXT,
              git_dirty INTEGER,
              git_diff_hash TEXT,
              py_version TEXT,
              uv_lock_hash TEXT,
              env_json TEXT,
              total_score REAL,
              task_count INTEGER,
              notes TEXT
            );

            CREATE TABLE IF NOT EXISTS trials(
              local_trial_id TEXT PRIMARY KEY,
              local_run_id TEXT NOT NULL,
              bitgn_trial_id TEXT,
              task_id TEXT NOT NULL,
              attempt_no INTEGER NOT NULL,
              attempt_id TEXT NOT NULL,
              instruction TEXT NOT NULL,
              instruction_hash TEXT NOT NULL,
              harness_url_hash TEXT,
              started_at TEXT,
              ended_at TEXT,
              status TEXT,
              agent_outcome TEXT,
              score_available INTEGER,
              score REAL,
              score_detail_json TEXT,
              exception_json TEXT,
              classification_id TEXT,
              failure_analysis_id TEXT,
              active_fix_id TEXT,
              active_hypothesis_id TEXT
            );

            CREATE TABLE IF NOT EXISTS events(
              event_id TEXT PRIMARY KEY,
              seq INTEGER NOT NULL,
              local_run_id TEXT,
              local_trial_id TEXT,
              task_id TEXT,
              attempt_id TEXT,
              step_id TEXT,
              parent_event_id TEXT,
              event_type TEXT NOT NULL,
              name TEXT NOT NULL,
              status TEXT NOT NULL,
              ts_start TEXT NOT NULL,
              ts_end TEXT,
              duration_ms INTEGER,
              input_preview TEXT,
              output_preview TEXT,
              input_hash TEXT,
              output_hash TEXT,
              input_artifact_path TEXT,
              output_artifact_path TEXT,
              error_json TEXT,
              counts_json TEXT,
              tags_json TEXT
            );

            CREATE TABLE IF NOT EXISTS llm_calls(
              llm_call_id TEXT PRIMARY KEY,
              local_run_id TEXT,
              local_trial_id TEXT,
              task_id TEXT,
              attempt_id TEXT,
              step_id TEXT,
              purpose TEXT,
              provider TEXT,
              model TEXT,
              started_at TEXT,
              ended_at TEXT,
              duration_ms INTEGER,
              prompt_chars INTEGER,
              completion_chars INTEGER,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              total_tokens INTEGER,
              approximate_tokens INTEGER,
              schema_hash TEXT,
              prompt_hash TEXT,
              raw_output_hash TEXT,
              parsed_output_json TEXT,
              raw_artifact_path TEXT,
              prompt_artifact_path TEXT,
              stdout_artifact_path TEXT,
              stderr_artifact_path TEXT,
              returncode INTEGER,
              status TEXT,
              error_json TEXT
            );

            CREATE TABLE IF NOT EXISTS tool_calls(
              tool_call_id TEXT PRIMARY KEY,
              local_run_id TEXT,
              local_trial_id TEXT,
              task_id TEXT,
              attempt_id TEXT,
              step_id TEXT,
              tool_scope TEXT,
              tool_name TEXT,
              started_at TEXT,
              ended_at TEXT,
              duration_ms INTEGER,
              input_json TEXT,
              output_preview TEXT,
              output_hash TEXT,
              output_artifact_path TEXT,
              status TEXT,
              error_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages(
              message_id TEXT PRIMARY KEY,
              local_trial_id TEXT,
              step_id TEXT,
              message_index INTEGER,
              role TEXT,
              content_preview TEXT,
              content_hash TEXT,
              content_artifact_path TEXT,
              tool_call_id TEXT
            );

            CREATE TABLE IF NOT EXISTS task_classifications(
              classification_id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL,
              instruction_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              model TEXT,
              provider TEXT,
              task_family TEXT,
              task_subtype TEXT,
              objective_summary TEXT,
              main_difficulty TEXT,
              expected_action_type TEXT,
              likely_tools_json TEXT,
              risk_flags_json TEXT,
              reusable_lessons_json TEXT,
              confidence REAL,
              raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS failure_analyses(
              failure_analysis_id TEXT PRIMARY KEY,
              local_trial_id TEXT NOT NULL,
              task_id TEXT NOT NULL,
              attempt_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              model TEXT,
              provider TEXT,
              failure_mode TEXT,
              root_cause_summary TEXT,
              evidence_json TEXT,
              missed_observations_json TEXT,
              bad_tool_calls_json TEXT,
              proposed_hypotheses_json TEXT,
              recommended_single_change TEXT,
              raw_json TEXT
            );

            CREATE TABLE IF NOT EXISTS hypotheses(
              hypothesis_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              updated_at TEXT,
              task_id TEXT,
              task_family TEXT,
              source_failure_analysis_id TEXT,
              statement TEXT NOT NULL,
              expected_effect TEXT,
              scope TEXT,
              status TEXT,
              evidence_for_json TEXT,
              evidence_against_json TEXT,
              attempts_json TEXT,
              superseded_by TEXT
            );

            CREATE TABLE IF NOT EXISTS changes(
              change_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              finished_at TEXT,
              task_id TEXT,
              attempt_id TEXT,
              hypothesis_id TEXT,
              failure_analysis_id TEXT,
              title TEXT,
              description TEXT,
              major_change_count INTEGER,
              files_touched_json TEXT,
              git_sha_before TEXT,
              git_sha_after TEXT,
              git_diff_hash_before TEXT,
              git_diff_hash_after TEXT,
              diff_artifact_path TEXT,
              status TEXT,
              result_summary TEXT
            );

            CREATE TABLE IF NOT EXISTS regression_runs(
              regression_run_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              trigger_change_id TEXT,
              trigger_hypothesis_id TEXT,
              trigger_task_id TEXT,
              local_run_id TEXT,
              task_ids_json TEXT,
              status TEXT,
              summary_json TEXT
            );

            CREATE TABLE IF NOT EXISTS regressions(
              regression_id TEXT PRIMARY KEY,
              regression_run_id TEXT,
              task_id TEXT,
              previous_best_score REAL,
              current_score REAL,
              severity TEXT,
              detected_at TEXT,
              suspected_cause_change_id TEXT,
              suspected_cause_hypothesis_id TEXT,
              notes TEXT
            );

            CREATE TABLE IF NOT EXISTS skill_snapshots(
              snapshot_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              title TEXT,
              skills_hash TEXT,
              paths_json TEXT,
              artifact_path TEXT,
              status TEXT
            );

            CREATE TABLE IF NOT EXISTS generalization_findings(
              finding_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              source TEXT,
              path TEXT,
              severity TEXT,
              kind TEXT,
              atom TEXT,
              message TEXT,
              line_no INTEGER,
              status TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trials_task_id ON trials(task_id);
            CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(local_run_id);
            CREATE INDEX IF NOT EXISTS idx_trials_attempt ON trials(attempt_id);
            CREATE INDEX IF NOT EXISTS idx_trials_score ON trials(score);
            CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);

            CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
            CREATE INDEX IF NOT EXISTS idx_events_run ON events(local_run_id);
            CREATE INDEX IF NOT EXISTS idx_events_trial ON events(local_trial_id);
            CREATE INDEX IF NOT EXISTS idx_events_attempt ON events(attempt_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);

            CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_calls(local_run_id);
            CREATE INDEX IF NOT EXISTS idx_llm_trial ON llm_calls(local_trial_id);
            CREATE INDEX IF NOT EXISTS idx_llm_task ON llm_calls(task_id);
            CREATE INDEX IF NOT EXISTS idx_llm_status ON llm_calls(status);

            CREATE INDEX IF NOT EXISTS idx_tool_run ON tool_calls(local_run_id);
            CREATE INDEX IF NOT EXISTS idx_tool_trial ON tool_calls(local_trial_id);
            CREATE INDEX IF NOT EXISTS idx_tool_task ON tool_calls(task_id);
            CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_calls(tool_name);
            CREATE INDEX IF NOT EXISTS idx_tool_status ON tool_calls(status);

            CREATE INDEX IF NOT EXISTS idx_messages_trial ON messages(local_trial_id);
            CREATE INDEX IF NOT EXISTS idx_class_task ON task_classifications(task_id);
            CREATE INDEX IF NOT EXISTS idx_class_family ON task_classifications(task_family);
            CREATE INDEX IF NOT EXISTS idx_class_created ON task_classifications(created_at);
            CREATE INDEX IF NOT EXISTS idx_failure_task ON failure_analyses(task_id);
            CREATE INDEX IF NOT EXISTS idx_failure_trial ON failure_analyses(local_trial_id);
            CREATE INDEX IF NOT EXISTS idx_failure_created ON failure_analyses(created_at);
            CREATE INDEX IF NOT EXISTS idx_hyp_task ON hypotheses(task_id);
            CREATE INDEX IF NOT EXISTS idx_hyp_family ON hypotheses(task_family);
            CREATE INDEX IF NOT EXISTS idx_hyp_status ON hypotheses(status);
            CREATE INDEX IF NOT EXISTS idx_hyp_created ON hypotheses(created_at);
            CREATE INDEX IF NOT EXISTS idx_changes_hyp ON changes(hypothesis_id);
            CREATE INDEX IF NOT EXISTS idx_changes_status ON changes(status);
            CREATE INDEX IF NOT EXISTS idx_changes_created ON changes(created_at);
            CREATE INDEX IF NOT EXISTS idx_reg_run ON regressions(regression_run_id);
            CREATE INDEX IF NOT EXISTS idx_reg_task ON regressions(task_id);
            CREATE INDEX IF NOT EXISTS idx_skill_snap_created ON skill_snapshots(created_at);
            CREATE INDEX IF NOT EXISTS idx_generalization_path ON generalization_findings(path);
            CREATE INDEX IF NOT EXISTS idx_generalization_status ON generalization_findings(status);
            CREATE INDEX IF NOT EXISTS idx_generalization_created ON generalization_findings(created_at);
            """
        )
        row = self.conn.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
        elif row["version"] > SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported analytics schema version {row['version']}; expected {SCHEMA_VERSION}"
            )
        elif row["version"] < SCHEMA_VERSION:
            self.conn.execute("UPDATE schema_meta SET version=?", (SCHEMA_VERSION,))
        self.conn.commit()

    def insert_or_replace(self, table: str, values: dict[str, Any]) -> None:
        keys = list(values.keys())
        placeholders = ", ".join("?" for _ in keys)
        columns = ", ".join(keys)
        sql = f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})"
        self.conn.execute(sql, [values[key] for key in keys])
        self.conn.commit()

    def insert(self, table: str, values: dict[str, Any]) -> None:
        keys = list(values.keys())
        placeholders = ", ".join("?" for _ in keys)
        columns = ", ".join(keys)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.conn.execute(sql, [values[key] for key in keys])
        self.conn.commit()

    def update(self, table: str, key_column: str, key_value: Any, values: dict[str, Any]) -> None:
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        sql = f"UPDATE {table} SET {assignments} WHERE {key_column}=?"
        self.conn.execute(sql, [*values.values(), key_value])
        self.conn.commit()

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def row(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def next_attempt_no(self, task_id: str) -> int:
        row = self.row(
            "SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_no FROM trials WHERE task_id=?",
            (task_id,),
        )
        return int(row["next_no"] if row else 1)

    def known_task_ids(self) -> list[str]:
        rows = self.rows(
            "SELECT DISTINCT task_id FROM trials UNION SELECT DISTINCT task_id FROM task_classifications ORDER BY task_id"
        )
        return [row["task_id"] for row in rows if row["task_id"]]

    def previous_best_scores(self, exclude_run_id: str | None = None) -> dict[str, float]:
        params: tuple[Any, ...] = ()
        where = "WHERE score_available=1"
        if exclude_run_id:
            where += " AND local_run_id != ?"
            params = (exclude_run_id,)
        rows = self.rows(
            f"SELECT task_id, MAX(score) AS best_score FROM trials {where} GROUP BY task_id",
            params,
        )
        return {row["task_id"]: float(row["best_score"]) for row in rows if row["best_score"] is not None}

    def latest_classification(self, task_id: str) -> sqlite3.Row | None:
        return self.row(
            """
            SELECT * FROM task_classifications
            WHERE task_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (task_id,),
        )

    def active_changes(self) -> list[sqlite3.Row]:
        return self.rows(
            "SELECT * FROM changes WHERE status IN ('started', 'active', 'testing') ORDER BY created_at DESC"
        )

    def trial_counters(self, local_trial_id: str) -> dict[str, Any]:
        llm = self.row(
            """
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(prompt_chars), 0) AS prompt_chars,
                   COALESCE(SUM(completion_chars), 0) AS completion_chars,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(approximate_tokens), 0) AS approximate_tokens
            FROM llm_calls
            WHERE local_trial_id=?
            """,
            (local_trial_id,),
        )
        tool_rows = self.rows(
            "SELECT tool_name, COUNT(*) AS n FROM tool_calls WHERE local_trial_id=? GROUP BY tool_name",
            (local_trial_id,),
        )
        sql_exec = self.row(
            """
            SELECT COUNT(*) AS n FROM tool_calls
            WHERE local_trial_id=? AND tool_name='exec' AND input_json LIKE '%/bin/sql%'
            """,
            (local_trial_id,),
        )
        writes = self.row(
            "SELECT COUNT(*) AS n FROM tool_calls WHERE local_trial_id=? AND tool_name='write'",
            (local_trial_id,),
        )
        deletes = self.row(
            "SELECT COUNT(*) AS n FROM tool_calls WHERE local_trial_id=? AND tool_name='delete'",
            (local_trial_id,),
        )
        errors = self.row(
            "SELECT COUNT(*) AS n FROM tool_calls WHERE local_trial_id=? AND status='error'",
            (local_trial_id,),
        )
        steps = self.row(
            "SELECT COUNT(*) AS n FROM events WHERE local_trial_id=? AND event_type='agent.step.started'",
            (local_trial_id,),
        )
        trial = self.row("SELECT * FROM trials WHERE local_trial_id=?", (local_trial_id,))
        by_name = {row["tool_name"]: int(row["n"]) for row in tool_rows}
        return {
            "llm_calls": int(llm["calls"] if llm else 0),
            "tool_calls_total": sum(by_name.values()),
            "tool_calls_by_name": by_name,
            "sql_exec_calls": int(sql_exec["n"] if sql_exec else 0),
            "write_calls": int(writes["n"] if writes else 0),
            "delete_calls": int(deletes["n"] if deletes else 0),
            "connect_errors": int(errors["n"] if errors else 0),
            "agent_steps": int(steps["n"] if steps else 0),
            "prompt_chars": int(llm["prompt_chars"] if llm else 0),
            "completion_chars": int(llm["completion_chars"] if llm else 0),
            "prompt_tokens": int(llm["prompt_tokens"] if llm else 0),
            "completion_tokens": int(llm["completion_tokens"] if llm else 0),
            "approximate_tokens": int(llm["approximate_tokens"] if llm else 0),
            "final_outcome": trial["agent_outcome"] if trial else None,
            "score": trial["score"] if trial else None,
        }

    def run_counters(self, local_run_id: str) -> dict[str, Any]:
        score_rows = self.rows(
            "SELECT score_available, score FROM trials WHERE local_run_id=?",
            (local_run_id,),
        )
        known_scores = [float(row["score"]) for row in score_rows if row["score_available"] and row["score"] is not None]
        passed = sum(1 for score in known_scores if score >= 1.0)
        failed = sum(1 for score in known_scores if score < 1.0)
        unknown = sum(1 for row in score_rows if not row["score_available"])
        llm = self.row("SELECT COUNT(*) AS n FROM llm_calls WHERE local_run_id=?", (local_run_id,))
        tools = self.row("SELECT COUNT(*) AS n FROM tool_calls WHERE local_run_id=?", (local_run_id,))
        regs = self.row(
            """
            SELECT COUNT(*) AS n FROM regressions r
            JOIN regression_runs rr ON rr.regression_run_id = r.regression_run_id
            WHERE rr.local_run_id=?
            """,
            (local_run_id,),
        )
        return {
            "tasks": len(score_rows),
            "passed": passed,
            "failed": failed,
            "unknown": unknown,
            "mean_score": (sum(known_scores) / len(known_scores)) if known_scores else None,
            "llm_calls": int(llm["n"] if llm else 0),
            "tool_calls_total": int(tools["n"] if tools else 0),
            "regressions": int(regs["n"] if regs else 0),
        }
