from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from analytics_store import AnalyticsStore, json_dumps


SECRET_KEY_RE = re.compile(r"(key|token|secret|password)", re.IGNORECASE)
AUTH_RE = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
QUERY_SECRET_KEYS = {"api_key", "apikey", "access_token", "token", "key", "password", "secret"}
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
SAFE_COUNT_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "approximate_tokens",
    "max_completion_tokens",
    "token_count_source",
}
PREVIEW_LIMIT = 1200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_local_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid.uuid4().hex[:8]}"


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "item"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8", "replace")
    else:
        data = stable_json(value).encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


def file_hash(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc or not parts.query:
        return value
    query = []
    changed = False
    for key, item in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in QUERY_SECRET_KEYS or SECRET_KEY_RE.search(key):
            query.append((key, "[REDACTED]"))
            changed = True
        else:
            query.append((key, item))
    if not changed:
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _looks_like_harness_url(value: str) -> bool:
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    host = parts.netloc.lower()
    path = parts.path.lower()
    return "bitgn" in host and path.startswith("/vm/")


def _redact_harness_urls(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ".,;:)]}":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        if not _looks_like_harness_url(raw):
            return match.group(0)
        return f"[HARNESS_URL sha256={stable_hash(raw)}]{trailing}"

    return URL_RE.sub(replace, value)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_harness_url_key(value: str) -> bool:
    normalized = _normalized_key(value)
    return "harness" in normalized and "url" in normalized


def sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        redacted = AUTH_RE.sub(lambda match: f"{match.group(1)} [REDACTED]", value)
        return _redact_harness_urls(_redact_url(redacted))
    if isinstance(value, bytes):
        return f"<{len(value)} bytes sha256={stable_hash(value)}>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _is_harness_url_key(key_str):
                out["harness_url_hash"] = stable_hash(str(item))
            elif key_str.lower() in SAFE_COUNT_KEYS:
                out[key_str] = sanitize(item)
            elif key_str.lower() == "authorization" or SECRET_KEY_RE.search(key_str):
                out[key_str] = "[REDACTED]"
            else:
                out[key_str] = sanitize(item)
        return out
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump())
    return sanitize(str(value))


def preview(value: Any, limit: int = PREVIEW_LIMIT) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def safe_env_snapshot() -> dict[str, str]:
    prefixes = ("BITGN_", "BENCH", "MODEL_ID", "LLM_PROVIDER", "CODEX_", "OPENAI_", "PYTHON")
    selected = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(prefixes) or SECRET_KEY_RE.search(key)
    }
    return sanitize(selected)


def git_snapshot(cwd: str | Path = ".") -> dict[str, Any]:
    root = Path(cwd)

    def run_git(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    sha = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--porcelain"])
    diff = run_git(["diff", "--no-ext-diff", "--binary"])
    return {
        "git_sha": sha,
        "git_dirty": None if status is None else int(bool(status)),
        "git_diff_hash": stable_hash(diff) if diff is not None else None,
    }


def protobuf_to_dict(value: Any) -> Any:
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(value)
    except Exception:
        return sanitize(str(value))


class Recorder:
    def __init__(self, obs_dir: str | Path = ".bitgn_obs", local_run_id: str | None = None) -> None:
        self.obs_dir = Path(obs_dir)
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.store = AnalyticsStore(self.obs_dir / "obs.db")
        self.local_run_id = local_run_id or make_local_run_id()
        self.run_dir = self.obs_dir / "runs" / self.local_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "trials").mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.seq = 0
        self._active: dict[str, str] = {}

    def close(self) -> None:
        self.store.close()

    def rel(self, path: str | Path | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        try:
            return p.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return p.as_posix()

    def trial_dir(self, task_id: str, attempt_no: int) -> Path:
        name = f"{safe_slug(task_id)}__attempt_{attempt_no}"
        path = self.run_dir / "trials" / name
        for child in ("llm_calls", "tool_calls", "harness_calls", "artifacts"):
            (path / child).mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: str | Path, value: Any) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2, default=str))
        return self.rel(p) or str(p)

    def write_text(self, path: str | Path, value: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(sanitize(value) if isinstance(value, str) else str(value))
        return self.rel(p) or str(p)

    def append_jsonl(self, path: str | Path, value: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sanitize(value), ensure_ascii=False, default=str) + "\n")

    def start_event(
        self,
        event_type: str,
        name: str,
        *,
        local_trial_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        step_id: str | None = None,
        parent_event_id: str | None = None,
        input_json: Any = None,
        tags_json: Any = None,
    ) -> str:
        event_id = short_id("evt")
        ts = utc_now()
        self._active[event_id] = ts
        self.record_event(
            event_id=event_id,
            event_type=event_type,
            name=name,
            status="started",
            ts_start=ts,
            ts_end=None,
            duration_ms=None,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            parent_event_id=parent_event_id,
            input_json=input_json,
            tags_json=tags_json,
        )
        return event_id

    def finish_event(
        self,
        started_event_id: str | None,
        event_type: str,
        name: str,
        *,
        status: str = "ok",
        local_trial_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        step_id: str | None = None,
        output_json: Any = None,
        error_json: Any = None,
        counts_json: Any = None,
        tags_json: Any = None,
        artifact_paths_json: list[str] | None = None,
    ) -> str:
        now = utc_now()
        ts_start = self._active.pop(started_event_id, None) if started_event_id else None
        ts_start = ts_start or now
        duration_ms = _duration_ms(ts_start, now)
        event_id = short_id("evt")
        self.record_event(
            event_id=event_id,
            event_type=event_type,
            name=name,
            status=status,
            ts_start=ts_start,
            ts_end=now,
            duration_ms=duration_ms,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            parent_event_id=started_event_id,
            output_json=output_json,
            error_json=error_json,
            counts_json=counts_json,
            tags_json=tags_json,
            artifact_paths_json=artifact_paths_json,
        )
        return event_id

    def record_event(
        self,
        *,
        event_id: str,
        event_type: str,
        name: str,
        status: str,
        ts_start: str,
        ts_end: str | None,
        duration_ms: int | None,
        local_trial_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        step_id: str | None = None,
        parent_event_id: str | None = None,
        input_json: Any = None,
        output_json: Any = None,
        error_json: Any = None,
        counts_json: Any = None,
        tags_json: Any = None,
        artifact_paths_json: list[str] | None = None,
    ) -> None:
        self.seq += 1
        safe_input = sanitize(input_json)
        safe_output = sanitize(output_json)
        safe_error = sanitize(error_json)
        safe_counts = sanitize(counts_json)
        safe_tags = sanitize(tags_json)
        artifact_paths = list(artifact_paths_json or [])
        input_path = None
        output_path = None
        if safe_input is not None and len(preview(safe_input, 10_000) or "") >= 10_000:
            input_path = self.write_json(self.run_dir / "artifacts" / f"{event_id}_input.json", safe_input)
            artifact_paths.append(input_path)
        if safe_output is not None and len(preview(safe_output, 10_000) or "") >= 10_000:
            output_path = self.write_json(self.run_dir / "artifacts" / f"{event_id}_output.json", safe_output)
            artifact_paths.append(output_path)
        event = {
            "event_id": event_id,
            "seq": self.seq,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "duration_ms": duration_ms,
            "local_run_id": self.local_run_id,
            "local_trial_id": local_trial_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "step_id": step_id,
            "parent_event_id": parent_event_id,
            "event_type": event_type,
            "name": name,
            "status": status,
            "input_json": safe_input,
            "output_json": safe_output,
            "error_json": safe_error,
            "counts_json": safe_counts,
            "tags_json": safe_tags,
            "artifact_paths_json": artifact_paths,
            "input_hash": stable_hash(safe_input) if safe_input is not None else None,
            "output_hash": stable_hash(safe_output) if safe_output is not None else None,
        }
        self.append_jsonl(self.events_path, event)
        self.store.insert(
            "events",
            {
                "event_id": event_id,
                "seq": self.seq,
                "local_run_id": self.local_run_id,
                "local_trial_id": local_trial_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "step_id": step_id,
                "parent_event_id": parent_event_id,
                "event_type": event_type,
                "name": name,
                "status": status,
                "ts_start": ts_start,
                "ts_end": ts_end,
                "duration_ms": duration_ms,
                "input_preview": preview(safe_input),
                "output_preview": preview(safe_output),
                "input_hash": event["input_hash"],
                "output_hash": event["output_hash"],
                "input_artifact_path": input_path,
                "output_artifact_path": output_path,
                "error_json": json_dumps(safe_error),
                "counts_json": json_dumps(safe_counts),
                "tags_json": json_dumps(safe_tags),
            },
        )

    def record_llm_call(self, values: dict[str, Any]) -> None:
        safe_values = sanitize(values)
        parsed = safe_values.get("parsed_output_json")
        error = safe_values.get("error_json")
        row = {
            **safe_values,
            "local_run_id": safe_values.get("local_run_id") or self.local_run_id,
            "parsed_output_json": json_dumps(parsed),
            "error_json": json_dumps(error),
        }
        self.store.insert("llm_calls", row)

    def record_tool_call(self, values: dict[str, Any]) -> None:
        safe_values = sanitize(values)
        row = {
            **safe_values,
            "local_run_id": safe_values.get("local_run_id") or self.local_run_id,
            "input_json": json_dumps(safe_values.get("input_json")),
            "error_json": json_dumps(safe_values.get("error_json")),
        }
        self.store.insert("tool_calls", row)

    def record_message(
        self,
        *,
        local_trial_id: str,
        trial_dir: Path,
        message_index: int,
        role: str,
        content: str,
        step_id: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: Any = None,
    ) -> None:
        safe_content = sanitize(content or "")
        content_hash = stable_hash(safe_content)
        artifact_path = None
        if len(safe_content) > PREVIEW_LIMIT:
            artifact_path = self.write_text(
                trial_dir / "artifacts" / f"message_{message_index:04d}_{role}.txt",
                safe_content,
            )
        self.append_jsonl(
            trial_dir / "transcript.jsonl",
            {
                "message_index": message_index,
                "step_id": step_id,
                "role": role,
                "content": safe_content,
                "tool_calls": sanitize(tool_calls or []),
                "ts": utc_now(),
            },
        )
        self.store.insert(
            "messages",
            {
                "message_id": short_id("msg"),
                "local_trial_id": local_trial_id,
                "step_id": step_id,
                "message_index": message_index,
                "role": role,
                "content_preview": preview(safe_content),
                "content_hash": content_hash,
                "content_artifact_path": artifact_path,
                "tool_call_id": tool_call_id,
            },
        )


def _duration_ms(ts_start: str, ts_end: str) -> int:
    try:
        start = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))
        return int((end - start).total_seconds() * 1000)
    except Exception:
        return 0


def python_version() -> str:
    return sys.version.replace("\n", " ")
