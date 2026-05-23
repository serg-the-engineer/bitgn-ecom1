from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from observability import Recorder, sanitize, short_id, stable_hash, utc_now


T = TypeVar("T", bound=BaseModel)


class StructuredLLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw: str = "",
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def resolve_provider(provider: str | None = None) -> str:
    selected = (provider or os.getenv("LLM_PROVIDER", "codex")).lower()
    if selected == "auto":
        return "openai" if os.getenv("OPENAI_API_KEY") else "codex"
    return selected


def strict_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        out = {key: strict_json_schema(item) for key, item in value.items()}
        if out.get("type") == "object" or "properties" in out:
            out.setdefault("additionalProperties", False)
            if isinstance(out.get("properties"), dict):
                out["required"] = list(out["properties"].keys())
        return out
    if isinstance(value, list):
        return [strict_json_schema(item) for item in value]
    return value


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _prompt_text(messages_or_prompt: list[dict[str, Any]] | str) -> str:
    if isinstance(messages_or_prompt, str):
        return messages_or_prompt
    return json.dumps(messages_or_prompt, ensure_ascii=False, indent=2, default=str)


def render_codex_prompt(
    messages_or_prompt: list[dict[str, Any]] | str,
    *,
    purpose: str,
) -> str:
    if isinstance(messages_or_prompt, str):
        return f"""
Return exactly one JSON object matching the provided output schema.
Do not include markdown fences or commentary outside the JSON.

Task:
{messages_or_prompt}
""".strip()
    return f"""
You are the reasoning component inside a BitGN ECOM benchmark agent.

Return exactly one JSON object matching the provided output schema. Do not run
shell commands or inspect local files. The only available actions are the tools
encoded in the schema. Pick the next single tool call that best advances the
task, or use report_completion when the task is done or blocked.

Purpose: {purpose}

Conversation transcript:
{json.dumps(messages_or_prompt, ensure_ascii=False, indent=2, default=str)}
""".strip()


def _artifact_base(
    recorder: Recorder | None,
    context: dict[str, Any],
    llm_call_id: str,
) -> Path | None:
    if recorder is None:
        return None
    trial_dir = context.get("trial_dir")
    if trial_dir:
        return Path(trial_dir) / "llm_calls" / llm_call_id
    return recorder.run_dir / "artifacts" / llm_call_id


def run_structured_llm(
    *,
    provider: str,
    model: str,
    messages_or_prompt: list[dict[str, Any]] | str,
    response_model: type[T],
    purpose: str,
    recorder: Recorder | None,
    context: dict[str, Any],
) -> T:
    provider = resolve_provider(provider)
    llm_call_id = short_id("llm")
    local_trial_id = context.get("local_trial_id")
    task_id = context.get("task_id")
    attempt_id = context.get("attempt_id")
    step_id = context.get("step_id")
    canonical_prompt_text = _prompt_text(messages_or_prompt)
    provider_prompt_text = (
        render_codex_prompt(messages_or_prompt, purpose=purpose)
        if provider == "codex"
        else canonical_prompt_text
    )
    prompt_chars = len(provider_prompt_text)
    schema = strict_json_schema(response_model.model_json_schema())
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    schema_hash = stable_hash(schema_json)
    prompt_hash = stable_hash(provider_prompt_text)
    base = _artifact_base(recorder, context, llm_call_id)
    prompt_artifact_path = None
    raw_artifact_path = None
    stdout_artifact_path = None
    stderr_artifact_path = None
    started_at = utc_now()
    start_event_id = None
    if recorder:
        if base:
            prompt_artifact_path = recorder.write_text(f"{base}_prompt.txt", provider_prompt_text)
            if canonical_prompt_text != provider_prompt_text:
                recorder.write_text(f"{base}_canonical_messages.txt", canonical_prompt_text)
            recorder.write_text(f"{base}_schema.json", schema_json)
        llm_input = {
            "llm_call_id": llm_call_id,
            "purpose": purpose,
            "provider": provider,
            "model": model,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
        }
        if provider == "codex":
            llm_input.update(
                {
                    "command_args": [
                        "codex",
                        "exec",
                        "--skip-git-repo-check",
                        "--ephemeral",
                        "--sandbox",
                        "read-only",
                        "--model",
                        model,
                        "--output-schema",
                        "<temp_schema_path>",
                        "--output-last-message",
                        "<temp_output_path>",
                        "-",
                    ],
                    "timeout": int(os.getenv("CODEX_EXEC_TIMEOUT", "600")),
                }
            )
        elif provider == "openai":
            llm_input.update({"api": "client.beta.chat.completions.parse"})
        start_event_id = recorder.start_event(
            "llm.call.started",
            purpose,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            input_json=llm_input,
            tags_json=context.get("tags_json"),
        )

    status = "ok"
    error_json = None
    raw = ""
    parsed: T | None = None
    returncode = None
    stdout = ""
    stderr = ""
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    try:
        if provider == "codex":
            parsed, raw, stdout, stderr, returncode = _run_codex(
                model=model,
                prompt=provider_prompt_text,
                schema=schema,
                response_model=response_model,
            )
        elif provider == "openai":
            parsed, raw, prompt_tokens, completion_tokens, total_tokens = _run_openai(
                model=model,
                messages_or_prompt=messages_or_prompt,
                response_model=response_model,
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    except Exception as exc:
        status = "error"
        error_json = {"type": exc.__class__.__name__, "message": str(exc)}
        if isinstance(exc, StructuredLLMError):
            raw = exc.raw
            stdout = exc.stdout
            stderr = exc.stderr
            returncode = exc.returncode
        if recorder:
            ended_at = utc_now()
            if base:
                if raw:
                    raw_artifact_path = recorder.write_text(f"{base}_raw.txt", raw)
                if stdout:
                    stdout_artifact_path = recorder.write_text(f"{base}_stdout.txt", stdout)
                if stderr:
                    stderr_artifact_path = recorder.write_text(f"{base}_stderr.txt", stderr)
            recorder.finish_event(
                start_event_id,
                "llm.call.error",
                purpose,
                status="error",
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                step_id=step_id,
                error_json=error_json,
                output_json={"stderr_tail": "\n".join(stderr.splitlines()[-20:])},
                artifact_paths_json=[
                    item
                    for item in (raw_artifact_path, stdout_artifact_path, stderr_artifact_path)
                    if item
                ],
            )
            _record_llm_row(
                recorder,
                llm_call_id=llm_call_id,
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                step_id=step_id,
                purpose=purpose,
                provider=provider,
                model=model,
                started_at=started_at,
                ended_at=ended_at,
                prompt_chars=prompt_chars,
                completion_chars=len(raw),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                schema_hash=schema_hash,
                prompt_hash=prompt_hash,
                raw_output_hash=stable_hash(raw) if raw else None,
                parsed_output_json=None,
                raw_artifact_path=raw_artifact_path,
                prompt_artifact_path=prompt_artifact_path,
                stdout_artifact_path=stdout_artifact_path,
                stderr_artifact_path=stderr_artifact_path,
                returncode=returncode,
                status=status,
                error_json=error_json,
            )
        raise

    ended_at = utc_now()
    if recorder:
        if base:
            raw_artifact_path = recorder.write_text(f"{base}_raw.txt", raw)
            if stdout:
                stdout_artifact_path = recorder.write_text(f"{base}_stdout.txt", stdout)
            if stderr:
                stderr_artifact_path = recorder.write_text(f"{base}_stderr.txt", stderr)
        parsed_json = parsed.model_dump() if parsed else None
        recorder.finish_event(
            start_event_id,
            "llm.call.finished",
            purpose,
            status=status,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            output_json={
                "llm_call_id": llm_call_id,
                "raw_output_hash": stable_hash(raw),
                "returncode": returncode,
                "parsed": parsed_json,
            },
            counts_json={
                "prompt_chars": prompt_chars,
                "completion_chars": len(raw),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "approximate_tokens": math.ceil((prompt_chars + len(raw)) / 4),
                "token_count_source": "provider_usage" if total_tokens is not None else "approx_chars_div_4",
            },
            artifact_paths_json=[
                item
                for item in (prompt_artifact_path, raw_artifact_path, stdout_artifact_path, stderr_artifact_path)
                if item
            ],
        )
        recorder.record_event(
            event_id=short_id("evt"),
            event_type="llm.parsed_output",
            name=purpose,
            status="ok",
            ts_start=ended_at,
            ts_end=ended_at,
            duration_ms=0,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            output_json=parsed_json,
        )
        _record_llm_row(
            recorder,
            llm_call_id=llm_call_id,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            purpose=purpose,
            provider=provider,
            model=model,
            started_at=started_at,
            ended_at=ended_at,
            prompt_chars=prompt_chars,
            completion_chars=len(raw),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            schema_hash=schema_hash,
            prompt_hash=prompt_hash,
            raw_output_hash=stable_hash(raw),
            parsed_output_json=parsed_json,
            raw_artifact_path=raw_artifact_path,
            prompt_artifact_path=prompt_artifact_path,
            stdout_artifact_path=stdout_artifact_path,
            stderr_artifact_path=stderr_artifact_path,
            returncode=returncode,
            status=status,
            error_json=error_json,
        )
    return parsed


def _run_codex(
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    response_model: type[T],
) -> tuple[T, str, str, str, int]:
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as schema_file:
        schema_file.write(json.dumps(schema, ensure_ascii=False, indent=2))
        schema_path = schema_file.name
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as output:
        output_path = output.name
    timeout = int(os.getenv("CODEX_EXEC_TIMEOUT", "600"))
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--model",
        model,
        "--output-schema",
        schema_path,
        "--output-last-message",
        output_path,
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        raw = Path(output_path).read_text() if Path(output_path).exists() else ""
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-20:])
            raise StructuredLLMError(
                f"codex exec failed with return code {proc.returncode}:\n{tail}",
                raw=raw,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )
        try:
            parsed = response_model.model_validate_json(extract_json_object(raw))
        except Exception as exc:
            raise StructuredLLMError(
                f"codex output did not match {response_model.__name__}: {exc}",
                raw=raw,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            ) from exc
        return parsed, raw, proc.stdout, proc.stderr, proc.returncode
    finally:
        Path(schema_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def _run_openai(
    *,
    model: str,
    messages_or_prompt: list[dict[str, Any]] | str,
    response_model: type[T],
) -> tuple[T, str, int | None, int | None, int | None]:
    client = OpenAI()
    messages = (
        messages_or_prompt
        if isinstance(messages_or_prompt, list)
        else [{"role": "user", "content": messages_or_prompt}]
    )
    resp = client.beta.chat.completions.parse(
        model=model,
        response_format=response_model,
        messages=sanitize(messages),
        max_completion_tokens=16384,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI response did not contain parsed output")
    try:
        raw = resp.model_dump_json(indent=2)
    except Exception:
        raw = str(resp)
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None
    return parsed, raw, prompt_tokens, completion_tokens, total_tokens


def _record_llm_row(
    recorder: Recorder,
    *,
    llm_call_id: str,
    local_trial_id: str | None,
    task_id: str | None,
    attempt_id: str | None,
    step_id: str | None,
    purpose: str,
    provider: str,
    model: str,
    started_at: str,
    ended_at: str,
    prompt_chars: int,
    completion_chars: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    schema_hash: str,
    prompt_hash: str,
    raw_output_hash: str | None,
    parsed_output_json: dict[str, Any] | None,
    raw_artifact_path: str | None,
    prompt_artifact_path: str | None,
    stdout_artifact_path: str | None,
    stderr_artifact_path: str | None,
    returncode: int | None,
    status: str,
    error_json: dict[str, Any] | None,
) -> None:
    duration_ms = _duration_ms(started_at, ended_at)
    approximate_tokens = math.ceil((prompt_chars + completion_chars) / 4)
    recorder.record_llm_call(
        {
            "llm_call_id": llm_call_id,
            "local_run_id": recorder.local_run_id,
            "local_trial_id": local_trial_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "step_id": step_id,
            "purpose": purpose,
            "provider": provider,
            "model": model,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "approximate_tokens": approximate_tokens,
            "schema_hash": schema_hash,
            "prompt_hash": prompt_hash,
            "raw_output_hash": raw_output_hash,
            "parsed_output_json": parsed_output_json,
            "raw_artifact_path": raw_artifact_path,
            "prompt_artifact_path": prompt_artifact_path,
            "stdout_artifact_path": stdout_artifact_path,
            "stderr_artifact_path": stderr_artifact_path,
            "returncode": returncode,
            "status": status,
            "error_json": error_json,
        }
    )


def _duration_ms(ts_start: str, ts_end: str) -> int:
    from datetime import datetime

    start = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))
    return int((end - start).total_seconds() * 1000)
