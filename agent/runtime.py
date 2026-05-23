from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bitgn.vm.ecom.ecom_connect import EcomRuntimeClientSync
from bitgn.vm.ecom.ecom_pb2 import (
    AnswerRequest,
    DeleteRequest,
    ExecRequest,
    FindRequest,
    ListRequest,
    NodeKind,
    Outcome,
    ReadRequest,
    SearchRequest,
    StatRequest,
    TreeRequest,
    WriteRequest,
)
from connectrpc.errors import ConnectError
from pydantic import BaseModel

from observability import Recorder, protobuf_to_dict, short_id, stable_hash, utc_now

from .formatting import format_result
from .models import (
    ReportTaskCompletion,
    Req_Delete,
    Req_Exec,
    Req_Find,
    Req_List,
    Req_Read,
    Req_Search,
    Req_Stat,
    Req_Tree,
    Req_Write,
)


OUTCOME_BY_NAME = {
    "OUTCOME_OK": Outcome.OUTCOME_OK,
    "OUTCOME_DENIED_SECURITY": Outcome.OUTCOME_DENIED_SECURITY,
    "OUTCOME_NONE_CLARIFICATION": Outcome.OUTCOME_NONE_CLARIFICATION,
    "OUTCOME_NONE_UNSUPPORTED": Outcome.OUTCOME_NONE_UNSUPPORTED,
    "OUTCOME_ERR_INTERNAL": Outcome.OUTCOME_ERR_INTERNAL,
}


def dispatch(
    vm: EcomRuntimeClientSync,
    cmd: BaseModel,
    *,
    recorder: Recorder | None = None,
    local_trial_id: str | None = None,
    trial_dir: Path | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
    step_id: str | None = None,
    tool_scope: str = "ecom_runtime",
):
    tool_name = getattr(cmd, "tool", cmd.__class__.__name__)
    tool_call_id = short_id("tool")
    tags = {"exec_kind": "sql"} if isinstance(cmd, Req_Exec) and cmd.path == "/bin/sql" else None
    input_json = cmd.model_dump() if hasattr(cmd, "model_dump") else {"command": str(cmd)}
    started_at = utc_now()
    start_event_id = None
    if recorder:
        start_event_id = recorder.start_event(
            "tool.call.started",
            tool_name,
            local_trial_id=local_trial_id,
            task_id=task_id,
            attempt_id=attempt_id,
            step_id=step_id,
            input_json=input_json,
            tags_json=tags,
        )
    status = "ok"
    error_json = None
    output_preview = None
    output_hash = None
    output_artifact_path = None
    try:
        result = _dispatch_raw(vm, cmd)
        output_preview = format_result(cmd, result)
        output_hash = stable_hash(output_preview)
        if recorder and trial_dir:
            output_artifact_path = recorder.write_json(
                trial_dir / "tool_calls" / f"{step_id or tool_call_id}_{tool_name}.json",
                {
                    "tool_call_id": tool_call_id,
                    "tool_scope": tool_scope,
                    "tool_name": tool_name,
                    "input_json": input_json,
                    "output_preview": output_preview,
                    "protobuf_json": protobuf_to_dict(result),
                },
            )
        return result
    except ConnectError as exc:
        status = "error"
        error_json = {"type": "ConnectError", "code": str(exc.code), "message": exc.message}
        raise
    except Exception as exc:
        status = "error"
        error_json = {"type": exc.__class__.__name__, "message": str(exc)}
        raise
    finally:
        ended_at = utc_now()
        duration_ms = _duration_ms(started_at, ended_at)
        if recorder:
            if error_json and trial_dir:
                output_artifact_path = recorder.write_json(
                    trial_dir / "tool_calls" / f"{step_id or tool_call_id}_{tool_name}_error.json",
                    {
                        "tool_call_id": tool_call_id,
                        "tool_scope": tool_scope,
                        "tool_name": tool_name,
                        "input_json": input_json,
                        "error_json": error_json,
                    },
                )
            recorder.record_tool_call(
                {
                    "tool_call_id": tool_call_id,
                    "local_run_id": recorder.local_run_id,
                    "local_trial_id": local_trial_id,
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "step_id": step_id,
                    "tool_scope": tool_scope,
                    "tool_name": tool_name,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_ms": duration_ms,
                    "input_json": input_json,
                    "output_preview": output_preview,
                    "output_hash": output_hash,
                    "output_artifact_path": output_artifact_path,
                    "status": status,
                    "error_json": error_json,
                }
            )
            recorder.finish_event(
                start_event_id,
                "tool.call.error" if error_json else "tool.call.finished",
                tool_name,
                status=status,
                local_trial_id=local_trial_id,
                task_id=task_id,
                attempt_id=attempt_id,
                step_id=step_id,
                output_json={"output_preview": output_preview, "output_hash": output_hash},
                error_json=error_json,
                tags_json=tags,
                artifact_paths_json=[output_artifact_path] if output_artifact_path else None,
            )


def _dispatch_raw(vm: EcomRuntimeClientSync, cmd: BaseModel):
    if isinstance(cmd, Req_Tree):
        return vm.tree(TreeRequest(root=cmd.root, level=cmd.level))
    if isinstance(cmd, Req_Find):
        return vm.find(
            FindRequest(
                root=cmd.root,
                name=cmd.name,
                kind={
                    "all": NodeKind.NODE_KIND_UNSPECIFIED,
                    "files": NodeKind.NODE_KIND_FILE,
                    "dirs": NodeKind.NODE_KIND_DIR,
                }[cmd.kind],
                limit=cmd.limit,
            )
        )
    if isinstance(cmd, Req_Search):
        return vm.search(
            SearchRequest(root=cmd.root, pattern=cmd.pattern, limit=cmd.limit)
        )
    if isinstance(cmd, Req_List):
        return vm.list(ListRequest(path=cmd.path))
    if isinstance(cmd, Req_Read):
        return vm.read(
            ReadRequest(
                path=cmd.path,
                number=cmd.number,
                start_line=cmd.start_line,
                end_line=cmd.end_line,
            )
        )
    if isinstance(cmd, Req_Write):
        return vm.write(WriteRequest(path=cmd.path, content=cmd.content))
    if isinstance(cmd, Req_Delete):
        return vm.delete(DeleteRequest(path=cmd.path))
    if isinstance(cmd, Req_Stat):
        return vm.stat(StatRequest(path=cmd.path))
    if isinstance(cmd, Req_Exec):
        return vm.exec(ExecRequest(path=cmd.path, args=cmd.args, stdin=cmd.stdin))
    if isinstance(cmd, ReportTaskCompletion):
        return vm.answer(
            AnswerRequest(
                message=cmd.message,
                outcome=OUTCOME_BY_NAME[cmd.outcome],
                refs=cmd.grounding_refs,
            )
        )
    raise ValueError(f"Unknown command: {cmd}")


def _duration_ms(ts_start: str, ts_end: str) -> int:
    start = datetime.fromisoformat(ts_start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(ts_end.replace("Z", "+00:00"))
    return int((end - start).total_seconds() * 1000)
