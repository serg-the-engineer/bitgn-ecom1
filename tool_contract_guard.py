from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    issues: list[str] = field(default_factory=list)
    feedback: str = ""
    tags: dict[str, Any] = field(default_factory=dict)


LOCAL_PATH_PREFIXES = ("/home/", "/Users/", "/tmp/", "/var/", "/workspace/")
BAD_EXEC_PATHS = {"/bin/cat", "/usr/bin/cat", "cat", "sql"}
RUNTIME_SQL_PATH = "/bin/sql"


def validate_tool_contract(command: Any) -> GuardDecision:
    payload = _payload(command)
    tool = str(payload.get("tool") or "")
    issues: list[str] = []

    if tool == "exec":
        path = str(payload.get("path") or "")
        if not path.startswith("/"):
            issues.append(
                f"exec path `{path}` is not an absolute runtime path. Use `{RUNTIME_SQL_PATH}` for SQL or use read/search/list/tree for files."
            )
        if path in BAD_EXEC_PATHS:
            issues.append(
                f"`{path}` is not part of the ECOM runtime contract for file inspection. Use read/search/list/tree instead."
            )
        if path.startswith(LOCAL_PATH_PREFIXES):
            issues.append(
                f"`{path}` is a local host path, not a runtime tool path. Use runtime paths only."
            )
        if path.endswith("/bin/sql") and path != RUNTIME_SQL_PATH:
            issues.append(f"SQL must use `{RUNTIME_SQL_PATH}`, not `{path}`.")

    if tool in {"read", "list", "stat"}:
        path = str(payload.get("path") or "")
        if path and not path.startswith("/"):
            issues.append(f"{tool} path `{path}` must be an absolute runtime workspace path.")

    if tool in {"search", "find"}:
        root = str(payload.get("root") or "/")
        if root and not root.startswith("/"):
            issues.append(f"{tool} root `{root}` must be an absolute runtime workspace path.")

    if tool == "tree":
        root = str(payload.get("root") or "/")
        if root and root != "/" and not root.startswith("/"):
            issues.append(f"tree root `{root}` must be an absolute runtime workspace path.")

    if not issues:
        return GuardDecision(True)
    return GuardDecision(
        False,
        issues=issues,
        feedback="TOOL_CONTRACT_GUARD_BLOCKED:\n" + "\n".join(f"- {issue}" for issue in issues),
        tags={"guard": "tool_contract"},
    )


def command_tool_name(command: Any) -> str:
    return str(_payload(command).get("tool") or command.__class__.__name__)


def command_path(command: Any) -> str:
    payload = _payload(command)
    return str(payload.get("path") or payload.get("root") or "")


def _payload(command: Any) -> dict[str, Any]:
    if hasattr(command, "model_dump"):
        return dict(command.model_dump())
    if isinstance(command, dict):
        return command
    return {"tool": getattr(command, "tool", command.__class__.__name__)}


def is_runtime_path_like(value: str) -> bool:
    try:
        path = PurePosixPath(value)
    except Exception:
        return False
    return path.is_absolute()
