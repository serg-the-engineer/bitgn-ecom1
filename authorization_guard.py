from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    issues: list[str] = field(default_factory=list)
    feedback: str = ""


SENSITIVE_WORDS = (
    "checkout",
    "payment",
    "payments",
    "3ds",
    "discount",
    "refund",
    "return",
    "order",
    "basket",
    "price",
    "pricing",
)

READ_ONLY_TOOLS = {"tree", "find", "search", "list", "read", "stat"}
SAFE_EXEC_PATHS = {"/bin/sql", "/bin/date", "/bin/id"}
POLICY_WORDS = ("policy", "security", "authorization", "authorised", "authorized", "approval", "allowed", "cap")


def validate_authorization_action(
    *,
    command: Any,
    task_text: str,
    tool_history: list[dict[str, Any]],
) -> AuthorizationDecision:
    payload = _payload(command)
    tool = str(payload.get("tool") or "")
    if tool != "exec" and tool not in {"write", "delete"}:
        return AuthorizationDecision(True)

    if not _task_or_command_is_sensitive(task_text, payload):
        return AuthorizationDecision(True)

    if tool == "exec" and str(payload.get("path") or "") in SAFE_EXEC_PATHS:
        return AuthorizationDecision(True)

    if not _has_prior_policy_or_authority_evidence(tool_history):
        issues = [
            "Sensitive side-effect requested before policy, authority, or safety evidence was observed.",
            "Inspect the relevant runtime policy/state first, then decide OK, clarification, unsupported, or denied.",
        ]
        return AuthorizationDecision(
            False,
            issues=issues,
            feedback="AUTHORIZATION_GUARD_BLOCKED:\n" + "\n".join(f"- {issue}" for issue in issues),
        )
    return AuthorizationDecision(True)


def sensitive_task(task_text: str, family: str | None = None) -> bool:
    if family in {"pricing_or_discount_update", "order_customer_refund_return", "security_or_policy_refusal"}:
        return True
    return any(word in str(task_text or "").lower() for word in SENSITIVE_WORDS)


def _task_or_command_is_sensitive(task_text: str, payload: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(task_text or ""),
            str(payload.get("path") or ""),
            " ".join(str(arg) for arg in payload.get("args") or []),
            str(payload.get("stdin") or ""),
        ]
    ).lower()
    return any(word in text for word in SENSITIVE_WORDS)


def _has_prior_policy_or_authority_evidence(tool_history: list[dict[str, Any]]) -> bool:
    for item in tool_history:
        tool = str(item.get("tool") or "")
        text = " ".join(
            [
                str(item.get("path") or ""),
                str(item.get("input") or ""),
                str(item.get("output_preview") or ""),
            ]
        ).lower()
        if tool in READ_ONLY_TOOLS and any(word in text for word in POLICY_WORDS):
            return True
        if tool == "exec" and str(item.get("path") or "") == "/bin/sql" and any(word in text for word in POLICY_WORDS):
            return True
    return False


def _payload(command: Any) -> dict[str, Any]:
    if hasattr(command, "model_dump"):
        return dict(command.model_dump())
    if isinstance(command, dict):
        return command
    return {"tool": getattr(command, "tool", command.__class__.__name__)}
