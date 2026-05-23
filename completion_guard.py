from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from authorization_guard import sensitive_task


@dataclass(frozen=True)
class CompletionDecision:
    allowed: bool
    issues: list[str] = field(default_factory=list)
    feedback: str = ""


REQUIRED_LITERAL_RE = re.compile(r"<[A-Z][A-Z0-9_]{1,40}>")
RUNTIME_REF_RE = re.compile(r"(?<!\w)/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+")
NUMBER_ONLY_HINTS = ("number only", "integer only", "plain integer", "bare integer", "digits only")
READ_OR_VERIFY_TOOLS = {"read", "stat", "list", "search", "find", "tree"}
MUTATING_EXEC_HINTS = ("checkout", "payment", "payments", "discount", "refund", "return", "order")


def validate_completion(
    *,
    command: Any,
    task_text: str,
    tool_history: list[dict[str, Any]],
    task_family: str | None = None,
) -> CompletionDecision:
    payload = _payload(command)
    if payload.get("tool") != "report_completion":
        return CompletionDecision(True)

    issues: list[str] = []
    outcome = str(payload.get("outcome") or "")
    message = str(payload.get("message") or "")
    refs = [str(ref) for ref in payload.get("grounding_refs") or [] if str(ref).strip()]
    text = str(task_text or "")
    lower_task = text.lower()

    if not outcome:
        issues.append("Completion outcome is missing.")
    if not message.strip():
        issues.append("Completion message is empty.")

    required_literals = sorted(set(REQUIRED_LITERAL_RE.findall(text)))
    if required_literals and outcome == "OUTCOME_OK":
        if not any(token in message for token in required_literals):
            issues.append(
                "Task contains a literal output marker; the final message must include the required literal marker."
            )

    if any(hint in lower_task for hint in NUMBER_ONLY_HINTS) and outcome == "OUTCOME_OK":
        if not re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", message):
            issues.append("Task asks for number-only output, but the final message is not only a number.")

    mentioned_refs = _task_reference_mentions(text)
    if mentioned_refs and outcome in {"OUTCOME_OK", "OUTCOME_DENIED_SECURITY"}:
        if not any(ref in refs or ref in message for ref in mentioned_refs):
            issues.append("Task mentions required evidence references; include the discovered required references in refs or message.")

    if outcome == "OUTCOME_OK" and _looks_like_grounded_task(lower_task, task_family) and not refs:
        issues.append("Successful grounded task completion needs grounding_refs.")

    if outcome == "OUTCOME_OK" and _has_mutation(tool_history) and not _has_post_mutation_verification(tool_history):
        issues.append("A mutation was attempted, but no later verification read/stat/search/list was observed.")

    if sensitive_task(text, task_family) and outcome == "OUTCOME_OK" and not _has_authorization_evidence(tool_history):
        issues.append("Sensitive task is being reported OK without observed authorization or policy evidence.")

    if not issues:
        return CompletionDecision(True)
    return CompletionDecision(
        False,
        issues=issues,
        feedback="COMPLETION_GUARD_BLOCKED:\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\nRevise the next step; do not call report_completion until the contract is satisfied.",
    )


def _payload(command: Any) -> dict[str, Any]:
    if hasattr(command, "model_dump"):
        return dict(command.model_dump())
    if isinstance(command, dict):
        return command
    return {"tool": getattr(command, "tool", command.__class__.__name__)}


def _task_reference_mentions(text: str) -> list[str]:
    refs: list[str] = []
    for ref in RUNTIME_REF_RE.findall(text):
        if ref.startswith("/bin/") or ref in {"/", "/AGENTS.MD", "/AGENTS.md"}:
            continue
        refs.append(ref)
    return sorted(set(refs))


def _looks_like_grounded_task(lower_task: str, family: str | None) -> bool:
    if family in {
        "catalog_search_and_identification",
        "inventory_or_availability_update",
        "pricing_or_discount_update",
        "order_customer_refund_return",
        "data_quality_reconciliation",
        "reporting_or_analysis",
    }:
        return True
    return any(
        word in lower_task
        for word in ("find", "count", "verify", "catalog", "catalogue", "availability", "checkout", "payment", "discount")
    )


def _has_mutation(history: list[dict[str, Any]]) -> bool:
    for item in history:
        tool = str(item.get("tool") or "")
        path = str(item.get("path") or "").lower()
        if tool in {"write", "delete"}:
            return True
        if tool == "exec" and any(word in path for word in MUTATING_EXEC_HINTS):
            return True
    return False


def _has_post_mutation_verification(history: list[dict[str, Any]]) -> bool:
    mutation_index = -1
    for idx, item in enumerate(history):
        tool = str(item.get("tool") or "")
        path = str(item.get("path") or "").lower()
        if tool in {"write", "delete"} or (tool == "exec" and any(word in path for word in MUTATING_EXEC_HINTS)):
            mutation_index = idx
    if mutation_index < 0:
        return True
    return any(str(item.get("tool") or "") in READ_OR_VERIFY_TOOLS for item in history[mutation_index + 1 :])


def _has_authorization_evidence(history: list[dict[str, Any]]) -> bool:
    words = ("policy", "security", "authorization", "authorised", "authorized", "approval", "allowed", "cap")
    return any(
        any(word in str(item.get("output_preview") or "").lower() or word in str(item.get("path") or "").lower() for word in words)
        for item in history
    )
