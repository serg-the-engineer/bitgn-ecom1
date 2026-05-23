from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics_store import json_loads
from skill_registry import load_skill


FAMILY_TO_PRIMARY_SKILL = {
    "catalog_search_and_identification": "catalog_lookup",
    "catalog_data_update": "catalog_lookup",
    "inventory_or_availability_update": "inventory_availability",
    "pricing_or_discount_update": "discount_pricing_policy",
    "order_customer_refund_return": "checkout_payment_recovery",
    "data_quality_reconciliation": "data_quality_reconciliation",
    "reporting_or_analysis": "read_only_investigation",
    "security_or_policy_refusal": "authorization_safety",
    "workspace_file_operation": "read_only_investigation",
}

SENSITIVE_FAMILIES = {
    "pricing_or_discount_update",
    "order_customer_refund_return",
    "security_or_policy_refusal",
}

READ_ONLY_HINTS = (
    "read-only",
    "do not change",
    "do not modify",
    "archive",
    "archived",
    "fraud",
    "report",
    "history",
    "inspect",
)

CHECKOUT_HINTS = (
    "checkout",
    "basket",
    "payment",
    "3ds",
    "refund",
    "return",
    "order",
)

DISCOUNT_HINTS = ("discount", "price", "pricing", "service_recovery", "manager")
INVENTORY_HINTS = ("availability", "available", "stock", "inventory", "branch", "store")
CATALOG_HINTS = ("catalog", "catalogue", "sku", "product", "count", "how many")


@dataclass(frozen=True)
class SkillRoute:
    primary_skill: str
    guard_skills: list[str] = field(default_factory=list)
    task_family: str | None = None
    risk_flags: list[str] = field(default_factory=list)

    @property
    def skill_ids(self) -> list[str]:
        out: list[str] = []
        for skill_id in [self.primary_skill, *self.guard_skills]:
            if skill_id and skill_id not in out:
                out.append(skill_id)
        return out


def route_skills(
    *,
    task_text: str,
    classification: Any | None = None,
) -> SkillRoute:
    family = _classification_value(classification, "task_family")
    risks = _classification_list(classification, "risk_flags")
    text = str(task_text or "").lower()

    primary = FAMILY_TO_PRIMARY_SKILL.get(family or "")
    if not primary:
        primary = _infer_primary_from_text(text)

    guards = ["runtime_tool_contract", "completion_contract"]
    if family in SENSITIVE_FAMILIES or any(h in text for h in CHECKOUT_HINTS + DISCOUNT_HINTS):
        guards.insert(0, "authorization_safety")
    if any(h in text for h in READ_ONLY_HINTS):
        guards.insert(0, "read_only_investigation")

    return SkillRoute(primary_skill=primary, guard_skills=_dedupe(guards), task_family=family, risk_flags=risks)


def render_skill_context(route: SkillRoute) -> str:
    sections: list[str] = [
        "Reusable task skills for this run.",
        "These are generalized workflows. Do not treat them as task facts.",
        f"Primary skill: {route.primary_skill}.",
    ]
    if route.task_family:
        sections.append(f"Classified task family: {route.task_family}.")
    if route.risk_flags:
        sections.append("Risk flags: " + "; ".join(route.risk_flags[:5]))
    for skill_id in route.skill_ids:
        prompt = load_skill(skill_id)
        if prompt:
            sections.append(prompt)
    return "\n\n".join(sections).strip()


def _infer_primary_from_text(text: str) -> str:
    if any(h in text for h in DISCOUNT_HINTS):
        return "discount_pricing_policy"
    if any(h in text for h in CHECKOUT_HINTS):
        return "checkout_payment_recovery"
    if any(h in text for h in INVENTORY_HINTS):
        return "inventory_availability"
    if any(h in text for h in READ_ONLY_HINTS):
        return "read_only_investigation"
    if any(h in text for h in CATALOG_HINTS):
        return "catalog_lookup"
    return "data_quality_reconciliation"


def _classification_value(classification: Any | None, key: str) -> str | None:
    if classification is None:
        return None
    if hasattr(classification, key):
        value = getattr(classification, key)
        return str(value) if value is not None else None
    if isinstance(classification, dict):
        value = classification.get(key)
        return str(value) if value is not None else None
    try:
        value = classification[key]
        return str(value) if value is not None else None
    except Exception:
        return None


def _classification_list(classification: Any | None, key: str) -> list[str]:
    if classification is None:
        return []
    raw: Any = None
    if hasattr(classification, key):
        raw = getattr(classification, key)
    elif isinstance(classification, dict):
        raw = classification.get(key) or classification.get(f"{key}_json")
    else:
        try:
            raw = classification[key]
        except Exception:
            raw = None
    if isinstance(raw, str):
        parsed = json_loads(raw, None)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
