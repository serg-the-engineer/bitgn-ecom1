from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class Skill:
    skill_id: str
    title: str
    description: str
    filename: str

    @property
    def path(self) -> Path:
        return SKILLS_DIR / self.filename

    def load(self) -> str:
        return self.path.read_text(encoding="utf-8").strip()


SKILLS: dict[str, Skill] = {
    "catalog_lookup": Skill(
        "catalog_lookup",
        "Catalog Lookup And Counting",
        "Catalog presence checks, product identification, and counts.",
        "catalog_lookup.md",
    ),
    "inventory_availability": Skill(
        "inventory_availability",
        "Inventory Availability",
        "Variant resolution and store or branch availability aggregation.",
        "inventory_availability.md",
    ),
    "data_quality_reconciliation": Skill(
        "data_quality_reconciliation",
        "Data Quality Reconciliation",
        "Support-note verification and canonical attribute comparison.",
        "data_quality_reconciliation.md",
    ),
    "checkout_payment_recovery": Skill(
        "checkout_payment_recovery",
        "Checkout And Payment Recovery",
        "Checkout, basket, payment authentication, refund, return, and recovery workflows.",
        "checkout_payment_recovery.md",
    ),
    "discount_pricing_policy": Skill(
        "discount_pricing_policy",
        "Discount And Pricing Policy",
        "Discounts, service recovery, price changes, authority, and policy caps.",
        "discount_pricing_policy.md",
    ),
    "read_only_investigation": Skill(
        "read_only_investigation",
        "Read-Only Investigation",
        "Archive, fraud-history, reporting, and no-write investigations.",
        "read_only_investigation.md",
    ),
    "authorization_safety": Skill(
        "authorization_safety",
        "Authorization And Safety",
        "Deny-first and clarify-first checks for sensitive side effects.",
        "authorization_safety.md",
    ),
    "completion_contract": Skill(
        "completion_contract",
        "Completion Contract",
        "Outcome, format, verification, and evidence discipline before final answer.",
        "completion_contract.md",
    ),
    "runtime_tool_contract": Skill(
        "runtime_tool_contract",
        "Runtime Tool Contract",
        "Runtime path and tool-surface discipline.",
        "runtime_tool_contract.md",
    ),
}


def list_skills() -> list[Skill]:
    return [SKILLS[key] for key in sorted(SKILLS)]


def get_skill(skill_id: str) -> Skill | None:
    return SKILLS.get(skill_id)


def load_skill(skill_id: str) -> str | None:
    skill = get_skill(skill_id)
    if not skill:
        return None
    return skill.load()
