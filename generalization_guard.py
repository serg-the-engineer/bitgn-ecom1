from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ALLOWED_PATHS = {"/", "/AGENTS.MD", "/AGENTS.md", "/bin/sql", "/bin/date", "/bin/id"}
ALLOWED_PATH_PREFIXES = ("/bin/",)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("task_id", re.compile(r"\bt\d{2,4}\b", re.IGNORECASE)),
    ("commerce_id", re.compile(r"\b(?:basket|pay|payment|cust|customer|order|refund|return)_[A-Za-z0-9]+\b", re.IGNORECASE)),
    ("sku_like_id", re.compile(r"\b[A-Z]{2,5}-[A-Z0-9]{6,12}\b")),
    ("iso_date", re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("local_host_path", re.compile(r"/(?:home|Users|tmp|var|workspace)/[A-Za-z0-9_./+-]+")),
    ("runtime_specific_path", re.compile(r"(?<!\w)/(?:docs|proc|catalog|orders|payments|customers|baskets|stores|inventory|reports|archives)/[A-Za-z0-9_./+-]+")),
]


@dataclass(frozen=True)
class GeneralizationFinding:
    kind: str
    atom: str
    message: str
    severity: str = "error"
    line_no: int | None = None


@dataclass(frozen=True)
class GeneralizationReport:
    ok: bool
    findings: list[GeneralizationFinding] = field(default_factory=list)

    def to_markdown(self) -> str:
        if self.ok:
            return "Generalization guard: OK"
        lines = ["Generalization guard: FAILED", ""]
        for finding in self.findings:
            loc = f" line {finding.line_no}" if finding.line_no else ""
            lines.append(f"- [{finding.severity}] {finding.kind}{loc}: `{finding.atom}` - {finding.message}")
        return "\n".join(lines)


def scan_text(text: str, *, source_label: str = "text") -> GeneralizationReport:
    findings: list[GeneralizationFinding] = []
    for line_no, line in enumerate(str(text or "").splitlines(), start=1):
        for kind, pattern in PATTERNS:
            for match in pattern.finditer(line):
                atom = match.group(0)
                if _is_allowed_atom(kind, atom):
                    continue
                findings.append(
                    GeneralizationFinding(
                        kind=kind,
                        atom=atom,
                        line_no=line_no,
                        message=f"{source_label} contains task-specific data; runtime rules and skills must use reusable abstractions.",
                    )
                )
    return GeneralizationReport(ok=not findings, findings=findings)


def scan_paths(paths: Iterable[str | Path]) -> dict[str, GeneralizationReport]:
    reports: dict[str, GeneralizationReport] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and _is_text_candidate(child):
                    reports[str(child)] = scan_text(child.read_text(encoding="utf-8", errors="replace"), source_label=str(child))
        elif path.exists() and _is_text_candidate(path):
            reports[str(path)] = scan_text(path.read_text(encoding="utf-8", errors="replace"), source_label=str(path))
    return reports


def generalize_runtime_guidance(text: str) -> str:
    out = str(text or "")
    replacements = [
        (PATTERNS[1][1], "[commerce_object_id]"),
        (PATTERNS[2][1], "[catalog_item_id]"),
        (PATTERNS[3][1], "[date]"),
        (PATTERNS[4][1], "[email]"),
        (PATTERNS[5][1], "[local_host_path]"),
        (PATTERNS[6][1], "[runtime_evidence_path]"),
    ]
    for pattern, replacement in replacements:
        out = pattern.sub(lambda m: m.group(0) if _is_allowed_atom("path", m.group(0)) else replacement, out)
    out = PATTERNS[0][1].sub("[task_id]", out)
    return out


def assert_generalized(text: str, *, source_label: str = "text") -> None:
    report = scan_text(text, source_label=source_label)
    if not report.ok:
        raise ValueError(report.to_markdown())


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yml", ".yaml"}


def _is_allowed_atom(kind: str, atom: str) -> bool:
    if atom in ALLOWED_PATHS:
        return True
    if kind in {"runtime_specific_path", "local_host_path"} and atom.startswith(ALLOWED_PATH_PREFIXES):
        return True
    return False
