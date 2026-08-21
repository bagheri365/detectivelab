from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .focused import build_focused_extracted_evidence

_WITNESS_RE = re.compile(
    r"^The witness says the (?P<label>.+?) is currently (?P<state>[a-z]+)\.$",
    re.IGNORECASE,
)
_STAGE_RE = re.compile(
    r"^(?P<key>EXISTENCE|PHYSICAL_STATE|AGREEMENT|VERDICT)\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConflictStages:
    existence: str
    physical_state: str
    agreement: str
    verdict: str


def _context_text(payload: dict, entry_type: str) -> str | None:
    for entry in payload.get("context", []):
        if entry.get("type") == entry_type:
            return str(entry.get("text", ""))
    return None


def _base_staged_lines(*, image_path: Path, payload: dict, witness_override: str | None = None) -> list[str]:
    if payload.get("family") != "conflict":
        raise ValueError("staged conflict conditions only support conflict items")

    evidence = build_focused_extracted_evidence(image_path=image_path, payload=payload)
    witness = witness_override if witness_override is not None else _context_text(payload, "witness_testimony")
    rule = _context_text(payload, "case_rule")
    if witness is None or rule is None:
        raise ValueError("Conflict payload requires witness testimony and case rule")

    return [
        evidence,
        f"Witness Testimony: {witness}",
        f"Case Rule: {rule}",
        f"Question: {payload['question']}",
        "Reason through the evidence in four explicit stages.",
        "Return exactly four lines in this format:",
        "EXISTENCE: present or absent",
        "PHYSICAL_STATE: observed state or not_applicable",
        "AGREEMENT: supports, contradicts, or unknown",
        "VERDICT: supported, contradicted, or unknown",
    ]


def build_conflict_staged_prompt(*, image_path: Path, payload: dict, witness_override: str | None = None) -> str:
    lines = _base_staged_lines(image_path=image_path, payload=payload, witness_override=witness_override)
    lines.append("Use unknown when the claimed object is absent from the current physical evidence.")
    return "\n".join(lines)


def build_conflict_epistemic_prompt(*, image_path: Path, payload: dict, witness_override: str | None = None) -> str:
    """Build the explicit epistemic-rule ablation prompt.

    The representation and four-stage output schema are identical to
    CONFLICT_STAGED. The only substantive change is a mandatory rule that
    distinguishes missing evidence from contradictory evidence.
    """

    lines = _base_staged_lines(image_path=image_path, payload=payload, witness_override=witness_override)
    lines.extend(
        [
            "Mandatory epistemic rule:",
            "- If EXISTENCE is absent, there is no observed physical state for the claimed object.",
            "- Therefore PHYSICAL_STATE must be not_applicable, AGREEMENT must be unknown, and VERDICT must be unknown.",
            "- Absence of the claimed object is insufficient evidence; it is NOT evidence that contradicts the witness.",
            "Apply this rule even if the witness states a specific physical state.",
        ]
    )
    return "\n".join(lines)


def _head(value: str) -> str:
    return value.strip().lower().split(" - ", 1)[0].strip()


def _canonicalize_stage(value: str, stage: str) -> str:
    head = _head(value)
    if stage == "existence":
        if head.startswith("present"):
            return "present"
        if head.startswith("absent"):
            return "absent"
    elif stage == "physical_state":
        if head.startswith("not_applicable") or head.startswith("not applicable"):
            return "not_applicable"
    elif stage == "agreement":
        if head.startswith("supports") or head.startswith("support"):
            return "supports"
        if head.startswith("contradicts") or head.startswith("contradict"):
            return "contradicts"
        if head.startswith("unknown"):
            return "unknown"
    elif stage == "verdict":
        if head.startswith("supported") or head.startswith("support"):
            return "supported"
        if head.startswith("contradicted") or head.startswith("contradicts") or head.startswith("contradict"):
            return "contradicted"
        if head.startswith("unknown"):
            return "unknown"
    return head


def parse_conflict_stages(raw_output: str) -> ConflictStages | None:
    values: dict[str, str] = {}
    for line in raw_output.splitlines():
        match = _STAGE_RE.match(line.strip())
        if match is None:
            continue
        values[match.group("key").upper()] = match.group("value").strip()

    required = {"EXISTENCE", "PHYSICAL_STATE", "AGREEMENT", "VERDICT"}
    if not required.issubset(values):
        return None
    return ConflictStages(
        existence=_canonicalize_stage(values["EXISTENCE"], "existence"),
        physical_state=_canonicalize_stage(values["PHYSICAL_STATE"], "physical_state"),
        agreement=_canonicalize_stage(values["AGREEMENT"], "agreement"),
        verdict=_canonicalize_stage(values["VERDICT"], "verdict"),
    )


def expected_stages_from_extracted_evidence(*, image_path: Path, payload: dict) -> ConflictStages:
    """Derive diagnostic stage targets from participant-visible extracted evidence.

    This intentionally does not use scene.json or benchmark gold labels. The
    result measures downstream reasoning relative to the extractor output.
    """

    evidence = build_focused_extracted_evidence(image_path=image_path, payload=payload)
    witness = _context_text(payload, "witness_testimony")
    if witness is None:
        raise ValueError("Conflict payload is missing witness testimony")
    witness_match = _WITNESS_RE.match(witness)
    if witness_match is None:
        raise ValueError(f"Unsupported witness statement: {witness!r}")
    claimed_state = witness_match.group("state").strip().lower()

    evidence_lines = [line[2:].strip() for line in evidence.splitlines() if line.startswith("- ")]
    if len(evidence_lines) != 1 or ":" not in evidence_lines[0]:
        raise ValueError(f"Unexpected focused conflict evidence: {evidence!r}")
    _, observed = evidence_lines[0].split(":", 1)
    observed = observed.strip().lower()

    if observed == "not present":
        return ConflictStages(
            existence="absent",
            physical_state="not_applicable",
            agreement="unknown",
            verdict="unknown",
        )

    if observed == claimed_state:
        agreement = "supports"
        verdict = "supported"
    else:
        agreement = "contradicts"
        verdict = "contradicted"

    return ConflictStages(
        existence="present",
        physical_state=observed,
        agreement=agreement,
        verdict=verdict,
    )
