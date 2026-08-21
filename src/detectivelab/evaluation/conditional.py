from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from detectivelab.adapters.base import AdapterRequest, ModelAdapter
from detectivelab.extraction import extract_scene_facts

from .focused import build_focused_extracted_evidence
from .staged import build_conflict_staged_prompt, parse_conflict_stages

_EXISTENCE_LINE_RE = re.compile(r"^EXISTENCE\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ConditionalConflictResult:
    prompt: str
    raw_output: str
    prediction: str
    latency_ms: float
    existence: str | None
    gated: bool
    model_calls: int


def _context_text(payload: dict, entry_type: str) -> str | None:
    for entry in payload.get("context", []):
        if entry.get("type") == entry_type:
            return str(entry.get("text", ""))
    return None


def build_conflict_existence_prompt(
    *, image_path: Path, payload: dict, witness_override: str | None = None
) -> str:
    """Ask only whether the witness's claimed target exists in extracted evidence."""

    if payload.get("family") != "conflict":
        raise ValueError("conditional conflict evaluation only supports conflict items")

    evidence = build_focused_extracted_evidence(image_path=image_path, payload=payload)
    witness = witness_override if witness_override is not None else _context_text(payload, "witness_testimony")
    if witness is None:
        raise ValueError("Conflict payload is missing witness testimony")

    return "\n".join(
        [
            evidence,
            f"Witness Testimony: {witness}",
            "Determine only whether the object named in the witness testimony is present in the current physical evidence.",
            "Return exactly one line:",
            "EXISTENCE: present or absent",
            "Do not decide agreement or verdict yet.",
        ]
    )


def parse_existence_decision(raw_output: str) -> str | None:
    for line in raw_output.splitlines():
        match = _EXISTENCE_LINE_RE.match(line.strip())
        if match is None:
            continue
        value = match.group("value").strip().lower()
        # Accept short explanations while canonicalizing the decision token.
        if value.startswith("present"):
            return "present"
        if value.startswith("absent"):
            return "absent"
    return None


def _absent_stage_output() -> str:
    return "\n".join(
        [
            "EXISTENCE: absent",
            "PHYSICAL_STATE: not_applicable",
            "AGREEMENT: unknown",
            "VERDICT: unknown",
        ]
    )



_WITNESS_RE = re.compile(
    r"^The witness says the (?P<label>.+?) is currently (?P<state>[a-z]+)\.$",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _target_label(payload: dict) -> str:
    witness = _context_text(payload, "witness_testimony")
    if witness is None:
        raise ValueError("Conflict payload is missing witness testimony")
    match = _WITNESS_RE.match(witness)
    if match is None:
        raise ValueError(f"Unsupported canonical witness statement: {witness!r}")
    return _norm(match.group("label"))


def extracted_target_presence(*, image_path: Path, payload: dict) -> tuple[str, str]:
    """Return (presence, target_label) using image-derived extractor facts only."""
    label = _target_label(payload)
    objects = extract_scene_facts(image_path)
    matches = [obj for obj in objects if _norm(obj.label) == label]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous extracted label: {label!r}")
    return ("present" if matches else "absent"), label


def run_extractor_gated_conflict(
    *,
    adapter: ModelAdapter,
    item_id: str,
    image_path: Path,
    payload: dict,
    witness_override: str | None = None,
) -> ConditionalConflictResult:
    """Gate the epistemic policy from extractor-derived target presence.

    No LLM call is used for the gate. Absent targets deterministically map to
    unknown; present targets use the unchanged staged prompt.
    """
    start = time.perf_counter()
    presence, label = extracted_target_presence(image_path=image_path, payload=payload)
    gate_text = f"EXTRACTOR_GATE: {label} -> {presence}"

    if presence == "absent":
        final_raw = _absent_stage_output()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ConditionalConflictResult(
            prompt=gate_text + "\n[Gate action: absent -> deterministic unknown policy]",
            raw_output="[EXTRACTOR GATE]\n" + gate_text + "\n\n[GATED RESULT]\n" + final_raw,
            prediction="unknown",
            latency_ms=latency_ms,
            existence="absent",
            gated=True,
            model_calls=0,
        )

    staged_prompt = build_conflict_staged_prompt(
        image_path=image_path,
        payload=payload,
        witness_override=witness_override,
    )
    staged_request = AdapterRequest(
        item_id=item_id,
        family="conflict",
        answer_type="evidence_verdict",
        prompt=staged_prompt,
        image_path=None,
    )
    staged_raw = adapter.predict(staged_request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    stages = parse_conflict_stages(staged_raw)
    prediction = stages.verdict if stages is not None else "invalid"
    return ConditionalConflictResult(
        prompt=gate_text + "\n\n[PRESENT-TARGET FOLLOW-UP]\n" + staged_prompt,
        raw_output="[EXTRACTOR GATE]\n" + gate_text + "\n\n[STAGED FOLLOW-UP]\n" + staged_raw.strip(),
        prediction=prediction,
        latency_ms=latency_ms,
        existence="present",
        gated=False,
        model_calls=1,
    )

def run_conditional_conflict(
    *,
    adapter: ModelAdapter,
    item_id: str,
    image_path: Path,
    payload: dict,
    witness_override: str | None = None,
) -> ConditionalConflictResult:
    """Run the deterministic v0.5 absence gate.

    Step 1 asks only for target existence. If the model says ``absent``, the
    benchmark's epistemic policy is applied deterministically and no second
    model call is made. If the model says ``present``, the unchanged staged
    prompt is used for the second call.
    """

    existence_prompt = build_conflict_existence_prompt(
        image_path=image_path,
        payload=payload,
        witness_override=witness_override,
    )
    existence_request = AdapterRequest(
        item_id=f"{item_id}::existence_gate",
        family="conflict",
        answer_type="evidence_verdict",
        prompt=existence_prompt,
        image_path=None,
    )

    start = time.perf_counter()
    existence_raw = adapter.predict(existence_request)
    existence = parse_existence_decision(existence_raw)

    if existence == "absent":
        final_raw = _absent_stage_output()
        latency_ms = (time.perf_counter() - start) * 1000.0
        combined_prompt = existence_prompt + "\n\n[Gate action: absent -> deterministic unknown policy]"
        combined_raw = "[EXISTENCE GATE]\n" + existence_raw.strip() + "\n\n[GATED RESULT]\n" + final_raw
        return ConditionalConflictResult(
            prompt=combined_prompt,
            raw_output=combined_raw,
            prediction="unknown",
            latency_ms=latency_ms,
            existence="absent",
            gated=True,
            model_calls=1,
        )

    if existence == "present":
        staged_prompt = build_conflict_staged_prompt(
            image_path=image_path,
            payload=payload,
            witness_override=witness_override,
        )
        staged_request = AdapterRequest(
            item_id=item_id,
            family="conflict",
            answer_type="evidence_verdict",
            prompt=staged_prompt,
            image_path=None,
        )
        staged_raw = adapter.predict(staged_request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        stages = parse_conflict_stages(staged_raw)
        prediction = stages.verdict if stages is not None else "invalid"
        combined_prompt = existence_prompt + "\n\n[PRESENT-TARGET FOLLOW-UP]\n" + staged_prompt
        combined_raw = "[EXISTENCE GATE]\n" + existence_raw.strip() + "\n\n[STAGED FOLLOW-UP]\n" + staged_raw.strip()
        return ConditionalConflictResult(
            prompt=combined_prompt,
            raw_output=combined_raw,
            prediction=prediction,
            latency_ms=latency_ms,
            existence="present",
            gated=False,
            model_calls=2,
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return ConditionalConflictResult(
        prompt=existence_prompt,
        raw_output="[EXISTENCE GATE]\n" + existence_raw.strip(),
        prediction="invalid",
        latency_ms=latency_ms,
        existence=None,
        gated=False,
        model_calls=1,
    )
