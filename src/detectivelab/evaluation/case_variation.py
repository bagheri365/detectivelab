from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from detectivelab.adapters.base import AdapterRequest, ModelAdapter
from detectivelab.extraction import extract_scene_facts

from .scoring import is_correct
from .staged import build_conflict_epistemic_prompt, build_conflict_staged_prompt, parse_conflict_stages

CASE_VARIATION_POLICIES = {"staged", "epistemic"}

_STATE_OPPOSITE = {
    "open": "closed",
    "closed": "open",
    "intact": "broken",
    "broken": "intact",
}
_STATEFUL_KIND_STATES = {
    "door": ("open", "closed"),
    "window": ("open", "closed"),
    "notebook": ("open", "closed"),
    "glass": ("intact", "broken"),
}
_COLORS = ("amber", "black", "blue", "green", "red", "white")


@dataclass(frozen=True)
class ConflictCaseVariant:
    variant_id: str
    label: str
    claimed_state: str
    gold: str

    @property
    def witness(self) -> str:
        return f"The witness says the {self.label} is currently {self.claimed_state}."


@dataclass(frozen=True)
class CaseVariationRunResult:
    output_path: Path
    condition: str
    written: int
    skipped: int
    total_records: int
    correct: int
    variant_accuracy: dict[str, float]
    gold_accuracy: dict[str, float]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_records if self.total_records else 0.0


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _context_text(payload: dict, entry_type: str) -> str | None:
    for entry in payload.get("context", []):
        if entry.get("type") == entry_type:
            return str(entry.get("text", ""))
    return None


def _conflict_payload(case_dir: Path) -> dict:
    payloads = _read_json(case_dir / "payloads.json")
    return next(payload for payload in payloads if payload.get("family") == "conflict")


def _variant_payload(base_payload: dict, variant: ConflictCaseVariant) -> dict:
    rule = _context_text(base_payload, "case_rule")
    if rule is None:
        raise ValueError("Conflict payload is missing case rule")
    return {
        **base_payload,
        "item_id": f"{base_payload['item_id']}::{variant.variant_id}",
        "context": [
            {"type": "witness_testimony", "text": variant.witness},
            {"type": "case_rule", "text": rule},
        ],
    }


def generate_case_variants(*, image_path: Path) -> tuple[ConflictCaseVariant, ...]:
    """Create new conflict claims from image-derived facts only.

    One stateful present object is used for both a supported and contradicted
    claim. A deterministic absent stateful color/kind label supplies an unknown
    claim. No scene JSON, IDs, seeds, provenance, or benchmark gold labels are
    consulted.
    """

    objects = extract_scene_facts(image_path)
    stateful = sorted(
        (obj for obj in objects if obj.state in _STATE_OPPOSITE),
        key=lambda obj: (obj.label, obj.center_y, obj.center_x),
    )
    if not stateful:
        raise ValueError(f"No stateful extracted object found in {image_path}")

    target = stateful[0]
    assert target.state is not None
    present_labels = {obj.label.lower() for obj in objects}

    absent_label = None
    absent_state = None
    for kind, states in _STATEFUL_KIND_STATES.items():
        for color in _COLORS:
            label = f"{color} {kind}"
            if label not in present_labels:
                absent_label = label
                # Alternate state choice deterministically from the label text
                # so unknown cases cover both state values across scenes.
                absent_state = states[sum(map(ord, label)) % len(states)]
                break
        if absent_label is not None:
            break
    if absent_label is None or absent_state is None:
        raise ValueError(f"Could not construct an absent stateful label for {image_path}")

    return (
        ConflictCaseVariant(
            variant_id="present_supported",
            label=target.label,
            claimed_state=target.state,
            gold="supported",
        ),
        ConflictCaseVariant(
            variant_id="present_contradicted",
            label=target.label,
            claimed_state=_STATE_OPPOSITE[target.state],
            gold="contradicted",
        ),
        ConflictCaseVariant(
            variant_id="absent_unknown",
            label=absent_label,
            claimed_state=absent_state,
            gold="unknown",
        ),
    )


def build_case_variation_prompt(
    *, image_path: Path, base_payload: dict, policy: str, variant: ConflictCaseVariant
) -> str:
    policy = policy.lower()
    if policy not in CASE_VARIATION_POLICIES:
        raise ValueError(f"policy must be one of {sorted(CASE_VARIATION_POLICIES)}")
    payload = _variant_payload(base_payload, variant)
    if policy == "staged":
        return build_conflict_staged_prompt(image_path=image_path, payload=payload)
    return build_conflict_epistemic_prompt(image_path=image_path, payload=payload)


def _iter_cases(benchmark_dir: Path) -> Iterable[tuple[Path, dict]]:
    manifest = _read_json(benchmark_dir / "manifest.json")
    for case in manifest["cases"]:
        case_dir = benchmark_dir / case["path"]
        yield case_dir, _conflict_payload(case_dir)


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case_variation_robustness(
    *, benchmark_dir: Path, policy: str, adapter: ModelAdapter, output_path: Path
) -> CaseVariationRunResult:
    """Evaluate new conflict claims derived from rendered scene evidence only."""

    policy = policy.lower()
    if policy not in CASE_VARIATION_POLICIES:
        raise ValueError(f"policy must be one of {sorted(CASE_VARIATION_POLICIES)}")
    if not (benchmark_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {benchmark_dir / 'manifest.json'}")

    condition = f"CONFLICT_{policy.upper()}_CASE_VARIATION"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_records(output_path)
    completed = {
        (record["item_id"], record["condition"], record["model"])
        for record in existing
    }
    written = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as stream:
        for case_dir, base_payload in _iter_cases(benchmark_dir):
            for variant in generate_case_variants(image_path=case_dir / "scene.png"):
                payload = _variant_payload(base_payload, variant)
                item_id = payload["item_id"]
                resume_key = (item_id, condition, adapter.name)
                if resume_key in completed:
                    skipped += 1
                    continue

                prompt = build_case_variation_prompt(
                    image_path=case_dir / "scene.png",
                    base_payload=base_payload,
                    policy=policy,
                    variant=variant,
                )
                request = AdapterRequest(
                    item_id=item_id,
                    family="conflict",
                    answer_type=payload["answer_type"],
                    prompt=prompt,
                    image_path=None,
                )
                start = time.perf_counter()
                raw_output = adapter.predict(request)
                latency_ms = (time.perf_counter() - start) * 1000.0
                stages = parse_conflict_stages(raw_output)
                prediction = stages.verdict if stages is not None else "invalid"

                record = {
                    "scene_id": base_payload["scene_id"],
                    "item_id": item_id,
                    "base_item_id": base_payload["item_id"],
                    "family": "conflict",
                    "condition": condition,
                    "policy": policy,
                    "variant_id": variant.variant_id,
                    "case_label": variant.label,
                    "claimed_state": variant.claimed_state,
                    "witness_testimony": variant.witness,
                    "model": adapter.name,
                    "prompt": prompt,
                    "image_path": None,
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "gold": variant.gold,
                    "correct": is_correct(prediction, variant.gold),
                    "latency_ms": round(latency_ms, 3),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                completed.add(resume_key)
                written += 1

    records = [
        record
        for record in _load_records(output_path)
        if record.get("condition") == condition and record.get("model") == adapter.name
    ]
    correct = sum(int(bool(record["correct"])) for record in records)

    variant_total = Counter(record["variant_id"] for record in records)
    variant_correct: dict[str, int] = defaultdict(int)
    gold_total = Counter(record["gold"] for record in records)
    gold_correct: dict[str, int] = defaultdict(int)
    for record in records:
        variant_correct[record["variant_id"]] += int(bool(record["correct"]))
        gold_correct[record["gold"]] += int(bool(record["correct"]))

    return CaseVariationRunResult(
        output_path=output_path,
        condition=condition,
        written=written,
        skipped=skipped,
        total_records=len(records),
        correct=correct,
        variant_accuracy={
            key: variant_correct[key] / total for key, total in sorted(variant_total.items())
        },
        gold_accuracy={key: gold_correct[key] / total for key, total in sorted(gold_total.items())},
    )
