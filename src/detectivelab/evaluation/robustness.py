from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from detectivelab.adapters.base import AdapterRequest, ModelAdapter

from .scoring import is_correct
from .staged import build_conflict_epistemic_prompt, build_conflict_staged_prompt, parse_conflict_stages

_WITNESS_RE = re.compile(
    r"^The witness says the (?P<label>.+?) is currently (?P<state>[a-z]+)\.$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParaphraseVariant:
    variant_id: str
    template: str

    def render(self, *, label: str, state: str) -> str:
        return self.template.format(label=label, state=state)


PARAPHRASE_VARIANTS: tuple[ParaphraseVariant, ...] = (
    ParaphraseVariant(
        "according_to",
        "According to the witness, the {label} is {state}.",
    ),
    ParaphraseVariant(
        "reports_now",
        "The witness reports that the {label} is {state} right now.",
    ),
    ParaphraseVariant(
        "claim_is",
        "The witness's claim is that the {label} is {state}.",
    ),
)

ROBUSTNESS_POLICIES = {"staged", "epistemic"}


@dataclass(frozen=True)
class ParaphraseRunResult:
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


def extract_witness_claim(payload: dict) -> tuple[str, str]:
    """Return the canonical label/state claim from the frozen participant payload."""

    if payload.get("family") != "conflict":
        raise ValueError("paraphrase robustness only supports conflict payloads")
    witness = _context_text(payload, "witness_testimony")
    if witness is None:
        raise ValueError("Conflict payload is missing witness testimony")
    match = _WITNESS_RE.match(witness)
    if match is None:
        raise ValueError(f"Unsupported frozen witness statement: {witness!r}")
    return match.group("label").strip(), match.group("state").strip().lower()


def paraphrase_witness(payload: dict, variant: ParaphraseVariant) -> str:
    label, state = extract_witness_claim(payload)
    return variant.render(label=label, state=state)


def build_paraphrase_prompt(
    *,
    image_path: Path,
    payload: dict,
    policy: str,
    variant: ParaphraseVariant,
) -> str:
    policy = policy.lower()
    if policy not in ROBUSTNESS_POLICIES:
        raise ValueError(f"policy must be one of {sorted(ROBUSTNESS_POLICIES)}")
    witness_override = paraphrase_witness(payload, variant)
    if policy == "staged":
        return build_conflict_staged_prompt(
            image_path=image_path,
            payload=payload,
            witness_override=witness_override,
        )
    return build_conflict_epistemic_prompt(
        image_path=image_path,
        payload=payload,
        witness_override=witness_override,
    )


def _iter_conflict_items(benchmark_dir: Path) -> Iterable[tuple[Path, dict, dict]]:
    manifest = _read_json(benchmark_dir / "manifest.json")
    for case in manifest["cases"]:
        case_dir = benchmark_dir / case["path"]
        payloads = _read_json(case_dir / "payloads.json")
        questions = _read_json(case_dir / "questions.json")
        gold_by_id = {item["item_id"]: item for item in questions}
        for payload in payloads:
            if payload.get("family") == "conflict":
                yield case_dir, payload, gold_by_id[payload["item_id"]]


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_paraphrase_robustness(
    *,
    benchmark_dir: Path,
    policy: str,
    adapter: ModelAdapter,
    output_path: Path,
) -> ParaphraseRunResult:
    """Evaluate deterministic testimony paraphrases without changing benchmark semantics."""

    policy = policy.lower()
    if policy not in ROBUSTNESS_POLICIES:
        raise ValueError(f"policy must be one of {sorted(ROBUSTNESS_POLICIES)}")
    if not (benchmark_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {benchmark_dir / 'manifest.json'}")

    condition = f"CONFLICT_{policy.upper()}_PARAPHRASE"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_records(output_path)
    completed = {
        (record["item_id"], record["condition"], record["model"])
        for record in existing
    }
    written = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as stream:
        for case_dir, payload, gold_item in _iter_conflict_items(benchmark_dir):
            for variant in PARAPHRASE_VARIANTS:
                variant_item_id = f"{payload['item_id']}::{variant.variant_id}"
                resume_key = (variant_item_id, condition, adapter.name)
                if resume_key in completed:
                    skipped += 1
                    continue

                prompt = build_paraphrase_prompt(
                    image_path=case_dir / "scene.png",
                    payload=payload,
                    policy=policy,
                    variant=variant,
                )
                request = AdapterRequest(
                    item_id=variant_item_id,
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
                gold = gold_item["answer"].strip().lower()

                record = {
                    "scene_id": payload["scene_id"],
                    "item_id": variant_item_id,
                    "base_item_id": payload["item_id"],
                    "family": "conflict",
                    "condition": condition,
                    "policy": policy,
                    "variant_id": variant.variant_id,
                    "witness_paraphrase": paraphrase_witness(payload, variant),
                    "model": adapter.name,
                    "prompt": prompt,
                    "image_path": None,
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "gold": gold,
                    "correct": is_correct(prediction, gold),
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

    return ParaphraseRunResult(
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
