from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from detectivelab.adapters.base import AdapterRequest, ModelAdapter

from .conditional import extracted_target_presence
from .scoring import is_correct
from .staged import build_conflict_staged_prompt, parse_conflict_stages

GATE_CORRUPTIONS = {"clean", "false_absence", "false_presence"}
GATE_CORRUPTION_RATES = (0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class GateCorruptionResult:
    output_path: Path
    condition: str
    corruption: str
    corruption_rate: float
    written: int
    skipped: int
    total_records: int
    correct: int
    corrupted_records: int
    eligible_records: int
    gold_accuracy: dict[str, float]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_records if self.total_records else 0.0


def apply_gate_corruption(
    clean_presence: str,
    corruption: str,
    *,
    enabled: bool = True,
) -> tuple[str, bool]:
    """Apply one directional gate corruption when enabled.

    false_absence flips only present -> absent.
    false_presence flips only absent -> present.
    The returned bool says whether the gate value was actually changed.
    """
    corruption = corruption.lower()
    if corruption not in GATE_CORRUPTIONS:
        raise ValueError(f"corruption must be one of {sorted(GATE_CORRUPTIONS)}")
    if clean_presence not in {"present", "absent"}:
        raise ValueError("clean_presence must be 'present' or 'absent'")

    if not enabled or corruption == "clean":
        return clean_presence, False
    if corruption == "false_absence" and clean_presence == "present":
        return "absent", True
    if corruption == "false_presence" and clean_presence == "absent":
        return "present", True
    return clean_presence, False


def _validate_rate(corruption: str, corruption_rate: float) -> float:
    rate = float(corruption_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("corruption_rate must be between 0 and 1")
    if corruption == "clean" and rate != 0.0:
        raise ValueError("clean corruption requires corruption_rate=0")
    return rate


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _stable_rank(item_id: str, corruption: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{corruption}:{item_id}".encode("utf-8")).hexdigest()
    return digest, item_id


def select_corrupted_items(
    eligible_item_ids: Iterable[str],
    *,
    corruption: str,
    corruption_rate: float,
) -> set[str]:
    """Select an exact deterministic subset of eligible items for corruption."""
    corruption = corruption.lower()
    if corruption not in GATE_CORRUPTIONS:
        raise ValueError(f"corruption must be one of {sorted(GATE_CORRUPTIONS)}")
    rate = _validate_rate(corruption, corruption_rate)

    eligible = sorted(set(eligible_item_ids), key=lambda x: _stable_rank(x, corruption))
    count = _round_half_up(rate * len(eligible))
    return set(eligible[:count])


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _conflict_payload(case_dir: Path) -> dict:
    payloads = _read_json(case_dir / "payloads.json")
    return next(payload for payload in payloads if payload.get("family") == "conflict")


def _gold_for_item(case_dir: Path, item_id: str):
    """Read the hidden scoring target from questions.json, never participant payloads."""
    questions = _read_json(case_dir / "questions.json")

    def walk(value):
        if isinstance(value, dict):
            if value.get("item_id") == item_id:
                for key in ("gold", "answer", "label", "expected"):
                    if key in value:
                        return value[key]
            for child in value.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None

    gold = walk(questions)
    if gold is None:
        raise KeyError(
            f"Could not find a scoring target for item_id={item_id!r} "
            f"in {case_dir / 'questions.json'}"
        )
    return gold


def _iter_cases(benchmark_dir: Path) -> Iterable[tuple[Path, dict]]:
    manifest = _read_json(benchmark_dir / "manifest.json")
    for case in manifest["cases"]:
        case_dir = benchmark_dir / case["path"]
        yield case_dir, _conflict_payload(case_dir)


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _absent_output() -> str:
    return "\n".join(
        [
            "EXISTENCE: absent",
            "PHYSICAL_STATE: not_applicable",
            "AGREEMENT: unknown",
            "VERDICT: unknown",
        ]
    )


def _condition_name(corruption: str, corruption_rate: float) -> str:
    if corruption == "clean":
        return "CONFLICT_EXTRACTOR_GATED_CLEAN"
    pct = int(round(corruption_rate * 100))
    return f"CONFLICT_EXTRACTOR_GATED_{corruption.upper()}_{pct:03d}"


def run_corrupted_gate_item(
    *,
    adapter: ModelAdapter,
    item_id: str,
    image_path: Path,
    payload: dict,
    corruption: str,
    corruption_rate: float = 1.0,
    corruption_enabled: bool = True,
) -> dict:
    """Run one conflict item with a controlled presence-gate corruption."""
    start = time.perf_counter()
    clean_presence, target_label = extracted_target_presence(
        image_path=image_path,
        payload=payload,
    )
    gate_presence, was_corrupted = apply_gate_corruption(
        clean_presence,
        corruption,
        enabled=corruption_enabled,
    )

    gate_line = (
        f"GATE: target={target_label} clean={clean_presence} "
        f"corruption={corruption} rate={corruption_rate:.2f} "
        f"selected={str(corruption_enabled).lower()} observed={gate_presence}"
    )

    if gate_presence == "absent":
        raw_output = (
            "[CORRUPTED EXTRACTOR GATE]\n"
            + gate_line
            + "\n\n[GATED RESULT]\n"
            + _absent_output()
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "prompt": gate_line + "\n[Gate action: absent -> deterministic unknown policy]",
            "raw_output": raw_output,
            "prediction": "unknown",
            "latency_ms": latency_ms,
            "clean_presence": clean_presence,
            "gate_presence": gate_presence,
            "was_corrupted": was_corrupted,
            "model_calls": 0,
            "target_label": target_label,
        }

    staged_prompt = build_conflict_staged_prompt(image_path=image_path, payload=payload)
    request = AdapterRequest(
        item_id=item_id,
        family="conflict",
        answer_type="evidence_verdict",
        prompt=staged_prompt,
        image_path=None,
    )
    raw = adapter.predict(request)
    latency_ms = (time.perf_counter() - start) * 1000.0
    stages = parse_conflict_stages(raw)
    prediction = stages.verdict if stages is not None else "invalid"
    return {
        "prompt": gate_line + "\n\n[PRESENT-TARGET FOLLOW-UP]\n" + staged_prompt,
        "raw_output": (
            "[CORRUPTED EXTRACTOR GATE]\n"
            + gate_line
            + "\n\n[STAGED FOLLOW-UP]\n"
            + raw.strip()
        ),
        "prediction": prediction,
        "latency_ms": latency_ms,
        "clean_presence": clean_presence,
        "gate_presence": gate_presence,
        "was_corrupted": was_corrupted,
        "model_calls": 1,
        "target_label": target_label,
    }


def run_gate_corruption(
    *,
    benchmark_dir: Path,
    corruption: str,
    adapter: ModelAdapter,
    output_path: Path,
    corruption_rate: float = 1.0,
) -> GateCorruptionResult:
    corruption = corruption.lower()
    if corruption not in GATE_CORRUPTIONS:
        raise ValueError(f"corruption must be one of {sorted(GATE_CORRUPTIONS)}")
    rate = _validate_rate(corruption, 0.0 if corruption == "clean" else corruption_rate)
    if not (benchmark_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {benchmark_dir / 'manifest.json'}")

    cases = list(_iter_cases(benchmark_dir))

    clean_presence_by_item: dict[str, str] = {}
    for case_dir, payload in cases:
        item_id = payload["item_id"]
        clean_presence, _ = extracted_target_presence(
            image_path=case_dir / "scene.png",
            payload=payload,
        )
        clean_presence_by_item[item_id] = clean_presence

    if corruption == "false_absence":
        eligible = [
            item_id for item_id, presence in clean_presence_by_item.items()
            if presence == "present"
        ]
    elif corruption == "false_presence":
        eligible = [
            item_id for item_id, presence in clean_presence_by_item.items()
            if presence == "absent"
        ]
    else:
        eligible = []

    selected = select_corrupted_items(
        eligible,
        corruption=corruption,
        corruption_rate=rate,
    )

    condition = _condition_name(corruption, rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_records(output_path)
    completed = {
        (record["item_id"], record["condition"], record["model"])
        for record in existing
    }
    written = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as stream:
        for case_dir, payload in cases:
            item_id = payload["item_id"]
            resume_key = (item_id, condition, adapter.name)
            if resume_key in completed:
                skipped += 1
                continue

            result = run_corrupted_gate_item(
                adapter=adapter,
                item_id=item_id,
                image_path=case_dir / "scene.png",
                payload=payload,
                corruption=corruption,
                corruption_rate=rate,
                corruption_enabled=item_id in selected,
            )
            gold = _gold_for_item(case_dir, item_id)
            record = {
                "scene_id": payload["scene_id"],
                "item_id": item_id,
                "family": "conflict",
                "condition": condition,
                "corruption": corruption,
                "corruption_rate": rate,
                "model": adapter.name,
                "prompt": result["prompt"],
                "image_path": None,
                "raw_output": result["raw_output"],
                "prediction": result["prediction"],
                "gold": gold,
                "correct": is_correct(result["prediction"], gold),
                "latency_ms": round(result["latency_ms"], 3),
                "target_label": result["target_label"],
                "clean_presence": result["clean_presence"],
                "gate_presence": result["gate_presence"],
                "was_corrupted": result["was_corrupted"],
                "selected_for_corruption": item_id in selected,
                "model_calls": result["model_calls"],
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            written += 1
            completed.add(resume_key)

    records = [
        r for r in _load_records(output_path)
        if r.get("condition") == condition and r.get("model") == adapter.name
    ]
    gold_total = Counter()
    gold_correct = Counter()
    corrupted_records = 0
    for record in records:
        gold_total[record["gold"]] += 1
        gold_correct[record["gold"]] += int(bool(record["correct"]))
        corrupted_records += int(bool(record.get("was_corrupted")))

    correct = sum(int(bool(r["correct"])) for r in records)
    return GateCorruptionResult(
        output_path=output_path,
        condition=condition,
        corruption=corruption,
        corruption_rate=rate,
        written=written,
        skipped=skipped,
        total_records=len(records),
        correct=correct,
        corrupted_records=corrupted_records,
        eligible_records=len(eligible),
        gold_accuracy={
            key: gold_correct[key] / total for key, total in sorted(gold_total.items())
        },
    )
