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
from detectivelab.evaluation.conditional import extracted_target_presence
from detectivelab.evaluation.gate_corruption import _gold_for_item
from detectivelab.evaluation.scoring import is_correct
from detectivelab.evaluation.staged import (
    build_conflict_staged_prompt,
    parse_conflict_stages,
)

PROTECTION_RATES = (0.00, 0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class AbstainingGateResult:
    output_path: Path
    condition: str
    protection_rate: float
    written: int
    skipped: int
    total_records: int
    correct: int
    eligible_false_absence_records: int
    protected_records: int
    residual_false_absence_records: int
    abstained_records: int
    model_calls: int
    gold_accuracy: dict[str, float]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_records if self.total_records else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.abstained_records / self.total_records if self.total_records else 0.0

    @property
    def model_call_rate(self) -> float:
        return self.model_calls / self.total_records if self.total_records else 0.0


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _stable_rank(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"abstention-protection:{item_id}".encode("utf-8")).hexdigest()
    return digest, item_id


def select_protected_items(
    eligible_item_ids: Iterable[str],
    protection_rate: float,
) -> set[str]:
    rate = float(protection_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("protection_rate must be between 0 and 1")

    eligible = sorted(set(eligible_item_ids), key=_stable_rank)
    count = _round_half_up(rate * len(eligible))
    return set(eligible[:count])


def apply_abstention_protection(
    *,
    clean_presence: str,
    false_absence_corrupted: bool,
    protected: bool,
) -> tuple[str, bool]:
    """Return the control state and whether the gate abstained.

    v0.7 starts from the v0.6 false-absence stress condition:
      clean present + corruption -> absent

    Protection converts selected would-be false absences to `uncertain`.
    `uncertain` is deliberately routed to staged reasoning.
    """
    if clean_presence not in {"present", "absent"}:
        raise ValueError("clean_presence must be 'present' or 'absent'")

    if clean_presence == "present" and false_absence_corrupted:
        if protected:
            return "uncertain", True
        return "absent", False

    return clean_presence, False


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _conflict_payload(case_dir: Path) -> dict:
    payloads = _read_json(case_dir / "payloads.json")
    return next(payload for payload in payloads if payload.get("family") == "conflict")


def _iter_cases(benchmark_dir: Path):
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


def _condition_name(protection_rate: float) -> str:
    pct = int(round(protection_rate * 100))
    return f"CONFLICT_ABSTAINING_GATE_PROTECTION_{pct:03d}"


def run_abstaining_gate_item(
    *,
    adapter: ModelAdapter,
    item_id: str,
    image_path: Path,
    payload: dict,
    protected: bool,
    protection_rate: float,
) -> dict:
    start = time.perf_counter()
    clean_presence, target_label = extracted_target_presence(
        image_path=image_path,
        payload=payload,
    )

    # Stress condition: every truly present target is subjected to the
    # v0.6 false-absence failure. Protection can rescue selected cases.
    false_absence_corrupted = clean_presence == "present"
    gate_state, abstained = apply_abstention_protection(
        clean_presence=clean_presence,
        false_absence_corrupted=false_absence_corrupted,
        protected=protected,
    )

    residual_false_absence = (
        clean_presence == "present" and gate_state == "absent"
    )

    gate_line = (
        f"GATE: target={target_label} clean={clean_presence} "
        f"stress=false_absence protection_rate={protection_rate:.2f} "
        f"protected={str(protected).lower()} state={gate_state}"
    )

    if gate_state == "absent":
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "prompt": gate_line + "\n[Gate action: absent -> deterministic unknown policy]",
            "raw_output": (
                "[ABSTAINING GATE]\n"
                + gate_line
                + "\n\n[GATED RESULT]\n"
                + _absent_output()
            ),
            "prediction": "unknown",
            "latency_ms": latency_ms,
            "clean_presence": clean_presence,
            "gate_state": gate_state,
            "abstained": abstained,
            "residual_false_absence": residual_false_absence,
            "model_calls": 0,
            "target_label": target_label,
        }

    staged_prompt = build_conflict_staged_prompt(
        image_path=image_path,
        payload=payload,
    )
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

    route_label = "UNCERTAIN FOLLOW-UP" if gate_state == "uncertain" else "PRESENT FOLLOW-UP"
    return {
        "prompt": gate_line + f"\n\n[{route_label}]\n" + staged_prompt,
        "raw_output": (
            "[ABSTAINING GATE]\n"
            + gate_line
            + f"\n\n[{route_label}]\n"
            + raw.strip()
        ),
        "prediction": prediction,
        "latency_ms": latency_ms,
        "clean_presence": clean_presence,
        "gate_state": gate_state,
        "abstained": abstained,
        "residual_false_absence": residual_false_absence,
        "model_calls": 1,
        "target_label": target_label,
    }


def run_abstaining_gate(
    *,
    benchmark_dir: Path,
    protection_rate: float,
    adapter: ModelAdapter,
    output_path: Path,
) -> AbstainingGateResult:
    rate = float(protection_rate)
    if rate not in PROTECTION_RATES:
        raise ValueError(f"protection_rate must be one of {PROTECTION_RATES}")
    if not (benchmark_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Missing benchmark manifest: {benchmark_dir / 'manifest.json'}"
        )

    cases = list(_iter_cases(benchmark_dir))

    clean_presence_by_item: dict[str, str] = {}
    for case_dir, payload in cases:
        item_id = payload["item_id"]
        clean_presence, _ = extracted_target_presence(
            image_path=case_dir / "scene.png",
            payload=payload,
        )
        clean_presence_by_item[item_id] = clean_presence

    eligible = [
        item_id
        for item_id, presence in clean_presence_by_item.items()
        if presence == "present"
    ]
    protected = select_protected_items(eligible, rate)

    condition = _condition_name(rate)
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

            result = run_abstaining_gate_item(
                adapter=adapter,
                item_id=item_id,
                image_path=case_dir / "scene.png",
                payload=payload,
                protected=item_id in protected,
                protection_rate=rate,
            )
            gold = _gold_for_item(case_dir, item_id)

            record = {
                "scene_id": payload["scene_id"],
                "item_id": item_id,
                "family": "conflict",
                "condition": condition,
                "protection_rate": rate,
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
                "gate_state": result["gate_state"],
                "protected": item_id in protected,
                "abstained": result["abstained"],
                "residual_false_absence": result["residual_false_absence"],
                "model_calls": result["model_calls"],
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            written += 1
            completed.add(resume_key)

    records = [
        r
        for r in _load_records(output_path)
        if r.get("condition") == condition and r.get("model") == adapter.name
    ]

    gold_total = Counter()
    gold_correct = Counter()
    for record in records:
        gold_total[record["gold"]] += 1
        gold_correct[record["gold"]] += int(bool(record["correct"]))

    correct = sum(int(bool(r["correct"])) for r in records)
    abstained_records = sum(int(bool(r.get("abstained"))) for r in records)
    residual_false_absence_records = sum(
        int(bool(r.get("residual_false_absence"))) for r in records
    )
    model_calls = sum(int(r.get("model_calls", 0)) for r in records)

    return AbstainingGateResult(
        output_path=output_path,
        condition=condition,
        protection_rate=rate,
        written=written,
        skipped=skipped,
        total_records=len(records),
        correct=correct,
        eligible_false_absence_records=len(eligible),
        protected_records=len(protected),
        residual_false_absence_records=residual_false_absence_records,
        abstained_records=abstained_records,
        model_calls=model_calls,
        gold_accuracy={
            key: gold_correct[key] / total
            for key, total in sorted(gold_total.items())
        },
    )
