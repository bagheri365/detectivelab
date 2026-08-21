from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from detectivelab.adapters.base import AdapterRequest, ModelAdapter
from detectivelab.evaluation.conditional import extracted_target_presence
from detectivelab.evaluation.gate_corruption import _gold_for_item
from detectivelab.evaluation.scoring import is_correct
from detectivelab.evaluation.staged import (
    build_conflict_staged_prompt,
    parse_conflict_stages,
)

# Calibrated in scripts/audit_perturbation_stability.py on the frozen benchmark.
# Brightness perturbations are intentionally excluded because they caused
# systematic extractor collapse rather than case-specific instability.
CALIBRATED_VIEWS = (
    "original",
    "blur_020",
    "blur_040",
    "blur_060",
    "downsample_090",
    "downsample_075",
    "downsample_060",
)

DEFAULT_VIEWS = CALIBRATED_VIEWS


@dataclass(frozen=True)
class EvidenceUncertainty:
    state: str
    present_votes: int
    absent_votes: int
    total_votes: int
    agreement: float
    votes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EvidenceUncertaintyResult:
    output_path: Path
    condition: str
    written: int
    skipped: int
    total_records: int
    correct: int
    uncertain_records: int
    hard_absent_records: int
    hard_present_records: int
    model_calls: int
    gold_accuracy: dict[str, float]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_records if self.total_records else 0.0

    @property
    def uncertainty_rate(self) -> float:
        return self.uncertain_records / self.total_records if self.total_records else 0.0

    @property
    def model_call_rate(self) -> float:
        return self.model_calls / self.total_records if self.total_records else 0.0


def classify_presence_votes(votes: list[str]) -> tuple[str, float]:
    if not votes:
        raise ValueError("at least one vote is required")
    if any(v not in {"present", "absent"} for v in votes):
        raise ValueError("votes must be 'present' or 'absent'")

    present = sum(v == "present" for v in votes)
    absent = len(votes) - present
    agreement = max(present, absent) / len(votes)

    if present == len(votes):
        return "present", agreement
    if absent == len(votes):
        return "absent", agreement
    return "uncertain", agreement


def _save_view(image: Image.Image, view: str, path: Path) -> None:
    if view == "original":
        transformed = image
    elif view.startswith("blur_"):
        radius = int(view.split("_", 1)[1]) / 100
        transformed = image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif view.startswith("downsample_"):
        scale = int(view.split("_", 1)[1]) / 100
        w, h = image.size
        small = image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.BILINEAR,
        )
        transformed = small.resize((w, h), Image.Resampling.NEAREST)
    else:
        raise ValueError(f"unknown evidence view: {view}")
    transformed.save(path)


def evidence_uncertainty(
    *,
    image_path: Path,
    payload: dict,
    views: tuple[str, ...] = DEFAULT_VIEWS,
) -> EvidenceUncertainty:
    if not views:
        raise ValueError("views must be non-empty")

    votes: list[tuple[str, str]] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        with tempfile.TemporaryDirectory(prefix="detectivelab_uncertainty_") as tmp:
            tmp_dir = Path(tmp)
            for idx, view in enumerate(views):
                variant_path = tmp_dir / f"{idx:02d}_{view}.png"
                _save_view(image, view, variant_path)
                presence, _ = extracted_target_presence(
                    image_path=variant_path,
                    payload=payload,
                )
                votes.append((view, presence))

    values = [presence for _, presence in votes]
    state, agreement = classify_presence_votes(values)
    present_votes = sum(v == "present" for v in values)
    absent_votes = len(values) - present_votes
    return EvidenceUncertainty(
        state=state,
        present_votes=present_votes,
        absent_votes=absent_votes,
        total_votes=len(values),
        agreement=agreement,
        votes=tuple(votes),
    )


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


def run_evidence_uncertainty_item(
    *,
    adapter: ModelAdapter,
    item_id: str,
    image_path: Path,
    payload: dict,
    views: tuple[str, ...] = DEFAULT_VIEWS,
) -> dict:
    start = time.perf_counter()
    signal = evidence_uncertainty(
        image_path=image_path,
        payload=payload,
        views=views,
    )

    vote_text = ", ".join(f"{name}={vote}" for name, vote in signal.votes)
    gate_line = (
        f"EVIDENCE_GATE: state={signal.state} agreement={signal.agreement:.3f} "
        f"present_votes={signal.present_votes}/{signal.total_votes} "
        f"absent_votes={signal.absent_votes}/{signal.total_votes}"
    )

    if signal.state == "absent":
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "prompt": gate_line + "\n" + vote_text,
            "raw_output": (
                "[EVIDENCE UNCERTAINTY GATE]\n"
                + gate_line
                + "\n"
                + vote_text
                + "\n\n[GATED RESULT]\n"
                + _absent_output()
            ),
            "prediction": "unknown",
            "latency_ms": latency_ms,
            "gate_state": signal.state,
            "agreement": signal.agreement,
            "votes": list(signal.votes),
            "model_calls": 0,
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

    route = "UNCERTAIN" if signal.state == "uncertain" else "PRESENT"
    return {
        "prompt": (
            gate_line
            + "\n"
            + vote_text
            + f"\n\n[{route} FOLLOW-UP]\n"
            + staged_prompt
        ),
        "raw_output": (
            "[EVIDENCE UNCERTAINTY GATE]\n"
            + gate_line
            + "\n"
            + vote_text
            + f"\n\n[{route} FOLLOW-UP]\n"
            + raw.strip()
        ),
        "prediction": prediction,
        "latency_ms": latency_ms,
        "gate_state": signal.state,
        "agreement": signal.agreement,
        "votes": list(signal.votes),
        "model_calls": 1,
    }


def run_evidence_uncertainty(
    *,
    benchmark_dir: Path,
    adapter: ModelAdapter,
    output_path: Path,
    views: tuple[str, ...] = DEFAULT_VIEWS,
) -> EvidenceUncertaintyResult:
    if not (benchmark_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Missing benchmark manifest: {benchmark_dir / 'manifest.json'}"
        )

    condition = "CONFLICT_EVIDENCE_UNCERTAINTY_CALIBRATED"
    cases = list(_iter_cases(benchmark_dir))
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

            result = run_evidence_uncertainty_item(
                adapter=adapter,
                item_id=item_id,
                image_path=case_dir / "scene.png",
                payload=payload,
                views=views,
            )
            gold = _gold_for_item(case_dir, item_id)
            record = {
                "scene_id": payload["scene_id"],
                "item_id": item_id,
                "family": "conflict",
                "condition": condition,
                "model": adapter.name,
                "prompt": result["prompt"],
                "image_path": None,
                "raw_output": result["raw_output"],
                "prediction": result["prediction"],
                "gold": gold,
                "correct": is_correct(result["prediction"], gold),
                "latency_ms": round(result["latency_ms"], 3),
                "gate_state": result["gate_state"],
                "agreement": result["agreement"],
                "votes": result["votes"],
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
    uncertain = sum(r.get("gate_state") == "uncertain" for r in records)
    hard_absent = sum(r.get("gate_state") == "absent" for r in records)
    hard_present = sum(r.get("gate_state") == "present" for r in records)
    model_calls = sum(int(r.get("model_calls", 0)) for r in records)

    return EvidenceUncertaintyResult(
        output_path=output_path,
        condition=condition,
        written=written,
        skipped=skipped,
        total_records=len(records),
        correct=correct,
        uncertain_records=uncertain,
        hard_absent_records=hard_absent,
        hard_present_records=hard_present,
        model_calls=model_calls,
        gold_accuracy={
            key: gold_correct[key] / total
            for key, total in sorted(gold_total.items())
        },
    )
