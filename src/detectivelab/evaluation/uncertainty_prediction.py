from __future__ import annotations

import json
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from detectivelab.adapters.base import AdapterRequest, ModelAdapter
from detectivelab.evaluation.conditional import extracted_target_presence
from detectivelab.evaluation.evidence_uncertainty import (
    CALIBRATED_VIEWS,
    evidence_uncertainty,
)
from detectivelab.evaluation.gate_corruption import _gold_for_item
from detectivelab.evaluation.scoring import is_correct
from detectivelab.evaluation.staged import (
    build_conflict_staged_prompt,
    parse_conflict_stages,
)

DEGRADATION_GRID = {
    "blur": (0.0, 0.4, 0.8, 1.2, 1.6),
    "downsample": (1.0, 0.8, 0.6, 0.4, 0.25),
    "contrast": (1.0, 0.8, 0.6, 0.4, 0.25),
    "occlusion": (0.0, 0.05, 0.10, 0.15, 0.20),
}


@dataclass(frozen=True)
class PredictionSummary:
    output_path: Path
    total_records: int
    written: int
    skipped: int
    extraction_failures: int
    uncertainty_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    downstream_correct: int
    model_calls: int

    @property
    def failure_recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def failure_precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def false_negative_rate(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.false_negative / denom if denom else 0.0

    @property
    def downstream_accuracy(self) -> float:
        return self.downstream_correct / self.total_records if self.total_records else 0.0

    @property
    def model_call_rate(self) -> float:
        return self.model_calls / self.total_records if self.total_records else 0.0


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


def _degrade(image: Image.Image, family: str, severity: float) -> Image.Image:
    if family == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))

    if family == "downsample":
        if severity <= 0 or severity > 1:
            raise ValueError("downsample severity must be in (0, 1]")
        w, h = image.size
        small = image.resize(
            (max(1, round(w * severity)), max(1, round(h * severity))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((w, h), Image.Resampling.NEAREST)

    if family == "contrast":
        if severity <= 0:
            raise ValueError("contrast severity must be > 0")
        return ImageEnhance.Contrast(image).enhance(severity)

    if family == "occlusion":
        if not 0 <= severity <= 1:
            raise ValueError("occlusion severity must be in [0, 1]")
        if severity == 0:
            return image
        out = image.copy()
        w, h = out.size
        box_w = max(1, round(w * severity))
        box_h = max(1, round(h * severity))
        x0 = max(0, (w - box_w) // 2)
        y0 = max(0, (h - box_h) // 2)
        draw = ImageDraw.Draw(out)
        draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(255, 255, 255))
        return out

    raise ValueError(f"unknown degradation family: {family}")


def _severity_rank(family: str, severity: float) -> int:
    return DEGRADATION_GRID[family].index(severity)


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


def run_degraded_item(
    *,
    adapter: ModelAdapter,
    item_id: str,
    degraded_image_path: Path,
    payload: dict,
) -> dict:
    signal = evidence_uncertainty(
        image_path=degraded_image_path,
        payload=payload,
        views=CALIBRATED_VIEWS,
    )

    if signal.state == "absent":
        return {
            "prediction": "unknown",
            "gate_state": signal.state,
            "agreement": signal.agreement,
            "votes": list(signal.votes),
            "model_calls": 0,
            "raw_output": _absent_output(),
        }

    staged_prompt = build_conflict_staged_prompt(
        image_path=degraded_image_path,
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
    stages = parse_conflict_stages(raw)
    prediction = stages.verdict if stages is not None else "invalid"
    return {
        "prediction": prediction,
        "gate_state": signal.state,
        "agreement": signal.agreement,
        "votes": list(signal.votes),
        "model_calls": 1,
        "raw_output": raw,
    }


def run_uncertainty_prediction(
    *,
    benchmark_dir: Path,
    degradation_family: str,
    adapter: ModelAdapter,
    output_path: Path,
) -> PredictionSummary:
    if degradation_family not in DEGRADATION_GRID:
        raise ValueError(
            f"degradation_family must be one of {sorted(DEGRADATION_GRID)}"
        )

    cases = list(_iter_cases(benchmark_dir))
    condition = f"UNCERTAINTY_PREDICTION_{degradation_family.upper()}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_records(output_path)
    completed = {
        (
            r["item_id"],
            r["condition"],
            r["model"],
            r["degradation_family"],
            r["severity"],
        )
        for r in existing
    }

    clean_presence: dict[str, str] = {}
    for case_dir, payload in cases:
        p, _ = extracted_target_presence(
            image_path=case_dir / "scene.png",
            payload=payload,
        )
        clean_presence[payload["item_id"]] = p

    written = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as stream:
        for case_dir, payload in cases:
            item_id = payload["item_id"]
            gold = _gold_for_item(case_dir, item_id)

            with Image.open(case_dir / "scene.png") as original:
                original = original.convert("RGB")

                for severity in DEGRADATION_GRID[degradation_family]:
                    resume_key = (
                        item_id,
                        condition,
                        adapter.name,
                        degradation_family,
                        severity,
                    )
                    if resume_key in completed:
                        skipped += 1
                        continue

                    degraded = _degrade(original, degradation_family, severity)

                    with tempfile.TemporaryDirectory(
                        prefix="detectivelab_v09_"
                    ) as tmp:
                        degraded_path = Path(tmp) / "degraded.png"
                        degraded.save(degraded_path)

                        degraded_presence, _ = extracted_target_presence(
                            image_path=degraded_path,
                            payload=payload,
                        )
                        extraction_failed = (
                            degraded_presence != clean_presence[item_id]
                        )

                        start = time.perf_counter()
                        result = run_degraded_item(
                            adapter=adapter,
                            item_id=item_id,
                            degraded_image_path=degraded_path,
                            payload=payload,
                        )
                        latency_ms = (time.perf_counter() - start) * 1000.0

                    uncertainty_positive = result["gate_state"] == "uncertain"

                    record = {
                        "scene_id": payload["scene_id"],
                        "item_id": item_id,
                        "family": "conflict",
                        "condition": condition,
                        "model": adapter.name,
                        "degradation_family": degradation_family,
                        "severity": severity,
                        "severity_rank": _severity_rank(
                            degradation_family, severity
                        ),
                        "clean_presence": clean_presence[item_id],
                        "degraded_presence": degraded_presence,
                        "extraction_failed": extraction_failed,
                        "gate_state": result["gate_state"],
                        "uncertainty_positive": uncertainty_positive,
                        "agreement": result["agreement"],
                        "votes": result["votes"],
                        "prediction": result["prediction"],
                        "gold": gold,
                        "correct": is_correct(result["prediction"], gold),
                        "model_calls": result["model_calls"],
                        "latency_ms": round(latency_ms, 3),
                        "raw_output": result["raw_output"],
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

    tp = fp = fn = tn = 0
    for r in records:
        failure = bool(r["extraction_failed"])
        uncertain = bool(r["uncertainty_positive"])
        if failure and uncertain:
            tp += 1
        elif not failure and uncertain:
            fp += 1
        elif failure and not uncertain:
            fn += 1
        else:
            tn += 1

    return PredictionSummary(
        output_path=output_path,
        total_records=len(records),
        written=written,
        skipped=skipped,
        extraction_failures=sum(bool(r["extraction_failed"]) for r in records),
        uncertainty_positive=sum(bool(r["uncertainty_positive"]) for r in records),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        downstream_correct=sum(bool(r["correct"]) for r in records),
        model_calls=sum(int(r["model_calls"]) for r in records),
    )


def earliest_events(records: list[dict]) -> dict[str, dict]:
    """Return earliest uncertainty and failure severity per item."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r["item_id"]].append(r)

    out = {}
    for item_id, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["severity_rank"])
        uncertainty = next(
            (r["severity"] for r in rows if r["uncertainty_positive"]),
            None,
        )
        failure = next(
            (r["severity"] for r in rows if r["extraction_failed"]),
            None,
        )
        out[item_id] = {
            "earliest_uncertainty": uncertainty,
            "earliest_failure": failure,
        }
    return out
