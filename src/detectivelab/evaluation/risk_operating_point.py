from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from detectivelab.adapters.base import AdapterRequest, ModelAdapter
from detectivelab.evaluation.scoring import is_correct
from detectivelab.evaluation.staged import (
    build_conflict_staged_prompt,
    parse_conflict_stages,
)
from detectivelab.evaluation.uncertainty_prediction import _degrade


POLICIES = (
    "NEVER_ESCALATE",
    "STABILITY_ONLY",
    "LOW_CONTRAST_ONLY",
    "LOW_EDGE_ONLY",
    "QUALITY_ANY",
    "ANY_SIGNAL",
    "TWO_PLUS",
    "ALWAYS_ESCALATE",
)


@dataclass(frozen=True)
class QualityThresholds:
    contrast_floor: float
    edge_floor: float
    clean_contrast_min: float
    clean_edge_min: float
    multiplier: float


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _quality_scores(image: Image.Image) -> tuple[float, float]:
    gray = image.convert("L")
    contrast = float(ImageStat.Stat(gray).stddev[0])

    if gray.width < 2 or gray.height < 2:
        return contrast, 0.0

    h_a = gray.crop((0, 0, gray.width - 1, gray.height))
    h_b = gray.crop((1, 0, gray.width, gray.height))
    h_diff = ImageChops.difference(h_a, h_b)
    h_edge = float(ImageStat.Stat(h_diff).mean[0])

    v_a = gray.crop((0, 0, gray.width, gray.height - 1))
    v_b = gray.crop((0, 1, gray.width, gray.height))
    v_diff = ImageChops.difference(v_a, v_b)
    v_edge = float(ImageStat.Stat(v_diff).mean[0])

    edge = (h_edge + v_edge) / 2.0
    return contrast, edge


def _case_map(benchmark_dir: Path) -> dict[str, Path]:
    manifest = _read_json(benchmark_dir / "manifest.json")
    out: dict[str, Path] = {}
    for case in manifest["cases"]:
        case_dir = benchmark_dir / case["path"]
        payloads = _read_json(case_dir / "payloads.json")
        for payload in payloads:
            if payload.get("family") == "conflict":
                out[payload["item_id"]] = case_dir
    return out


def _payload_for(case_dir: Path, item_id: str) -> dict:
    payloads = _read_json(case_dir / "payloads.json")
    return next(p for p in payloads if p.get("item_id") == item_id)


def calibrate_quality_thresholds(
    benchmark_dir: Path,
    *,
    multiplier: float = 0.90,
) -> QualityThresholds:
    """Calibrate conservative quality floors from clean benchmark images only.

    No degraded outcomes or extraction-failure labels are used.
    """
    if not 0 < multiplier <= 1:
        raise ValueError("multiplier must be in (0, 1]")

    case_dirs = sorted(set(_case_map(benchmark_dir).values()))
    contrast_scores: list[float] = []
    edge_scores: list[float] = []

    for case_dir in case_dirs:
        with Image.open(case_dir / "scene.png") as image:
            contrast, edge = _quality_scores(image.convert("RGB"))
        contrast_scores.append(contrast)
        edge_scores.append(edge)

    if not contrast_scores:
        raise ValueError("benchmark contains no conflict cases")

    contrast_min = min(contrast_scores)
    edge_min = min(edge_scores)
    return QualityThresholds(
        contrast_floor=contrast_min * multiplier,
        edge_floor=edge_min * multiplier,
        clean_contrast_min=contrast_min,
        clean_edge_min=edge_min,
        multiplier=multiplier,
    )


def _signals(
    *,
    gate_state: str,
    contrast_score: float,
    edge_score: float,
    thresholds: QualityThresholds,
) -> dict[str, bool]:
    return {
        "instability": gate_state == "uncertain",
        "low_contrast": contrast_score < thresholds.contrast_floor,
        "low_edge": edge_score < thresholds.edge_floor,
    }


def policy_escalates(policy: str, signals: dict[str, bool]) -> bool:
    if policy == "NEVER_ESCALATE":
        return False
    if policy == "STABILITY_ONLY":
        return signals["instability"]
    if policy == "LOW_CONTRAST_ONLY":
        return signals["low_contrast"]
    if policy == "LOW_EDGE_ONLY":
        return signals["low_edge"]
    if policy == "QUALITY_ANY":
        return signals["low_contrast"] or signals["low_edge"]
    if policy == "ANY_SIGNAL":
        return any(signals.values())
    if policy == "TWO_PLUS":
        return sum(bool(v) for v in signals.values()) >= 2
    if policy == "ALWAYS_ESCALATE":
        return True
    raise ValueError(f"unknown policy: {policy}")


def _staged_prediction(
    *,
    adapter: ModelAdapter,
    image_path: Path,
    payload: dict,
) -> tuple[str, str]:
    prompt = build_conflict_staged_prompt(
        image_path=image_path,
        payload=payload,
    )
    request = AdapterRequest(
        item_id=payload["item_id"],
        family="conflict",
        answer_type="evidence_verdict",
        prompt=prompt,
        image_path=None,
    )
    raw = adapter.predict(request)
    stages = parse_conflict_stages(raw)
    prediction = stages.verdict if stages is not None else "invalid"
    return prediction, raw


def _counterfactual_cache(path: Path) -> dict[tuple, dict]:
    cache = {}
    if not path.exists():
        return cache
    for row in _load_jsonl(path):
        key = (
            row["item_id"],
            row["model"],
            row["degradation_family"],
            row["severity"],
        )
        cache[key] = row
    return cache


def build_risk_records(
    *,
    benchmark_dir: Path,
    v09_paths: list[Path],
    adapter: ModelAdapter,
    cache_path: Path,
    quality_multiplier: float = 0.90,
) -> tuple[list[dict], QualityThresholds]:
    """Build one reusable v0.10 record per v0.9 scene × severity.

    For records that originally took zero model calls, this function obtains a
    single counterfactual staged prediction and caches it. Policies can then be
    compared offline without repeated model calls.
    """
    thresholds = calibrate_quality_thresholds(
        benchmark_dir,
        multiplier=quality_multiplier,
    )
    cases = _case_map(benchmark_dir)
    cache = _counterfactual_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in v09_paths:
        rows.extend(_load_jsonl(path))

    built: list[dict] = []

    with cache_path.open("a", encoding="utf-8") as cache_stream:
        for row in rows:
            item_id = row["item_id"]
            case_dir = cases[item_id]
            payload = _payload_for(case_dir, item_id)

            with Image.open(case_dir / "scene.png") as original:
                original = original.convert("RGB")
                degraded = _degrade(
                    original,
                    row["degradation_family"],
                    row["severity"],
                )

                contrast_score, edge_score = _quality_scores(degraded)

                with tempfile.TemporaryDirectory(
                    prefix="detectivelab_v010_"
                ) as tmp:
                    degraded_path = Path(tmp) / "degraded.png"
                    degraded.save(degraded_path)

                    if int(row["model_calls"]) > 0:
                        staged_prediction = row["prediction"]
                        staged_raw = row.get("raw_output", "")
                        cf_model_call = 0
                    else:
                        key = (
                            item_id,
                            row["model"],
                            row["degradation_family"],
                            row["severity"],
                        )
                        cached = cache.get(key)
                        if cached is None:
                            start = time.perf_counter()
                            staged_prediction, staged_raw = _staged_prediction(
                                adapter=adapter,
                                image_path=degraded_path,
                                payload=payload,
                            )
                            cached = {
                                "item_id": item_id,
                                "model": row["model"],
                                "degradation_family": row["degradation_family"],
                                "severity": row["severity"],
                                "staged_prediction": staged_prediction,
                                "raw_output": staged_raw,
                                "latency_ms": round(
                                    (time.perf_counter() - start) * 1000.0, 3
                                ),
                            }
                            cache_stream.write(
                                json.dumps(cached, sort_keys=True) + "\n"
                            )
                            cache_stream.flush()
                            cache[key] = cached
                            cf_model_call = 1
                        else:
                            staged_prediction = cached["staged_prediction"]
                            staged_raw = cached.get("raw_output", "")
                            cf_model_call = 0

            signals = _signals(
                gate_state=row["gate_state"],
                contrast_score=contrast_score,
                edge_score=edge_score,
                thresholds=thresholds,
            )

            built.append(
                {
                    **row,
                    "contrast_score": round(contrast_score, 6),
                    "edge_score": round(edge_score, 6),
                    "signals": signals,
                    "staged_prediction": staged_prediction,
                    "staged_correct": is_correct(staged_prediction, row["gold"]),
                    "counterfactual_call_written_this_run": cf_model_call,
                }
            )

    return built, thresholds


def evaluate_policy(records: list[dict], policy: str) -> dict:
    total = len(records)
    failures = sum(bool(r["extraction_failed"]) for r in records)

    escalated = 0
    covered_failures = 0
    correct = 0
    model_calls = 0

    for r in records:
        fire = policy_escalates(policy, r["signals"])
        # Present/uncertain records already go to the model under the base gate.
        base_calls_model = int(r["model_calls"]) > 0
        effective_escalation = fire and not base_calls_model

        if effective_escalation:
            escalated += 1

        if r["extraction_failed"] and fire:
            covered_failures += 1

        if base_calls_model or fire:
            prediction = r["staged_prediction"]
            model_calls += 1
        else:
            prediction = r["prediction"]

        if is_correct(prediction, r["gold"]):
            correct += 1

    nonfailures = total - failures
    fired = sum(policy_escalates(policy, r["signals"]) for r in records)
    fired_failures = sum(
        bool(r["extraction_failed"]) and policy_escalates(policy, r["signals"])
        for r in records
    )
    fired_nonfailures = fired - fired_failures

    recall = fired_failures / failures if failures else 0.0
    precision = fired_failures / fired if fired else 0.0
    fnr = 1.0 - recall if failures else 0.0

    return {
        "policy": policy,
        "records": total,
        "failures": failures,
        "failure_recall": recall,
        "failure_precision": precision,
        "false_negative_rate": fnr,
        "signal_fire_rate": fired / total if total else 0.0,
        "incremental_escalation_rate": escalated / total if total else 0.0,
        "model_call_rate": model_calls / total if total else 0.0,
        "downstream_accuracy": correct / total if total else 0.0,
        "covered_failures": covered_failures,
        "signal_false_positives": fired_nonfailures,
    }


def evaluate_all_policies(records: list[dict]) -> list[dict]:
    return [evaluate_policy(records, policy) for policy in POLICIES]
