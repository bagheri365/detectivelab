from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from detectivelab.adapters.base import AdapterRequest, ModelAdapter

from .records import PredictionRecord
from .scoring import is_correct, normalize_prediction
from .structured import build_structured_evidence


VALID_CONDITIONS = {"QUESTION", "RAW", "ORACLE_STRUCTURED"}


@dataclass(frozen=True)
class EvaluationResult:
    output_path: Path
    written: int
    skipped: int
    total_records: int
    correct: int
    family_accuracy: dict[str, float]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_records if self.total_records else 0.0


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_prompt(payload: dict, structured_evidence: str | None = None) -> str:
    lines: list[str] = []
    if structured_evidence is not None:
        lines.append(structured_evidence)
    for entry in payload.get("context", []):
        entry_type = entry.get("type", "context").replace("_", " ").title()
        lines.append(f"{entry_type}: {entry['text']}")
    lines.append(f"Question: {payload['question']}")
    if payload["answer_type"] == "yes_no":
        lines.append("Answer with exactly one label: yes or no.")
    elif payload["answer_type"] == "evidence_verdict":
        lines.append("Answer with exactly one label: supported, contradicted, or unknown.")
    else:
        raise ValueError(f"Unsupported answer_type: {payload['answer_type']}")
    return "\n".join(lines)


def _iter_items(benchmark_dir: Path) -> Iterable[tuple[Path, dict, dict]]:
    manifest = _read_json(benchmark_dir / "manifest.json")
    for case in manifest["cases"]:
        case_dir = benchmark_dir / case["path"]
        payloads = _read_json(case_dir / "payloads.json")
        questions = _read_json(case_dir / "questions.json")
        gold_by_id = {item["item_id"]: item for item in questions}
        for payload in payloads:
            yield case_dir, payload, gold_by_id[payload["item_id"]]


def _resume_keys(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line_no, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {output_path}:{line_no}") from exc
        keys.add((record["item_id"], record["condition"], record["model"]))
    return keys


def _load_records(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    return [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_evaluation(
    *,
    benchmark_dir: Path,
    condition: str,
    adapter: ModelAdapter,
    output_path: Path,
) -> EvaluationResult:
    condition = condition.upper()
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}")

    if not (benchmark_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing benchmark manifest: {benchmark_dir / 'manifest.json'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _resume_keys(output_path)
    written = 0
    skipped = 0

    with output_path.open("a", encoding="utf-8") as stream:
        for case_dir, payload, gold_item in _iter_items(benchmark_dir):
            resume_key = (payload["item_id"], condition, adapter.name)
            if resume_key in completed:
                skipped += 1
                continue

            image_path = case_dir / "scene.png" if condition == "RAW" else None
            structured_evidence = None
            if condition == "ORACLE_STRUCTURED":
                scene = _read_json(case_dir / "scene.json")
                structured_evidence = build_structured_evidence(scene)

            request = AdapterRequest(
                item_id=payload["item_id"],
                family=payload["family"],
                answer_type=payload["answer_type"],
                prompt=_build_prompt(payload, structured_evidence),
                image_path=image_path,
            )

            start = time.perf_counter()
            raw_output = adapter.predict(request)
            latency_ms = (time.perf_counter() - start) * 1000.0
            prediction = normalize_prediction(raw_output, payload["answer_type"])
            gold = gold_item["answer"].strip().lower()

            record = PredictionRecord(
                scene_id=payload["scene_id"],
                item_id=payload["item_id"],
                family=payload["family"],
                condition=condition,
                model=adapter.name,
                prompt=request.prompt,
                image_path=str(image_path) if image_path is not None else None,
                raw_output=raw_output,
                prediction=prediction,
                gold=gold,
                correct=is_correct(prediction, gold),
                latency_ms=round(latency_ms, 3),
            )
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            stream.flush()
            completed.add(resume_key)
            written += 1

    records = [
        record
        for record in _load_records(output_path)
        if record.get("condition") == condition and record.get("model") == adapter.name
    ]
    correct = sum(bool(record["correct"]) for record in records)
    family_totals = Counter(record["family"] for record in records)
    family_correct: dict[str, int] = defaultdict(int)
    for record in records:
        family_correct[record["family"]] += int(bool(record["correct"]))
    family_accuracy = {
        family: family_correct[family] / count
        for family, count in sorted(family_totals.items())
    }

    return EvaluationResult(
        output_path=output_path,
        written=written,
        skipped=skipped,
        total_records=len(records),
        correct=correct,
        family_accuracy=family_accuracy,
    )
