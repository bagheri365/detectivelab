from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters import DummyAdapter
from detectivelab.evaluation.runner import run_evaluation
from detectivelab.evaluation.scoring import normalize_prediction


def _benchmark_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "benchmark_v0_0"


def test_normalize_prediction_closed_form() -> None:
    assert normalize_prediction("YES", "yes_no") == "yes"
    assert normalize_prediction("The answer is no.", "yes_no") == "no"
    assert normalize_prediction("contradicted", "evidence_verdict") == "contradicted"
    assert normalize_prediction("maybe", "yes_no") == "invalid"


def test_dummy_question_run_writes_30_records_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    result = run_evaluation(
        benchmark_dir=_benchmark_root(),
        condition="QUESTION",
        adapter=DummyAdapter(),
        output_path=output,
    )
    assert result.written == 30
    assert result.total_records == 30

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 30
    assert all(record["condition"] == "QUESTION" for record in records)
    assert all(record["image_path"] is None for record in records)
    assert all("raw_output" in record for record in records)

    rerun = run_evaluation(
        benchmark_dir=_benchmark_root(),
        condition="QUESTION",
        adapter=DummyAdapter(),
        output_path=output,
    )
    assert rerun.written == 0
    assert rerun.skipped == 30
    assert len(output.read_text().splitlines()) == 30


def test_raw_run_points_to_scene_images(tmp_path: Path) -> None:
    output = tmp_path / "raw.jsonl"
    result = run_evaluation(
        benchmark_dir=_benchmark_root(),
        condition="RAW",
        adapter=DummyAdapter(),
        output_path=output,
    )
    assert result.total_records == 30
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(record["image_path"] for record in records)
    assert all(Path(record["image_path"]).name == "scene.png" for record in records)
