from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters.dummy import DummyAdapter
from detectivelab.evaluation.focused import build_focused_extracted_evidence
from detectivelab.evaluation.runner import VALID_CONDITIONS, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "benchmark_v0_0_1"


def _payload(scene_id: str, family: str) -> dict:
    payloads = json.loads((BENCHMARK / scene_id / "payloads.json").read_text())
    return next(p for p in payloads if p["family"] == family)


def _one_case_benchmark(tmp_path: Path, scene_id: str = "scene_0000") -> Path:
    copied = tmp_path / "benchmark"
    copied.mkdir()
    source_case = BENCHMARK / scene_id
    shutil.copytree(source_case, copied / scene_id)
    manifest = json.loads((BENCHMARK / "manifest.json").read_text())
    case = next(case for case in manifest["cases"] if case["scene_id"] == scene_id)
    manifest["cases"] = [case]
    manifest["scene_count"] = 1
    manifest["item_count"] = 3
    (copied / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return copied


def test_condition_is_registered() -> None:
    assert "EXTRACTED_FOCUSED" in VALID_CONDITIONS


def test_spatial_focused_evidence_contains_only_queried_pair() -> None:
    payload = _payload("scene_0001", "spatial")
    text = build_focused_extracted_evidence(
        image_path=BENCHMARK / "scene_0001" / "scene.png",
        payload=payload,
    )
    assert "green clock: present" in text
    assert "amber envelope: present" in text
    assert "green clock is left of amber envelope" in text
    assert "amber footprint" not in text
    assert "red door" not in text


def test_state_focused_evidence_contains_only_target() -> None:
    payload = _payload("scene_0002", "state")
    text = build_focused_extracted_evidence(
        image_path=BENCHMARK / "scene_0002" / "scene.png",
        payload=payload,
    )
    assert "blue window: closed" in text
    assert "white notebook" not in text
    assert "red door" not in text


def test_unknown_conflict_preserves_absence() -> None:
    payload = _payload("scene_0000", "conflict")
    text = build_focused_extracted_evidence(
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=payload,
    )
    assert "amber glass: not present" in text
    assert "red glass" not in text


def test_focused_evidence_is_deterministic() -> None:
    payload = _payload("scene_0005", "conflict")
    image = BENCHMARK / "scene_0005" / "scene.png"
    assert build_focused_extracted_evidence(image_path=image, payload=payload) == build_focused_extracted_evidence(image_path=image, payload=payload)


def test_focused_condition_uses_pixels_not_scene_json_and_attaches_no_image(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    (copied / "scene_0000" / "scene.json").unlink()

    output = tmp_path / "focused.jsonl"
    result = run_evaluation(
        benchmark_dir=copied,
        condition="EXTRACTED_FOCUSED",
        adapter=DummyAdapter(),
        output_path=output,
    )
    assert result.total_records == 3
    assert result.written == 3

    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    forbidden = ("object_id", "subject_id", "seed", "center_x", "center_y", "gold")
    for record in records:
        prompt = record["prompt"].lower()
        assert all(token not in prompt for token in forbidden)
        assert record["image_path"] is None
