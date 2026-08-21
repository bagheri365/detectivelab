from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters import DummyAdapter
from detectivelab.evaluation.runner import run_evaluation
from detectivelab.extraction import extract_scene_facts, extract_structured_evidence


BENCHMARK = Path("artifacts/benchmark_v0_0_1")


def _gold_visible(scene_dir: Path):
    scene = json.loads((scene_dir / "scene.json").read_text())
    return sorted(
        (obj["color"], obj["kind"], obj.get("state"))
        for obj in scene["objects"]
        if obj.get("visibility") == "visible"
    )


def test_extractor_recovers_frozen_visible_objects() -> None:
    for scene_dir in sorted(BENCHMARK.glob("scene_*")):
        extracted = sorted(
            (obj.color, obj.kind, obj.state)
            for obj in extract_scene_facts(scene_dir / "scene.png")
        )
        assert extracted == _gold_visible(scene_dir)


def test_extractor_is_deterministic() -> None:
    image = BENCHMARK / "scene_0002" / "scene.png"
    assert extract_structured_evidence(image) == extract_structured_evidence(image)


def test_extracted_evidence_has_no_hidden_metadata() -> None:
    evidence = extract_structured_evidence(BENCHMARK / "scene_0002" / "scene.png")
    forbidden = ["window_1", "seed", "object_id", "subject_id", "0.3162", "answer"]
    assert not any(token in evidence for token in forbidden)
    assert "blue window: closed" in evidence


def test_extracted_condition_does_not_require_scene_json(tmp_path: Path) -> None:
    # Copy the frozen benchmark, then delete every hidden-state scene file.
    # EXTRACTED_STRUCTURED must remain runnable because its evidence comes only
    # from rendered PNGs plus participant-facing payload/question files.
    benchmark = tmp_path / "benchmark"
    shutil.copytree(BENCHMARK, benchmark)
    for scene_json in benchmark.glob("scene_*/scene.json"):
        scene_json.unlink()

    output = tmp_path / "predictions.jsonl"
    result = run_evaluation(
        benchmark_dir=benchmark,
        condition="EXTRACTED_STRUCTURED",
        adapter=DummyAdapter(),
        output_path=output,
    )
    assert result.total_records == 30
    assert result.written == 30


def test_extracted_condition_attaches_no_image_to_reasoner(tmp_path: Path) -> None:
    class SpyAdapter:
        name = "spy"

        def __init__(self) -> None:
            self.image_paths = []

        def predict(self, request):
            self.image_paths.append(request.image_path)
            return "yes" if request.answer_type == "yes_no" else "unknown"

    adapter = SpyAdapter()
    run_evaluation(
        benchmark_dir=BENCHMARK,
        condition="EXTRACTED_STRUCTURED",
        adapter=adapter,
        output_path=tmp_path / "spy.jsonl",
    )
    assert adapter.image_paths
    assert all(path is None for path in adapter.image_paths)
