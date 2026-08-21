from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters.base import AdapterRequest
from detectivelab.evaluation.robustness import (
    PARAPHRASE_VARIANTS,
    build_paraphrase_prompt,
    paraphrase_witness,
    run_paraphrase_robustness,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "benchmark_v0_0_1"


def _payload(scene_id: str) -> dict:
    payloads = json.loads((BENCHMARK / scene_id / "payloads.json").read_text())
    return next(p for p in payloads if p["family"] == "conflict")


class RobustnessDummy:
    @property
    def name(self) -> str:
        return "robustness-dummy"

    def predict(self, request: AdapterRequest) -> str:
        assert request.family == "conflict"
        return "\n".join([
            "EXISTENCE: absent",
            "PHYSICAL_STATE: not_applicable",
            "AGREEMENT: unknown",
            "VERDICT: unknown",
        ])


def _one_case_benchmark(tmp_path: Path, scene_id: str = "scene_0000") -> Path:
    copied = tmp_path / "benchmark"
    copied.mkdir()
    shutil.copytree(BENCHMARK / scene_id, copied / scene_id)
    manifest = json.loads((BENCHMARK / "manifest.json").read_text())
    case = next(case for case in manifest["cases"] if case["scene_id"] == scene_id)
    manifest["cases"] = [case]
    manifest["scene_count"] = 1
    manifest["item_count"] = 3
    (copied / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return copied


def test_paraphrases_are_deterministic_and_semantics_preserving() -> None:
    payload = _payload("scene_0005")
    variants = [paraphrase_witness(payload, variant) for variant in PARAPHRASE_VARIANTS]
    assert len(variants) == 3
    assert len(set(variants)) == 3
    for text in variants:
        lowered = text.lower()
        assert "blue notebook" in lowered
        assert "closed" in lowered


def test_paraphrase_prompt_keeps_focused_visual_evidence_fixed() -> None:
    payload = _payload("scene_0005")
    prompts = [
        build_paraphrase_prompt(
            image_path=BENCHMARK / "scene_0005" / "scene.png",
            payload=payload,
            policy="epistemic",
            variant=variant,
        )
        for variant in PARAPHRASE_VARIANTS
    ]
    for prompt in prompts:
        assert "Focused visible scene evidence:\n- blue notebook: open" in prompt
        assert "Mandatory epistemic rule:" in prompt
    assert len({p.split("Witness Testimony: ", 1)[1].split("\n", 1)[0] for p in prompts}) == 3


def test_paraphrase_run_needs_no_scene_json_and_writes_three_variants(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    (copied / "scene_0000" / "scene.json").unlink()
    output = tmp_path / "paraphrase.jsonl"

    result = run_paraphrase_robustness(
        benchmark_dir=copied,
        policy="epistemic",
        adapter=RobustnessDummy(),
        output_path=output,
    )

    assert result.total_records == 3
    assert result.correct == 3
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert {row["variant_id"] for row in rows} == {v.variant_id for v in PARAPHRASE_VARIANTS}
    assert {row["condition"] for row in rows} == {"CONFLICT_EPISTEMIC_PARAPHRASE"}
    assert all(row["image_path"] is None for row in rows)
    assert all("scene.json" not in row["prompt"] for row in rows)
    for row in rows:
        for forbidden in ("object_id", "subject_id", "seed", "center_x", "center_y", "gold"):
            assert forbidden not in row["prompt"].lower()


def test_paraphrase_run_is_resumable(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    output = tmp_path / "paraphrase.jsonl"
    adapter = RobustnessDummy()

    first = run_paraphrase_robustness(
        benchmark_dir=copied,
        policy="staged",
        adapter=adapter,
        output_path=output,
    )
    second = run_paraphrase_robustness(
        benchmark_dir=copied,
        policy="staged",
        adapter=adapter,
        output_path=output,
    )
    assert first.written == 3
    assert second.written == 0
    assert second.skipped == 3
    assert second.total_records == 3
