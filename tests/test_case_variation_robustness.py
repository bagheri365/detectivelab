from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters.base import AdapterRequest
from detectivelab.evaluation.case_variation import (
    build_case_variation_prompt,
    generate_case_variants,
    run_case_variation_robustness,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "benchmark_v0_0_1"


def _payload(scene_id: str) -> dict:
    payloads = json.loads((BENCHMARK / scene_id / "payloads.json").read_text())
    return next(p for p in payloads if p["family"] == "conflict")


class UnknownDummy:
    @property
    def name(self) -> str:
        return "case-variation-dummy"

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


def test_case_variants_are_balanced_and_image_derived() -> None:
    variants = generate_case_variants(image_path=BENCHMARK / "scene_0000" / "scene.png")
    assert [variant.variant_id for variant in variants] == [
        "present_supported",
        "present_contradicted",
        "absent_unknown",
    ]
    assert {variant.gold for variant in variants} == {"supported", "contradicted", "unknown"}
    assert variants[0].label == variants[1].label
    assert variants[0].claimed_state != variants[1].claimed_state
    assert variants[2].label != variants[0].label


def test_case_variation_prompt_changes_claim_but_not_policy() -> None:
    image = BENCHMARK / "scene_0005" / "scene.png"
    payload = _payload("scene_0005")
    variants = generate_case_variants(image_path=image)
    prompts = [
        build_case_variation_prompt(
            image_path=image,
            base_payload=payload,
            policy="epistemic",
            variant=variant,
        )
        for variant in variants
    ]
    assert all("Mandatory epistemic rule:" in prompt for prompt in prompts)
    assert len(set(prompts)) == 3
    for variant, prompt in zip(variants, prompts):
        assert variant.witness in prompt


def test_case_variation_run_needs_no_scene_json_and_writes_three_variants(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    (copied / "scene_0000" / "scene.json").unlink()
    output = tmp_path / "case_variation.jsonl"

    result = run_case_variation_robustness(
        benchmark_dir=copied,
        policy="epistemic",
        adapter=UnknownDummy(),
        output_path=output,
    )

    assert result.total_records == 3
    assert result.correct == 1
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert {row["variant_id"] for row in rows} == {
        "present_supported",
        "present_contradicted",
        "absent_unknown",
    }
    assert {row["condition"] for row in rows} == {"CONFLICT_EPISTEMIC_CASE_VARIATION"}
    assert all(row["image_path"] is None for row in rows)
    assert all("scene.json" not in row["prompt"] for row in rows)
    for row in rows:
        for forbidden in ("object_id", "subject_id", "seed", "center_x", "center_y"):
            assert forbidden not in row["prompt"].lower()


def test_case_variation_is_balanced_across_full_slice() -> None:
    totals = {"supported": 0, "contradicted": 0, "unknown": 0}
    for scene_index in range(10):
        image = BENCHMARK / f"scene_{scene_index:04d}" / "scene.png"
        for variant in generate_case_variants(image_path=image):
            totals[variant.gold] += 1
    assert totals == {"supported": 10, "contradicted": 10, "unknown": 10}


def test_case_variation_run_is_resumable(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    output = tmp_path / "case_variation.jsonl"
    adapter = UnknownDummy()
    first = run_case_variation_robustness(
        benchmark_dir=copied,
        policy="staged",
        adapter=adapter,
        output_path=output,
    )
    second = run_case_variation_robustness(
        benchmark_dir=copied,
        policy="staged",
        adapter=adapter,
        output_path=output,
    )
    assert first.written == 3
    assert second.written == 0
    assert second.skipped == 3
    assert second.total_records == 3
