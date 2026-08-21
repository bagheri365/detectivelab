from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters.base import AdapterRequest
from detectivelab.evaluation.runner import VALID_CONDITIONS, run_evaluation
from detectivelab.evaluation.staged import (
    ConflictStages,
    build_conflict_staged_prompt,
    expected_stages_from_extracted_evidence,
    parse_conflict_stages,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "benchmark_v0_0_1"


def _payload(scene_id: str) -> dict:
    payloads = json.loads((BENCHMARK / scene_id / "payloads.json").read_text())
    return next(p for p in payloads if p["family"] == "conflict")


class StagedDummy:
    @property
    def name(self) -> str:
        return "staged-dummy"

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
    assert "CONFLICT_STAGED" in VALID_CONDITIONS


def test_staged_prompt_uses_focused_extracted_evidence() -> None:
    payload = _payload("scene_0005")
    prompt = build_conflict_staged_prompt(
        image_path=BENCHMARK / "scene_0005" / "scene.png",
        payload=payload,
    )
    assert "Focused visible scene evidence:" in prompt
    assert "blue notebook: open" in prompt
    assert "EXISTENCE:" in prompt
    assert "PHYSICAL_STATE:" in prompt
    assert "AGREEMENT:" in prompt
    assert "VERDICT:" in prompt


def test_parse_staged_output() -> None:
    parsed = parse_conflict_stages(
        "EXISTENCE: present\nPHYSICAL_STATE: open\nAGREEMENT: contradicts\nVERDICT: contradicted"
    )
    assert parsed == ConflictStages("present", "open", "contradicts", "contradicted")


def test_expected_stages_are_relative_to_extractor() -> None:
    supported = expected_stages_from_extracted_evidence(
        image_path=BENCHMARK / "scene_0001" / "scene.png",
        payload=_payload("scene_0001"),
    )
    unknown = expected_stages_from_extracted_evidence(
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=_payload("scene_0000"),
    )
    assert supported.verdict == "supported"
    assert unknown == ConflictStages("absent", "not_applicable", "unknown", "unknown")


def test_staged_condition_runs_only_conflict_and_needs_no_scene_json(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    (copied / "scene_0000" / "scene.json").unlink()
    output = tmp_path / "staged.jsonl"

    result = run_evaluation(
        benchmark_dir=copied,
        condition="CONFLICT_STAGED",
        adapter=StagedDummy(),
        output_path=output,
    )

    assert result.total_records == 1
    assert result.written == 1
    record = json.loads(output.read_text().strip())
    assert record["family"] == "conflict"
    assert record["image_path"] is None
    prompt = record["prompt"].lower()
    for forbidden in ("object_id", "subject_id", "seed", "center_x", "center_y", "gold"):
        assert forbidden not in prompt


def test_epistemic_condition_is_registered() -> None:
    assert "CONFLICT_EPISTEMIC" in VALID_CONDITIONS


def test_epistemic_prompt_makes_absence_mapping_mandatory() -> None:
    from detectivelab.evaluation.staged import build_conflict_epistemic_prompt

    payload = _payload("scene_0000")
    prompt = build_conflict_epistemic_prompt(
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=payload,
    )
    assert "Mandatory epistemic rule:" in prompt
    assert "AGREEMENT must be unknown" in prompt
    assert "VERDICT must be unknown" in prompt
    assert "is NOT evidence that contradicts the witness" in prompt


def test_parser_canonicalizes_explanatory_suffixes_and_aliases() -> None:
    parsed = parse_conflict_stages(
        "\n".join([
            "EXISTENCE: present - the object is visible",
            "PHYSICAL_STATE: open - observed in evidence",
            "AGREEMENT: contradicts - testimony says closed",
            "VERDICT: contradicts - physical evidence has priority",
        ])
    )
    assert parsed == ConflictStages("present", "open", "contradicts", "contradicted")


def test_epistemic_condition_runs_only_conflict_and_needs_no_scene_json(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path)
    (copied / "scene_0000" / "scene.json").unlink()
    output = tmp_path / "epistemic.jsonl"

    result = run_evaluation(
        benchmark_dir=copied,
        condition="CONFLICT_EPISTEMIC",
        adapter=StagedDummy(),
        output_path=output,
    )

    assert result.total_records == 1
    record = json.loads(output.read_text().strip())
    assert record["family"] == "conflict"
    assert record["prediction"] == "unknown"
    assert record["image_path"] is None
    assert "Mandatory epistemic rule:" in record["prompt"]
