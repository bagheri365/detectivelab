from __future__ import annotations

import json
import shutil
from pathlib import Path

from detectivelab.adapters.base import AdapterRequest
from detectivelab.evaluation.case_variation import CASE_VARIATION_POLICIES
from detectivelab.evaluation.conditional import (
    build_conflict_existence_prompt,
    parse_existence_decision,
    run_conditional_conflict,
)
from detectivelab.evaluation.robustness import ROBUSTNESS_POLICIES
from detectivelab.evaluation.runner import VALID_CONDITIONS, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "benchmark_v0_0_1"


def _payload(scene_id: str) -> dict:
    payloads = json.loads((BENCHMARK / scene_id / "payloads.json").read_text())
    return next(p for p in payloads if p["family"] == "conflict")


class GateDummy:
    def __init__(self, existence: str, staged_verdict: str = "contradicted") -> None:
        self.existence = existence
        self.staged_verdict = staged_verdict
        self.calls: list[AdapterRequest] = []

    @property
    def name(self) -> str:
        return "gate-dummy"

    def predict(self, request: AdapterRequest) -> str:
        self.calls.append(request)
        if "Determine only whether" in request.prompt:
            return f"EXISTENCE: {self.existence}"
        if self.staged_verdict == "contradicted":
            return "\n".join([
                "EXISTENCE: present",
                "PHYSICAL_STATE: open",
                "AGREEMENT: contradicts",
                "VERDICT: contradicted",
            ])
        return "\n".join([
            "EXISTENCE: present",
            "PHYSICAL_STATE: open",
            "AGREEMENT: supports",
            "VERDICT: supported",
        ])


def _one_case_benchmark(tmp_path: Path, scene_id: str) -> Path:
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


def test_conditional_policy_is_registered_everywhere() -> None:
    assert "CONFLICT_CONDITIONAL" in VALID_CONDITIONS
    assert "conditional" in ROBUSTNESS_POLICIES
    assert "conditional" in CASE_VARIATION_POLICIES


def test_existence_parser_accepts_explanatory_suffix() -> None:
    assert parse_existence_decision("EXISTENCE: absent - target not found") == "absent"
    assert parse_existence_decision("EXISTENCE: present (visible)") == "present"


def test_existence_prompt_uses_image_derived_focused_evidence() -> None:
    prompt = build_conflict_existence_prompt(
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=_payload("scene_0000"),
    )
    assert "Focused visible scene evidence:" in prompt
    assert "EXISTENCE: present or absent" in prompt
    assert "Do not decide agreement or verdict yet." in prompt


def test_absent_gate_uses_one_call_and_forces_unknown() -> None:
    adapter = GateDummy("absent")
    result = run_conditional_conflict(
        adapter=adapter,
        item_id="scene_0000__conflict",
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=_payload("scene_0000"),
    )
    assert result.prediction == "unknown"
    assert result.gated is True
    assert result.model_calls == 1
    assert len(adapter.calls) == 1
    assert "VERDICT: unknown" in result.raw_output


def test_present_gate_uses_unchanged_staged_followup() -> None:
    adapter = GateDummy("present", staged_verdict="contradicted")
    result = run_conditional_conflict(
        adapter=adapter,
        item_id="scene_0005__conflict",
        image_path=BENCHMARK / "scene_0005" / "scene.png",
        payload=_payload("scene_0005"),
    )
    assert result.prediction == "contradicted"
    assert result.gated is False
    assert result.model_calls == 2
    assert len(adapter.calls) == 2
    assert "Mandatory epistemic rule:" not in adapter.calls[1].prompt
    assert "Reason through the evidence in four explicit stages." in adapter.calls[1].prompt


def test_conditional_runner_needs_no_scene_json(tmp_path: Path) -> None:
    copied = _one_case_benchmark(tmp_path, "scene_0000")
    (copied / "scene_0000" / "scene.json").unlink()
    output = tmp_path / "conditional.jsonl"
    adapter = GateDummy("absent")

    result = run_evaluation(
        benchmark_dir=copied,
        condition="CONFLICT_CONDITIONAL",
        adapter=adapter,
        output_path=output,
    )

    assert result.total_records == 1
    assert result.correct == 1
    record = json.loads(output.read_text().strip())
    assert record["prediction"] == "unknown"
    assert record["condition"] == "CONFLICT_CONDITIONAL"
    assert record["image_path"] is None

from detectivelab.evaluation.conditional import (
    extracted_target_presence,
    run_extractor_gated_conflict,
)


def test_extractor_gated_policy_is_registered_everywhere() -> None:
    assert "CONFLICT_EXTRACTOR_GATED" in VALID_CONDITIONS
    assert "extractor_gated" in ROBUSTNESS_POLICIES
    assert "extractor_gated" in CASE_VARIATION_POLICIES


def test_extractor_presence_finds_present_and_absent_targets() -> None:
    present, label = extracted_target_presence(
        image_path=BENCHMARK / "scene_0005" / "scene.png",
        payload=_payload("scene_0005"),
    )
    assert present == "present"
    assert label == "blue notebook"

    absent, label = extracted_target_presence(
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=_payload("scene_0000"),
    )
    assert absent == "absent"
    assert label == "amber glass"


def test_extractor_absent_gate_uses_zero_model_calls() -> None:
    adapter = GateDummy("present")
    result = run_extractor_gated_conflict(
        adapter=adapter,
        item_id="scene_0000__conflict",
        image_path=BENCHMARK / "scene_0000" / "scene.png",
        payload=_payload("scene_0000"),
    )
    assert result.prediction == "unknown"
    assert result.model_calls == 0
    assert result.gated is True
    assert adapter.calls == []


def test_extractor_present_gate_uses_one_unchanged_staged_call() -> None:
    adapter = GateDummy("absent", staged_verdict="contradicted")
    result = run_extractor_gated_conflict(
        adapter=adapter,
        item_id="scene_0005__conflict",
        image_path=BENCHMARK / "scene_0005" / "scene.png",
        payload=_payload("scene_0005"),
    )
    assert result.prediction == "contradicted"
    assert result.model_calls == 1
    assert result.gated is False
    assert len(adapter.calls) == 1
    assert "Determine only whether" not in adapter.calls[0].prompt
    assert "Mandatory epistemic rule:" not in adapter.calls[0].prompt
    assert "Reason through the evidence in four explicit stages." in adapter.calls[0].prompt
