from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from detectivelab.adapters.base import AdapterRequest
from detectivelab.evaluation.runner import run_evaluation
from detectivelab.evaluation.structured import build_structured_evidence


def _benchmark_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "benchmark_v0_0_1"


@dataclass
class CaptureAdapter:
    requests: list[AdapterRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "capture"

    def predict(self, request: AdapterRequest) -> str:
        self.requests.append(request)
        if request.answer_type == "evidence_verdict":
            return "unknown"
        return "no"


def test_structured_evidence_contains_observable_state_and_relations() -> None:
    scene = json.loads((_benchmark_root() / "scene_0002" / "scene.json").read_text())
    evidence = build_structured_evidence(scene)

    assert "Visible scene evidence:" in evidence
    assert "- blue window: closed" in evidence
    assert "- white notebook: open" in evidence
    assert "- black envelope: present" in evidence
    assert "Spatial relations:" in evidence
    assert "- white glass is right of black envelope" in evidence


def test_structured_evidence_excludes_hidden_internal_fields() -> None:
    scene = json.loads((_benchmark_root() / "scene_0002" / "scene.json").read_text())
    evidence = build_structured_evidence(scene)

    forbidden = [
        "window_1",
        "object_id",
        "subject_id",
        "seed",
        "0.3162",
        '"x"',
        '"y"',
        "verified",
    ]
    for token in forbidden:
        assert token not in evidence


def test_oracle_structured_run_uses_text_only_structured_facts(tmp_path: Path) -> None:
    adapter = CaptureAdapter()
    output = tmp_path / "oracle.jsonl"

    result = run_evaluation(
        benchmark_dir=_benchmark_root(),
        condition="ORACLE_STRUCTURED",
        adapter=adapter,
        output_path=output,
    )

    assert result.total_records == 30
    assert len(adapter.requests) == 30
    assert all(request.image_path is None for request in adapter.requests)
    assert all("Visible scene evidence:" in request.prompt for request in adapter.requests)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(record["condition"] == "ORACLE_STRUCTURED" for record in records)
    assert all(record["image_path"] is None for record in records)


def test_oracle_unknown_case_does_not_invent_absent_object() -> None:
    scene = json.loads((_benchmark_root() / "scene_0000" / "scene.json").read_text())
    evidence = build_structured_evidence(scene)

    assert "red glass: broken" in evidence
    assert "amber clock: present" in evidence
    assert "amber glass" not in evidence
