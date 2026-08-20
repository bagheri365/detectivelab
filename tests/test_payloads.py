from detectivelab.benchmark.payloads import build_payload
from detectivelab.domain.schema import CaseFamily
from detectivelab.generation.questions import generate_questions
from detectivelab.generation.scenes import generate_scene


def _by_family(seed=0):
    scene = generate_scene(seed)
    return scene, {item.family: item for item in generate_questions(scene)}


def test_spatial_payload_exposes_no_text_evidence():
    scene, items = _by_family()
    payload = build_payload(scene, items[CaseFamily.SPATIAL])
    assert payload["context"] == []


def test_state_payload_does_not_leak_witness_alternative():
    scene, items = _by_family()
    payload = build_payload(scene, items[CaseFamily.STATE])
    assert payload["context"] == []
    assert scene.witness_statements[0].claim not in str(payload)


def test_conflict_payload_contains_testimony_and_rule():
    scene, items = _by_family()
    payload = build_payload(scene, items[CaseFamily.CONFLICT])
    assert [entry["type"] for entry in payload["context"]] == ["witness_testimony", "case_rule"]
    assert scene.witness_statements[0].claim in str(payload)
    assert scene.rules[0].description in str(payload)


def test_conflict_rule_text_is_constant_across_frozen_slice():
    rule_texts = set()
    for seed in range(10):
        scene, items = _by_family(seed)
        payload = build_payload(scene, items[CaseFamily.CONFLICT])
        rule_texts.add(payload["context"][1]["text"])
    assert len(rule_texts) == 1
