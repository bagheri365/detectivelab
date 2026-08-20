import json

import pytest

from detectivelab.domain.schema import EvidencePriority
from detectivelab.generation.scenes import generate_scene


def test_same_seed_produces_identical_scene():
    first = generate_scene(42)
    second = generate_scene(42)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_different_seeds_produce_different_scene_state():
    assert generate_scene(1).to_dict() != generate_scene(2).to_dict()


def test_generated_scene_satisfies_v0_guardrails():
    scene = generate_scene(7)
    assert len(scene.objects) == 6
    assert len(scene.relations) >= 1
    assert len(scene.witness_statements) == 1
    assert scene.rules[0].evidence_priority in {
        EvidencePriority.PHYSICAL_OVER_TESTIMONY,
        EvidencePriority.NO_PRIORITY,
    }


def test_generated_witness_claim_conflicts_with_physical_state():
    scene = generate_scene(15)
    statement = scene.witness_statements[0]
    subject = scene.object_by_id(statement.subject_id)
    assert statement.predicate == "state"
    assert subject.state is not None
    assert statement.value != subject.state


def test_generated_scene_is_json_serializable():
    scene = generate_scene(99)
    encoded = json.dumps(scene.to_dict(), sort_keys=True)
    assert '"scene_id": "scene_0099"' in encoded


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        generate_scene(-1)


def test_generated_objects_do_not_overlap():
    for seed in range(25):
        scene = generate_scene(seed)
        for index, first in enumerate(scene.objects):
            for second in scene.objects[index + 1 :]:
                overlap_x = abs(first.position.x - second.position.x) < (
                    first.size.width + second.size.width
                ) / 2
                overlap_y = abs(first.position.y - second.position.y) < (
                    first.size.height + second.size.height
                ) / 2
                assert not (overlap_x and overlap_y)


def test_orientation_stays_frozen_until_renderer_supports_it():
    for seed in range(10):
        assert all(obj.orientation_deg == 0 for obj in generate_scene(seed).objects)


def test_v0_objects_are_fully_visible() -> None:
    for seed in range(25):
        assert all(obj.visibility.value == "visible" for obj in generate_scene(seed).objects)


def test_clock_is_not_used_as_single_frame_state() -> None:
    for seed in range(25):
        for obj in generate_scene(seed).objects:
            if obj.kind.value == "clock":
                assert obj.state is None
