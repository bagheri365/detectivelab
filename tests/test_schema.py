import pytest

from detectivelab.domain.schema import (
    CaseRule,
    EvidencePriority,
    ObjectKind,
    Point,
    RelationKind,
    Scene,
    SceneObject,
    Size,
    SpatialRelation,
    Visibility,
    WitnessStatement,
)


def make_objects():
    return (
        SceneObject("key_1", ObjectKind.KEY, Point(0.25, 0.70), Size(0.08, 0.04), color="blue"),
        SceneObject("lamp_1", ObjectKind.LAMP, Point(0.30, 0.45), Size(0.18, 0.28), state="on"),
        SceneObject("window_1", ObjectKind.WINDOW, Point(0.75, 0.25), Size(0.25, 0.30), state="closed"),
        SceneObject("desk_1", ObjectKind.NOTEBOOK, Point(0.50, 0.70), Size(0.30, 0.15), state="open"),
    )


def test_scene_round_trips_to_json_friendly_dict():
    scene = Scene(
        scene_id="scene_0001",
        seed=7,
        objects=make_objects(),
        relations=(SpatialRelation("key_1", RelationKind.UNDER, "lamp_1"),),
        witness_statements=(
            WitnessStatement(
                "stmt_1",
                "witness_a",
                "The blue key was on the shelf.",
                subject_id="key_1",
                predicate="location",
                value="shelf",
            ),
        ),
        rules=(
            CaseRule(
                "rule_1",
                "Current physical evidence overrides unverified recollection.",
                EvidencePriority.PHYSICAL_OVER_TESTIMONY,
            ),
        ),
    )

    data = scene.to_dict()
    assert data["scene_id"] == "scene_0001"
    assert data["objects"][0]["kind"] == "key"
    assert data["relations"][0]["relation"] == "under"
    assert data["rules"][0]["evidence_priority"] == "physical_over_testimony"


def test_scene_rejects_duplicate_object_ids():
    objects = list(make_objects())
    objects[-1] = SceneObject("key_1", ObjectKind.CHAIR, Point(0.8, 0.8), Size(0.1, 0.1))
    with pytest.raises(ValueError, match="unique"):
        Scene("bad", 1, tuple(objects))


def test_scene_rejects_relation_to_missing_object():
    with pytest.raises(ValueError, match="relations"):
        Scene(
            "bad",
            1,
            make_objects(),
            relations=(SpatialRelation("key_1", RelationKind.UNDER, "missing"),),
        )


def test_point_bounds_are_enforced():
    with pytest.raises(ValueError, match="normalized"):
        Point(1.2, 0.5)


def test_object_count_guardrail_is_enforced():
    with pytest.raises(ValueError, match="4-8"):
        Scene("too_small", 1, make_objects()[:3])
