from detectivelab.domain.schema import CaseFamily, EvidencePriority
from detectivelab.generation.questions import generate_questions
from detectivelab.generation.scenes import generate_scene


def test_generates_exactly_one_item_per_family() -> None:
    items = generate_questions(generate_scene(11))
    assert len(items) == 3
    assert [item.family for item in items] == [
        CaseFamily.SPATIAL,
        CaseFamily.STATE,
        CaseFamily.CONFLICT,
    ]


def test_question_generation_is_deterministic() -> None:
    first = [item.to_dict() for item in generate_questions(generate_scene(42))]
    second = [item.to_dict() for item in generate_questions(generate_scene(42))]
    assert first == second


def test_spatial_answer_is_derived_from_scene_relation() -> None:
    even_scene = generate_scene(4)
    odd_scene = generate_scene(5)
    even_spatial = generate_questions(even_scene)[0]
    odd_spatial = generate_questions(odd_scene)[0]
    assert even_spatial.answer == "yes"
    assert odd_spatial.answer == "no"
    relation = odd_scene.relations[0]
    assert relation.subject_id in odd_spatial.evidence_ids
    assert relation.object_id in odd_spatial.evidence_ids


def test_state_answer_is_balanced_binary_proposition() -> None:
    answers = [generate_questions(generate_scene(seed))[1].answer for seed in range(10)]
    assert answers.count("yes") == 5
    assert answers.count("no") == 5


def test_conflict_uses_constant_rule_and_three_image_dependent_outcomes() -> None:
    scenes = [generate_scene(seed) for seed in range(10)]
    rule_texts = {scene.rules[0].description for scene in scenes}
    priorities = {scene.rules[0].evidence_priority for scene in scenes}
    answers = [generate_questions(scene)[2].answer for scene in scenes]

    assert len(rule_texts) == 1
    assert priorities == {EvidencePriority.PHYSICAL_OVER_TESTIMONY}
    assert answers.count("supported") == 3
    assert answers.count("contradicted") == 3
    assert answers.count("unknown") == 4

    for scene, answer in zip(scenes, answers):
        statement = scene.witness_statements[0]
        if answer == "unknown":
            assert statement.subject_id is None
        else:
            assert statement.subject_id is not None
            target = scene.object_by_id(statement.subject_id)
            if answer == "supported":
                assert statement.value == target.state
            else:
                assert statement.value != target.state


def test_question_ids_are_scene_scoped_and_unique() -> None:
    scene = generate_scene(13)
    items = generate_questions(scene)
    assert len({item.item_id for item in items}) == 3
    assert all(item.item_id.startswith(f"{scene.scene_id}__") for item in items)


def test_answers_are_closed_form() -> None:
    items = generate_questions(generate_scene(17))
    spatial, state, conflict = items
    assert spatial.answer in {"yes", "no"}
    assert state.answer in {"yes", "no"}
    assert conflict.answer in {"supported", "contradicted", "unknown"}
