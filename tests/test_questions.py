from detectivelab.domain.schema import CaseFamily
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


def test_conflict_answer_depends_on_evidence_priority() -> None:
    even_scene = generate_scene(8)
    odd_scene = generate_scene(9)
    even_conflict = generate_questions(even_scene)[2]
    odd_conflict = generate_questions(odd_scene)[2]
    for scene in (even_scene, odd_scene):
        statement = scene.witness_statements[0]
        target = scene.object_by_id(statement.subject_id)
        assert statement.value != target.state
    assert even_conflict.answer == "contradicted"
    assert odd_conflict.answer == "unknown"


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
