"""Deterministic question generation from canonical hidden scene state.

Questions are derived mechanically from a :class:`Scene`. No model, prompt,
renderer output, or hand-authored answer key participates in labeling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from detectivelab.domain.schema import CaseFamily, EvidencePriority, Scene


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One closed-form benchmark question and its canonical answer."""

    item_id: str
    scene_id: str
    family: CaseFamily
    question: str
    answer: str
    answer_type: str
    evidence_ids: tuple[str, ...]
    rationale: str
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.item_id or not self.scene_id:
            raise ValueError("item_id and scene_id must be non-empty")
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("question and answer must be non-empty")
        if not self.answer_type:
            raise ValueError("answer_type must be non-empty")
        if not self.evidence_ids:
            raise ValueError("each benchmark item must name canonical evidence")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["family"] = self.family.value
        data["evidence_ids"] = list(self.evidence_ids)
        return data


def _label(scene: Scene, object_id: str) -> str:
    obj = scene.object_by_id(object_id)
    return f"{obj.color} {obj.kind.value}" if obj.color else obj.kind.value


def _generate_spatial(scene: Scene) -> BenchmarkItem:
    if not scene.relations:
        raise ValueError("spatial question generation requires a scene relation")

    relation = scene.relations[0]
    subject = _label(scene, relation.subject_id)
    reference = _label(scene, relation.object_id)

    # Deterministically balance yes/no labels. On odd seeds ask the inverse of
    # the canonical left/right relation, producing a mechanically known "no".
    inverse = {"left_of": "right_of", "right_of": "left_of"}
    canonical = relation.relation.value
    asked = canonical if scene.seed % 2 == 0 else inverse.get(canonical, canonical)
    answer = "yes" if asked == canonical else "no"
    wording = asked.replace("_", " ")
    return BenchmarkItem(
        item_id=f"{scene.scene_id}__spatial",
        scene_id=scene.scene_id,
        family=CaseFamily.SPATIAL,
        question=f"Is the {subject} {wording} the {reference}?",
        answer=answer,
        answer_type="yes_no",
        evidence_ids=(relation.subject_id, relation.object_id),
        rationale=(
            f"The canonical scene relation records {relation.subject_id} "
            f"as {relation.relation.value} {relation.object_id}."
        ),
    )


def _generate_state(scene: Scene) -> BenchmarkItem:
    """Generate a balanced yes/no question about an observable object state.

    v0.0.1 intentionally decouples STATE from witness testimony so conflict
    construction cannot leak or constrain the state-family label.
    """

    state_objects = [obj for obj in scene.objects if obj.state is not None]
    if not state_objects:
        raise ValueError("state question generation requires a state-bearing object")
    target = state_objects[0]

    alternatives = {
        "open": "closed",
        "closed": "open",
        "intact": "broken",
        "broken": "intact",
    }
    if target.state not in alternatives:
        raise ValueError(f"unsupported observable state: {target.state}")

    ask_true_state = ((scene.seed ^ (scene.seed >> 1)) & 1) == 0
    asked_state = str(target.state if ask_true_state else alternatives[target.state])
    answer = "yes" if ask_true_state else "no"

    return BenchmarkItem(
        item_id=f"{scene.scene_id}__state",
        scene_id=scene.scene_id,
        family=CaseFamily.STATE,
        question=f"Is the {_label(scene, target.object_id)} currently {asked_state}?",
        answer=answer,
        answer_type="yes_no",
        evidence_ids=(target.object_id,),
        rationale=(
            f"The hidden scene state records {target.object_id}.state={target.state}; "
            f"the question asks whether it is {asked_state}."
        ),
    )


def _generate_conflict(scene: Scene) -> BenchmarkItem:
    if not scene.witness_statements:
        raise ValueError("conflict question generation requires witness testimony")
    if not scene.rules:
        raise ValueError("conflict question generation requires an evidence rule")

    statement = scene.witness_statements[0]
    rule = scene.rules[0]
    if statement.predicate != "state":
        raise ValueError("v0.0.1 conflict questions require state testimony")
    if rule.evidence_priority != EvidencePriority.PHYSICAL_OVER_TESTIMONY:
        raise ValueError("v0.0.1 uses one constant physical-over-testimony rule")

    evidence_ids = [statement.statement_id, rule.rule_id]
    if statement.subject_id is None:
        answer = "unknown"
        rationale = (
            "The testimony names an object-state claim that the current scene does not "
            "contain, so the physical evidence cannot resolve the claim."
        )
    else:
        target = scene.object_by_id(statement.subject_id)
        if target.state is None:
            raise ValueError("conflict testimony must target a state-bearing object")
        evidence_ids.append(target.object_id)
        if statement.value == target.state:
            answer = "supported"
            rationale = (
                f"The witness claim state={statement.value} matches physical state={target.state}."
            )
        else:
            answer = "contradicted"
            rationale = (
                f"The witness claims state={statement.value}, while physical evidence "
                f"records state={target.state}; the constant case rule prioritizes physical evidence."
            )

    return BenchmarkItem(
        item_id=f"{scene.scene_id}__conflict",
        scene_id=scene.scene_id,
        family=CaseFamily.CONFLICT,
        question=(
            "Given the current physical scene, witness testimony, and case rule, "
            "is the testimony supported, contradicted, or unknown?"
        ),
        answer=answer,
        answer_type="evidence_verdict",
        evidence_ids=tuple(evidence_ids),
        rationale=rationale,
    )


def generate_questions(scene: Scene) -> tuple[BenchmarkItem, BenchmarkItem, BenchmarkItem]:
    """Return exactly one spatial, one state, and one conflict item for ``scene``."""

    items = (
        _generate_spatial(scene),
        _generate_state(scene),
        _generate_conflict(scene),
    )
    if {item.family for item in items} != set(CaseFamily):
        raise AssertionError("question generator must emit exactly one item per case family")
    return items
