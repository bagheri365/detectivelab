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

    v0.0 deliberately uses the conflict anchor as the state target because its
    witness statement already contains a mechanically valid alternative state.
    The evaluator must *not* expose witness testimony for STATE-family items.
    """

    if not scene.witness_statements:
        raise ValueError("state question generation requires the conflict anchor")
    statement = scene.witness_statements[0]
    if statement.subject_id is None or statement.predicate != "state":
        raise ValueError("v0.0 state questions require state testimony about an object")

    target = scene.object_by_id(statement.subject_id)
    if target.state is None:
        raise ValueError("state question target must carry an observable state")
    if statement.value == target.state:
        raise ValueError("state question requires a distinct alternative state")

    # Gray-code parity gives a deterministic 5/5 yes/no split for seeds 0..9
    # without simply mirroring the spatial seed-parity pattern.
    ask_true_state = ((scene.seed ^ (scene.seed >> 1)) & 1) == 0
    asked_state = str(target.state if ask_true_state else statement.value)
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
    if statement.subject_id is None or statement.predicate != "state":
        raise ValueError("v0.0 conflict questions require state testimony about an object")

    target = scene.object_by_id(statement.subject_id)
    if target.state is None:
        raise ValueError("conflict testimony must target a state-bearing object")

    physical_conflict = statement.value != target.state
    physical_priority = any(
        rule.evidence_priority == EvidencePriority.PHYSICAL_OVER_TESTIMONY
        for rule in scene.rules
    )

    if physical_conflict and physical_priority:
        answer = "contradicted"
        rationale = (
            f"The witness claims state={statement.value}, while physical evidence "
            f"records state={target.state}; the case rule prioritizes physical evidence."
        )
    elif not physical_conflict:
        answer = "supported"
        rationale = (
            f"The witness claim state={statement.value} matches physical state={target.state}."
        )
    else:
        answer = "unknown"
        rationale = "The sources disagree and no applicable evidence-priority rule resolves them."

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
        evidence_ids=(statement.statement_id, statement.subject_id, *[r.rule_id for r in scene.rules]),
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
