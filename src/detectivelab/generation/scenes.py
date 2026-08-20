"""Deterministic synthetic scene generator.

The generator intentionally produces simple, renderer-friendly hidden states.
It does not attempt to create photorealistic worlds. Its job is to provide a
stable source of truth for later rendering, question generation, and audits.
"""

from __future__ import annotations

import random

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

_COLORS = ("blue", "red", "green", "amber", "black", "white")
_STATES: dict[ObjectKind, tuple[str, ...]] = {
    ObjectKind.WINDOW: ("open", "closed"),
    ObjectKind.DOOR: ("open", "closed"),
    ObjectKind.NOTEBOOK: ("open", "closed"),
    ObjectKind.GLASS: ("intact", "broken"),
}
_CONFLICT_KINDS = tuple(_STATES)
_CONFLICT_RULE = "Current physical evidence overrides unverified witness testimony."


def _object_size(kind: ObjectKind) -> Size:
    sizes = {
        ObjectKind.KEY: Size(0.08, 0.04),
        ObjectKind.NOTEBOOK: Size(0.14, 0.10),
        ObjectKind.GLASS: Size(0.07, 0.11),
        ObjectKind.LAMP: Size(0.12, 0.20),
        ObjectKind.CLOCK: Size(0.11, 0.11),
        ObjectKind.BRIEFCASE: Size(0.17, 0.11),
        ObjectKind.PAINTING: Size(0.18, 0.16),
        ObjectKind.CHAIR: Size(0.16, 0.20),
        ObjectKind.DOOR: Size(0.17, 0.34),
        ObjectKind.WINDOW: Size(0.20, 0.22),
        ObjectKind.FOOTPRINT: Size(0.08, 0.12),
        ObjectKind.ENVELOPE: Size(0.12, 0.07),
    }
    return sizes[kind]


def _sample_position(rng: random.Random, size: Size) -> Point:
    half_w = size.width / 2
    half_h = size.height / 2
    return Point(
        round(rng.uniform(half_w + 0.03, 1 - half_w - 0.03), 4),
        round(rng.uniform(half_h + 0.03, 1 - half_h - 0.03), 4),
    )


def _overlaps(candidate: Point, size: Size, objects: list[SceneObject], *, gap: float = 0.025) -> bool:
    for other in objects:
        overlap_x = abs(candidate.x - other.position.x) < (size.width + other.size.width) / 2 + gap
        overlap_y = abs(candidate.y - other.position.y) < (size.height + other.size.height) / 2 + gap
        if overlap_x and overlap_y:
            return True
    return False


def _sample_non_overlapping_position(
    rng: random.Random, size: Size, objects: list[SceneObject]
) -> Point:
    for _ in range(500):
        candidate = _sample_position(rng, size)
        if not _overlaps(candidate, size, objects):
            return candidate
    raise RuntimeError("could not place object without overlap")


def _alternate_state(kind: ObjectKind, current: str) -> str:
    for state in _STATES[kind]:
        if state != current:
            return state
    raise AssertionError("state vocabulary must contain an alternative")


def _conflict_mode(seed: int) -> str:
    """Deterministic 10-scene balance: 4 unknown / 3 supported / 3 contradicted."""

    slot = seed % 10
    if slot in {0, 3, 6, 9}:
        return "unknown"
    if slot in {1, 4, 7}:
        return "supported"
    return "contradicted"


def _absent_claim(rng: random.Random, objects: list[SceneObject]) -> tuple[str, str, str]:
    """Return a plausible state claim for a color+kind pair absent from the scene."""

    visual_labels = {(obj.color, obj.kind) for obj in objects}
    candidates = [
        (color, kind)
        for kind in _CONFLICT_KINDS
        for color in _COLORS
        if (color, kind) not in visual_labels
    ]
    color, kind = rng.choice(candidates)
    state = rng.choice(_STATES[kind])
    return color, kind.value, state


def generate_scene(seed: int, *, scene_id: str | None = None) -> Scene:
    """Generate one deterministic, schema-valid scene from ``seed``.

    v0.0.1 conflict guardrail:
    - every case uses the exact same evidence-priority rule;
    - testimony can be supported, contradicted, or unresolved by the image;
    - the verdict is therefore not encoded by rule wording.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = random.Random(seed)
    resolved_scene_id = scene_id or f"scene_{seed:04d}"

    anchor_kind = rng.choice(_CONFLICT_KINDS)
    remaining_kinds = [kind for kind in ObjectKind if kind != anchor_kind]
    chosen_kinds = [anchor_kind, *rng.sample(remaining_kinds, k=5)]

    objects: list[SceneObject] = []
    for index, kind in enumerate(chosen_kinds, start=1):
        size = _object_size(kind)
        state = rng.choice(_STATES[kind]) if kind in _STATES else None
        objects.append(
            SceneObject(
                object_id=f"{kind.value}_{index}",
                kind=kind,
                position=_sample_non_overlapping_position(rng, size, objects),
                size=size,
                color=rng.choice(_COLORS),
                state=state,
                visibility=Visibility.VISIBLE,
                orientation_deg=0,
            )
        )

    subject, reference = rng.sample(objects, 2)
    relation_kind = (
        RelationKind.LEFT_OF if subject.position.x <= reference.position.x else RelationKind.RIGHT_OF
    )
    relation = SpatialRelation(subject.object_id, relation_kind, reference.object_id)

    anchor = objects[0]
    assert anchor.state is not None
    mode = _conflict_mode(seed)
    if mode == "supported":
        claimed_state = str(anchor.state)
        statement = WitnessStatement(
            statement_id="stmt_1",
            witness_id="witness_a",
            claim=(
                f"The witness says the {anchor.color} {anchor.kind.value} "
                f"is currently {claimed_state}."
            ),
            subject_id=anchor.object_id,
            predicate="state",
            value=claimed_state,
            verified=False,
        )
    elif mode == "contradicted":
        claimed_state = _alternate_state(anchor.kind, str(anchor.state))
        statement = WitnessStatement(
            statement_id="stmt_1",
            witness_id="witness_a",
            claim=(
                f"The witness says the {anchor.color} {anchor.kind.value} "
                f"is currently {claimed_state}."
            ),
            subject_id=anchor.object_id,
            predicate="state",
            value=claimed_state,
            verified=False,
        )
    else:
        color, kind, claimed_state = _absent_claim(rng, objects)
        statement = WitnessStatement(
            statement_id="stmt_1",
            witness_id="witness_a",
            claim=f"The witness says the {color} {kind} is currently {claimed_state}.",
            subject_id=None,
            predicate="state",
            value=claimed_state,
            verified=False,
        )

    rule = CaseRule(
        rule_id="rule_1",
        description=_CONFLICT_RULE,
        evidence_priority=EvidencePriority.PHYSICAL_OVER_TESTIMONY,
    )

    return Scene(
        scene_id=resolved_scene_id,
        seed=seed,
        objects=tuple(objects),
        relations=(relation,),
        witness_statements=(statement,),
        rules=(rule,),
    )
