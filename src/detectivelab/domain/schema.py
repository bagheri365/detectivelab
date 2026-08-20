"""Canonical hidden-state schema for DetectiveLab scenes.

This module deliberately contains no rendering or model code. It defines the
smallest stable contract shared by generation, rendering, question creation,
and evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ObjectKind(StrEnum):
    KEY = "key"
    NOTEBOOK = "notebook"
    GLASS = "glass"
    LAMP = "lamp"
    CLOCK = "clock"
    BRIEFCASE = "briefcase"
    PAINTING = "painting"
    CHAIR = "chair"
    DOOR = "door"
    WINDOW = "window"
    FOOTPRINT = "footprint"
    ENVELOPE = "envelope"


class Visibility(StrEnum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    HIDDEN = "hidden"


class RelationKind(StrEnum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    UNDER = "under"
    INSIDE = "inside"
    BEHIND = "behind"
    IN_FRONT_OF = "in_front_of"
    NEAR = "near"
    FAR = "far"
    PARTIALLY_OCCLUDED_BY = "partially_occluded_by"


class CaseFamily(StrEnum):
    SPATIAL = "spatial"
    STATE = "state"
    CONFLICT = "conflict"


class EvidencePriority(StrEnum):
    PHYSICAL_OVER_TESTIMONY = "physical_over_testimony"
    TESTIMONY_OVER_PHYSICAL = "testimony_over_physical"
    NO_PRIORITY = "no_priority"


@dataclass(frozen=True, slots=True)
class Point:
    """Normalized 2D point in renderer coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("Point coordinates must be normalized to [0, 1].")


@dataclass(frozen=True, slots=True)
class Size:
    """Normalized object footprint."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if not (0.0 < self.width <= 1.0 and 0.0 < self.height <= 1.0):
            raise ValueError("Size dimensions must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class SceneObject:
    """One canonical object in the hidden world state."""

    object_id: str
    kind: ObjectKind
    position: Point
    size: Size
    color: str | None = None
    state: str | None = None
    visibility: Visibility = Visibility.VISIBLE
    orientation_deg: int = 0
    container_id: str | None = None
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id or any(ch.isspace() for ch in self.object_id):
            raise ValueError("object_id must be non-empty and contain no whitespace.")
        if not (0 <= self.orientation_deg < 360):
            raise ValueError("orientation_deg must be in [0, 360).")
        if self.container_id == self.object_id:
            raise ValueError("An object cannot contain itself.")


@dataclass(frozen=True, slots=True)
class SpatialRelation:
    subject_id: str
    relation: RelationKind
    object_id: str

    def __post_init__(self) -> None:
        if self.subject_id == self.object_id:
            raise ValueError("A spatial relation must connect two different objects.")


@dataclass(frozen=True, slots=True)
class WitnessStatement:
    statement_id: str
    witness_id: str
    claim: str
    subject_id: str | None = None
    predicate: str | None = None
    value: str | int | float | bool | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.statement_id or not self.witness_id or not self.claim.strip():
            raise ValueError("Witness statements require IDs and non-empty claim text.")


@dataclass(frozen=True, slots=True)
class CaseRule:
    rule_id: str
    description: str
    evidence_priority: EvidencePriority = EvidencePriority.NO_PRIORITY

    def __post_init__(self) -> None:
        if not self.rule_id or not self.description.strip():
            raise ValueError("Case rules require an ID and description.")


@dataclass(frozen=True, slots=True)
class Scene:
    """Complete hidden state for one deterministic DetectiveLab scene."""

    scene_id: str
    seed: int
    objects: tuple[SceneObject, ...]
    relations: tuple[SpatialRelation, ...] = ()
    witness_statements: tuple[WitnessStatement, ...] = ()
    rules: tuple[CaseRule, ...] = ()
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty.")
        if not (4 <= len(self.objects) <= 8):
            raise ValueError("Primary benchmark scenes must contain 4-8 objects.")

        object_ids = [obj.object_id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("Scene object IDs must be unique.")
        object_id_set = set(object_ids)

        statement_ids = [s.statement_id for s in self.witness_statements]
        if len(statement_ids) != len(set(statement_ids)):
            raise ValueError("Witness statement IDs must be unique within a scene.")

        rule_ids = [r.rule_id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Rule IDs must be unique within a scene.")

        for obj in self.objects:
            if obj.container_id is not None and obj.container_id not in object_id_set:
                raise ValueError(
                    f"Object {obj.object_id!r} references missing container "
                    f"{obj.container_id!r}."
                )

        for rel in self.relations:
            if rel.subject_id not in object_id_set or rel.object_id not in object_id_set:
                raise ValueError("All spatial relations must reference scene objects.")

        for stmt in self.witness_statements:
            if stmt.subject_id is not None and stmt.subject_id not in object_id_set:
                raise ValueError(
                    f"Witness statement {stmt.statement_id!r} references missing "
                    f"object {stmt.subject_id!r}."
                )

    def object_by_id(self, object_id: str) -> SceneObject:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(object_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable enum values."""

        def normalize(value: Any) -> Any:
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(v) for v in value]
            return value

        return normalize(asdict(self))
