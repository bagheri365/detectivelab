from __future__ import annotations

import re
from pathlib import Path

from detectivelab.extraction import extract_scene_facts
from detectivelab.extraction.base import ExtractedObject

_SPATIAL_RE = re.compile(
    r"^Is the (?P<a>.+?) (?P<relation>left|right) of the (?P<b>.+?)\?$",
    re.IGNORECASE,
)
_STATE_RE = re.compile(
    r"^Is the (?P<label>.+?) currently (?P<state>[a-z]+)\?$",
    re.IGNORECASE,
)
_WITNESS_RE = re.compile(
    r"^The witness says the (?P<label>.+?) is currently (?P<state>[a-z]+)\.$",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _match_object(objects: list[ExtractedObject], label: str) -> ExtractedObject | None:
    wanted = _norm(label)
    matches = [obj for obj in objects if _norm(obj.label) == wanted]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous extracted label: {label!r}")
    return matches[0] if matches else None


def _context_text(payload: dict, entry_type: str) -> str | None:
    for entry in payload.get("context", []):
        if entry.get("type") == entry_type:
            return str(entry.get("text", ""))
    return None


def build_focused_extracted_evidence(*, image_path: Path, payload: dict) -> str:
    """Build minimal task-relevant evidence from image-derived facts only.

    The participant-facing payload is used only to identify the queried entity
    or entities. Visual facts come exclusively from ``scene.png`` through the
    existing deterministic extractor. Hidden scene JSON, object IDs, seeds,
    provenance, and gold answers are not consulted.
    """

    objects = extract_scene_facts(image_path)
    family = payload["family"]

    if family == "spatial":
        match = _SPATIAL_RE.match(payload["question"])
        if match is None:
            raise ValueError(f"Unsupported spatial question: {payload['question']!r}")

        first_label = _norm(match.group("a"))
        second_label = _norm(match.group("b"))
        first = _match_object(objects, first_label)
        second = _match_object(objects, second_label)

        lines = ["Focused visible scene evidence:"]
        if first is None:
            lines.append(f"- {first_label}: not present")
        else:
            lines.append(f"- {first.label}: present")
        if second is None:
            lines.append(f"- {second_label}: not present")
        else:
            lines.append(f"- {second.label}: present")

        if first is not None and second is not None:
            if first.center_x < second.center_x:
                relation = "left of"
            elif first.center_x > second.center_x:
                relation = "right of"
            else:
                relation = "horizontally aligned with"
            lines.append("Relevant spatial relation:")
            lines.append(f"- {first.label} is {relation} {second.label}")
        return "\n".join(lines)

    if family == "state":
        match = _STATE_RE.match(payload["question"])
        if match is None:
            raise ValueError(f"Unsupported state question: {payload['question']!r}")
        label = _norm(match.group("label"))
        obj = _match_object(objects, label)
        lines = ["Focused visible scene evidence:"]
        if obj is None:
            lines.append(f"- {label}: not present")
        else:
            value = obj.state if obj.state is not None else "present"
            lines.append(f"- {obj.label}: {value}")
        return "\n".join(lines)

    if family == "conflict":
        witness = _context_text(payload, "witness_testimony")
        if witness is None:
            raise ValueError("Conflict payload is missing witness testimony")
        match = _WITNESS_RE.match(witness)
        if match is None:
            raise ValueError(f"Unsupported witness statement: {witness!r}")
        label = _norm(match.group("label"))
        obj = _match_object(objects, label)
        lines = ["Focused visible scene evidence:"]
        if obj is None:
            lines.append(f"- {label}: not present")
        else:
            value = obj.state if obj.state is not None else "present"
            lines.append(f"- {obj.label}: {value}")
        return "\n".join(lines)

    raise ValueError(f"Unsupported family for focused evidence: {family}")
