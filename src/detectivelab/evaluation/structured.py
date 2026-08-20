from __future__ import annotations


def _object_label(obj: dict) -> str:
    return f"{obj['color']} {obj['kind']}"


def build_structured_evidence(scene: dict) -> str:
    """Render participant-safe oracle facts from hidden scene state.

    Only facts that are visually observable in the rendered scene are exposed.
    Internal identifiers, coordinates, seeds, provenance, and gold labels are
    intentionally excluded.
    """

    visible_objects = [
        obj for obj in scene.get("objects", []) if obj.get("visibility") == "visible"
    ]
    label_by_id = {obj["object_id"]: _object_label(obj) for obj in visible_objects}

    lines = ["Visible scene evidence:"]
    for obj in visible_objects:
        label = _object_label(obj)
        state = obj.get("state")
        if state is None:
            lines.append(f"- {label}: present")
        else:
            lines.append(f"- {label}: {state}")

    relation_lines: list[str] = []
    for relation in scene.get("relations", []):
        subject = label_by_id.get(relation.get("subject_id"))
        obj = label_by_id.get(relation.get("object_id"))
        if subject is None or obj is None:
            continue

        relation_name = str(relation.get("relation", "")).replace("_", " ")
        relation_lines.append(f"- {subject} is {relation_name} {obj}")

    if relation_lines:
        lines.append("Spatial relations:")
        lines.extend(relation_lines)

    return "\n".join(lines)
