"""Participant-facing benchmark payload construction.

This module is intentionally the only place that decides which evidence each
case family may see. It prevents cross-family leakage during evaluation.
"""
from __future__ import annotations

from typing import Any

from detectivelab.domain.schema import CaseFamily, Scene
from detectivelab.generation.questions import BenchmarkItem


def build_payload(scene: Scene, item: BenchmarkItem) -> dict[str, Any]:
    """Build the exact participant-facing input for one benchmark item.

    RAW evaluators attach ``scene.png`` separately. This JSON payload contains
    only the text evidence allowed for the item's family.
    """
    payload: dict[str, Any] = {
        "item_id": item.item_id,
        "scene_id": item.scene_id,
        "family": item.family.value,
        "question": item.question,
        "answer_type": item.answer_type,
    }

    if item.family == CaseFamily.SPATIAL:
        # The image and question are sufficient. No testimony/rules are exposed.
        payload["context"] = []
    elif item.family == CaseFamily.STATE:
        # Deliberately exclude witness testimony, which contains the alternate
        # state and would create a shortcut.
        payload["context"] = []
    elif item.family == CaseFamily.CONFLICT:
        if not scene.witness_statements or not scene.rules:
            raise ValueError("conflict payload requires testimony and a case rule")
        payload["context"] = [
            {"type": "witness_testimony", "text": scene.witness_statements[0].claim},
            {"type": "case_rule", "text": scene.rules[0].description},
        ]
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"unsupported family: {item.family}")

    return payload
