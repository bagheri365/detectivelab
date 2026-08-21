from __future__ import annotations

from .base import AdapterRequest


class DummyAdapter:
    """Deterministic non-ML adapter used to exercise the evaluation harness.

    It intentionally does not inspect the image. It returns a fixed legal answer
    for each answer type so that end-to-end plumbing can be tested before any
    real model dependency is introduced.
    """

    @property
    def name(self) -> str:
        return "dummy"

    def predict(self, request: AdapterRequest) -> str:
        if request.answer_type == "yes_no":
            return "yes"
        if request.answer_type == "evidence_verdict":
            return "unknown"
        raise ValueError(f"Unsupported answer_type: {request.answer_type}")
