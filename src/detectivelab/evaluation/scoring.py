from __future__ import annotations

import re


_ALLOWED = {
    "yes_no": ("yes", "no"),
    "evidence_verdict": ("supported", "contradicted", "unknown"),
}


def normalize_prediction(raw_output: str, answer_type: str) -> str:
    """Map a model response to one closed-form benchmark label.

    Exact labels are preferred. For future natural-language model adapters, the
    first standalone allowed label found in the response is accepted. If no
    valid label is present, ``invalid`` is returned rather than guessing.
    """

    allowed = _ALLOWED.get(answer_type)
    if allowed is None:
        raise ValueError(f"Unsupported answer_type: {answer_type}")

    text = raw_output.strip().lower()
    if text in allowed:
        return text

    for label in allowed:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label

    return "invalid"


def is_correct(prediction: str, gold: str) -> bool:
    return prediction == gold.strip().lower()
