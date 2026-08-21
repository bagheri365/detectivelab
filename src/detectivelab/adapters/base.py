from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AdapterRequest:
    """Model-facing request produced by the evaluation runner."""

    item_id: str
    family: str
    answer_type: str
    prompt: str
    image_path: Path | None = None


class ModelAdapter(Protocol):
    """Minimal interface shared by local and API-backed model adapters."""

    @property
    def name(self) -> str: ...

    def predict(self, request: AdapterRequest) -> str: ...
