from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    scene_id: str
    item_id: str
    family: str
    condition: str
    model: str
    prompt: str
    image_path: str | None
    raw_output: str
    prediction: str
    gold: str
    correct: bool
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
