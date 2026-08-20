from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedObject:
    kind: str
    color: str
    state: str | None
    center_x: float
    center_y: float

    @property
    def label(self) -> str:
        return f"{self.color} {self.kind}"
