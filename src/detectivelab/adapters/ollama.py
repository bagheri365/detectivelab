from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import AdapterRequest


@dataclass(frozen=True)
class OllamaAdapter:
    """Dependency-free adapter for a local Ollama server.

    The adapter uses Ollama's /api/generate endpoint with streaming disabled.
    v0.1-direct intentionally supports text-only QUESTION evaluation first;
    RAW image support is added only when the visual condition is promoted.
    """

    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_predict: int = 8
    seed: int = 0
    timeout_s: float = 120.0

    @property
    def name(self) -> str:
        # Include the concrete model in the resume/provenance key.
        return f"ollama:{self.model}"

    def predict(self, request: AdapterRequest) -> str:
        if request.image_path is not None:
            raise ValueError(
                "OllamaAdapter v0.1 supports QUESTION only; RAW image input "
                "has not been promoted yet."
            )

        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            ) from exc

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc

        if "response" not in result:
            raise RuntimeError(f"Ollama response missing 'response': {result}")
        return str(result["response"]).strip()
