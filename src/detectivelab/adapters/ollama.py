from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import AdapterRequest


@dataclass(frozen=True)
class OllamaAdapter:
    """Dependency-free adapter for a local Ollama server.

    The adapter uses Ollama's /api/generate endpoint with streaming disabled.
    QUESTION requests are text-only. RAW requests attach the rendered scene
    image using Ollama's base64 ``images`` field while keeping the prompt and
    decoding settings unchanged.
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

        if request.image_path is not None:
            image_path = request.image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing RAW image: {image_path}")
            payload["images"] = [
                base64.b64encode(image_path.read_bytes()).decode("ascii")
            ]

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
