from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from detectivelab.adapters.base import AdapterRequest
from detectivelab.adapters.ollama import OllamaAdapter


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def _request(image_path: Path | None = None) -> AdapterRequest:
    return AdapterRequest(
        item_id="scene_0000_state",
        family="state",
        answer_type="yes_no",
        prompt="Question: Is the window closed?\nAnswer with exactly one label: yes or no.",
        image_path=image_path,
    )


def test_ollama_adapter_posts_deterministic_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"response": "yes", "done": True})

    monkeypatch.setattr("detectivelab.adapters.ollama.urlopen", fake_urlopen)
    adapter = OllamaAdapter(model="qwen3:test", timeout_s=3.5)

    assert adapter.predict(_request()) == "yes"
    assert adapter.name == "ollama:qwen3:test"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 3.5
    assert captured["payload"] == {
        "model": "qwen3:test",
        "prompt": _request().prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 8, "seed": 0},
    }


def test_ollama_adapter_posts_raw_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}
    image_path = tmp_path / "scene.png"
    image_bytes = b"fake-png-bytes"
    image_path.write_bytes(image_bytes)

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"response": "no", "done": True})

    monkeypatch.setattr("detectivelab.adapters.ollama.urlopen", fake_urlopen)
    adapter = OllamaAdapter(model="gemma3:test")

    assert adapter.predict(_request(image_path)) == "no"
    assert captured["payload"]["images"] == [
        base64.b64encode(image_bytes).decode("ascii")
    ]
    assert captured["payload"]["model"] == "gemma3:test"
    assert captured["payload"]["prompt"] == _request(image_path).prompt
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {
        "temperature": 0.0,
        "num_predict": 8,
        "seed": 0,
    }


def test_ollama_adapter_reports_missing_raw_image() -> None:
    adapter = OllamaAdapter(model="gemma3:test")
    with pytest.raises(FileNotFoundError, match="Missing RAW image"):
        adapter.predict(_request(Path("missing-scene.png")))


def test_ollama_adapter_reports_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("detectivelab.adapters.ollama.urlopen", fail)
    adapter = OllamaAdapter(model="qwen3:test")
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        adapter.predict(_request())
