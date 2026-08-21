from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen


BENCHMARK = Path("artifacts/benchmark_v0_0_1")
MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Pick 2 supported, 2 contradicted, 2 unknown cases.
CASE_IDS = [
    "scene_0001",
    "scene_0004",
    "scene_0002",
    "scene_0005",
    "scene_0000",
    "scene_0003",
]


def ollama(prompt: str, image_path: Path | None = None) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 32,
            "seed": 0,
        },
    }

    if image_path is not None:
        payload["images"] = [
            base64.b64encode(image_path.read_bytes()).decode("ascii")
        ]

    req = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=120) as response:
        body = json.loads(response.read().decode())

    return str(body["response"]).strip()


def load_conflict(scene_id: str) -> tuple[dict, dict]:
    scene_dir = BENCHMARK / scene_id

    payloads = json.loads((scene_dir / "payloads.json").read_text())
    questions = json.loads((scene_dir / "questions.json").read_text())

    payload = next(x for x in payloads if x["family"] == "conflict")
    question = next(x for x in questions if x["family"] == "conflict")

    return payload, question


def witness_text(payload: dict) -> str:
    return next(
        x["text"]
        for x in payload["context"]
        if x["type"] == "witness_testimony"
    )


def main() -> None:
    for scene_id in CASE_IDS:
        payload, question = load_conflict(scene_id)
        image = BENCHMARK / scene_id / "scene.png"

        witness = witness_text(payload)
        gold = question["answer"]

        print()
        print("=" * 90)
        print(scene_id)
        print(f"GOLD: {gold}")
        print(f"WITNESS: {witness}")

        # Stage 1 — general visual observation
        perception_prompt = f"""
Look carefully at the image.

A witness made this claim:

{witness}

Identify the object mentioned by the witness and report only its visible current state.

Do not judge the witness statement yet.
Answer with a short factual phrase only.
""".strip()

        perception = ollama(perception_prompt, image)

        print()
        print("1. PERCEPTION")
        print(perception)

        # Stage 2 — direct comparison
        comparison_prompt = f"""
Look carefully at the image.

Witness statement:
{witness}

Compare the witness statement with what is visibly shown in the image.

Answer exactly one word:

matches
conflicts
unknown

Definitions:
matches = the visible scene supports the witness statement
conflicts = the visible scene shows the opposite
unknown = the relevant object or state cannot be determined from the scene
""".strip()

        comparison = ollama(comparison_prompt, image)

        print()
        print("2. COMPARISON")
        print(comparison)

        # Stage 3 — benchmark label mapping
        verdict_prompt = f"""
Look carefully at the image.

Witness statement:
{witness}

Use these definitions:

supported = the visible scene matches the testimony
contradicted = the visible scene conflicts with the testimony
unknown = the scene does not contain enough evidence to decide

Return exactly one label:

supported
contradicted
unknown
""".strip()

        verdict = ollama(verdict_prompt, image)

        print()
        print("3. VERDICT")
        print(verdict)


if __name__ == "__main__":
    main()