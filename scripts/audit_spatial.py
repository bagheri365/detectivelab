from __future__ import annotations

import json
from pathlib import Path

from detectivelab.evaluation.structured import build_structured_evidence
from detectivelab.extraction import (
    extract_scene_facts,
    extract_structured_evidence,
)


BENCHMARK = Path("artifacts/benchmark_v0_0_1")

ORACLE_RESULTS = Path(
    "artifacts/evaluation/v0_2_oracle_structured_gemma3_4b.jsonl"
)
EXTRACTED_RESULTS = Path(
    "artifacts/evaluation/v0_2_extracted_structured_gemma3_4b.jsonl"
)


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {record["item_id"]: record for record in records}


def spatial_payload(scene_dir: Path) -> dict:
    payloads = json.loads((scene_dir / "payloads.json").read_text())
    return next(p for p in payloads if p["family"] == "spatial")


def spatial_question(scene_dir: Path) -> dict:
    questions = json.loads((scene_dir / "questions.json").read_text())
    return next(q for q in questions if q["family"] == "spatial")


def main() -> None:
    oracle_results = load_jsonl(ORACLE_RESULTS)
    extracted_results = load_jsonl(EXTRACTED_RESULTS)

    for scene_dir in sorted(BENCHMARK.glob("scene_*")):
        payload = spatial_payload(scene_dir)
        question = spatial_question(scene_dir)

        item_id = payload["item_id"]

        scene = json.loads((scene_dir / "scene.json").read_text())
        oracle_evidence = build_structured_evidence(scene)

        image_path = scene_dir / "scene.png"
        extracted_facts = extract_scene_facts(image_path)
        extracted_evidence = extract_structured_evidence(image_path)

        oracle = oracle_results[item_id]
        extracted = extracted_results[item_id]

        print()
        print("=" * 100)
        print(item_id)
        print(f"QUESTION: {payload['question']}")
        print(f"GOLD: {question['answer']}")
        print(
            f"ORACLE PREDICTION: {oracle['prediction']} "
            f"correct={oracle['correct']}"
        )
        print(
            f"EXTRACTED PREDICTION: {extracted['prediction']} "
            f"correct={extracted['correct']}"
        )

        print()
        print("ORACLE EVIDENCE")
        if isinstance(oracle_evidence, str):
            print(oracle_evidence)
        else:
            for line in oracle_evidence:
                print(f"  - {line}")

        print()
        print("EXTRACTED OBJECTS")
        for fact in extracted_facts:
            print(f"  - {fact}")

        print()
        print("EXTRACTED STRUCTURED EVIDENCE")
        print(extracted_evidence)


if __name__ == "__main__":
    main()