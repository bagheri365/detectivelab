from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_DIR = Path("artifacts/benchmark_v0_0_1")
QUESTION_RESULTS = Path(
    "artifacts/evaluation/v0_1_question_gemma3_4b.jsonl"
)
RAW_RESULTS = Path(
    "artifacts/evaluation/v0_1_raw_gemma3_4b.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_item(records: list[dict]) -> dict[str, dict]:
    return {record["item_id"]: record for record in records}


def load_conflict_payloads() -> dict[str, dict]:
    items = {}

    for scene_dir in sorted(BENCHMARK_DIR.glob("scene_*")):
        payload_path = scene_dir / "payloads.json"
        payloads = json.loads(payload_path.read_text())

        for payload in payloads:
            if payload["family"] == "conflict":
                items[payload["item_id"]] = payload

    return items


def context_text(payload: dict, context_type: str) -> str:
    for entry in payload.get("context", []):
        if entry.get("type") == context_type:
            return entry.get("text", "")
    return ""


def main() -> None:
    question = index_by_item(load_jsonl(QUESTION_RESULTS))
    raw = index_by_item(load_jsonl(RAW_RESULTS))
    payloads = load_conflict_payloads()

    print("=" * 100)
    print("DetectiveLab v0.1 — Conflict Error Audit")
    print("=" * 100)

    raw_correct = 0
    question_correct = 0

    for item_id in sorted(payloads):
        payload = payloads[item_id]
        q = question[item_id]
        r = raw[item_id]

        if q["correct"]:
            question_correct += 1
        if r["correct"]:
            raw_correct += 1

        print()
        print(f"ITEM: {item_id}")
        print(f"GOLD: {r['gold']}")
        print(f"QUESTION: {q['prediction']}  correct={q['correct']}")
        print(f"RAW:      {r['prediction']}  correct={r['correct']}")
        print()
        print("WITNESS:")
        print(context_text(payload, "witness_testimony"))
        print()
        print("RULE:")
        print(context_text(payload, "case_rule"))
        print()
        print("QUESTION TEXT:")
        print(payload["question"])
        print()
        print("RAW MODEL OUTPUT:")
        print(repr(r.get("raw_output", "")))
        print("-" * 100)

    total = len(payloads)

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"QUESTION conflict accuracy: "
        f"{question_correct}/{total} ({question_correct / total:.1%})"
    )
    print(
        f"RAW conflict accuracy: "
        f"{raw_correct}/{total} ({raw_correct / total:.1%})"
    )


if __name__ == "__main__":
    main()