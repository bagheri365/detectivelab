from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from detectivelab.evaluation.staged import (
    expected_stages_from_extracted_evidence,
    parse_conflict_stages,
)


FIELDS = ("existence", "physical_state", "agreement", "verdict")
LINE_LABELS = {
    "EXISTENCE": "existence",
    "PHYSICAL_STATE": "physical_state",
    "AGREEMENT": "agreement",
    "VERDICT": "verdict",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _payload(benchmark: Path, scene_id: str) -> dict:
    payloads = _read_json(benchmark / scene_id / "payloads.json")
    return next(p for p in payloads if p["family"] == "conflict")


def _head(value: str) -> str:
    """Return the decision token before any free-form explanation."""
    value = value.strip().lower()
    head = re.split(r"\s+-\s+", value, maxsplit=1)[0].strip()
    # Strip trailing parenthetical explanations, e.g.
    # "closed (the scene shows closed)" -> "closed".
    head = re.sub(r"\s*\([^)]*\)\s*$", "", head).strip()
    return head


def _canonicalize(value: str, field: str) -> str:
    """Normalize semantic aliases without forgiving genuinely wrong decisions."""
    head = _head(value)

    if field == "existence":
        if head.startswith("present"):
            return "present"
        if head.startswith("absent"):
            return "absent"
        return head

    if field == "physical_state":
        if head in {"not applicable", "not-applicable", "n/a", "na"}:
            return "not_applicable"
        if head.startswith("not_applicable"):
            return "not_applicable"
        return head

    if field == "agreement":
        if head.startswith("support"):
            return "supports"
        if head.startswith("contradict"):
            return "contradicts"
        if head.startswith("unknown"):
            return "unknown"
        return head

    if field == "verdict":
        if head.startswith("support"):
            return "supported"
        if head.startswith("contradict"):
            return "contradicted"
        if head.startswith("unknown"):
            return "unknown"
        return head

    raise ValueError(f"Unknown field: {field}")


def _parse_partial(raw_output: str) -> dict[str, str]:
    """Recover any emitted stages even when the full staged parser fails/truncates."""
    values: dict[str, str] = {}
    for line in raw_output.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        field = LINE_LABELS.get(label.strip().upper())
        if field and value.strip():
            values[field] = value.strip()
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit CONFLICT_STAGED intermediate decisions with semantic normalization."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("CONFLICT_STAGED", "CONFLICT_EPISTEMIC", "CONFLICT_STAGED_PARAPHRASE", "CONFLICT_EPISTEMIC_PARAPHRASE"),
        default=None,
        help="Optional condition filter. If omitted, audit all staged conflict conditions.",
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    allowed_conditions = {"CONFLICT_STAGED", "CONFLICT_EPISTEMIC", "CONFLICT_STAGED_PARAPHRASE", "CONFLICT_EPISTEMIC_PARAPHRASE"}
    if args.condition is not None:
        allowed_conditions = {args.condition}
    records = [r for r in records if r.get("condition") in allowed_conditions]

    correct = Counter()
    emitted = Counter()
    full_parse_failures = 0
    total = 0

    for record in records:
        scene_id = record["scene_id"]
        payload = _payload(args.benchmark, scene_id)
        expected = expected_stages_from_extracted_evidence(
            image_path=args.benchmark / scene_id / "scene.png",
            payload=payload,
        )

        parsed = parse_conflict_stages(record["raw_output"])
        partial = _parse_partial(record["raw_output"])
        if parsed is None:
            full_parse_failures += 1

        total += 1
        print("=" * 88)
        print(record["item_id"])
        print(
            f"benchmark gold: {record['gold']} | "
            f"final prediction: {record['prediction']} | "
            f"correct={record['correct']}"
        )
        print(f"expected from extracted evidence: {expected}")
        print(f"model stages: {parsed}")
        if parsed is None and partial:
            print(f"partial stages recovered: {partial}")

        for field in FIELDS:
            expected_value = _canonicalize(getattr(expected, field), field)

            if parsed is not None:
                raw_value = getattr(parsed, field)
            else:
                raw_value = partial.get(field)

            if raw_value is None:
                print(f"  {field}: MISSING (expected={expected_value})")
                continue

            emitted[field] += 1
            model_value = _canonicalize(raw_value, field)
            ok = model_value == expected_value
            correct[field] += int(ok)
            print(
                f"  {field}: {'PASS' if ok else 'FAIL'} "
                f"(model={model_value!r}, expected={expected_value!r})"
            )

    print("=" * 88)
    print(f"Items: {total}")
    if total:
        for field in FIELDS:
            all_items_rate = correct[field] / total
            coverage = emitted[field] / total
            emitted_rate = correct[field] / emitted[field] if emitted[field] else 0.0
            print(
                f"{field}: {correct[field]}/{total} ({all_items_rate:.1%}) overall; "
                f"{correct[field]}/{emitted[field]} ({emitted_rate:.1%}) among emitted; "
                f"coverage={coverage:.1%}"
            )
        if full_parse_failures:
            print(f"full-format parse failures: {full_parse_failures}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
