from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(record: dict[str, Any]) -> tuple[str, str]:
    # Robustness runners may include a variant field in addition to item_id.
    variant = (
        record.get("variant")
        or record.get("paraphrase_variant")
        or record.get("case_variant")
        or ""
    )
    return str(record["item_id"]), str(variant)


def _index(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {_key(r): r for r in records}


def _compact_raw(record: dict[str, Any]) -> str:
    raw = str(record.get("raw_output", "")).strip()
    return raw if raw else "<empty>"


def _print_record(label: str, record: dict[str, Any]) -> None:
    print(f"{label}:")
    print(f"  condition:  {record.get('condition')}")
    print(f"  gold:       {record.get('gold')}")
    print(f"  prediction: {record.get('prediction')}")
    print(f"  correct:    {record.get('correct')}")
    for name in ("variant", "paraphrase_variant", "case_variant"):
        if record.get(name) is not None:
            print(f"  {name}: {record.get(name)}")
    print("  raw:")
    for line in _compact_raw(record).splitlines():
        print(f"    {line}")


def compare_pair(staged_path: Path, epistemic_path: Path, title: str) -> None:
    staged = _index(_read_jsonl(staged_path))
    epistemic = _index(_read_jsonl(epistemic_path))

    keys = sorted(set(staged) & set(epistemic))
    if not keys:
        print(f"\n{title}: no overlapping records")
        return

    changed = []
    helped = []
    hurt = []
    unchanged_wrong = []

    for key in keys:
        s = staged[key]
        e = epistemic[key]
        s_ok = bool(s.get("correct"))
        e_ok = bool(e.get("correct"))

        if s.get("prediction") != e.get("prediction") or s_ok != e_ok:
            changed.append(key)
        if not s_ok and e_ok:
            helped.append(key)
        elif s_ok and not e_ok:
            hurt.append(key)
        elif not s_ok and not e_ok:
            unchanged_wrong.append(key)

    print("=" * 100)
    print(title)
    print(f"records compared: {len(keys)}")
    print(f"changed outcomes: {len(changed)}")
    print(f"helped by epistemic: {len(helped)}")
    print(f"hurt by epistemic:   {len(hurt)}")
    print(f"wrong in both:        {len(unchanged_wrong)}")

    for bucket_name, bucket in (
        ("HURT BY EPISTEMIC", hurt),
        ("HELPED BY EPISTEMIC", helped),
        ("WRONG IN BOTH", unchanged_wrong),
    ):
        if not bucket:
            continue
        print("\n" + "-" * 100)
        print(bucket_name)
        for key in bucket:
            print("\n" + "." * 100)
            print(f"key: {key}")
            _print_record("STAGED", staged[key])
            _print_record("EPISTEMIC", epistemic[key])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit where CONFLICT_EPISTEMIC helps or hurts relative to CONFLICT_STAGED."
    )
    parser.add_argument("--canonical-staged", type=Path)
    parser.add_argument("--canonical-epistemic", type=Path)
    parser.add_argument("--paraphrase-staged", type=Path)
    parser.add_argument("--paraphrase-epistemic", type=Path)
    parser.add_argument("--case-staged", type=Path)
    parser.add_argument("--case-epistemic", type=Path)
    args = parser.parse_args()

    pairs = [
        (
            args.canonical_staged,
            args.canonical_epistemic,
            "CANONICAL CONFLICT",
        ),
        (
            args.paraphrase_staged,
            args.paraphrase_epistemic,
            "PARAPHRASE ROBUSTNESS",
        ),
        (
            args.case_staged,
            args.case_epistemic,
            "CASE VARIATION",
        ),
    ]

    ran = False
    for staged, epistemic, title in pairs:
        if staged and epistemic:
            compare_pair(staged, epistemic, title)
            ran = True

    if not ran:
        parser.error("provide at least one staged/epistemic result pair")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
