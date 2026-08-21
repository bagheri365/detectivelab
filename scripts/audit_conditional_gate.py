from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["item_id"]): r for r in records}


def _extract_existence(raw: str) -> str:
    text = raw.strip().lower()
    m = re.search(r"(?mi)^\s*existence\s*:\s*([^\n\r]+)", text)
    if not m:
        return "<missing>"
    value = m.group(1).strip()
    value = re.split(r"\s+-\s+", value, maxsplit=1)[0].strip()
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    if value.startswith("present"):
        return "present"
    if value.startswith("absent"):
        return "absent"
    return value


def _compact(raw: str) -> str:
    raw = raw.strip()
    return raw if raw else "<empty>"


def _gate(record: dict[str, Any]) -> tuple[str, str]:
    """
    Best-effort gate reconstruction from conditional raw output.
    Returns (existence, branch).
    """
    raw = str(record.get("raw_output", ""))
    existence = _extract_existence(raw)
    if existence == "absent":
        return existence, "forced_unknown"
    if existence == "present":
        return existence, "staged_present"
    return existence, "unknown"


def _print_record(label: str, record: dict[str, Any]) -> None:
    print(f"{label}:")
    print(f"  condition:  {record.get('condition')}")
    print(f"  gold:       {record.get('gold')}")
    print(f"  prediction: {record.get('prediction')}")
    print(f"  correct:    {record.get('correct')}")
    print("  raw:")
    for line in _compact(str(record.get("raw_output", ""))).splitlines():
        print(f"    {line}")


def audit_model(
    name: str,
    conditional_path: Path,
    staged_path: Path,
    epistemic_path: Path,
) -> None:
    conditional = _index(_read_jsonl(conditional_path))
    staged = _index(_read_jsonl(staged_path))
    epistemic = _index(_read_jsonl(epistemic_path))

    keys = sorted(set(conditional) & set(staged) & set(epistemic))

    print("=" * 110)
    print(name)
    print(f"records compared: {len(keys)}")

    stats = {
        "conditional_correct": 0,
        "gate_absent": 0,
        "gate_present": 0,
        "gate_unknown": 0,
        "conditional_only_wrong": 0,
        "all_wrong": 0,
    }

    for key in keys:
        c = conditional[key]
        s = staged[key]
        e = epistemic[key]
        existence, branch = _gate(c)

        stats["conditional_correct"] += int(bool(c.get("correct")))
        if existence == "absent":
            stats["gate_absent"] += 1
        elif existence == "present":
            stats["gate_present"] += 1
        else:
            stats["gate_unknown"] += 1

        c_ok = bool(c.get("correct"))
        s_ok = bool(s.get("correct"))
        e_ok = bool(e.get("correct"))

        if not c_ok and (s_ok or e_ok):
            stats["conditional_only_wrong"] += 1
            print("\n" + "-" * 110)
            print(f"{key}")
            print(
                f"gate existence={existence} branch={branch} | "
                f"gold={c.get('gold')} | "
                f"conditional={c.get('prediction')} | "
                f"staged={s.get('prediction')} | "
                f"epistemic={e.get('prediction')}"
            )
            _print_record("CONDITIONAL", c)
            _print_record("STAGED", s)
            _print_record("EPISTEMIC", e)
        elif not c_ok and not s_ok and not e_ok:
            stats["all_wrong"] += 1

    print("\n" + "." * 110)
    print("SUMMARY")
    print(f"conditional accuracy: {stats['conditional_correct']}/{len(keys)}")
    print(f"gate absent:  {stats['gate_absent']}")
    print(f"gate present: {stats['gate_present']}")
    print(f"gate unknown: {stats['gate_unknown']}")
    print(f"conditional-only failures: {stats['conditional_only_wrong']}")
    print(f"wrong in all three: {stats['all_wrong']}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Audit canonical CONFLICT_CONDITIONAL gating against staged and epistemic baselines."
    )
    p.add_argument("--gemma-conditional", type=Path, required=True)
    p.add_argument("--gemma-staged", type=Path, required=True)
    p.add_argument("--gemma-epistemic", type=Path, required=True)
    p.add_argument("--qwen-conditional", type=Path, required=True)
    p.add_argument("--qwen-staged", type=Path, required=True)
    p.add_argument("--qwen-epistemic", type=Path, required=True)
    args = p.parse_args()

    audit_model(
        "GEMMA 3 4B",
        args.gemma_conditional,
        args.gemma_staged,
        args.gemma_epistemic,
    )
    audit_model(
        "QWEN3 4B INSTRUCT",
        args.qwen_conditional,
        args.qwen_staged,
        args.qwen_epistemic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
