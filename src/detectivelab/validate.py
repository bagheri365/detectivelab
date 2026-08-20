# src/detectivelab/validate.py

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_CASE_FILES = {
    "scene.json",
    "scene.png",
    "questions.json",
    "payloads.json",
    "provenance.json",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m detectivelab.validate <benchmark_dir>")
        return 2

    benchmark_dir = Path(sys.argv[1])

    manifest_path = benchmark_dir / "manifest.json"
    audit_path = benchmark_dir / "AUDIT.json"

    errors: list[str] = []

    if not benchmark_dir.is_dir():
        errors.append(f"Benchmark directory does not exist: {benchmark_dir}")

    if not manifest_path.exists():
        errors.append("Missing manifest.json")

    if not audit_path.exists():
        errors.append("Missing AUDIT.json")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())

    cases = manifest.get("cases", [])

    for case in cases:
        case_dir = benchmark_dir / case["path"]

        if not case_dir.is_dir():
            errors.append(f"Missing case directory: {case['path']}")
            continue

        actual_files = {p.name for p in case_dir.iterdir() if p.is_file()}
        missing = REQUIRED_CASE_FILES - actual_files

        if missing:
            errors.append(
                f"{case['path']} missing files: {', '.join(sorted(missing))}"
            )

    if manifest.get("scene_count") != len(cases):
        errors.append(
            "scene_count does not match number of manifest cases"
        )

    expected_items = manifest.get("scene_count", 0) * 3
    if manifest.get("item_count") != expected_items:
        errors.append(
            f"Expected {expected_items} items, found "
            f"{manifest.get('item_count')}"
        )

    if audit.get("status") != "PASS":
        errors.append(
            f"AUDIT.json status is {audit.get('status')!r}, not 'PASS'"
        )

    if audit.get("scene_count") != manifest.get("scene_count"):
        errors.append("AUDIT and manifest scene counts disagree")

    if audit.get("item_count") != manifest.get("item_count"):
        errors.append("AUDIT and manifest item counts disagree")

    if audit.get("family_counts") != manifest.get("family_counts"):
        errors.append("AUDIT and manifest family counts disagree")

    if audit.get("answer_counts") != manifest.get("answer_counts"):
        errors.append("AUDIT and manifest answer counts disagree")

    if errors:
        print(f"Benchmark: {benchmark_dir}")
        for error in errors:
            print(f"FAIL: {error}")
        print("Status: FAIL")
        return 1

    print(f"Benchmark: {benchmark_dir}")
    print(f"Version: {manifest.get('benchmark_version')}")
    print(f"Scenes: {manifest.get('scene_count')}")
    print(f"Items: {manifest.get('item_count')}")
    print("Required files: PASS")
    print("Manifest/AUDIT consistency: PASS")
    print("Audit status: PASS")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())