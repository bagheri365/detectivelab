"""Validation for exported DetectiveLab benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_CASE_FILES = ("scene.json", "scene.png", "questions.json", "payloads.json", "provenance.json")
EXPECTED_FAMILIES = {"spatial", "state", "conflict"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_benchmark(root: str | Path) -> dict[str, Any]:
    """Validate file completeness, hashes, IDs, counts, and basic label balance.

    Raises ``ValueError`` on the first failed invariant. Returns a compact audit
    report when the export is internally consistent.
    """

    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("missing manifest.json")

    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    if manifest.get("scene_count") != len(cases):
        raise ValueError("manifest scene_count does not match cases")

    family_counts: Counter[str] = Counter()
    conflict_rule_texts: set[str] = set()
    answer_counts: dict[str, Counter[str]] = {}
    item_ids: set[str] = set()

    for case in cases:
        case_dir = root / case["path"]
        for filename in REQUIRED_CASE_FILES:
            if not (case_dir / filename).exists():
                raise ValueError(f"{case['scene_id']}: missing {filename}")

        provenance_path = case_dir / "provenance.json"
        if _sha256_file(provenance_path) != case["provenance_sha256"]:
            raise ValueError(f"{case['scene_id']}: provenance hash mismatch")
        provenance = json.loads(provenance_path.read_text())

        for filename in ("scene.json", "scene.png", "questions.json", "payloads.json"):
            expected = provenance["artifacts"].get(filename)
            actual = _sha256_file(case_dir / filename)
            if actual != expected:
                raise ValueError(f"{case['scene_id']}: hash mismatch for {filename}")

        scene = json.loads((case_dir / "scene.json").read_text())
        if scene["scene_id"] != case["scene_id"] or scene["seed"] != case["seed"]:
            raise ValueError(f"{case['scene_id']}: scene identity mismatch")
        if not (4 <= len(scene["objects"]) <= 8):
            raise ValueError(f"{case['scene_id']}: object-count guardrail violated")

        objects = scene["objects"]
        object_by_id = {obj["object_id"]: obj for obj in objects}
        visual_labels = [(obj.get("color"), obj["kind"]) for obj in objects]
        if len(visual_labels) != len(set(visual_labels)):
            raise ValueError(f"{case['scene_id']}: duplicate color+kind visual label")

        # v0.0 intentionally excludes occlusion as an experimental variable.
        if any(obj.get("visibility") != "visible" for obj in objects):
            raise ValueError(f"{case['scene_id']}: v0.0 contains non-visible object")

        # Exported relations must agree with canonical geometry.
        for rel in scene.get("relations", []):
            subject = object_by_id[rel["subject_id"]]
            reference = object_by_id[rel["object_id"]]
            sx = subject["position"]["x"]
            rx = reference["position"]["x"]
            if rel["relation"] == "left_of" and not sx < rx:
                raise ValueError(f"{case['scene_id']}: left_of relation contradicts geometry")
            if rel["relation"] == "right_of" and not sx > rx:
                raise ValueError(f"{case['scene_id']}: right_of relation contradicts geometry")

        # No object boxes may overlap in the v0.0 clean benchmark.
        for index, first in enumerate(objects):
            for second in objects[index + 1:]:
                overlap_x = abs(first["position"]["x"] - second["position"]["x"]) < (
                    first["size"]["width"] + second["size"]["width"]
                ) / 2
                overlap_y = abs(first["position"]["y"] - second["position"]["y"]) < (
                    first["size"]["height"] + second["size"]["height"]
                ) / 2
                if overlap_x and overlap_y:
                    raise ValueError(f"{case['scene_id']}: overlapping objects in clean benchmark")

        questions = json.loads((case_dir / "questions.json").read_text())
        payloads = json.loads((case_dir / "payloads.json").read_text())
        if len(payloads) != 3:
            raise ValueError(f"{case['scene_id']}: expected exactly 3 participant payloads")
        payload_by_family = {payload["family"]: payload for payload in payloads}
        if set(payload_by_family) != EXPECTED_FAMILIES:
            raise ValueError(f"{case['scene_id']}: payload families do not match benchmark families")
        if payload_by_family["spatial"].get("context") != []:
            raise ValueError(f"{case['scene_id']}: spatial payload leaks text evidence")
        if payload_by_family["state"].get("context") != []:
            raise ValueError(f"{case['scene_id']}: state payload leaks text evidence")
        conflict_context = payload_by_family["conflict"].get("context", [])
        conflict_types = [entry.get("type") for entry in conflict_context]
        if conflict_types != ["witness_testimony", "case_rule"]:
            raise ValueError(f"{case['scene_id']}: conflict payload is missing testimony or case rule")
        conflict_rule_texts.add(conflict_context[1]["text"])
        if len(questions) != 3:
            raise ValueError(f"{case['scene_id']}: expected exactly 3 questions")
        families = {item["family"] for item in questions}
        if families != EXPECTED_FAMILIES:
            raise ValueError(f"{case['scene_id']}: missing or duplicate case family")

        for item in questions:
            if item["scene_id"] != case["scene_id"]:
                raise ValueError(f"{case['scene_id']}: question scene_id mismatch")
            if item["item_id"] in item_ids:
                raise ValueError(f"duplicate item_id: {item['item_id']}")
            item_ids.add(item["item_id"])
            for evidence_id in item.get("evidence_ids", []):
                obj = object_by_id.get(evidence_id)
                if obj is not None and obj.get("visibility") != "visible":
                    raise ValueError(
                        f"{case['scene_id']}: question depends on non-visible evidence {evidence_id}"
                    )
            family_counts[item["family"]] += 1
            answer_counts.setdefault(item["family"], Counter())[item["answer"]] += 1

    if manifest.get("item_count") != len(item_ids):
        raise ValueError("manifest item_count does not match exported questions")
    if dict(sorted(family_counts.items())) != manifest.get("family_counts"):
        raise ValueError("manifest family_counts mismatch")

    # Shortcut-leak checks. Spatial/state remain balanced binary tasks. Conflict
    # must expose all three verdicts while keeping rule wording constant so the
    # QUESTION-only condition cannot infer the label from the policy text.
    for family in ("spatial", "state"):
        if len(answer_counts.get(family, {})) < 2:
            raise ValueError(f"{family} labels are degenerate")
    if set(answer_counts.get("conflict", {})) != {"supported", "contradicted", "unknown"}:
        raise ValueError("conflict family must contain supported, contradicted, and unknown labels")
    if len(conflict_rule_texts) != 1:
        raise ValueError("conflict case_rule text must be identical across the benchmark")

    report = {
        "scene_count": len(cases),
        "item_count": len(item_ids),
        "family_counts": dict(sorted(family_counts.items())),
        "answer_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(answer_counts.items())
        },
        "status": "PASS",
    }
    return report
