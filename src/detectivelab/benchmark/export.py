"""Deterministic benchmark export for DetectiveLab v0.0.

Exports one folder per scene with canonical hidden state, rendered PNG,
three benchmark questions, and provenance hashes. A top-level manifest records
the frozen slice and label counts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from detectivelab.generation.questions import generate_questions
from detectivelab.generation.scenes import generate_scene
from detectivelab.rendering.renderer import render_scene_bytes
from detectivelab.benchmark.payloads import build_payload

BENCHMARK_VERSION = "v0.0"
DEFAULT_CANVAS_SIZE = 256


def _canonical_json_bytes(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return _sha256(data)


def export_benchmark(
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = tuple(range(10)),
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> dict[str, Any]:
    """Export a deterministic benchmark slice and return its manifest."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds must be unique")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    answer_counts: dict[str, Counter[str]] = {}

    for seed in seeds:
        scene = generate_scene(seed)
        items = generate_questions(scene)
        case_dir = root / scene.scene_id
        case_dir.mkdir(parents=True, exist_ok=True)

        scene_bytes = _canonical_json_bytes(scene.to_dict())
        questions_data = [item.to_dict() for item in items]
        questions_bytes = _canonical_json_bytes(questions_data)
        payloads_data = [build_payload(scene, item) for item in items]
        payloads_bytes = _canonical_json_bytes(payloads_data)
        image_bytes = render_scene_bytes(scene, canvas_size=canvas_size)

        hashes = {
            "scene.json": _write_bytes(case_dir / "scene.json", scene_bytes),
            "scene.png": _write_bytes(case_dir / "scene.png", image_bytes),
            "questions.json": _write_bytes(case_dir / "questions.json", questions_bytes),
            "payloads.json": _write_bytes(case_dir / "payloads.json", payloads_bytes),
        }

        for item in items:
            family = item.family.value
            family_counts[family] += 1
            answer_counts.setdefault(family, Counter())[item.answer] += 1

        provenance = {
            "benchmark_version": BENCHMARK_VERSION,
            "scene_id": scene.scene_id,
            "seed": seed,
            "schema_version": scene.schema_version,
            "renderer": {
                "canvas_size": canvas_size,
                "format": "PNG",
            },
            "artifacts": hashes,
        }
        provenance_bytes = _canonical_json_bytes(provenance)
        provenance_hash = _write_bytes(case_dir / "provenance.json", provenance_bytes)

        manifest_cases.append(
            {
                "scene_id": scene.scene_id,
                "seed": seed,
                "path": scene.scene_id,
                "provenance_sha256": provenance_hash,
            }
        )

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "scene_count": len(seeds),
        "item_count": len(seeds) * 3,
        "seeds": list(seeds),
        "canvas_size": canvas_size,
        "family_counts": dict(sorted(family_counts.items())),
        "answer_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(answer_counts.items())
        },
        "cases": manifest_cases,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _write_bytes(root / "manifest.json", manifest_bytes)
    return manifest
