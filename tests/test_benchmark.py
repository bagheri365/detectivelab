import hashlib
import json

import pytest

from detectivelab.benchmark.export import export_benchmark
from detectivelab.benchmark.validate import validate_benchmark


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_export_has_expected_shape(tmp_path):
    manifest = export_benchmark(tmp_path, seeds=(0, 1))
    assert manifest["scene_count"] == 2
    assert manifest["item_count"] == 6
    for seed in (0, 1):
        case = tmp_path / f"scene_{seed:04d}"
        assert {p.name for p in case.iterdir()} == {
            "scene.json", "scene.png", "questions.json", "payloads.json", "provenance.json"
        }


def test_export_is_byte_deterministic(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    export_benchmark(first, seeds=(0, 1, 2))
    export_benchmark(second, seeds=(0, 1, 2))
    assert _tree_hash(first) == _tree_hash(second)


def test_validator_passes_clean_export(tmp_path):
    export_benchmark(tmp_path, seeds=tuple(range(10)))
    report = validate_benchmark(tmp_path)
    assert report["status"] == "PASS"
    assert report["scene_count"] == 10
    assert report["item_count"] == 30
    assert report["answer_counts"]["spatial"] == {"no": 5, "yes": 5}
    assert report["answer_counts"]["conflict"] == {"contradicted": 5, "unknown": 5}


def test_validator_catches_tampering(tmp_path):
    export_benchmark(tmp_path, seeds=(0,))
    scene_path = tmp_path / "scene_0000" / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["seed"] = 999
    scene_path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_benchmark(tmp_path)
