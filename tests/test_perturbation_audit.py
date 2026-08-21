from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.audit_perturbation_stability import _save_view


def test_blur_view_preserves_size(tmp_path: Path):
    image = Image.new("RGB", (20, 10), "white")
    out = tmp_path / "blur.png"
    _save_view(image, "blur_020", out)
    with Image.open(out) as result:
        assert result.size == (20, 10)


def test_downsample_view_restores_original_size(tmp_path: Path):
    image = Image.new("RGB", (20, 10), "white")
    out = tmp_path / "down.png"
    _save_view(image, "downsample_060", out)
    with Image.open(out) as result:
        assert result.size == (20, 10)
