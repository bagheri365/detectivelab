from __future__ import annotations

from PIL import Image

from detectivelab.evaluation.uncertainty_prediction import (
    DEGRADATION_GRID,
    _degrade,
)


def test_degradation_grid_has_four_families():
    assert set(DEGRADATION_GRID) == {
        "blur",
        "downsample",
        "contrast",
        "occlusion",
    }


def test_each_degradation_family_has_five_levels():
    assert all(len(levels) == 5 for levels in DEGRADATION_GRID.values())


def test_blur_preserves_image_size():
    image = Image.new("RGB", (20, 10), "white")
    assert _degrade(image, "blur", 0.8).size == (20, 10)


def test_downsample_restores_image_size():
    image = Image.new("RGB", (20, 10), "white")
    assert _degrade(image, "downsample", 0.6).size == (20, 10)


def test_contrast_preserves_image_size():
    image = Image.new("RGB", (20, 10), "white")
    assert _degrade(image, "contrast", 0.6).size == (20, 10)


def test_occlusion_preserves_image_size():
    image = Image.new("RGB", (20, 10), "white")
    assert _degrade(image, "occlusion", 0.1).size == (20, 10)
