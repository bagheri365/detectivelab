import hashlib

import pytest
from PIL import Image

from detectivelab.domain.schema import ObjectKind, Point, Scene, SceneObject, Size, Visibility
from detectivelab.generation.scenes import generate_scene
from detectivelab.rendering.renderer import render_scene, render_scene_bytes


def test_renderer_is_byte_deterministic_for_same_scene():
    scene = generate_scene(42)
    first = render_scene_bytes(scene)
    second = render_scene_bytes(scene)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_different_scene_state_changes_rendered_bytes():
    assert render_scene_bytes(generate_scene(1)) != render_scene_bytes(generate_scene(2))


def test_renderer_returns_small_rgb_image():
    image = render_scene(generate_scene(7), canvas_size=256)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (256, 256)


def test_renderer_rejects_large_canvas():
    with pytest.raises(ValueError, match="128 and 384"):
        render_scene(generate_scene(7), canvas_size=512)


def test_hidden_object_does_not_change_pixels():
    visible_anchor = (
        SceneObject("key_1", ObjectKind.KEY, Point(0.25, 0.25), Size(0.08, 0.04), color="blue"),
        SceneObject("lamp_1", ObjectKind.LAMP, Point(0.75, 0.25), Size(0.12, 0.20), color="amber", state="on"),
        SceneObject("chair_1", ObjectKind.CHAIR, Point(0.25, 0.75), Size(0.16, 0.20), color="green"),
        SceneObject("window_1", ObjectKind.WINDOW, Point(0.75, 0.75), Size(0.20, 0.22), color="white", state="closed"),
    )
    with_hidden = Scene(
        "hidden_scene",
        1,
        visible_anchor + (
            SceneObject(
                "glass_1",
                ObjectKind.GLASS,
                Point(0.50, 0.50),
                Size(0.07, 0.11),
                color="red",
                state="intact",
                visibility=Visibility.HIDDEN,
            ),
        ),
    )
    without_hidden = Scene("plain_scene", 1, visible_anchor)
    assert render_scene_bytes(with_hidden) == render_scene_bytes(without_hidden)
