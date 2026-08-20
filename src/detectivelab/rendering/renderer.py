"""Deterministic, CPU-light renderer for DetectiveLab hidden states.

The renderer deliberately uses only geometric primitives. There are no font,
OCR, network, or model dependencies, which keeps images stable across runs and
makes visual state an explicit function of the canonical scene schema.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from detectivelab.domain.schema import ObjectKind, Scene, SceneObject, Visibility

DEFAULT_CANVAS_SIZE = 256
_BACKGROUND = (244, 241, 233)
_OUTLINE = (30, 30, 30)
_MUTED = (120, 120, 120)
_WHITE = (255, 255, 255)

_COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (70, 120, 190),
    "red": (190, 75, 70),
    "green": (80, 150, 100),
    "amber": (210, 150, 55),
    "black": (45, 45, 45),
    "white": (235, 235, 235),
}


def _box(obj: SceneObject, canvas_size: int) -> tuple[int, int, int, int]:
    half_w = obj.size.width * canvas_size / 2
    half_h = obj.size.height * canvas_size / 2
    cx = obj.position.x * canvas_size
    cy = obj.position.y * canvas_size
    return (
        round(cx - half_w),
        round(cy - half_h),
        round(cx + half_w),
        round(cy + half_h),
    )


def _inset(box: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return x0 + amount, y0 + amount, x1 - amount, y1 - amount


def _midpoint(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return (x0 + x1) // 2, (y0 + y1) // 2


def _draw_key(draw: ImageDraw.ImageDraw, box, fill) -> None:
    x0, y0, x1, y1 = box
    cy = (y0 + y1) // 2
    radius = max(2, (y1 - y0) // 3)
    draw.ellipse((x0, cy - radius, x0 + 2 * radius, cy + radius), fill=fill, outline=_OUTLINE, width=2)
    draw.line((x0 + 2 * radius, cy, x1, cy), fill=_OUTLINE, width=3)
    tooth = max(2, (y1 - y0) // 4)
    draw.line((x1 - tooth * 2, cy, x1 - tooth * 2, cy + tooth), fill=_OUTLINE, width=2)
    draw.line((x1 - tooth, cy, x1 - tooth, cy + tooth), fill=_OUTLINE, width=2)


def _draw_lamp(draw: ImageDraw.ImageDraw, box, fill, state: str | None) -> None:
    x0, y0, x1, y1 = box
    cx, _ = _midpoint(box)
    shade_bottom = y0 + round((y1 - y0) * 0.48)
    draw.polygon(((cx, y0), (x0, shade_bottom), (x1, shade_bottom)), fill=fill, outline=_OUTLINE)
    draw.line((cx, shade_bottom, cx, y1 - 4), fill=_OUTLINE, width=3)
    draw.line((x0 + 3, y1 - 3, x1 - 3, y1 - 3), fill=_OUTLINE, width=3)
    if state == "on":
        draw.ellipse(_inset((x0, y0, x1, shade_bottom), 4), outline=_WHITE, width=3)
    elif state == "off":
        draw.line((x0 + 4, y0 + 4, x1 - 4, shade_bottom - 4), fill=_OUTLINE, width=2)


def _draw_openable(draw: ImageDraw.ImageDraw, box, fill, state: str | None, *, vertical: bool) -> None:
    """Draw a door/window as an actual frame plus physical panel.

    Closed means the panel fills the frame. Open means the panel is visibly
    displaced/angled, leaving empty background visible inside the frame. This
    avoids arbitrary slash/cross state symbols.
    """
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=_BACKGROUND, outline=_OUTLINE, width=3)

    if state == "open":
        if vertical:
            # Door hinged on the left: narrow angled panel occupies only part
            # of the doorway, leaving a conspicuous open gap.
            hinge_x = x0 + 3
            panel_right = x0 + max(7, round((x1 - x0) * 0.38))
            draw.polygon(
                ((hinge_x, y0 + 3), (panel_right, y0 + 8),
                 (panel_right, y1 - 8), (hinge_x, y1 - 3)),
                fill=fill, outline=_OUTLINE,
            )
            knob_y = (y0 + y1) // 2
            draw.ellipse((panel_right - 4, knob_y - 2, panel_right, knob_y + 2), fill=_OUTLINE)
        else:
            # Window: two panes swung away from the center, exposing the
            # background through a broad central opening.
            cx = (x0 + x1) // 2
            gap = max(5, round((x1 - x0) * 0.22))
            draw.polygon(((x0 + 3, y0 + 4), (cx - gap, y0 + 8),
                          (cx - gap, y1 - 8), (x0 + 3, y1 - 4)),
                         fill=fill, outline=_OUTLINE)
            draw.polygon(((x1 - 3, y0 + 4), (cx + gap, y0 + 8),
                          (cx + gap, y1 - 8), (x1 - 3, y1 - 4)),
                         fill=fill, outline=_OUTLINE)
    else:
        # Closed: solid panel fills the frame. A frame seam communicates the
        # object identity, not the state.
        inner = _inset(box, 4)
        draw.rectangle(inner, fill=fill, outline=_OUTLINE, width=2)
        if vertical:
            ix0, iy0, ix1, iy1 = inner
            knob_y = (iy0 + iy1) // 2
            draw.ellipse((ix1 - 7, knob_y - 2, ix1 - 3, knob_y + 2), fill=_OUTLINE)
        else:
            ix0, iy0, ix1, iy1 = inner
            cx, cy = _midpoint(inner)
            draw.line((cx, iy0, cx, iy1), fill=_OUTLINE, width=2)
            draw.line((ix0, cy, ix1, cy), fill=_OUTLINE, width=2)

def _draw_notebook(draw: ImageDraw.ImageDraw, box, fill, state: str | None) -> None:
    """Draw open as two splayed pages and closed as a single bound cover."""
    x0, y0, x1, y1 = box
    cx, cy = _midpoint(box)
    if state == "open":
        # Two page polygons with a central gutter; silhouette differs clearly
        # from the single rectangular closed book.
        draw.polygon(((cx, y0 + 3), (x0 + 2, y0 + 7), (x0 + 4, y1 - 3),
                      (cx, y1 - 6)), fill=fill, outline=_OUTLINE)
        draw.polygon(((cx, y0 + 3), (x1 - 2, y0 + 7), (x1 - 4, y1 - 3),
                      (cx, y1 - 6)), fill=fill, outline=_OUTLINE)
        draw.line((cx, y0 + 3, cx, y1 - 6), fill=_OUTLINE, width=2)
        draw.line((x0 + 7, cy, cx - 4, cy), fill=_WHITE, width=1)
        draw.line((cx + 4, cy, x1 - 7, cy), fill=_WHITE, width=1)
    else:
        draw.rounded_rectangle(box, radius=2, fill=fill, outline=_OUTLINE, width=2)
        # Visible spine makes this read as a closed book, not a generic box.
        draw.line((x0 + 4, y0 + 3, x0 + 4, y1 - 3), fill=_OUTLINE, width=2)
        draw.line((x0 + 8, y0 + 6, x1 - 5, y0 + 6), fill=_WHITE, width=1)

def _draw_briefcase(draw: ImageDraw.ImageDraw, box, fill, state: str | None) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=fill, outline=_OUTLINE, width=2)
    cx, cy = _midpoint(box)
    handle_w = max(4, (x1 - x0) // 4)
    draw.arc((cx - handle_w, y0 - 5, cx + handle_w, y0 + 8), 180, 360, fill=_OUTLINE, width=2)
    if state == "latched":
        draw.rectangle((cx - 3, cy - 2, cx + 3, cy + 4), fill=_OUTLINE)
    elif state == "unlatched":
        draw.arc((cx - 4, cy - 4, cx + 4, cy + 5), 180, 360, fill=_WHITE, width=2)


def _draw_clock(draw: ImageDraw.ImageDraw, box, fill, state: str | None) -> None:
    draw.ellipse(box, fill=fill, outline=_OUTLINE, width=2)
    cx, cy = _midpoint(box)
    radius = max(3, min(box[2] - box[0], box[3] - box[1]) // 3)
    draw.line((cx, cy, cx, cy - radius), fill=_OUTLINE, width=2)
    if state == "running":
        draw.line((cx, cy, cx + radius, cy), fill=_OUTLINE, width=2)
    else:
        draw.line((cx - radius, cy + radius, cx + radius, cy - radius), fill=_WHITE, width=2)


def _draw_glass(draw: ImageDraw.ImageDraw, box, fill, state: str | None) -> None:
    x0, y0, x1, y1 = box
    draw.polygon(((x0 + 2, y0), (x1 - 2, y0), (x1 - 5, y1), (x0 + 5, y1)), fill=fill, outline=_OUTLINE)
    if state == "broken":
        cx, cy = _midpoint(box)
        draw.line((cx, y0 + 2, cx - 3, cy, cx + 3, y1 - 2), fill=_WHITE, width=2)


def _draw_generic(draw: ImageDraw.ImageDraw, obj: SceneObject, box, fill) -> None:
    x0, y0, x1, y1 = box
    kind = obj.kind
    if kind == ObjectKind.PAINTING:
        draw.rectangle(box, fill=fill, outline=_OUTLINE, width=3)
        draw.line((x0 + 4, y1 - 4, (x0 + x1) // 2, y0 + 4, x1 - 4, y1 - 4), fill=_WHITE, width=2)
    elif kind == ObjectKind.CHAIR:
        draw.rectangle((x0 + 3, y0, x1 - 3, (y0 + y1) // 2), fill=fill, outline=_OUTLINE, width=2)
        draw.line((x0 + 5, (y0 + y1) // 2, x0 + 2, y1), fill=_OUTLINE, width=3)
        draw.line((x1 - 5, (y0 + y1) // 2, x1 - 2, y1), fill=_OUTLINE, width=3)
    elif kind == ObjectKind.FOOTPRINT:
        draw.ellipse((x0 + 2, y0, x1 - 2, y1 - 5), fill=fill, outline=_OUTLINE, width=2)
        draw.ellipse((x0 + 4, y0 - 2, x0 + 8, y0 + 3), fill=fill, outline=_OUTLINE)
        draw.ellipse((x1 - 8, y0 - 2, x1 - 4, y0 + 3), fill=fill, outline=_OUTLINE)
    elif kind == ObjectKind.ENVELOPE:
        draw.rectangle(box, fill=fill, outline=_OUTLINE, width=2)
        draw.line((x0, y0, (x0 + x1) // 2, (y0 + y1) // 2, x1, y0), fill=_OUTLINE, width=2)
    else:
        draw.rounded_rectangle(box, radius=3, fill=fill, outline=_OUTLINE, width=2)


def _draw_object(draw: ImageDraw.ImageDraw, obj: SceneObject, canvas_size: int) -> None:
    if obj.visibility == Visibility.HIDDEN:
        return

    box = _box(obj, canvas_size)
    fill = _COLORS.get(obj.color or "", _MUTED)

    if obj.kind == ObjectKind.KEY:
        _draw_key(draw, box, fill)
    elif obj.kind == ObjectKind.LAMP:
        _draw_lamp(draw, box, fill, obj.state)
    elif obj.kind == ObjectKind.WINDOW:
        _draw_openable(draw, box, fill, obj.state, vertical=False)
    elif obj.kind == ObjectKind.DOOR:
        _draw_openable(draw, box, fill, obj.state, vertical=True)
    elif obj.kind == ObjectKind.NOTEBOOK:
        _draw_notebook(draw, box, fill, obj.state)
    elif obj.kind == ObjectKind.BRIEFCASE:
        _draw_briefcase(draw, box, fill, obj.state)
    elif obj.kind == ObjectKind.CLOCK:
        _draw_clock(draw, box, fill, obj.state)
    elif obj.kind == ObjectKind.GLASS:
        _draw_glass(draw, box, fill, obj.state)
    else:
        _draw_generic(draw, obj, box, fill)

    # PARTIAL is encoded as deterministic physical occlusion, not transparency.
    if obj.visibility == Visibility.PARTIAL:
        x0, y0, x1, y1 = box
        draw.rectangle(
            (round(x0 + (x1 - x0) * 0.58), round(y0 + (y1 - y0) * 0.50), x1 + 1, y1 + 1),
            fill=_BACKGROUND,
        )


def render_scene(scene: Scene, *, canvas_size: int = DEFAULT_CANVAS_SIZE) -> Image.Image:
    """Render a scene into a deterministic RGB image.

    ``canvas_size`` is intentionally bounded to keep the benchmark laptop-safe.
    """

    if not (128 <= canvas_size <= 384):
        raise ValueError("canvas_size must be between 128 and 384 pixels")

    image = Image.new("RGB", (canvas_size, canvas_size), _BACKGROUND)
    draw = ImageDraw.Draw(image)

    # Stable object order ensures deterministic overlap semantics.
    for obj in scene.objects:
        _draw_object(draw, obj, canvas_size)

    return image


def render_scene_bytes(scene: Scene, *, canvas_size: int = DEFAULT_CANVAS_SIZE) -> bytes:
    """Render a scene and return deterministic PNG bytes."""

    output = BytesIO()
    render_scene(scene, canvas_size=canvas_size).save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()
