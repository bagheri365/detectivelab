"""Deterministic image-only extractor for DetectiveLab's synthetic renderer.

This extractor is intentionally benchmark-specific. It reverses the renderer's
small visual grammar using connected components and template matching. Runtime
input is *only* ``scene.png``: no scene JSON, seeds, object IDs, provenance, or
gold labels are consulted.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from detectivelab.domain.schema import ObjectKind, Point, SceneObject, Size, Visibility
from detectivelab.rendering.renderer import _draw_object

from .base import ExtractedObject

_BACKGROUND = (244, 241, 233)
_OUTLINE = (30, 30, 30)
_WHITE = (255, 255, 255)
_COLORS = {
    "blue": (70, 120, 190),
    "red": (190, 75, 70),
    "green": (80, 150, 100),
    "amber": (210, 150, 55),
    "black": (45, 45, 45),
    "white": (235, 235, 235),
}
_SIZES = {
    ObjectKind.KEY: Size(0.08, 0.04),
    ObjectKind.NOTEBOOK: Size(0.14, 0.10),
    ObjectKind.GLASS: Size(0.07, 0.11),
    ObjectKind.LAMP: Size(0.12, 0.20),
    ObjectKind.CLOCK: Size(0.11, 0.11),
    ObjectKind.BRIEFCASE: Size(0.17, 0.11),
    ObjectKind.PAINTING: Size(0.18, 0.16),
    ObjectKind.CHAIR: Size(0.16, 0.20),
    ObjectKind.DOOR: Size(0.17, 0.34),
    ObjectKind.WINDOW: Size(0.20, 0.22),
    ObjectKind.FOOTPRINT: Size(0.08, 0.12),
    ObjectKind.ENVELOPE: Size(0.12, 0.07),
}
_STATES = {
    ObjectKind.DOOR: ("open", "closed"),
    ObjectKind.WINDOW: ("open", "closed"),
    ObjectKind.NOTEBOOK: ("open", "closed"),
    ObjectKind.GLASS: ("intact", "broken"),
}


def _dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _is_foreground(pixel: tuple[int, int, int]) -> bool:
    return _dist(pixel, _BACKGROUND) > 9


def _raw_components(image: Image.Image) -> list[tuple[int, int, int, int, list[tuple[int, int]]]]:
    image = image.convert("RGB")
    w, h = image.size
    px = image.load()
    seen: set[tuple[int, int]] = set()
    comps = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or not _is_foreground(px[x, y]):
                continue
            q = deque([(x, y)])
            seen.add((x, y))
            pts: list[tuple[int, int]] = []
            while q:
                cx, cy = q.popleft()
                pts.append((cx, cy))
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        if (nx, ny) in seen or not _is_foreground(px[nx, ny]):
                            continue
                        seen.add((nx, ny))
                        q.append((nx, ny))
            if len(pts) >= 8:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                comps.append((min(xs), min(ys), max(xs), max(ys), pts))
    return comps


def _boxes_near(a, b, margin: int = 2) -> bool:
    ax0, ay0, ax1, ay1, _ = a
    bx0, by0, bx1, by1, _ = b
    return not (
        ax1 + margin < bx0 or bx1 + margin < ax0 or
        ay1 + margin < by0 or by1 + margin < ay0
    )


def _components(image: Image.Image):
    """Merge renderer subcomponents such as frames/panels and handles."""
    comps = [list(c) for c in _raw_components(image)]
    changed = True
    while changed:
        changed = False
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                if not _boxes_near(comps[i], comps[j]):
                    continue
                a, b = comps[i], comps[j]
                merged = [
                    min(a[0], b[0]), min(a[1], b[1]),
                    max(a[2], b[2]), max(a[3], b[3]),
                    a[4] + b[4],
                ]
                comps[i] = merged
                del comps[j]
                changed = True
                break
            if changed:
                break
    return comps


def _classify_color(image: Image.Image, pts: list[tuple[int, int]]) -> str:
    px = image.load()
    counts = {name: 0 for name in _COLORS}
    for x, y in pts:
        pixel = px[x, y]
        if _dist(pixel, _OUTLINE) < 64 or _dist(pixel, _WHITE) < 64:
            continue
        name = min(_COLORS, key=lambda n: _dist(pixel, _COLORS[n]))
        if _dist(pixel, _COLORS[name]) < 144:
            counts[name] += 1
    return max(counts, key=counts.get)


def _crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def _normalized_pixels(image: Image.Image, size: int = 48) -> list[tuple[int, int, int]]:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    data = resized.tobytes()
    return [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data), 3)]


def _image_score(a: Image.Image, b: Image.Image) -> float:
    ap = _normalized_pixels(a)
    bp = _normalized_pixels(b)
    return sum(_dist(x, y) for x, y in zip(ap, bp)) / len(ap)


def _candidate_states(kind: ObjectKind):
    return _STATES.get(kind, (None,))


@lru_cache(maxsize=None)
def _prototype(kind_value: str, state: str | None, color: str) -> Image.Image:
    kind = ObjectKind(kind_value)
    canvas = Image.new("RGB", (256, 256), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    obj = SceneObject(
        object_id="prototype",
        kind=kind,
        color=color,
        state=state,
        position=Point(0.5, 0.5),
        size=_SIZES[kind],
        visibility=Visibility.VISIBLE,
        orientation_deg=0,
    )
    _draw_object(draw, obj, 256)
    comps = _components(canvas)
    if len(comps) != 1:
        raise AssertionError(f"prototype {kind_value}/{state} split into {len(comps)} components")
    x0, y0, x1, y1, _ = comps[0]
    return _crop(canvas, (x0, y0, x1, y1))


def _classify_object(image: Image.Image, comp) -> tuple[str, str | None, str]:
    x0, y0, x1, y1, pts = comp
    observed = _crop(image, (x0, y0, x1, y1))
    color = _classify_color(image, pts)
    candidates: list[tuple[float, str, str | None]] = []
    for kind in ObjectKind:
        for state in _candidate_states(kind):
            proto = _prototype(kind.value, state, color)
            # Shape mismatch is useful in addition to resized pixel mismatch.
            ow, oh = observed.size
            pw, ph = proto.size
            shape_penalty = 2500 * (abs(ow - pw) / pw + abs(oh - ph) / ph)
            score = _image_score(observed, proto) + shape_penalty
            candidates.append((score, kind.value, state))
    _, kind, state = min(candidates, key=lambda row: row[0])
    return kind, state, color


def extract_scene_facts(image_path: Path) -> list[ExtractedObject]:
    image = Image.open(image_path).convert("RGB")
    objects: list[ExtractedObject] = []
    for x0, y0, x1, y1, pts in _components(image):
        kind, state, color = _classify_object(image, (x0, y0, x1, y1, pts))
        objects.append(
            ExtractedObject(
                kind=kind,
                color=color,
                state=state,
                center_x=((x0 + x1) / 2) / image.width,
                center_y=((y0 + y1) / 2) / image.height,
            )
        )
    return sorted(objects, key=lambda obj: (obj.center_y, obj.center_x, obj.label))


def extract_structured_evidence(image_path: Path) -> str:
    objects = extract_scene_facts(image_path)
    lines = ["Visible scene evidence:"]
    for obj in objects:
        value = obj.state if obj.state is not None else "present"
        lines.append(f"- {obj.label}: {value}")

    # Reconstruct pairwise order from image-space centers. This intentionally
    # supplies more relations than the frozen benchmark requires, but all are
    # direct consequences of the extracted geometry.
    if len(objects) >= 2:
        lines.append("Spatial relations:")
        for i, first in enumerate(objects):
            for second in objects[i + 1:]:
                if first.center_x < second.center_x:
                    lines.append(f"- {first.label} is left of {second.label}")
                elif first.center_x > second.center_x:
                    lines.append(f"- {first.label} is right of {second.label}")

    return "\n".join(lines)
