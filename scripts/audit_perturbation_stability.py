from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageFilter

from detectivelab.evaluation.conditional import extracted_target_presence


CANDIDATE_VIEWS = (
    "original",
    "blur_020",
    "blur_040",
    "blur_060",
    "downsample_090",
    "downsample_075",
    "downsample_060",
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _conflict_payload(case_dir: Path) -> dict:
    payloads = _read_json(case_dir / "payloads.json")
    return next(payload for payload in payloads if payload.get("family") == "conflict")


def _iter_cases(benchmark_dir: Path):
    manifest = _read_json(benchmark_dir / "manifest.json")
    for case in manifest["cases"]:
        case_dir = benchmark_dir / case["path"]
        yield case_dir, _conflict_payload(case_dir)


def _save_view(image: Image.Image, view: str, path: Path) -> None:
    if view == "original":
        transformed = image
    elif view.startswith("blur_"):
        radius = int(view.split("_", 1)[1]) / 100
        transformed = image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif view.startswith("downsample_"):
        scale = int(view.split("_", 1)[1]) / 100
        w, h = image.size
        small = image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.Resampling.BILINEAR,
        )
        transformed = small.resize((w, h), Image.Resampling.NEAREST)
    else:
        raise ValueError(f"unknown view: {view}")
    transformed.save(path)


def collect_view_votes(
    *,
    image_path: Path,
    payload: dict,
    views: tuple[str, ...] = CANDIDATE_VIEWS,
) -> dict[str, str]:
    votes: dict[str, str] = {}
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        with tempfile.TemporaryDirectory(prefix="detectivelab_perturb_") as tmp:
            tmp_dir = Path(tmp)
            for idx, view in enumerate(views):
                path = tmp_dir / f"{idx:02d}_{view}.png"
                _save_view(image, view, path)
                presence, _ = extracted_target_presence(
                    image_path=path,
                    payload=payload,
                )
                votes[view] = presence
    return votes


def summarize(benchmark_dir: Path, views: tuple[str, ...]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    per_view = defaultdict(Counter)

    for case_dir, payload in _iter_cases(benchmark_dir):
        votes = collect_view_votes(
            image_path=case_dir / "scene.png",
            payload=payload,
            views=views,
        )
        clean = votes["original"]

        row = {
            "scene_id": payload["scene_id"],
            "item_id": payload["item_id"],
            "clean": clean,
            "votes": votes,
        }
        rows.append(row)

        for view, vote in votes.items():
            per_view[view]["present"] += int(vote == "present")
            per_view[view]["absent"] += int(vote == "absent")
            per_view[view]["agree_clean"] += int(vote == clean)
            per_view[view]["total"] += 1

    return rows, per_view


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit candidate extractor perturbations before using them as uncertainty probes."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--min-clean-agreement",
        type=float,
        default=0.90,
        help="Suggested admissibility threshold for a perturbation view.",
    )
    args = parser.parse_args()

    rows, per_view = summarize(args.benchmark, CANDIDATE_VIEWS)

    print("=== Per-view extractor stability ===")
    for view in CANDIDATE_VIEWS:
        counts = per_view[view]
        agreement = counts["agree_clean"] / counts["total"]
        admissible = view == "original" or agreement >= args.min_clean_agreement
        print(
            f"{view:18s} "
            f"present={counts['present']:2d} "
            f"absent={counts['absent']:2d} "
            f"agree_clean={counts['agree_clean']:2d}/{counts['total']:2d} "
            f"({agreement:.1%}) "
            f"admissible={'yes' if admissible else 'no'}"
        )

    print("\n=== Case-specific disagreements ===")
    found = False
    for row in rows:
        clean = row["clean"]
        disagreements = [
            f"{view}:{vote}"
            for view, vote in row["votes"].items()
            if view != "original" and vote != clean
        ]
        if disagreements:
            found = True
            print(
                f"{row['scene_id']} clean={clean} "
                + " ".join(disagreements)
            )
    if not found:
        print("(none)")

    print("\n=== Suggested admissible views ===")
    kept = []
    for view in CANDIDATE_VIEWS:
        if view == "original":
            kept.append(view)
            continue
        counts = per_view[view]
        agreement = counts["agree_clean"] / counts["total"]
        if agreement >= args.min_clean_agreement:
            kept.append(view)
    print(", ".join(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
