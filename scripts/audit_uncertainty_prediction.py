from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_relation(items: list[dict]) -> tuple[str, dict | None, dict | None]:
    items = sorted(items, key=lambda r: r["severity_rank"])
    first_warning = next((r for r in items if r["uncertainty_positive"]), None)
    first_failure = next((r for r in items if r["extraction_failed"]), None)

    if first_warning is None and first_failure is None:
        return "neither", None, None
    if first_warning is None:
        return "missed_failure", None, first_failure
    if first_failure is None:
        return "warning_without_failure", first_warning, None
    if first_warning["severity_rank"] < first_failure["severity_rank"]:
        return "warning_before_failure", first_warning, first_failure
    if first_warning["severity_rank"] == first_failure["severity_rank"]:
        return "warning_at_failure", first_warning, first_failure
    return "warning_after_failure", first_warning, first_failure


def _pointwise(rows: list[dict]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for r in rows:
        failure = bool(r["extraction_failed"])
        warning = bool(r["uncertainty_positive"])
        if failure and warning:
            tp += 1
        elif not failure and warning:
            fp += 1
        elif failure and not warning:
            fn += 1
        else:
            tn += 1

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pointwise failure detection and prospective event-level "
            "warning before/at first extraction failure."
        )
    )
    parser.add_argument("jsonl", nargs="+", type=Path)
    args = parser.parse_args()

    all_rows: list[dict] = []
    overall_relations = Counter()

    for path in args.jsonl:
        rows = _load(path)
        all_rows.extend(rows)
        family = rows[0]["degradation_family"] if rows else path.stem

        grouped = defaultdict(list)
        for row in rows:
            grouped[row["item_id"]].append(row)

        relations = Counter()
        print(f"\n=== {family} ===")
        print("item_id earliest_uncertainty earliest_failure relation")

        for item_id, items in sorted(grouped.items()):
            relation, warning, failure = _event_relation(items)
            relations[relation] += 1
            overall_relations[relation] += 1
            print(
                item_id,
                None if warning is None else warning["severity"],
                None if failure is None else failure["severity"],
                relation,
            )

        failing_items = (
            relations["warning_before_failure"]
            + relations["warning_at_failure"]
            + relations["warning_after_failure"]
            + relations["missed_failure"]
        )
        timely = (
            relations["warning_before_failure"]
            + relations["warning_at_failure"]
        )
        warning_items = (
            relations["warning_before_failure"]
            + relations["warning_at_failure"]
            + relations["warning_after_failure"]
            + relations["warning_without_failure"]
        )

        event_recall = timely / failing_items if failing_items else 0.0
        event_precision = timely / warning_items if warning_items else 0.0

        point = _pointwise(rows)
        print("\nPointwise:")
        print(
            f"  TP={point['tp']} FP={point['fp']} FN={point['fn']} TN={point['tn']} "
            f"recall={point['recall']:.1%} precision={point['precision']:.1%}"
        )
        print("Prospective event-level:")
        print(f"  failing items={failing_items}")
        print(f"  timely warnings={timely}")
        print(f"  warning-only/late items={warning_items - timely}")
        print(f"  event recall={event_recall:.1%}")
        print(f"  event precision={event_precision:.1%}")

    if len(args.jsonl) > 1:
        failing = (
            overall_relations["warning_before_failure"]
            + overall_relations["warning_at_failure"]
            + overall_relations["warning_after_failure"]
            + overall_relations["missed_failure"]
        )
        timely = (
            overall_relations["warning_before_failure"]
            + overall_relations["warning_at_failure"]
        )
        warnings = (
            overall_relations["warning_before_failure"]
            + overall_relations["warning_at_failure"]
            + overall_relations["warning_after_failure"]
            + overall_relations["warning_without_failure"]
        )
        print("\n=== Overall event-level ===")
        print(f"failing item-family events={failing}")
        print(f"timely warnings={timely}")
        print(f"event recall={timely / failing if failing else 0.0:.1%}")
        print(f"event precision={timely / warnings if warnings else 0.0:.1%}")
        print("relations:", dict(sorted(overall_relations.items())))

        point = _pointwise(all_rows)
        print("\n=== Overall pointwise ===")
        print(
            f"TP={point['tp']} FP={point['fp']} FN={point['fn']} TN={point['tn']} "
            f"recall={point['recall']:.1%} precision={point['precision']:.1%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
