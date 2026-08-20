from __future__ import annotations

import argparse
from pathlib import Path

from detectivelab.adapters import DummyAdapter
from detectivelab.evaluation.runner import VALID_CONDITIONS, run_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DetectiveLab benchmark condition.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(VALID_CONDITIONS), type=str.upper, required=True)
    parser.add_argument("--adapter", choices=["dummy"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    adapter = DummyAdapter()

    result = run_evaluation(
        benchmark_dir=args.benchmark,
        condition=args.condition,
        adapter=adapter,
        output_path=args.output,
    )

    print(f"Condition: {args.condition}")
    print(f"Adapter: {adapter.name}")
    print(f"Records: {result.total_records}")
    print(f"Written this run: {result.written}")
    print(f"Skipped as complete: {result.skipped}")
    print(f"Accuracy: {result.correct}/{result.total_records} ({result.accuracy:.1%})")
    for family, accuracy in result.family_accuracy.items():
        print(f"  {family}: {accuracy:.1%}")
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
