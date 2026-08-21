from __future__ import annotations

import argparse
from pathlib import Path

from detectivelab.adapters import OllamaAdapter
from detectivelab.evaluation.gate_corruption import (
    GATE_CORRUPTIONS,
    GATE_CORRUPTION_RATES,
    run_gate_corruption,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure controlled corruption of the extractor-derived conflict gate."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--corruption", choices=sorted(GATE_CORRUPTIONS), required=True)
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Fraction of eligible gate values to flip. Typical values: 0.25, 0.5, 0.75, 1.0.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.corruption != "clean" and args.rate not in GATE_CORRUPTION_RATES:
        raise SystemExit(
            f"--rate must be one of {', '.join(str(x) for x in GATE_CORRUPTION_RATES)}"
        )

    adapter = OllamaAdapter(
        model=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        seed=args.seed,
    )
    result = run_gate_corruption(
        benchmark_dir=args.benchmark,
        corruption=args.corruption,
        corruption_rate=0.0 if args.corruption == "clean" else args.rate,
        adapter=adapter,
        output_path=args.output,
    )

    print(f"Condition: {result.condition}")
    print(f"Adapter: {adapter.name}")
    print(f"Records: {result.total_records}")
    print(f"Written this run: {result.written}")
    print(f"Skipped as complete: {result.skipped}")
    print(f"Eligible gate values: {result.eligible_records}")
    print(
        f"Gate values actually flipped: "
        f"{result.corrupted_records}/{result.eligible_records} eligible "
        f"({result.corruption_rate:.0%} requested)"
    )
    print(f"Accuracy: {result.correct}/{result.total_records} ({result.accuracy:.1%})")
    print("By gold label:")
    for label, accuracy in result.gold_accuracy.items():
        print(f"  {label}: {accuracy:.1%}")
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
