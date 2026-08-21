from __future__ import annotations

import argparse
from pathlib import Path

from detectivelab.adapters import OllamaAdapter
from detectivelab.evaluation.uncertainty_prediction import (
    DEGRADATION_GRID,
    run_uncertainty_prediction,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prospectively test whether v0.8 uncertainty predicts extraction failure."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--degradation",
        choices=sorted(DEGRADATION_GRID),
        required=True,
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
    adapter = OllamaAdapter(
        model=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        seed=args.seed,
    )

    result = run_uncertainty_prediction(
        benchmark_dir=args.benchmark,
        degradation_family=args.degradation,
        adapter=adapter,
        output_path=args.output,
    )

    print(f"Adapter: {adapter.name}")
    print(f"Degradation: {args.degradation}")
    print(f"Records: {result.total_records}")
    print(f"Written this run: {result.written}")
    print(f"Skipped as complete: {result.skipped}")
    print(f"Extraction failures: {result.extraction_failures}/{result.total_records}")
    print(
        f"Uncertainty positives: "
        f"{result.uncertainty_positive}/{result.total_records}"
    )
    print(
        f"Failure detection: TP={result.true_positive} "
        f"FP={result.false_positive} FN={result.false_negative} "
        f"TN={result.true_negative}"
    )
    print(f"Failure recall: {result.failure_recall:.1%}")
    print(f"Failure precision: {result.failure_precision:.1%}")
    print(f"False-negative rate: {result.false_negative_rate:.1%}")
    print(
        f"Downstream accuracy: "
        f"{result.downstream_correct}/{result.total_records} "
        f"({result.downstream_accuracy:.1%})"
    )
    print(
        f"Model calls: {result.model_calls}/{result.total_records} "
        f"({result.model_call_rate:.1%})"
    )
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
