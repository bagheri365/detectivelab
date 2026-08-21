from __future__ import annotations

import argparse
from pathlib import Path

from detectivelab.adapters import OllamaAdapter
from detectivelab.evaluation.abstaining_gate import (
    PROTECTION_RATES,
    run_abstaining_gate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether abstention rescues catastrophic false-absence "
            "gate errors."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--protection-rate",
        type=float,
        required=True,
        choices=PROTECTION_RATES,
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

    result = run_abstaining_gate(
        benchmark_dir=args.benchmark,
        protection_rate=args.protection_rate,
        adapter=adapter,
        output_path=args.output,
    )

    print(f"Condition: {result.condition}")
    print(f"Adapter: {adapter.name}")
    print(f"Records: {result.total_records}")
    print(f"Written this run: {result.written}")
    print(f"Skipped as complete: {result.skipped}")
    print(
        f"Protected would-be false absences: "
        f"{result.protected_records}/{result.eligible_false_absence_records}"
    )
    print(
        f"Residual false absences: "
        f"{result.residual_false_absence_records}/"
        f"{result.eligible_false_absence_records}"
    )
    print(
        f"Abstentions: {result.abstained_records}/{result.total_records} "
        f"({result.abstention_rate:.1%})"
    )
    print(
        f"Model calls: {result.model_calls}/{result.total_records} "
        f"({result.model_call_rate:.1%})"
    )
    print(
        f"Accuracy: {result.correct}/{result.total_records} "
        f"({result.accuracy:.1%})"
    )
    print("By gold label:")
    for label, accuracy in result.gold_accuracy.items():
        print(f"  {label}: {accuracy:.1%}")
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
