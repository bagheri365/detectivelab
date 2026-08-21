from __future__ import annotations

import argparse
from pathlib import Path

from detectivelab.adapters import OllamaAdapter
from detectivelab.evaluation.robustness import run_paraphrase_robustness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled conflict-testimony paraphrase robustness."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--policy", choices=["staged", "epistemic"], required=True)
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
    result = run_paraphrase_robustness(
        benchmark_dir=args.benchmark,
        policy=args.policy,
        adapter=adapter,
        output_path=args.output,
    )

    print(f"Condition: {result.condition}")
    print(f"Adapter: {adapter.name}")
    print(f"Records: {result.total_records}")
    print(f"Written this run: {result.written}")
    print(f"Skipped as complete: {result.skipped}")
    print(f"Accuracy: {result.correct}/{result.total_records} ({result.accuracy:.1%})")
    print("By paraphrase variant:")
    for variant, accuracy in result.variant_accuracy.items():
        print(f"  {variant}: {accuracy:.1%}")
    print("By gold label:")
    for label, accuracy in result.gold_accuracy.items():
        print(f"  {label}: {accuracy:.1%}")
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
