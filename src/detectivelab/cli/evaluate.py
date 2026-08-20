from __future__ import annotations

import argparse
from pathlib import Path

from detectivelab.adapters import DummyAdapter, OllamaAdapter
from detectivelab.evaluation.runner import VALID_CONDITIONS, run_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DetectiveLab benchmark condition.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(VALID_CONDITIONS), type=str.upper, required=True)
    parser.add_argument("--adapter", choices=["dummy", "ollama"], required=True)
    parser.add_argument("--model", help="Model name for model-backed adapters, e.g. qwen3:4b-instruct-2507-q4_K_M")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build_adapter(args: argparse.Namespace):
    if args.adapter == "dummy":
        if args.model:
            raise SystemExit("--model is not used with --adapter dummy")
        return DummyAdapter()

    if args.adapter == "ollama":
        if not args.model:
            raise SystemExit("--model is required with --adapter ollama")
        if args.condition != "QUESTION":
            raise SystemExit(
                "The v0.1 Ollama adapter is intentionally QUESTION-only. "
                "RAW image support has not been promoted yet."
            )
        return OllamaAdapter(
            model=args.model,
            base_url=args.ollama_url,
            temperature=args.temperature,
            num_predict=args.num_predict,
            seed=args.seed,
        )

    raise SystemExit(f"Unsupported adapter: {args.adapter}")


def main() -> int:
    args = _parse_args()
    adapter = _build_adapter(args)

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
