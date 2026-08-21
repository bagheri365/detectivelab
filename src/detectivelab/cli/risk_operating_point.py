from __future__ import annotations

import argparse
import json
from pathlib import Path

from detectivelab.adapters import OllamaAdapter
from detectivelab.evaluation.risk_operating_point import (
    build_risk_records,
    evaluate_all_policies,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate interpretable risk policies on the fixed v0.9 "
            "degradation trajectories."
        )
    )
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--v09", nargs="+", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quality-multiplier", type=float, default=0.90)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    adapter = OllamaAdapter(
        model=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        num_predict=args.num_predict,
        seed=args.seed,
    )

    records, thresholds = build_risk_records(
        benchmark_dir=args.benchmark,
        v09_paths=args.v09,
        adapter=adapter,
        cache_path=args.cache,
        quality_multiplier=args.quality_multiplier,
    )
    results = evaluate_all_policies(records)

    payload = {
        "model": adapter.name,
        "quality_calibration": {
            "multiplier": thresholds.multiplier,
            "clean_contrast_min": thresholds.clean_contrast_min,
            "contrast_floor": thresholds.contrast_floor,
            "clean_edge_min": thresholds.clean_edge_min,
            "edge_floor": thresholds.edge_floor,
        },
        "policies": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Adapter: {adapter.name}")
    print(
        "Quality calibration: "
        f"contrast<{thresholds.contrast_floor:.4f}, "
        f"edge<{thresholds.edge_floor:.4f} "
        f"(clean-min × {thresholds.multiplier:.2f})"
    )
    print()
    print(
        f"{'policy':<20} {'fail-rec':>9} {'fail-prec':>10} "
        f"{'esc-rate':>9} {'call-rate':>10} {'accuracy':>9}"
    )
    for r in results:
        print(
            f"{r['policy']:<20} "
            f"{r['failure_recall']:>8.1%} "
            f"{r['failure_precision']:>9.1%} "
            f"{r['incremental_escalation_rate']:>8.1%} "
            f"{r['model_call_rate']:>9.1%} "
            f"{r['downstream_accuracy']:>8.1%}"
        )
    print()
    print(f"Output: {args.output}")
    print(f"Counterfactual cache: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
