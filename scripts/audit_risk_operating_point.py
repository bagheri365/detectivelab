from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Audit v0.10 operating points and Pareto-efficient policies."
    )
    p.add_argument("result", type=Path)
    args = p.parse_args()

    data = json.loads(args.result.read_text(encoding="utf-8"))
    rows = data["policies"]

    print("policy failure_recall failure_precision escalation model_calls accuracy")
    for r in rows:
        print(
            r["policy"],
            f"{r['failure_recall']:.1%}",
            f"{r['failure_precision']:.1%}",
            f"{r['incremental_escalation_rate']:.1%}",
            f"{r['model_call_rate']:.1%}",
            f"{r['downstream_accuracy']:.1%}",
        )

    efficient = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse_cost = (
                other["model_call_rate"] <= candidate["model_call_rate"]
            )
            no_worse_acc = (
                other["downstream_accuracy"] >= candidate["downstream_accuracy"]
            )
            strictly_better = (
                other["model_call_rate"] < candidate["model_call_rate"]
                or other["downstream_accuracy"] > candidate["downstream_accuracy"]
            )
            if no_worse_cost and no_worse_acc and strictly_better:
                dominated = True
                break
        if not dominated:
            efficient.append(candidate)

    print("\nPareto-efficient by model-call rate vs downstream accuracy:")
    for r in sorted(efficient, key=lambda x: x["model_call_rate"]):
        print(
            f"  {r['policy']}: calls={r['model_call_rate']:.1%} "
            f"accuracy={r['downstream_accuracy']:.1%} "
            f"failure_recall={r['failure_recall']:.1%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
