## v0.7 Abstaining Gate

`v0.7-abstaining-gate` tests whether an abstention path can rescue the catastrophic false-absence errors identified in `v0.6`.

The gate now permits:

```text
present
absent
uncertain
```

with:

```text
present   -> staged reasoning
absent    -> deterministic unknown
uncertain -> staged reasoning
```

The experiment starts from the 100% false-absence stress condition and converts deterministic subsets of would-be false absences to `uncertain`.

### Accuracy / compute tradeoff

Gemma and Qwen produce the same curve:

| Protection | Residual false absences | Model-call rate | Accuracy |
| ---: | ---: | ---: | ---: |
| 0% | 6/6 | 0% | 40% |
| 25% | 4/6 | 20% | 60% |
| 50% | 3/6 | 30% | 70% |
| 75% | 1/6 | 50% | 90% |
| 100% | 0/6 | 60% | 100% |

The central result is:

> **Abstention converts catastrophic false-negative gate errors into recoverable reasoning calls.**

At full protection, the system restores 100% accuracy while still using zero model calls on the four genuinely absent cases.

This is not equivalent to always invoking the reasoner.

The important limitation is that `v0.7` uses controlled protection. It measures the **value of abstention**, not the quality of an uncertainty estimator.

See [`docs/results/v0_7_abstaining_gate.md`](./docs/results/v0_7_abstaining_gate.md).

### Next research question

> Can uncertainty be derived from the extractor itself rather than supplied by controlled oracle protection?

The next step should remain evidence-grounded and interpretable before any learned routing is introduced.
