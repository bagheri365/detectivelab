# v0.10 — Risk Operating Point

Branch: `v0.10-risk-operating-point`

## Research question

> **Can multiple complementary, interpretable evidence-risk signals produce a useful failure-coverage versus escalation-cost operating point without learned routing?**

`v0.9` established that the fixed `v0.8` perturbation-stability signal is not a reliable general predictor of extractor failure across degradation types.

`v0.10` asks the final operational question:

> If risk detection is broadened with complementary interpretable signals, does escalation actually improve the system?

No learned router is introduced.

## Signals

The milestone evaluates three simple evidence-risk signals:

```text
perturbation instability
low global contrast
low local edge strength
```

The image-quality thresholds are calibrated from clean benchmark images only.

No degraded extraction-failure labels are used to set the thresholds.

Calibration:

```text
contrast floor = 33.3471
edge floor     = 1.4261
rule           = clean minimum × 0.90
```

## Policies

Eight fixed policies are compared:

```text
NEVER_ESCALATE
STABILITY_ONLY
LOW_CONTRAST_ONLY
LOW_EDGE_ONLY
QUALITY_ANY
ANY_SIGNAL
TWO_PLUS
ALWAYS_ESCALATE
```

The evaluation reuses the frozen `v0.9` degradation trajectories.

Where `v0.9` originally took a deterministic zero-model-call path, `v0.10` obtains one counterfactual staged-model prediction and caches it. All policies are then compared against the same cached fallback outputs.

## Test status

```text
125 passed
```

## Gemma operating-point results

Model:

```text
gemma3:4b
```

| Policy | Failure recall | Failure precision | Incremental escalation | Model-call rate | Downstream accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| NEVER_ESCALATE | 0.0% | 0.0% | 0.0% | 43.0% | **83.0%** |
| STABILITY_ONLY | 2.9% | 7.7% | 0.0% | 43.0% | **83.0%** |
| LOW_CONTRAST_ONLY | 51.4% | 58.1% | 14.5% | 57.5% | 80.5% |
| LOW_EDGE_ONLY | 45.7% | 44.4% | 14.0% | 57.0% | 80.5% |
| QUALITY_ANY | 57.1% | 46.5% | 17.5% | 60.5% | 79.5% |
| ANY_SIGNAL | **60.0%** | 38.2% | 17.5% | 60.5% | 79.5% |
| TWO_PLUS | 40.0% | 58.3% | 11.0% | 54.0% | 81.5% |
| ALWAYS_ESCALATE | **100.0%** | 17.5% | 57.0% | 100.0% | **59.0%** |

## Pareto result

The audit identifies only two accuracy-versus-model-call Pareto-efficient policies:

```text
NEVER_ESCALATE
STABILITY_ONLY
```

Both operate at:

```text
model-call rate = 43.0%
accuracy        = 83.0%
```

`STABILITY_ONLY` detects one additional failure but does not change the actual compute or accuracy operating point because the base gate already sends unstable cases to the reasoner.

Every policy that increases model calls is dominated on the current accuracy-versus-compute objective.

## Main finding

> **Better failure detection does not automatically produce better downstream performance when the escalation path is itself unreliable under degraded evidence.**

The clearest example is `ANY_SIGNAL`.

Failure recall improves:

```text
2.9% → 60.0%
```

but model-call rate also increases:

```text
43.0% → 60.5%
```

while downstream accuracy falls:

```text
83.0% → 79.5%
```

The `ALWAYS_ESCALATE` baseline makes the limitation explicit:

```text
failure recall = 100%
model calls    = 100%
accuracy       = 59%
```

Perfect failure coverage produces the worst downstream accuracy of all tested policies.

## Interpretation

The earlier architecture implicitly assumed:

```text
detect risk
→ escalate
→ recover
```

`v0.10` shows that the second implication is not guaranteed.

A useful escalation policy requires both:

```text
1. the current path is risky
2. the fallback path has positive expected value
```

Risk detection alone is therefore insufficient.

The more appropriate target is not simply:

```text
failure recall
```

but something closer to:

```text
recoverable-risk recall
```

or:

```text
expected value of escalation
```

## Relationship to previous milestones

The final experimental arc is:

```text
v0.1
perception can masquerade as reasoning failure

v0.2
focused structure can outperform dense correct structure

v0.3
residual failure can be epistemic policy

v0.4
policy interventions can be model-dependent

v0.5
control works better from reliable representation than re-inference

v0.6
gate errors are directionally asymmetric

v0.7
abstention can rescue dangerous gate errors

v0.8
evidence-derived instability can trigger abstention

v0.9
instability is not a general failure predictor

v0.10
risk detection alone does not justify escalation
```

## Final project conclusion

> **Reliable multimodal control requires not only detecting when the current evidence path is risky, but also knowing whether the fallback path is likely to recover rather than amplify that risk.**

A compact operational version is:

> **Uncertainty is useful only when escalation has positive expected value.**

## Why the project stops here

`v0.10` answers the operating-point question without requiring a learned router.

The interpretable risk signals do increase failure coverage, but none produces a superior accuracy-compute operating point because the fallback reasoner is itself fragile under degraded evidence.

A learned router would therefore not yet solve the identified systems problem.

Before routing could be justified, the system would first need a fallback whose recovery probability is demonstrably higher on the cases being escalated.

For the current research scope, that is a natural stopping point.

No `v0.11` milestone is planned.
