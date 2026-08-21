# v0.8 — Evidence-Derived Uncertainty

Branch: `v0.8-evidence-uncertainty`

## Research question

Can uncertainty be derived from extractor evidence itself rather than supplied by controlled oracle protection?

`v0.7` showed that abstention can convert catastrophic false-absence gate errors into recoverable reasoning calls. However, the protection signal in `v0.7` was supplied by the experiment.

`v0.8` removes that oracle protection and asks whether uncertainty can be estimated directly from the extractor.

## Initial approach: perturbation stability

The first uncertainty signal was based on repeated extraction across deterministic image views.

Initial views:

- original
- brightness 0.90
- brightness 1.10
- Gaussian blur 0.60
- downsample 0.75

The gate used a conservative unanimity rule:

```text
all present  -> present
all absent   -> absent
disagreement -> uncertain
```

Routing remained:

```text
present   -> staged reasoning
absent    -> deterministic unknown
uncertain -> staged reasoning
```

## Initial result

Both models produced:

| Model | Hard present | Hard absent | Uncertain | Model-call rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 4B | 0/10 | 4/10 | 6/10 | 60% | 100% |
| Qwen3 4B | 0/10 | 4/10 | 6/10 | 60% | 100% |

At first glance this reproduced the `v0.7` full-protection operating point.

However, the perturbation audit showed that this result was misleading.

## Perturbation audit

Per-view extractor behavior:

| View | Present | Absent | Agreement with original |
| --- | ---: | ---: | ---: |
| original | 6 | 4 | 100% |
| brightness 0.90 | 0 | 10 | 40% |
| brightness 1.10 | 0 | 10 | 40% |
| blur 0.60 | 5 | 5 | 90% |
| downsample 0.75 | 6 | 4 | 100% |

Both brightness views caused catastrophic extractor collapse on every present case.

The six `uncertain` decisions were therefore not evidence of nuanced case-specific ambiguity. They were largely induced by globally destructive perturbations.

This creates an important negative result:

> **Naive perturbation disagreement is not automatically epistemic uncertainty. A perturbation can measure extractor non-invariance rather than ambiguity.**

## Calibrating the perturbation probes

A dedicated calibration audit evaluated a wider set of candidate views before allowing them into the uncertainty ensemble:

- original
- blur 0.20
- blur 0.40
- blur 0.60
- downsample 0.90
- downsample 0.75
- downsample 0.60

Results:

| View | Present | Absent | Clean agreement | Admissible at 90% |
| --- | ---: | ---: | ---: | --- |
| original | 6 | 4 | 100% | yes |
| blur 0.20 | 5 | 5 | 90% | yes |
| blur 0.40 | 6 | 4 | 100% | yes |
| blur 0.60 | 5 | 5 | 90% | yes |
| downsample 0.90 | 6 | 4 | 100% | yes |
| downsample 0.75 | 6 | 4 | 100% | yes |
| downsample 0.60 | 6 | 4 | 100% | yes |

Only one case showed disagreement:

```text
scene_0002
clean=present
blur_020=absent
blur_060=absent
```

This is qualitatively different from the brightness failure: instability is localized rather than universal.

## Calibrated evidence uncertainty

The final calibrated ensemble uses:

```text
original
blur_020
blur_040
blur_060
downsample_090
downsample_075
downsample_060
```

Brightness transforms are excluded.

The same conservative rule is retained:

```text
unanimous present -> hard present
unanimous absent  -> hard absent
any disagreement  -> uncertain
```

Condition:

```text
CONFLICT_EVIDENCE_UNCERTAINTY_CALIBRATED
```

## Final result

Both models produce the same gate distribution and accuracy:

| Model | Hard present | Hard absent | Uncertain | Model-call rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 4B | 5/10 | 4/10 | 1/10 | 60% | 100% |
| Qwen3 4B | 5/10 | 4/10 | 1/10 | 60% | 100% |

Label accuracy:

- supported: 100%
- contradicted: 100%
- unknown: 100%

Current test status:

```text
114 passed
```

## Main finding

> **An interpretable uncertainty signal can be derived from extractor stability, but only after the perturbation probes themselves are calibrated against extractor invariance.**

The calibrated system separates the benchmark into:

- 5 stable present cases
- 4 stable absent cases
- 1 unstable case

The unstable case is escalated through the abstention path and solved correctly.

## Efficiency interpretation

The model-call rate remains 60%, but the meaning of those calls is different from the initial diagnostic ensemble.

Final routing:

```text
5 hard present -> staged reasoning
4 hard absent  -> deterministic unknown
1 uncertain    -> staged reasoning
```

Only 1/10 cases incurs uncertainty-specific escalation.

The other five model calls are normal present-target reasoning.

Therefore the uncertainty overhead is 10% of cases, not 60%.

## Relationship to v0.7

`v0.7` demonstrated:

> If risky false-absence decisions can be identified, abstention can restore accuracy with a controlled compute cost.

`v0.8` demonstrates:

> A simple evidence-derived instability signal can identify a localized risky case on the current benchmark, provided the perturbation probes are first validated.

This moves the architecture from controlled oracle protection to an evidence-grounded abstention signal.

## Methodological lesson

The failed brightness ensemble is part of the result and should not be discarded.

It demonstrates:

> **Uncertainty estimation can itself become a source of spurious evidence if the probe changes the extractor's operating regime.**

Perturbation-based uncertainty therefore requires a calibration stage:

```text
candidate perturbations
-> measure extractor invariance
-> reject globally destructive probes
-> retain locally discriminative probes
-> construct uncertainty signal
```

This mirrors the broader DetectiveLab principle:

> evidence before complexity

The uncertainty estimator should itself be validated as a measurement instrument before being trusted as a control signal.

## What v0.8 does not establish

The benchmark remains small and synthetic.

The result does not establish that:

- perturbation stability is calibrated probability
- the 90% admissibility threshold is optimal
- the same views transfer to natural images
- disagreement always corresponds to semantic ambiguity
- `scene_0002` is intrinsically ambiguous rather than simply near an extractor decision boundary
- learned confidence or routing is necessary

The current signal is interpretable and useful, but not probabilistically calibrated.

## Recommended next step

The next milestone should test whether the evidence-derived uncertainty signal generalizes beyond this single clean benchmark slice.

A narrow next question is:

> **Does calibrated extractor instability predict actual extraction failure under controlled image degradation?**

That would move from clean-image stability to prospective validation:

1. degrade images at controlled severity levels;
2. measure when the extractor's clean decision becomes wrong;
3. test whether the calibrated instability signal rises before or at those failures;
4. evaluate precision/recall of abstention as a failure detector.

This would test whether the current uncertainty signal is predictive rather than merely descriptive.

No learned router is justified yet.
