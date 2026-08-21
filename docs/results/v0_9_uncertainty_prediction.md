# v0.9 — Uncertainty Prediction

Branch: `v0.9-uncertainty-prediction`

## Research question

Does calibrated extractor instability predict actual extraction failure under controlled image degradation?

`v0.8` established that calibrated perturbation disagreement can identify localized clean-image instability and support an evidence-derived abstention path.

`v0.9` tests whether that signal is prospectively useful when the image itself is degraded.

The `v0.8` uncertainty rule is held fixed.

No perturbation set is retuned on the `v0.9` degradation results.

## Controlled degradation families

Four degradation families were applied at five severity levels each:

- blur
- downsampling
- contrast reduction
- centered occlusion

For every scene × severity, the harness records:

- clean extractor presence
- degraded extractor presence
- whether extraction actually failed
- calibrated uncertainty state
- whether uncertainty fired
- downstream verdict correctness
- model-call count

## Why two evaluation views are necessary

A pointwise classifier metric treats each severity independently.

That creates a problem for prospective warnings:

```text
warning at severity 1
failure at severity 3
```

At the record level:

```text
severity 1 -> false positive
severity 3 -> false negative
```

But prospectively, the earlier warning may be exactly the desired behavior.

Therefore `v0.9` reports both:

1. **pointwise detection**
2. **event-level prospective warning**

The event-level audit asks whether the first uncertainty warning occurs before or at the first extraction failure for each item/degradation-family trajectory.

## Gemma results

Model:

```text
gemma3:4b
```

Current test status:

```text
120 passed
```

### Blur

50 records.

```text
Extraction failures: 4/50
Uncertainty positives: 5/50
Pointwise TP=1 FP=4 FN=3 TN=42
Pointwise recall=25.0%
Pointwise precision=20.0%
Downstream accuracy=94.0%
Model-call rate=54.0%
```

Event-level:

```text
failing items=2
timely warnings=1
event recall=50.0%
event precision=33.3%
```

Notable cases:

```text
scene_0002
uncertainty at 0.0
failure at 0.8
warning_before_failure

scene_0005
no uncertainty
failure at 1.6
missed_failure
```

### Downsampling

50 records.

```text
Extraction failures: 4/50
Uncertainty positives: 2/50
Pointwise TP=0 FP=2 FN=4 TN=44
Pointwise recall=0.0%
Pointwise precision=0.0%
Downstream accuracy=92.0%
Model-call rate=52.0%
```

Event-level:

```text
failing items=4
timely warnings=1
event recall=25.0%
event precision=100.0%
```

Three failing items receive no warning.

### Contrast reduction

50 records.

```text
Extraction failures: 24/50
Uncertainty positives: 1/50
Pointwise TP=0 FP=1 FN=24 TN=25
Pointwise recall=0.0%
Pointwise precision=0.0%
Downstream accuracy=52.0%
Model-call rate=12.0%
```

Event-level:

```text
failing items=6
timely warnings=1
event recall=16.7%
event precision=100.0%
```

Contrast is the strongest failure mode in this milestone.

The extractor fails frequently, but the calibrated instability signal almost never fires.

This produces a severe downstream accuracy collapse.

### Occlusion

50 records.

```text
Extraction failures: 3/50
Uncertainty positives: 5/50
Pointwise TP=0 FP=5 FN=3 TN=42
Pointwise recall=0.0%
Pointwise precision=0.0%
Downstream accuracy=94.0%
Model-call rate=54.0%
```

Event-level:

```text
failing items=2
timely warnings=2
event recall=100.0%
event precision=100.0%
```

Occlusion is the strongest positive result.

Both failing item trajectories receive uncertainty warnings before failure.

## Overall detection result

Across all four degradation families:

```text
failing item-family events=14
timely warnings=5
missed failures=9
```

Event-level:

| Metric | Result |
| --- | ---: |
| Recall | 35.7% |
| Precision | 71.4% |

Pointwise:

| Metric | Result |
| --- | ---: |
| TP | 1 |
| FP | 12 |
| FN | 34 |
| TN | 153 |
| Recall | 2.9% |
| Precision | 7.7% |

The event-level metric is the more appropriate prospective measure, but even there the signal misses most failures.

## Main finding

> **Calibrated clean-image instability is not a reliable general predictor of extractor failure under distribution shift.**

It has predictive value for some degradation modes, but that value does not generalize.

The clearest contrast is:

```text
occlusion:
  event recall = 100%

contrast:
  event recall = 16.7%
```

This indicates that the uncertainty signal is degradation-specific rather than universally predictive.

## Interpretation

The `v0.8` perturbation ensemble probes a particular set of extractor invariances:

- blur tolerance
- resolution tolerance

That can reveal fragility related to similar local perturbations.

It does not necessarily probe:

- contrast sensitivity
- all forms of occlusion
- every extractor decision boundary
- arbitrary distribution shift

The resulting methodological lesson is:

> **Uncertainty probes inherit assumptions about the failure modes they perturb.**

A calibrated perturbation ensemble may be a valid measurement instrument for one class of fragility while remaining blind to another.

## Relationship to v0.8

`v0.8` established:

> calibrated perturbation disagreement can provide a localized, interpretable evidence-derived uncertainty signal on the clean benchmark.

`v0.9` establishes the boundary:

> descriptive instability does not automatically imply general predictive uncertainty.

This does not invalidate `v0.8`.

It clarifies what the signal means.

The current perturbation-based uncertainty measure is best interpreted as:

```text
local extractor fragility under a specific family of perturbations
```

not:

```text
general probability that extraction is wrong
```

## Why the signal should not be retuned in v0.9

The `v0.9` degradation sweep is prospective validation.

Retuning the `v0.8` perturbation set on these same degradation outcomes would convert the evaluation set into a tuning set.

Therefore the correct scientific response is to preserve the failure result.

The weak predictive performance is evidence about the limit of the current signal.

## Why a second language model is not required for the core conclusion

The uncertainty detector and extraction failures occur upstream of the language model.

The main `v0.9` question is therefore model-independent:

```text
does uncertainty predict extraction failure?
```

A different downstream reasoner may change final verdict accuracy, but it cannot change the detector's event recall.

For that reason, the Gemma degradation sweep is sufficient to establish the central uncertainty-prediction result.

## Architectural implication

The previous milestones progressively removed hidden assumptions:

```text
v0.5
reliable structured gate works

v0.6
gate error direction matters

v0.7
abstention can rescue risky gate errors

v0.8
uncertainty can be derived from evidence stability

v0.9
that uncertainty is not a general failure predictor
```

The resulting principle is:

> **A useful uncertainty mechanism must cover the failure modes that matter operationally; no single perturbation probe should be assumed to measure generic uncertainty.**

## Recommended final milestone

A narrow final milestone should ask:

> **Can multiple complementary, interpretable evidence-risk signals produce a useful operating point without learned routing?**

The goal should not be to maximize accuracy by adding arbitrary detectors.

Instead, combine a small number of clearly motivated signals such as:

- perturbation instability
- raw extractor score or match margin
- competing candidate evidence
- threshold proximity
- simple image-quality indicators

Then measure:

```text
failure recall
failure precision
false-negative rate
abstention / escalation rate
model-call rate
downstream accuracy
```

The final question is operational:

> How much failure coverage can be gained for how much escalation cost?

Only after that comparison would a learned confidence model or router be justified.
