# DetectiveLab v0.4 Epistemic Robustness — Results Note

## Status

**Milestone:** `v0.4-epistemic-robustness`  
**Benchmark base:** `v0.0.1`  
**Primary models:** `gemma3:4b`, `qwen3:4b-instruct-2507-q4_K_M`  
**Robustness axes:** testimony paraphrase, case variation, model variation

This milestone tests whether the explicit epistemic rule discovered in `v0.3` remains useful when wording, cases, and model choice vary.

The rule under test is:

```text
if EXISTENCE = absent:
  PHYSICAL_STATE = not_applicable
  AGREEMENT = unknown
  VERDICT = unknown
```

The intended semantics are:

> Missing physical evidence is insufficient evidence, not contradictory evidence.

## Research Question

Does the explicit “missing evidence → unknown” policy remain effective when surface form, conflict cases, and model choice change?

This milestone deliberately avoids new architecture.

It compares the same two reasoning policies:

- `CONFLICT_STAGED`
- `CONFLICT_EPISTEMIC`

while varying only:

1. witness wording
2. conflict case construction
3. model identity

## Robustness Axis 1 — Testimony Paraphrases

Each of the 10 canonical conflict cases is expressed with three deterministic paraphrase variants:

- `according_to`
- `claim_is`
- `reports_now`

This produces 30 records per policy without changing benchmark semantics.

### Gemma 3 4B

| Policy | Overall | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| staged paraphrases | 66.7% | 100.0% | 100.0% | 16.7% |
| epistemic paraphrases | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

By paraphrase variant:

| Variant | Staged | Epistemic |
| --- | ---: | ---: |
| according_to | 80.0% | 100.0% |
| claim_is | 60.0% | 100.0% |
| reports_now | 60.0% | 100.0% |

### Interpretation

The staged baseline remains highly sensitive to `unknown` cases under wording variation.

The explicit epistemic rule completely stabilizes Gemma across all three paraphrase forms.

This supports:

> The original missing-evidence failure is not tied to one exact witness wording.

## Robustness Axis 2 — Controlled Case Variation

A dedicated image-only case-variation harness creates three deterministic conflict cases per scene:

- `present_supported`
- `present_contradicted`
- `absent_unknown`

This produces 30 new conflict cases per policy:

- 10 supported
- 10 contradicted
- 10 unknown

The harness derives case content from `scene.png` and extractor output rather than hidden benchmark state.

### Gemma 3 4B

| Policy | Overall | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| staged case variation | 66.7% | 100.0% | 100.0% | **0.0%** |
| epistemic case variation | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

### Interpretation

The staged policy fails all 10 new absent-target cases while remaining perfect on present supported and contradicted cases.

The explicit epistemic policy fixes all 10 unknown cases without harming the other labels.

This is stronger evidence that the Gemma failure is structural:

> When the target is absent, Gemma systematically tends to convert insufficient evidence into contradiction unless the epistemic policy is made explicit.

## Robustness Axis 3 — Model Variation

The same staged-versus-epistemic comparison is repeated on:

```text
qwen3:4b-instruct-2507-q4_K_M
```

No benchmark, extractor, or reasoning-stage change is introduced.

### Canonical Conflict

| Policy | Accuracy |
| --- | ---: |
| Qwen staged | **100.0%** |
| Qwen epistemic | 90.0% |

The epistemic policy hurts one canonical contradicted case.

### Paraphrase Robustness

| Policy | Overall | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Qwen staged | 86.7% | 100.0% | 88.9% | 75.0% |
| Qwen epistemic | **93.3%** | 100.0% | 77.8% | **100.0%** |

The epistemic policy improves unknown handling from 75.0% to 100.0%, but reduces contradicted accuracy.

### Case Variation

| Policy | Overall | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Qwen staged | **100.0%** | 100.0% | 100.0% | 100.0% |
| Qwen epistemic | 93.3% | 100.0% | 80.0% | 100.0% |

The epistemic rule preserves unknown and supported accuracy, but introduces two contradicted-to-supported errors.

## Qwen Error Audit

A direct staged-versus-epistemic audit confirms the Qwen regressions are semantic rather than parser artifacts.

The repeated regression pattern is:

```text
EXISTENCE: present
PHYSICAL_STATE: open
AGREEMENT: contradicts
VERDICT: contradicted
```

under the staged condition, changing to:

```text
EXISTENCE: present
PHYSICAL_STATE: open
AGREEMENT: supports
VERDICT: supported
```

under the epistemic condition.

This occurs in:

- one canonical contradicted case
- one paraphrase contradicted case
- two case-variation contradicted cases

The explicit rule therefore changes Qwen's reasoning behavior outside the absent-target branch even though the rule was intended to govern missing evidence.

## Primary Finding

The strongest `v0.4` result is:

> **The epistemic rule robustly protects unknown-case handling, but its net value is model-dependent.**

For Gemma:

- the native failure mode is a strong absent-target → contradiction bias
- the explicit epistemic rule fixes that failure across wording and case variation
- all tested robustness slices reach 100%

For Qwen:

- native missing-evidence handling is already stronger
- the explicit policy improves unknown robustness under paraphrases
- but it can introduce new contradiction-to-support errors

Therefore:

> **An intervention that fixes one model's epistemic failure can degrade another model that does not share the same failure profile.**

## Architectural Implication

The project now distinguishes four independent failure locations:

```text
pixels
→ extracted facts
→ task-relevant representation
→ epistemic reasoning policy
→ verdict
```

But `v0.4` adds an important qualification:

> The optimal reasoning policy is not necessarily model-invariant.

A fixed policy layer can help one model and over-constrain another.

This is the first DetectiveLab result that could eventually justify adaptive policy selection.

However, routing is still not justified yet.

The evidence currently supports only:

- different models exhibit different native failure profiles
- one fixed epistemic intervention is not universally optimal

It does not yet show that a reliable selector can predict when to use which policy.

## What This Milestone Supports

The current evidence supports these working claims:

1. Gemma's missing-evidence failure is robust to testimony paraphrases.
2. Gemma's failure persists on newly constructed absent-target cases.
3. The explicit epistemic policy reliably fixes those Gemma failures.
4. Qwen has a different native error profile.
5. The same epistemic intervention can improve unknown handling while harming contradicted cases on Qwen.
6. Reasoning-policy interventions should be evaluated against the model's native error distribution.
7. Negative cross-model results are important evidence, not noise to be hidden.

## What This Milestone Does Not Support

This milestone does **not** establish that:

- the epistemic policy universally improves multimodal reasoning
- Gemma is generally worse than Qwen
- Qwen is generally better calibrated
- a routing system is now warranted
- one small second-model check is sufficient for broad model-generalization claims
- the result transfers to natural images or larger models
- the benchmark is large enough for statistical population claims

The current results are controlled mechanism tests on small local models and synthetic scenes.

## Milestone Conclusion

`v0.4` began with a simple robustness hypothesis:

> If the explicit missing-evidence rule is correct, it should remain helpful when wording, cases, and model choice vary.

The evidence partially supports that hypothesis.

It is strongly robust for Gemma across:

- paraphrase variation
- new conflict cases
- unknown-case expansion

But the model axis reveals a critical boundary condition:

> **The rule is failure-mode-specific rather than universally beneficial.**

That negative result is the most important outcome of the milestone.

## Recommended Next Step

Do not add routing yet.

The next useful experiment should test whether the model-specific policy effect is stable enough to predict.

Good candidates:

1. repeat the same staged-versus-epistemic comparison on one additional local model
2. audit which prompt components cause Qwen's contradicted-to-supported regression
3. test a narrower conditional prompt that activates the epistemic rule only after `EXISTENCE: absent`
4. measure whether a deterministic stage gate can avoid changing present-target cases

The simplest next mechanism to test is:

> Can the epistemic rule be applied only after the model has already classified the target as absent?

That would preserve the intended intervention boundary and test whether Qwen's collateral contradicted-case regressions disappear without introducing routing or learned control.
