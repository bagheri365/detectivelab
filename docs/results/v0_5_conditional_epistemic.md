# DetectiveLab v0.5 Conditional Epistemic Gating — Results Note

## Status

**Milestone:** `v0.5-conditional-epistemic`  
**Benchmark base:** `v0.0.1`  
**Models:** `gemma3:4b`, `qwen3:4b-instruct-2507-q4_K_M`  
**Robustness slices:** canonical conflict, testimony paraphrases, controlled case variation

This milestone tests whether the missing-evidence policy discovered in `v0.3` and stress-tested in `v0.4` can be applied conditionally without causing collateral regressions on present-target cases.

## Research Question

Can the missing-evidence rule be activated only when the target is actually absent, preserving unknown-case gains without perturbing present-target reasoning?

The milestone compares four policy-control strategies:

1. `CONFLICT_STAGED`
2. `CONFLICT_EPISTEMIC`
3. `CONFLICT_CONDITIONAL`
4. `CONFLICT_EXTRACTOR_GATED`

The final comparison isolates not only the reasoning policy, but also **where the control decision lives**.

## Policy Variants

### CONFLICT_STAGED

Baseline staged reasoning:

```text
EXISTENCE
→ PHYSICAL_STATE
→ AGREEMENT
→ VERDICT
```

No explicit missing-evidence rule is enforced.

### CONFLICT_EPISTEMIC

Global explicit rule:

```text
if EXISTENCE = absent:
  PHYSICAL_STATE = not_applicable
  AGREEMENT = unknown
  VERDICT = unknown
```

This policy fixed Gemma's unknown-case failure but caused model-specific regressions on Qwen.

### CONFLICT_CONDITIONAL

A separate model call first predicts:

```text
EXISTENCE: present | absent
```

If the model predicts `absent`, the system deterministically returns:

```text
PHYSICAL_STATE: not_applicable
AGREEMENT: unknown
VERDICT: unknown
```

If the model predicts `present`, the unchanged staged prompt is used.

This was intended to constrain the epistemic rule to the absent branch only.

### CONFLICT_EXTRACTOR_GATED

The final condition removes the standalone LLM existence call.

Instead, the already-validated image-derived structured extractor determines whether the claimed target is present.

```text
image
→ extracted target evidence
→ target absent?
   ├─ yes → not_applicable → unknown → unknown
   └─ no  → unchanged CONFLICT_STAGED reasoning
```

For absent targets, no model call is needed.

For present targets, the model receives the unchanged staged reasoning task.

## Canonical Results

### Gemma 3 4B

| Policy | Accuracy |
| --- | ---: |
| staged | 70.0% |
| global epistemic | **100.0%** |
| LLM-gated conditional | 70.0% |
| extractor-gated | **100.0%** |

### Qwen3 4B Instruct

| Policy | Accuracy |
| --- | ---: |
| staged | **100.0%** |
| global epistemic | 90.0% |
| LLM-gated conditional | 70.0% |
| extractor-gated | **100.0%** |

The extractor-gated policy matches the best observed canonical performance for both models.

## Why the LLM Gate Failed

A direct audit of `CONFLICT_CONDITIONAL` showed that all new canonical failures were introduced by the standalone existence gate.

For both models, the gate falsely labeled the same present contradicted targets as absent.

The resulting path was:

```text
true state:
  target present
  physical state conflicts with testimony

standalone gate:
  EXISTENCE: absent

forced branch:
  PHYSICAL_STATE: not_applicable
  AGREEMENT: unknown
  VERDICT: unknown
```

The full staged and global epistemic prompts were able to recognize those same targets as present.

The conditional system therefore introduced a new bottleneck:

> **A model-based gate can be less reliable than the reasoning policy it is intended to control.**

For Gemma:

```text
conditional accuracy: 7/10
gate absent: 7
gate present: 3
conditional-only failures: 3
```

For Qwen:

```text
conditional accuracy: 7/10
gate absent: 5
gate present: 5
conditional-only failures: 3
```

In both models:

```text
wrong in all three: 0
```

Every canonical conditional failure was newly introduced by gating.

## Extractor-Gated Canonical Result

Replacing the standalone model gate with the image-derived presence signal restores:

```text
Gemma: 10/10
Qwen:  10/10
```

This isolates the key mechanism:

> **Conditional control works when the gate is grounded in reliable structured evidence, but can fail when the reasoner is asked to re-infer a fact the representation layer already knows.**

## Robustness — Testimony Paraphrases

### Gemma 3 4B

`CONFLICT_EXTRACTOR_GATED_PARAPHRASE`:

```text
30/30 = 100.0%
```

By paraphrase:

| Variant | Accuracy |
| --- | ---: |
| according_to | 100.0% |
| claim_is | 100.0% |
| reports_now | 100.0% |

By gold label:

| Label | Accuracy |
| --- | ---: |
| supported | 100.0% |
| contradicted | 100.0% |
| unknown | 100.0% |

### Qwen3 4B Instruct

`CONFLICT_EXTRACTOR_GATED_PARAPHRASE`:

```text
29/30 = 96.7%
```

By paraphrase:

| Variant | Accuracy |
| --- | ---: |
| according_to | 90.0% |
| claim_is | 100.0% |
| reports_now | 100.0% |

By gold label:

| Label | Accuracy |
| --- | ---: |
| supported | 100.0% |
| contradicted | 88.9% |
| unknown | 100.0% |

The remaining Qwen error is a present-target contradiction error.

Because extractor gating does not alter present-target reasoning, this miss is best interpreted as an ordinary reasoning error rather than a gating failure.

## Robustness — Controlled Case Variation

### Gemma 3 4B

`CONFLICT_EXTRACTOR_GATED_CASE_VARIATION`:

```text
30/30 = 100.0%
```

All three case variants reach 100%:

- `absent_unknown`
- `present_contradicted`
- `present_supported`

### Qwen3 4B Instruct

`CONFLICT_EXTRACTOR_GATED_CASE_VARIATION`:

```text
30/30 = 100.0%
```

Again, all three case variants reach 100%.

## Cross-Model Summary

| Model | Slice | Extractor-gated |
| --- | --- | ---: |
| Gemma | canonical | **100.0%** |
| Gemma | paraphrases | **100.0%** |
| Gemma | case variation | **100.0%** |
| Qwen | canonical | **100.0%** |
| Qwen | paraphrases | **96.7%** |
| Qwen | case variation | **100.0%** |

The extractor-gated policy preserves:

- `unknown = 100%` across all tested slices
- `supported = 100%` across all tested slices

The only remaining miss is one Qwen paraphrase contradicted case.

## Primary Finding

The strongest `v0.5` result is:

> **The location of the control decision matters.**

The same conditional policy behaves very differently depending on how the gate is implemented.

### Model-derived gate

```text
reasoner re-infers target existence
→ false-absence errors
→ forced unknown verdicts
→ 70% canonical accuracy on both models
```

### Representation-derived gate

```text
extractor supplies target presence
→ unknown rule applies only when supported by structured evidence
→ present-target reasoning remains unchanged
→ 100% canonical accuracy on both models
```

This suggests a broader architectural principle:

> **When a control decision depends on a fact already available in the structured representation, use that fact directly rather than delegating the same decision back to the reasoner.**

## Relationship to Earlier Milestones

DetectiveLab now has a progressively sharper architecture story.

### v0.1 — Mixed multimodal failure

RAW performance showed that perception and reasoning errors were entangled.

### v0.2 — Representation matters

Focused image-derived structure matched oracle performance, while dense correct structure degraded spatial reasoning.

### v0.3 — Epistemic policy matters

Gemma confused missing evidence with contradictory evidence.

Making the missing-evidence policy explicit fixed the conflict slice.

### v0.4 — Policies are model-specific

The same global epistemic rule helped Gemma but caused Qwen contradiction regressions.

### v0.5 — Gate location matters

A standalone model gate created new false-absence failures.

An extractor-derived gate preserved the intended benefit without perturbing present-target reasoning.

The current working decomposition is:

```text
pixels
→ extracted facts
→ task-relevant representation
→ representation-grounded control
→ reasoning policy
→ verdict
```

## What This Milestone Supports

The current evidence supports these working claims:

1. Conditional epistemic control can preserve the missing-evidence benefit without globally changing the reasoning prompt.
2. A standalone LLM existence gate can introduce new errors even when the full staged prompt handles the same cases correctly.
3. The existence gate became the dominant bottleneck in the failed conditional system.
4. Reusing the validated extractor's presence signal removes those false-absence errors.
5. The extractor-gated policy reaches 100% canonical conflict accuracy on both tested models.
6. The extractor-gated policy is robust across Gemma paraphrases and case variation.
7. The extractor-gated policy remains strong on Qwen, including 100% canonical and case-variation performance.
8. Control decisions should preferentially use reliable structured evidence when that evidence already contains the required fact.

## What This Milestone Does Not Support

This result does **not** establish that:

- deterministic gating is universally better than learned routing
- extractor-derived gates will remain reliable under natural-image perception noise
- the current synthetic extractor is suitable for production use
- all policy-selection decisions should live in the representation layer
- 100% on these small slices implies broad multimodal reliability
- routing is now justified
- the remaining Qwen paraphrase miss is unimportant

The current result is a controlled architectural mechanism test.

## Milestone Conclusion

`v0.5` began with the hypothesis that the missing-evidence policy could be applied only after absence was detected.

The first implementation failed because the model-based existence gate became a new bottleneck.

That negative result led to a simpler alternative: use the presence fact already available in the structured evidence.

The resulting extractor-gated policy:

- restores Gemma's missing-evidence gains
- avoids Qwen's global epistemic regressions
- eliminates the false-absence errors introduced by the standalone LLM gate
- remains strong across paraphrase and case variation

The main conclusion is:

> **Policy selection does not require a learned router when the trigger is already explicit in reliable structured evidence.**

## Recommended Next Step

Do not add learned routing yet.

The next useful question is whether this representation-grounded control remains effective when the gate signal itself becomes imperfect.

A natural next milestone would introduce controlled extraction corruption:

- false target absence
- false target presence
- state corruption
- confidence thresholds

The research question would be:

> How brittle is representation-grounded control when the structured evidence is no longer perfect?

That directly tests the main assumption behind the `v0.5` success rather than adding new architecture.
