# DetectiveLab v0.3 Conflict Arbitration — Results Note

## Status

**Milestone:** `v0.3-conflict-arbitration`  
**Benchmark:** `v0.0.1`  
**Model:** `gemma3:4b` via Ollama  
**Conflict items:** 10  
**Full benchmark size:** 30 items across 10 scenes

This milestone investigates the remaining conflict-reasoning bottleneck after focused extracted evidence matched the oracle representation on state and spatial items.

## Research Question

Why does conflict accuracy remain below the state/spatial ceiling even when the relevant physical evidence is extracted correctly and presented concisely?

The milestone decomposes conflict reasoning into four stages:

1. target existence
2. physical state
3. agreement between testimony and evidence
4. final verdict

It then tests whether an explicit epistemic rule about missing evidence removes the residual errors.

## Conditions

### CONFLICT_STAGED

Uses the same focused image-derived evidence as `EXTRACTED_FOCUSED`, but requires the model to emit explicit intermediate decisions:

```text
EXISTENCE:
PHYSICAL_STATE:
AGREEMENT:
VERDICT:
```

No new visual capability is introduced.

### CONFLICT_EPISTEMIC

Uses the same staged decomposition, plus one explicit rule:

```text
if EXISTENCE = absent:
  PHYSICAL_STATE = not_applicable
  AGREEMENT = unknown
  VERDICT = unknown
```

The rule makes the epistemic distinction explicit:

> absence of evidence is insufficient evidence, not contradictory evidence.

## Results

Conflict accuracy:

| Condition | Conflict Accuracy |
| --- | ---: |
| QUESTION | 30% |
| RAW | 30% |
| EXTRACTED_FOCUSED | 60% |
| CONFLICT_STAGED | 70% |
| CONFLICT_EPISTEMIC | **100%** |

The staged baseline improves over the focused direct verdict condition, but still leaves systematic errors.

The epistemic-rule condition reaches 10/10 on the current conflict slice.

## Stage Audit

The corrected semantic audit of the original staged run showed:

| Stage | Accuracy |
| --- | ---: |
| existence | 90% |
| physical state | 90% |
| agreement | 70% |
| verdict | 60% overall |
| verdict among emitted | 66.7% |

The largest drop occurred after perception and state recovery.

This localized the remaining bottleneck to evidence comparison and verdict policy rather than visual extraction.

## Primary Failure Pattern

The most important staged failures followed this pattern:

```text
EXISTENCE: absent
PHYSICAL_STATE: not_applicable
AGREEMENT: contradicts
VERDICT: contradicted
```

The model correctly recognized that the claimed target was absent, but then treated that absence as evidence against the witness.

The benchmark policy instead requires:

```text
EXISTENCE: absent
PHYSICAL_STATE: not_applicable
AGREEMENT: unknown
VERDICT: unknown
```

Therefore the residual error is best characterized as epistemic rather than perceptual.

## Primary Finding

> **After perception is controlled, conflict reasoning can still fail because the model applies the wrong epistemic policy.**

On this benchmark slice, explicitly stating that missing physical evidence implies uncertainty rather than contradiction eliminates the remaining conflict errors.

This result sharpens the DetectiveLab thesis.

The architecture question is not only:

> when should perception become explicit structure?

It is also:

> what reasoning policy should operate over that structure when evidence is incomplete?

## Interpretation

The progression across milestones now separates three bottlenecks:

### 1. Perception bottleneck

RAW state and spatial performance lagged structured evidence.

Explicit image-derived structure removed those errors.

### 2. Representation bottleneck

Dense correct structure degraded spatial reasoning.

Task-focused structure restored performance to the oracle ceiling.

### 3. Epistemic arbitration bottleneck

Even with focused correct evidence, conflict remained below ceiling.

Explicitly distinguishing contradiction from insufficient evidence removed the remaining measured errors.

This yields the current working decomposition:

```text
pixels
→ extracted facts
→ task-relevant representation
→ epistemic arbitration policy
→ verdict
```

Failure can occur independently at each stage.

## What This Milestone Supports

The current evidence supports these working claims:

1. The remaining conflict errors were not primarily visual once focused extraction was used.
2. Staging made the residual reasoning failure observable.
3. The dominant remaining error involved treating absent evidence as contradictory evidence.
4. An explicit epistemic rule removed that failure on all 10 current conflict items.
5. Reasoning quality depends not only on the facts supplied, but also on the policy used to interpret missing evidence.

## What This Milestone Does Not Support

This result does **not** establish that:

- explicit epistemic prompting will generalize beyond this benchmark;
- the model has learned a durable uncertainty policy;
- 10 conflict items are enough for strong statistical claims;
- the same behavior will hold across other models;
- explicit rules will always outperform natural-language reasoning;
- the current benchmark is robust to all prompt-level shortcuts;
- routing, hybrid evidence, or fine-tuning is now justified.

The 100% score is a controlled-mechanism result, not a broad capability claim.

## Next Step

Before adding new architecture, the next useful move is to test whether the epistemic rule survives controlled variation.

Candidates include:

- paraphrasing the testimony;
- changing object/state combinations while preserving semantics;
- adding more unknown cases;
- testing the same rule on a second small local model;
- introducing controlled evidence corruption later.

The next complexity should test robustness of the discovered mechanism, not add orchestration for its own sake.

## Milestone Conclusion

The strongest `v0.3` result is:

> **Conflict failures persisted after perception was solved because the model conflated missing evidence with contradictory evidence. Making the uncertainty policy explicit raised conflict accuracy from 70% in the staged baseline to 100% on the current slice.**

DetectiveLab now has a three-part architectural story:

- **v0.1:** raw multimodal failure mixes perception and reasoning
- **v0.2:** focused structured evidence can match oracle performance, while dense structure can hurt
- **v0.3:** even correct focused evidence needs an explicit epistemic policy when evidence is incomplete
