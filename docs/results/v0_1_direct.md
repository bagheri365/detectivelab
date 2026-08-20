# DetectiveLab v0.1 Direct — Results Note

## Status

**Milestone:** `v0.1-direct`  
**Benchmark:** `v0.0.1`  
**Model:** `gemma3:4b` via Ollama  
**Evaluation size:** 30 items across 10 scenes  
**Families:** spatial, state, conflict

This milestone compares three evidence conditions while holding the benchmark, model, decoding path, and answer format fixed.

## Research Question

How much of DetectiveLab's difficulty comes from visual perception and grounding versus downstream reasoning?

The three conditions are:

- **QUESTION** — participant-facing question/context only; no scene image.
- **RAW** — same task plus the rendered scene image.
- **ORACLE_STRUCTURED** — no image; instead the model receives correct symbolic scene facts derived from the benchmark's hidden state.

The oracle condition is diagnostic. It does not represent a deployable perception system. Its purpose is to estimate performance when perception and grounding are assumed correct.

## Results

| Condition | Overall | Conflict | Spatial | State |
| --- | ---: | ---: | ---: | ---: |
| QUESTION | 50.0% | 30.0% | 70.0% | 50.0% |
| RAW | 53.3% | 30.0% | 50.0% | 80.0% |
| ORACLE_STRUCTURED | **86.7%** | **60.0%** | **100.0%** | **100.0%** |

### Condition deltas

| Comparison | Overall | Conflict | Spatial | State |
| --- | ---: | ---: | ---: | ---: |
| RAW - QUESTION | +3.3 pp | 0 pp | -20 pp | +30 pp |
| ORACLE - RAW | +33.4 pp | +30 pp | +50 pp | +20 pp |
| ORACLE - QUESTION | +36.7 pp | +30 pp | +30 pp | +50 pp |

## Primary Finding

The strongest result is the gap between **RAW** and **ORACLE_STRUCTURED**:

> The same model improves from 53.3% to 86.7% when raw visual input is replaced with correct structured scene evidence.

On this controlled benchmark, that indicates that a substantial portion of the observed difficulty occurs before or during visual grounding rather than in the final reasoning step alone.

The family-level results make that distinction clearer.

## Spatial

- QUESTION: 70.0%
- RAW: 50.0%
- ORACLE_STRUCTURED: 100.0%

Oracle structure completely solves the current spatial family.

The 70% QUESTION result should not be interpreted as evidence of text-only spatial reasoning because the family contains only 10 items. With such a small slice, a 7/10 result can arise from answer-pattern variance.

The more important comparison is RAW versus ORACLE:

> When the correct spatial relation is provided symbolically, the model solves all spatial items; when it must recover that relation from the rendered image, performance falls to chance-level on this slice.

**Working interpretation:** spatial performance is strongly representation/perception limited.

## State

- QUESTION: 50.0%
- RAW: 80.0%
- ORACLE_STRUCTURED: 100.0%

RAW improves substantially over QUESTION, showing that the model is using visual evidence for state questions.

Oracle structure closes the remaining gap.

**Working interpretation:** the model can reason correctly over discrete state facts once those facts are available, while raw perception still introduces some errors.

## Conflict

- QUESTION: 30.0%
- RAW: 30.0%
- ORACLE_STRUCTURED: 60.0%

Conflict remains the hardest family.

The earlier `v0.0` conflict benchmark was invalidated after the QUESTION-only baseline reached 100%, revealing a shortcut from case-rule wording to the gold label. `v0.0.1` removes that shortcut by using the same evidence rule across conflict cases and making the verdict depend on the scene/testimony relationship.

Under the corrected benchmark, RAW does not improve over QUESTION, but oracle structure doubles accuracy from 30% to 60%.

This suggests two bottlenecks:

1. **Perception / grounding**
2. **Evidence comparison / arbitration**

Oracle structure removes the first bottleneck but does not solve the second completely.

## Conflict Error Audit

A six-case diagnostic probe exposed multiple distinct failure modes.

### Grounding failures

In `scene_0000`, the witness refers to an amber glass that is not present. The model instead grounded onto a different glass and treated it as relevant evidence.

In `scene_0003`, the witness refers to an amber notebook that is not present, but the model reported a state for that nonexistent object rather than returning uncertainty.

These are not final-verdict errors alone. They are object-grounding failures upstream of reasoning.

### State perception failures

In `scene_0002`, the benchmark state contains a closed blue window while the witness claims it is open. The model's visual probe reported the window as open.

In `scene_0005`, the benchmark state contains an open blue notebook while the witness claims it is closed. The model's visual probe reported the notebook as closed.

These cases show direct visual-state perception errors.

### Comparison instability

In `scene_0001`, the model correctly perceived the red door as closed and ultimately returned the correct `supported` verdict, but an intermediate comparison probe returned `conflicts`.

This indicates that even when perception is correct, evidence comparison can be unstable.

### Correct full-chain case

In `scene_0004`, perception, comparison, and final verdict were all correct.

## What This Milestone Supports

The current evidence supports the following working claims:

1. **Raw multimodal input is not sufficient to expose the model's full reasoning capability.**
2. **Correct structured evidence substantially improves performance for the same model.**
3. **State and spatial errors are largely upstream of final reasoning on this benchmark.**
4. **Conflict tasks retain a reasoning/arbitration bottleneck even after oracle perception is supplied.**
5. **Multimodal failure should not be treated as a single category; grounding, state perception, comparison, and verdict mapping can fail independently.**

## What This Milestone Does Not Support

This experiment does **not** yet show that:

- structured pipelines are universally better than end-to-end VLMs;
- a practical extractor can approach oracle performance;
- routing between RAW and structured paths is justified;
- the current percentages will generalize to a larger benchmark;
- Gemma 3 4B represents other multimodal models;
- the current 30-item benchmark is large enough for strong statistical claims.

The oracle condition is intentionally optimistic and should be treated as an upper-bound diagnostic.

## Architectural Implication

The current results justify the next mechanism:

> Replace oracle scene facts with automatically extracted structured evidence and measure how much of the oracle gap can be recovered.

This mechanism is now evidence-motivated rather than added speculatively.

## Next Milestone

### `v0.2-extracted-structure`

Research question:

> Can a lightweight automatic perception layer recover a meaningful fraction of the RAW-to-ORACLE performance gap without introducing a brittle new bottleneck?

Primary comparison:

| Condition | Purpose |
| --- | --- |
| RAW | End-to-end visual baseline |
| EXTRACTED_STRUCTURED | Practical explicit-perception path |
| ORACLE_STRUCTURED | Upper-bound diagnostic |

Primary quantity of interest:

```text
oracle_gap = ORACLE_STRUCTURED - RAW
recovered_gap = EXTRACTED_STRUCTURED - RAW
gap_recovery = recovered_gap / oracle_gap
```

The next milestone should remain small and CPU-friendly. It should not introduce routing, hybrid reasoning, agents, tool use, fine-tuning, or benchmark expansion until the extracted-structure result justifies further complexity.

## Guardrail

**Do not add routing or hybrid evidence yet.**

The next question is not "which path should the system choose?"

It is:

> **Can an explicit perception layer produce useful structured evidence at all?**

Only if different evidence paths later show stable, meaningful advantages should routing be considered.
