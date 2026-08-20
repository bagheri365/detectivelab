# DetectiveLab v0.2 Extracted Structure — Results Note

## Status

**Milestone:** `v0.2-extracted-structure`  
**Benchmark:** `v0.0.1`  
**Model:** `gemma3:4b` via Ollama  
**Evaluation size:** 30 items across 10 scenes  
**Families:** spatial, state, conflict

This milestone tests whether image-derived structured evidence can recover the performance gap between raw visual reasoning and oracle structured evidence.

## Research Question

Can a lightweight automatic perception layer recover a meaningful fraction of the RAW-to-ORACLE performance gap without introducing a brittle new bottleneck?

The milestone compares five conditions:

- **QUESTION** — participant-facing question/context only; no scene image.
- **RAW** — participant-facing task plus the rendered scene image.
- **EXTRACTED_STRUCTURED** — dense symbolic evidence recovered from `scene.png` by a deterministic image-only extractor.
- **EXTRACTED_FOCUSED** — only task-relevant facts selected from the image-derived structured evidence.
- **ORACLE_STRUCTURED** — correct symbolic scene facts derived from hidden benchmark state; diagnostic upper bound.

The extractor is synthetic-renderer-specific and reads only image pixels. It does not access `scene.json`, gold labels, hidden IDs, or benchmark provenance during extraction.

## Results

| Condition | Overall | Conflict | Spatial | State |
| --- | ---: | ---: | ---: | ---: |
| QUESTION | 50.0% | 30.0% | 70.0% | 50.0% |
| RAW | 53.3% | 30.0% | 50.0% | 80.0% |
| EXTRACTED_STRUCTURED | 70.0% | 60.0% | 50.0% | 100.0% |
| EXTRACTED_FOCUSED | **86.7%** | **60.0%** | **100.0%** | **100.0%** |
| ORACLE_STRUCTURED | **86.7%** | **60.0%** | **100.0%** | **100.0%** |

## Primary Finding

The strongest result is not simply that structured evidence helps.

It is:

> **Focused image-derived structure matched oracle performance, while dense correct structure degraded reasoning.**

`EXTRACTED_FOCUSED` reaches the same 86.7% overall accuracy as `ORACLE_STRUCTURED`.

By contrast, `EXTRACTED_STRUCTURED` reaches only 70.0%.

This indicates that the extractor itself was not the primary reason for the remaining gap. The way correct extracted evidence was serialized to the reasoner materially affected performance.

## Oracle Gap Recovery

Using RAW as the practical end-to-end baseline:

```text
RAW                = 53.3%
ORACLE_STRUCTURED  = 86.7%

oracle_gap = 33.4 percentage points
```

Dense extracted structure recovers:

```text
70.0 - 53.3 = 16.7 pp

gap_recovery ≈ 16.7 / 33.4 ≈ 50%
```

Focused extracted structure recovers:

```text
86.7 - 53.3 = 33.4 pp

gap_recovery = 100%
```

On this benchmark slice, task-relevant extracted structure recovers the full measured oracle gap.

## State

| Condition | Accuracy |
| --- | ---: |
| QUESTION | 50.0% |
| RAW | 80.0% |
| EXTRACTED_STRUCTURED | 100.0% |
| EXTRACTED_FOCUSED | 100.0% |
| ORACLE_STRUCTURED | 100.0% |

The state family shows the cleanest benefit from explicit perception.

RAW visual reasoning improves over QUESTION, demonstrating that the model uses the image. However, it still makes state-perception errors.

Both dense and focused extracted structure reach the oracle result.

**Working interpretation:**

> For discrete object state, explicit image-derived structure removes the remaining measurable perception bottleneck on the current benchmark.

## Spatial

| Condition | Accuracy |
| --- | ---: |
| QUESTION | 70.0% |
| RAW | 50.0% |
| EXTRACTED_STRUCTURED | 50.0% |
| EXTRACTED_FOCUSED | 100.0% |
| ORACLE_STRUCTURED | 100.0% |

At first, the 50.0% dense extracted result appeared to suggest that spatial relations were difficult to recover from the image.

A direct audit rejected that explanation.

The image-only extractor correctly recovered the relevant queried spatial relation in the audited scenes. The dense representation instead exposed all pairwise left/right relations among the six detected objects.

For six objects, this produced 15 pairwise relation statements.

The model then showed a strong `yes` bias:

- all five gold-`yes` spatial items were correct;
- all five gold-`no` spatial items were predicted `yes`.

The relevant relation was present in the evidence, but surrounded by many irrelevant correct relations.

When the representation was reduced to the queried objects and their relevant relation, spatial accuracy increased from 50.0% to 100.0%.

**Working interpretation:**

> The spatial bottleneck was not extraction quality. It was evidence density and representation design.

This result motivates a broader principle:

> **More correct structure is not necessarily better structure.**

## Conflict

| Condition | Accuracy |
| --- | ---: |
| QUESTION | 30.0% |
| RAW | 30.0% |
| EXTRACTED_STRUCTURED | 60.0% |
| EXTRACTED_FOCUSED | 60.0% |
| ORACLE_STRUCTURED | 60.0% |

Conflict remains the hardest family.

The image-derived structured conditions reach the oracle ceiling, which means the measured perception gap is closed for this family.

However, all structured variants stop at 60.0%.

This isolates the remaining bottleneck:

> **The unresolved conflict errors are downstream of perception.**

Earlier probes showed multiple failure modes under RAW input:

- object-grounding errors;
- hallucinated grounding for absent objects;
- visual-state perception errors;
- inconsistent comparison behavior.

Structured evidence removes the first three measurable sources of error, but the final evidence-comparison / verdict-reasoning problem remains.

## Representation Ablation

The most important ablation in this milestone is:

```text
EXTRACTED_STRUCTURED
vs
EXTRACTED_FOCUSED
```

Both conditions use the same:

- benchmark;
- image-only extractor;
- model;
- decoding configuration;
- output labels.

They differ primarily in how much extracted evidence is exposed to the reasoner.

### Dense

The dense condition exposes all detected objects, states, and pairwise spatial relations.

Result:

```text
70.0% overall
50.0% spatial
```

### Focused

The focused condition uses participant-facing task text to select only the extracted facts relevant to that item.

Result:

```text
86.7% overall
100.0% spatial
```

The focused representation therefore matches the oracle result without accessing oracle state.

## What This Milestone Supports

The current evidence supports these working claims:

1. **Explicit image-derived structure can outperform direct RAW reasoning on this controlled benchmark.**
2. **A lightweight deterministic extractor can recover the full measured oracle advantage when evidence is task-focused.**
3. **Representation density matters independently of extraction correctness.**
4. **Correct but irrelevant symbolic facts can degrade reasoning.**
5. **State and spatial failures under RAW input are largely upstream of final reasoning on this benchmark.**
6. **Conflict retains a downstream reasoning/arbitration bottleneck after perception errors are removed.**
7. **The useful abstraction is not merely pixels versus symbols; it is pixels versus task-relevant symbols versus overly dense symbols.**

## What This Milestone Does Not Support

This experiment does **not** show that:

- the reference extractor generalizes to natural images;
- structured pipelines are universally better than VLMs;
- focused evidence will always outperform dense evidence;
- a learned extractor would behave the same way;
- the current 30-item benchmark is large enough for strong statistical claims;
- the result generalizes beyond Gemma 3 4B;
- routing is justified yet;
- hybrid RAW + structured evidence is necessary yet.

The extractor is deliberately specific to the synthetic rendering grammar.

Its role is to isolate the architecture question, not to establish production computer-vision performance.

## Architectural Implication

The original architectural question was:

> Can explicit perception close the RAW-to-ORACLE gap?

The answer on this benchmark is qualified:

> **Yes, but only when the extracted representation is selective enough for the reasoning task.**

The experiment therefore shifts the project from a simple extraction question to a representation question.

The next system should not automatically expose every extracted fact.

It should preserve task-relevant evidence while avoiding unnecessary symbolic clutter.

## Remaining Bottleneck

After `EXTRACTED_FOCUSED`, state and spatial reach 100.0%, while conflict remains at 60.0%.

Therefore the clearest remaining measured failure is:

> **evidence arbitration / conflict reasoning**

The next milestone should investigate that failure before adding routing, hybrid evidence, or additional orchestration.

## Recommended Next Milestone

### Conflict-Arbitration Study

Research question:

> Why does the same model remain at 60% conflict accuracy even when the relevant visual evidence is represented correctly and concisely?

The next experiment should separate:

1. target existence / absence;
2. state comparison;
3. witness-evidence agreement;
4. final verdict mapping.

A controlled staged prompt or intermediate-decision protocol can test whether the remaining failure comes from comparison semantics or label mapping.

## Guardrail

**Do not add routing yet.**

Routing would only be justified if different evidence paths later show stable, meaningful advantages that a selector could exploit.

At the end of `v0.2`, the strongest result is instead:

> **Task-relevant image-derived structure can match oracle performance, while dense correct structure can substantially degrade reasoning.**

The next complexity must be motivated by the remaining 60% conflict ceiling, not by architectural ambition.
