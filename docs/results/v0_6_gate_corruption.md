# v0.6 — Gate Corruption

Branch: `v0.6-gate-corruption`

## Research question

How brittle is representation-grounded control when the extractor-derived gate signal is imperfect?

`v0.5` showed that using a reliable structured presence signal directly was better than asking the language model to re-infer target existence in a separate gate call. `v0.6` tests the main assumption behind that result: what happens when the gate itself is wrong?

The experiment keeps the benchmark, extractor, staged reasoning prompt, model settings, and gold labels fixed. Only the gate value used for control flow is corrupted.

## Conditions

Two directional corruption modes are tested.

### False absence

A genuinely present target is flipped to `absent`.

This forces the deterministic missing-evidence branch:

```text
EXISTENCE: absent
PHYSICAL_STATE: not_applicable
AGREEMENT: unknown
VERDICT: unknown
```

The downstream language model receives no opportunity to recover the suppressed evidence.

### False presence

A genuinely absent target is flipped to `present`.

The case bypasses the deterministic `unknown` branch and is instead sent to the unchanged staged conflict reasoner.

Unlike false absence, this error is potentially recoverable downstream.

## Deterministic corruption rates

For each corruption direction, eligible cases are deterministically ranked and nested subsets are corrupted at:

- 25%
- 50%
- 75%
- 100%

Because the same eligible subsets are used across models, Gemma and Qwen are directly comparable.

Eligible cases:

- false absence: 6 present-target cases
- false presence: 4 absent-target cases

Actual deterministic flip counts:

| Requested rate | False absence | False presence |
| --- | ---: | ---: |
| 25% | 2/6 | 1/4 |
| 50% | 3/6 | 2/4 |
| 75% | 5/6 | 3/4 |
| 100% | 6/6 | 4/4 |

## Models

- `gemma3:4b`
- `qwen3:4b-instruct-2507-q4_K_M`

Runs use deterministic Ollama settings with `--num-predict 128`.

## Test status

```text
96 passed
```

## Results

### Overall accuracy

| Model | Corruption | 25% | 50% | 75% | 100% |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 3 4B | false absence | 80% | 70% | 50% | 40% |
| Gemma 3 4B | false presence | 100% | 90% | 80% | 70% |
| Qwen3 4B | false absence | 80% | 70% | 50% | 40% |
| Qwen3 4B | false presence | 100% | 100% | 100% | 100% |

The clean extractor-gated baseline from `v0.5` is 100% canonical conflict accuracy for both models.

## False-absence curve

### Gemma

| Rate | Accuracy | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| 25% | 80% | 33.3% | 100% | 100% |
| 50% | 70% | 33.3% | 66.7% | 100% |
| 75% | 50% | 0% | 33.3% | 100% |
| 100% | 40% | 0% | 0% | 100% |

### Qwen

| Rate | Accuracy | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| 25% | 80% | 33.3% | 100% | 100% |
| 50% | 70% | 33.3% | 66.7% | 100% |
| 75% | 50% | 0% | 33.3% | 100% |
| 100% | 40% | 0% | 0% | 100% |

The curves are identical across models.

This is expected from the architecture: once a truly present target is falsely declared absent, the control flow deterministically returns `unknown`. The reasoner is never invoked and therefore cannot compensate.

The error is upstream of model reasoning.

## False-presence curve

### Gemma

| Rate | Accuracy | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| 25% | 100% | 100% | 100% | 100% |
| 50% | 90% | 100% | 100% | 75% |
| 75% | 80% | 100% | 100% | 50% |
| 100% | 70% | 100% | 100% | 25% |

Gemma degrades gradually as more truly absent cases are incorrectly routed back to staged reasoning.

This reproduces the earlier Gemma-specific epistemic weakness: absent evidence can be misinterpreted once the deterministic missing-evidence policy is bypassed.

### Qwen

| Rate | Accuracy | Supported | Contradicted | Unknown |
| --- | ---: | ---: | ---: | ---: |
| 25% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 75% | 100% | 100% | 100% | 100% |
| 100% | 100% | 100% | 100% | 100% |

Qwen recovers every false-presence corruption in the current benchmark slice.

Even when all four absent targets are incorrectly admitted to the staged reasoner, Qwen still returns the correct `unknown` verdict.

## Main finding

Gate corruption is directionally asymmetric.

> **False absence is substantially more dangerous than false presence because it suppresses evidence and bypasses downstream reasoning entirely.**

A false absence creates an unrecoverable control-flow error in the current architecture.

A false presence merely delegates an otherwise deterministic `unknown` case to the downstream reasoner. Whether that becomes an error depends on the model's native epistemic behavior.

This creates a second distinction:

> **Gate recall requirements are architecture-level, while gate precision requirements can depend on downstream model robustness.**

Both models fail identically under false absence because neither model is allowed to reason.

Under false presence:

- Gemma degrades from 100% to 70%
- Qwen remains at 100%

## Architectural implication

For this representation-grounded control design:

> **Target-presence recall matters more than target-presence precision.**

Missing a truly present target is catastrophic because evidence is discarded before reasoning.

Falsely admitting an absent target is less severe because the reasoner may still recover the correct uncertainty judgment.

More generally:

> **Errors before a hard control boundary are not equivalent. False negatives that suppress evidence can be more damaging than false positives that permit additional reasoning.**

This refines the `v0.5` conclusion.

`v0.5`:

> When a control decision depends on a fact already available in structured evidence, use that fact directly rather than asking the reasoner to re-infer it.

`v0.6`:

> Direct structured control is only as safe as its asymmetric error profile; false-negative gate errors can be unrecoverable even when false positives are recoverable.

## Relationship to earlier milestones

### v0.1

Direct raw vision showed a large perception/representation bottleneck.

### v0.2

Focused image-derived structure matched oracle performance and showed that dense correct structure can still degrade reasoning.

### v0.3

After perception was controlled, Gemma's remaining conflict failures were traced to missing-evidence policy.

### v0.4

The explicit epistemic rule was robust for Gemma but model-dependent; Qwen sometimes regressed under the same intervention.

### v0.5

A separate LLM gate failed because it created false-absence errors. Reusing the already-validated extractor signal produced strong representation-grounded control.

### v0.6

Controlled corruption shows why the false-absence failure was especially damaging: false-negative gate errors suppress evidence before reasoning and create a model-independent failure mode.

## What v0.6 does not show

The current experiment is deliberately narrow.

It does not establish that:

- the synthetic extractor will remain reliable on natural images
- all gate variables have the same asymmetric risk profile
- state corruption behaves like existence corruption
- Qwen will recover arbitrary false-positive routing outside this benchmark
- confidence scoring or learned routing is necessary

The present benchmark is small and synthetic. The result should be interpreted as a controlled architectural finding, not a broad claim about multimodal systems.

## Recommended next step

Do not add a learned router yet.

The next milestone should test whether the same asymmetric principle holds when the structured evidence itself becomes uncertain rather than manually flipped after extraction.

A clean next question is:

> **Can an abstaining or confidence-aware gate reduce catastrophic false-absence errors without reintroducing unnecessary model reasoning?**

That would test a simple safety mechanism around the gate before introducing learned routing.
