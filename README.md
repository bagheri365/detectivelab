# DetectiveLab

**A controlled multimodal research lab for studying when visual evidence should stay raw, when it should become structured, and when escalation is actually worth the cost.**

DetectiveLab uses a small frozen synthetic benchmark to isolate failure causes one capability at a time. The project does not optimize for maximum benchmark score; it optimizes for **diagnosability**.

> **Evidence before complexity.**

## Status

- Final milestone: `v0.10`
- Branch: `v0.10-risk-operating-point`
- Tests: `125 passed`
- Frozen benchmark: `artifacts/benchmark_v0_0_1`
- Main models: `gemma3:4b`, `qwen3:4b-instruct-2507-q4_K_M`
- Research status: **complete**

Final systems principle:

> **Uncertainty is useful only when escalation has positive expected value.**

## Quick read

The research arc established five main results:

1. **Perception can masquerade as reasoning failure.** Oracle structure substantially outperformed raw input.
2. **Focused structure can beat dense correct structure.** More evidence is not always better evidence.
3. **Conflict reasoning depends on epistemic policy.** Missing evidence was often misread as contradiction.
4. **Control works better from reliable structured facts than from re-inference.** Gate location mattered.
5. **Risk detection alone is insufficient.** In `v0.10`, broader failure coverage increased model calls but reduced downstream accuracy because the fallback was itself unreliable.

## One real benchmark example

![Real benchmark example from `scene_0002`](docs/figures/benchmark_example_scene_0002.png)

`scene_0002` is a real conflict item from the frozen benchmark:

- physical evidence: **blue window = closed**
- witness testimony: **blue window = open**
- rule: **current physical evidence overrides unverified witness testimony**
- gold verdict: **contradicted**

This captures the core benchmark pattern:

```text
scene image
→ focused evidence extraction
→ compare physical evidence with testimony
→ apply an explicit policy
→ final verdict
```

## Benchmark

The corrected benchmark is frozen at:

```text
artifacts/benchmark_v0_0_1
```

It contains:

```text
10 scenes × 3 families = 30 items
```

Families:

- `spatial`
- `state`
- `conflict`

Label balance:

```text
spatial:  5 yes / 5 no
state:    5 yes / 5 no
conflict: 3 supported / 3 contradicted / 4 unknown
```

Validate it with:

```bash
python -m detectivelab.validate artifacts/benchmark_v0_0_1
```

The original `v0.0` conflict benchmark was invalidated because its wording leaked a shortcut. It is preserved rather than silently rewritten.

## Results at a glance

| Milestone | Main intervention | Main result |
| --- | --- | --- |
| `v0.1` | raw vs oracle structure | Oracle structure: **86.7%** overall vs raw: **53.3%** |
| `v0.2` | dense vs focused extracted structure | Focused structure matched oracle at **86.7%** |
| `v0.3` | explicit missing-evidence policy | Gemma conflict: **70% → 100%** |
| `v0.4` | cross-model robustness | Same policy can help one model and hurt another |
| `v0.5` | LLM gate vs extractor gate | Extractor-gated conflict: **100%** on both models |
| `v0.6` | directional gate corruption | False absence is much more damaging than false presence |
| `v0.7` | abstaining gate | Abstention trades compute for recovery |
| `v0.8` | calibrated extractor instability | Evidence-grounded uncertainty recovered **100%** canonical conflict accuracy |
| `v0.9` | prospective degradation test | Event-level failure recall only **35.7%** overall |
| `v0.10` | multi-signal risk policies | Best accuracy remained **83.0% at 43.0% model calls**; broader escalation was dominated |

Detailed write-ups live in [`docs/results/`](docs/results/).

## Final architecture studied

```text
scene.png
   ↓
deterministic image-derived structure
   ↓
focused evidence
   ↓
evidence-risk signals
   ↓
control decision
   ├─ stable absent   → deterministic unknown
   ├─ stable present  → staged reasoning when needed
   └─ risky/uncertain → escalate only when fallback value is justified
```

This is intentionally **not** a learned router.

`v0.10` showed why: detecting more risky cases did not improve the accuracy-compute frontier when the fallback reasoner could not reliably recover them.

## Final `v0.10` operating point

Gemma results:

| Policy | Failure recall | Model-call rate | Accuracy |
| --- | ---: | ---: | ---: |
| `NEVER_ESCALATE` | 0.0% | 43.0% | **83.0%** |
| `STABILITY_ONLY` | 2.9% | 43.0% | **83.0%** |
| `TWO_PLUS` | 40.0% | 54.0% | 81.5% |
| `ANY_SIGNAL` | 60.0% | 60.5% | 79.5% |
| `ALWAYS_ESCALATE` | 100.0% | 100.0% | 59.0% |

The key result is not that risk detection failed. It is that **better risk detection did not imply better outcomes**.

```text
detect risk
→ escalate
→ recover
```

The last step cannot be assumed.

## Setup

Designed for local CPU execution on macOS with Ollama.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
```

Expected test result:

```text
125 passed
```

Main local models:

```text
gemma3:4b
qwen3:4b-instruct-2507-q4_K_M
```

Deterministic defaults:

```text
temperature = 0
seed = 0
think = false
num_predict = 128
```

## Run the final experiment

```bash
python -m detectivelab.cli.risk_operating_point \
  --benchmark artifacts/benchmark_v0_0_1 \
  --v09 \
    artifacts/evaluation/v0_9_uncertainty_blur_gemma3_4b.jsonl \
    artifacts/evaluation/v0_9_uncertainty_downsample_gemma3_4b.jsonl \
    artifacts/evaluation/v0_9_uncertainty_contrast_gemma3_4b.jsonl \
    artifacts/evaluation/v0_9_uncertainty_occlusion_gemma3_4b.jsonl \
  --model gemma3:4b \
  --num-predict 128 \
  --cache artifacts/evaluation/v0_10_counterfactual_staged_gemma3_4b.jsonl \
  --output artifacts/evaluation/v0_10_risk_operating_point_gemma3_4b.json
```

Audit:

```bash
python scripts/audit_risk_operating_point.py \
  artifacts/evaluation/v0_10_risk_operating_point_gemma3_4b.json
```

## Research record

Each milestone has a dedicated result document:

```text
docs/results/v0_1_direct.md
docs/results/v0_2_extracted_structure.md
docs/results/v0_3_conflict_arbitration.md
docs/results/v0_4_epistemic_robustness.md
docs/results/v0_5_conditional_epistemic.md
docs/results/v0_6_gate_corruption.md
docs/results/v0_7_abstaining_gate.md
docs/results/v0_8_evidence_uncertainty.md
docs/results/v0_9_uncertainty_prediction.md
docs/results/v0_10_risk_operating_point.md
```

Important diagnostic scripts:

```text
scripts/audit_spatial.py
scripts/audit_conflict_staged.py
scripts/audit_epistemic_model_effect.py
scripts/audit_conditional_gate.py
scripts/audit_perturbation_stability.py
scripts/audit_uncertainty_prediction.py
scripts/audit_risk_operating_point.py
```

## What the evidence does not establish

This project does **not** establish that:

- the synthetic extractor transfers to natural images;
- perturbation stability is a calibrated probability;
- the current uncertainty probes generalize to new visual domains;
- learned routing is necessary;
- escalation improves accuracy simply because risk was detected;
- a larger model would eliminate the observed architectural effects;
- this small controlled benchmark supports broad generalization claims.

## Reproducibility principles

- freeze completed benchmarks;
- preserve negative results;
- keep participant inputs separate from gold labels;
- prefer deterministic experiments;
- audit surprising results before adding complexity;
- do not add learned routing without evidence.

## Research status

`v0.10` is the final planned milestone for this research arc.

The project stops here because the final experiment shows that the key remaining problem is not merely **when to escalate**, but **whether escalation has positive expected value on the cases selected**.

> **Reliable multimodal control requires not only detecting when the current evidence path is risky, but also knowing whether the fallback path is likely to recover rather than amplify that risk.**
