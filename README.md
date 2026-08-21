# DetectiveLab

**Where did the AI fail: seeing, reasoning, or deciding when to escalate?**

DetectiveLab is a controlled multimodal research lab for studying how AI systems should use visual evidence.

It asks a simple question:

> **When a model gets an answer wrong, was the problem perception, reasoning, evidence representation, or the fallback strategy?**

The project uses a small frozen synthetic benchmark where the correct world state is known in advance. That makes failures easier to diagnose instead of treating the model as one opaque end-to-end system.

> **Evidence before complexity.**

---

## 10-second explanation

DetectiveLab creates visual scenes with known answers and compares different ways of solving them:

```text
scene image
    ↓
extract evidence
    ↓
reason over evidence
    ↓
decide whether escalation is needed
    ↓
final answer
```

The research tests whether AI performs better with:

```text
raw image
structured facts
focused structured facts
image + structure
uncertainty signals
fallback reasoning
```

The main conclusion is:

> **Detecting risk is not enough. Escalation only helps when the fallback is actually likely to recover the failure.**

---

## One concrete example

A benchmark scene might contain:

```text
Physical evidence:
blue window = closed

Witness testimony:
blue window = open

Rule:
current physical evidence overrides
unverified witness testimony
```

Question:

```text
Is the witness testimony supported,
contradicted, or unknown?
```

Correct answer:

```text
contradicted
```

The system can now test different evidence paths.

### Raw-image path

```text
scene.png
    ↓
model must recognize the window
    ↓
model must determine closed/open
    ↓
model must compare that with testimony
    ↓
model must apply the rule
    ↓
answer
```

If the answer is wrong, several things could have failed.

### Structured-evidence path

```text
blue window = closed
witness = open
rule = physical evidence wins
    ↓
reasoner
    ↓
contradicted
```

This removes most of the perception problem.

Comparing the two helps answer:

> **Was the model bad at seeing the evidence, or bad at reasoning over correct evidence?**

---

## Why DetectiveLab exists

Multimodal systems often combine several capabilities into one prediction:

```text
pixels
  ↓
perception
  ↓
representation
  ↓
reasoning
  ↓
control / routing
  ↓
answer
```

When the final answer is wrong, an end-to-end accuracy score does not tell us where the failure happened.

DetectiveLab separates these stages experimentally.

The goal is **not maximum benchmark accuracy**.

The goal is:

> **Diagnosability.**

---

## What the research found

The project established five main results.

### 1. Perception can look like reasoning failure

When the model received perfect structured evidence instead of raw visual input, accuracy improved substantially.

```text
Raw input:        53.3%
Oracle structure: 86.7%
```

That means some apparent reasoning failures were actually perception failures.

---

### 2. More evidence is not always better evidence

Dense structured evidence did not automatically help.

Focused evidence matched oracle performance:

```text
Focused structure: 86.7%
```

The useful lesson is:

> **Relevant structure can outperform more complete structure.**

---

### 3. Conflict reasoning requires an explicit evidence policy

The model sometimes interpreted missing evidence as contradiction.

Adding an explicit missing-evidence policy improved Gemma conflict accuracy:

```text
70% → 100%
```

So conflict reasoning depends not only on facts, but also on rules for interpreting absence, disagreement, and evidence priority.

---

### 4. Control works better from reliable evidence than from re-inference

DetectiveLab compared different places to make routing or gating decisions.

An extractor-based gate performed better than asking the model to infer the same control signal again.

For conflict cases:

```text
Extractor-gated accuracy:
100% on both tested models
```

This suggests that control decisions should use the most reliable evidence representation available.

---

### 5. Better risk detection does not guarantee better outcomes

This became the key result of the final experiment.

The system learned to identify more risky cases.

But escalating more of those cases caused accuracy to fall.

```text
Policy             Model calls    Accuracy

NEVER_ESCALATE        43.0%         83.0%
STABILITY_ONLY        43.0%         83.0%
TWO_PLUS              54.0%         81.5%
ANY_SIGNAL            60.5%         79.5%
ALWAYS_ESCALATE      100.0%         59.0%
```

The failure was not simply:

```text
cannot detect risky cases
```

It was:

```text
detect risky case
      ↓
escalate
      ↓
fallback also fails
```

So the final systems principle is:

> **Uncertainty is useful only when escalation has positive expected value.**

---

## The architecture studied

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
    │
    ├── stable absent
    │       ↓
    │   deterministic unknown
    │
    ├── stable present
    │       ↓
    │   staged reasoning when needed
    │
    └── risky / uncertain
            ↓
        escalate only when
        fallback value is justified
```

This is intentionally **not a learned router**.

The experiments showed that detecting more risky cases did not improve the accuracy/compute tradeoff when the fallback reasoner could not reliably recover them.

---

## Research progression

The project evolved through a sequence of controlled experiments:

```text
v0.1
raw image
vs
oracle structure
    ↓
Where is the failure:
perception or reasoning?

v0.2
dense structure
vs
focused structure
    ↓
How much evidence should the model receive?

v0.3
explicit conflict policy
    ↓
How should missing and conflicting evidence be handled?

v0.4
cross-model test
    ↓
Do the same interventions generalize across models?

v0.5
LLM gate
vs
extractor gate
    ↓
Where should control decisions happen?

v0.6
gate corruption
    ↓
Which control mistakes are most damaging?

v0.7
abstaining gate
    ↓
Can uncertainty trade compute for recovery?

v0.8
evidence instability
    ↓
Can uncertainty grounded in evidence help?

v0.9
prospective degradation
    ↓
Do those uncertainty signals predict real failures?

v0.10
multi-signal risk policies
    ↓
Does detecting more risk actually improve outcomes?
```

---

## Benchmark

The corrected frozen benchmark is:

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
spatial:   5 yes / 5 no
state:     5 yes / 5 no
conflict:  3 supported / 3 contradicted / 4 unknown
```

Validate it with:

```bash
python -m detectivelab.validate artifacts/benchmark_v0_0_1
```

The original `v0.0` conflict benchmark was invalidated because its wording leaked a shortcut.

It is preserved rather than silently rewritten.

---

## Results at a glance

| Milestone | Main intervention | Main result |
| --- | --- | --- |
| `v0.1` | raw vs oracle structure | Oracle: **86.7%** vs raw: **53.3%** |
| `v0.2` | dense vs focused structure | Focused structure matched oracle at **86.7%** |
| `v0.3` | missing-evidence policy | Gemma conflict: **70% → 100%** |
| `v0.4` | cross-model robustness | Same policy can help one model and hurt another |
| `v0.5` | LLM gate vs extractor gate | Extractor-gated conflict: **100%** on both models |
| `v0.6` | gate corruption | False absence was more damaging than false presence |
| `v0.7` | abstaining gate | Abstention trades compute for recovery |
| `v0.8` | evidence uncertainty | Recovered **100%** canonical conflict accuracy |
| `v0.9` | degradation prediction | Event-level failure recall: **35.7%** overall |
| `v0.10` | multi-signal risk policies | Best result remained **83.0% accuracy at 43.0% model calls** |

Detailed results:

```text
docs/results/
```

---

## Repository mental model

A benchmark case separates truth from what the model is allowed to see.

```text
canonical scene truth
        │
        ├── render ───────► scene.png
        │
        ├───────────────► gold answer
        │
        └───────────────► participant evidence
```

That separation is important.

The model should never receive hidden gold information while supposedly being tested on perception.

A scene typically contains artifacts such as:

```text
scene.json
scene.png
questions.json
payloads.json
provenance.json
```

Conceptually:

```text
scene.json
    =
what is actually true

scene.png
    =
what the visual system sees

questions.json
    =
what is being asked + gold labels

payloads.json
    =
what evidence an experimental condition receives

provenance.json
    =
how the artifact was generated and tracked
```

---

## Setup

Designed for local CPU execution on macOS with Ollama.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python -m pytest
```

Expected result:

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

---

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

---

## Research record

Each milestone has a dedicated write-up:

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

Diagnostic scripts:

```text
scripts/audit_spatial.py
scripts/audit_conflict_staged.py
scripts/audit_epistemic_model_effect.py
scripts/audit_conditional_gate.py
scripts/audit_perturbation_stability.py
scripts/audit_uncertainty_prediction.py
scripts/audit_risk_operating_point.py
```

---

## What this project does not claim

DetectiveLab does **not** establish that:

- the synthetic extractor transfers directly to natural images;
- perturbation stability is a calibrated probability;
- the current uncertainty probes generalize to new visual domains;
- learned routing is necessary;
- escalation improves accuracy simply because risk was detected;
- a larger model would eliminate the observed architectural effects;
- a 30-item controlled benchmark supports broad generalization claims.

The benchmark is deliberately small because the goal is controlled diagnosis, not broad benchmark leadership.

---

## Reproducibility principles

- freeze completed benchmarks;
- preserve negative results;
- keep participant inputs separate from gold labels;
- prefer deterministic experiments;
- audit surprising results before adding complexity;
- do not add learned routing without evidence.

---

## Project status

```text
Final milestone:   v0.10
Branch:            v0.10-risk-operating-point
Tests:             125 passed
Frozen benchmark:  artifacts/benchmark_v0_0_1
Research status:   complete
```

`v0.10` is the final planned milestone for this research arc.

The project stops here because the remaining problem is no longer simply:

```text
When should the system escalate?
```

It is:

```text
If the system escalates,
is the fallback actually more likely
to recover the case?
```

The final takeaway is:

> **Reliable multimodal control requires not only detecting when the current evidence path is risky, but also knowing whether the fallback path is likely to recover rather than amplify that risk.**