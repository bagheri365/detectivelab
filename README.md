# DetectiveLab

**A controlled multimodal research lab for studying when visual evidence should stay raw, when it should become structured, and how reasoning systems should arbitrate conflicting evidence.**

DetectiveLab is a research-first repository built around a deliberately small synthetic benchmark. The goal is not to maximize benchmark accuracy as quickly as possible. The goal is to isolate *why* a multimodal system succeeds or fails by changing one capability at a time.

The project currently studies four linked questions:

1. **Perception:** can the model recover the relevant evidence from an image?
2. **Representation:** should that evidence remain raw, become dense structure, or become focused structure?
3. **Epistemic policy:** how should the system reason when physical evidence and testimony conflict?
4. **Control:** when should the system trust a structured signal, abstain, or escalate to the language model?

The guiding rule throughout the repository is:

> **Evidence before complexity.**

Before adding routing, learned confidence, fine-tuning, or larger architectures, DetectiveLab first asks whether the failure can be explained by perception, representation density, reasoning policy, gate location, or uncertainty estimation.

---

## Status

- **Current branch:** `v0.8-evidence-uncertainty`
- **Current milestone:** `v0.8` complete
- **Current test suite:** `114 passed`
- **Frozen benchmark:** `artifacts/benchmark_v0_0_1`
- **Primary local models:**
  - `gemma3:4b`
  - `qwen3:4b-instruct-2507-q4_K_M`
- **Current best canonical conflict result:** 100% on both models
- **Current calibrated uncertainty result:** 100% accuracy with 60% model-call rate and only 10% uncertainty-specific escalation

The current architectural path is:

```text
raw image
   ↓
deterministic image-derived structure
   ↓
calibrated evidence-stability gate
   ├─ stable absent   → deterministic unknown
   ├─ stable present  → staged reasoning
   └─ uncertain       → staged reasoning
```

The strongest current finding is:

> **An evidence-derived abstention signal is useful only after the perturbation probes themselves are validated for extractor invariance.**

---

# Why DetectiveLab Exists

Multimodal systems often fail in ways that are difficult to diagnose because several components are entangled.

A wrong final answer might come from:

- failing to see the relevant object,
- seeing it but encoding it poorly,
- providing too much structured information,
- applying the wrong reasoning policy,
- placing control logic in the wrong component,
- or trusting an unreliable uncertainty signal.

If all of those are changed at once, a benchmark score says very little about the cause.

DetectiveLab instead treats the system like an experimental object.

The benchmark is frozen. One intervention is added. The result is measured. Negative results are preserved. Only then does the next milestone begin.

---

# Research Thesis

The project began with a simple question:

> **When should visual evidence remain raw, and when should it become explicit structured evidence?**

That question evolved as the experiments exposed new bottlenecks.

The current research story is:

```text
perception failures
    ↓
representation bottlenecks
    ↓
representation density effects
    ↓
epistemic policy failures
    ↓
model-specific policy effects
    ↓
gate-location failures
    ↓
asymmetric gate corruption
    ↓
abstention as risk control
    ↓
evidence-derived uncertainty
```

The resulting architecture is intentionally simpler than a learned router.

No routing model has yet been justified by the evidence.

---

# Benchmark

## Frozen corrected benchmark: `v0.0.1`

Path:

```text
artifacts/benchmark_v0_0_1
```

The corrected benchmark contains:

```text
10 scenes
× 3 question families
= 30 items
```

Families:

- spatial
- state
- conflict

Label balance:

```text
spatial:
  yes  = 5
  no   = 5

state:
  yes  = 5
  no   = 5

conflict:
  supported     = 3
  contradicted  = 3
  unknown       = 4
```

Conflict items use the constant participant-facing rule:

> **Current physical evidence overrides unverified witness testimony.**

Unknown conflict cases are scene-dependent and refer to a plausible target that is absent from the current physical evidence.

For those cases:

```text
subject_id = null
```

The benchmark intentionally separates participant-facing inputs from hidden scoring information.

Gold labels are not placed in participant payloads.

Validate the benchmark with:

```bash
python -m detectivelab.validate artifacts/benchmark_v0_0_1
```

---

## Why `v0.0` Was Not Used

The original `v0.0` conflict family contained a shortcut.

The wording of the rule allowed a model to infer the conflict answer from the question itself without properly grounding the physical scene.

This made the conflict family invalid as a perception-and-reasoning test.

The original benchmark is preserved rather than rewritten.

The corrected `v0.0.1` benchmark removes that shortcut and is frozen for all subsequent experiments.

This is an important project rule:

> **Invalid benchmarks are preserved as evidence; they are not silently repaired in place.**

---

# Experimental Philosophy

Every milestone follows the same discipline:

1. freeze the benchmark;
2. identify one failure hypothesis;
3. add one intervention;
4. measure it on the same slice;
5. audit surprising failures;
6. preserve negative results;
7. avoid adding architecture until the evidence requires it.

This means DetectiveLab intentionally includes failed approaches.

Examples include:

- question-only shortcut leakage in `v0.0`;
- dense correct structure degrading spatial reasoning in `v0.2`;
- a standalone LLM gate creating false-absence errors in `v0.5`;
- globally applied epistemic policy degrading Qwen in `v0.4`;
- brightness perturbations manufacturing uncertainty in early `v0.8`.

These are not discarded implementation mistakes.

They are part of the research result.

---

# Experiment Ladder

## v0.1 — Direct Evidence Conditions

Branch:

```text
v0.1-direct
```

Commit:

```text
3adb657
```

The first controlled comparison tested three evidence conditions with Gemma 3 4B.

### Conditions

```text
QUESTION
RAW
ORACLE_STRUCTURED
```

### Results

| Condition | Overall | Conflict | Spatial | State |
| --- | ---: | ---: | ---: | ---: |
| QUESTION | 50.0% | 30% | 70% | 50% |
| RAW | 53.3% | 30% | 50% | 80% |
| ORACLE_STRUCTURED | **86.7%** | 60% | **100%** | **100%** |

### Finding

The large jump from raw/question conditions to oracle structure showed that much of the apparent reasoning failure was actually upstream.

> **Perception and representation were major bottlenecks.**

However, conflict remained at only 60% even with oracle structure.

That suggested a second failure mode downstream of perception.

Results:

```text
docs/results/v0_1_direct.md
```

---

## v0.2 — Extracted Structure

Branch:

```text
v0.2-extracted-structure
```

README milestone commit:

```text
469ce47
```

A deterministic image-only extractor was added.

Key implementation:

```text
src/detectivelab/extraction/base.py
src/detectivelab/extraction/synthetic.py
```

Public functions include:

```python
extract_scene_facts(image_path)
extract_structured_evidence(image_path)
```

The extractor reads only:

```text
scene.png
```

It does not read:

```text
scene.json
gold labels
hidden provenance
```

### Dense extracted structure

Condition:

```text
EXTRACTED_STRUCTURED
```

Gemma result:

| Family | Accuracy |
| --- | ---: |
| Overall | 70.0% |
| Conflict | 60% |
| Spatial | 50% |
| State | 100% |

The spatial audit revealed an unexpected result:

- the extracted spatial relations were correct;
- the model still failed;
- all 15 pairwise relations were presented;
- Gemma developed a strong yes-bias.

The representation was correct but too dense.

### Focused extracted structure

Condition:

```text
EXTRACTED_FOCUSED
```

Gemma result:

| Family | Accuracy |
| --- | ---: |
| Overall | **86.7%** |
| Conflict | 60% |
| Spatial | **100%** |
| State | **100%** |

This exactly matched the oracle-structured condition.

### Finding

> **Focused image-derived structure matched oracle performance, while dense correct structure degraded reasoning.**

This established two distinct representation lessons:

1. converting visual evidence into explicit structure can remove a perception bottleneck;
2. more correct structure is not necessarily better structure.

The current benchmark therefore shows a **representation-density effect**, not merely an extraction-quality effect.

Audit:

```text
scripts/audit_spatial.py
```

Results:

```text
docs/results/v0_2_extracted_structure.md
```

---

## v0.3 — Conflict Arbitration

Branch:

```text
v0.3-conflict-arbitration
```

README milestone commit:

```text
65c29af
```

After `v0.2`, spatial and state performance were effectively solved on the current slice.

Conflict remained at 60%.

The next question was therefore:

> **What reasoning failure remains after perception is controlled?**

### Staged conflict reasoning

Condition:

```text
CONFLICT_STAGED
```

The model was required to produce:

```text
EXISTENCE:
PHYSICAL_STATE:
AGREEMENT:
VERDICT:
```

A semantic audit showed that the major failure pattern was:

```text
target absent
→ physical state not applicable
→ model still says contradicts
→ verdict contradicted
```

The model was conflating:

> absence of evidence

with:

> contradictory evidence

### Explicit epistemic policy

Condition:

```text
CONFLICT_EPISTEMIC
```

Added rule:

```text
if EXISTENCE = absent:
  PHYSICAL_STATE = not_applicable
  AGREEMENT = unknown
  VERDICT = unknown
```

Gemma canonical conflict results:

| Condition | Accuracy |
| --- | ---: |
| CONFLICT_STAGED | 70% |
| CONFLICT_EPISTEMIC | **100%** |

Final semantic audit:

```text
existence       100%
physical state  100%
agreement       100%
verdict         100%
```

### Finding

> **After perception is controlled, the remaining Gemma conflict failure is an epistemic-policy failure.**

More specifically:

> **Absence of evidence was being treated as contradictory evidence.**

Audit:

```text
scripts/audit_conflict_staged.py
```

Results:

```text
docs/results/v0_3_conflict_arbitration.md
```

---

## v0.4 — Epistemic Robustness

Branch:

```text
v0.4-epistemic-robustness
```

README milestone commit:

```text
45c632c1
```

`v0.3` fixed the canonical Gemma conflict slice.

`v0.4` tested whether that fix was robust.

The scope was deliberately limited to:

1. paraphrase robustness;
2. controlled case variation;
3. cross-model robustness.

No routing was added.

---

### Gemma paraphrase robustness

Three witness paraphrases were generated for every canonical conflict item:

```text
according_to
claim_is
reports_now
```

30 records per policy.

Results:

| Policy | Accuracy |
| --- | ---: |
| staged | 66.7% |
| epistemic | **100%** |

Under the staged policy, `unknown` accuracy fell to 16.7%.

Under the explicit epistemic policy, every variant and label reached 100%.

---

### Gemma case variation

Three deterministic conflict variants per scene:

```text
present_supported
present_contradicted
absent_unknown
```

30 records per policy.

Results:

| Policy | Accuracy |
| --- | ---: |
| staged | 66.7% |
| epistemic | **100%** |

Again:

```text
supported     100%
contradicted  100%
unknown       100%
```

under the explicit policy.

---

### Cross-model Qwen robustness

Qwen did not behave like Gemma.

Canonical:

| Policy | Qwen |
| --- | ---: |
| staged | **100%** |
| epistemic | 90% |

Paraphrase:

| Policy | Qwen |
| --- | ---: |
| staged | 86.7% |
| epistemic | 93.3% |

Case variation:

| Policy | Qwen |
| --- | ---: |
| staged | **100%** |
| epistemic | 93.3% |

The explicit rule improved missing-evidence handling but introduced genuine present-target regressions.

The repeated pattern was:

```text
present / open contradicted case

AGREEMENT:
  contradicts → supports

VERDICT:
  contradicted → supported
```

### Finding

> **The same explicit reasoning intervention can fix one model and degrade another.**

The missing-evidence rule is highly effective for Gemma, but its net value is model-dependent.

This was the first strong evidence that a global reasoning policy was not the right control mechanism.

Audit:

```text
scripts/audit_epistemic_model_effect.py
```

Results:

```text
docs/results/v0_4_epistemic_robustness.md
```

---

## v0.5 — Conditional Epistemic Control

Branch:

```text
v0.5-conditional-epistemic
```

Implementation commit:

```text
1c47060
```

README milestone commit:

```text
35458c6
```

Research question:

> **Can the missing-evidence rule activate only after absence is detected, preserving unknown-case gains without perturbing present-target reasoning?**

Four policies were compared:

| Policy | Control mechanism |
| --- | --- |
| `CONFLICT_STAGED` | no explicit missing-evidence rule |
| `CONFLICT_EPISTEMIC` | global explicit epistemic rule |
| `CONFLICT_CONDITIONAL` | standalone LLM existence gate |
| `CONFLICT_EXTRACTOR_GATED` | gate from existing image-derived structure |

### Canonical conflict

| Model | Staged | Global epistemic | LLM-gated | Extractor-gated |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 4B | 70% | **100%** | 70% | **100%** |
| Qwen3 4B | **100%** | 90% | 70% | **100%** |

### Negative result: LLM gate

The standalone existence gate created a new failure mode.

Both models falsely classified the same three present contradicted targets as absent.

That forced:

```text
not_applicable
unknown
unknown
```

even though the full staged reasoning prompt recognized the targets as present.

The gate was less reliable than the reasoning policy it was supposed to control.

### Successful result: extractor gate

Instead of asking the model to re-infer existence, DetectiveLab reused the already validated image-derived presence signal.

Routing:

```text
target absent
→ zero model calls
→ deterministic unknown

target present
→ unchanged staged reasoning
```

Canonical result:

```text
Gemma 10/10
Qwen  10/10
```

Robustness:

| Model | Slice | Extractor-gated |
| --- | --- | ---: |
| Gemma | canonical | 100% |
| Gemma | paraphrases | 100% |
| Gemma | case variation | 100% |
| Qwen | canonical | 100% |
| Qwen | paraphrases | 96.7% |
| Qwen | case variation | 100% |

### Finding

> **Gate location matters.**

More specifically:

> **When control depends on a fact already available in reliable structured evidence, use that fact directly rather than asking the reasoner to re-infer it.**

This result did not justify a learned router.

It justified simpler representation-grounded control.

Audit:

```text
scripts/audit_conditional_gate.py
```

Results:

```text
docs/results/v0_5_conditional_epistemic.md
```

---

## v0.6 — Gate Corruption

Branch:

```text
v0.6-gate-corruption
```

Research question:

> **How brittle is representation-grounded control when the gate signal is wrong?**

Two directional corruptions were introduced.

### False absence

```text
present → absent
```

This suppresses evidence and forces the deterministic unknown path.

### False presence

```text
absent → present
```

This sends a genuinely absent case back to staged reasoning.

Corruption was applied deterministically at:

```text
25%
50%
75%
100%
```

of eligible cases.

### Full degradation grid

| Model | Corruption | 25% | 50% | 75% | 100% |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 3 4B | false absence | 80% | 70% | 50% | 40% |
| Gemma 3 4B | false presence | 100% | 90% | 80% | 70% |
| Qwen3 4B | false absence | 80% | 70% | 50% | 40% |
| Qwen3 4B | false presence | 100% | 100% | 100% | 100% |

### Finding

Gate corruption is strongly asymmetric.

False absence is model-independent because the model never receives the evidence.

False presence is model-dependent because the downstream reasoner still has an opportunity to recover.

The architectural result is:

> **Errors before a hard control boundary are not equivalent. False negatives that suppress evidence can be substantially more dangerous than false positives that permit additional reasoning.**

For this gate:

> **Target-presence recall matters more than target-presence precision.**

Results:

```text
docs/results/v0_6_gate_corruption.md
```

---

## v0.7 — Abstaining Gate

Branch:

```text
v0.7-abstaining-gate
```

Commit:

```text
b2b82a4
```

Research question:

> **Can an abstaining gate reduce catastrophic false-absence errors without sending every case to the reasoner?**

The binary gate became three-way:

```text
present
absent
uncertain
```

Routing:

```text
present   → staged reasoning
absent    → deterministic unknown
uncertain → staged reasoning
```

The experiment started from the `v0.6` 100% false-absence stress condition.

A controlled protection signal converted selected false-absence decisions to `uncertain`.

### Accuracy / compute tradeoff

Gemma and Qwen produced the same curve:

| Protection | Residual false absences | Abstention rate | Model-call rate | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 6/6 | 0% | 0% | 40% |
| 25% | 4/6 | 20% | 20% | 60% |
| 50% | 3/6 | 30% | 30% | 70% |
| 75% | 1/6 | 50% | 50% | 90% |
| 100% | 0/6 | 60% | 60% | **100%** |

At full protection:

```text
6 present cases → model reasoning
4 absent cases  → zero-call deterministic unknown
```

### Finding

> **Abstention converts catastrophic false-negative gate errors into recoverable reasoning calls.**

This produced a direct accuracy-versus-compute tradeoff.

However, `v0.7` did **not** solve uncertainty estimation.

The protection signal was controlled by the experiment.

Therefore:

> **v0.7 measures the value of abstention, not the quality of an uncertainty estimator.**

Results:

```text
docs/results/v0_7_abstaining_gate.md
```

---

## v0.8 — Evidence-Derived Uncertainty

Branch:

```text
v0.8-evidence-uncertainty
```

Research question:

> **Can uncertainty be derived from extractor evidence itself rather than supplied by controlled oracle protection?**

The first approach estimated uncertainty from extractor stability under deterministic image perturbations.

---

### Phase 1: naive perturbation ensemble

Initial views:

```text
original
brightness_090
brightness_110
blur_060
downsample_075
```

Rule:

```text
unanimous present → present
unanimous absent  → absent
any disagreement  → uncertain
```

Initial result on both models:

```text
hard present: 0/10
hard absent:  4/10
uncertain:    6/10
model calls:  6/10
accuracy:     10/10
```

This looked promising but was misleading.

### Audit of the probes

Per-view extractor behavior:

| View | Present | Absent | Clean agreement |
| --- | ---: | ---: | ---: |
| original | 6 | 4 | 100% |
| brightness_090 | 0 | 10 | 40% |
| brightness_110 | 0 | 10 | 40% |
| blur_060 | 5 | 5 | 90% |
| downsample_075 | 6 | 4 | 100% |

Both brightness perturbations destroyed every present detection.

The apparent uncertainty signal was therefore mostly a perturbation-induced extractor failure.

### Negative finding

> **Perturbation disagreement is not automatically epistemic uncertainty. Destructive probes can manufacture abstention.**

This is an important measurement lesson:

> **The uncertainty probe must itself be validated before its output is trusted.**

---

### Phase 2: perturbation calibration

A dedicated audit evaluated candidate perturbations before using them for gating.

Candidate set:

```text
original
blur_020
blur_040
blur_060
downsample_090
downsample_075
downsample_060
```

Audit:

| View | Present | Absent | Agreement with clean |
| --- | ---: | ---: | ---: |
| original | 6 | 4 | 100% |
| blur_020 | 5 | 5 | 90% |
| blur_040 | 6 | 4 | 100% |
| blur_060 | 5 | 5 | 90% |
| downsample_090 | 6 | 4 | 100% |
| downsample_075 | 6 | 4 | 100% |
| downsample_060 | 6 | 4 | 100% |

Only one case was unstable:

```text
scene_0002
clean=present
blur_020=absent
blur_060=absent
```

This is qualitatively different from the brightness failure.

The instability is localized rather than global.

Audit script:

```text
scripts/audit_perturbation_stability.py
```

---

### Phase 3: calibrated evidence gate

Final calibrated views:

```text
original
blur_020
blur_040
blur_060
downsample_090
downsample_075
downsample_060
```

Condition:

```text
CONFLICT_EVIDENCE_UNCERTAINTY_CALIBRATED
```

Final gate distribution:

```text
5 hard present
4 hard absent
1 uncertain
```

Results:

| Model | Hard present | Hard absent | Uncertain | Model-call rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 4B | 5/10 | 4/10 | 1/10 | 60% | **100%** |
| Qwen3 4B | 5/10 | 4/10 | 1/10 | 60% | **100%** |

All labels:

```text
supported     100%
contradicted  100%
unknown       100%
```

Only 1/10 cases incurs uncertainty-specific escalation.

The other five model calls are normal present-target reasoning.

### Finding

> **An interpretable uncertainty signal can be derived from extractor stability, provided the perturbation probes themselves are calibrated not to induce systematic extractor failure.**

This moves DetectiveLab from:

```text
oracle protection
```

to:

```text
evidence-grounded abstention
```

without introducing a learned confidence model.

Results:

```text
docs/results/v0_8_evidence_uncertainty.md
```

---

# Results at a Glance

## Representation results

| Milestone | Condition | Key result |
| --- | --- | --- |
| v0.1 | RAW | 53.3% overall |
| v0.1 | ORACLE_STRUCTURED | 86.7% overall |
| v0.2 | EXTRACTED_STRUCTURED | 70.0% overall |
| v0.2 | EXTRACTED_FOCUSED | **86.7% overall** |

Core lesson:

> **Representation quality is not only about correctness; relevance and density matter.**

---

## Conflict-control results

| Milestone | Gemma | Qwen | Main lesson |
| --- | ---: | ---: | --- |
| staged conflict | 70% | 100% | native epistemic behavior differs |
| global epistemic rule | 100% | 90% | policy intervention is model-dependent |
| standalone LLM gate | 70% | 70% | gate can become a new bottleneck |
| extractor gate | **100%** | **100%** | reuse reliable structured facts directly |
| calibrated uncertainty gate | **100%** | **100%** | calibrated instability supports evidence-grounded abstention |

---

# Current Architecture

The current evidence/control pipeline is:

```text
scene.png
   ↓
deterministic synthetic extractor
   ↓
focused structured evidence
   ↓
presence stability across calibrated perturbations
   ↓
┌───────────────────────────────────────┐
│ stable absent                         │
│ → deterministic unknown               │
│ → zero language-model calls           │
├───────────────────────────────────────┤
│ stable present                        │
│ → staged conflict reasoning           │
├───────────────────────────────────────┤
│ unstable / uncertain                  │
│ → abstain from hard gate decision     │
│ → staged conflict reasoning           │
└───────────────────────────────────────┘
```

This architecture is intentionally not a learned router.

The current evidence does not justify one.

---

# Setup

The repository is designed to run locally on a Mac CPU.

Example environment:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run tests:

```bash
python -m pytest
```

Current expected result:

```text
114 passed
```

---

# Local Models

Experiments have been run through Ollama.

Known local models include:

```text
gemma3:4b
qwen3:4b-instruct-2507-q4_K_M
qwen3:8b
llama3.2:3b
```

The main controlled comparisons in the current research arc use:

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

The adapter uses Ollama's local generation API.

No fine-tuning is required.

---

# Running the Current v0.8 Experiment

Gemma:

```bash
python -m detectivelab.cli.evidence_uncertainty \
  --benchmark artifacts/benchmark_v0_0_1 \
  --model gemma3:4b \
  --num-predict 128 \
  --output artifacts/evaluation/v0_8_evidence_uncertainty_calibrated_gemma3_4b.jsonl
```

Qwen:

```bash
python -m detectivelab.cli.evidence_uncertainty \
  --benchmark artifacts/benchmark_v0_0_1 \
  --model qwen3:4b-instruct-2507-q4_K_M \
  --num-predict 128 \
  --output artifacts/evaluation/v0_8_evidence_uncertainty_calibrated_qwen3_4b.jsonl
```

Audit candidate perturbations:

```bash
python scripts/audit_perturbation_stability.py \
  --benchmark artifacts/benchmark_v0_0_1
```

Expected calibrated gate shape:

```text
Hard present: 5/10
Hard absent:  4/10
Uncertain:    1/10
Model calls:  6/10
Accuracy:     10/10
```

---

# Diagnostic Scripts

The repository includes targeted audits used to explain failures rather than merely report aggregate accuracy.

```text
scripts/audit_spatial.py
scripts/audit_conflict_staged.py
scripts/audit_epistemic_model_effect.py
scripts/audit_conditional_gate.py
scripts/audit_perturbation_stability.py
```

These are part of the experimental method.

A surprising benchmark score should trigger an audit before a new architecture is added.

---

# Research Evolution

| Milestone | Research question | Intervention | Status |
| --- | --- | --- | --- |
| v0.0 | Can the initial benchmark test grounded conflict reasoning? | initial benchmark | INVALIDATED |
| v0.0.1 | Can shortcut leakage be removed while freezing the task? | corrected benchmark | COMPLETE |
| v0.1 | Is failure primarily perception or reasoning? | raw vs oracle structure | COMPLETE |
| v0.2 | Can image-derived structure recover the oracle gap? | dense vs focused extractor output | COMPLETE |
| v0.3 | What conflict failure remains after perception is controlled? | staged reasoning + explicit missing-evidence rule | COMPLETE |
| v0.4 | Is the epistemic fix robust across wording, cases, and models? | robustness harnesses | COMPLETE |
| v0.5 | Can policy be applied conditionally without collateral regressions? | LLM gate vs extractor gate | COMPLETE |
| v0.6 | How brittle is representation-grounded control to gate errors? | directional corruption curves | COMPLETE |
| v0.7 | Can abstention trade compute for recovery? | controlled three-way gate | COMPLETE |
| v0.8 | Can uncertainty come from evidence rather than oracle protection? | calibrated extractor-stability signal | COMPLETE |
| next | Does instability predict real extraction failure? | controlled image degradation | NEXT |

The project trajectory is:

```text
benchmark
→ direct perception
→ oracle structure
→ extracted structure
→ focused representation
→ conflict arbitration
→ epistemic robustness
→ representation-grounded control
→ gate corruption
→ abstention
→ evidence-derived uncertainty
→ prospective uncertainty validation
```

---

# What the Evidence Supports So Far

The current benchmark supports the following conclusions.

### 1. Apparent reasoning failures can originate in perception

Oracle structure substantially outperformed raw visual input.

### 2. Correct structure can still be harmful when it is too dense

Dense pairwise spatial structure reduced reasoning quality despite being correct.

### 3. Focused image-derived structure can match oracle structure

On the current benchmark, task-relevant extracted evidence recovered essentially the full oracle gap.

### 4. Missing evidence requires explicit epistemic handling for some models

Gemma repeatedly treated absence as contradiction until the missing-evidence policy was made explicit.

### 5. Reasoning-policy interventions are model-dependent

The same global policy that fixed Gemma introduced regressions in Qwen.

### 6. A control gate can be less reliable than the reasoner it controls

The isolated LLM existence gate created false-absence errors not present in full staged reasoning.

### 7. Reusing reliable structured evidence can outperform re-inference

The extractor-derived presence gate restored 100% canonical conflict accuracy on both models.

### 8. Gate errors are directionally asymmetric

False absence suppresses evidence and is effectively unrecoverable downstream.

False presence can be recoverable.

### 9. Abstention can convert catastrophic control errors into recoverable computation

A three-way gate restored accuracy by escalating risky hard decisions.

### 10. Uncertainty probes must themselves be calibrated

Destructive perturbations can manufacture disagreement and create fake uncertainty.

### 11. Calibrated extractor instability can support evidence-grounded abstention

On the frozen benchmark, the final calibrated signal isolates one unstable case while preserving all stable absent and most stable present cases.

---

# What the Evidence Does Not Yet Support

DetectiveLab does **not** currently establish that:

- the synthetic extractor transfers to natural images;
- perturbation stability is a calibrated probability;
- the current perturbation set transfers to another visual domain;
- `scene_0002` is semantically ambiguous rather than near an extractor threshold;
- confidence thresholds have been optimized;
- learned routing is necessary;
- a larger model would remove the observed architectural effects;
- the current benchmark is large enough for broad generalization claims.

The benchmark is intentionally small and controlled.

The project prioritizes causal interpretability over scale.

---

# Next Milestone

The next research question should be:

> **Does calibrated extractor instability predict actual extraction failure under controlled image degradation?**

This is a prospective validation problem.

A clean next experiment would:

1. apply controlled image degradation at increasing severity;
2. measure when the extractor's clean presence decision becomes wrong;
3. measure whether the calibrated stability signal becomes uncertain before or at failure;
4. compute failure-detection precision and recall;
5. measure downstream accuracy and model-call cost under abstention.

Candidate degradation families:

```text
blur severity
resolution loss
occlusion
contrast reduction
localized corruption
```

The goal is not to add more architecture.

The goal is to determine whether the current uncertainty signal is **predictive**, rather than merely descriptive on the clean benchmark.

A learned router or learned confidence model should only be introduced if this simpler evidence-derived mechanism reaches a clear limit.

---

# Reproducibility Rules

DetectiveLab follows several repository-level rules.

### Freeze completed benchmarks

Do not rewrite `v0.0.1` to make a new experiment easier.

### Preserve negative results

Failed approaches remain documented when they reveal a real mechanism.

### Keep participant evidence separate from gold labels

Gold belongs in scoring artifacts, not participant-facing payloads.

### Prefer deterministic experiments

Use fixed seeds, deterministic synthetic generation, deterministic corruption subsets, and explicit model settings.

### Audit before escalating complexity

A surprising result should be decomposed before adding a new component.

### Do not add learned routing without evidence

Routing remains a hypothesis, not a default architecture.

---

# Repository Layout

Key paths:

```text
artifacts/
  benchmark_v0_0_1/

docs/
  results/
    v0_1_direct.md
    v0_2_extracted_structure.md
    v0_3_conflict_arbitration.md
    v0_4_epistemic_robustness.md
    v0_5_conditional_epistemic.md
    v0_6_gate_corruption.md
    v0_7_abstaining_gate.md
    v0_8_evidence_uncertainty.md

scripts/
  audit_spatial.py
  audit_conflict_staged.py
  audit_epistemic_model_effect.py
  audit_conditional_gate.py
  audit_perturbation_stability.py

src/detectivelab/
  adapters/
  cli/
  evaluation/
  extraction/

tests/
```

Evaluation JSONL outputs are treated as experiment artifacts and are not required to be committed to Git.

---

# Branch History

Current milestone branches include:

```text
main
v0.1-direct
v0.2-extracted-structure
v0.3-conflict-arbitration
v0.4-epistemic-robustness
v0.5-conditional-epistemic
v0.6-gate-corruption
v0.7-abstaining-gate
v0.8-evidence-uncertainty
```

Selected milestone commits:

```text
main                       9eb3a9c
v0.1-direct                3adb657
v0.2-extracted-structure   469ce47
v0.3-conflict-arbitration  65c29af
v0.4-epistemic-robustness  45c632c1
v0.5 implementation        1c47060
v0.5 README                35458c6
v0.7 implementation        b2b82a4
```

---

# Core Takeaway

DetectiveLab started as a comparison between raw and structured visual evidence.

The experiments now support a broader systems view:

> **Multimodal reliability depends not only on what evidence is available, but on how it is represented, how missing evidence is interpreted, where control decisions are made, which gate errors are allowed to suppress evidence, and whether uncertainty signals are themselves trustworthy.**

The current architecture remains deliberately simple:

```text
extract
→ focus
→ validate stability
→ hard decision when stable
→ abstain when unstable
→ reason only when needed
```

That simplicity is intentional.

The next component should be added only when the current evidence says it is necessary.
