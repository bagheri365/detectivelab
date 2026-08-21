# DetectiveLab

DetectiveLab is a controlled multimodal research project for testing when an AI system should reason from raw visual evidence, structured observations, or both.

Evidence before complexity is the project rule: freeze the benchmark, add one capability at a time, measure it, and only keep additional machinery when the evidence justifies it.

## At a Glance

- **Research question:** when multimodal evidence conflicts, should an AI reason from raw perception, structured observations, or both?
- **Current milestone:** `v0.4-epistemic-robustness` is **COMPLETE**.
- **Current benchmark:** `v0.0.1`, with 10 deterministic scenes, 30 total items, and three case families: spatial, state, and conflict.
- **Best current result:** the explicit epistemic policy reaches **100%** on Gemma across canonical conflict, paraphrase robustness, and controlled case variation; however, Qwen shows model-specific regressions under the same intervention.
- **Key finding:** task-relevant image-derived structure can match oracle performance, while dense correct structure can materially degrade reasoning.
- **Key v0.4 finding:** the missing-evidence rule is highly effective for Gemma but not universally beneficial; on Qwen it improves `unknown` handling while sometimes introducing contradiction-to-support errors.
- **Compute constraint:** the reference benchmark and evaluation workflow must remain runnable on a consumer Mac CPU without required model training.
- **Next experiment:** test whether the epistemic rule can be applied conditionally only after `EXISTENCE: absent`, preserving its benefit while avoiding collateral regressions on present-target cases.
- **Project charter:** see [`DETECTIVELAB_PROJECT.md`](./DETECTIVELAB_PROJECT.md).

## Why This Project Exists

DetectiveLab is not a generic visual-question-answering or detective-game application. It is a research system for studying a specific architecture question:

> When should visual evidence remain raw, and when should it become explicit structured evidence before reasoning?

The project uses small synthetic detective scenes so that visual state, witness claims, case rules, and ground-truth answers can be controlled exactly. This makes it possible to separate perception failures from reasoning failures instead of comparing end-to-end systems with many confounded differences.

## Research Question

DetectiveLab follows the progression:

`frozen benchmark -> simplest baseline -> measurable failure -> add one capability -> evaluate -> retain / retain experimentally / reject`

The system is designed to separate three things that are often conflated:

- the quality of visual perception
- the quality of reasoning over explicit evidence
- the quality of evidence arbitration when sources disagree

The long-term hypothesis is that different case requirements may favor different evidence paths, but routing will only be introduced if earlier controlled experiments demonstrate a stable advantage over a simpler fixed path.

## Current Findings

The current controlled comparison uses the same `v0.0.1` benchmark and `gemma3:4b` model across five evidence conditions:

| Condition | Overall | Conflict | Spatial | State |
| --- | ---: | ---: | ---: | ---: |
| QUESTION | 50.0% | 30.0% | 70.0% | 50.0% |
| RAW | 53.3% | 30.0% | 50.0% | 80.0% |
| EXTRACTED_STRUCTURED | 70.0% | 60.0% | 50.0% | 100.0% |
| EXTRACTED_FOCUSED | **86.7%** | **60.0%** | **100.0%** | **100.0%** |
| ORACLE_STRUCTURED | **86.7%** | **60.0%** | **100.0%** | **100.0%** |

The strongest `v0.2` result was:

> **Focused image-derived structure matched oracle performance, while dense correct structure degraded reasoning.**

This means the important distinction is not only raw pixels versus symbolic structure. Representation density also matters.

`v0.3` then isolates the remaining conflict bottleneck:

| Conflict condition | Accuracy |
| --- | ---: |
| QUESTION | 30% |
| RAW | 30% |
| EXTRACTED_FOCUSED | 60% |
| CONFLICT_STAGED | 70% |
| CONFLICT_EPISTEMIC | **100%** |

The staged diagnostic decomposes conflict handling into target existence, physical state, testimony/evidence agreement, and final verdict. Under the explicit epistemic rule, all four stages score **10/10** in the semantic audit.

The resulting failure mechanism is specific:

> **The model often treated an absent target as contradictory evidence instead of insufficient evidence.**

Making the policy explicit—

```text
if EXISTENCE = absent:
  PHYSICAL_STATE = not_applicable
  AGREEMENT = unknown
  VERDICT = unknown
```

—removed the remaining conflict errors on the current 10-item slice.


## v0.4 Robustness Findings

`v0.4-epistemic-robustness` tests the `v0.3` missing-evidence rule across three axes:

1. testimony paraphrases
2. controlled case variation
3. model variation

The same two policies are compared throughout:

- `CONFLICT_STAGED`
- `CONFLICT_EPISTEMIC`

### Gemma 3 4B

Gemma remains strongly sensitive to missing-evidence cases under the staged policy, but the explicit epistemic rule is robust across wording and case variation.

| Slice | Staged | Epistemic |
| --- | ---: | ---: |
| canonical conflict | 70.0% | **100.0%** |
| paraphrase robustness | 66.7% | **100.0%** |
| case variation | 66.7% | **100.0%** |

The label-level pattern is especially clear:

- paraphrases: `unknown` improves from 16.7% to 100%
- case variation: `unknown` improves from 0% to 100%
- supported and contradicted cases remain at 100% under the epistemic policy

For Gemma, the intervention is therefore stable across the tested surface-form and case changes.

### Qwen3 4B Instruct

The second-model check reveals an important boundary condition.

| Slice | Staged | Epistemic |
| --- | ---: | ---: |
| canonical conflict | **100.0%** | 90.0% |
| paraphrase robustness | 86.7% | **93.3%** |
| case variation | **100.0%** | 93.3% |

On Qwen, the epistemic rule consistently protects `unknown` cases, but it can hurt `contradicted` cases:

- paraphrases: `unknown` rises from 75.0% to 100%, while `contradicted` falls from 88.9% to 77.8%
- case variation: `unknown` stays at 100%, while `contradicted` falls from 100% to 80%

A direct audit confirms that these regressions are semantic rather than parser artifacts. The repeated failure changes:

```text
EXISTENCE: present
PHYSICAL_STATE: open
AGREEMENT: contradicts
VERDICT: contradicted
```

into:

```text
EXISTENCE: present
PHYSICAL_STATE: open
AGREEMENT: supports
VERDICT: supported
```

under the epistemic prompt.

The resulting conclusion is:

> **The epistemic intervention is failure-mode-specific rather than universally beneficial. A rule that fixes one model can over-constrain another model that does not share the same native error profile.**

See [`docs/results/v0_4_epistemic_robustness.md`](./docs/results/v0_4_epistemic_robustness.md).

### State

State accuracy progresses:

`50% QUESTION -> 80% RAW -> 100% EXTRACTED -> 100% ORACLE`

The model can reason correctly over state facts once those facts are represented explicitly. The remaining RAW gap is therefore primarily perceptual on this benchmark.

### Spatial

Spatial accuracy progresses:

`70% QUESTION -> 50% RAW -> 50% DENSE EXTRACTED -> 100% FOCUSED EXTRACTED -> 100% ORACLE`

A direct audit showed that the image-only extractor was recovering the relevant left/right relation correctly. The failure came from exposing all pairwise relations among detected objects.

With six objects, the dense condition emitted 15 pairwise spatial statements. The relevant relation was present, but surrounded by irrelevant correct structure. The model then developed a strong `yes` bias.

When the evidence was reduced to the queried entities and their relevant relation, spatial accuracy rose to 100%.

This supports the working principle:

> **More correct structure is not necessarily better structure.**

### Conflict

Conflict accuracy now progresses:

`30% QUESTION -> 30% RAW -> 60% EXTRACTED_FOCUSED -> 70% CONFLICT_STAGED -> 100% CONFLICT_EPISTEMIC`

The structured conditions first close the measurable perception gap, but direct conflict verdicting remains imperfect.

The staged diagnostic then shows where the remaining failure occurs:

- target existence: 90% in the baseline staged audit
- physical state: 90%
- agreement: 70%
- final verdict: 60% overall in the original semantic audit

The dominant error occurs on `unknown` cases. The model can correctly recognize:

```text
EXISTENCE: absent
PHYSICAL_STATE: not_applicable
```

but still produce:

```text
AGREEMENT: contradicts
VERDICT: contradicted
```

instead of propagating uncertainty.

`CONFLICT_EPISTEMIC` adds only an explicit rule stating that absence of physical evidence implies `unknown`, not contradiction. With that rule, the semantic audit reaches:

- existence: 100%
- physical state: 100%
- agreement: 100%
- verdict: 100%

**Working interpretation:**

> The residual conflict bottleneck was epistemic rather than perceptual: the model needed an explicit policy for distinguishing contradiction from insufficient evidence.

See:

- [`docs/results/v0_1_direct.md`](./docs/results/v0_1_direct.md)
- [`docs/results/v0_2_extracted_structure.md`](./docs/results/v0_2_extracted_structure.md)
- [`docs/results/v0_3_conflict_arbitration.md`](./docs/results/v0_3_conflict_arbitration.md)

## Benchmark History

The first real QUESTION-only run exposed a benchmark bug before any RAW-image claim was made. On preserved `v0.0`, Qwen3 4B scored:

- spatial: 50%
- state: 50%
- conflict: 100%

The conflict family was therefore **invalidated for shortcut leakage**: its verdict could be inferred from case-rule wording without seeing the scene. `v0.0` remains preserved as the original frozen artifact and should not be rewritten.

`v0.0.1` corrects that flaw while keeping the benchmark small and deterministic:

- 10 deterministic scenes (`seed=0..9`)
- 30 total items: one spatial, one state, and one conflict item per scene
- spatial: 5 `yes`, 5 `no`
- state: 5 `yes`, 5 `no`
- conflict: 3 `supported`, 3 `contradicted`, 4 `unknown`
- one identical conflict rule across all scenes
- STATE questions generated independently from witness testimony
- deterministic CPU-light rendering
- participant-facing family-specific payloads
- per-case provenance and SHA-256 hashes
- automated validation status: `PASS`

See [`artifacts/benchmark_v0_0_1/BENCHMARK_FIX.md`](./artifacts/benchmark_v0_0_1/BENCHMARK_FIX.md) for the correction record.

## Benchmark Design

Each generated scene has a canonical hidden state that is never inferred from the rendered image. The renderer produces the participant-facing visual scene, while questions and answers are derived mechanically from the hidden state.

The three current case families are:

- **spatial:** tests relationships such as left/right placement from the rendered scene
- **state:** tests directly observable physical states such as open/closed or intact/broken
- **conflict:** tests whether witness testimony is supported, contradicted, or unresolved by physical evidence under a constant case rule

Family-specific participant payloads are intentionally separated so irrelevant evidence cannot leak answers. Spatial and state items do not receive witness testimony, while conflict items receive only the testimony and rule required for arbitration.

## Evidence Conditions

DetectiveLab currently supports five evaluation conditions:

| Condition | Evidence path |
| --- | --- |
| `QUESTION` | participant-facing text only |
| `RAW` | participant-facing text + rendered scene image |
| `ORACLE_STRUCTURED` | correct symbolic facts from hidden benchmark state |
| `EXTRACTED_STRUCTURED` | dense symbolic facts recovered from `scene.png` only |
| `EXTRACTED_FOCUSED` | task-relevant subset of image-derived symbolic facts |
| `CONFLICT_STAGED` | focused conflict evidence + explicit intermediate decisions |
| `CONFLICT_EPISTEMIC` | staged conflict reasoning + explicit missing-evidence policy |
| robustness harnesses | paraphrase, case-variation, and cross-model checks over staged vs epistemic policies |

`ORACLE_STRUCTURED` is a diagnostic upper bound, not a deployable perception system.

`EXTRACTED_STRUCTURED` and `EXTRACTED_FOCUSED` both use a deterministic CPU-light extractor that reads only `scene.png`.

The focused formatter may use participant-facing task text to select relevant extracted facts, but it does not read `scene.json`, gold labels, object IDs, seeds, or provenance.

## Research Evolution

> benchmark -> direct perception -> oracle structure -> extracted structure -> focused representation -> conflict arbitration -> epistemic robustness -> conditional policy control -> hybrid / routing only if justified
>
> evidence before complexity

| Version | Research question | Capability introduced | Status |
| --- | --- | --- | --- |
| `v0.0` | Can we create a deterministic multimodal benchmark without obvious shortcut leakage? | Synthetic scene schema, renderer, question generation, provenance, audits | **PRESERVED / CONFLICT INVALIDATED** |
| `v0.0.1` | Can we remove the conflict-rule shortcut without changing the research question? | Constant rule, image-dependent 3-way conflict verdicts, leakage guards | **VALIDATED** |
| `v0.1-direct` | How much can the model solve from priors versus raw visual evidence? | QUESTION and RAW evaluation harness; oracle diagnostic added | **COMPLETE** |
| `v0.2-extracted-structure` | Can explicit image-derived structure recover the oracle gap? | Dense and focused image-only structured evidence | **COMPLETE** |
| `v0.3-conflict-arbitration` | Why does conflict remain below ceiling after perception is controlled? | Staged reasoning + explicit epistemic policy | **COMPLETE** |
| `v0.4-epistemic-robustness` | Does the epistemic policy survive wording, case, and model variation? | Paraphrase, case-variation, and second-model robustness checks | **COMPLETE** |
| next | Can the epistemic rule be applied only when absence is detected? | Conditional policy application after the existence stage | **NEXT** |
| later | Does retaining both pixels and structure improve robustness? | RAW + STRUCTURED hybrid path | Conditional |
| later | When does structured evidence become brittle? | Controlled evidence corruption | Conditional |
| later | Do different case types justify different evidence paths? | Routing | Conditional |

## Engineering Highlights

- standard-library-first typed Python domain model
- deterministic seed-based scene generation
- CPU-light Pillow rendering
- closed-form answers and deterministic scoring targets
- family-specific participant payloads to reduce leakage
- per-case provenance and SHA-256 hashes
- reproducible benchmark export
- resumable JSONL evaluation harness
- local Ollama adapter with deterministic decoding defaults
- RAW image transport through Ollama
- oracle structured evidence formatter
- deterministic image-only reference extractor
- focused representation ablation
- staged conflict reasoning diagnostic
- semantic stage audit with alias/suffix normalization
- explicit epistemic missing-evidence rule
- deterministic paraphrase robustness harness
- image-only controlled case-variation harness
- cross-model staged-vs-epistemic audit
- automated leakage and hidden-state access tests
- manual and blind visual audits before freeze
- explicit benchmark-version freeze rule

## Repository Structure

- `README.md`: project overview, current status, and local workflow
- `DETECTIVELAB_PROJECT.md`: project charter, computational guardrails, and anti-drift rules
- `src/detectivelab/`: domain, generation, rendering, extraction, evaluation, adapters, export, and validation code
- `scripts/`: targeted diagnostics such as conflict and spatial audits
- `tests/`: offline regression coverage for benchmark and evaluation behavior
- `docs/results/`: milestone-level experimental results
- `artifacts/benchmark_v0_0/`: preserved original benchmark; conflict family invalidated after leakage detection
- `artifacts/benchmark_v0_0_1/`: corrected benchmark used for current experiments
- `artifacts/evaluation/`: local evaluation outputs; JSONL runs are ignored by Git
- `pyproject.toml`: packaging and development dependency configuration

## Running Locally

Start from the repository root.

On macOS with Homebrew Python 3.12:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
```

DetectiveLab requires Python 3.11 or newer.

After the virtual environment has been created once, future sessions only need:

```bash
source .venv/bin/activate
```

The current `v0.4-epistemic-robustness` branch should report:

```text
75 passed
```

## Reproducing the Benchmark Checks

Run the benchmark integrity and metadata check with:

```bash
python -m detectivelab.validate artifacts/benchmark_v0_0_1
```

A healthy benchmark reports:

```text
Benchmark: artifacts/benchmark_v0_0_1
Version: v0.0.1
Scenes: 10
Items: 30
Required files: PASS
Manifest/AUDIT consistency: PASS
Audit status: PASS
Status: PASS
```

You can also inspect the benchmark metadata directly:

```bash
cat artifacts/benchmark_v0_0_1/AUDIT.json
cat artifacts/benchmark_v0_0_1/manifest.json
```

Each scene directory contains:

```text
scene_0000/
├── scene.json
├── scene.png
├── questions.json
├── payloads.json
└── provenance.json
```

## Running the Current Experiments

A local Ollama server is required for model-backed runs. The current reference model is `gemma3:4b`.

QUESTION:

```bash
python -m detectivelab.cli.evaluate   --benchmark artifacts/benchmark_v0_0_1   --condition QUESTION   --adapter ollama   --model gemma3:4b   --output artifacts/evaluation/v0_1_question_gemma3_4b.jsonl
```

RAW:

```bash
python -m detectivelab.cli.evaluate   --benchmark artifacts/benchmark_v0_0_1   --condition RAW   --adapter ollama   --model gemma3:4b   --output artifacts/evaluation/v0_1_raw_gemma3_4b.jsonl
```

ORACLE_STRUCTURED:

```bash
python -m detectivelab.cli.evaluate   --benchmark artifacts/benchmark_v0_0_1   --condition ORACLE_STRUCTURED   --adapter ollama   --model gemma3:4b   --output artifacts/evaluation/v0_2_oracle_structured_gemma3_4b.jsonl
```

EXTRACTED_STRUCTURED:

```bash
python -m detectivelab.cli.evaluate   --benchmark artifacts/benchmark_v0_0_1   --condition EXTRACTED_STRUCTURED   --adapter ollama   --model gemma3:4b   --output artifacts/evaluation/v0_2_extracted_structured_gemma3_4b.jsonl
```

EXTRACTED_FOCUSED:

```bash
python -m detectivelab.cli.evaluate   --benchmark artifacts/benchmark_v0_0_1   --condition EXTRACTED_FOCUSED   --adapter ollama   --model gemma3:4b   --output artifacts/evaluation/v0_2_extracted_focused_gemma3_4b.jsonl
```

Paraphrase robustness:

```bash
python -m detectivelab.cli.paraphrase_robustness \
  --benchmark artifacts/benchmark_v0_0_1 \
  --policy epistemic \
  --model gemma3:4b \
  --num-predict 128 \
  --output artifacts/evaluation/v0_4_paraphrase_epistemic_gemma3_4b.jsonl
```

Case-variation robustness:

```bash
python -m detectivelab.cli.case_robustness \
  --benchmark artifacts/benchmark_v0_0_1 \
  --policy epistemic \
  --model gemma3:4b \
  --num-predict 128 \
  --output artifacts/evaluation/v0_4_case_variation_epistemic_gemma3_4b.jsonl
```

The same robustness commands can be rerun with:

```text
qwen3:4b-instruct-2507-q4_K_M
```

to reproduce the second-model comparison.

Current deterministic defaults:

- temperature: `0.0`
- output budget: `8` tokens
- seed: `0`
- Ollama URL: `http://localhost:11434`
- thinking: disabled

Evaluation JSONL outputs are local artifacts and should not be committed.

## Diagnostic Scripts

Conflict audit:

```bash
python scripts/audit_conflicts.py
```

Conflict perception/comparison probe:

```bash
python scripts/probe_conflict.py
```

Spatial dense-vs-focused audit:

```bash
python scripts/audit_spatial.py
```

Cross-model policy-effect audit:

```bash
python scripts/audit_epistemic_model_effect.py \
  --canonical-staged artifacts/evaluation/v0_4_conflict_staged_qwen3_4b.jsonl \
  --canonical-epistemic artifacts/evaluation/v0_4_conflict_epistemic_qwen3_4b.jsonl \
  --paraphrase-staged artifacts/evaluation/v0_4_paraphrase_staged_qwen3_4b.jsonl \
  --paraphrase-epistemic artifacts/evaluation/v0_4_paraphrase_epistemic_qwen3_4b.jsonl \
  --case-staged artifacts/evaluation/v0_4_case_variation_staged_qwen3_4b.jsonl \
  --case-epistemic artifacts/evaluation/v0_4_case_variation_epistemic_qwen3_4b.jsonl
```

These scripts exist to diagnose measured failures. They are not separate benchmark conditions.

## Limitations

- The benchmark is intentionally tiny: 10 scenes and 30 items are enough to validate mechanisms, not to support broad claims about multimodal intelligence.
- The rendered scenes are synthetic and deliberately simple.
- The reference extractor is specific to the DetectiveLab rendering grammar and is not a natural-image perception system.
- The current result is based primarily on one multimodal model, `gemma3:4b`.
- The current percentages are descriptive and should not be treated as statistically stable population estimates.
- Current state questions cover only visually self-evident physical states.
- The benchmark does not currently cover OCR, natural photographs, video, audio, tool use, or open-ended generation.
- `ORACLE_STRUCTURED` is an upper-bound diagnostic and should not be interpreted as a deployable architecture.
- `EXTRACTED_FOCUSED` uses task text to select relevant extracted facts; this is deliberate and should be distinguished from blind scene summarization.
- `CONFLICT_EPISTEMIC` encodes the benchmark's missing-evidence policy explicitly; its 100% Gemma result is a controlled mechanism result, not evidence of broad uncertainty calibration.
- Cross-model results show that the same explicit rule can introduce regressions on a model with a different native error profile.

## What the Current Evidence Suggests

The project currently supports several methodological lessons:

- benchmark leakage can appear through label priors and rule wording, not only direct answer leakage
- participant-facing payloads need their own validation contract
- multimodal failure is not a single category: grounding, perception, comparison, and verdict mapping can fail independently
- correct structured evidence can outperform raw visual reasoning
- dense correct structure can still hurt reasoning
- task-relevant structure can recover the full measured oracle gap on the current slice
- perception can be fully controlled while evidence arbitration remains difficult
- missing evidence and contradictory evidence are distinct epistemic states, and some models may need that distinction made explicit
- reasoning interventions can be model-specific: a policy that fixes one model can degrade another
- model complexity should only be added after the previous experiment identifies a measured bottleneck

In short:

> **Representation choice matters twice: first in whether visual evidence becomes explicit structure, and again in how much of that structure is exposed to the reasoner.**

## Next Milestone

### Conditional epistemic policy application

`v0.4-epistemic-robustness` is complete.

The next research question is:

> Can the missing-evidence rule be applied only after the system has already determined that the target is absent?

This is motivated directly by the cross-model result.

The current global epistemic prompt helps Gemma substantially, but on Qwen it can alter present-target contradiction judgments that the rule was never intended to govern.

The next experiment should therefore keep the staged decomposition but apply the uncertainty rule only when:

```text
EXISTENCE = absent
```

The goal is to test whether a deterministic stage gate can:

1. preserve the unknown-case benefit
2. avoid changing present-target supported/contradicted reasoning
3. reduce the Qwen contradiction-to-support regression
4. remain simple enough to justify before any routing or learned control

No learned router, hybrid RAW+STRUCTURED path, fine-tuning, or larger orchestration should be added yet.

## Freeze Rule

Any later change to scene semantics, rendering conventions, participant payload composition, labels, or benchmark-generation behavior requires a **new benchmark version**.

Do not rewrite `v0.0`.

Benchmark corrections require a new version; `v0.0.1` is the corrected benchmark used for current experiments.
