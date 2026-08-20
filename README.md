# DetectiveLab

DetectiveLab is a controlled multimodal research project for testing when an AI system should reason from raw visual evidence, structured observations, or both.

Evidence before complexity is the project rule: freeze the benchmark, add one capability at a time, measure it, and only keep additional machinery when the evidence justifies it.

## At a Glance

- **Research question:** when multimodal evidence conflicts, should an AI reason from raw perception, structured observations, or both?
- **Current milestone:** `v0.0.1-benchmark-fix` is **VALIDATED**; rerun the QUESTION baseline before RAW.
- **Current benchmark:** 10 deterministic scenes, 30 total items, and three case families: spatial, state, and conflict.
- **Preserved history:** `v0.0` remains frozen for provenance but its conflict family is invalidated for QUESTION-only shortcut leakage.
- **Compute constraint:** the reference benchmark and evaluation workflow must remain runnable on a consumer Mac CPU without required model training.
- **Next experiment:** rerun the `v0.1-direct` QUESTION-only baseline on `v0.0.1`; run RAW only if the conflict shortcut is gone.
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

The long-term hypothesis is that different case requirements may favor different evidence paths, but routing will only be introduced if earlier controlled experiments demonstrate a stable advantage over an always-hybrid baseline.

## Current Findings

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


# v0.1 Ollama adapter

Copy `src/` and `tests/` into the repository root, preserving paths.

No new Python dependency is required. The adapter uses Python's standard-library HTTP client and expects a local Ollama server on `http://localhost:11434`.

Run tests:

```bash
python -m pytest
```

Run the first real QUESTION baseline:

```bash
python -m detectivelab.cli.evaluate \
  --benchmark artifacts/benchmark_v0_0_1 \
  --condition QUESTION \
  --adapter ollama \
  --model qwen3:4b-instruct-2507-q4_K_M \
  --output artifacts/evaluation/v0_1_question_qwen3_4b.jsonl
```

Defaults are frozen for this first pass:

- temperature: `0.0`
- output budget: `8` tokens
- seed: `0`
- Ollama URL: `http://localhost:11434`
- thinking: disabled

The adapter intentionally rejects `RAW` for now. That keeps `v0.1-direct` focused on validating the text-only QUESTION baseline before image transport is introduced.


## Benchmark Design

Each generated scene has a canonical hidden state that is never inferred from the rendered image. The renderer produces the participant-facing visual scene, while questions and answers are derived mechanically from the hidden state.

The three current case families are:

- **spatial:** tests relationships such as left/right placement from the rendered scene
- **state:** tests directly observable physical states such as open/closed or intact/broken
- **conflict:** tests whether witness testimony is contradicted by, or unresolved against, physical evidence under an explicit case rule

Family-specific participant payloads are intentionally separated so irrelevant evidence cannot leak answers. For example, spatial and state items do not receive witness testimony, while conflict items receive the testimony and exact rule required for arbitration.

## Research Evolution

> benchmark -> direct perception -> oracle structure -> extracted structure -> hybrid evidence -> robustness -> routing only if justified
>
> evidence before complexity

| Version | Research question | Capability introduced | Status |
| --- | --- | --- | --- |
| `v0.0` | Can we create a deterministic multimodal benchmark without obvious shortcut leakage? | Synthetic scene schema, renderer, question generation, provenance, audits | **PRESERVED / CONFLICT INVALIDATED** |
| `v0.0.1` | Can we remove the conflict-rule shortcut without changing the research question? | Constant rule, image-dependent 3-way conflict verdicts, leakage guards | **VALIDATED / QUESTION RERUN NEXT** |
| `v0.1` | How much can the model solve from priors versus raw visual evidence? | QUESTION-only and RAW-image evaluation harness | **IN PROGRESS** |
| `v0.2` | How much failure comes from perception versus reasoning? | Oracle structured visual state | Planned |
| `v0.3` | Can explicit perception close the oracle gap? | Extracted structured evidence | Planned |
| `v0.4` | Does retaining both pixels and structure improve robustness? | RAW + STRUCTURED hybrid path | Planned |
| `v0.5` | When does structured evidence become brittle? | Controlled evidence corruption | Planned |
| `v0.6` | Do different case types justify different evidence paths? | Routing, only if prior results justify it | Conditional |

## Engineering Highlights

- standard-library-first typed Python domain model
- deterministic seed-based scene generation
- CPU-light Pillow rendering
- closed-form answers and deterministic scoring targets
- family-specific participant payloads to reduce leakage
- per-case provenance and SHA-256 hashes
- reproducible benchmark export
- automated schema, rendering, generation, and benchmark tests
- manual and blind visual audits before freeze
- explicit benchmark-version freeze rule

## Repository Structure

- `README.md`: project overview, current status, and local workflow
- `DETECTIVELAB_PROJECT.md`: project charter, computational guardrails, and anti-drift rules
- `src/detectivelab/`: domain schema, deterministic generation, rendering, export, and validation code
- `tests/`: offline regression coverage for the frozen benchmark machinery
- `artifacts/benchmark_v0_0/`: preserved original benchmark; conflict family invalidated after leakage detection
- `artifacts/benchmark_v0_0_1/`: corrected benchmark used for the next evaluation gate
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

The current branch should report:

```text
41 passed
```

## Reproducing the Frozen Benchmark Checks

Run the benchmark integrity and metadata check with:

```bash
python -m detectivelab.validate artifacts/benchmark_v0_0_1
```

A healthy frozen benchmark reports:

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

You can also inspect the frozen metadata directly:

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

Top-level freeze artifacts include:

- `artifacts/benchmark_v0_0/manifest.json`
- `artifacts/benchmark_v0_0/AUDIT.json`
- `artifacts/benchmark_v0_0/FINAL_BLIND_AUDIT.md`
- `artifacts/benchmark_v0_0/FINAL_BLIND_CONTACT.png`

## Limitations

- `v0.0` is intentionally tiny: 10 scenes and 30 items are enough to validate the protocol, not to support broad claims about multimodal intelligence.
- The rendered scenes are synthetic and deliberately simple.
- Current state questions cover only visually self-evident physical states; ambiguous symbolic conventions were removed before freeze.
- No model-performance conclusions have been established yet.
- The benchmark does not currently cover OCR, natural photographs, video, audio, tool use, or open-ended generation.

## What the Current Evidence Suggests

The project has already reinforced a few methodological rules:

- benchmark leakage can appear through label priors as easily as through prompt text
- a visually deterministic benchmark can still fail a blind human audit if its symbols require learned conventions
- participant-facing payloads need their own validation contract, separate from hidden-state correctness
- provenance is only credible if current source code can regenerate the frozen artifacts
- model complexity should not be added until the benchmark itself is trustworthy

In short, DetectiveLab currently suggests that multimodal architecture experiments should begin by controlling representation and evidence flow before comparing sophisticated models.

## Future Research

- rerun the QUESTION-only baseline on corrected `v0.0.1`; run RAW only after the conflict shortcut gate passes
- compare RAW inference with oracle structured evidence to separate perception from reasoning error
- introduce explicit visual extraction only after the oracle gap is measured
- test whether hybrid RAW + STRUCTURED evidence improves conflict handling or merely adds redundancy
- corrupt extracted evidence systematically to measure brittleness and error propagation
- introduce evidence-path routing only if different case families show stable, meaningful path preferences
- scale from 10 to 30 scenes only after the evaluation harness is trustworthy

## Freeze Rule

Any later change to scene semantics, rendering conventions, participant payload composition, labels, or benchmark-generation behavior requires a **new benchmark version**.

Do not rewrite `v0.0`. Benchmark corrections require a new version; `v0.0.1` is the first such correction.



# DetectiveLab v0.2 Oracle Structured update

Drop these files into the repo root while on `v0.1-direct`:

```text
src/detectivelab/evaluation/runner.py
src/detectivelab/evaluation/structured.py
tests/test_oracle_structured.py
```

The update adds a third evaluation condition:

```text
ORACLE_STRUCTURED
```

It derives participant-safe symbolic evidence from each frozen `scene.json` and sends no image to the model. The same adapter, model, decoding settings, payload context, questions, and scoring remain unchanged.

Run:

```bash
python -m pytest
```

Expected after this update:

```text
46 passed
```

Then evaluate:

```bash
python -m detectivelab.cli.evaluate \
  --benchmark artifacts/benchmark_v0_0_1 \
  --condition ORACLE_STRUCTURED \
  --adapter ollama \
  --model gemma3:4b \
  --output artifacts/evaluation/v0_2_oracle_structured_gemma3_4b.jsonl
```

Note: your local `src/detectivelab/cli/evaluate.py` must not contain the old QUESTION-only Ollama guard. You already removed that guard when promoting RAW support.


# v0.2 Extracted Structure Update

This drop-in update adds the `EXTRACTED_STRUCTURED` evaluation condition.

## Files

Copy these paths into the repository root:

```text
src/detectivelab/extraction/__init__.py
src/detectivelab/extraction/base.py
src/detectivelab/extraction/synthetic.py
src/detectivelab/evaluation/runner.py
tests/test_extracted_structured.py
```

## What the extractor does

The reference extractor is deliberately synthetic-specific and CPU-light. It reads only `scene.png` and reverses DetectiveLab's small rendering grammar using:

- connected visual components,
- deterministic component merging,
- renderer-template matching for object kind/state,
- pixel color recovery,
- image-space center ordering for spatial relations.

It does **not** read `scene.json`, seeds, object IDs, provenance, or gold labels at runtime.

On the frozen 10-scene `v0.0.1` slice, the extractor reconstructs all visible object color/kind/state tuples exactly. This is a reference extraction ceiling for the synthetic renderer, not a claim about natural-image perception.

## Verify

```bash
python -m pytest
```

Expected:

```text
51 passed
```

## Run

```bash
python -m detectivelab.cli.evaluate \
  --benchmark artifacts/benchmark_v0_0_1 \
  --condition EXTRACTED_STRUCTURED \
  --adapter ollama \
  --model gemma3:4b \
  --output artifacts/evaluation/v0_2_extracted_structured_gemma3_4b.jsonl
```

Compare against:

```text
RAW                53.3%
ORACLE_STRUCTURED  86.7%
EXTRACTED_STRUCTURED ?
```


# EXTRACTED_FOCUSED update

Adds a task-focused representation ablation on top of the existing image-only synthetic extractor.

## New condition

`EXTRACTED_FOCUSED`

- spatial: exposes only the two queried objects plus their extracted left/right relation
- state: exposes only the queried object's extracted state
- conflict: exposes only the object named in testimony, or `not present` when the extractor cannot find it

The formatter uses participant-facing question/testimony text only to select relevant entities. All visual facts come from `scene.png` through the existing extractor. It does not read `scene.json`, gold labels, object IDs, seeds, or provenance.

## Files

- `src/detectivelab/evaluation/focused.py`
- `src/detectivelab/evaluation/runner.py`
- `tests/test_extracted_focused.py`

## Test

```bash
python -m pytest
```

Expected in the patched repo:

```text
57 passed
```

## Run

```bash
python -m detectivelab.cli.evaluate \
  --benchmark artifacts/benchmark_v0_0_1 \
  --condition EXTRACTED_FOCUSED \
  --adapter ollama \
  --model gemma3:4b \
  --output artifacts/evaluation/v0_2_extracted_focused_gemma3_4b.jsonl
```
