# DetectiveLab

> **Research question:** When multimodal evidence conflicts, should an AI reason from raw perception, structured observations, or both?

DetectiveLab is a small, controlled multimodal research project disguised as a fictional detective puzzle benchmark. It studies **evidence representation and arbitration**, not generic visual question answering and not agentic orchestration.

The project is intentionally constrained so the full reference experiment can run on a **consumer Mac CPU**.

---

## 1. Project Thesis

Different case types may benefit from different evidence representations:

- **RAW** visual evidence may be best when geometry, occlusion, viewpoint, or spatial relationships matter.
- **STRUCTURED** evidence may be best when reasoning depends on discrete state such as open/closed, present/missing, or object attributes.
- **HYBRID** evidence may be best when sources conflict or an extracted interpretation needs verification against the original image.

The project must **test** this hypothesis, not assume it.

Routing is justified only if different evidence paths show stable, meaningful advantages over a strong always-hybrid baseline.

---

## 2. What DetectiveLab Is Actually Studying

The central architectural choice is:

```text
raw perception -> reasoning
```

versus:

```text
raw perception -> explicit representation -> reasoning
```

versus:

```text
                 +-> raw evidence --------+
image -----------|                        |-> reasoner
                 +-> structured evidence -+
```

The detective theme provides controlled scenarios in which physical evidence, witness testimony, and case rules can agree or conflict.

The serious research problem is:

> **How should an AI system represent and arbitrate competing sources of evidence?**

---

## 3. Non-Goals

DetectiveLab is **not** intended to become:

- a generic VQA benchmark;
- an image-captioning project;
- an OCR benchmark;
- a real-world forensic system;
- a document-understanding system;
- a video or audio project;
- a web-search agent;
- a multi-agent system;
- a LangGraph / orchestration showcase;
- an image-generation project;
- a model-training project;
- a leaderboard comparing unrelated VLMs;
- a study claiming universal superiority of early/late neural fusion.

If a proposed feature does not help answer the central research question, it does not belong in the primary project.

---

## 4. Hard Compute Guardrails

The reference experiment must remain practical on a Mac CPU.

### Primary limits

- **No required model training.**
- **No discrete GPU requirement.**
- **Primary frozen benchmark: <= 300 items.**
- **Development benchmark: ~90 items.**
- **Rendered image size: <= 384 x 384.**
- **Objects per scene: usually 4-8.**
- **Closed-form answers whenever possible.**
- **Every expensive inference result must be cached.**
- **Experiments must be resumable.**
- **Routing and corruption analyses should operate on cached predictions whenever possible.**

### Development scale

```text
30 scenes
x 3 question families per scene
= 90 benchmark items
```

### Frozen reference scale

```text
100 scenes
x 3 question families per scene
= 300 benchmark items
```

For three primary evidence conditions:

```text
300 items x 3 conditions = 900 primary predictions
```

Do not expand the benchmark until the smaller development set shows a meaningful experimental signal.

---

## 5. Controlled World Design

Scenes must be **programmatically generated from explicit hidden state**.

Example hidden state:

```json
{
  "key": {
    "color": "blue",
    "location": "under_lamp",
    "visibility": "partial"
  },
  "window": {
    "state": "closed",
    "latch": "engaged"
  },
  "notebook": {
    "location": "desk",
    "state": "open"
  }
}
```

The renderer converts this state into a simple fictional detective scene.

```text
scene specification
       |
       v
 deterministic renderer
       |
       v
      PNG
```

The model sees the rendered evidence. The benchmark retains the hidden state as ground truth.

### Why synthetic scenes

This gives DetectiveLab:

- exact scene ground truth;
- deterministic regeneration;
- controlled visual complexity;
- known provenance;
- exact spatial and state labels;
- controlled contradictions;
- reproducible corruptions;
- no dependence on large external datasets.

---

## 6. Scene Grammar

Keep the visual grammar deliberately small.

### Objects

Examples:

- key
- notebook
- glass
- lamp
- clock
- briefcase
- painting
- chair
- door
- window
- footprint marker
- envelope

### Object attributes

```text
identity
color
state
position
containment
visibility
orientation
```

### Relationships

```text
left_of
right_of
above
below
under
inside
behind
in_front_of
near
far
partially_occluded_by
```

Do not add new visual primitives unless an existing benchmark family requires them.

---

## 7. The Three Primary Case Families

Use the **same scene grammar** across families. Ideally, multiple questions are derived from the same underlying scene so the visual environment is held constant.

### A. Spatial

Tests geometry, viewpoint, relative position, and occlusion.

Example:

> Could the blue key have been visible from the doorway?

Expected hypothesis:

> RAW may outperform STRUCTURED when compression into symbolic state discards useful geometric information.

---

### B. State

Tests discrete facts and rule-based reasoning.

Example:

> Is the study secured if the window is closed, the latch is engaged, and the key is absent?

Expected hypothesis:

> STRUCTURED may outperform RAW when the task mainly requires precise discrete state followed by simple reasoning.

---

### C. Conflict

Tests arbitration between physical evidence and linguistic claims.

Example:

```text
Witness:
"The blue key was on the bookshelf."

Physical scene:
The blue key is under the lamp.

Rule:
Current physical evidence overrides unverified recollection.

Question:
Does the witness statement conflict with the scene?
```

Expected hypothesis:

> HYBRID may outperform either representation alone because the system can use explicit facts while retaining access to the original evidence.

---

## 8. Primary Evidence Conditions

Every primary comparison must preserve the same:

- case;
- question;
- case rules;
- decoding settings;
- answer format;
- scoring logic;
- model family where technically possible.

### QUESTION

No scene evidence.

Purpose: detect dataset shortcuts and prior leakage.

---

### RAW

```text
image + testimony + rules + question -> multimodal model -> answer
```

Purpose: establish the direct multimodal baseline.

---

### ORACLE_STRUCTURED

```text
ground-truth scene state + testimony + rules + question -> reasoner -> answer
```

Purpose: separate perception difficulty from reasoning difficulty.

This condition must use benchmark ground truth, not a learned extractor.

---

### EXTRACTED_STRUCTURED

```text
image -> visual extractor -> structured state -> reasoner -> answer
```

Purpose: measure how much of the oracle advantage survives a practical perception stage.

This is introduced only after RAW and ORACLE_STRUCTURED are understood.

---

### HYBRID

```text
image + extracted structured state + testimony + rules -> reasoner -> answer
```

Purpose: test whether structure is more useful as an additional view than as a replacement for raw perception.

---

## 9. Preferred Answer Space

Prefer deterministic, closed-form outputs:

```text
true / false
supported / contradicted / unknown
A / B / C / D
object_id
location_id
```

Avoid free-form detective narratives in the primary benchmark.

Free-form explanations may be collected as diagnostics but must not be required for primary scoring.

---

## 10. Error Decomposition

DetectiveLab should make errors attributable.

At minimum distinguish:

```text
PERCEPTION ERROR
The scene was interpreted incorrectly.

REASONING ERROR
The correct scene facts were available, but the conclusion was wrong.

ARBITRATION ERROR
The system had competing evidence but trusted the wrong source.

ROUTING ERROR
A later router selected a suboptimal evidence path.
```

The project is successful if it makes these failure modes measurable even if the more elaborate architecture does not improve final accuracy.

---

## 11. Metrics

### Primary

- final decision accuracy;
- paired accuracy delta between conditions.

### Diagnostic

- visual-state accuracy;
- reasoning accuracy given oracle state;
- contradiction-detection accuracy;
- evidence-source attribution accuracy;
- abstention / unknown accuracy where applicable;
- extraction accuracy;
- error type counts.

### Operational

- wall-clock latency;
- model calls;
- text tokens where measurable;
- image-processing calls;
- cache hit rate.

Do not make cost metrics more sophisticated than the system being evaluated requires.

---

## 12. Paired Evaluation

Overall accuracy is not enough.

For every condition comparison, compute per-example paired outcomes:

```text
RAW -> STRUCTURED

wins:   N
losses: N
ties:   N
```

Results should also be broken down by case family.

A routing claim requires family-level differences that are repeatable and large enough to matter operationally.

---

## 13. Routing Guardrail

**Do not implement routing early.**

Routing is allowed only if the completed evidence-path experiments show a pattern resembling:

| Family | RAW | STRUCTURED | HYBRID |
|---|---:|---:|---:|
| Spatial | best | worse | competitive |
| State | worse | best | competitive |
| Conflict | worse | competitive | best |

The exact result need not match this table. What matters is that different paths show systematic conditional advantages.

### Required routing baselines

Before building a learned or heuristic router, calculate:

```text
ALWAYS_RAW
ALWAYS_STRUCTURED
ALWAYS_HYBRID
ORACLE_ROUTER
```

If ORACLE_ROUTER does not meaningfully beat ALWAYS_HYBRID, stop.

**Routing is not justified.**

If the oracle gap is meaningful, then and only then test:

```text
PREDICTED_ROUTER
```

Start with the simplest deterministic router possible.

A learned router must earn its complexity over a deterministic one.

---

## 14. Extraction Reliability Experiment

If structured extraction proves useful, test how its value changes as reliability falls.

Prefer cheap deterministic corruption of cached structured evidence before expensive image perturbation.

Example levels:

```text
0%
5%
10%
20%
30%
```

Example corruption:

```json
// correct
{"window": "closed"}

// corrupted
{"window": "open"}
```

The goal is to identify whether there are regimes where:

```text
high extraction reliability -> STRUCTURED wins
medium reliability          -> HYBRID wins
low reliability             -> RAW wins
```

This is a hypothesis, not an expected result.

---

## 15. Milestones and Promotion Gates

Each milestone must answer **one primary question**.

Every milestone ends with one disposition:

```text
RETAIN
REJECT
INVALIDATE
```

Negative results remain in the repository.

### v0.0-benchmark

**Question:** Can we build a deterministic benchmark without obvious shortcuts?

Deliverables:

- scene schema;
- deterministic renderer;
- development split;
- case generation;
- answer generation;
- benchmark validator;
- provenance manifest;
- hashes.

Promotion gate:

- scenes regenerate deterministically;
- labels match hidden state;
- QUESTION-only performance does not reveal a major shortcut;
- benchmark validator passes.

---

### v0.1-direct

**Question:** How well does direct multimodal reasoning solve the benchmark?

Conditions:

```text
QUESTION
RAW
```

Promotion gate:

- RAW produces a measurable signal above shortcut baseline;
- failures can be inspected by family.

---

### v0.2-oracle-structure

**Question:** How much failure is perceptual versus reasoning-related?

Conditions:

```text
RAW
ORACLE_STRUCTURED
```

Promotion gate:

- quantify oracle gap;
- classify representative RAW failures;
- decide whether explicit extraction is experimentally justified.

If there is no meaningful oracle gap, explicit perception may not be worth adding.

---

### v0.3-explicit-perception

**Question:** Can an explicit extractor close enough of the oracle gap to be useful?

Conditions:

```text
RAW
EXTRACTED_STRUCTURED
ORACLE_STRUCTURED
```

Promotion gate:

- extractor quality is independently measurable;
- end-to-end error can be separated into perception and reasoning components.

Do not train a large detector for this milestone.

---

### v0.4-hybrid

**Question:** Is structure more useful as an additional view than as a replacement for pixels?

Conditions:

```text
RAW
EXTRACTED_STRUCTURED
HYBRID
```

Promotion gate:

- compare paired outcomes by family;
- identify whether hybrid resolves or amplifies extraction errors.

---

### v0.5-corruption

**Question:** How does extraction reliability change the preferred evidence path?

Use cached outputs and deterministic structured-state corruption wherever possible.

Promotion gate:

- identify whether path preference changes systematically with reliability.

---

### v0.6-routing - OPTIONAL

**Question:** Does choosing evidence paths by case type/reliability beat a single strong default?

Prerequisite:

> ORACLE_ROUTER must meaningfully outperform ALWAYS_HYBRID.

If not, this milestone is rejected without implementation.

---

## 16. Stop Conditions

The project should stop rather than expand if any of these occur:

### No multimodal signal

If RAW does not meaningfully beat QUESTION-only after benchmark audit, fix or invalidate the benchmark.

### No oracle gap

If ORACLE_STRUCTURED does not meaningfully outperform RAW, do not force an extraction storyline.

### Extraction cannot approach oracle performance

Record the bottleneck. Do not hide it by adding agents or tools.

### Hybrid dominates everything

If ALWAYS_HYBRID performs essentially as well as oracle routing, do not build routing.

### Routing gain is trivial

If routing saves negligible compute or adds negligible quality, reject it.

### CPU budget breaks

If the full experiment cannot reasonably run within the declared laptop budget, reduce benchmark size, image complexity, model size, or number of conditions before adding hardware requirements.

---

## 17. Drift Prevention Rules

Before adding any capability, answer all five questions:

1. **Which measured failure motivates this capability?**
2. **What single variable does the experiment change?**
3. **What baseline will it be compared against?**
4. **What result would cause us to reject it?**
5. **Can it stay inside the Mac CPU compute budget?**

If any answer is unclear, do not add the feature.

### Explicit drift warnings

The following phrases should trigger a scope review:

```text
"while we're here..."
"it would be cool if..."
"let's also add an agent..."
"we could support video..."
"let's compare a few more models..."
"we should build a UI first..."
"maybe add web search..."
"let's fine-tune it..."
```

These ideas can go into `docs/FUTURE_WORK.md`, not the active milestone.

---

## 18. Reproducibility Contract

Before a frozen evaluation, record and hash where practical:

- benchmark version;
- scene specifications;
- renderer version;
- generation seed;
- split membership;
- rendered image hashes;
- model identifier/revision;
- quantization/settings if local;
- prompt templates;
- answer schema;
- decoding parameters;
- extraction settings;
- evaluation code version;
- scoring rules;
- corruption seed/config;
- candidate-selection policy.

A completed experiment must be reproducible from committed configuration plus documented model dependencies.

---

## 19. Caching Contract

Every model or extraction call should have a deterministic cache key derived from relevant inputs, for example:

```text
benchmark_version
case_id
condition
model_id
prompt_version
image_hash
extractor_version
```

Never rerun expensive inference merely to regenerate plots or routing results.

Analysis should consume saved prediction artifacts.

---

## 20. Suggested Repository Skeleton

```text
detectivelab/
├── README.md
├── PROJECT.md
├── pyproject.toml
├── configs/
│   ├── benchmark/
│   ├── models/
│   └── experiments/
├── data/
│   ├── development/
│   └── frozen/
├── artifacts/
│   ├── predictions/
│   ├── evaluation/
│   ├── provenance/
│   └── audits/
├── docs/
│   ├── EXPERIMENTAL_METHOD.md
│   ├── FUTURE_WORK.md
│   ├── decisions/
│   └── milestones/
├── src/detectivelab/
│   ├── domain/
│   │   ├── schema.py
│   │   ├── rules.py
│   │   └── cases.py
│   ├── rendering/
│   │   └── renderer.py
│   ├── benchmark/
│   │   ├── generate.py
│   │   ├── validate.py
│   │   └── splits.py
│   ├── evidence/
│   │   ├── oracle.py
│   │   ├── extract.py
│   │   └── corrupt.py
│   ├── conditions/
│   │   ├── question.py
│   │   ├── raw.py
│   │   ├── structured.py
│   │   └── hybrid.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── local.py
│   │   └── api.py
│   ├── evaluation/
│   │   ├── score.py
│   │   ├── paired.py
│   │   └── errors.py
│   └── routing/
│       └── oracle.py
├── scripts/
│   ├── generate_benchmark.py
│   ├── run_experiment.py
│   └── evaluate.py
└── tests/
    ├── test_renderer.py
    ├── test_generation.py
    ├── test_scoring.py
    └── test_determinism.py
```

Do not create modules for future milestones until the current milestone needs them.

---

## 21. First Build Target

The first implementation should contain **no multimodal model integration**.

Build only:

```text
scene schema
    ->
scene generator
    ->
renderer
    ->
question generator
    ->
ground-truth answer
    ->
validator
```

Target:

```text
10 scenes
x 3 questions
= 30 completely deterministic examples
```

Success means:

- every scene can be regenerated from seed/spec;
- every question is answerable from declared evidence;
- every answer is generated from hidden state rather than manually typed;
- every image has a stable hash;
- tests verify determinism.

Only after this works should the development benchmark grow to 90 items and model adapters be introduced.

---

## 22. Decision Log Template

For every meaningful architectural addition, create a short decision record.

```markdown
# Decision: <name>

## Observed problem
What measured failure are we addressing?

## Proposed change
What exactly changes?

## Controlled comparison
What stays fixed, and what is the baseline?

## Success criterion
What result justifies retaining the change?

## Rejection criterion
What result means we remove or stop pursuing it?

## Compute impact
Does this remain inside the Mac CPU budget?

## Disposition
RETAIN / REJECT / INVALIDATE
```

---

## 23. Project-Level Success Criteria

DetectiveLab does **not** need to prove that routing, structure, or hybrid reasoning is superior.

The project succeeds if it produces a credible answer to these questions:

1. How much multimodal error comes from perception versus reasoning?
2. When does explicit visual structure help or hurt?
3. Does retaining raw evidence improve robustness to extraction errors?
4. Do different case requirements genuinely favor different evidence paths?
5. If so, is the advantage large enough to justify routing?

A strong negative result is preferable to an unjustified positive claim.

---

## 24. Candidate Headline Conclusions

These are **possible outcomes**, not targets.

### Outcome A - Conditional representations

> Raw perception performs best for geometry-heavy cases, structured evidence for discrete state reasoning, and hybrid evidence for conflict resolution.

### Outcome B - Hybrid default wins

> Keeping both raw and structured evidence is sufficiently robust that routing adds complexity without meaningful benefit.

### Outcome C - Structure is diagnostic, not predictive

> Explicit visual structure does not improve end-to-end accuracy substantially, but it makes perception and reasoning failures attributable and debuggable.

### Outcome D - Direct VLM wins

> On this controlled benchmark, direct multimodal inference is more robust than explicit decomposition, and additional representation machinery is not justified.

All four are valid research outcomes.

---

## 25. One-Sentence Guardrail

> **Add no multimodal complexity until a measured failure in the smallest controlled experiment gives us a reason to add it.**

## v0.0 freeze rule

The 10-scene / 30-item `v0.0` benchmark is frozen after final blind audit. Do not modify its scene semantics, renderer conventions, participant payloads, labels, or scoring in place. Any such change requires a new benchmark version and a new audit trail.

## v0.0.1 benchmark correction rule

The first QUESTION-only model run revealed that `v0.0` conflict labels were predictable from case-rule wording. `v0.0` remains preserved for provenance and must not be rewritten.

`v0.0.1` is the corrected benchmark version. Its conflict family must satisfy all of the following before RAW evaluation:

- every conflict item uses identical case-rule text;
- supported, contradicted, and unknown outcomes all occur in the frozen slice;
- STATE question generation is independent of conflict testimony;
- a conflict verdict must depend on visual evidence availability or visible state, not policy wording;
- the QUESTION-only conflict baseline must no longer reproduce the `v0.0` 100% shortcut result.

If the corrected QUESTION-only baseline remains materially above chance, stop and audit again before adding image inference.
