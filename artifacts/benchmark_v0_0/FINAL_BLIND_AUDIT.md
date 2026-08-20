# DetectiveLab v0.0 — Final Blind Audit

Status: **PASS — freeze approved**

## Protocol

The 10 rendered scenes and their 30 participant-facing questions were reviewed without using hidden-state rationales to infer the answers. The review focused on whether the evidence needed by each family is available and naturally interpretable from the participant-facing materials.

## Changes since the HOLD audit

- Removed briefcase `latched/unlatched` from the v0.0 state/question vocabulary.
- Removed lamp `on/off` from the v0.0 state/question vocabulary.
- Redesigned doors so `open` visibly leaves an empty doorway and `closed` fills the frame.
- Redesigned windows so `open` visibly exposes a central opening and `closed` fills the frame with panes.
- Redesigned notebooks so `open` has a two-page silhouette and `closed` is a single bound cover.
- Retained glass `intact/broken`, where breakage is represented by a physical crack.
- Kept family-specific payload isolation: spatial/state receive no testimony; conflict receives testimony plus the exact evidence-priority rule.

## Blind-review result

All 10 state items are answerable from ordinary visible geometry rather than a benchmark-specific legend:

- broken/intact glass: physical crack vs intact silhouette,
- open/closed door: displaced door panel and visible gap vs filled doorway,
- open/closed window: displaced panes and visible opening vs filled window frame,
- open/closed notebook: two-page spread vs closed bound cover.

Spatial items remain directly answerable from left/right geometry. Conflict items contain the testimony and rule required to distinguish `contradicted` from `unknown`, while the visual state remains the physical evidence source.

## Automated freeze gates

- [x] deterministic export and SHA-256 provenance
- [x] 10 scenes / 30 items
- [x] exactly one spatial, state, and conflict item per scene
- [x] spatial labels balanced 5 yes / 5 no
- [x] state labels balanced 5 yes / 5 no
- [x] conflict labels balanced 5 contradicted / 5 unknown
- [x] no overlap or partial occlusion
- [x] unique color+kind visual labels per scene
- [x] geometry-consistent spatial relations
- [x] participant payloads hashed and validated
- [x] state/spatial payloads contain no witness leakage
- [x] conflict payloads contain testimony + case rule
- [x] all tests pass
- [x] final blind visual review passes all 30 items

## Disposition

**FREEZE v0.0.**

The benchmark contract may now be treated as fixed for subsequent model experiments. Changes to scene semantics, renderer state conventions, question labeling, payload composition, or scoring should require a new benchmark version rather than silently modifying v0.0.
