# DetectiveLab benchmark correction: v0.0 -> v0.0.1

## Why v0.0 was not used for model conclusions

The first real QUESTION-only run on the frozen `v0.0` benchmark exposed a text-side shortcut in the conflict family:

- spatial: 50%
- state: 50%
- conflict: 100%

The conflict label was correlated with the wording of the case rule. Cases using a physical-evidence-priority rule mapped to `contradicted`, while cases using a no-priority rule mapped to `unknown`. A text-only model could therefore infer the conflict verdict without seeing the scene.

`v0.0` remains preserved for provenance. It is not rewritten or deleted.

## v0.0.1 correction

`v0.0.1` changes only the benchmark-generation logic required to remove that shortcut:

1. Every conflict item uses the same rule text:
   `Current physical evidence overrides unverified witness testimony.`
2. Conflict outcomes now depend on the relationship between testimony and the rendered scene:
   - `supported`: the claimed object is present and its visible state matches the testimony;
   - `contradicted`: the claimed object is present and its visible state conflicts with the testimony;
   - `unknown`: the testimony concerns a plausible object-state pair that is not present in the current scene.
3. STATE questions are generated independently from conflict testimony.
4. Validation now rejects a benchmark if conflict rule text varies across cases or if any of the three conflict verdicts is absent.

## Frozen v0.0.1 slice

- scenes: 10
- items: 30
- spatial: 5 yes / 5 no
- state: 5 yes / 5 no
- conflict: 3 supported / 3 contradicted / 4 unknown
- validator status: PASS

The next gate is to rerun the QUESTION-only model baseline on `v0.0.1`. RAW evaluation should not proceed until the conflict result falls away from the previous 100% shortcut behavior.
