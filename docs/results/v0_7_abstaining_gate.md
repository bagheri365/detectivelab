# v0.7 — Abstaining Gate

Branch: `v0.7-abstaining-gate`

## Research question

Can an abstaining gate reduce catastrophic false-absence errors without unnecessarily sending every case back to the reasoner?

`v0.6` established a directional asymmetry in representation-grounded control:

- false absence is catastrophic because it suppresses evidence and bypasses reasoning;
- false presence can be recoverable downstream.

`v0.7` tests a simple consequence of that result: if a risky hard-absence decision is converted to `uncertain`, can accuracy be recovered by reopening the reasoning path?

## Gate states

The control layer now permits three states:

```text
present
absent
uncertain
```

Routing policy:

```text
present   -> staged reasoning
absent    -> deterministic unknown policy
uncertain -> staged reasoning
```

The experiment deliberately starts from the `v0.6` 100% false-absence stress condition on the six genuinely present conflict targets.

A deterministic protection subset converts selected would-be false absences to `uncertain`.

The four genuinely absent cases retain the normal zero-call deterministic `unknown` branch.

## Controlled protection rates

Protection rates:

- 0%
- 25%
- 50%
- 75%
- 100%

With six eligible present-target cases, deterministic protected counts are:

| Protection rate | Protected | Residual false absences |
| --- | ---: | ---: |
| 0% | 0/6 | 6/6 |
| 25% | 2/6 | 4/6 |
| 50% | 3/6 | 3/6 |
| 75% | 5/6 | 1/6 |
| 100% | 6/6 | 0/6 |

## Models

- `gemma3:4b`
- `qwen3:4b-instruct-2507-q4_K_M`

## Results

Gemma and Qwen produce the same full protection curve.

| Protection rate | Protected false absences | Residual false absences | Abstention rate | Model-call rate | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0/6 | 6/6 | 0% | 0% | 40% |
| 25% | 2/6 | 4/6 | 20% | 20% | 60% |
| 50% | 3/6 | 3/6 | 30% | 30% | 70% |
| 75% | 5/6 | 1/6 | 50% | 50% | 90% |
| 100% | 6/6 | 0/6 | 60% | 60% | 100% |

### Label-level behavior

At 0% protection:

- supported: 0%
- contradicted: 0%
- unknown: 100%

At 100% protection:

- supported: 100%
- contradicted: 100%
- unknown: 100%

Intermediate rates recover exactly those present-target cases whose false-absence decisions are converted to `uncertain`.

## Main finding

> **Abstention converts catastrophic false-negative gate errors into recoverable reasoning calls, producing a direct accuracy-compute tradeoff.**

Every protected false absence reopens the staged reasoning path instead of forcing `unknown`.

In the current slice, each protected case recovers exactly one error.

The progression is therefore transparent:

```text
0 protected -> 40%
2 protected -> 60%
3 protected -> 70%
5 protected -> 90%
6 protected -> 100%
```

## Efficiency result

Full protection restores 100% accuracy while invoking the model on only 6/10 cases.

The four genuinely absent cases still terminate through the deterministic zero-call path.

This distinguishes abstaining control from simply sending every item to the reasoner:

```text
hard corrupted gate:
  0% model-call rate
  40% accuracy

full abstention protection:
  60% model-call rate
  100% accuracy

always reason:
  100% model-call rate
```

The controlled result therefore supports:

> **When false negatives are more costly than false positives, selective abstention can preserve a cheap deterministic path while escalating only uncertain high-risk decisions.**

## Cross-model interpretation

The identical Gemma and Qwen curves are important.

In `v0.6`, false-presence recovery was model-dependent because the downstream reasoner had to recover missing-evidence semantics.

In `v0.7`, the rescued cases are genuinely present targets. Both staged reasoners solve those cases correctly once they are allowed to reason.

The remaining error source is therefore the control layer, not a model-specific downstream capability.

## What v0.7 does not establish

The experiment uses controlled protection.

It does **not** show that the system already knows which hard absence decisions should become `uncertain`.

The protection signal is intentionally supplied by the experiment so that the value of abstention can be measured independently of confidence estimation quality.

Therefore:

> **v0.7 measures the value of an abstention mechanism, not the quality of a learned or calibrated uncertainty estimator.**

This distinction is essential.

The experiment establishes that abstention has architectural value if risky false-absence decisions can be identified.

It does not yet solve how to identify them from evidence.

## Relationship to earlier milestones

### v0.5

Representation-grounded gating succeeded when control reused a reliable extractor-derived fact rather than asking the reasoner to re-infer existence.

### v0.6

Controlled corruption showed that gate errors are asymmetric. False absence is especially dangerous because it suppresses evidence before reasoning.

### v0.7

Controlled abstention shows that this catastrophic failure mode can be converted into a recoverable reasoning path, with an explicit accuracy-versus-compute tradeoff.

The resulting progression is:

> reliable structured control -> asymmetric corruption -> selective abstention

## Architectural principle

The combined `v0.6` and `v0.7` result suggests:

> **Hard control decisions should reflect asymmetric downstream risk. When one error direction suppresses recoverable evidence, an abstention path can be preferable to a forced binary decision.**

For the current presence gate:

- hard false absence is high-risk;
- uncertain is recoverable;
- genuine absence remains cheap.

This gives a concrete systems reason for three-way gating rather than introducing a learned router prematurely.

## Recommended next step

The missing piece is now uncertainty estimation.

A natural next milestone is:

> **Can uncertainty be derived from the extractor itself rather than supplied by controlled oracle protection?**

Potential first signals should remain simple and interpretable:

- template-match margin
- connected-component ambiguity
- distance from expected object-size range
- competing candidate scores
- threshold proximity

The goal should be to construct an evidence-derived abstention signal before considering a learned confidence model or router.

No learned routing is justified yet.
