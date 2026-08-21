## v0.6 Gate Corruption

`v0.6-gate-corruption` tests how representation-grounded control behaves when the extractor-derived presence gate is intentionally corrupted.

The benchmark, extractor, staged prompt, and gold labels remain frozen. Only the gate value used for control flow is perturbed.

Two error directions are tested:

- **false absence:** present target → absent
- **false presence:** absent target → present

Corruption is applied deterministically at 25%, 50%, 75%, and 100% of eligible cases.

### Degradation curves

| Model | Corruption | 25% | 50% | 75% | 100% |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 3 4B | false absence | 80% | 70% | 50% | 40% |
| Gemma 3 4B | false presence | 100% | 90% | 80% | 70% |
| Qwen3 4B | false absence | 80% | 70% | 50% | 40% |
| Qwen3 4B | false presence | 100% | 100% | 100% | 100% |

The clean extractor-gated baseline is 100% for both models.

The central result is:

> **Gate corruption is directionally asymmetric: false absence suppresses evidence before reasoning and causes model-independent accuracy loss, while false presence remains recoverable according to the downstream model's native epistemic capability.**

For this architecture:

> **Target-presence recall matters more than target-presence precision.**

Gemma gradually loses `unknown` accuracy under false presence, while Qwen recovers all false-presence cases even at 100% corruption. Neither model can recover false absence because the reasoner is bypassed.

See [`docs/results/v0_6_gate_corruption.md`](./docs/results/v0_6_gate_corruption.md).

Current test status:

```text
96 passed
```

### Next research question

> Can an abstaining or confidence-aware gate reduce catastrophic false-absence errors without reintroducing unnecessary model reasoning?

No learned router is justified yet.
