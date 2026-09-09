# Recurrent PPO diagnostic v1

This directory preserves the rejected recurrent-PPO experiment run from the promoted
`gate-motor-interface-es-v1` controller. It is evidence, not a deployable checkpoint:
no `controller.pt` is present and the promoted controller remains unchanged.

The actor used only current FPV, roll/pitch, body-Z specific force, and its persistent
native connectome state. The common API supplied stick position, but this graph had no
proprioception nodes and ignored it. Mass, gate pose, physical state, reward bookkeeping,
and the critic were available during training only.

- `report.json` records the protocol, numerical audits, per-iteration training summaries,
  validation, fresh 1,024-flight evaluation, causal controls, and failed promotion checks.
- `candidate-vector.json` records the exact selected 198 edge magnitudes and 26 motor
  biases. Its float32 vector hash is
  `22cf35378b211c14bd51d2b7584d1ecd120f55ad7d5168b694c7308836d90829`.
- `archive.pt` retains the five validation snapshots. SHA-256:
  `63c2da669991186e83bcbd36c36d296b40d64dfefb471469383ce54111fbd517`.

On fresh matched flights, the selected candidate changed success from 47.66% to 47.56%:
light mass improved from 3.71% to 5.27%, while heavy mass declined from 91.60% to
89.84%. Constant-1g acceleration slightly outperformed live acceleration, and swapping
acceleration traces within matched mass pairs changed no successes. The experiment did
not demonstrate acceleration-specific adaptation and failed promotion.
