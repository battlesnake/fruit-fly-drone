# FP64-primary corrected update-21 audit v1

This is the formal restored one-step result registered in commit `d6fa381` and
implemented in `191e8ba`. It consumed the already-qualified frozen update-21 proposal,
reconstructed exactly one pending Adam transaction, ran the original ordinary scales and
unchanged guard-band repair, evaluated the one selected candidate on the fixed development
bank, and restored update 20 without retaining a candidate.

The audit passed. The reconstructed raw displacement and pending Adam state matched their
qualified hashes exactly. The FP64 projection reproduced the qualified displacement and
complete primal objective exactly, converged in three rounds, and passed all primary,
independent, canonical, bound, linear, descent, and finite-difference controls.

No ordinary scale passed. Scale 1/16 was repair-eligible: it improved training endpoint-D
NRMSE by 0.002262, and endpoint-C25 was its only failed outer gate. The unchanged
guard-band correction passed at its first scale, 1.0. The selected training candidate had
endpoint-D NRMSE `1.35364211`, a 0.002261 improvement from update 20, and endpoint-C25
NRMSE `1.68359399`, leaving `9.61e-5` headroom to the source-plus-0.0199 acceptance limit.
Its repair finite-difference disagreement was 0.0580%, and the one pending Adam state was
unchanged throughout correction.

On the fixed development bank, endpoint-D NRMSE improved from update 20's `1.32103145` to
`1.31926715`, a 0.001764 improvement, while every original-source preservation gate
passed. Sign fraction nevertheless remained zero and teacher-aligned gain remained
negative on both training and development, so this is a validated incremental update—not
useful damping or a flight result.

Parameters and Adam state were restored exactly, all locked inputs remained byte-for-byte
unchanged, and no candidate, checkpoint, fresh cohort, closed-loop run, or promotion
resulted. Compact measurements are in [`report.json`](report.json). The complete ignored
report is
`runs/variable-height-hover/full-native-d-first-fp64-corrected-step-audit-001/report.json`
(SHA-256 `8f76e9b5bd9cb1cecc782b207c3694b332926fb9f34b5be1fea8fd31b2deb6dd`).
