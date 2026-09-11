# FP64-primary corrected fitting continuation v1

This is the compact record of the formal continuation registered in commit `ba6fcc3` and
implemented in `abf67b1`. It restored total update 20, exactly reconstructed the frozen
update-21 Adam/FP64 primary transaction, and attempted the unchanged ordinary-then-repair
selection before any new-update generation.

The run stopped safely at attempted update 21 with 20 accepted updates. The source
controller, raw displacement, pending Adam state, FP64 displacement and complete projection
objective all reproduced exactly. The ordinary scale-1/16 candidate also had the exact
registered tensor hash and again failed only endpoint-C25's outer gate.

The repair itself passed every safety and nonlinear gate at correction scale 1.0. Its
training endpoint-D NRMSE was `1.35364258`, endpoint-C25 was `1.68359423`, and all 25
registered metric values agreed within the established numerical tolerance. The recomputed
repaired tensor hash was `7d5f2b6...`, however, rather than the preregistered `3d8d96f...`.
That exact-identity gate therefore rejected update 21. This is a cross-process reconstructed-
tensor control failure, not evidence of a damping or safety regression.

The failed transaction restored the controller and Adam state exactly to update 20 and
persisted a stopped resume. No development selection, terminal checkpoint, fresh cohort,
closed-loop hover or promotion occurred. The next registered experiment is an independent-
process qualification of a complete accepted update-21 transaction snapshot; a future
continuation may load it exactly rather than recompute the GPU-sensitive repair.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-corrected-fitting-001/report.json`
(SHA-256 `7790d61241280dc315323563058858e1c16f9e625f459ad2b84f5d8cbc74ee40`).
