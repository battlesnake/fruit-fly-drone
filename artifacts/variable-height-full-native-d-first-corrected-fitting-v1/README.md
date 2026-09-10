# Corrected D-first fitting v1

This is the formal result of the corrected fitting protocol frozen in commit
`4e9f646` and implemented in commit `a352dc3`. The run forked—not reopened—the
hash-locked canonical update-8 resume, retained the native recurrent actor and
sensor contract, and added only the previously audited nonlinear endpoint-C
guard-band repair.

The run accepted 12 new updates, reaching total update 20. Eight were ordinary
backtracked proposals. Updates 9, 12, 14 and 18 exhausted ordinary backtracking
and passed the fixed repair at proposal scale 1/16 and correction scale 1. Every
one-step Adam counter check and every repaired-candidate pending-optimizer check
passed. The update-9 proposal and starting candidate also reproduced the
authorizing guard-band audit within its registered numerical tolerances.

Training endpoint-D NRMSE fell from 1.45840085 at the original source and
1.37624633 at update 8 to 1.35590315 at update 20, a 7.03% improvement from the
source. Development endpoint-D NRMSE was 1.33429182 at update 10 and 1.32103133
at update 20, a 5.57% improvement from its 1.39894068 source. Both scheduled
development preservation checks passed. This was still far from useful damping:
training and development sign fractions remained zero and their gains remained
negative.

Update 21 stopped before nonlinear trial replay because the bound-aware
active-set projection failed its solver control. Round one fixed 5,508 crossing
edges. On round two, L-BFGS-B returned `ABNORMAL` with `solver_success=false`.
Its maximum continuous linearized violation was nevertheless only `2.16e-7`,
inside the `1e-6` residual limit, but authoritative materialization left a
`3.87e-6` violation, outside that same limit. The frozen protocol requires both
solver success and post-materialization feasibility, and explicitly forbids
repairing a projection/numerical failure, so no retry or nonlinear repair ran.

The rejected Adam transaction advanced every counter exactly once while forming
the proposal, then the stopped resume restored the optimizer to its recorded
pre-proposal semantic hash and retained the update-20 controller. The run state
is terminal, the original update-8 resume is unchanged, and no terminal
checkpoint, fresh qualification, closed-loop hover comparison or promotion was
authorized. Compact measurements are in [`report.json`](report.json). The full
ignored report is
`runs/variable-height-hover/full-native-d-first-corrected-fitting-001/report.json`
(SHA-256 `9b236fc6a6744b1b06984958ebc2c3ff3681cc45fc1fa6da73d1f96ae56b458b`).
