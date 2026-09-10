# Nonlinear D-first feasibility-correction audit v1

This is the formal result of the restored correction protocol frozen in commit
`236671c` and implemented in commit `2b38fd0`. It reconstructed the rejected
update-9 Adam proposal from the immutable update-8 resume state, persisted and
hashed only its scale-1/16 starting parameters, and attempted one minimum-norm
correction without retaining a corrected controller.

All source, update-8, proposal and scale-1/16 replay checks reproduced. The
update-8 controller also transferred cleanly to the already-exposed development
bank: endpoint-D NRMSE improved from 1.398940 to 1.336928, and every original
C/P/RPY, validity and motor-output preservation gate passed.

The audit nevertheless failed two frozen numerical controls. The independently
computed endpoint-D Jacobian rows differed by `2.60e-5` at their worst
coordinate versus the preregistered combined limit of `1.41e-5`, although their
relative L2 difference, `8.37e-6`, passed its `1e-5` limit. The continuous
active-set solve reached a maximum linear residual of `3.04e-11`; converting
that correction to authoritative float32 controller parameters increased the
residual to `1.33e-5`, above the fixed `1e-6` gate.

The scale-1 correction was very close but does not count as a pass. Its actual
endpoint-C NRMSE was 1.68369424, missing the source-plus-0.0199 interior target
by `4.14e-6`, while remaining inside the original source-plus-0.02 outer gate.
It retained 0.002232 endpoint-D improvement from update 8 and passed all other
outer preservation gates. Smaller correction scales did not meet the linear or
interior-C requirements.

Parameters, the complete Adam state and the source resume file were restored
exactly. No corrected candidate, promotion, closed-loop test or fitting
authorization resulted. Compact measurements are in
[`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-nonlinear-correction-audit-001/report.json`
(SHA-256 `fcdd9842ba9e8d0ed652525409af76cb5cd18353ce02d606df5b8bbc6cf42b5f`).
