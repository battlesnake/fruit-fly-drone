# Update-25 exhaustive-primary solver audit v1

This is the compact record of the frozen-input numerical audit registered in commit
`18a237b` and implemented in commits `c1395d2` and `1ba167a`. The producer restored exact
accepted update 24, reproduced both training metric trees, generated the update-25 rows,
gradient and one-step Adam displacement once, archived them before any projection, and
restored the controller and optimizer exactly.

The producer passed. Its ignored 200 MiB archive has SHA-256 `5fc3b908...`; all seven
tensor-tree semantic hashes and the source files remained exact. The separate verifier
loaded that archive three times.

The exhaustive-support primary solution was identical on every reload and numerically
excellent: it found the sole full-KKT-feasible support (mask 840), with normalized KKT
residual `1.16e-17` and original-unit primal violation `5.03e-16`. The required SLSQP
reference nevertheless failed the registered accuracy gates. Although SLSQP reported
success and its objective differed by only `8.72e-11`, its KKT residual was `7.00e-7` and
its Gram-induced primal distance from primary was `9.34e-6`, above the `1e-8` and `1e-6`
limits. The audit therefore failed in round 1 before adding the 5,506 identified bound
crossings. The complete bound-aware projection and all post-projection controls were not
tested.

This is an independent-reference accuracy failure, not evidence that the primary QP is
infeasible. No controller candidate was installed, no Adam step retained, no development
or fresh data evaluated, no hover run made, and nothing promoted.

The one final registered numerical follow-up reuses this exact archive without regenerating
anything. It keeps SLSQP unchanged, derives support only from SLSQP's own coefficients,
then performs one fixed FP64 SVD support-polishing solve before applying the existing
reference gates. Failure pauses this solver-integration route.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-update25-exhaustive-audit-001/report.json`
(SHA-256 `7b9532798c7bb55ea8d27bd06dd026d6de4c54847a10c108120a22bfc0bfc47a`).
