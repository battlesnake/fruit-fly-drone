# Frozen-input FP64 projection qualification v1

This is the formal result of the numerical qualification initially registered in
commit `df0910d`, mathematically corrected before the run in `cb7c7aa`, and implemented
in `8fd4782`. It reconstructed update 21 once from the hash-locked update-20 controller,
persisted every actual numerical input to one ignored CPU archive before any solve, and
reloaded that exact archive for three complete FP64 projections.

The qualification passed. All three projections were byte-identical, including the final
displacement hash and complete bound-aware primal objective. Each run fixed 5,508 then 16
edges and converged in round three with 5,524 fixed edges. The maximum original-unit
primal violation was `1.02e-10`; the maximum normalized primary KKT residual was
`5.99e-10`. All were comfortably inside their respective `1e-6` and `1e-8` limits.

Every round also passed the independent exhaustive active-support NNLS check against the
original rank-deficient dual QP. The 10-row Gram matrices had rank 8 and nullity 2, and
their violation vectors had substantial components outside the retained Gram eigenspace,
confirming why the pre-run protocol correction was necessary. The maximum Gram-induced
primal disagreement from the primary solver was `8.14e-9`, below `1e-6`; the maximum
relative original-objective disagreement was `6.14e-16`, below `1e-8`.

Canonicalization then passed idempotence and native bounds. Its post-materialization
linear violation was `1.18e-7`, the endpoint-D directional derivative was negative, and
the scale-1/16 finite difference disagreed by only 0.113%, below the 20% limit. The
production projector also completed three rounds, but remained diagnostic-only.

Parameters and Adam state were restored exactly, every source and prior failed-audit file
remained byte-for-byte unchanged, and no candidate, development/fresh result, closed-loop
run, or promotion resulted. This pass authorizes only the separately registered one-step
FP64-projected plus guard-band-repaired audit; it does not accept update 21 by itself.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-frozen-input-audit-001/report.json`
(SHA-256 `e1ead1779c568cacfcebda5df435175f505c9e7218592c37b1187faeaf929f25`).
The ignored 153 MiB tensor archive has SHA-256
`e228a3920c47f56a7eac6d1452f996d9709721f82536240d96dbe6e0c6e1830f`.
