# Bound-aware D-first fitting preflight v1

This is the formal zero-update result of the bound-aware projection protocol
frozen in commit `fe4401f` and implemented in commit `97aac02`. It preserved
the earlier failed preflight and changed only the handling of native edge
magnitudes that crossed their fixed [0, 8] bounds.

The active-set correction worked substantively. It converged in three rounds,
fixed 4,699 edges at their actual boundary displacements, left no box
violation, and kept maximum post-bound linearized constraint violation at
`7.18e-7`, below the `1e-6` limit. It retained 73.97% of the raw damping
descent. The scale-1/16 complete replay passed every C/P/RPY/validity/output
gate, improved endpoint-D NRMSE by 0.002525, and agreed with autograd within
0.18%.

The preregistered preflight still failed. Re-materializing the projected
parameters through the controller changed the float32 displacement by at most
`1.1902e-7`, just over the frozen `1e-7` idempotence limit. The threshold is not
relaxed after observing the result. This is a numerical representation failure,
not evidence against the projected damping direction.

No update was accepted, no optimizer or resume state was retained, and no
development candidate, fresh qualification, closed-loop test or promotion
resulted. Compact measurements are in [`report.json`](report.json). The
complete ignored report is
`runs/variable-height-hover/full-native-d-first-bound-aware-fitting-001/report.json`
(SHA-256 `1c8964221f646f5ad9f61003229da6edbf7fe95cb3019a197268ae8306f81171`).
