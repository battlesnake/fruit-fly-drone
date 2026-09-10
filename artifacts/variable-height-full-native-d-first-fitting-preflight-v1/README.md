# Constraint-aware D-first fitting preflight v1

This is the formal zero-update result of the fitting protocol frozen in commit
`420df40` and implemented in commit `e129961`. It attempted to project the
full-native endpoint-damping Adam proposal onto the eight registered aggregate
and per-horizon C/P error half-spaces before any bounded fitting.

The unconstrained half-space problem was well behaved. Its dual solver
converged, reduced maximum linearized violation to `1.50e-7`, and retained
74.1% of the original damping descent. Applying the fixed edge-magnitude bounds
then altered that solution: the actual post-bound displacement had linearized
MSE violations of 0.006976 for endpoint common response and 0.000179 for
step-20 height response. The preregistered post-bound projection gate therefore
failed and fitting did not start.

Other controls were positive. A 1/16 replay of the post-bound direction passed
all actual C/P/RPY/validity/output gates, improved endpoint-D NRMSE by 0.002529,
and had 0.15% finite-difference disagreement. This supports repairing the
projection's bound handling, but does not retroactively pass the required
full-direction linearized gate.

No update was accepted, no optimizer or resume state was retained, no
development candidate or fresh qualification was run, and parameters remained
bit-identical to the source. Compact measurements are in
[`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fitting-001/report.json`
(SHA-256 `9703ac092d1b4b61e9ef7c3618b7982c325750a1e63da6eb6338084c98f1842b`).
