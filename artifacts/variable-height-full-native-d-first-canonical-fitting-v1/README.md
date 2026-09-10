# Canonical constraint-aware D-first fitting v1

This is the formal result of the numerical-canonicalization protocol frozen in
commit `99b95f5` and implemented in commit `6ad8f44`. It made the once-
materialized parameter tensors authoritative while retaining the same native
actor, endpoint-D objective, C/P/RPY constraints, optimizer, data and gates.

The preflight passed. A second direct parameter projection changed nothing.
The largest prior displacement discrepancy was identified at edge 145701: it
was `1.1911e-7`, or 0.4996 local float32 ULP. The canonical effective direction
satisfied every linear and box constraint, retained 73.97% of damping descent,
passed the nonlinear 1/16 replay, and had 0.14% finite-difference disagreement.

Eight fitting updates were accepted at scales 1/2, 1/2, 1/2, 1/8, 1/8, 1/16,
1/16 and 1/4. Training endpoint-D NRMSE fell from 1.458400 to 1.376246 (5.63%),
and aligned gain moved from -0.43297 to -0.35893; damping sign remained wrong
in all eight scenes. RPY and all cumulative C/P gates passed through update 8.

The deterministic update-9 proposal had no acceptable registered scale, so the
run stopped immediately. Endpoint common response had reached 1.68378794 versus
its source-relative limit of 1.68379011—only `2.17e-6` NRMSE of slack. Although
the projected tangent satisfied its linear inequality, nonlinear replay crossed
that boundary even at 1/32. The 1/32 trial would otherwise have improved
endpoint-D NRMSE by 0.001118 and passed every other gate.

The run stopped before the first scheduled development evaluation at update 10.
Its update-8 resume state remains an ignored, nonpromotional run artifact; it is
not committed or tracked with Git LFS. No terminal checkpoint, fresh
qualification, closed-loop test or promotion resulted. Compact measurements are
in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-canonical-fitting-001/report.json`
(SHA-256 `bd7d27595d0cec185542e0ede5f873cdc3057252138ae7b700b6b5faa8e22e43`).
