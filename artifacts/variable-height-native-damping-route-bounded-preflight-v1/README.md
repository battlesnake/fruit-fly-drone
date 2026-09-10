# Bounded native damping-route preflight v1

This separately preregistered retry kept the exact shallow route mask, loss, step cap
and acceptance thresholds from the rejected equality-projected preflight. It changed
only the constraint formulation: pair-common throttle became a bounded inequality, and
candidate directions were normalized to the final selected-family step cap before full
replay.

The preflight passed. All five equality-to-raw blends were linearly admissible, and the
raw selected-metric descent was the strongest. At scale 1 it reduced fixed-bank motion
NRMSE from 1.449651 to 1.447269, an improvement of 0.002382 against the unchanged
`1e-4` gate. The predicted loss change was -0.006913 and the measured change was
-0.006901. Forty-one source-zero edge magnitudes were clipped at their fixed nonnegative
bound before screening.

Full zero-state replay—not the linear model—made the acceptance decision. Source-global
common-throttle RMS was 0.000426 against 0.0025, maximum drift was 0.000546 against
0.005, and per-update common RMS was 0.000426 against 0.001. Small and medium visual
height contrasts retained 0.99574 and 0.99572 of source. All legacy and dynamic R/P/Y
checks passed. The rescaled equality direction also passed independently and improved
motion NRMSE by 0.000328, confirming that the v1 failure came from leaving its projected
step far below the declared cap rather than from a missing gradient.

This is only local credit-assignment evidence. Complete-replay damping remained
wrong-signed with gain -0.547 after the accepted trial, so no controller was retained or
promoted. Parameters were restored with zero maximum error. The result authorizes the
preregistered maximum-50-attempt training run on this exact mask, with its meaningful-
progress gate at attempt 25.

Compact measurements are in [`report.json`](report.json). The complete ignored report,
including mask indices, candidate screens and all functional replay fields, is
`runs/variable-height-hover/damping-route-bounded-preflight-001/report.json` (SHA-256
`f4c79d1d561e2d63d1d02c40c4afc77300818fa3ada2451309308cf74379fc01`).
