# Upstream native damping-route preflight v1

This diagnostic restarted from the preserved source and retained the shallow downstream
route while adding every native afferent and bias of the 612 descending neurons already
feeding its intermediates or throttle motors. The frozen mask contains 80,454 edge
magnitudes and 883 biases. To make the intervention comparable with the shallow test,
its parameter metric still divides by the original 12,314 edges and 303 biases.

The one-step preflight passed and restored every parameter exactly. The raw direction
was selected at full scale. Its autograd derivative was -0.0077051 and the forward
finite-difference estimate was -0.0076752, a 0.389% relative error against the
preregistered 20% limit. Fixed-source-prefix motion NRMSE improved by 0.002654, and the
separately seeded complete zero-state native replay improved by 0.002219; both exceeded
the unchanged `1e-4` floor. Damping remained wrong-signed, as expected from one step.

All preservation checks passed. Source-global common-throttle RMS was 0.000577 against
0.0025, per-update common RMS was also 0.000577 against 0.001, maximum common drift was
0.000749 against 0.005, and both height-response ratios were about 0.9953.

The comparison is promising but qualified. On the identical fixed-prefix bank the
expanded route improved NRMSE 11.4% more than the shallow preflight, but used 35.5% more
common-output drift; improvement per unit common RMS was 4.60 versus 5.59. The local
step therefore does not yet demonstrate better damping/collective separation. It does
establish valid local credit assignment and permits a separately frozen bounded training
test to determine whether joint upstream/downstream adaptation becomes more selective.
It does not identify a biological damping module or promote a controller.

Compact measurements are in [`report.json`](report.json). The complete ignored report,
including mask indices, superclass counts, candidate screens and all replay fields, is
`runs/variable-height-hover/upstream-damping-route-preflight-001/report.json` (SHA-256
`4a93a09acee7efc96e55043cddfd089f198ed253c44cab1e419972e8c5c8f6ea`).
