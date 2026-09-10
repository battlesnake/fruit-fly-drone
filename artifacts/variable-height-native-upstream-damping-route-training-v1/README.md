# Upstream native damping-route training v1

This is the terminal result of the expanded-route protocol frozen in commit `aa65d79`.
It restarted from the preserved source and jointly trained the frozen 80,454-edge,
883-bias upstream/downstream mask under the original shallow metric denominators and all
source-preservation limits.

The run was rejected before its attempt-25 milestone. Ten updates were accepted, but
attempts 17–21 produced five consecutive inadmissible directions and triggered the
preregistered stop. On the previously unconsumed 64-pair terminal cohort, motion NRMSE
fell from 1.380882 to 1.366758, a 1.023% improvement. Damping remained wrong-signed on
all 64 pairs and teacher-aligned gain moved only from -0.5671 to -0.5516. No checkpoint
was promoted and no closed-loop hover test was run.

All terminal preservation measurements were finite and passed. Small and medium visual
height contrasts retained 0.97304 and 0.97389 of source, source-global common-throttle
RMS was 0.002432 against the 0.0025 limit, and maximum common drift was 0.003906 against
0.005. The fixed-denominator parameter metric reached only 0.000134 against its 0.0005
radius, so parameter distance was not the bottleneck.

The last five candidate screens make the failure specific. Equality-heavy directions
kept predicted common RMS inside its limit but had positive damping-loss derivatives.
Directions with negative damping derivatives predicted source-common RMS of roughly
0.00253–0.00311 and were inadmissible. The mean accepted-step diagnostic improvement per
unit common RMS was 4.12. Thus adding descending-neuron afferents did not produce a
selective fixed-sign damping route before exhausting the permitted collective-output
change.

This closes further anatomical expansion under the current source-common preservation
strategy. It does not show that the connectome cannot learn visual altitude control; the
constraint preserves a source policy whose tonic/height and motion responses may need to
be relearned jointly. The next experiment should change that control decomposition rather
than automatically open more anatomy or relax thresholds post hoc.

Compact measurements are in [`report.json`](report.json). The complete ignored report,
including per-bank data and every candidate screen, is
`runs/variable-height-hover/upstream-damping-route-train-001/report.json` (SHA-256
`137da6fee2f0fac564931f5db8cdaa7cb3c340747af9286dfa9ef0d3f96fbcec`). The ignored
endpoint is retained only for audit and remains explicitly nonpromotional.
