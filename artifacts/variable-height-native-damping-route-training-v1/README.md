# Native damping-route training v1

This is the terminal result of the bounded training protocol frozen in commit
`6673469`. It trained only the 12,314 edge magnitudes and 303 intermediate biases on
existing direct or two-edge descending/VNC-intrinsic routes to the seven throttle motor
neurons. Inputs, recurrent state, neuron time constants, motor biases and the other
connectome parameters were unchanged.

The route was rejected at its preregistered attempt-25 gate. Sixteen of 25 proposed
updates passed both independent guard banks and all functional constraints. On the
64-pair milestone cohort, fixed-scale motion NRMSE fell from 1.365113 to 1.349864: a
1.117% improvement against the required 25%. The disjoint 64-pair terminal cohort
improved from 1.395011 to 1.378718 (1.168%), but damping remained wrong-signed on every
pair and teacher-aligned gain moved only from -0.5691 to -0.5519. No closed-loop hover
test was run and no checkpoint was promoted.

The preservation result is informative. Small and medium visual height responses kept
0.97023 and 0.96999 of their source contrast, all legacy and dynamic R/P/Y replay checks
passed, and every measured value was finite. Meanwhile source-global common-throttle RMS
reached 0.002491 against its 0.0025 limit, although the selected-route source metric was
only 0.000175 against 0.0005. Later directions were frequently inadmissible or rejected
on common-output drift. Thus this shallow route can weaken the erroneous response only
slightly before colliding with the requirement to preserve tonic/height control; it does
not provide a selectable correctly signed visual-motion channel.

The negative result closes this exact mask and constraint formulation. The next test
should change the anatomical/credit-assignment hypothesis, not relax these thresholds
after observing the result.

Compact measurements are in [`report.json`](report.json). The complete ignored report,
including all per-bank samples, candidate screens and replay fields, is
`runs/variable-height-hover/damping-route-train-001/report.json` (SHA-256
`69977eafacd02352dfedb5de337aabec3874a3c68961059f96a40d25e7d1f4c6`). The ignored
nonpromotional endpoint is retained only for audit, not as a controller release.
