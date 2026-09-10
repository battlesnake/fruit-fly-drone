# Native damping-route preflight v1

This preflight tested whether the preregistered downstream native mask contains a
measurable safe direction toward correctly signed vertical damping. The mask contains
12,314 existing edge magnitudes and 303 nonmotor intermediate biases on direct or
two-synapse paths from descending/VNC-intrinsic neurons to the seven throttle-pool motor
neurons. It excludes all motor neurons as intermediates and freezes topology,
transmitter signs, motor biases, every time constant and all other parameters.

The result is a clean rejection, not an optimizer failure. The source's fixed motion
bank had 1.270499 fixed-scale NRMSE and teacher-aligned gain -0.44565. The desired branch
outputs had 0.567 motor-unit margin to their mathematical bounds, so output saturation
was not the obstacle. A raw equal-family descent step used the full `2e-5` selected-edge
RMS cap. Projecting out 16 pair-common throttle rows reduced it to `2.657e-6`; its
first-order derivative remained negative.

At scale 1, the predicted loss change was `-7.378e-5` and the measured finite-difference
change was `-7.653e-5`, confirming the gradient and implementation. Every complete
zero-state replay guard passed: common-throttle RMS was `4.57e-6`, maximum common drift
was `8.73e-6`, and small/medium height-response ratios were 0.99995/0.99996. However,
motion NRMSE improved only `3.016e-5`, below the preregistered `1e-4` measurement floor.
All smaller backtracking scales were weaker, and correct damping sign remained 0%.

No candidate was accepted and parameters were restored with zero maximum error. The
planned 50-update run is therefore not authorized from this preflight. The result says
this exact shallow downstream mask plus exact linearized common-output equality is too
weak at the declared scale; it does not say visual motion is absent, since the preceding
responsibility audit decoded it strongly under identical endpoint pixels.

Compact measurements are in [`report.json`](report.json). The complete ignored report,
including all mask indices, constraint spectra and trial replay metrics, is
`runs/variable-height-hover/damping-route-preflight-001/report.json` (SHA-256
`0c723a12cab461c8abc4e7ba704cc538e4e6b5cd6133d836f615da52d4f0084c`).
