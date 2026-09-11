# Bounded RK4 readout-capacity fit

This is the compact terminal record of the capacity test registered in commit `3cda916` and
implemented in `4362cb6`. It restarted the original controller, trained only the existing
493 incoming throttle-motor edges plus seven motor biases/time constants, and enforced the
unchanged source-relative preservation limits separately on four 24-pair visual blocks.

Two updates were accepted: scale 8 reduced full-prefix NRMSE to `1.246492`, and scale 2 reduced
it to `1.245232`, from source `1.251558`. On attempt 3, the gradient remained a valid descent
direction and the derivative ladder passed. Scale 2 still supplied `0.001257` real-replay
improvement, but three blocks narrowly exceeded the `0.005` common-throttle RMS ceiling.
Scale 1 was safe on all blocks but supplied only `0.000629` improvement, below the immutable
`0.001` floor. Smaller scales were likewise safe and sub-threshold.

The result isolates a responsibility conflict: useful last-hop damping motion consumes the
same interface budget needed to preserve collective throttle. It is not a numerical failure
or a lack of local descent. The response remained wrong-signed in all 96 cases; aligned gain
moved only from `-0.238209` to `-0.232473`.

The rejected transaction restored the accepted update-2 state exactly. Development and
qualification stayed unopened. This closes the unchanged last-hop fitting family and sends
work upstream into native recurrent routing. It authorizes no hover, gate flight or promotion.

Compact measurements are in [`report.json`](report.json). The complete ignored run occupies
about 3.0 GiB at `runs/variable-height-hover/native-rk4-readout-capacity-001/`; its report
SHA-256 is `7f0dbd25aff387a67ec992e2ea7468c7c9c7b74e7d569914565f99665501b09c`.
