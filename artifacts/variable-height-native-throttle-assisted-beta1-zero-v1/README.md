# Teacher-attitude-assisted native throttle, beta1=0

This is the compact terminal record of the source-restarted curriculum registered in commit
`ddd3c5c` and implemented in `3a7348b`. It changed only fresh Adam's betas from
`(0.9, 0.999)` to `(0, 0.999)` relative to the assisted-throttle protocol. The actor still
received 320x200 RGB plus roll and pitch, retained only native MaleCNS recurrence, and drove
the two abstract forelegs. The analytical teacher owned physical roll, pitch and yaw; the fly
alone owned throttle. No accelerometer, privileged input, external state machine or engineered
history was added.

Before the formal run, the registered nonformal CUDA integration accepted one β1=0 update at
scale 1/2. Its fixed-burn finite-difference relative error was `0.061712`, and its controller,
Adam counters, RNG and frozen-bank identities survived a CPU-canonical resume round trip.

The formal teacher positive control again passed with 100% hover success. Its foreleg motion
control had the correct sign for all 24 pairs and maximum steady RC error `0.001070`; every
paired endpoint image was exactly equal. The unchanged source was wrong-signed on all 24
training motion pairs, with teacher-aligned gain `-0.029650` and NRMSE `1.030300`.

The run accepted its first five updates, at scales 1/2, 1, 1/2, 1/2 and 1/4. Attempt 6 had
finite gradients, recurrent states, outputs, objectives and projected parameters. Its β1=0
transaction advanced all three pending counters from 5 to 6 and produced a valid descent
direction: autograd and independently materialized directional derivatives were `-0.785668`
and `-0.785674`. The mandatory scale-1/16 finite difference also decreased the objective from
`1.046161` to `1.020021`, but its measured derivative was only `-0.418238`; the `46.77%`
relative error exceeded the fixed 20% limit. Under the registered rules this was a fatal
numerical-control failure, so no ordinary scale trials ran.

The pending transaction was discarded. The stopped resume retains optimizer hash
`9ecbe922bcfc0bf5a6a14edc49836ac803af600363211bd75903aea7723d441f`, exactly the
attempt-6 pre-transaction hash, with all counters at 5. The update-50 midpoint and update-100
final data were never generated or evaluated. No assisted-hover claim, native-attitude
reintegration, full-native hover, gate flight or promotion was authorized. This run is closed
and must not be resumed under its protocol.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-throttle-assisted-beta1-zero-001/report.json` (SHA-256
`1b951c0613ebe904cdb39186a053fcba72d5da7408170b73fac85693b915cc36`). The ignored
stopped resume has SHA-256
`330fa83864a711a7630df8109f1a0550360b111f950e6572b25a4919b978acb3`.
