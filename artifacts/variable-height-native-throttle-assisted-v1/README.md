# Teacher-attitude-assisted native throttle v1

This is the compact terminal record of the curriculum registered in commit `6d2c8c2`
and implemented in `b11efa3`. The experiment kept the actor contract at 320x200 RGB plus
roll and pitch, native MaleCNS recurrence, and two-foreleg motor output. The analytical
teacher owned roll, pitch and yaw through the same foreleg stick plant; the fly alone owned
throttle. No accelerometer, privileged actor input, external state machine or engineered
history was added.

The 64-case end-to-end teacher positive control passed with 100% hover success. Its foreleg
motion-pair control had the correct sign for all 24 pairs and a maximum steady RC error of
`0.001070`. Every motion pair ended with exactly equal RGB and attitude. The untrained source
was wrong-signed on all 24 motion pairs, with teacher-aligned gain `-0.029651` and NRMSE
`1.030300`.

The formal run accepted 18 updates in 20 attempts. Attempt 8 was one ordinary, fully restored
rejection. The accepted scales were eight at 1, seven at 1/2, and one each at 1/4, 1/8 and
1/16. Attempt 20 generated finite gradients, recurrent states, outputs and current-objective
reports, and its Adam transaction advanced all three pending counters from 18 to 19 exactly
once. It nevertheless failed the mandatory direction control: the materialized Adam proposal
had `grad(J) dot displacement = +0.961656`, so momentum made it an ascent direction for the
current sampled objective. The finite-difference and ordinary scale trials were therefore not
run.

The candidate transaction was restored. The stopped resume retains Adam hash
`5b9df11d8396df8808af3fab1b916fa9d1b0a1b88031df2439e917152b52da0e`, exactly the
attempt-20 pre-transaction hash, with all counters at 18. The update-50 midpoint and update-100
final data were never generated or evaluated. No assisted-hover claim, native-attitude
reintegration, full-native hover, gate flight or promotion was authorized. This stopped
controller is audit-only and must not be resumed under the closed protocol.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-throttle-assisted-001/report.json` (SHA-256
`ca1562b8764174ac804185104eeb365fa2826241c7f292430d0a81700c177a34`). The ignored
stopped resume has SHA-256
`46d2d9314b347f9d62168ca5aadbc9538f8889e9d82503095b0488af71d48d68`.
