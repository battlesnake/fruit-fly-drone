# Native visual-motion learnability diagnostic

This is the compact terminal record of the motion-only experiment registered in commit
`77340c1` and implemented in `28c8ba9`. It restarted from the original native visual-hover
checkpoint and used the complete 165,122-neuron MaleCNS graph. The actor received only
320x200 RGB at 125-degree HFOV plus roll and pitch; native recurrence was its only history.
No accelerometer, velocity, target height, timer, phase, engineered history or external
state entered the fly.

The fixed training bank at seed `450991` contained the exact 24-combination factorial of two
height errors, two height signs, two endpoint-speed magnitudes and three response horizons.
Every update used the whole bank in fixed order and optimized only normalized paired
equal-endpoint throttle contrast. The teacher supplied labels only. The source replay,
teacher/foreleg sign control, endpoint-image identity, actor-input, trajectory-sign,
normalization and output-bound preflights all passed.

The registered sampled-trajectory limitation was material: last-frame displacement implied
1.27-2.93 times the declared continuous endpoint speed, although every sign matched. This
was recorded before optimization and was not corrected post hoc.

All first 17 updates passed the derivative ladder and were accepted at ordinary scale 1.
They reduced full-prefix motion NRMSE from `1.031749` to `1.001594`, only a 2.92% relative
improvement. More importantly, they did not learn braking. Correct-sign fraction remained
0/24 and teacher-aligned gain moved from `-0.030875` to `-0.001592`: the wrong-signed
response was attenuated almost to zero rather than reversed. Prediction RMS fell 94.75%,
from `0.002292` to `0.000120` motor units, while the target remained `0.043661`.

This was a degenerate sensitivity-erasure path. The descriptive height-sign contrast also
collapsed from `0.038864` to `0.001979` motor units, while pair-common throttle moved by
`0.270700` RMS (maximum `0.343440`) relative to source. The terminal throttle range was
0.6247-0.6663 rather than the source's 0.3156-0.4382. Motion-only squared error therefore
found an easy route that removed visual response and shifted common output, but it did not
show a useful recurrent damping computation.

Attempt 18 had finite recurrence, outputs, gradients, bounds and a valid 17-to-18 pending
Adam transaction. Its full materialized direction was still negative (`-2.19597e-4`). The
scale-1/16 probe passed local agreement and changed the objective by `-1.35303e-5`, but the
adjacent scale-1/32 change (`-6.13431e-6`) fell just below the registered replay-noise
threshold (`6.91662e-6`). All smaller changes were also below threshold, so the required two
adjacent above-noise probes could not qualify. The run stopped as
`derivative_probe_noise_limited_inconclusive`; the pending transaction was discarded and
controller/Adam restoration passed exactly with all counters at 17.

The update-25 milestone was not reached and the disjoint development bank was never
generated. No staged curriculum, hover, native-attitude reintegration, gate flight or
promotion was authorized. This endpoint is closed and must not be resumed.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-throttle-motion-only-001/report.json` (SHA-256
`ad0f4f0524be5f9a769b74d890d8f8b6925ba25921bb283ac52b8089ddc22c5c`). The ignored
stopped resume has SHA-256
`04c4b4e0b598f37c3808a7740bad0160c2826d20584149c841694696d1d0e28d`.
