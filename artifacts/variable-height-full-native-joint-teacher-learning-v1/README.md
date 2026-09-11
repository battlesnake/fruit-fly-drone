# Full-native joint teacher learning v1

This is the compact terminal record of the joint C/P/D experiment registered in
commit `f0e5614` and implemented in `6595d56`. It restarted from the unchanged
original native visual-hover checkpoint with fresh Adam state and optimized the
fixed, source-normalized joint teacher objective. The deployed actor contract stayed
at 320x200 RGB plus roll and pitch, native recurrence, and the two foreleg motor
outputs; no privileged actor input or external memory was added.

All 25 proposed updates were accepted. The joint objective fell from `1.000000` to
`0.541016`. Aggregate common-component NRMSE fell from `2.533490` to `0.294855`, and
height-component NRMSE fell from `0.593303` to `0.548998`. Endpoint damping NRMSE fell
from `1.458401` to `1.265771`, a `13.208%` source-relative improvement.

The run nevertheless failed its preregistered update-25 gate, which required at least
15% endpoint-damping improvement. More importantly, endpoint damping remained
wrong-signed in every training scene: correct-sign fraction stayed at zero and
teacher-aligned gain moved only from `-0.432969` to `-0.254692`. Thus the loss learned
to attenuate the anti-damping response but did not reverse it into useful braking.
The early damping horizons changed little, while most improvement appeared at the
25-step endpoint.

Roll, pitch and yaw source-relative NRMSE remained just inside their 0.05 limits at
`0.039849`, `0.049969` and `0.049507`. The failed milestone was persisted and the run
stopped with accepted update 25 and Adam counters of 25. No development candidate,
closed-loop hover or gate flight was evaluated, and no controller was promoted.

Compact measurements are in [`report.json`](report.json). The complete ignored report
is `runs/variable-height-hover/full-native-joint-teacher-learning-001/report.json`
(SHA-256 `85199ca29943c7a01b1a18aede8c4425f29dad03f74c67c3b1cd5b92b3b352c9`).
The ignored stopped resume is retained for audit only; it is not an authorized source
for subsequent training.
