# Factorial damping-route training v1

This is the terminal result of the bounded height-null training protocol frozen in
commit `da95d2a`. It restarted from the preserved source, used the expanded native
80,454-edge/883-bias mask, and generated each damping gradient and its transient
marker-error Jacobian from the same 2x2 height-by-velocity histories. Common throttle
and the height-by-velocity interaction remained diagnostic rather than gated.

All 25 attempted updates were accepted at full scale. This confirms that the
factorial decomposition and per-scene height-null projection are operational: on the
64-scene development cohort, every matched transient height-response ratio remained in
0.98100-1.00578, and the independent existing small/medium height assays retained
0.98155/0.98207 of source. RPY, validity and motor-bound checks also passed.

The useful-progress gate nevertheless failed by a wide margin. Development damping
NRMSE fell only from 0.594106 to 0.589698, a 0.742% improvement against the required
25%. Damping remained wrong-signed in all 64 scenes, and teacher-aligned gain moved only
from -0.3393 to -0.3300. The interaction NRMSE fell from 0.06081 to 0.05932, also a small
attenuation rather than a qualitative change.

The run used most of its declared local parameter radius: the fixed-denominator source
metric reached 0.0004214 of 0.0005. Absolute common-throttle drift, deliberately
unanchored, reached 0.005645 RMS and 0.007661 maximum on the source-pair diagnostic.
Together these measurements show that the weak result is not another rejection caused
by the old common anchor or by the new height-null gates. The current expanded mask and
local metric did not demonstrate useful independent damping authority. It found a safe
local descent direction, but no correctly signed braking response was demonstrated; the
result does not claim that the unrestricted native graph lacks that authority.

The fresh cohort was not exposed. The retained endpoint is audit-only: no checkpoint
was promoted and no closed-loop hover test was run. This closes further local damping
optimization in the current mask/metric family. A next attempt must be declared as a
broader control/representation redesign rather than a threshold relaxation or a longer
continuation of this endpoint.

Compact measurements are in [`report.json`](report.json). The complete ignored report
is `runs/variable-height-hover/factorial-damping-train-001/report.json` (SHA-256
`26449d62329907284a3492ccb688b42fa2bf07a71bbeedbf4fcb91fbea812b33`). The ignored
training state and nonpromotional endpoint are retained for audit and hashed in the
compact report.
