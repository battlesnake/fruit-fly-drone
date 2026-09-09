# Native motor-interface ES checkpoint v1

This is the best promoted native gate checkpoint so far, but it does **not** pass the 90%
annular-gate milestone.

A 24-parameter mirrored evolution strategy changed one internal bias and two signed
incoming-edge gains for each of the eight existing motor pools. The selected vector was
compiled into 196 existing edge magnitudes and 26 motor-neuron biases. Topology,
transmitter signs, zero edges, time constants, renderer, foreleg/stick mechanics, and
quadcopter physics remain fixed. No search vector, decoder, action scaler, estimator,
clock, or external history remains in the deployed actor.

On 1,024 fresh balanced paired flights, success rose from `42.97%` to `49.51%`. The paired
gain was `6.54` percentage points (95% normal-approximation CI `4.87–8.22`), both lateral
sides improved, and mean radial error at the gate plane fell from `0.748 m` to `0.663 m`.
Freezing the initial FPV frame reduced success to `4.88%`.

This checkpoint is not evidence of mass inference. Replacing acceleration with constant
1g produced `50.0%`, versus `49.51%` with live acceleration. The improvement is a better
static motor calibration: lower-mass success remains about `6%`, while higher-mass success
reaches `93%`. A subsequent experiment must improve both mass halves and demonstrate causal
use of live sensor history.

[`controller.pt`](controller.pt) is a 159 KiB generated checkpoint and therefore remains a
normal Git object rather than Git LFS, consistent with the earlier checkpoints. It uses the
shared [`gate-accel-v2` connectome](../gate-accel-v2/connectome.npz), whose hash is bound in
both the checkpoint and [`report.json`](report.json). The derived graph follows the upstream
MaleCNS CC BY 4.0 terms in [`data/README.md`](../../data/README.md).
