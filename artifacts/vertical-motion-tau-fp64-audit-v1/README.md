# Vertical T4/T5 FP64 tau audit v1

The independent full-FP64 diagnostic passed every preregistered gate. Both tau probes
matched finite differences at all four tested scales, the FP32 and FP64 analytic
gradients agreed closely, and the fixed-state exponential-Euler derivative matched its
closed form to machine precision.

The conditional production FP32 Adam step then reduced the fixed minibatch loss by
18.19% at the first backtracking multiplier. It was disposable: source and local
identity were restored, no candidate was retained, development and acceptance remained
unopened, and peak CUDA reserved memory was 9.44 GiB.

This result authorizes the separately bounded 100-update vertical-motion commissioning
run in FP32. It does not authorize hover, gate flight, or promotion. The compact
[`report.json`](report.json) hashes the complete ignored report and start marker.
