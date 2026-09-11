# RK4 readout trust-region extension

This is the compact terminal record of the extension registered in commit `06c904f`,
implemented in `67de0bd`, and given a pre-start CLI-only fix in `4c27a7f`. Its purpose was to
reconstruct the prior masked RK4/Adam direction exactly, then test only new upward scales
16, 8, 4 and 2 under the unchanged output-preservation gates.

The scalar computation reproduced extremely closely. The three source objectives differed
from their registered values by at most `3.58e-7`, the gradient objective by `4.77e-7`, and
the directional derivative by `6.37e-11`, all far inside the declared `2e-5` tolerance. The
mask controls and empty-to-one Adam counters also reproduced.

The byte identities did not reproduce across CUDA processes. The pending-controller semantic
hash was `4c84cc…` rather than `73d633…`, and the optimizer hash was `5ae04a…` rather than
`0692f5…`. The small scalar and gradient-family changes are consistent with GPU reduction
nondeterminism, but the frozen protocol required exact hashes, so the audit correctly stopped
without evaluating scale 16 or any later scale. It did not test the trust-region hypothesis.

Source and optimizer were restored exactly. No candidate was retained and no fitting, hover,
gate flight or promotion was authorized. A corrected repeat must persist one reconstructed
direction and validate it by numerical and functional replay rather than assuming cross-process
byte identity from floating-point CUDA reductions.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-rk4-readout-trust-region-001/report.json` (SHA-256
`03c4ee348d51c351611b425de184a3d64cf3211f1a75ddf1c6f9e50006dcfc5e`).
