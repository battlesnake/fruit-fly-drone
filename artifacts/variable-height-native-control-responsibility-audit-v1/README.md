# Native hover control-responsibility audit v1

This diagnostic tested the unchanged full-MaleCNS visual-height source with three
controlled paired sensory histories. No parameter was changed, no probe output was fed
to the actor, and no result is a closed-loop hover claim.

The useful result is that the visual interface and recurrent graph are not missing the
state needed for damping. A 26-neuron sparse linear probe recovered the conventional
teacher's height-error correction from every anatomical partition with held-out R2
between 0.964 and 0.989. It recovered vertical-motion damping from every partition with
R2 between 0.898 and 0.933 even though the paired trajectories ended at the exact same
pose and retinal input. Separately selected shuffled-label controls reached at most
0.299 R2.

Native use is the problem. Height response remained correct in all 128 test pairs and
was 0.910 of the teacher contrast. The motion response was wrong-signed in every pair
and had magnitude 0.625 of the teacher contrast in the wrong direction. This may be
lagged position feedback rather than an explicitly inverted velocity estimate, but
either interpretation predicts poor damping.

The command-history assay also explains why a tonic-state claim would be premature.
After 0.26 seconds of identical input, the native output retained 0.463 of the preceding
command contrast with the correct sign. During the next 0.20 seconds it fell through
zero after roughly 0.08 seconds and reversed, reaching -0.246 before decaying. Recent
command history is richly decodable, but the deployed dynamics do not maintain it as a
stable neutral-throttle estimate.

No coarse anatomical partition passed the preregistered selective-causality gate. For
height, one-shot optic-lobe/visual replacement removed 54.7% of response but changed
common motor output by 0.00352, over the 0.0025 limit. Descending-neuron replacement was
selective but removed only 27.8%, just below the fixed 30% threshold. These near misses
are useful engineering clues, not permission to rename either region a damping module.
Identical-history and whole-state controls stayed at or below `1.2e-7` motor units.

The preregistered decision is therefore **native routing and readout training**, not new
sensors, a graph rebuild, or full progressive handoff yet. The next preflight opens only
real at-most-two-synapse routes from descending/VNC interneurons to the throttle motor
pools, excluding other motor neurons as intermediates. It targets correctly signed
motion damping while protecting common throttle, visual height response, R/P/Y behavior,
topology and transmitter signs. FeCO remains a later controlled test of physical leg
state; this audit neither requires it nor proves that it would be useless.

Compact measurements are in [`report.json`](report.json). The complete report, including
all selected body IDs, ridge validation results, bootstrap intervals and intervention
measurements, is ignored at
`runs/visual-hover/responsibility-audit-001/report.json` (SHA-256
`6c3072fceb8c2e1348968a7fe08ff46c685f51be84e9e48c4c34fa008db3fa59`).
