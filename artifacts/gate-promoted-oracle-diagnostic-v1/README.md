# Promoted-controller mass-oracle calibration

This bounded diagnostic recalibrated the privileged throttle-pool mass bias for the
promoted controller on the harder diverse-gate distribution. It searched 25 declared
light/heavy endpoint pairs, performed one 3×3 half-step refinement, selected on a disjoint
validation seed, and froze the winner before a 1,024-flight preflight.

The selected endpoint biases were -0.1625 at mass scale 0.92 and +0.05 at 1.08. Exact mass
was causally useful: success was 79.6%, versus 20.6% unchanged, 13.5% for the selected
constant trim, and 6.1% with labels swapped inside matched pairs. But the candidate failed
the fixed 90% teacher gate. Negative-lateral success was 100%, while positive-lateral
success was only 59.2%, showing that throttle-only calibration had reached a steering
ceiling. This is a rejected training-only oracle and does not count toward fly-controlled
flight.

See [`report.json`](report.json) for the full grid, validation, and preflight record. Re-run
with:

```bash
scripts/run_gate_promoted_oracle.sh \
  --output-dir runs/gate/promoted-oracle-v1
```
