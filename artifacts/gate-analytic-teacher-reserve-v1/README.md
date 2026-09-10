# Mass-free diverse-gate analytical teacher

The visual/accelerometer reserve teacher achieved 100% success on 256 balanced selection
flights and again on 1,024 fresh flights after a 0.5-second promoted-controller prefix.
Success was 100% separately for light and heavy drones, both lateral sides, and both gate
obliquity signs, with no ring collisions or misses. Mean gate-plane radial error was
0.0798 m.

Unlike the lower-error exact-mass teacher, this policy does not read or correct for the
simulator's mass. It is therefore the observation-compatible target for the next native
recurrent distillation run. The teacher still receives training-only relative gate
geometry; neither that geometry nor any teacher state enters the deployed fly actor, and
this result does not count as fly-controlled flight.

See [`candidate.json`](candidate.json) and [`report.json`](report.json). Re-run with:

```bash
scripts/run_gate_analytic_teachers.sh \
  --output-dir runs/gate/analytic-teacher-reserve-v1 \
  --teacher-modes visual_accelerometer_reserve \
  --selection-seed 1001031 --preflight-seed 1002031
```
