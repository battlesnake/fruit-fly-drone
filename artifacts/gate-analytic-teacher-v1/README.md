# Diverse-gate analytical teacher

Four fixed analytical teachers were compared on 256 balanced diverse-gate cases after a
0.5-second promoted-controller prefix. All four achieved 100% success; the selected
visual/accelerometer teacher with exact training-only mass had the lowest mean gate-plane
radial error (0.0166 m).

The frozen choice then achieved 100% success overall, on both mass halves, both lateral
sides, and both obliquity signs in 1,024 fresh flights, with 0% ring collisions and misses.
It therefore passes the fixed teacher preflight for multi-time native distillation. The
teacher uses privileged relative geometry and exact simulator mass only to make training
labels; neither may enter the deployed fly actor, and this result does not itself count
toward the direct-sensor goal.

See [`candidate.json`](candidate.json) and [`report.json`](report.json). Re-run with:

```bash
scripts/run_gate_analytic_teachers.sh \
  --output-dir runs/gate/analytic-teacher-v1
```
