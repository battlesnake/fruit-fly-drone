# Privileged mass oracle v1

This artifact measures an upper bound; it is **not a deployable fly sensor and does not
count toward the direct-sensor gate goal**. The frozen `gate-accel-v2` controller and
connectome are unchanged. At every neural step, the simulator's exact episode mass
selects an external differential bias on the existing throttle-positive and
throttle-negative motor pools:

```text
z = clamp((mass_scale - 1) / 0.08, -1, 1)
bias = -0.05 + 0.10 z
```

On the fresh balanced 1,024-flight audit, the correct mass-conditioned bias scored
1024/1024. The unchanged controller scored 438/1024, the selected mass-independent
constant trim scored 416/1024, and shuffling mass labels within matched gate-geometry
strata scored 103/1024. The result establishes that the present task is strongly limited
by mass-dependent collective-throttle calibration. It does not establish that a physical
vehicle can measure its own mass directly.

`candidate.json` records the two oracle parameters. `report.json` contains the complete
development grid, held-out validation, final results, hashes, and causal checks. No new
controller checkpoint is needed because no controller parameter changed.
