# Sampled-horizon mass-free distillation diagnostic

This rejected run distilled the validated mass-free visual/accelerometer reserve teacher
at seven recurrent endpoints, sampling one endpoint per optimizer update. The 0.5-second
teacher throttle contrast was genuinely tiny (mean 0.000248, standard deviation 0.000169
on holdout cases), and the fixed 0.01 motor-command normalization floor behaved correctly:
its error remained about 0.028 of that scale. This confirms that removing exact-mass
labels eliminated the artificial early mass-identification requirement.

The fixed 200-update first round nevertheless failed four-axis fidelity. Improvements at
the middle endpoints interfered with the endpoints at 0.5 and 5 seconds: by update 200,
the 5-second throttle pair-mean error had risen to 4.24 normalized RMSE, while roll error
remained 0.78--1.08 across endpoints. The best global snapshot was update 25 with a worst
component error of 2.26, far above the fixed 0.25 limit. DAgger, final flight evaluation,
and promotion therefore did not run, and no checkpoint was produced.

[`report.json`](report.json) contains the full run. [`candidate-vector.json`](candidate-vector.json)
and [`archive.pt`](archive.pt) preserve the rejected selected vector and snapshots. Re-run
the sampled-horizon protocol from commit `7b8ec69` with:

```bash
scripts/run_gate_multitime_distillation.sh \
  --output-dir runs/gate/multitime-reserve-v1
```
