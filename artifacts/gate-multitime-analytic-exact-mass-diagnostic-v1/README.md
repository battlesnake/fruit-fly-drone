# Exact-mass analytical distillation diagnostic

This rejected run started from the successful 0.75-second conditional-overfit parameter
vector and trained every native edge magnitude, neuron bias, and time constant through
complete recurrent prefixes at seven endpoints from 0.5 to 5 seconds. Its exact-mass
analytical target had passed every diverse-flight preflight stratum at 100%.

After the fixed 200-update first round, the best snapshot failed the preregistered action
fidelity gate, so DAgger, final flight evaluation, and promotion did not run. It learned
the 0.75-second light/heavy throttle contrast (20.7% normalized RMSE at the selected
snapshot; 14.9% at update 200), but its 0.5-second contrast stayed at 68.6% with essentially
zero predicted separation. The selected snapshot also missed 0.5-second roll, pitch, and
throttle pair-mean targets. This is evidence against demanding an unnecessary privileged
mass correction before the allowed sensor history makes it observable—not evidence that
removing that target alone solves the remaining imitation problem.

No checkpoint was produced. [`report.json`](report.json) contains the full run;
[`candidate-vector.json`](candidate-vector.json) and [`archive.pt`](archive.pt) preserve
the rejected selected vector and all selection snapshots.

Reproduce the stopped protocol with the earlier exact-mass teacher explicitly selected:

```bash
scripts/run_gate_multitime_distillation.sh \
  --teacher-spec artifacts/gate-analytic-teacher-v1/candidate.json \
  --allow-exact-mass-teacher \
  --output-dir runs/gate/multitime-analytic-v1
```
