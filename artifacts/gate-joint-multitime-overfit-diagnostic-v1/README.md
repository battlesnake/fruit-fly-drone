# Joint multi-time capacity diagnostic

This bounded audit replaced sampled endpoints with one complete 500-step recurrent replay
per update and a fixed weighted average of all seven endpoint losses. Each of roll, pitch,
yaw, throttle contrast, and throttle pair-mean had roughly equal effective weight. Eight
balanced, all-horizon-valid training pairs and eight disjoint holdout pairs were evaluated
every 25 updates; holdout data never selected the snapshot.

The 150-update configuration did not meet its fixed training-fidelity gate. Its worst
threshold-normalized training error fell monotonically from 18.75 to 5.03, while the
holdout score improved from 19.52 to 4.52. All native parameter families had finite,
nonzero complete-prefix gradients. The result rejects this budget and initialization, not
the fly network's recurrent capacity.

The initialization was the earlier 0.75-second exact-mass conditional-overfit diagnostic
vector. That vector begins with a wrong-sign reserve-teacher contrast at 0.75 seconds and
was never a promoted checkpoint, so the next controlled audit should initialize directly
from the promoted source controller. No checkpoint was promoted here.

See [`report.json`](report.json) and the rejected [`selected-vector.json`](selected-vector.json).
Re-run from commit `ff4add9` with:

```bash
scripts/run_gate_joint_multitime_overfit.sh \
  --output-dir runs/gate/joint-multitime-overfit-v1
```
