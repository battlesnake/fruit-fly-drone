# Source-initialized joint multi-time diagnostic

This controlled repeat changed only the joint audit's update-0 controller from the
unpromoted exact-mass overfit vector back to the promoted source controller. It reused the
same cases, recorded histories, targets, scales, seeds, optimizer, 150-update budget, and
even the same exact-mass-vector regularization reference as the preceding audit.

Source initialization reduced the initial worst threshold-normalized training margin from
18.75 to 5.00. The selected update-150 result reached 4.37 on training and 4.05 on disjoint
holdout, compared with its own 5.46 holdout initialization. It fit the 0.75-second throttle
contrast to 0.254 normalized RMSE, but 1--5-second contrast remained near 1.0 and roll
fidelity remained about 0.81--0.90. It therefore failed the unchanged training-fit gate.

The result rejects this 150-update exact-action configuration. It does not reject the
native recurrent graph, and it does not justify a longer post-hoc budget. No checkpoint
was promoted. See [`report.json`](report.json) and the rejected
[`selected-vector.json`](selected-vector.json). Re-run with:

```bash
scripts/run_gate_joint_multitime_overfit.sh \
  --initialization source \
  --output-dir runs/gate/joint-multitime-source-init-v1
```
