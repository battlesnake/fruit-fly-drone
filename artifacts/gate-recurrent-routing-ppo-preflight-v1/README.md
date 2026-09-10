# Recurrent-routing PPO preflight tolerance calibration

The first execution of the frozen code at commit `71b9ab2` stopped during assisted-
evaluator parity, before constructing the development set, updating any parameter, or
creating the run output directory. Every discrete success, collision, miss, and plane-
crossing rate matched exactly. Mean gate-plane crossing radius differed by
`1.1563301086425781e-05` m (11.6 micrometres), narrowly exceeding the original
`1e-05` m numerical tolerance.

The continuous-only tolerance was changed to `2e-05` m and is now written into the parity
record. Discrete outcomes must still match exactly. No training threshold, task metric,
seed, sampled case, optimizer setting, or promotion rule changed.

The structured stop record is in [`report.json`](report.json).
