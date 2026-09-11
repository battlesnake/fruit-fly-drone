# Canonical RK4 readout trust region

This is the compact terminal record of the corrected repeat registered in commit `3d8d1d9`
and implemented in `ae2f2c5`. It reconstructed the prior masked RK4/Adam direction once,
persisted a canonical CPU archive, reproduced the old scale-1 result functionally, and only
then opened the preregistered upward scale ladder.

All reproduction gates passed. Scalar differences were at most `7.16e-7`, the canonical
archive reloaded bit-exactly, and the old scale-1 output tensors and metrics differed by at
most `9.24e-7`. Scale 1 remained rejected under its original verdict, so this repeat did not
retroactively reinterpret the failed audit.

Scale 16 produced about `0.00956` NRMSE improvement in both replay modes, but failed the
source-relative pair-common throttle RMS constraint (`0.006773 > 0.005`). Scale 8 was the
first fully passing candidate: fixed/full NRMSE improvements were `0.004801`/`0.004800`,
pair-common throttle drift was `0.003412` RMS and `0.004342` maximum, maximum motor magnitude
was `0.434785`, and RPY drift stayed at float32 noise scale. All native bounds, endpoints,
finiteness, mask identity, source restoration and optimizer restoration checks passed.

No candidate was retained. The result authorizes only preregistration of bounded last-hop
fitting with the complete fixed line-search ladder and unchanged source-relative preservation
constraints. It does not authorize training execution, hover, gate flight or promotion.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-rk4-readout-trust-region-canonical-001/report.json`
(SHA-256 `cd65d0d77fe88af1335d7b6baf1b57cf1b95c9e22bfa7962324dbc80cfc99d73`). The ignored
48 MiB canonical direction archive has physical SHA-256
`08564d6a1bf330b2e06015f93942da969ebdfe3becb5d51996c30a86a8cd57b2`.
