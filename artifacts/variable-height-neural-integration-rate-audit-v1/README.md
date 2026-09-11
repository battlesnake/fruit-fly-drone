# Native neural-integration-rate audit

This is the compact terminal record of the source-only numerical audit registered in commit
`e0d07a9` and implemented in `e7971df`. It tested whether coupling the fly's recurrent-state
update to the 50 Hz camera/action clock was too coarse. The camera and prospective stick
commands remained at 50 Hz. Only the number of native MaleCNS state integrations performed
while holding each RGB and roll/pitch observation fixed changed.

The audit loaded the original `paired-dynamic-001` controller and the already-opened 24-pair
motion bank. It generated no new cases or seeds and performed no learning. One immutable,
lossless 721 MiB RGB/attitude/target cache was shared by K=1, 2, 4, 8 and 16. Its semantic
SHA-256 is `43a576a2abc01283a7b5a72560e3c9ff921bfaaf3e5a481ac6715019843f7d0e`;
the ignored physical file's SHA-256 is
`68ea98cc60a8969eb38c1bc62db698bb232ddb6ddaebac426a850ee31596ecb8`.
All paired terminal images were bit-identical.

The K=1 control reproduced the earlier source NRMSE, aligned gain and prediction RMS within
`4.23e-8`, `4.10e-8` and `8.15e-9`, respectively, comfortably inside the preregistered
`2e-5` tolerance. Every condition loaded and restored the source controller exactly and had
finite recurrent state, outputs and metrics.

No adjacent refinement passed both registered convergence limits. From K=1→2, 2→4, 4→8
and 8→16, normalized contrast-vector RMS differences were 0.11592, 0.06839, 0.03553 and
0.01595; the limit was 0.01. Terminal four-axis motor RMS differences were 0.01746, 0.00684,
0.00433 and 0.00265; the limit was 0.005. Thus the last two pairs passed the motor-output
limit, but K=8→16 still missed the contrast limit by about 60%. The audit correctly selected
no internal rate.

The behavioral measurements are descriptive and were not used to select K. Faster native
integration did not repair the sign. Correct-sign fraction was 0/24 at every rate, while
aligned gain became more negative: -0.03088 at K=1, -0.11459 at K=2, -0.16881 at K=4,
-0.19787 at K=8 and -0.21081 at K=16. NRMSE correspondingly worsened from 1.03175 to
1.22419. This is useful evidence that the continuous-time limit of the unchanged source
circuit likely has a stronger wrong-signed response; it is not yet a numerical convergence
result because the registered ladder ended before reaching tolerance.

No training rate, hover, gate flight or promotion was authorized. The registered next branch
is a separately declared finer reference (or higher-order solver), after which any temporal-
credit/time-constant training still requires its own preregistration.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-neural-integration-rate-audit-001/report.json` (SHA-256
`ded09b91a114d08039d5334a937e36812b69ba55183634ae4bd21ed9864baff7`).
