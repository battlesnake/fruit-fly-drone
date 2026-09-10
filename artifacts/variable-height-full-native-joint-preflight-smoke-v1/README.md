# Full-native joint preflight smoke v1

This preserves the reduced four-scene CUDA smoke test of the restored one-step
full-native joint-fitting preflight implemented in commit `63f4614`. It used the
registered training and development seed numbers with smaller banks, so those seeds
have been exposed; the later eight-scene run must not call them unseen. The full
protocol, thresholds and learning rates were committed before this recorded smoke and
were not changed in response.

The implementation checks passed. Cached source replay differed by at most
`5.96e-8`, opposite-motion endpoint images were exactly equal, the measured factorial
identity reconstruction passed, and the teacher's constant commands reached the
requested RC through the foreleg/stick plant within 0.00878. The actual post-bound Adam
displacement had a negative derivative whose forward finite difference agreed within
2.20%, and all parameters were restored exactly.

The smoke candidate reduced joint normalized MSE strongly on both reduced banks, but it
failed the frozen RPY safety gate. Training roll/pitch source NRMSE became
0.05242/0.08798 and development roll/pitch became 0.05748/0.08716, against 0.05 per
axis. Endpoint damping also remained wrong-signed in every scene. No parameter was
retained, no checkpoint was written, and no closed-loop test ran.

This is an implementation smoke, not the registered eight-scene preflight result. It
predicts that the exact one-step proposal may be too disruptive to attitude output,
but it does not authorize changing the preregistered gate and does not test whether a
different optimizer or multi-step schedule could learn full-native joint control.

Compact measurements are in [`report.json`](report.json). The complete ignored report
is `runs/variable-height-hover/full-native-joint-preflight-smoke/report.json` (SHA-256
`fc55bc178ffc47d54612f1c476ec911128896bd2bcec76badd00d3022eb38833`).
