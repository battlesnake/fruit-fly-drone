# Assisted-throttle Taylor-convergence audit v1

This is the compact record of the training-only audit registered in commit `33ce76d` and
implemented in `9f2e114`. It used only the stopped accepted-5 β1=0 controller, exact Adam
state, frozen block-zero histories and persisted failed attempt-6 sample. An exclusive start
marker closed the audit against replay before computation. No fresh, midpoint or final data
were generated or opened.

The audit reproduced the stopped run within the registered `2e-5` tolerance. Its three
fixed-burn baseline replays measured only `3.5763e-7` maximum objective noise, giving a
`3.5763e-6` objective-change threshold. The pending β1=0 transaction again advanced all three
counters from 5 to 6 and produced a full materialized directional derivative of `-0.785671`.

The registered scale 1/16 failure reproduced: its objective decreased by `0.026140`, but the
measured derivative `-0.418240` differed from the predicted `-0.785665` by 46.77%. Scale 1/32
still missed the 20% limit at 23.33%. Scales 1/64, 1/128, 1/256 and 1/512 all passed, with
relative errors of 11.65%, 5.81%, 2.90% and 1.41%. All objective changes were far above the
noise threshold and all recurrent, output, parameter-bound and canonical controls passed.

More decisively, the absolute Taylor residual decreased by a factor of approximately four at
every halving: the five smaller/larger ratios were `0.24946`, `0.24966`, `0.24951`, `0.24942`
and `0.24371`. This is local derivative convergence consistent with ordinary finite-step
curvature; it is not retroactive acceptance of attempt 6.

The controller and optimizer were restored exactly to accepted-5 hashes, and no candidate,
optimizer transaction or continuation scale was retained. This pass authorizes only a new,
separately preregistered source restart that separates a sufficiently local derivative probe
from ordinary optimization-step selection. It establishes neither useful motion damping nor
assisted hover, native-attitude reintegration, gate flight or promotion.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-throttle-beta1-zero-taylor-audit-001/report.json` (SHA-256
`0ee33e31f913a3de8e3b27934c1205706f4a417c7b3a33f12e190d4084930e50`). The exclusive
start marker has SHA-256
`938c82562e3f4bb644b141b792f820ada030d3da80f18669c0a0d5d2ffdc16bb`.
