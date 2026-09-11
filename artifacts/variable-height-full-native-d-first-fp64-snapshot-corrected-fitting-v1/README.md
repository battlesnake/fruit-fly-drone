# Snapshot-seeded FP64 corrected fitting continuation v1

This is the compact record of the continuation registered in commit `36e573e` and
implemented in `f038f11`. It loaded the qualified accepted update-21 controller and pending
Adam transaction exactly, reproduced both 25-value training metric trees, and began fresh
proposal generation at update 22.

Updates 22, 23 and 24 passed the unchanged projection, preservation and nonlinear selection
gates at ordinary scales 1/16, 1/32 and 1/32. Training endpoint-D NRMSE improved from
`1.35364187` at the snapshot boundary to `1.34913635`; sign fraction remains zero and gain
remains negative, so the controller is not ready for hover.

The run stopped safely before installing update 25. Its first free-coordinate L-BFGS solve
returned `ABNORMAL`, which was a preregistered hard failure, even though the computed
solution passed the original-unit primal and normalized-KKT gates and agreed with the
exhaustive-support reference. The active-set loop therefore returned before incorporating
5,506 edges that crossed their lower bound. Clamping that unfinished first-round result
produced a `0.048034` common-step-25 linearized violation; this does not demonstrate that a
completed bound-aware problem is infeasible.

The failed update-25 Adam transaction was restored exactly. The stopped resume contains
accepted update 24, optimizer counters of 24, and no update-25 candidate. There was no new
development evaluation, terminal checkpoint, fresh qualification, closed-loop hover or
promotion.

The next registered experiment is a frozen-input exhaustive-support primary-solver audit
of update 25. It will archive the proposal inputs once, use the existing exhaustive-support
solution as primary, complete the unchanged eight-round bound-aware loop, require three
identical reloads and all existing numerical gates, then evaluate the fixed six ordinary
scales on training only. It cannot retain a candidate or authorize continuation by itself.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-snapshot-corrected-fitting-001/report.json`
(SHA-256 `55aa8e0848d2b93b5493c2c6478b93f30485bd810520f0814b126a918aec1691`).
