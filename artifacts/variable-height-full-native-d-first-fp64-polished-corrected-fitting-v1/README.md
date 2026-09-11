# Polished FP64 corrected fitting continuation v1

This is the compact record of the continuation registered in commit `d1e5c82` and
implemented in `95e027f`. It restarted exact accepted update 24, reproduced the frozen
update-25 proposal and polished projection exactly, and loaded the archived pending Adam
transaction without regenerating the gradient or calling Adam again.

Update 25 passed every frozen-input, projection, canonical, finite-difference and optimizer
control. The existing C25-only nonlinear repair accepted it at proposal scale 1/16 and
correction scale 1. Updates 26 through 29 then passed ordinarily at scale 1/32. Training
endpoint-D NRMSE improved from `1.34913635` at the restart boundary to `1.34239793`, or
`7.95%` relative to the original source. Endpoint-D sign fraction nevertheless remained
zero and teacher-aligned gain remained negative, so this controller is not useful for hover.

The run stopped safely before installing update 30. Its polished projection, numerical
controls and one-step Adam accounting all passed, but no fixed nonlinear scale preserved
the source-output limits. At scale 1/32, only endpoint C25 failed, by `3.12e-5`. The existing
scale-1/16 repair was ineligible because height P20 exceeded its limit by `1.95e-6`. The
rejected controller and Adam transaction were restored exactly; the stopped resume contains
accepted update 29 and optimizer counters of 29.

No new development evaluation, terminal checkpoint, fresh qualification, closed-loop
hover or promotion occurred. This closes the source-preserving local-repair route; it does
not demonstrate infeasibility or inadequate recurrent neural capacity.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-polished-corrected-fitting-001/report.json`
(SHA-256 `48dedef889546a84916e04052ae7d6207bc6368159327e90025236b32455a3f7`).
