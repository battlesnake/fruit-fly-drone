# Assisted-throttle optimizer attribution v1

This is the compact record of the training-only audit registered in commit `cecdafe` and
implemented in `f09a869`. It used only the stopped accepted-18 controller, its exact Adam
state, frozen block-one histories, and the persisted failed attempt-20 sample. An exclusive
start marker closed the audit against replay before computation. No midpoint, final or fresh
data were opened.

The original Adam transaction reproduced the failed materialized direction at `+0.961657`
(absolute difference `9.03e-7`). Its edge and bias contributions were both positive. Changing
only the loaded optimizer's first-moment coefficient to zero changed the materialized direction
to `-5.662634`; the unprojected direction was `-5.672789`. All three counters advanced from 18
to 19 in both counterfactual transactions. The β1=0 transaction replaced every first moment
exactly with the current clipped gradient while preserving and updating the archived second
moments within dtype-aware rounding bounds.

The β1=0 scale-1/16 finite difference decreased the fixed objective by `0.317212`; its measured
direction was `-5.075396` versus autograd's `-5.662610`, a `10.370%` relative error under the
fixed 20% limit. Scale 1 failed both objective improvements, while scale 1/2 improved the fixed-
burn objective by `0.558523` and the independently replayed full-prefix objective by `0.494581`.
That scale was qualified for audit only.

On all frozen block-one teacher frames, accepted update 18 reduced teacher-target throttle RMSE
from `0.079912` to `0.067264`, a `15.83%` training-only improvement. Motion-pair NRMSE changed
only from `1.030300` to `1.025025`; all 24 pairs remained wrong-signed, though teacher-aligned
gain became slightly less negative (`-0.029651` to `-0.024602`). Thus this audit establishes
optimizer attribution, not learned braking, generalization or hover.

Controller and optimizer hashes after the audit exactly matched their inputs. No candidate or
optimizer state was retained. The pass authorizes only a separately preregistered run restarted
from the original source with `Adam(betas=(0, 0.999))`. It does not authorize resuming accepted
update 18, assisted hover, native attitude reintegration, gate flight or promotion.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-throttle-optimizer-attribution-001/report.json` (SHA-256
`cc4f66e168e4b6f2b1f7157c87584c24bfbfbf5eca579d8abdd4c3b13f53030c`).
