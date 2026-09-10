# Variable-height visual-response bridge v2

Bridge v2 changed one factor from the rejected v1 experiment: all four wall/floor style
combinations were balanced in every small-pair, medium-pair and dynamic-replay batch.
Absolute marker heights `[0.90, 1.10]` m, continuous texture realizations and trajectory
seeds remained held out. Fixed training-support and fresh-height matrices covered both
contrast amplitudes and all four style combinations at updates 0 and 25.

The added coverage produced a small, real in-support improvement, but not enough to be a
useful bridge. Mean training-support NRMSE improved 3.57%, with positive change in every
style combination; the preregistered update-50 requirement was 25%. Fresh-height matrix
NRMSE improved only 0.60%, and the primary genuine-pair NRMSE improved 0.75%.

More importantly, closed-loop marker-step altitude RMSE regressed 27.4%, from 0.4726 m
to 0.6020 m, so the update-25 safety gate stopped training. Legacy fixed-scene response,
tilt, validity and ground-contact safety were retained. The selected checkpoint was
restored to update 0 and no candidate was promoted.

Because v2 measured and slightly improved what it trained on while substantially
damaging absolute closed-loop calibration, further domain-bridge optimizer guesses are
paused. The next diagnostic measures lesson-gradient conflict and native visual signal
propagation at the source and rejected update-25 checkpoint.

The full report and checkpoints remain under ignored
`runs/variable-height-hover/bridge-all-001/`. Compact evidence is in
[`report.json`](report.json); no large binary is committed.
