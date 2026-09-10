# Variable-height visual-response bridge v1

This rejected diagnostic asked whether a response-preserving bridge could teach the
source controller to ignore randomized scene nuisance before closed-loop hover training.
Each optimizer update accumulated gradients from small- and medium-amplitude paired
marker lessons, legal cross-band source-response preservation, and dynamic source replay.
All parameters were updated once after all four gradients had been computed. Every
training and prefix marker stayed outside the held-out `[0.90, 1.10]` m interval.

The v1 scene split paired matching wall/floor style families in training and reserved
opposite-family combinations for evaluation. It failed quickly and was stopped at the
update-25 safety check. Genuine held-out pair NRMSE was unchanged (0.7883 to 0.7890),
while closed-loop altitude RMSE regressed 16.7% (0.4726 m to 0.5516 m). The legacy
fixed-scene response remained passing at 0.1066 NRMSE, and there were no ground contacts
or invalid flights.

This result does not establish an optimization conflict because v1 never measured a
fixed training-support pair matrix. It establishes only that adaptation did not improve
the deliberately withheld style combinations safely. Bridge v2 changes one factor: all
four style combinations occur uniformly in training, while absolute marker heights,
texture realizations and trajectory seeds remain held out. It also adds fixed
training-support and fresh-height audit matrices.

No candidate was promoted and no large binary is committed. The full ignored report and
checkpoint remain under `runs/variable-height-hover/bridge-001/`; compact evidence is in
[`report.json`](report.json).
