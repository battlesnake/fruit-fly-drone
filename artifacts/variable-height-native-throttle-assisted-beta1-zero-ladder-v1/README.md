# Assisted native throttle with a derivative ladder

This is the compact terminal record of the source-restarted curriculum registered in
commit `4b25f56` and implemented in `7fcc1ec`. It retained the original actor contract:
320x200 RGB at 125-degree HFOV plus roll and pitch entered the complete MaleCNS graph,
native recurrence was the only memory, and the native front-leg outputs supplied throttle
through the foreleg/stick plant. The analytical teacher supplied physical roll, pitch and
yaw during this diagnostic. No accelerometer, velocity, target height, phase, external
state or engineered history entered the fly.

The revised derivative control worked as intended. All 53 attempted updates found two
adjacent, above-noise probes with local derivative agreement. Thirty-three attempts needed
two probes, 15 needed three and five needed four; no attempt exhausted the registered
ladder. The probes remained diagnostic and ordinary backtracking independently selected 50
updates. Attempts 6, 9 and 26 were ordinary, exactly restored rejections. Adam's accepted
counters ended at 50 for every parameter family.

The fixed update-50 midpoint gate failed cleanly. Assisted hover succeeded in 23 of 64
cases (`0.359375`) against the registered 50% threshold. There were no ground contacts or
invalid cases, mean final-window height RMSE was `0.247881 m`, and mean vertical-speed RMS
was `0.063173 m/s`.

The more decisive failure was visual motion feedback. All 64 equal-endpoint motion pairs
had the wrong throttle-contrast sign. The native prediction RMS was only `0.001636` motor
units against a teacher target RMS of `0.043659`; teacher-aligned gain was `-0.024702` and
NRMSE was `1.025002`. Correct-sign fraction was zero independently at 15-, 20- and 25-frame
horizons. Every terminal paired image was exactly equal, and all recurrent states, outputs
and metrics were finite.

The accepted-step record explains why the combined objective did not repair damping.
Across the 50 independently sampled accepted candidates, fixed-burn dense-loss improvement
summed to `125.304640`, while motion-loss improvement summed to only `0.006764`; 21 accepted
candidates worsened their sampled motion loss. Full-prefix improvements showed the same
imbalance (`159.758883` dense versus `0.012661` motion). These sums are within-step selection
diagnostics rather than a trajectory-wide loss curve, but they show that equal `0.5/0.5`
scalar weights did not create comparable optimization pressure. The result is consistent
with the easier static/common throttle task dominating a much weaker recurrent motion
Jacobian.

The run stopped at its registered midpoint and is closed. Final-evaluation seeds were not
opened; no controller was promoted, and native-attitude reintegration, full-native hover
and gate flight remain unauthorized. Compact measurements are in
[`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-throttle-assisted-beta1-zero-ladder-001/report.json`
(SHA-256 `549259e08db3dd1dc43afda0ea85b76f86fac115a5a28bb0aca6700dbccb9ed9`).
The ignored stopped resume has SHA-256
`51ee3e289b77c7ece63f239732a5b41fcac5ccbc7efec4a38fbe5e8833ab3a08`.
