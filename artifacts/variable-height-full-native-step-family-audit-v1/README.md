# Full-native step-size and family audit v1

This is the formal result of the diagnostic protocol implemented in commit
`14d32d3870f0061b05e120aa32c541069225d960`. It regenerated the failed
full-native Adam displacement, replayed complete-update scales 1, 1/2, 1/4 and
1/8 on the frozen training bank, selected the largest safe training scale, and
tested only that candidate on development. Edge-only, bias-only and
time-constant-only full-scale replays were diagnostic and were not combined.

The run is formally an audit-control failure. The factorial caches reproduced
byte-for-byte and all registered scalar controls reproduced within tolerance,
including the directional derivative and family displacement RMS values. The
source-driven CUDA attitude rollouts were numerically reproducible to
`5.96e-8`, but their regenerated tensor hashes differed from the hashes frozen
by the protocol. The failed hash gate is preserved; the run was not revised or
rerun after seeing it.

The counterfactual substantive result is still useful. Scale 1/2 was the
largest training-safe displacement and also passed the development safety
gates. It strongly improved the collective-dominated joint loss while keeping
roll, pitch and yaw below 0.05. Damping NRMSE improved by only 0.000263 on
training and worsened by 0.000466 on development; endpoint damping remained
wrong-signed in every scene. This is evidence only for safe collective
calibration, not transferable damping learning.

The family probes explain the first-step behavior. Edge updates caused most of
the collective improvement but violated the pitch guard; bias updates were
safe but produced only a tiny damping change; time-constant updates were
negligible at the registered learning rate. They do not establish additive
family effects and do not authorize selecting or combining families after the
fact.

No candidate was retained or promoted, no multi-step fit was authorized, and
no closed-loop hover ran. Compact measurements are in
[`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-step-family-audit-001/report.json`
(SHA-256 `edfea71ba0c15d8fa512150ddfbe01bb7a3b4159b2c2676be63d9f713e168677`).
