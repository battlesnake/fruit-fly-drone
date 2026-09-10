# Variable-height hover bounded curriculum v1

This is the rejected first execution of the mixed variable-height hover curriculum in
[`docs/HOVER_TO_GATE_PLAN.md`](../../docs/HOVER_TO_GATE_PLAN.md). It started from the
preserved paired-dynamic controller and used the exact 50% closed-loop DAgger, 25%
paired-marker and 25% attitude-recovery lesson cycle. Current-policy prefixes covered
early, middle and late flight; physical evolution was detached while gradients flowed
through the connectome's native recurrent state.

The source controller was safe but did not hover at the commanded marker height under
the new independently randomized scene protocol. On the fresh 256-flight audit it had
0% marker-step success and 0.474 m mean altitude RMSE. It nevertheless retained a strong
causal response to the live marker step (slope 1.085), zero ground contacts, zero invalid
flights and a passing fixed-scene legacy response.

Training stopped at the first scheduled decision point, update 50. Absolute-height RMSE
worsened from 0.472 m to 0.611 m on the 64-case development suite, 17.2% of cases touched
the ground, and the legacy paired response crossed its rejection threshold (NRMSE 0.267,
limit 0.25). The selected checkpoint was therefore restored to update 0; no trained
candidate was promoted.

This separates two problems that the first curriculum tried to solve at once. The source
understands the marker in its original scene but generalizes poorly to randomized and
held-out wall/floor combinations. The next bounded experiment is a response-preserving
scene/amplitude bridge; only a bridge passing both the legacy and randomized pair gates
may seed another mixed flight curriculum.

The large checkpoints and full raw report remain below ignored
`runs/variable-height-hover/bounded-001/`. The compact machine-readable evidence is in
[`report.json`](report.json). No large binary is committed by this diagnostic.
