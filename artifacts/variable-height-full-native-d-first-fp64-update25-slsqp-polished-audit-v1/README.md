# Update-25 SLSQP-support-polished reference audit v1

This is the compact record of the final bounded numerical audit registered in commit
`3340c27` and implemented in `e2966ab`. It loaded the existing frozen update-25 archive;
it did not instantiate the fly controller or regenerate metrics, gradients, constraint
rows, or Adam state.

The audit passed. Each of three fresh CPU-archive reloads completed the bound-aware
projection in three rounds. The active set fixed 5,506 edges in round 1, another 15 in
round 2, and converged with 5,521 fixed edges and zero box violation in round 3. Every
active/fixed-set path was identical, and all pairwise primal and objective repeat
differences were zero.

In every round, unchanged SLSQP selected support mask 840 from its own coefficients. The
fixed one-shot FP64 SVD solve found that four-coordinate support full rank and reproduced
the exhaustive primary solution to at most `4.22e-16` Gram-induced primal distance and
`1.98e-16` objective difference. Polished KKT residuals were at most `3.45e-17`; polished
original-unit primal violations were at most `2.81e-16`. All registered gates passed.

No actor candidate was materialized, no optimizer or state retained, no development or
fresh data used, no hover run made, and nothing promoted. This result qualifies only the
frozen-input polished numerical reference and permits a separately registered continuation
from exact accepted update 24.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-update25-slsqp-polished-audit-001/report.json`
(SHA-256 `9abe6a4b433927e6136395dbe7abec869bce8daccd5b03284fc91cb076f0e075`).
