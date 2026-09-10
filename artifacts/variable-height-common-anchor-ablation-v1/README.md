# Common-throttle anchor ablation v1

This is the terminal result of the diagnostic protocol frozen in commit `4eafdf0`.
It restarted from the preserved source and used the same 80,454-edge/883-bias expanded
native mask, source-relative parameter metric, gradient cases and guard cases as the
upstream damping run. The only intervention was to remove every absolute
throttle-to-source objective, projection and gate. Absolute collective drift and error
against the analytical throttle teacher remained diagnostics.

The absolute common-throttle anchor was binding, but it was not the main obstacle. The
ablation accepted 24 of 25 attempted updates, including 21 full-scale updates, and let
source-pair common-throttle RMS rise to 0.011228 (4.49 times the old 0.0025 limit).
Nevertheless, motion NRMSE on the deliberately reused 64-pair benchmark improved only
3.552% (1.380882 to 1.331832), far below the preregistered 25% useful-progress gate.
Damping remained wrong-signed on all 64 pairs, and teacher-aligned gain remained
negative, moving from -0.5671 to -0.5133.

The endpoint remained finite and passed the ablated preservation checks. Dynamic R/P/Y
errors were 0.00676/0.00906/0.00642, and the legacy RPY aggregate was 0.00671 against
the 0.05 limit. Small and medium height contrasts retained 0.90202 and 0.90013 of the
source response, respectively, leaving almost no margin above the 0.9 floor. The
source-relative parameter metric reached 0.000437 of its 0.0005 radius. Attempt 25 was
rejected at every scale because the height ratio fell below 0.9; the smallest trial also
failed the per-guard-bank progress floor.

Because useful progress failed, the previously unconsumed fresh cohort was never
evaluated. No checkpoint was promoted and no closed-loop hover run was started. This
rules out the simple explanation that preserving absolute collective output was by
itself preventing the existing damping objective from finding a useful native route.
The retained endpoint is audit-only and must not be used as a controller.

Compact measurements are in [`report.json`](report.json). The complete ignored report,
including every attempt and guard bank, is
`runs/variable-height-hover/common-anchor-ablation-001/report.json` (SHA-256
`6106e1e60a16fb7e868a02dcca67bb740ea022946f99da6dc6fe0e011b17b0ce`). The ignored
training state and endpoint are retained only for audit; their hashes are recorded in
the compact report.
