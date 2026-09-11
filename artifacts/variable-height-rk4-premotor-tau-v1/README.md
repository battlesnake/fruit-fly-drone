# RK4 premotor time-constant preflight

This compact artifact records the premotor temporal-shaping audit preregistered through commit
`393bbfb`, implemented in `27acf55`, corrected under the fail-closed amendment `c318a1f`, and
executed from correction commit `169cbce`.

The first `-001` run stopped safely on a missing combined-summary convenience key. Its report
and valid 61 MiB direction archive remain ignored under `runs/`; neither is duplicated here.
The correction hash-locked and reused that exact direction and wrote only to `-002`.

The physical-gradient and finite-difference controls passed. The largest 4 ms-RMS request was
selected; source-relative per-cell bounds reduced its actual tau change to 2.8769 ms RMS. It
improved full zero-state NRMSE by 0.03196 while preserving collective throttle and all RPY axes
on every training block. The response nevertheless remained wrong-signed in all 96 cases.

The audit stopped before development because the unchanged source narrowly missed the stricter
per-block RK4-M1 versus K32 contrast tolerance on two blocks. The candidate passed that solver
test on all four, but the gate required both. No tolerance was relaxed after inspection, no
held-out data were generated, the source was restored exactly, and no candidate was retained.

This closes the fixed 883-neuron premotor-tau-only route while retaining one useful design clue:
native time-scale separation has measurable damping leverage. The next intervention must use a
different internal decomposition, such as signed optic-flow temporal pathways feeding native
descending/VNC control circuits.

Compact measurements are in [`report.json`](report.json). The full corrected report remains at
`runs/variable-height-hover/native-rk4-premotor-tau-preflight-002/report.json` with SHA-256
`c5062fb18039d248d045eebbbfc17ccacc43522d5a34c2d239f7a14b309ba64a`.
