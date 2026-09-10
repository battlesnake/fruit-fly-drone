# Frozen throttle-routing diagnostic v1

This diagnostic tests whether the source controller's existing recurrent activity can
represent the throttle correction required after steering has been stabilized. It records
64 development and 128 disjoint held-out matched geometry pairs through 1.5 seconds, with
the source supplying throttle and the mass-free reserve teacher supplying only steering
after 0.5 seconds. One clock-free readout is shared across three time windows and both mass
halves.

Two unconstrained analysis probes passed on the untouched held-out set. A ridge readout of
the 93 acceleration-to-throttle path neurons had a worst-window/mass normalized RMSE of
0.090 and improved aggregate RMSE by 87.5% over the best constant correction. A stricter
probe using only the 19 native throttle-return sources plus seven throttle motor states
also passed, narrowly: worst-group NRMSE was 0.242 against the 0.25 limit, with a 63.4%
aggregate improvement.

The exact native readout did not pass. Fitting only the 37 existing nonnegative,
fixed-transmitter-sign incoming magnitudes through the real one-step neural update and
motor-pool mapping reached a worst-group NRMSE of 0.499 and improved aggregate RMSE by
30.2%, short of the required 50%. Its loss was still declining at the fixed 200-update
budget, so this run does not distinguish a bounded-optimizer failure from a constraint of
the native signed readout.

Fixed-history controls add an important qualification. Replacing live acceleration with
constant 1g changed the 93-node probe prediction by 0.0172 motor-drive units on average,
but changed the return-source probe by only 0.00030 and the native fit by 0.000072. Thus
action-relevant information is present at the return sources, but this audit does not show
that the accelerometer is its carrier there; unchanged image, angle, and stick histories
can also contain the relevant trajectory information.

All source-forward, live-replay, and zero-change native-readout parity checks passed within
`1e-6`. The complete metrics and protocol are in [`report.json`](report.json), and fitted
diagnostic coefficients are in [`fitted-readouts.json`](fitted-readouts.json). The exact
267,205,077-byte replay tensor remains under the ignored `runs/` tree rather than bloating
Git history; its SHA-256 is recorded in [`replay-histories.sha256`](replay-histories.sha256)
and it is deterministically regenerated from the reported seeds. No readout was compiled
or promoted.

Re-run the code at commit `44ab7bf` with:

```bash
scripts/run_gate_throttle_routing_audit.sh \
  --output-dir runs/gate/throttle-routing-audit-v1
```
