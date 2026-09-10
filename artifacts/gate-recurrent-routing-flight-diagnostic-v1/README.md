# Recurrent-routing assisted-flight diagnostic v1

This read-only diagnostic evaluates the source controller and both stopped
recurrent-routing continuation arms on the same 256 fresh, matched 12-second flights.
Each condition starts its native neural state from zero. During the first 0.5 seconds the
source drives all four axes; afterwards, the analytical reserve controller supplies only
steering while the fly controller retains throttle. A separate positive control switches
all four axes to the reserve controller after the same source prefix.

The positive control passed all 256 flights and every mass, lateral-side, and obliquity
stratum. Its mean gate-plane radial error was 0.0788 m and its p90 was 0.1636 m, validating
the cases, flight classifier, and intervention path. All conditions reached the gate
plane, so their conditional radial-error summaries cover every episode.

The source succeeded on 38.28% of flights: 0% for lower-mass cases and 76.56% for
higher-mass cases. The equal-window and fourfold-early continuation arms both fell to 0%
success, missed every gate, and increased mean radial error from 1.224 m to approximately
3.46 m. Each arm's geometry-paired success difference from source was -38.28 percentage
points, with a cluster-aware 95% interval of `[-41.96, -34.60]` points. Neither arm met
the preregistered light-mass and worst-mass/lateral improvement gates.

This valid negative result closes the tested 125-edge replay-imitation continuation
family: lower open-loop replay error did not transfer to complete closed-loop flight and
in fact damaged assisted flight. The audit does not separate changed sensory trajectories,
accumulated action error, or behavior beyond the 1.5-second training horizon, and it does
not show that native connectome recurrence is inadequate. The next training family should
optimize complete-flight task outcomes while keeping recurrence and memory inside the fly
network.

No optimizer update, selection, interpolation, merge, compilation, or promotion occurred
in this audit. The source controller remains unchanged. The complete protocol and metrics
are in [`report.json`](report.json). Re-run the code introduced at commit `9f36877` with:

```bash
scripts/run_gate_recurrent_routing_flight_audit.sh \
  --output-dir runs/gate/recurrent-routing-flight-audit-v1
```
