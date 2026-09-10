# Paired recurrent-routing continuation diagnostic v1

This paired experiment tests two explanations for the shared early-window fitting error.
Both arms start from the retained update-100 magnitudes on the same 125 existing signed
edges, reset Adam state, use learning rate 0.003 and norm clipping at 1, and see an
identical sequence of balanced eight-pair minibatches. The control retains equal weighting
across the three throttle windows. The treatment assigns weights `(4,1,1)/6` to early,
middle, and late window losses. The normalized first-0.2-second four-axis source anchor is
unchanged.

The control's validation worst early-group NRMSE improved from 0.6082 to 0.5909 after 50
additional updates and 0.5540 after 100. The treatment reached 0.5664 and 0.5184. These
are 8.9% and 14.8% improvements at the continuation checkpoint, both short of the
required 20% (approximately 0.4865). Every middle/late safety check passed, and prefix
motor RMSE remained very small at 0.000516 and 0.000595.

Neither arm passed the continuation gate, so both stopped at 100 additional updates
rather than using the maximum 200. The treatment was the diagnostic development selection
with 0.518 worst-group NRMSE and 22.25% aggregate RMSE improvement over the constant, far
short of the final 0.25/50% replay gates. Equal-window continuation reached 24.03%
aggregate improvement but worse early error. This rules out the tested amounts of simple
additional training and fourfold early weighting as sufficient fixes for the current
125-edge circuit. It does not distinguish a stronger temporal objective from missing
upstream anatomical degrees of freedom.

No fresh held-out case was consumed, no assisted flight ran, and no parameter was merged,
compiled, or promoted. The source controller remains unchanged. The complete protocol and
metrics are in [`report.json`](report.json); the stopped diagnostic selection is in
[`selected-magnitudes.json`](selected-magnitudes.json). Re-run the code introduced at
commit `e8dd932` with:

```bash
scripts/run_gate_recurrent_routing_continuation.sh \
  --output-dir runs/gate/recurrent-routing-continuation-v1
```
