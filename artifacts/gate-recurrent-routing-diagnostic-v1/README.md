# One-layer recurrent-routing diagnostic v1

This experiment moved beyond isolated saved neural states. It replayed every 1.5-second
sensory prefix from zero through the entire native MaleCNS recurrent controller while
training magnitudes on 125 existing, fixed-transmitter-sign edges. The preregistered and
hashed mask contains 88 audited four-hop-path edges entering the 19 throttle-return
sources from outside that set, plus all 37 edges from those sources into the seven
throttle motor neurons. The 19 return-to-return edges, every other magnitude, all biases,
and all time constants remained frozen.

Training used 48 of the 64 saved development geometry pairs in balanced eight-pair
minibatches. The target after 0.5 seconds was the saved absolute reserve throttle motor
drive, equally weighted across three time windows and both mass halves. An independently
normalized loss anchored all four motor outputs to the source during the first 0.2
seconds. The other 16 development pairs were reserved for validation. Source replay
matched the saved throttle trace within `1.79e-7`, and the preregistered directional
gradient check passed (analytic `0.764231`, finite difference `0.764191`).

Minibatch normalized throttle MSE fell from 0.944 at update 1 to 0.101 at update 80. At
the update-100 validation gate, however, worst-group NRMSE was 0.608 against the 0.50
continuation limit and aggregate improvement over the constant baseline was only 20.5%.
The protected prefix remained close to source at 0.000604 RMSE. The run therefore stopped
at the preregistered midpoint. Magnitudes were not ceiling-limited: the maximum selected
value was 3.03 under the legal limit of 8.

This is a failed midpoint continuation gate, not a fresh held-out replay or flight
failure; neither later stage ran. The logged minibatch normalized MSE and validation
worst-group NRMSE are different statistics, so their headline values alone do not
establish a train/validation generalization gap. A separate frozen-checkpoint audit is
needed to compare like with like. The result also does not show that native recurrence is
absent. No magnitude was compiled, merged, or promoted, and the source is unchanged.

The complete protocol and metrics are in [`report.json`](report.json), the exact ordered
mask and signs are in [`preregistered-mask.json`](preregistered-mask.json), and the stopped
diagnostic vector is in [`selected-magnitudes.json`](selected-magnitudes.json). Re-run the
code at commit `57d24e1` after regenerating the saved histories if necessary:

```bash
scripts/run_gate_recurrent_routing.sh \
  --output-dir runs/gate/recurrent-routing-v1
```
