# Throttle readout constraint diagnostic v1

This paired frozen-history diagnostic separates an inadequate optimizer budget from the
fixed transmitter-sign constraint at the 37 native throttle-incoming edges. Both arms use
the same development histories, exact nonlinear one-step membrane update and motor-pool
mapping, target normalization, and two starts. FP64 bounded trust-region least squares
receives an analytic Jacobian and at most 300 function evaluations per start. The only
difference is legal magnitudes in `[0, 8]` versus diagnostic effective weights in
`[-8, 8]`.

Both legal fits converged to the same development loss from independent source and Adam
update-200 starts, with projected-gradient optimality below `1.1e-9`. On 128 newly seeded
held-out geometry pairs, the selected legal fit reached a worst-window/mass NRMSE of
0.366 and improved aggregate RMSE by 46.75% over the constant baseline. It therefore
missed both preregistered gates. Thirty-two of its 37 magnitudes ended at a bound (22 at
zero and ten at eight), consistent with an active constraint rather than unfinished local
optimization.

The relaxed-sign arm passed the same untouched held-out test: worst-group NRMSE was 0.235
and aggregate improvement was 57.41%. It reversed 15 of the 37 original edge signs. The
relaxed optimizer exhausted its budget rather than converging, but this does not weaken
the discriminator: it had already crossed both held-out sufficiency thresholds, whereas
the legal arm had fully converged and failed.

This is evidence that the fixed signs of this final 37-edge readout obstruct the required
throttle correction. It is not a licence to flip MaleCNS transmitter signs. The useful
next search must find a different path through existing anatomically signed circuitry.
Neither fitted arm was compiled or promoted, and the source controller is unchanged.

The full protocol and metrics are in [`report.json`](report.json); the two diagnostic
weight vectors are in [`selected-weights.json`](selected-weights.json). Re-run the exact
code at commit `1823b74` after regenerating the preceding frozen histories, if necessary:

```bash
scripts/run_gate_throttle_readout_constraint.sh \
  --output-dir runs/gate/throttle-readout-constraint-v1
```
