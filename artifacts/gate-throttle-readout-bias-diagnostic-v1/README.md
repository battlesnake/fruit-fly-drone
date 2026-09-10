# Native throttle-motor bias diagnostic v1

This final frozen-history readout audit fitted the same 37 legal fixed-sign magnitudes in
`[0,32]` together with bias deltas in `[-2,+2]` for the seven existing throttle motor
neurons. It added no edge, neuron, state, input, or clock. All other parameters and the
recorded recurrent histories were fixed. FP64 trust-region least squares used the same
64-pair development histories and selected between source and bound-32 starts using
development loss only.

The selected source-start fit converged after 287 function evaluations, with maximum
projected gradient `1.47e-9`. On 128 newly seeded held-out geometry pairs it improved
aggregate RMSE by 53.98% over the constant baseline, passing the aggregate criterion. Its
worst time/mass NRMSE was 0.274, however, above the required 0.25. Only the earliest
light-mass group failed; late light mass reached 0.243 and all other groups were at most
0.211. Twenty of 37 edge magnitudes were effectively zero, five were at 32, and five of
seven bias deltas were at a bound.

The replay gate therefore failed and the planned teacher-steering-assisted flight test
did not run. Per the preregistration, this closes the frozen-history 37-edge/seven-bias
readout family. The result shows that existing motor-neuron biases improve the fit but do
not make this bounded family sufficient, particularly during the early light-mass
transient. It is not a flight failure or evidence that native recurrence is insufficient:
the histories were frozen, upstream anatomical routing was not adapted, and several
selected parameters remained bound-limited.

The complete protocol and metrics are in [`report.json`](report.json), and the selected
diagnostic parameters are in [`selected-parameters.json`](selected-parameters.json). No
parameter was compiled, merged, promoted, or tested in flight; the source controller is
unchanged. Re-run the code at commit `e134c3d` after regenerating the frozen histories if
necessary:

```bash
scripts/run_gate_throttle_readout_bias.sh \
  --output-dir runs/gate/throttle-readout-bias-v1
```
