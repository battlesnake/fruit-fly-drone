# Legal throttle-readout ceiling diagnostic v1

This frozen-history diagnostic made one preregistered increase of the legal magnitude
ceiling, from 8 to 32, for the same 37 fixed-transmitter-sign edges entering the seven
native throttle motor neurons. It changed no edge sign, topology, bias, time constant,
sensor input, or recurrent state. FP64 trust-region least squares used the same 64-pair
development histories as the preceding readout audits and selected between source and
converged bound-8 starts using development loss only.

Both starts converged under the fixed optimality rule. On 128 newly seeded held-out
geometry pairs, the selected fit improved aggregate RMSE by 52.09% over the constant
baseline, passing that half of the screen. Its worst time/mass NRMSE was 0.303, however,
above the required 0.25; the two earliest groups were 0.303 for light mass and 0.294 for
heavy mass. Twenty-one of 37 magnitudes were effectively zero and four remained at the
new upper bound.

The replay gate therefore failed and the planned teacher-steering-assisted flight test
did not run. This is not a failed flight experiment. It shows that a one-time 8-to-32
ceiling increase does not make this frozen-history, fixed-bias readout sufficient. It
does not prove that transmitter signs alone prevent native recurrent control: the fit is
nonconvex, four magnitudes remain ceiling-limited, intrinsic motor-neuron biases were
frozen, and recurrent histories were held fixed. The earlier sign-relaxed result remains
evidence of a restriction in this particular readout family, not permission to reverse
MaleCNS transmitter signs or evidence of impossibility for the native controller.

The complete protocol and metrics are in [`report.json`](report.json), and the selected
diagnostic vector is in [`selected-magnitudes.json`](selected-magnitudes.json). No vector
was compiled, merged, promoted, or tested in flight; the source controller is unchanged.
Re-run the code at commit `9dde390` after regenerating the frozen histories if necessary:

```bash
scripts/run_gate_throttle_readout_ceiling.sh \
  --output-dir runs/gate/throttle-readout-ceiling-v1
```
