# Dense recurrent DAgger diagnostic v1

This bounded run trained every native parameter in the 1,138-neuron controller on dense
successful trajectories from the validated mass-free reserve teacher, then refreshed the
replay buffer with current-student histories. Deployment still used only current FPV,
roll/pitch, body-Z specific force, and the connectome's native recurrent state.

The teacher flew 64/64 diverse cases. FP32 sensor replay matched collection within
`2.1e-7`, and central finite differences agreed with the exact 100-step truncated
gradient for edge magnitudes, biases, and time constants. The fixed development suite
started at 19.9% success. The best snapshot, update 50, reached 21.1% overall by improving
heavy-mass success from 37.5% to 42.2%, but light-mass success fell from 2.3% to zero.
No checkpoint achieved the preregistered five-point midpoint gain, so training stopped at
update 200 and nothing was promoted.

The run also exposed an objective-scaling error: throttle MAE was divided by the RMS of
the full hover command (`0.388`) while roll used `0.0153`. This made an equal raw throttle
error about 25 times cheaper than roll. The controlled follow-up changes only throttle's
scale to the RMS teacher correction relative to the promoted controller's shadow output.

The rejected state is retained in [`selected-vector.json`](selected-vector.json), the
validation snapshots in [`archive.pt`](archive.pt), and the complete protocol and metrics
in [`report.json`](report.json). Re-run the exact v1 code at commit `cee0d4d` with:

```bash
scripts/run_gate_dense_dagger.sh \
  --output-dir runs/gate/dense-dagger-v1
```
