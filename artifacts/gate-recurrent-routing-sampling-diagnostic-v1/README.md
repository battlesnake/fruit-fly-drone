# Recurrent-routing sampling diagnostic v1

This no-training audit compares like-for-like errors for the retained update-100
recurrent-routing vector. It replays the full native controller from zero on all 48
training and 16 validation geometry pairs, using the same fixed sensory histories,
development normalization, source reference, and six-group aggregation. It also
reconstructs the exact 100 balanced eight-pair minibatches and bootstraps geometry pairs
within each split for 2,000 replicates.

Training aggregate NRMSE was 0.38081 and validation aggregate NRMSE was 0.38190, a ratio
of only 1.0029. The observed validation-minus-training normalized-MSE gap was 0.00083;
its pair-bootstrap 95% interval was `[-0.02959, 0.02801]`, which spans zero. The early
heavy group remained the limiting error on both splits (0.595 training, 0.608 validation),
followed by early light (0.482 and 0.479).

The 100 minibatches contained 800 pair exposures, balanced equally between the two course
halves. Exposure-weighted training normalized MSE was 0.144999 versus 0.145019 under
uniform weighting, an absolute relative difference of 0.014%; the preregistered sampling
distortion threshold was 10%. Thus neither a generalization gap nor sampling distortion
was established. The correct classification is unresolved early-window fitting error in
the present circuit/objective, not demonstrated overfitting.

No optimizer update ran, no fresh held-out case was consumed, and no flight, topology
expansion, compilation, merge, or promotion occurred. The complete protocol, per-group
metrics, per-pair losses, exposure sequence, and bootstrap result are in
[`report.json`](report.json). Re-run the code introduced at commit `a53f06a` with:

```bash
scripts/run_gate_recurrent_routing_sampling_audit.sh \
  --output-dir runs/gate/recurrent-routing-sampling-audit-v1
```
