# Native recurrence diagnostic v1

This is a **causal memory demonstration and rejected controller experiment**, not a new
flight checkpoint. It asks whether the recurrent MaleCNS graph can store a launch-derived
quantity without an external history feature.

The source `gate-accel-v2` graph contains 1,138 nodes and 4,469 directed edges. Of those,
718 nodes are in non-singleton strongly connected components; the largest component has
558 nodes. A bounded training run changed only a selected 18-neuron, 21-edge anatomical
motif made from fixed-sign sensory paths and positive-feedback cycles.

In the retention intervention, every runtime input was made uninformative after 0.75
seconds: FPV was black, roll/pitch was zero, and acceleration was a constant 1g. At the
last training checkpoint, the motif's aggregate activity still ordered vehicle mass at
1.5 seconds with Pearson `r=0.921`. That persistence must be in the numerical connectome
state; no external history buffer, mass estimate, clock, or actor state machine was used.

The result does **not** solve calibration. The retained 1.5-second signal had slope
`0.447` and R² `-1.915`; continuing live inputs reduced it to `r=0.151`, slope `0.013`,
and R² `-8.504`. The selected stable checkpoint failed its preregistered criterion, so
the experiment skipped return-edge fitting and all flight evaluation. No checkpoint is
promoted.

[`report.json`](report.json) is a compact, committed record of the topology, interventions,
results, and hashes of the ignored full run reports. It is small JSON and therefore does
not require Git LFS. The derived graph continues to follow the upstream MaleCNS CC BY 4.0
terms recorded in [`data/README.md`](../../data/README.md).
