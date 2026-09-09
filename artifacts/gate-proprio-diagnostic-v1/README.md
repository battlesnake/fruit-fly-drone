# Throttle-stick proprioception diagnostic v1

This is a **rejected sensor-causality diagnostic**, not the current flight controller.
It extends `gate-accel-v2` with four traced left-prothoracic MaleCNS `SNpp50`/`SNpp51`
cells annotated as femoral chordotonal-organ claw homologues. The cells receive
complementary measurements of the completed-step virtual throttle-stick position. Their
mapping to high/low position is an engineering choice: the MaleCNS release does not
resolve flexion-versus-extension tuning for these four cells.

Only 73 edges touching the added nine-cell circuit were searched; the 1,138 old neurons
and 4,469 old edges were copied from `gate-accel-v2`. On a fresh balanced 1,024-flight
suite the candidate scored 49.41%, versus 41.41% for the unchanged warm start. However,
holding the position input constant still scored 48.73%, swapping it between episodes
scored exactly 49.41%, and a position-by-acceleration intervention was effectively zero.
The extra anatomical capacity improved aiming, but these controls do not demonstrate
useful throttle-position sensing.

Files:

- `connectome.npz`: 1,147-neuron, 4,542-edge derived MaleCNS subgraph;
- `connectome-manifest.json`: source hashes, selection, attribution, and interface caveat;
- `candidate.pt`: rejected diagnostic checkpoint, retained for auditability;
- `report.json`: full search, validation, intervention, and causal-control results.

The derived graph follows the upstream MaleCNS CC BY 4.0 terms recorded in
[`data/README.md`](../../data/README.md). The physiological context is the open-access
[2025 femoral chordotonal organ circuit study](https://doi.org/10.1038/s41467-025-59302-3).
