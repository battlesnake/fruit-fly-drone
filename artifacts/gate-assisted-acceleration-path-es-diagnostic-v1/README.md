# Assisted acceleration-path ES diagnostic v1

This bounded experiment asked whether a wider recurrent throttle pathway could solve the
mass asymmetry that the six-parameter static throttle readout could not. It searched the
282 existing signed edge magnitudes on all paths of at most four hops from the three
accelerometer inputs to the throttle motor pools. Those paths contain 93 neurons. The
successful steering vector from the preceding assisted experiment was preserved, but not
used: a mass-free teacher supplied roll, pitch, and yaw after the same 0.5-second native
prefix so the search could isolate throttle learning.

The fixed generation-20 gate failed. On 256 diverse matched cases, the source achieved
39.84% overall success, split between 0% light-mass and 79.69% heavy-mass success. The
best candidate reached 44.14% overall and 88.28% heavy-mass success, but light-mass
success remained exactly zero. Its early throttle commands were also almost identical
between light and heavy cases. The search therefore stopped without spending the final
1024-case causal audit budget.

This rejects the tested edge-magnitude ES configuration, not recurrence in the MaleCNS
graph generally. The path parameters were directly disjoint from the successful steering
readout, although recurrent behavioral coupling remains possible; 36 of the 37 throttle
readout gain edges were inside the searched path. Nothing was compiled, merged, or
promoted, and reserve steering remains a training-only intervention.

The complete preregistered protocol and measurements are in [`report.json`](report.json),
the rejected parameter vectors are in [`archive.pt`](archive.pt), and the diagnostic
selection is in [`selected-theta.json`](selected-theta.json). The earlier successful
18-parameter steering vector is retained independently in
[`preserved-steering-vector.json`](preserved-steering-vector.json). Re-run the exact code
at commit `08f77b5` with:

```bash
scripts/run_gate_assisted_acceleration_path_es.sh \
  --output-dir runs/gate/assisted-acceleration-path-es-v1
```
