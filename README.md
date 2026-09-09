# Fly connectome quadcopter pilot

Research workspace for a **connectome-constrained artificial pilot**: a controller whose
recurrent core is constrained by the MaleCNS v1.0 fruit-fly connectome. Its front-leg
motor activity moves the two sticks of a virtual Mode-2 transmitter; the measured stick
positions are the four FPV acro controls (roll, pitch, yaw, throttle).

The important scientific boundary is that the connectome is a static wiring diagram.
Its published "weights" are counts of detected synapses between segments, not measured
electrophysiological strengths and not pretrained artificial-network weights. The data do
not provide neuron dynamics, receptor-specific signs, delays, plasticity, or a runnable
fly mind. A successful project would show that a controller constrained by fly anatomy
can be trained for the task—not that a biological fly knows how to fly a quadcopter.

## Repository map

- [`docs/PLAN.md`](docs/PLAN.md): architecture, simulator decision, curriculum,
  experiments, safety, and milestones.
- [`docs/HEADLESS_SPIKE.md`](docs/HEADLESS_SPIKE.md): reproducible WSL2/MJWarp
  acceptance tests and initial RTX 5080 results.
- [`data/README.md`](data/README.md): exact MaleCNS v1.0 snapshot, provenance,
  licensing, and what was intentionally not downloaded.
- `data/raw/malecns-v1.0/`: locally downloaded Feather tables (ignored by Git because
  the compact core is about 1.9 GB).
- [`papers/README.md`](papers/README.md): downloaded paper bundle and primary-source
  reading list.
- [`scripts/fetch-malecns-v1.0.sh`](scripts/fetch-malecns-v1.0.sh): resumable,
  checksum-verifying reproduction of the download.

Run `scripts/fetch-malecns-v1.0.sh` to restore or verify the official source files.

## Proposed first implementation slice

1. Ingest and profile the Feather graph; join connection endpoints to annotations and
   neurotransmitter predictions without modifying raw counts.
2. Define fixed, stateless mappings from FPV pixels and inertial values into identified
   sensory neurons, and from identified front-leg motor neurons into simulated joints.
3. Build a headless one-gate MuJoCo environment and a conventional privileged-state
   teacher for training only; an episode supervisor owns arm/disarm.
4. Tether a biomechanical fly over a virtual Mode-2 transmitter and make measured
   fore-tarsus/stick motion—not a separate learned decoder—the quadcopter action.
5. Train a small, explicitly selected visual-to-motor MaleCNS subgraph and compare it
   with matched generic and rewired baselines.
6. After the CPU loop is correct, batch physics and FPV rendering with MJWarp; only then
   expand the graph, add multi-gate role changes, AirSim visual evaluation, and validated
   Betaflight SITL.

## Primary sources

- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [HHMI Janelia MaleCNS project](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [MaleCNS v1.0 downloads](https://male-cns.janelia.org/download/)
- [Peer-reviewed paper DOI](https://doi.org/10.1016/j.cell.2026.08.015)
- [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
