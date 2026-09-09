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
- [`docs/HOVER_MILESTONE.md`](docs/HOVER_MILESTONE.md): the first passing
  connectome-to-forelegs-to-sticks takeoff and visual-height hover result.
- [`docs/GATE_MILESTONE.md`](docs/GATE_MILESTONE.md): the first recorded annular-gate
  traversal and the larger randomized evaluation, including failed acceptance criteria.
- [`data/README.md`](data/README.md): exact MaleCNS v1.0 snapshot, provenance,
  licensing, and what was intentionally not downloaded.
- `data/raw/malecns-v1.0/`: locally downloaded Feather tables (ignored by Git because
  the compact core is about 1.9 GB).
- [`papers/README.md`](papers/README.md): downloaded paper bundle and primary-source
  reading list.
- [`scripts/fetch-malecns-v1.0.sh`](scripts/fetch-malecns-v1.0.sh): resumable,
  checksum-verifying reproduction of the download.

Run `scripts/fetch-malecns-v1.0.sh` to restore or verify the official source files.

## Current milestone

The first differentiable surrogate experiment now passes its goal-level acceptance test:
40 randomized 20-second episodes produced 100% lift-off, 92.5% successful visual-height
hold, 0.188 m final-ten-second altitude RMSE, 3.66 degree roll/pitch RMS, and no ground
recontacts.  A paired mid-flight test compared a changed world-space height band with an
unchanged continuation from identical physical and recurrent state; the band-induced
height difference had the correct direction in 100% of trials.  Freezing the initial FPV
image reduced goal success to zero.

This result uses a compact, auditable 498-neuron MaleCNS subgraph and abstract bilateral
two-axis foreleg/stick kinematics.  It is not an articulated fly-leg, FlyGym, or independent
MuJoCo transfer result.  See the milestone document and committed machine-readable report
for the exact boundary and remaining limitations.

The next annular-gate checkpoint is also implemented and recorded, but is not yet a
passing goal-level result.  A denser 1,122-neuron MaleCNS subgraph lifts and traverses a
1.24 m inner-diameter gate whose centre begins 0.8 m off-axis and whose plane is 20 degrees
oblique.  It succeeds on 40.5% of 1,024 held-out trials with randomized side, obliquity
sign, and mass; freezing the first FPV frame reduces success to zero.  The committed
[`showcase.mp4`](artifacts/gate-v1/showcase.mp4) contains one complete traversal with a
schematic rendering of both forelegs moving the virtual transmitter sticks.  The broad
random-geometry task remains harder, and neither result is simulator transfer.

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
