# First connectome-driven hover milestone

## Result

The v1 controller lifts a differentiable quadcopter from the floor and maintains a
mostly level visual-height hover.  In the locked 40-episode, 20-second evaluation it
passed 37/40 episodes (92.5%): every episode lifted and acquired the height band within
six seconds, none touched the floor again, final-ten-second altitude RMSE averaged
0.188 m, and roll/pitch tilt RMS averaged 3.66 degrees.

This is a **differentiable surrogate milestone**, not yet a MuJoCo/FlyGym transfer
result.  The quad is a six-degree-of-freedom rigid body with a lumped thrust-and-torque
actuator, an acro body-rate controller, actuator lag, thrust limits, drag, and ground
contact.  Available torque is coupled to collective, but there are no explicit rotors or
mixer.  Each abstract foreleg is a two-axis gimbal with joint state and forward kinematics
to a constrained stick-cap endpoint; this is not articulated fly morphology, identified
muscle mechanics, or FlyGym contact.  The next fidelity gate is frozen-controller replay
against an independently implemented MuJoCo plant and then bilateral FlyGym leg mechanics.

## Information and action boundary

The deployed actor gets only:

- a 32 by 32 monochrome FPV image containing world geometry;
- drone roll and pitch, encoded into fixed push/pull sensory populations;
- its own persistent MaleCNS neuron states.

It does not get height, velocity, target height, elapsed episode time, a previous-action
buffer, a gate/marker detector, or a takeoff/hover phase bit.  There is no external
learned adapter, recurrent module, planner, or state machine.

The controller's four outputs are differences between eight fixed, named front-leg motor
pools.  They drive two two-axis foreleg endpoints coupled to spring-loaded Mode-2 sticks.
The flight model reads the **measured stick positions**, never a parallel neural action:

| RC axis | Side | Positive MaleCNS pool | Negative MaleCNS pool |
| --- | --- | --- | --- |
| Roll | Right | Tergopleural/Pleural promotor MN | Pleural remotor/abductor MN |
| Pitch | Right | Ti extensor MN | Ti flexor MN |
| Yaw | Left | Tergopleural/Pleural promotor MN | Pleural remotor/abductor MN |
| Throttle | Left | Ti extensor MN | Ti flexor MN |

The frozen controller uses a 240 degree/second maximum roll/pitch acro profile.  The
normal inner rate controller is part of the aircraft; it is not an external leveling
controller.

## MaleCNS scaffold

[`connectome-manifest.json`](../artifacts/hover-v1/connectome-manifest.json) records the
raw-file hashes, selection parameters, neuron IDs, and transmitter-sign assumptions.
The extraction is deterministic:

- 165,122 traced MaleCNS segments were eligible;
- paths use real directed edges with at least 20 counted synapses and at most eight hops;
- all retained edges among selected nodes with at least 10 synapses are restored;
- 48 retinotopic L1 cells (24 per eye) receive fixed samples from the FPV image;
- 32 `wind_gravity` mechanosensory cells receive fixed roll/pitch push/pull values;
- 26 T1 front-leg motor neurons make up the eight antagonist output pools;
- the resulting graph has 498 neurons and 2,013 anatomical edges, with every selected
  sensory cell reaching every output pool in three to seven strong-edge hops.

The release's 6,098 `ol_sensory` photoreceptor rows do not contain `assignedOlHex1/2`, so
v1 injects pixels into hex-indexed L1 cells.  This is an explicit engineered visual
interface, not an inferred biological retina.  Synapse counts initialize magnitudes;
nonnegative magnitudes, neuron biases, and time constants are trained inside the graph.
The anatomical mask and transmitter-derived sign hypothesis remain fixed.

## Visual height reference

The target is a physical horizontal bright band on a wall five metres in front of the
start.  Analytic pinhole rays generate the FPV pixels from the drone pose and world
geometry.  Target height is never converted into a screen coordinate or supplied as a
scalar to the actor.  A lightly patterned grey floor supplies weak optic flow.

The locked evaluation randomizes target height from 0.8 to 1.2 m, initial attitude, and
mass from 90% to 110% of nominal.  Maximum thrust is fixed from nominal mass, so this also
varies effective thrust-to-weight ratio rather than cancelling mass out.  A second test
forks identical physical, stick, and recurrent neural state halfway through an unbroken
20-second episode.  It changes only the band height in one branch and leaves the other as
a paired no-step counterfactual.

## Acceptance and evidence

The goal-level per-episode test requires lift-off, entry into the target-height plus or
minus 0.15 m band within six seconds, final-ten-second altitude RMSE no greater than
0.30 m, roll/pitch RMS no greater than 5 degrees, 95th-percentile tilt no greater than
10 degrees, no recontact, and no sustained stick saturation.  At least 90% of episodes
must pass.  In the paired band-step test, at least 90% of band-induced height differences
must have the correct direction and their mean response ratio must be between 0.5 and 1.5.

| Metric | Connectome actor | Frozen-initial-image ablation |
| --- | ---: | ---: |
| Goal success | **92.5%** | 0% |
| Lift-off | **100%** | 72.5% |
| Height band acquired by 6 s | **100%** | 47.5% |
| Final 10 s altitude RMSE | **0.188 m** | 3.17 m |
| Final 10 s tilt RMS | **3.66 degrees** | 11.27 degrees |
| Ground recontact | **0%** | 7.5% |
| Visual target tracking slope | 1.059 | not meaningful (unstable) |
| Paired mid-flight band-step direction | **100%** | not tested |
| Paired mid-flight response ratio | 1.018 | not tested |

The controller still misses the stricter research target of 0.10 m altitude RMSE and
0.75 m horizontal drift: mean maximum drift is 1.98 m because no world-position target is
visible. The 0.30 m goal-level RMSE limit also permits visibly loose, beginner-like
throttle hunting—informally, a roughly 60 cm-tall tolerance box, although RMSE is not a
hard peak-to-peak bound. This milestone establishes lift, causal visual altitude
regulation, mostly level hover, and the complete four-axis foreleg/stick action path; it
does not establish well-damped altitude hold, station keeping, high-rate acro performance,
full-brain fidelity, or simulator transfer. The proposed disturbance and damping audit is
recorded in [`CONTROL_ARCHITECTURE.md`](CONTROL_ARCHITECTURE.md).

The committed [`report.json`](../artifacts/hover-v1/report.json) binds the checkpoint and
graph SHA-256 values, evaluation seeds, image resolution, physical config, and training
lineage; it is the machine-readable source for these numbers.  Re-run the frozen
acceptance suite under WSL2 with:

```bash
scripts/run_hover_training.sh \
  --graph artifacts/hover-v1/connectome.npz \
  --checkpoint artifacts/hover-v1/controller.pt \
  --output-dir runs/hover/recheck \
  --evaluate-only --evaluation-episodes 40 --evaluation-seconds 20
```

Rebuild the derived graph from the official raw tables with:

```bash
aira confine -- .venv/bin/python scripts/build_hover_connectome.py
```

The build writes ignored working files under `data/derived/`; the compact graph and its
provenance manifest used for this result are frozen under `artifacts/hover-v1/`.
