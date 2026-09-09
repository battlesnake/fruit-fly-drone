# First connectome-driven annular-gate checkpoint

## Status

The repository now contains a complete, headless takeoff-to-annular-gate flight and an
auditable randomized evaluation.  This is a **research checkpoint, not a passed final
milestone**: the connectome actor succeeds on 415 of 1,024 trials (40.5%), below the
predeclared 90% goal-level threshold.

The result nevertheless establishes the full behavior once and measures where it fails.
The committed [`showcase.mp4`](../artifacts/gate-v1/showcase.mp4) records a successful
5.86-second traversal.  Its left pane is the actor's 32 by 32 monochrome FPV input.  The
right pane is a schematic fly: the orange front leg moves the roll/pitch stick and the
blue front leg moves the yaw/throttle stick.  A green FPV border marks the swept-sphere
gate crossing; it is a presentation indicator and is not supplied to the actor.

## Task and acceptance boundary

Every evaluation flight starts on the grey floor against a black background.  The bright
annulus has a 0.62 m inner radius and 0.76 m outer radius.  Its centre is 4.6 m ahead,
0.8 m to either side, and 1.1 m high.  The gate plane is always 20 degrees oblique to the
launch-to-centre displacement, with independently randomized sign.  Vehicle mass is
uniformly randomized from 92% to 108% of nominal.

The actor must lift, place the entire 0.09 m-radius drone sphere inside the aperture at a
swept plane crossing, travel at least 0.4 m beyond the plane, remain airborne for at least
one further second, stay below 40 degrees roll/pitch tilt, and avoid sustained stick
saturation.  A plane crossing outside the clean radius is classified as a ring collision
or miss rather than being silently retried.

The deployed actor receives only:

- the live 32 by 32 monochrome FPV image;
- roll and pitch angles through fixed push/pull sensory populations;
- its own recurrent MaleCNS neuron state.

It does not receive gate geometry, position, velocity, time, a gate detector, a pass bit,
or an external recurrent state.  Eight fixed front-leg motor-neuron pools form four
antagonist differences.  Those differences accelerate two spring-loaded, two-axis leg
endpoints, and only the measured stick positions enter the quadcopter's normal acro rate
controller.  Arm/disarm is outside this milestone.

## Connectome and training

The checkpoint expands the original hover graph from 48 to 192 hex-indexed L1 cells while
preserving all shared learned parameters by MaleCNS body ID and anatomical edge.  The
frozen artifact has 1,122 neurons and 4,324 directed anatomical edges.  Synapse-count
magnitudes initialize trainable nonnegative strengths; transmitter-derived signs and the
edge mask remain fixed.  Neuron biases and time constants are learned inside the same
recurrent graph.

Training used a privileged teacher only to label actions.  The final teacher is deliberately
observation-compatible: its commands depend on gate bearing/elevation (visible in FPV) and
roll/pitch, not hidden translational velocity or angular rate.  It also supplies a short
post-crossing continuation target that the recurrent connectome must internalize.  Exact
left/right paired trajectories and DAgger recovery experiments addressed an early
one-sided-steering failure.  No teacher, planner, memory, or learned decoder runs during
evaluation.

## Locked evidence

The committed [`report.json`](../artifacts/gate-v1/report.json) is a 1,024-episode CUDA
evaluation on the RTX 5080 under WSL2.  It binds the graph and controller hashes and uses
the same initial conditions for the actor, teacher, and frozen-image control.

| Metric | Connectome actor | Frozen-first-frame | Training teacher |
| --- | ---: | ---: | ---: |
| Full success | **40.5%** | **0%** | 100% |
| Lift-off | **100%** | 78.8% | 100% |
| Negative-offset success | 38.5% | 0% | 100% |
| Positive-offset success | 42.8% | 0% | 100% |
| Ring collision | 20.7% | 7.3% | 0% |
| Aperture miss | 38.8% | 92.7% | 0% |
| Mean maximum tilt | 17.7 degrees | 16.7 degrees | 22.4 degrees |

Every nominal episode reaches the gate plane; unsuccessful flights cross too far from its
centre.  Median radial crossing error is 0.614 m and the 90th percentile is 1.610 m.  A
teacher takeover after three seconds recovers 77.1% of trials, while a takeover after four
seconds cannot improve the 40.5% outcome.  The remaining problem is therefore early visual
aiming and closed-loop correction, not lift authority or failure to reach the gate.

## Reproduction

Re-run the frozen checkpoint evaluation with:

```bash
scripts/run_gate_training.sh \
  --graph artifacts/gate-v1/connectome.npz \
  --checkpoint artifacts/gate-v1/controller.pt \
  --output-dir runs/gate/recheck --evaluate-only \
  --evaluation-stage paired --evaluation-episodes 1024 \
  --evaluation-seconds 12 --teacher-takeover-audit-seconds 3 4
```

Record another seeded successful flight and foreleg/stick visualization with:

```bash
aira confine -- .venv/bin/python scripts/record_gate_flight.py \
  --graph artifacts/gate-v1/connectome.npz \
  --checkpoint artifacts/gate-v1/controller.pt \
  --output-dir runs/gate/recording --episodes 128 --seed 10031
```

The next acceptance target is at least 90% on this paired task, followed by the existing
broad suite that randomizes distance, lateral offset, height, and 10–30 degree obliquity.
Only after that should the same frozen actor move to an independent physics engine and a
multi-gate colour-role course.
