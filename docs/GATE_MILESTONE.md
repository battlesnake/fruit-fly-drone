# First connectome-driven annular-gate checkpoint

## Status

The repository now contains a complete, headless takeoff-to-annular-gate flight and an
auditable randomized evaluation.  This is a **research checkpoint, not a passed final
milestone**: the best connectome actor succeeds on 467 of 1,024 trials (45.6%), below
the predeclared 90% goal-level threshold.

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
centre. Median radial crossing error is 0.614 m and the 90th percentile is 1.610 m. A
later component audit found 0.420 m mean absolute lateral error and 0.656 m mean absolute
vertical error, with a 0.457 m upward bias. The hidden mass randomization is especially
important: the checkpoint succeeds on 72.8% of heavier-than-nominal vehicles but only
5.3% of lighter ones. A teacher takeover after three seconds recovers 77.1% of trials,
while a takeover after four seconds cannot improve the 40.5% outcome. The next work must
therefore address both early visual aiming and mass-robust vertical feedback.

## Direct accelerometer checkpoints

[`gate-accel-v1`](../artifacts/gate-accel-v1/) tests a minimal inertial interface without
changing the old pilot. Eight additional annotated MaleCNS `wind_gravity` Johnston's-organ
cells receive push/pull body-Z specific force centered on 1g. They connect through real
MaleCNS paths to the two throttle motor pools. The sensor-to-neuron assignment is an
explicit engineering mapping, not a claim about the cells' calibrated physiology.

All 1,122 old neurons and 4,324 old edges are frozen. New-to-old boundaries begin at
zero; old-to-new edges and new biases are held at zero so a constant-1g sensor exactly
reproduces the original controller. Only 109 edges whose presynaptic neuron belongs to
the new acceleration pathway can learn. The resulting graph has 1,138 neurons and 4,469
edges.

On the same 1,024-flight suite, live acceleration improves success from 40.53% to 41.89%
and reduces mean absolute vertical crossing error from 0.656 m to 0.601 m. Light-mass
success rises from 5.31% to 7.35%. Holding the trained controller's sensor at 1g returns
success to exactly 40.53%, establishing that the change depends on time-varying inertial
input. Swapping sensor traces by mass rank yields 41.80%, however, so this small benefit
looks like generic damping rather than learned vehicle-mass identification. It is useful
causal progress, but still far below the 90% goal.

[`gate-accel-v2`](../artifacts/gate-accel-v2/) keeps the same graph, restores every old
bias and time constant, and searches only those 109 acceleration-path edge magnitudes.
The bounded derivative-free search used eight generations of 16 antithetic candidates.
Every candidate was judged by complete flights on common samples exactly balanced across
mass range, gate side, and obliquity sign. The deployed checkpoint adds no parameter or
state outside the connectome.

On the original locked 1,024-flight protocol, v2 succeeds on **467 flights (45.61%)**.
The same weights score 40.53% with their accelerometer held at 1g and 0% with the first
FPV frame frozen. Compared with v1, mean absolute lateral crossing error falls from 0.397
m to 0.329 m, ring collisions fall from 23.24% to 19.82%, light-mass success rises from
7.35% to 8.57%, and heavy-mass success rises from 73.60% to 79.59%. Vertical error rises
slightly from 0.601 m to 0.623 m, so the gain is not an across-the-board improvement.

A separate exactly balanced 1,024-flight audit measured 43.85% live success. Disabling
the above-1g channel reduced that to 41.70%; disabling the below-1g channel reduced it to
40.92%; constant 1g produced 38.96%. A 200 ms intervention from matched recurrent and
leg state also gives the intended signed response: above 1g lowers throttle and below 1g
raises it. However, mass-rank-swapped sensor traces still score essentially the same as
live traces (45.61% on the locked suite). The circuit has learned useful bidirectional
feedback, but the evidence still does not support mass identification.

A two-parameter bias/gain calibration was also rejected. It could balance light and
heavy outcomes, but reduced fresh 1,024-flight completion from 38.09% to 35.25%, and its
constant-1g control retained nearly all of the effect. This is why v2 changes pathway
edges only and keeps the original throttle biases.

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

Re-run the best direct-accelerometer checkpoint and its constant-1g and mass-rank-swapped-trace
controls with:

```bash
scripts/run_gate_training.sh \
  --graph artifacts/gate-accel-v2/connectome.npz \
  --checkpoint artifacts/gate-accel-v2/controller.pt \
  --output-dir runs/gate/accel-recheck --evaluate-only \
  --evaluation-stage paired --evaluation-episodes 1024 \
  --evaluation-seconds 12
```

Re-run the bounded 109-edge search under AIRA with:

```bash
scripts/run_gate_acceleration_es.sh \
  --output-dir runs/gate/acceleration-es-recheck
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
