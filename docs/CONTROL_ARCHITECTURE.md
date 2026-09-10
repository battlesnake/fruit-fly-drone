# Control-theory decomposition for the native fly pilot

This note preserves two complementary development routes:

1. use a conventional multicopter stack as a decomposition and training oracle, while
   keeping every deployed estimator, transition and controller state inside MaleCNS;
2. use a progressively withdrawn teacher to place the aircraft in recoverable states and
   transfer control to the native graph earlier and earlier.

These are training and analysis strategies. Neither permits an external controller,
decoded neural state, recurrent model, mode bit, timer, mass value or history buffer in
the deployed actor.

## The boundary: the fly is an acro pilot, not the flight controller firmware

PX4's multicopter stack is a cascade: estimation feeds mission/mode setpoint generation,
position and velocity control produce an acceleration demand, that demand becomes an
attitude and thrust setpoint, attitude control produces body-rate setpoints, rate PID
produces torque, and control allocation maps torque and thrust to motors. The outer loops
are bypassed in modes that do not need them. See the official
[PX4 controller diagrams](https://docs.px4.io/main/en/flight_stack/controller_diagrams),
[controller module reference](https://docs.px4.io/main/en/modules/modules_controller), and
[EKF2 description](https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf).

Our interface deliberately cuts through that cascade at the acro sticks:

```text
FPV + estimated roll/pitch + body acceleration
                    |
          fixed sensory-neuron mappings
                    |
        recurrent MaleCNS-derived graph
                    |
       T1 foreleg motor-neuron pools
                    |
       physical forelegs -> Mode-2 sticks
                    |
      roll/pitch/yaw rate demands + throttle
                    |
    aircraft rate PID -> allocation -> motors -> motion
                    |
               sensors and FPV
```

In PX4 Acro, roll, pitch and yaw sticks command angular rates, while throttle passes to
control allocation; centered sticks stop rotation but do not level or position-hold the
vehicle. That matches this project's intended boundary. The aircraft's gyro-based rate
PID and mixer are aircraft machinery, just as the fly's muscles do not individually
commutate a brushless motor. They should not be duplicated inside the connectome.
[PX4 Acro mode](https://docs.px4.io/main/en/flight_modes_mc/acro)

The present actor receives estimated roll and pitch as signed push/pull values, not raw
gyro samples. Supplying those angles means that attitude estimation is already outside
the fly. That is acceptable under the declared observation contract, but we should not
claim that the deployed graph implements a complete inertial estimator.

Arm/disarm also remains with the episode supervisor and, eventually, an independent
safety pilot. The fly owns the four flight axes only.

## What a traditional controller does during this task

| Traditional responsibility | Information normally used | Output | Native-pilot interpretation |
| --- | --- | --- | --- |
| Takeoff/land state and constraints | Armed/landed state, local altitude and position, attitude, angular rate | A bounded climb trajectory and takeoff ramp | Learn the sensorimotor transition from grounded scene to climb; do not provide a phase bit. |
| Mission/waypoint selection | Mission list, vehicle position, acceptance radius | Current position/velocity/yaw setpoints | The world's role-colour change makes the current target visually observable, so most of this job is intentionally externalized into the scene. |
| State estimation | IMU plus height/position sources such as barometer, range, GNSS, optical flow or vision | Attitude, velocity, position, biases and uncertainty | Learn only task-sufficient latent quantities from legal sensors; do not reproduce a full navigation EKF. |
| Position control | Position error | Velocity demand | Gate bearing/elevation and height-marker displacement can directly play this role in an egocentric visual servo. |
| Velocity control | Estimated velocity, acceleration feedforward and limits | Acceleration demand | Recurrent optic-flow and accelerometer pathways should supply damping and time-to-contact. |
| Attitude control | Attitude setpoint and estimated attitude | Body-rate demand | The fly must perform this outer attitude loop because it emits acro rate sticks. |
| Angular-rate control | Rate demand and filtered gyro/angular acceleration | Torque demand | Keep this in the aircraft's conventional flight controller. |
| Control allocation | Torque/thrust and frame geometry | Individual motors | Keep this in the aircraft; mass and frame dimensions do not become fly inputs. |

PX4 Takeoff requires angular velocity, attitude, local altitude and local position, then
climbs to a configured altitude and holds. Position mode turns position error into
braking and hold behavior. Its position controller is a position P loop followed by a
velocity PID whose output is split into a thrust direction and magnitude.
[PX4 Takeoff](https://docs.px4.io/main/en/flight_modes_mc/takeoff),
[PX4 Position mode](https://docs.px4.io/main/en/flight_modes_mc/position)

That is a useful list of *responsibilities*, but not evidence that a named fly region is
literally an EKF, PID or PX4 mode manager.

## The hover result is regulation, not yet good damping

The hover milestone showed that the loop was causal and vision-dependent: it lifted on
every evaluation and followed a changed physical height marker. Its control quality was
deliberately accepted at a beginner-like level. Mean final-ten-second altitude RMSE was
0.188 m, the 95th percentile across episodes was 0.324 m, and the goal criterion allowed
up to 0.30 m RMSE. Describing the acceptance region informally as a roughly 60 cm-tall
box is fair, although RMSE is not a hard peak-to-peak bound. The stricter 0.10 m research
criterion failed.

The v1 hover graph had no accelerometer input. It corrected height from a visual band and
received roll/pitch feedback. The raster was only 32 by 32 over a 90-degree field of view.
In the exact ray generator, the two central rows are 3.69 degrees apart, corresponding to
0.3226 m on the wall at the initial 5 m range. The physical band's Gaussian scale was
0.18 m—0.56 image row, or about 1.31 rows at full width half maximum. Coarse and delayed
visual correction can therefore plausibly produce the observed throttle hunting. We must
measure the dominant period, overshoot and settling time before calling the mechanism
proven.

This reframes the next control problem: preserve the demonstrated visual height reference,
then add a native fast vertical-motion estimate and damping path without adding mass or
external history.

### Visual-only is reasonable; the present visual bottleneck may not be

A remote human acro pilot also receives no vehicle IMU stream. The aircraft gyro closes
the inner rate loop while the pilot estimates orientation, motion and closure from FPV.
Visual-only piloting is therefore the right conceptual baseline. The mismatch is visual
capacity, not the absence of an external EKF output.

The official MaleCNS annotations contain 103,268 traced neurons in the visual-sensory,
optic-lobe-intrinsic, visual-projection and visual-centrifugal superclasses—about 63% of
all 165,122 traced neurons. The current gate scaffold retains only 868 of those, about
0.84%, and supplies pixels to 192 selected L1 cells; hover v1 supplied only 48. The 868
visual-class cells are downstream processing capacity, not independent measurements. Its
missing and unbalanced motion and colour pathways are detailed below.

The annular gate is similarly undersampled. Across its 4.2--5.2 m starting range, its
1.52 m outer diameter spans only about 4.5--5.6 pixels at 32 by 32, while each 0.14 m rim
is about 0.42--0.52 pixel thick. Its 0.53 m collision-cleared radius is only 1.6--2.0
central pixel rows. Bilinear retinal sampling retains subpixel intensity information, but
this is still a credible observation bottleneck rather than a hard pixel-quantization
limit.

The mass-free reserve teacher currently consumes exact continuous gate bearing and
elevation derived from pose. Those quantities are physically represented by the rendered
gate, so the teacher is observation-compatible in kind, but not necessarily in precision.
Its success does not prove that the 32 by 32/192-L1 interface can recover its targets
accurately enough. That assumption now needs a direct observability test.

## A full EKF is the wrong first target

PX4's EKF2 estimates a navigation quaternion, three-dimensional velocity and position,
gyro and accelerometer biases, magnetic fields and bias, wind, terrain altitude, and
uncertainty. It also handles measurement delay. Our one-gate task does not need most of
those states.

An EKF is not merely a dense recurrent network. It is a structured recursive estimator
with a process model, measurement models and covariance propagation. Dense recurrence
may be useful for cross-modal fusion, but density alone says nothing about observability,
stability or whether downstream motor circuits use the representation.

The first native estimator should retain only sufficient control variables:

- slow visual height or gate-elevation error;
- vertical velocity or a visual time derivative;
- a short-timescale acceleration transient;
- a slowly adapting neutral/hover-throttle value;
- gate bearing, elevation and time-to-contact;
- a behavior context such as grounded, climbing, approaching or just passed, inferred
  from sensory events rather than supplied as a discrete state.

Body-Z acceleration cannot by itself reveal absolute height or constant vertical
velocity. The promising arrangement is complementary: acceleration provides fast
damping, visual marker/gate motion corrects drift, and recurrent state joins them.

### The most promising PX4-derived idea: estimate hover thrust, not mass

PX4 contains a separate multicopter hover-thrust estimator. Its published interface
reports a scalar hover-thrust estimate and variance plus acceleration innovation
statistics; the position controller uses the estimate as the base thrust for zero
vertical acceleration. This is the control-relevant quantity that mass, battery voltage,
propeller efficiency and payload jointly change.
[PX4 hover-thrust estimator interface](https://docs.px4.io/main/en/msg_docs/HoverThrustEstimate)

For the connectome, the analogous native computation is:

1. retain an efference copy of recent throttle activity through real recurrent and
   descending/ascending edges;
2. compare it with body-Z specific-force transients;
3. adapt a slow internal neutral-throttle state;
4. use visual height motion to correct drift and provide velocity damping;
5. compensate collective for roll/pitch tilt using the already legal attitude input.

The scalar state is never decoded and fed back at deployment. Training-only probes may
ask whether candidate populations contain it, and a conventional hover-thrust estimator
may provide an auxiliary target. Native downstream connections must ultimately turn the
representation into foreleg motor activity.

This is substantially more promising than feeding exact mass to the actor. Previous
experiments showed that accelerometer histories contain vehicle-load information and that
native recurrence can retain a launch-derived ordering, but the tested short-path
readouts and optimizers did not turn it into robust light/heavy throttle control. A
control-relevant scalar target and module-preserving routes give that result a more useful
form.

## Candidate anatomical division of labour

The assignments below are hypotheses used to select populations and training losses. All
runtime signals between them must use actual MaleCNS edges.

| Candidate circuitry | Engineering responsibility | Selection rule and caution |
| --- | --- | --- |
| R1-R6/L1/L2, T4/T5 and lobula-plate motion pathways | Luminance, image motion, horizon/ground flow, gate expansion and time-to-contact | Preserve ON and OFF pathways, all four motion directions, retinotopy and bilateral homologues. T4/T5 are the first direction-selective cells in the fly ON/OFF motion pathways, but drone use remains an engineering adaptation. [Motion-circuit evidence](https://www.nature.com/articles/s41586-024-07939-3) |
| R7/R8 and downstream colour-opponent pathways | Decode the fixed current/next gate role colours | The current monochrome one-gate renderer cannot test this. RGB-to-photoreceptor mapping is necessarily engineered because ordinary RGB lacks fly UV bands and the raw MaleCNS sensory rows lack usable hex coordinates. |
| Recurrent visual, central-brain, DN and AN loops | Fuse slow visual error, fast acceleration, action/efference state and context | Select by cross-modal convergence, complete strongly connected motifs and causal probe results—not soma position or path length alone. |
| Central complex EPG/PEN/PEG/Delta7, FC2/PFL and related populations | Candidate heading/goal comparison and persistent steering context | There is good evidence for heading-goal comparison through FC2/EPG/Delta7/PFL3 and onward to lateral accessory lobe/descending pathways, mainly in walking assays. It is a navigation hypothesis, not an altitude controller. [Heading-goal evidence](https://www.nature.com/articles/s41586-023-07006-3) |
| Visual and central descending neurons | Convert sensory/goal population activity into steering and behavioral commands | Preserve bilateral pairs and DN-DN interactions. DNs are the anatomical bottleneck from brain to VNC; ANs return processed sensory and motor-state information. [DN/AN evidence](https://www.nature.com/articles/s41586-025-08925-z) |
| T1 VNC interneurons, FeCO pathways and foreleg motor neurons | Stick-axis synergies, local feedback and final actuation | Keep the required front-leg embodiment even when flight-related wing DNs are used as internal features. FeCO direction tuning must be established rather than guessed. |

The central complex is most attractive when we introduce multiple visible gates, heading
changes or longer occlusions. For the first annular gate, an egocentric visual servo may
need no allocentric map at all. Role-colour changes already remove the hardest mission
manager question—"which gate is current?"—from internal memory.

## What the current extracted graph can and cannot support

The current direct-accelerometer graph contains 1,138 neurons and 4,469 induced edges.
Joining its body IDs to the official annotations gives:

| MaleCNS superclass | Selected neurons |
| --- | ---: |
| Optic-lobe intrinsic | 736 |
| Visual projection | 130 |
| Central-brain intrinsic | 85 |
| Descending | 81 |
| VNC intrinsic | 26 |
| Foreleg motor | 26 |
| Ascending | 12 |
| Central sensory (`wind_gravity`) | 40 |
| Visual centrifugal | 2 |

It is recurrent—718 nodes are in non-singleton strongly connected components—but it is
not a module-preserved miniature flight brain. The deterministic builder selects one
short route from every chosen sensor to every motor pool and then restores induced edges
among the selected nodes. That objective favors reachability, not functional completeness.

In particular, the graph has 192 L1 cells and 47 T4 cells distributed 18/10/2/17 across
the four directional subtypes, but no L2, T5, R7 or R8 cells. It contains no EPG, PEN,
PEG, Delta7, FC2, PFL, PFN or hDelta cells, although all of those families are present in
the raw MaleCNS release. Therefore:

- it cannot yet support an anatomically faithful central-complex navigation claim;
- its motion vision is strongly unbalanced and lacks the OFF pathway;
- its gate images are monochrome, so the proposed role-colour language is not yet an
  actor observation;
- adding a few shortest paths to named cell classes would not repair the problem.

The next extractor should preserve declared modules: complete bilateral type cohorts or
retinotopic samples, ON/OFF and direction balance, SCC closure, cross-modal convergence,
and real downstream routes through DNs/VNC to the existing foreleg pools. Exact included
and excluded body IDs must remain manifest data. The official release provides the full
connectivity graph and curated type/class annotations needed for this audit.
[MaleCNS download documentation](https://male-cns.janelia.org/download/)

## Two experimental tracks to retain

### Track A — control decomposition inside native anatomy

1. **Measure the existing hover loop.** Run the frozen 40-episode milestone again with
   per-episode peak-to-peak height, vertical speed, throttle spectrum, dominant oscillation
   period, overshoot and settling time. Add paired vertical-velocity impulses. This tests
   the throttle-bounce recollection rather than inferring it from RMSE.
2. **Audit visual observability.** Factor renderer resolution, apparent gate size and
   retinal sampling density. Measure whether legal retinal samples and native activations
   can recover height error, gate bearing/elevation, image velocity and time-to-contact
   to the accuracy required for a clean crossing.
3. **Audit candidate modules before rebuilding.** On the full MaleCNS annotations and
   graph, identify balanced visual-motion, colour, recurrent cross-modal, DN/AN/VNC and
   foreleg populations. Report SCCs, loop lengths, bottlenecks and edge closure.
4. **Use diagnostic neural probes.** On conventional-teacher trajectories, fit
   training-only readouts for visual height error, vertical velocity, hover thrust, gate
   bearing/elevation, time-to-contact and behavior phase. A successful probe establishes
   representation only; lesion/edge-intervention tests must show native downstream use.
5. **Train vertical damping first.** Keep teacher steering while the native graph learns
   the four-step hover-thrust computation above, then test zero-assistance hover under
   mass, thrust, delay and vertical-impulse variation.
6. **Train visual gate guidance.** With a stable native vertical loop, transfer
   gate-relative roll/pitch/yaw guidance, then train coupled four-axis corrections.
7. **Add colour-role courses.** Only after RGB/spectral opponent channels and balanced
   motion pathways exist, progress from one gate to two or three visible role colours.

### Track B — progressively withdrawn teacher assistance

The verified mass-free reserve controller can place the plant on a recoverable approach.
Use it as training scaffolding while the fly's native recurrent state runs continuously
from reset:

1. collect full teacher trajectories and train native action/motor targets at every step;
2. let the teacher drive the early plant, then hand each axis to the native graph at
   randomized recoverable states;
3. move the handoff earlier and widen disturbances as competence passes fixed gates;
4. overlap vertical and steering transfer, because the axis-takeover experiments show
   that the final gate problem is coupled;
5. finish with the native forelegs controlling all four axes from the ground, with zero
   assistance and no handoff signal presented to the actor.

Randomizing handoff time and state prevents an episode clock from being the solution.
Teacher actions, privileged state and decoded probe values remain training-only. Final
acceptance always resets and runs the unassisted actor exactly as deployment will.

## Immediate stop/go experiments

### Visual observability and scale audit

Before another optimizer run, run a frozen 2-by-2 information audit over identical
physical poses and short trajectories:

| Representation | Current raster | Higher-resolution raster |
| --- | --- | --- |
| Raw image | 32 by 32 pixels | 128 by 128 pixels |
| Existing retinal interface | 32 by 32 sampled at the unchanged 192-L1 grid | 128 by 128 sampled at the identical normalized L1 grid |

Include smooth vertical sweeps through several raster phases, matched 5 and 10 cm vertical
displacements, and opposite vertical-motion histories that finish at the same pose. Cover
both gate sides and obliquity signs. Split development and holdout by complete trajectory
or geometry, never by neighboring frames.

Measure raw and retinal sensitivity, then fit the same modest offline-only probes for
height/gate elevation and vertical-motion direction. Two-frame probes at a fixed interval
are allowed for this information diagnostic; their outputs never enter the actor. Read
the result as follows:

- raw 32 succeeds but retinal 32 fails: retinal coverage/transduction is implicated;
- both 32 representations fail and both 128 representations succeed: raster resolution
  is implicated;
- raw 128 succeeds but retinal 128 fails: more rendered pixels alone are insufficient;
- the existing retinal representation succeeds: prioritize neural dynamics and control.

Keep FOV, geometry, paint, softness, contrast and retinal coordinates fixed. A frozen
controller's failure at 128 would be inconclusive because it changes the learned input
distribution, so this first audit measures information rather than flight success.

If resolution is implicated, a later training positive control can use a nearer/larger
gate with a rim at least two pixels thick, or increase resolution with physical geometry
held fixed. The final annular geometry remains a held-out target, not something silently
made easier in its acceptance test.

### Frozen vertical-disturbance audit

Before any new training, evaluate the successful hover checkpoint on at least 256 fresh,
balanced vehicle cases. Apply one hidden airborne vertical-velocity impulse in each
direction, paired with an unperturbed continuation. Report recovery time to the original
visual band, overshoot, height peak-to-peak, dominant throttle/height frequency, ground
contact and stick saturation separately by mass half. A conventional analytical
controller is a recoverability control, not an actor input.

This tells us whether the existing graph already contains useful disturbance rejection
that should be protected, or only the loose visual regulation seen in the original test.

### Module-preserving extraction audit

Do not train the rebuilt graph first. Produce three candidate manifests—vertical-control,
egocentric one-gate, and central-complex navigation—and compare node/edge count, SCC
closure, bilateral/type coverage, sensory-to-foreleg reachability and GPU memory estimate.
The central-complex candidate should be retained for later even if the simpler egocentric
graph is chosen for the first gate.

## Decisions retained

- Do not feed mass or frame size to the fly.
- Treat accelerometer input as a testable aid, not a prerequisite: preserve a visual-only
  baseline and ablate inertial channels explicitly.
- Do not feed EKF state, height, vertical velocity, gate pose or mode to the deployed
  actor; they are allowed only as teacher, critic, probe and evaluation labels.
- Do not add an external GRU, integrator, PID, phase machine, action history or clock.
- Keep the conventional acro angular-rate loop and motor allocation in the aircraft.
- Keep arm/disarm under the independent supervisor.
- Preserve the foreleg/stick embodiment as the only four-axis action path.
- Use anatomical regions to constrain selection, connectivity and auxiliary objectives,
  not to make unsupported one-region/one-PX4-module claims.
