# Plan: a connectome-constrained acro pilot

## End goal and current priorities — clarified 2026-09-12

Train the fly-derived recurrent nervous system to pilot a quadcopter around a complete
drone racecourse in acro mode. FPV video is the final sensory input; activity in identified
front-leg motor neurons moves the simulated front legs, their movement deflects the two
transmitter sticks, and measured stick positions supply roll, pitch, yaw and throttle.
The current roll/pitch observation is temporary support for learning and should eventually
be removed. The aircraft can still use its own IMU for its ordinary inner angular-rate
controller without exposing that telemetry to the fly.

For this project, representative racing means courses combining straights, varying gate
spacing and orientation, left/right turns, reversals, climbs and descents. The fly must
learn coordinated roll/pitch/yaw/throttle control, anticipate the following gate, brake
and accelerate, and retain useful motion and target information when a gate leaves the
camera view. Begin with one complete lap, then test repeated laps with continuous neural,
leg, stick and aircraft state. These are proposed course families, not a claim to match a
particular racing standard or a requirement for competitive lap times at the first proof.

The simplified visual presentation is part of the intended task. Keep surfaces visually
distinct, with faint texture fixed in world coordinates to provide motion cues, and
colour gates by relative order: current, next and later. Passed gates become dark and the
roles advance after a valid crossing. Photorealistic scenery and ordinary race signage
are optional extensions. Course role assignment belongs to the environment; the actor
must read it from pixels. Texture seed/phase should vary independently of course layout
so texture can support motion estimation without identifying a memorized route.

The recent variable five-gate result (17/128 complete flights versus 13/128 for its source)
is an early progress measure. It covers mild offsets along a mostly forward course and
does not establish full-course racing or video-only control. The active progression is:

1. Improve uninterrupted completion on the present course distribution and preserve
   takeoff/hover capability as separate building blocks.
2. Add short courses with real heading changes: a single bend, alternating bends and a
   reversal, alongside independent height and spacing changes. Test roll/yaw coordination
   and throttle control, rather than assuming a roll-only adaptation will cover them.
3. Combine these elements into held-out complete laps, then join takeoff to the lap and
   evaluate repeated laps without controller resets or teacher intervention.
4. Introduce matched roll/pitch-input ablations as competence grows, then train the same
   native network for video-only flight and evaluate the complete lap again.

Select learning methods by complete-flight outcomes. Short teacher/recovery lessons can
seed useful behaviour; restricted anatomical-parameter evolution search or recurrent RL
can optimize the deployed trajectory. A teacher used for a new course must first fly it
through the actual foreleg/stick plant at the relevant speed. The current staged teacher
is much slower than successful native flights, so lower imitation error alone is not
progress toward racing.

During longer training and evaluation runs, periodically reconsider the design using
flight traces: perception and field-of-view limits, neural timing and memory, motor
authority, teacher mismatch, curriculum coverage and reward incentives. Record promising
alternatives and a small experiment that can distinguish them. Retain failed approaches
as evidence without letting exact replay or extensive certification displace the next
behavioural demonstration. Compare new controllers on matched unseen layouts using full
lap completion, cumulative gates, collisions and completion time; a few extra successes
on a small bank are preliminary evidence, not established reliability.

This clarification governs the earlier architectural options and research milestones
below. Current implementation status is recorded in
[the execution plan](HOVER_TO_GATE_PLAN.md) and its linked experiment records.
The next multi-gate design is recorded in
[anticipation, direction cues and teacher curriculum](RACING_ANTICIPATION.md), including
implemented all-gate event checks and the planned hint-fading experiment.

## 1. Research claim

The target is a **connectome-constrained artificial pilot**. MaleCNS determines the
allowed recurrent edges and supplies count-derived initial connection magnitudes. We
learn the missing dynamics and interfaces from tasks.

We should not call the result a whole-brain emulation or say that a fruit fly was trained.
MaleCNS v1.0 provides anatomy, segment-to-segment synapse counts, annotations, and
predicted neurotransmitter identity. It does not provide physiological synaptic strength,
receptor expression for every connection, neuron dynamics, axonal delays, internal state,
plasticity rules, or activity from the imaged animal.

Two useful precedents define the reasonable envelope:

- [Flyvis / Lappalainen et al.](https://www.nature.com/articles/s41586-024-07939-3)
  uses a recurrent, connectome-constrained visual network. Synapse counts set relative
  magnitudes and cell-type-shared parameters are task-optimized for optic flow. Its
  [official PyTorch implementation](https://github.com/TuragaLab/flyvis) is the starting
  architectural reference.
- [Shiu et al.](https://www.nature.com/articles/s41586-024-07763-9) demonstrates a
  whole-central-brain leaky integrate-and-fire model built from connectivity and predicted
  transmitter identity. Its [official code](https://github.com/philshiu/Drosophila_brain_model)
  is a useful second reference, but the new MaleCNS graph is larger and includes the VNC.

## 2. System boundary

```text
rendered FPV frames                         temporary roll/pitch support
         |                                             |
fixed hexagonal resampling and                 fixed normalization and
R1-R8-like channel response                    push-pull population coding
         |                                             |
         +----------- identified MaleCNS sensory neurons -----------+
                                     |
          fixed MaleCNS directed edge mask + count-derived magnitudes
               learned type-shared dynamics and synaptic gains
                                     |
                 descending / flight-motor neuron populations
                                     |
                  identified left/right foreleg motor neurons
                                     |
              fixed motor-to-joint/muscle transduction
                                     |
                simulated forelegs physically move two sticks
                                     |
          measured Mode-2 stick positions: yaw/throttle + roll/pitch
                                     |
           pinned rate/expo mapping -> body-rate controller -> motors
```

The interfaces perform only fixed, stateless transduction: image resampling, channel
response, normalization, push-pull coding, population aggregation, and unit calibration.
There is no external CNN, recurrent module, gate detector, planner, state machine,
action-history buffer, learned input adapter, or learned output head. Every trainable
temporal computation and all task memory live in the simulated MaleCNS neurons and
synapses. A privileged teacher and critic may see simulator state during training, but
they are absent from the deployed controller.

### Action contract

The flight model receives four normalized RC-like values measured from the sticks:

- roll, pitch, yaw in `[-1, 1]`, mapped through one pinned Betaflight rate/expo profile;
- throttle in `[0, 1]`, mapped through one pinned thrust curve and idle policy.

Acro mode commands body angular rates; neutral roll/pitch does not command level
attitude. The normal inner rate loop remains flight-control machinery, much as muscles
and local reflexes are not the brain's high-level task.

The motor interface uses the fly's two front legs as the physical control outputs. Mount
the virtual fly by its thorax above a Mode-2 transmitter and place its fore-tarsi on the
sticks:

- left foreleg lateral/medial motion -> yaw;
- left foreleg fore/aft motion -> throttle;
- right foreleg lateral/medial motion -> roll;
- right foreleg fore/aft motion -> pitch.

Exact coordinate signs are frozen and unit-tested. Identified front-leg motor-neuron
activity drives joint torques or target positions through a fixed, predeclared
motor-to-joint synergy matrix. The simulated leg then moves; the measured stick
deflection at the tarsus is the actual flight-controller input. There is no parallel
four-axis neural decoder.

For the first implementation, use a stable kinematic constraint (the tarsus remains
coupled to the stick cap) and ordinary bilateral joint actuators. Later replace it with
contact mechanics, tendons, and muscles if they improve the scientific question. The
current FlyGym experimental muscle model covers only the left front leg, so mirroring it
to the right before the basic demo would add risk without validating the connectome.

The mapping from fly motor neurons to leg actuators is necessarily engineered because
MaleCNS does not provide a complete calibrated neuron-to-muscle dynamical model. It may
contain scaling and mechanics, but no learned policy or task memory. Leg proprioception
(joint position, force, and contact) should return through fixed mappings to appropriate
sensory/ascending populations; that is biological peripheral feedback, not an external
controller state machine.

Arm/disarm belongs to an independent episode and safety supervisor. A later policy may
emit an **arm request**, but it should never own the sole authority to arm or disarm.

### Observations and timing

The current prototype uses 320×200 FPV at 125° horizontal FOV, a 50 Hz policy and 100 Hz
foreleg/stick/aircraft updates. Roll and pitch enter through fixed sensory coding during
the initial milestones; the final target uses video alone. Accelerometer and other
telemetry experiments are optional diagnostics, not requirements for the racing actor.
Preserve every neuron's membrane/rate state between ticks; that state must infer motion
and retain useful information between frames.
Model timestamping, exposure, transport delay, dropped frames, noise, and actuator lag
before hardware transfer. World pose, velocity, and gate coordinates are
teacher/critic/evaluation information only.

Mass and frame dimensions are vehicle configuration, not live sensor measurements. A
fixed value is already absorbed by the trained dynamics and adds no information. If a
known configuration value is tested across vehicles, it must enter through a declared
input population and any memory or calibration must remain in recurrent neural state;
directly changing motor-pool bias is only a privileged upper-bound experiment. Prefer
measurements that reveal realized dynamics—stick/joint position, accelerometer and gyro,
then modeled rotor-speed or motor-current telemetry—to nominal frame dimensions. A
single frame-size scalar cannot describe inertia, thrust authority, actuator response,
or drag.

## 3. Connectome model

Keep the raw Feather files immutable. Build a versioned derived graph with every filter,
join, threshold, and transformation recorded.

For an edge from neuron `j` to neuron `i`, begin with a differentiable rate model:

```text
tau[type(i)] * dv_i/dt = -v_i + bias[type(i)]
    + sum_j sign(j,i) * gain[type(j),type(i)] * f(count(j,i)) * activation(v_j)
    + sensory_input_i
```

Initial modelling choices:

- The adjacency mask is fixed to MaleCNS edges.
- Preserve raw counts, while testing `count`, `log1p(count)`, and normalized count as
  derived magnitudes. Normalize recurrent operators to prevent exploding dynamics.
- Learn positive, cell-type-pair-shared unitary gains, cell-type time constants, biases,
  and thresholds. This is much more identifiable than one free parameter per synapse.
- Treat transmitter-derived signs as hypotheses. Acetylcholine is usually excitatory and
  GABA usually inhibitory, but transmitter identity alone does not determine every
  postsynaptic effect; glutamate and modulators need explicit uncertainty/ablation.
- Train an ensemble of seeds. Similar task scores need not imply the same biological
  dynamics.

Profile the full graph first. If full-graph backpropagation is prohibitive, start with a
documented sensorimotor subgraph selected by annotation and connectivity: forward paths
from visual and mechanosensory inputs, backward paths from descending/flight motor
outputs, strong-edge retention, and explicit cell-type coverage. Record exact body IDs
and selection rules. Expand only after a reduced graph beats its controls.

## 4. Gate language

The proposed role-changing gates are a good way to externalize course order through
vision rather than give the actor a waypoint vector.

The world can remain deliberately austere through the final course task: a textured grey
floor, a dark background, distinct surfaces where present, and role-coloured gates.
Textures remain fixed in world coordinates during a flight. The current renderer uses
green for current, red for next, blue for all later gates and black for passed gates.
A possible bounded lookahead extension would assign colours as follows:

- gate `0` in the visible task horizon has fixed role color `C0` and is always current;
- gates `1..N-1` have fixed role colors `C1..C(N-1)`;
- passed gates and gates beyond the next `N` are black;
- after a valid pass, remaining colored gates shift down one role and the newly exposed
  gate receives `C(N-1)`.

Choose strongly separated role colors and map RGB through a fixed approximation of R1-R8
photoreceptor channels. An ordinary RGB camera cannot reproduce the fly's UV channels,
so this is an engineered retinal interface rather than a claim of biological spectral
vision. Randomize texture placement independently of the course while retaining this
visual language; ordinary unmarked racing visuals are not an acceptance requirement.

A valid pass is a directed crossing of the current gate plane with the whole quad inside
an aperture reduced by collision clearance. Sweep the trajectory between physics steps,
award each gate once, and reject reverse crossings. Merely touching a trigger volume,
skimming the frame, or oscillating across the plane must not score.

## 5. Simulator decision

The first proof should run fully headlessly in one Linux process. Use two synchronized
MuJoCo model instances with a deterministic multi-rate scheduler:

1. a FlyGym model in its native millimetre/gram convention, stepped at the rate needed
   for stable foreleg and stick mechanics;
2. a simple SI-unit quadcopter model, stepped at its own physics rate, with the FPV camera,
   gates, rotor thrust and motor lag.

The two models share simulated time but not state. Their only forward coupling is the four
measured stick positions; roll, pitch and acceleration return through the declared sensory
interface. This does not add a controller or task memory outside the connectome. Keeping
the models separate avoids forcing FlyGym's 0.1 ms, millimetre-scale body and a
metre-scale racecourse into one poorly scaled physics scene.

Begin with ordinary headless MuJoCo for correctness. Then batch both plants with
[MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp/) and use its GPU batch
renderer for low-resolution FPV. [FlyGym 2.x](https://neuromechfly.org/) exposes
GPU-parallel simulation and GPU batch rendering directly. Warp arrays can be shared with
PyTorch without copying, so pixels, sensors, neural state, actions and simulator state can
stay on the RTX 5080. Rendering is still sampled at the FPV rate; neither the camera nor
the connectome needs to run at the fly mechanics timestep.

| Component | Decision | Purpose |
| --- | --- | --- |
| [FlyGym 2.x](https://neuromechfly.org/) + MuJoCo/MJWarp | Primary stack | Tethered fly, bilateral forelegs, stick mechanics, quad plant, batched headless physics and low-resolution FPV without an RPC boundary. |
| [PyFlyt](https://github.com/jjshoots/PyFlyt) | Immediate drone fallback and oracle | Ready-made QuadX dynamics, Gymnasium tasks and angular-rate-plus-thrust control if implementing the small MuJoCo quad delays the first closed loop. Expect CPU/PyBullet camera transfer to limit large batches. |
| [Betaflight SITL](https://betaflight.com/docs/development/SITL) | Required fidelity gate | Exercise the intended rate mapping, PID/filter configuration, arming behavior, and firmware timing before hardware. |
| [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | SITL bridge reference | Reuse its Betaflight protocol work, but do not assume its demo bridge is a production training environment. |
| [AirSim](https://microsoft.github.io/AirSim/) | Evaluation and presentation adapter | Familiar, visually capable FPV target. Its `NoDisplay` mode keeps API camera rendering active, but the Unreal process and RPC boundary make it less attractive for high-throughput training. |
| [Godot](https://docs.godotengine.org/en/stable/tutorials/physics/rigid_body.html) | Optional course/presentation tool | Add only if the MuJoCo and AirSim visuals are insufficient; never let it become a second authoritative drone plant. |
| Isaac Lab / Aerial Gym | Defer | Powerful GPU robotics stacks, but large installation/VRAM cost and WSL graphics/version risk do not buy us a shorter first path. |

The quad model need not be photorealistic: the specified black world, grey floor and
colored primitive gates are an excellent match for MJWarp's batched ray renderer. Use
primitive geometry, no shadows and the smallest retinally useful image. An observer
camera for presentation is recorded only in evaluation runs.

PyFlyt's thrust command is not automatically Betaflight throttle. Whichever fast plant is
used, implement and test the RC-rate and throttle mapping as its own fixed module. Compare
step responses, maximum rates, motor saturation, propwash approximation, battery sag and
latency across it, Betaflight SITL, AirSim and eventually the real craft.

AirSim remains useful, especially because it is already familiar. On this machine the
most robust arrangement is a Windows-hosted AirSim process with the WSL2 actor as an RPC
client. `ViewMode: NoDisplay` disables the main view, not camera-image rendering. Both
processes share the same physical GPU, so it is an evaluation target rather than a
parallel rollout engine.

### Headless acceptance spike

Before building the environment, benchmark four minimal paths and record complete
observation/action steps per second, VRAM, latency and determinism:

1. FlyGym CPU with both front legs actuated and two constrained sticks;
2. FlyGym `GPUSimulation` under WSL2;
3. MJWarp RGB batch rendering of primitive gates at candidate FPV resolutions and batch
   sizes;
4. zero-copy Warp-to-PyTorch pixel/state views on one CUDA stream.

The third test is decisive. NVIDIA currently documents CUDA in WSL2 but not OpenGL-CUDA
interoperability; MJWarp's BVH batch ray tracer is not the ordinary OpenGL renderer, so
test its returned RGB buffers directly. If the FlyGym/MJWarp path fails or is slower than
the target after profiling, use PyFlyt for the first flying demo and retain FlyGym solely
for the synchronized leg/stick plant. Do not reach for Isaac Lab or a custom renderer
before this fallback is measured.

### Available compute

The development machine exposes an RTX 5080 with 16,303 MiB VRAM to WSL2 and PyTorch
CUDA 12.8. That is a good target for a neuron-level sensorimotor subgraph and batched
truncated recurrent training. It is not enough to treat all 151.9 million raw segment
edges as a convenient first differentiable model. Use GPU CSR/segment-reduction kernels;
use Warp/MJWarp for GPU environment kernels and reserve Numba for a demonstrated CPU
preprocessing bottleneck. Numba will not accelerate AirSim RPC, MuJoCo's compiled solver,
MJWarp or PyTorch CUDA kernels.

## 6. Training curriculum

### Phase 0 — measurement harness

- Inspect graph schema, row counts, join coverage, duplicate edges, self-edges, strongly
  connected components, neuron/type distributions, and transmitter confidence.
- Lock coordinate frames and signs with deterministic roll/pitch/yaw/throttle tests.
- Unit-test gate crossing, role transitions, resets, seeding, action holding, and latency.
- Train a conventional privileged-state pilot to prove the plant and reward are solvable.

### Phase 1 — sensory pretraining

- Train the visual portion on synthetic optic flow, ego-rotation, contrast transitions,
  gate segmentation, gate-role classification, and time-to-contact.
- Compare training from MaleCNS-derived visual circuitry with importing or adapting the
  Flyvis approach. Do not silently mix its female/consensus visual graph with MaleCNS.
- Verify motion and role information is present in connectome activations before adding
  the flight objective.

### Phase 2 — teacher and imitation

- Teacher sees pose, velocity, angular rates, and gate geometry and generates acro stick
  demonstrations.
- Convert each teacher stick command into training-only foreleg pose/motor targets. At
  deployment, no inverse-kinematics teacher remains: the connectome drives the legs and
  measured stick motion drives the quad.
- Begin airborne: recovery from small attitude/rate perturbations, hover, then translation.
- Add takeoff only after airborne stabilization works.
- Train only parameters inside the connectome while keeping both stateless interfaces
  frozen. Use behavior cloning followed by DAgger-style recovery data.

### Phase 3 — recurrent reinforcement learning

- Use recurrent PPO with truncated backpropagation and a privileged critic as the first
  robust baseline; keep the actor observation-constrained.
- Curriculum: one large gate -> one angled gate -> two role-changing gates -> short
  randomized courses -> tight turns, speed, occlusion, and distractor gates.
- Reward completion and valid gate crossings; penalize collision and elapsed time. Use a
  modest course-consistent potential for progress and a small action-rate penalty. Avoid
  a persistent survival reward that makes hovering optimal.

### Phase 4 — robustness and fidelity

- Randomize mass/inertia, motor constants, battery voltage, drag, camera intrinsics and
  mounting, textures, lighting, latency, and sensor noise.
- Add wind and model mismatch after nominal behavior is stable.
- Freeze evaluation course families and never train on them.
- Run the frozen actor through a validated Betaflight SITL bridge with the intended rates,
  filters, mixer, motor protocol, and firmware revision.

### Phase 5 — cautious hardware transfer

Begin only after simulation acceptance tests pass. Use a low-mass ducted tiny-whoop in a
netted cage, soft props, hard geofence, independent watchdog, remote kill/disarm, action
and rate limits, and a human safety pilot. The supervisor, not the neural policy, owns arm
authority. Log every frame, sensor packet, action, firmware state, and intervention.

## 7. Baselines and falsification

To show that the connectome contributes rather than merely supplying a large recurrent
network, compare under matched observations, fixed motor interfaces, parameter budget, environment
steps, seeds, and tuning effort:

1. compact CNN + GRU engineering baseline (not a valid fly controller);
2. random sparse recurrent graph with matched degree distribution;
3. degree-preserving rewired MaleCNS graph;
4. MaleCNS with shuffled or uniform edge counts;
5. MaleCNS without transmitter signs, and with uncertain signs varied;
6. feed-forward/no-recurrence ablation;
7. input/output-interface-only control, which must fail because the interfaces are fixed
   and stateless;
8. selected subgraph versus progressively expanded and full graph.

Report success rate, gates per episode, course completion time, collision and intervention
rates, wrong-gate passes, control saturation, inference latency, sample efficiency, and
performance under delay/dynamics shifts. Use multiple seeds and held-out course families.
Inspect pathway and cell-type ablations, but do not infer biological causality from a
drone-trained model without experimental validation.

## 8. Milestones and stop/go tests

1. **Data integrity:** every downloaded file matches the official size/hash; derived graph
   construction is deterministic and reports annotation/transmitter join coverage.
2. **Environment integrity:** deterministic replay; validated gate geometry; a conventional
   teacher reliably completes one- and multi-gate tasks.
3. **Embodiment:** commanded front-leg motor populations visibly move the correct sticks;
   measured stick deflection is the only four-axis command received by the flight model.
4. **Minimal experiment:** a selected connectome sensorimotor graph stabilizes from an
   airborne start and passes one gate better than chance, with no privileged actor input.
5. **Scientific signal:** connectome model improves at least one predeclared metric over
   matched rewired/random controls across seeds. If not, report the negative result before
   scaling up.
6. **Course task:** held-out multi-gate completion with delayed role changes, distractors,
   and randomized visuals/dynamics.
7. **Firmware fidelity:** frozen policy retains acceptable performance through Betaflight
   SITL under the target RC-rate profile and measured delays.
8. **Hardware readiness:** only after an explicit safety review and intervention-rate test.

The first decisive deliverable should be milestone 4 plus honest baselines—not a costly
full-graph racing run.
