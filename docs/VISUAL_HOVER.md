# Full-connectome visual hover experiment

## Status

The new visual stack has passed a **causal marker-response test**, but it has not yet
passed closed-loop hover. This distinction matters: the experiment now proves that the
full recurrent MaleCNS model can see a displaced physical marker and move its throttle
leg in the correct direction. It does not yet keep the vehicle at the marker height for
a complete episode.

The compact committed result is in
[`artifacts/full-visual-hover-diagnostic-v1`](../artifacts/full-visual-hover-diagnostic-v1/).
Large generated graphs, checkpoints and traces remain under ignored `data/derived/` and
`runs/` directories.

## Deployed actor boundary

The actor receives only:

- a 320 by 200 linear-RGB FPV frame with 125-degree horizontal field of view;
- estimated roll and pitch angles through fixed push/pull sensory channels;
- the recurrent state of the MaleCNS-derived network itself.

It receives no acceleration, mass, hover-thrust value, target coordinate, image
difference, optical flow, external history, phase bit or controller state. Its outputs
are 26 identified T1 foreleg motor neurons grouped into antagonist pairs. Those pools
drive an abstract two-axis left foreleg and two-axis right foreleg, which move the four
Mode-2 sticks. The ordinary 100 Hz aircraft rate controller and mixer remain part of the
aircraft, not the fly.

The fly policy runs at 50 Hz. The foreleg, stick, aircraft physics and rate controller
run at 100 Hz with the most recent neural command held between policy updates. This is a
good first compromise: it preserves 20 ms sensory/action updates while leaving enough
RTX 5080 capacity for several simultaneous training environments.

On the local RTX 5080, the complete renderer-plus-recurrent-graph path measured 274
neural steps/s at batch 1, 320 batched steps/s (1,278 camera frames/s) at batch 4, and
179 batched steps/s (2,861 frames/s) at batch 16. A batch-4, ten-step forward/backward
unroll measured 77.7 neural steps/s and about 3.0 GB peak CUDA allocation. GPU execution
is therefore already useful. Numba would not accelerate the dominant sparse PyTorch
gather/scatter and autograd work; it remains an option only for a future CPU simulator or
reward kernel found to be a measured bottleneck.

## Camera, scene and retina

[`visual_hover.py`](../src/flydrone/visual_hover.py) is a differentiable, analytic,
headless renderer. The cool wall and warmer floor have faint, deterministic multiscale
patterns fixed in world coordinates, so motion produces ordinary pixel motion. The
height marker is physical paint on the wall rather than a screen-space overlay. No
precomputed horizon or flow reaches the actor.

[`full_connectome_data.py`](../src/flydrone/full_connectome_data.py) retains all 165,122
traced MaleCNS v1.0 neurons and 2,749,407 released edges with at least ten synapses. It
does not use shortest-path extraction. The fixed camera interface directly drives 2,197
photoreceptors:

| Receptor type | Count | Fixed RGB transduction |
| --- | ---: | --- |
| R1-R6 | 1,386 | linear RGB luminance |
| R8p | 330 | blue channel |
| R8y | 481 | green channel |

Retinal positions are inferred only from released receptor-to-retinotopic-partner
connectivity and official optic-lobe hex annotations. R7 and the remaining unclear or
dorsal-specialized receptors stay in the recurrent graph but are not given an invented
signal: an RGB camera has no ultraviolet channel, and the missing tuning is not supplied
by the connectome release.

## Marker-height anti-shortcut design

The paired learner deliberately uses disconnected absolute-height ranges:

- training marker heights: 0.60-0.85 m and 1.15-1.40 m;
- held-out marker heights: 0.90-1.10 m.

For each contrast trial, the two branches have exactly the same vehicle pose, attitude,
velocity, world-fixed floor and wall texture, and initial neural state. Only the physical
marker height differs by plus or minus 0.20 m. Identical-marker and swapped-marker
controls are acceptance requirements. This is stronger than merely randomizing height:
background position cannot explain a difference between paired branches, and the
unseen-height interval tests generalization rather than memorization of the training
bands.

Future closed-loop training must preserve these paired batches alongside ordinary
episodes. It should also randomize texture phase, wall range and illumination
independently of marker height, then retain a disjoint combination test.

## Results

The first broad imitation run did not learn useful visual feedback. In 24 ten-second
episodes it passed 1/24, had 0.683 m mean final-window altitude RMSE, contacted the ground
in 50% of episodes, and behaved almost identically with the first image frozen. Its mean
settled throttle was close to the population-average hover value, but its correlation
with required throttle across randomized mass was only 0.066. It learned an average trim,
not hover-thrust estimation. Removing roll/pitch caused every episode to tumble and
contact the ground.

The follow-up paired dynamic-prefix learner passed its preregistered marker-response
gate on 64 held-out pairs:

| Metric | Before | After |
| --- | ---: | ---: |
| Normalized contrast RMSE | 1.000 | 0.134 |
| Correct response direction | 43.8% | 100% |
| Response slope | approximately 0 | 0.964 |
| Predicted throttle contrast RMS | 0.00009 | 0.0836 |
| Desired throttle contrast RMS | 0.0859 | 0.0859 |

The acceptance limits were NRMSE at most 0.25, direction at least 95%, and slope between
0.5 and 1.5. Identical images produced no meaningful contrast and swapping the two marker
images reversed the response.

A four-second closed-loop step test then showed the complete causal chain from motor
pool to foreleg/stick, actuator, vertical velocity and height. It kept all 32 nominal-mass
episodes airborne and had 4.19-degree RMS tilt. The signed response ratio averaged 1.025,
but final absolute target error was still 0.503 m and the full flight criterion failed.
The remaining problem is absolute collective calibration and vertical damping, not
whether the marker reaches the throttle output.

## Are we using the ventral nerve cord?

Yes, for the learned marker response—but only part of it is presently demonstrated as
useful.

The full graph contains 20,350 VNC-class neurons, including 13,151 `vnc_intrinsic`, 6,365
`vnc_sensory`, and 708 `vnc_motor` neurons. Only the declared 26 T1 motor neurons are read
out, but marker-dependent activity is distributed through 19,642 non-motor VNC neurons.

The causal assay removes only the *difference* between paired marker conditions in a
selected population after every recurrent update. It replaces those states by their
pairwise mean, leaving their common activity and every unselected state intact:

| Contrast clamp | Nodes | Remaining marker-to-throttle contrast |
| --- | ---: | ---: |
| None | 0 | 100% |
| All non-motor VNC | 19,642 | 33.4% |
| `vnc_intrinsic` | 13,151 | 33.4% |
| `vnc_sensory` | 6,365 | 105.1% |

Thus VNC intrinsic recurrence is causally carrying roughly two-thirds of the learned
marker-to-throttle response in this assay; the result is not explained merely by reading
VNC motor neurons. Conversely, current VNC sensory circuitry is not helping this
response. This does not identify a biological control algorithm or prove a closed-loop
flight benefit.

The natural next VNC responsibility is local foreleg/stick feedback through identified
FeCO pathways. That has not been connected yet. We should add it only after the nominal
visual loop is stable, then require matched live, constant and pair-shuffled
proprioception controls. The directional tuning must be documented rather than assigned
post hoc.

## Learning hover thrust without supplying mass

The controller should learn the action required for zero vertical motion, not the drone's
mass. The next curriculum is:

1. Preserve the held-out paired marker response while fitting nominal-mass closed-loop
   proportional control and visual vertical-motion damping.
2. Randomize thrust-to-weight, battery effectiveness and modest tilt only after nominal
   hover passes. Use small training-only throttle excursions or teacher handoffs so the
   recurrent graph observes how its own action changes textured-scene motion.
3. Add real foreleg/FeCO stick-position feedback as the cleanest action observation.
   Native motor recurrence is already an imperfect efference copy; no external history
   channel is needed.
4. Use the exact steady hover command only as a privileged loss, critic input and probe
   target. Never feed its value, mass or a decoded latent back to the deployed actor.
5. Accept adaptation only if stable airborne episodes show the correct settled-throttle
   ordering on unseen thrust-to-weight values and if live action/visual histories beat
   constant and pair-shuffled controls.

Only after that loop passes should we zero estimated roll/pitch and attempt a truly
visual-only policy. The current roll/pitch ablation is a failure, not evidence that the
network can yet infer attitude from the image.

## Reproducing the experiment

Build the ignored full graph from the locally restored official release:

```bash
aira confine -- scripts/build_full_visual_connectome.py
```

Run the primary trainer and the focused causal learner:

```bash
aira confine -- scripts/run_visual_hover_training.sh
aira confine -- scripts/run_visual_height_response_training.sh
```

Then run the closed-loop and VNC audits:

```bash
aira confine -- .venv/bin/python scripts/evaluate_visual_height_step.py \
  --checkpoint runs/visual-hover/paired-dynamic-001/controller.pt
aira confine -- .venv/bin/python scripts/audit_visual_vnc_use.py \
  --checkpoint runs/visual-hover/paired-dynamic-001/controller.pt
```

The graph is a transformed derivative of the official MaleCNS v1.0 data under CC BY
4.0. Provenance, checksums and the transformation notice are written into its generated
manifest and described in [`data/README.md`](../data/README.md). No new downloaded or
generated binary is committed by this experiment, so no additional Git LFS object is
needed.
