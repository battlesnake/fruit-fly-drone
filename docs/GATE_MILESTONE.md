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

### New full-network feasibility result

The older compact milestone below remains the takeoff and broad-evaluation record. A
separate behavior-first run now demonstrates substantially stronger visual steering with
the complete 165,122-neuron recurrent graph and the requested 320×200 RGB camera. On 64
untouched airborne flights arranged as 32 physically matched left/right pairs, it made
50 clean passes (78.1%), exactly 25/32 on each side. Eighteen pairs passed in both
directions, 38 flights met the stricter post-crossing success definition, and none hit the
ground or became invalid. Freezing RGB after 0.5 seconds cut clean passes to 24 and caused
the positive-offset side to score 0/32.

This does not supersede the formal milestone: it starts airborne, uses nominal mass and a
narrower nearby-gate distribution, and remains below 90% strict success. It does establish
the feasibility needed to proceed directly to a continuous two-gate experiment. Full
details and limitations are in the
[`pragmatic full-network gate record`](../artifacts/pragmatic-full-native-gate-v1/).

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

## Throttle-stick proprioception diagnostic

[`gate-proprio-diagnostic-v1`](../artifacts/gate-proprio-diagnostic-v1/) adds four traced
left-prothoracic `SNpp50`/`SNpp51` MaleCNS cells annotated as femoral chordotonal-organ
claw homologues. The published MaleCNS annotations do not resolve the flexion/extension
tuning of these particular cells, so assigning the two anatomical types to complementary
high/low virtual-stick positions is explicitly an engineering mapping. The relevant
position-coding physiology comes from the open-access
[FeCO circuit study](https://doi.org/10.1038/s41467-025-59302-3).

The resulting graph has 1,147 neurons and 4,542 edges. It is an exact warm extension of
the v2 graph: 1,138 neurons and 4,469 edges are shared by body ID and endpoints, and the
search can change only 73 edges touching the nine added circuit cells. On a fresh,
balanced 1,024-flight suite the selected candidate improved the unchanged warm start
from 41.41% to 49.41%, mostly by reducing lateral error. This is not a successful
proprioception result. Constant position retained 48.73%, mass-rank-swapped position
retained exactly 49.41%, and the measured position-by-acceleration intervention was only
`-5.96e-8`. The added pathway supplied useful anatomical capacity, but no meaningful
dependence on live stick position was demonstrated. The checkpoint is retained as a
negative result and is not promoted as the current actor.

## Privileged exact-mass upper bound

[`gate-mass-oracle-v1`](../artifacts/gate-mass-oracle-v1/) freezes every v2 connectome
parameter and tests one narrow question: would correct mass-dependent collective trim be
enough? Exact simulator mass selects an external differential bias on the existing
throttle motor pools, `b = -0.05 + 0.10 z`, where
`z = clamp((mass_scale - 1) / 0.08, -1, 1)`. This adds two runtime calibration parameters
outside the connectome and therefore **does not count toward the direct-sensor goal**.

The calibration scored 100% on its 256-flight held-out validation and again passed all
1,024 fresh balanced final flights. On the identical final suite, the unchanged
controller scored 42.77%, the validation-selected mass-independent constant trim scored
40.62%, and shuffling mass labels inside matched gate-side/obliquity strata scored only
10.06%. True physics mass was not shuffled. The result is strong causal evidence that
the current fixed-geometry task is limited by mass-dependent throttle calibration. The
shuffled control is actively harmful because it applies the wrong correction; it does
not imply that the fly inferred mass.

This does not establish that exact mass should be a deployed input, nor that injecting
mass into sensory neurons would reproduce the oracle. The current plant also scales
inertia with its mass multiplier, so that scalar identifies more than weight alone.

## Delayed oracle and native recurrence diagnostics

The delayed-oracle audit freezes every controller parameter and starts the same exact-mass
trim later in flight. Success on 1,024 balanced flights was 100.0% at 0 and 0.25 seconds,
91.21% at 0.5 seconds, 90.92% at both 0.75 and 1.0 seconds, 87.70% at 1.5 seconds, 76.27%
at 2.0 seconds, and 62.40% at 3.0 seconds. The deployed actor still never receives mass;
this only establishes that it has roughly 0.5–1.0 seconds in which to infer a useful
collective correction.

A separate held-out ridge diagnostic used only the actual causal input traces. Body-Z
acceleration history decoded normalized mass at 0.25 seconds with R²=0.983 and 96.9%
mass-sign accuracy, rising to R²=0.999 at 0.5 seconds. Adding completed-step stick
position raised the 0.25-second result to R²=0.9998. The linear decoder is an analysis
tool, not part of the actor; these results say the sensor history is informative, not
that the fly has learned to use it.

The 1,138-node direct-accelerometer graph is strongly recurrent: 718 nodes belong to
non-singleton strongly connected components, and the largest such component contains
558 nodes. Every controller tick passes the previous leaky membrane state into the next
tick; it is reset only between episodes. Training detach boundaries limit how far a
gradient is propagated but do not erase the numerical state during a rollout.

The bounded native-motif experiment selected only existing transmitter-signed anatomical
edges: an 18-neuron, 21-edge set of sensory paths and positive-feedback cycles, with eight
existing return edges available to the throttle motor pools. During its causal retention
condition, the FPV image became black, roll/pitch became zero, and acceleration became a
constant 1g after 0.75 seconds. At update 200 the motif still ordered mass at 1.0 and 1.5
seconds with Pearson correlations 0.906 and 0.921. This is direct evidence of internal
hysteresis; no external history tensor, estimator, clock, or state machine supplied the
memory.

It is a negative controller result. The retained code was not calibrated: at 1.5 seconds
its signed slope was 0.447 and R² was -1.915. With continuing live inputs, the same
checkpoint fell to r=0.151, slope=0.013, and R²=-8.504.

A subsequent bounded test retained every 20-update encoder checkpoint and selected update
180 by neutral-suffix ordering. It froze the encoder and fitted all 37 existing signed
edges from 19 source neurons into the two throttle motor pools. The readout was one
time-independent function shared by all conditions and times. It also allowed one learned
antagonist motor-pool bias (`c=0.04662`, below its ±0.2 bound) to cancel common-mode offset.
Exact mass still appeared only in the fitting target; deployment inputs and state were
unchanged.

That more permissive readout also failed held-out replay. Its overall correction slope was
0.260, R² was 0.240, and normalized RMSE was 0.872. Under clean live sensing, slope fell
from 0.197 at 0.75 seconds to 0.099 at 1.0 seconds and 0.008 at 1.5 seconds. The neutral
suffix preserved ordering better but not usable gain: slope was 0.224 at 1.0 seconds and
0.094 at 1.5 seconds. The linearized training replay was already poor, so recurrent
feedback after fitting is not the main explanation. The preregistered screen rejected the
readout before flight evaluation. The conclusion is therefore **memory capacity is
present, but this hand-selected mass-code-and-static-readout scaffold cannot supply robust
control calibration**. It should not be broadened into another motif search; the next
experiment should optimize bounded end-to-end trajectory loss through the recurrent actor
and plant. Mass remains a training label only.

That end-to-end route was audited before any update. A fresh `gate-accel-v2` controller
was unrolled for the complete eight-second horizon through the recurrent graph, measured
foreleg sticks, renderer, and vehicle. Applying identical teacher commands to cloned plants
gave exactly zero position and velocity loss, and the gradient-enabled actor call matched
the frozen deployment call exactly. Late trajectory loss also had a measurable causal
response to a perturbation of the first roll command.

Nevertheless, the full-horizon parameter gradients were unusable. Three unchanged CUDA
replays varied by only 0.000153 in a loss of about 3.02, but raw gradient norms reached
`1.37e17` for edges, `6.93e16` for biases, and `8.33e14` for time constants. The analytic
mixed-parameter directional derivative was `1.44e16`, whereas central finite differences
at perturbations from `1e-4` to `1e-5` were only 52–124. For the roll-motor bias contrast,
the corresponding values were `-1.94e16` versus -18 to -62. Late-loss sensitivity to the
first roll command likewise differed by fifteen orders of magnitude (`-5.80e15` analytic,
`-2.74` measured). The harness therefore made no optimizer update.

This is not evidence against recurrence. It shows that differentiating through 800 closed-
loop recurrent/physics steps is catastrophically ill-conditioned at this checkpoint.
Clipping would hide the magnitude without repairing the direction. The next training path
must use complete-flight rollout scores—such as antithetic evolution strategies or policy
gradients—without differentiating through the flight history. The deployed actor will
still retain only its internal connectome state.

## Native motor-interface evolution strategy

The first rollout optimizer used a deliberately small 24-dimensional parameterization.
Each of the eight existing antagonist motor pools received one shared internal bias delta,
one gain on all transmitter-positive incoming edges, and one gain on all transmitter-
negative incoming edges. The gains were compiled into the 196 existing edge magnitudes and
the biases into 26 motor neurons; topology, transmitter signs, zero edges, time constants,
retinal mapping, leg mechanics, and quad dynamics stayed fixed. No search coordinate or
scaler remains outside the connectome at runtime.

Sixteen antithetic directions were evaluated as one 34-policy CUDA batch for 60 generations.
Each policy saw the same 32 cases in a generation, balanced across the eight intersections
of mass, lateral side, and obliquity sign. The eight-second reward prioritized complete
success and clean crossing, with smaller progress, approach, hazard, and stick-saturation
terms. Fitness mixed mean reward with the worst intersection stratum. Independent
256-flight validation ran every ten generations; every archived vector was retained, and
the selected vector had to avoid a mass-half collapse before the fresh final audit.

The promoted checkpoint improved 1,024-flight success from 42.97% to 49.51%. There were 74
baseline failures repaired by the candidate and seven baseline successes lost, giving a
paired gain of 6.54 percentage points with a 95% normal-approximation interval of 4.87–8.22
points. Negative-side success rose from 47.27% to 52.15%, positive-side success from 38.67%
to 46.88%, and mean crossing radial error fell from 0.748 m to 0.663 m. Lower-mass success
moved only from 6.64% to 5.86%, within the declared five-point loss bound, while higher-mass
success rose from 79.30% to 93.16%. Frozen-first-frame success was 4.88%.

This is a material native-controller improvement, but it did **not** learn mass adaptation.
Replacing body acceleration with a constant 1g slightly increased success to 50.0%, with
nearly identical mass strata. The small motor-interface search found a better static trim,
not a history-dependent calibration policy. The next selective rollout experiment must
target recurrent acceleration-to-throttle paths and use live-versus-constant and
matched-trace controls before claiming sensor-dependent adaptation. Ordinary flight
improvement can still count without such a claim. The original 90% criterion remains unmet.

## Recurrent acceleration-path evolution strategy

The next rollout experiment started from the promoted motor-interface controller and
expanded plasticity to the 282 real edges on acceleration-to-throttle paths of at most
four hops. It froze all biases, time constants, nonselected edges, topology, and
transmitter signs. Search cases were adjacent light/heavy pairs with identical gate
geometry; the objective rewarded light-mass improvement while penalizing any heavy-mass
reward loss. Thirty-two antithetic directions and 32 common-seed flights per policy were
used for 60 generations. Normalized center updates were capped at one quarter of the
current exploration sigma.

The generation-20 continuation test passed. The selected validation candidate improved
light success by 9.38 percentage points and reduced heavy success by 1.56 points on 256
matched flights. Thirteen selected edges had zero magnitude at the start; their initial
0.000777 exploration magnitude produced a measurable `1.98e-5` early throttle-stick
change, so the search genuinely allowed those anatomical edges to activate.

Fresh testing rejected the candidate. On 1,024 matched flights, light success rose from
4.88% to 12.89% (paired gain 8.01 points, 95% CI 5.65–10.36), heavy success moved from
91.60% to 90.62%, and overall success rose from 48.24% to 51.76%. The preregistered light
gain was ten points. A separate balanced evaluation improved overall success from 47.95%
to 50.88% and mean radial error from 0.674 m to 0.614 m, but frozen-first-frame vision
still scored 18.07%. Live acceleration beat constant 1g by 1.07 points, while swapping
the acceleration traces within each matched light/heavy pair caused exactly zero net
success change. This does not establish mass-specific acceleration adaptation.

The four-hop mask is therefore closed without promotion or automatic expansion to six
hops. The compact audit is in
[`gate-acceleration-path-diagnostic-v1`](../artifacts/gate-acceleration-path-diagnostic-v1/).

## Recurrent PPO at the native motor interface

A recurrent-PPO follow-up independently exposed all 198 existing edges entering the 26
front-leg motor neurons plus their 26 intrinsic biases. The remaining connectome edges,
all time constants, topology, and transmitter signs stayed frozen. Each rollout carried
the native 1,138-neuron state continuously; 32-step training chunks used eight preceding
steps of burn-in. There was no observation-history stack or added recurrent module.

The full simulator stayed outside autograd. A training-only critic could see physical
state, mass, gate pose, foreleg state, reward bookkeeping, and a detached copy of the
native neural state. The deployed actor still received only current FPV, roll/pitch,
body-Z specific force, and its native recurrent state. The common controller API also
accepted stick position, but this graph had no proprioception nodes and ignored it.
Exploration was calibrated separately for light and heavy cases and settled at a motor
sigma of 0.00375. Native-forward parity, the previously validated batched evaluator,
25-step finite-difference gradients, unchanged-policy recurrent replay, and truncated
burn-in all passed their numerical audits.

The run stopped at its iteration-10 checkpoint. Validation light success had gained 10.16
points, but heavy success lost 7.03 points, so no archived point met the continuation
rule. The selected noninferior validation snapshot was then tested on 1,024 new matched
flights: reference overall/light/heavy success was 47.66%/3.71%/91.60%, versus
47.56%/5.27%/89.84% for the candidate. The paired light gain was significant but only
1.56 points; overall performance was flat and the heavy paired noninferiority interval
failed. Frozen-first-frame vision scored 2.64%. Constant-1g acceleration slightly beat
live acceleration, while matched-pair acceleration swapping changed no successes.

The candidate was rejected and no controller checkpoint was emitted. This demonstrates
that recurrent PPO can train through the native state without numerical or external-memory
shortcuts, but this last-layer plasticity trades mass strata rather than learning useful
acceleration-dependent compensation. The complete compact record is in
[`gate-recurrent-ppo-diagnostic-v1`](../artifacts/gate-recurrent-ppo-diagnostic-v1/).

A final bounded diagnostic opened the complete 1,138-neuron model: all 4,469 fixed-sign
edge magnitudes, 1,138 biases, and 1,138 native membrane time constants. A frozen native
controller with a training-only privileged mass bias generated absolute four-axis action
labels on current-student histories. It shadowed the student's actual observations with
its own recurrent state and never drove student collection physics. Twenty-five percent
of each batch used clean oracle-driven trajectories only as a stabilizer. Prefixes were
reconstructed from reset under the current student without gradients; gradients flowed
through the following 50 native recurrent steps. No mass, timer, history features, teacher
state, or added recurrence entered the student.

Teacher-label-on/off CPU trajectories were bit identical. Separate edge-, bias-, and
time-constant-only finite differences passed at both 25 and 50 steps on early and late
windows. The oracle also recovered 95.3% of trials after shadowing and taking control at
two seconds, so the target remained valid on student-visited states. The run stopped at
its predeclared update-100 checkpoint because no snapshot jointly improved action fidelity,
light flight, and heavy non-degradation.

The student reduced average oracle-action error but did so by learning a shared throttle
offset. On fresh student histories its matched light/heavy throttle slopes at 0.75, 1.5,
and 3 seconds were 0.024, -0.0007, and 0.008. On 1,024 new matched flights, reference
overall/light/heavy success was 47.46%/3.91%/91.02%, versus 43.07%/50.78%/35.35% for the
selected diagnostic candidate. Constant-1g scored 42.68%, and pair-swapped acceleration
scored 43.26%, compared with 43.07% live. This rules out useful accelerometer-conditioned
hysteresis in this run despite the network's structural recurrence. The candidate was
rejected and no checkpoint was emitted. The compact record is in
[`gate-full-network-oracle-diagnostic-v1`](../artifacts/gate-full-network-oracle-diagnostic-v1/).

To separate representational capacity from the failed long-horizon optimization, a tiny
follow-up froze 0.75-second promoted-student sensor prefixes for eight distinct gate
geometries, each paired at exact mass scales 0.92 and 1.08. It trained the same complete
native parameter set through all 75 recurrent steps. Light-minus-heavy throttle contrast
and pair-mean correction had separate fixed normalization and equal loss weight, so the
unconditional average learned by the full-flight experiment could not pass.

The selected run achieved 2.12% normalized contrast error and 4.60% pair-mean error on its
training pairs. When both members received the same complete prefix, their four-axis
outputs were bit identical. On eight disjoint geometries, the 0.75-second contrast error
was 5.71%, though pair-mean error rose to 17.84%; the untrained 0.50-second endpoint did not
generalize. This proves the native recurrent state can support a conditional light/heavy
action and points specifically to full-prefix, multi-time temporal credit assignment as
the next flight-training step. The diagnostic parameters are not promoted. See
[`gate-conditional-overfit-diagnostic-v1`](../artifacts/gate-conditional-overfit-diagnostic-v1/).

That next experiment first applied a fixed teacher-quality gate. A hybrid of the promoted
controller's steering and the old oracle's throttle scored 87.5% overall and on each mass
half across 64 deliberately diverse cases, below the preregistered 90%, so it stopped
before update 1. A bounded recalibration of the promoted controller's privileged mass bias
then achieved 79.6% over 1,024 fresh flights. Swapping mass labels reduced it to 6.1%, but
the remaining error was strongly directional: 100% success for negative lateral offsets
and 59.2% for positive offsets. This rejects the two-parameter throttle-only oracle family
for the broader geometry distribution.

Four steering-aware analytical teachers were then compared on a disjoint selection set.
All achieved 100%; the chosen visual/accelerometer teacher with exact training-only mass
had the smallest radial error. Frozen before evaluation, it retained 100% overall, light,
heavy, negative/positive lateral, and negative/positive obliquity success over 1,024 fresh
flights after a 0.5-second promoted-controller prefix. This validates the next teacher but
does not count as direct-sensor flight. Records are in
[`gate-multitime-preflight-diagnostic-v1`](../artifacts/gate-multitime-preflight-diagnostic-v1/),
[`gate-promoted-oracle-diagnostic-v1`](../artifacts/gate-promoted-oracle-diagnostic-v1/), and
[`gate-analytic-teacher-v1`](../artifacts/gate-analytic-teacher-v1/).

The first full-prefix, seven-horizon distillation of that exact-mass teacher stopped after
its fixed 200-update first round. Although the network learned the 0.75-second throttle
contrast, its predicted 0.5-second contrast stayed essentially zero and the selected
snapshot also failed early roll, pitch, and throttle pair-mean fidelity. No DAgger round,
final flight evaluation, or promotion followed. The failure record is in
[`gate-multitime-analytic-exact-mass-diagnostic-v1`](../artifacts/gate-multitime-analytic-exact-mass-diagnostic-v1/).

Exact mass is unnecessary for the task: the visual/accelerometer reserve teacher was also
perfect on the original selection set and then achieved 100% in every declared stratum on
1,024 newly seeded cases, with no collisions or misses. It is now the frozen target for
mass-free native distillation. Its action labels are invariant to changing only the mass
argument, and near-zero throttle contrasts use a fixed actuator-scale normalization floor
instead of a data-dependent microscopic denominator. See
[`gate-analytic-teacher-reserve-v1`](../artifacts/gate-analytic-teacher-reserve-v1/).

The first mass-free distillation confirmed that the 0.5-second issue was fixed: its raw
teacher contrast was only about 0.0003 motor units and the learned error was 0.028 of the
predeclared 0.01 actuator scale. The sampled-endpoint optimizer still failed its first-round
fidelity gate. Updates that improved the middle horizons damaged the 0.5- and 5-second
means, while roll error stayed high, exposing multi-objective interference rather than a
need for a mass input. The rejected record is in
[`gate-multitime-reserve-sampled-diagnostic-v1`](../artifacts/gate-multitime-reserve-sampled-diagnostic-v1/).

A follow-up capacity audit placed all seven weighted endpoint losses in every complete
500-step replay and gave each steering axis equal effective weight. Over its fixed 150
updates, the worst threshold-normalized training error improved from 18.75 to 5.03 and
disjoint holdout improved from 19.52 to 4.52, but the training fit still failed. Every
native parameter family had finite, nonzero gradients. The initialization was the earlier
unpromoted exact-mass conditional-overfit vector, whose 0.75-second reserve contrast began
with the wrong sign; the audit therefore rejects that initialization and budget, not
native recurrence. See
[`gate-joint-multitime-overfit-diagnostic-v1`](../artifacts/gate-joint-multitime-overfit-diagnostic-v1/).

A strict initialization control then restored the promoted source parameters at update 0
while keeping the same cases, histories, scales, seeds, optimizer, budget, and old-vector
regularization reference. Its initial training margin was 5.00 rather than 18.75, and the
selected update-150 snapshot reached 4.37 on training and 4.05 on disjoint holdout. It
learned the 0.75-second contrast but left later contrast and roll far outside the fit gate.
The unchanged failure stops this exact-action configuration without a post-hoc extension;
see
[`gate-joint-multitime-source-init-diagnostic-v1`](../artifacts/gate-joint-multitime-source-init-diagnostic-v1/).

The next bounded experiment replaced endpoint fitting with dense trajectory imitation and
DAgger. It used 1-second native recurrent windows, recomputed current-parameter burn-in
state from the sensor prefix, and opened all fixed-sign edge magnitudes, biases, and time
constants. The 64-flight mass-free expert set was perfect; FP32 replay agreed within
`2.1e-7`, and every parameter family passed central finite differences. Fixed-suite
success nevertheless moved only from 19.9% to 21.1%, entirely on the heavy-mass half, and
the run stopped at its update-200 midpoint gate. See
[`gate-dense-dagger-diagnostic-v1`](../artifacts/gate-dense-dagger-diagnostic-v1/).

V2 tested the concrete weighting defect revealed by that record. It normalized throttle
MAE by the teacher-minus-source correction RMS (`0.0787`) rather than the absolute hover
command (`0.388`) and measured fixed expert-history throttle errors by mass and time.
Crossing-window light-mass error improved slightly at update 50, while the takeover-window
error worsened and flight stayed below baseline. Later checkpoints collapsed to misses;
the source was restored and no parameters were promoted. This rejects loss rescaling as a
sufficient fix, not dense imitation or native recurrent capacity in general. See
[`gate-dense-dagger-diagnostic-v2`](../artifacts/gate-dense-dagger-diagnostic-v2/).

A preregistered frozen-policy factorial audit next replaced motor-command axes after the
same 0.5-second native prefix. Across 512 fresh matched cases, source success was 19.9%,
reserve throttle alone reached 56.6%, reserve roll/pitch/yaw alone reached 36.9%, and the
full reserve teacher reached 100% in every declared stratum. Steering-only takeover left
light-mass success at zero, while throttle-only takeover remained strongly asymmetric by
lateral side. The rejected v2 update-200 controller also recovered from zero to 100% under
full takeover, ruling out irrecoverable damage during its native prefix. This localizes the
task to coupled post-takeover steering and throttle control; see
[`gate-axis-takeover-factorial-v1`](../artifacts/gate-axis-takeover-factorial-v1/).

A staged complete-flight evolution strategy then optimized the existing native readout
while the complementary teacher axes stabilized each flight. The 18 steering parameters
passed their bounded gate: fixed assisted success rose from 53.9% to 75.4%, and the worst
mass/lateral stratum rose from 10.2% to 53.1%. The six throttle-pool gains and biases did
not: under teacher steering, light-mass success remained zero and overall success moved
only from 35.9% to 39.1%. The run stopped at throttle generation 20, before merge or native
joint polish. This demonstrates useful steering readout capacity but rejects a static
six-parameter throttle trim for the diverse task; see
[`gate-assisted-motor-es-diagnostic-v1`](../artifacts/gate-assisted-motor-es-diagnostic-v1/).

The next bounded diagnostic kept teacher steering but widened throttle adaptation to the
complete four-hop accelerometer-to-throttle path: 282 fixed-sign edge magnitudes across 93
neurons. The preserved successful steering vector was not merged into this search. On the
fixed 256-case suite, the source scored 39.8% overall (0% light, 79.7% heavy); the best
candidate scored 44.1% (0% light, 88.3% heavy). Early throttle commands remained almost
identical between matched light and heavy cases. Because no candidate simultaneously
raised light success and the worst mass/lateral stratum by ten points, generation 20
stopped the run before final causal controls. No vector was compiled or promoted; see
[`gate-assisted-acceleration-path-es-diagnostic-v1`](../artifacts/gate-assisted-acceleration-path-es-diagnostic-v1/).

A frozen-replay audit next tested action-relevant routing directly instead of running a
wider optimizer. It recorded the source's real sensor and recurrent histories for 64
development and 128 held-out geometry pairs and fitted one time-independent throttle
correction across 0.5--1.5 seconds. A ridge probe of the 93 path neurons passed easily
(worst time/mass NRMSE 0.090); a probe restricted to the 19 throttle-return sources plus
seven motor states also passed at 0.242. The exact nonlinear fit of all 37 existing
fixed-sign throttle-incoming magnitudes failed at 0.499 and only improved aggregate RMSE
30.2% over a constant, below the required 50%.

On identical saved image, angle, and stick histories, constant-1g acceleration changed the
wide path prediction by 0.0172 motor-drive units but the return-source prediction by only
0.00030. The controller therefore contains a usable action correlate at the return
boundary, while this audit does not attribute that correlate specifically to acceleration.
The native fit was still improving when its 200-update budget ended, so the result also
does not distinguish optimizer limitation from signed-readout constraint. All fits were
diagnostic only and the source was preserved; see
[`gate-throttle-routing-diagnostic-v1`](../artifacts/gate-throttle-routing-diagnostic-v1/).

A paired FP64 trust-region discriminator then used the same frozen development histories
to distinguish optimizer budget from the final readout's fixed signs. Both legal `[0,8]`
magnitude fits converged from different starts to the same loss, with projected gradients
below `1.1e-9`. On 128 fresh held-out pairs they still missed both gates: worst-group
NRMSE was 0.366 and aggregate improvement over the constant baseline was 46.8%. Thirty-two
of 37 magnitudes were pinned to a bound.

A diagnostic-only `[-8,8]` effective-weight relaxation passed the held-out screen at 0.235
worst-group NRMSE and 57.4% improvement, reversing 15 original signs. This establishes a
restriction in the bounded legal readout family rather than the earlier Adam budget, but
does not isolate transmitter signs: ten legal weights also hit the magnitude ceiling and
the relaxed optimizer did not converge. It does not justify changing MaleCNS transmitter
signs. No fitted weight was compiled or promoted; see
[`gate-throttle-readout-constraint-diagnostic-v1`](../artifacts/gate-throttle-readout-constraint-diagnostic-v1/).

A preregistered follow-up raised that ceiling once, from 8 to 32, while retaining the same
37 fixed-sign magnitudes, frozen histories, targets, biases, and exact nonlinear mapping.
Both legal starts converged. On 128 new held-out pairs, the selected fit passed aggregate
improvement at 52.1% but failed the every-group criterion: worst NRMSE was 0.303 against
the 0.25 limit, and four weights remained at the new ceiling. The replay gate stopped the
experiment before any assisted flight. Thus the one-time increase did not make this
fixed-bias readout sufficient, while the active bound, nonconvex fit, and frozen recurrence
still preclude a transmitter-sign or whole-controller impossibility claim. No fitted
weight was compiled or promoted; see
[`gate-throttle-readout-ceiling-diagnostic-v1`](../artifacts/gate-throttle-readout-ceiling-diagnostic-v1/).

The final frozen-history readout test opened bias deltas in `[-2,+2]` for the seven
existing throttle motor neurons alongside the same legal `[0,32]` magnitudes. The selected
source-start FP64 fit converged with projected gradient below `1.5e-9`. On 128 new held-out
pairs it passed aggregate improvement at 54.0%, but its earliest light-mass NRMSE was
0.274 against the 0.25 limit; all other groups were at most 0.243. Five edge magnitudes
and five bias deltas were bound-active. The replay gate stopped the experiment before
assisted flight and, as preregistered, closes this 37-edge/seven-bias frozen-history
family. It does not test adapted upstream recurrence or closed-loop flight. No parameter
was compiled or promoted; see
[`gate-throttle-readout-bias-diagnostic-v1`](../artifacts/gate-throttle-readout-bias-diagnostic-v1/).

A one-layer anatomical experiment then replaced isolated-state fitting with genuine
recurrent replay from zero. Its hashed mask contained 88 existing four-hop-path edges
entering the 19 throttle-return sources from outside that set and all 37 edges entering
the seven throttle motor neurons. The 19 return-to-return edges, every other magnitude,
all biases, and all time constants stayed frozen. The absolute reserve throttle target
was trained on a balanced 48-pair split, while a normalized first-0.2-second four-axis
anchor protected the source motor response.

The source replay parity error was `1.79e-7` and an analytic directional derivative of
0.764231 agreed with finite difference at 0.764191. Minibatch normalized throttle MSE fell
from 0.944 to 0.101 by update 80. At the update-100 reserved-development gate, however,
worst-group NRMSE was 0.608 against the 0.50 continuation limit and aggregate improvement
was only 20.5%; prefix motor RMSE remained safe at 0.000604. The run stopped before fresh
held-out replay or assisted flight. The selected magnitudes were not ceiling-limited,
with a maximum of 3.03 under the bound of 8. This identifies a failed one-layer
continuation gate, not by itself a generalization gap, missing native recurrence, or a
closed-loop flight result: logged minibatch MSE and validation worst-group NRMSE are not
directly comparable. No magnitude was compiled or promoted; see
[`gate-recurrent-routing-diagnostic-v1`](../artifacts/gate-recurrent-routing-diagnostic-v1/).

A no-training sampling audit resolved that statistical ambiguity using the frozen
update-100 vector. Aggregate NRMSE was 0.38081 over all 48 training pairs and 0.38190 over
all 16 validation pairs. The validation-minus-training normalized-MSE pair-bootstrap 95%
interval was `[-0.02959, 0.02801]`, so the preregistered 25% generalization-gap gate did
not pass. Reweighting frozen per-pair loss by the exact 800 recovered minibatch exposures
changed training MSE by only 0.014%, far below the 10% distortion gate. Early heavy error
was similarly limiting in both splits (0.595 and 0.608 NRMSE). This leaves unresolved
early-window fitting error, not demonstrated overfitting or sampling bias. The audit made
no update and consumed no fresh held-out case; see
[`gate-recurrent-routing-sampling-diagnostic-v1`](../artifacts/gate-recurrent-routing-sampling-diagnostic-v1/).

A paired continuation subsequently tested extra optimization against early-window
prioritization. Both arms reset Adam at the retained update-100 vector, used the same
125-edge mask and identical balanced minibatches, and preserved the normalized four-axis
prefix anchor. The control retained equal window weights; the treatment used `(4,1,1)/6`
for early, middle, and late throttle losses. At 100 additional updates, worst early-group
NRMSE improved from 0.6082 to 0.5540 for control and 0.5184 for treatment—8.9% and 14.8%,
short of the 20% continuation requirement. Every middle/late regression check and both
prefix checks passed. Neither arm continued to update 200, consumed fresh held-out data,
or ran assisted flight. This rejects the tested duration and fourfold reweighting as
sufficient fixes for the 125-edge circuit, without attributing the residual to recurrence
itself. No parameter was compiled or promoted; see
[`gate-recurrent-routing-continuation-diagnostic-v1`](../artifacts/gate-recurrent-routing-continuation-diagnostic-v1/).

A subsequent read-only assisted-flight audit recovered both stopped vectors
deterministically and evaluated them with the source on 256 identical fresh, matched
12-second cases. The source drove the first 0.5 seconds; the analytical reserve then
supplied steering while native fly output retained throttle. A full-reserve positive
control passed every flight and every reported stratum, with 0.0788 m mean and 0.1636 m
p90 gate-plane radial error. The source succeeded on 38.28% overall (0% lower mass,
76.56% higher mass), whereas both continuation arms succeeded on 0%, missed every gate,
and increased mean radial error from 1.224 m to about 3.46 m. The geometry-paired success
difference was -38.28 points for each arm, with 95% interval `[-41.96, -34.60]` points.
This closes the tested replay-imitation continuation family because its improving
open-loop objective failed to transfer to closed-loop flight. The audit does not isolate
changed sensory trajectories, accumulated action errors, or behavior past the 1.5-second
training horizon, and it does not reject native recurrence itself. No parameter was
updated or promoted; see
[`gate-recurrent-routing-flight-diagnostic-v1`](../artifacts/gate-recurrent-routing-flight-diagnostic-v1/).

A source-initialized recurrent-PPO experiment next optimized complete 12-second assisted
flight rather than replay fidelity. Only the same 125 fixed-sign magnitudes were
trainable. A parallel frozen source supplied steering during the first 0.5 seconds, the
analytical reserve supplied it thereafter, and the candidate's recurrent actor supplied
native throttle throughout. Exploration and PPO likelihood were scalar throttle only;
the critic's simulator state, mass, gate pose, and reward bookkeeping remained
training-only.

Native parity, assisted-evaluator parity, scalar likelihood finite differences, unchanged-
policy replay, and every truncated burn-in audit passed. Exact replay KL was `7.54e-13`,
burn-in mean exact KL stayed below `4.45e-7`, and PPO's maximum accepted post-update KL
was 0.00134 under the 0.01 cap. The development source scored 38.28% overall, 0% light,
76.56% heavy, and 0% in the worst mass/lateral stratum. Iteration 5 reached 39.06%, 0%,
78.12%, and 0%; iteration 10 reached 35.94%, 0%, 71.88%, and 0%, triggering the fixed
stop. Independent per-step exploration produced no light success across the 320 light
training episodes. The fresh 1,024-case set was not consumed, and nothing was compiled or
promoted. This rejects the tested PPO/exploration configuration, not recurrence itself;
see
[`gate-recurrent-routing-ppo-diagnostic-v1`](../artifacts/gate-recurrent-routing-ppo-diagnostic-v1/).

A no-learning exploration-timescale audit then evaluated the unchanged source on the
same 256 development cases. Two matched innovation seeds compared independent per-step
throttle noise with stationary AR(1) noise at identical latent standard deviation 0.03
and 0.2-second correlation time. Motor perturbation RMS remained about 0.0247 in both
modes, but correlated noise increased the measured closed-loop foreleg-throttle trace
difference from roughly 0.0061 to 0.032--0.034 RMS. Despite that real timescale effect,
both modes achieved 0% light success in both seeds. Light cases crossed the gate plane
about 2.09 m vertically off-centre. Correlated heavy success was 67.97% and 67.19%, a
9.375-point drop from each matched independent run. Both heavy noninferiority checks
passed, but the required 10% correlated light success failed twice. No correlated PPO was
run; the tested 125-edge PPO/exploration family is paused. No parameter changed and
nothing was promoted; see
[`gate-throttle-exploration-timescale-diagnostic-v1`](../artifacts/gate-throttle-exploration-timescale-diagnostic-v1/).

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

Re-run the matched recurrent acceleration-path search with:

```bash
scripts/run_gate_acceleration_path_es.sh
```

Re-run the teacher-steering-assisted recurrent acceleration-path diagnostic with:

```bash
scripts/run_gate_assisted_acceleration_path_es.sh
```

Re-run the frozen throttle-routing diagnostic with:

```bash
scripts/run_gate_throttle_routing_audit.sh
```

Re-run the paired fixed-sign/readout-constraint diagnostic with:

```bash
scripts/run_gate_throttle_readout_constraint.sh
```

Re-run the one-time legal readout-ceiling diagnostic with:

```bash
scripts/run_gate_throttle_readout_ceiling.sh
```

Re-run the native throttle-motor bias diagnostic with:

```bash
scripts/run_gate_throttle_readout_bias.sh
```

Re-run the one-layer native recurrent-routing diagnostic with:

```bash
scripts/run_gate_recurrent_routing.sh
```

Re-run its frozen sampling/generalization audit with:

```bash
scripts/run_gate_recurrent_routing_sampling_audit.sh
```

Re-run the paired recurrent-routing continuation with:

```bash
scripts/run_gate_recurrent_routing_continuation.sh
```

Re-run the recurrent-PPO diagnostic under AIRA with:

```bash
scripts/run_gate_recurrent_ppo.sh \
  --output-dir runs/gate/recurrent-ppo-v1
```

Re-run full-network oracle-action distillation with:

```bash
scripts/run_gate_full_network_oracle.sh \
  --output-dir runs/gate/full-network-oracle-v1
```

Re-run the tiny conditional representability audit with:

```bash
scripts/run_gate_conditional_overfit.sh \
  --output-dir runs/gate/conditional-overfit-v1
```

Re-run the diverse teacher diagnostics with:

```bash
scripts/run_gate_promoted_oracle.sh \
  --output-dir runs/gate/promoted-oracle-v1

scripts/run_gate_analytic_teachers.sh \
  --output-dir runs/gate/analytic-teacher-v1
```

Re-run the negative throttle-stick proprioception search and the privileged mass oracle
with:

```bash
scripts/run_gate_proprioception_es.sh \
  --graph artifacts/gate-proprio-diagnostic-v1/connectome.npz \
  --output-dir runs/gate/proprioception-recheck

scripts/run_gate_mass_oracle.sh \
  --output-dir runs/gate/mass-oracle-recheck
```

Re-run the delayed-oracle, sensor-observability, and native-recurrence diagnostics with:

```bash
scripts/run_gate_mass_oracle_delay.sh \
  --output-dir runs/gate/mass-oracle-delay-recheck

scripts/run_gate_sensor_observability.sh \
  --output-dir runs/gate/sensor-observability-recheck

scripts/run_gate_native_latent_motif.sh \
  --output-dir runs/gate/native-latent-motif-recheck
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
