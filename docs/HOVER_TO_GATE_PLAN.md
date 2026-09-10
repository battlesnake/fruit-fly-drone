# Variable-height hover to annular-gate plan

## Objective and current starting point

Train the recurrent full-MaleCNS controller to:

1. hold a quadcopter at a height indicated only by a physical visual marker whose
   absolute height varies; then
2. take off, approach and fly cleanly through one physical annular gate.

The starting checkpoint is `runs/visual-hover/paired-dynamic-001/controller.pt`, derived
from the full 165,122-neuron graph. It has a correct causal marker-to-throttle response
and acceptable short-horizon attitude stability, but it is **not** a hover checkpoint:
the current closed-loop marker-step test ends with 0.503 m mean absolute height error.

This plan advances one promotion gate at a time. Gate-training infrastructure may be
built while hover runs, but controller optimization for the gate begins only after the
variable-height hover gate passes.

## Fixed actor boundary

Throughout both goals, the deployed actor receives:

- 320 by 200 linear-RGB FPV at 125-degree horizontal field of view;
- estimated roll and pitch angles for the initial milestones;
- only the recurrent state belonging to the MaleCNS-derived network.

The actor emits roll, pitch, yaw and throttle through identified T1 front-leg motor
pools, the abstract forelegs and Mode-2 sticks. It receives no mass, hover-thrust value,
position, velocity, optical flow, target coordinates, gate detector, timer, phase bit,
external history or added recurrent model. Simulator state may be used for training
labels, critics, rewards and evaluation only. The aircraft's 100 Hz angular-rate loop
and mixer remain aircraft machinery; the fly policy runs at 50 Hz.

## Protocol correction before the next run

The completed paired diagnostic held out the **base vehicle/reference-height** interval
0.90-1.10 m from base intervals 0.60-0.85 m and 1.15-1.40 m. Its two branches then moved
the marker by plus or minus 0.20 m. Those resulting marker-height ranges overlap, so the
diagnostic must not be described as holding out every absolute marker height.

It still proves marker causality: pose, texture, attitude and recurrent state are
identical between each pair, and only the physical marker changes. Before new training,
make the broader generalization protocol explicit:

- vary camera height and marker height independently;
- reserve genuinely unseen absolute marker-height bands for final evaluation;
- reserve unseen marker-height/camera-height combinations even when each individual
  value appeared during training;
- randomize wall range, texture phase, floor texture and illumination independently of
  marker height;
- retain identical-marker and swapped-marker paired controls.

Record the sampled ranges and combination split in every run report.

## Goal 1 — variable-height visual hover

### Training sequence

Start from `paired-dynamic-001`, not the failed broad-imitation source. Use nominal mass
and nominal thrust-to-weight until hover passes.

Run an initial bounded 200-update experiment. Each update type is sampled in this fixed
mixture:

| Share | Lesson | Purpose |
| ---: | --- | --- |
| 50% | Current-policy closed-loop trajectories with teacher labels | Learn absolute collective calibration, vertical braking and recovery from the states the fly actually visits. |
| 25% | Paired physical-marker contrasts | Preserve the already verified signed visual-height response and anti-shortcut controls. |
| 25% | Dynamic attitude-recovery histories | Prevent throttle learning from destroying roll/pitch stability. |

Generate current-policy prefixes and detach plant transitions between bounded recurrent
windows; do not backpropagate naively through a complete flight. Use 25-policy-step
windows initially, source-relative parameter regularization and normalized per-axis
functional losses. Normalize throttle error by the useful correction range rather than
the much larger steady collective value.

The training-only teacher includes proportional height error and vertical-velocity
damping. The actor never receives vertical velocity: it must infer motion from successive
textured images and its native recurrent state. Include stationary targets, randomized
marker steps and small vertical impulses. Evaluate at updates 0, 50, 100, 150 and 200,
and select checkpoints by fresh closed-loop performance rather than imitation loss.

Continue an update block only while absolute height error improves without breaking the
paired marker-response gate or attitude safety. If current-policy imitation reaches a
stable plateau, recurrent PPO or a small evolution-strategy search may fine-tune the
same native parameters. RL is a fallback for the closed-loop objective, not the first
stage and not permission to add actor state.

Do not add accelerometer, FeCO feedback, randomized mass or attitude-input ablation in
this milestone. Once nominal visual hover passes, FeCO stick-position feedback is the
next controlled addition for action observation and hover-thrust adaptation. It requires
live, constant and pair-shuffled controls.

### Promotion gate

Evaluate 256 fresh nominal-mass episodes over unseen marker/pose/texture combinations,
reported overall and separately for upward and downward marker steps.

| Requirement | Threshold |
| --- | ---: |
| Successful episodes | at least 90% overall and in each step direction |
| Settled altitude RMSE | at most 0.15 m per successful episode |
| Settled vertical-speed RMS | at most 0.20 m/s |
| Roll/pitch tilt RMS | at most 5 degrees |
| Marker-step settling time | at most 3 seconds |
| Ground contact after takeoff | none |
| Sustained stick/actuator saturation | none |

The existing paired marker gate must still pass: contrast NRMSE at most 0.25, at least
95% correct sign, response slope 0.5-1.5, identical-image equality and swapped-image
reversal. Dynamic attitude recovery must also remain passing. A frozen-before-step image
must remove marker-step tracking; it need not make a correctly trimmed aircraft crash.

Only a checkpoint satisfying every requirement is promoted as the Goal 1 source.

## Goal 2 — one annular gate

Use the accepted Goal 1 controller and the same retina, recurrence, foreleg/stick plant,
camera and aircraft dynamics. First validate any teacher through that complete action
path. The gate is physical geometry with collision and aperture-clearance tests, not a
screen-space target or privileged actor input.

Train in this order while retaining hover, paired-marker and attitude-recovery replay:

1. Begin airborne and aligned with a large, nearby gate; learn a slow centered approach
   and continue flying after the plane crossing.
2. Reduce the annulus to its final aperture and require whole-vehicle clearance.
3. Add lateral and vertical offsets, balanced on both sides.
4. Add gate-plane obliquity and the yaw/roll coordination needed for acro control.
5. Restore takeoff and require one uninterrupted takeoff-to-gate flight.

Use a visually unambiguous current-gate colour, a grey textured floor and dark background.
For one gate, no course memory or pass-state input is needed. The passed-gate-dark and
fixed current/next colour sequence belongs to the later multi-gate milestone.

### Promotion gate

Evaluate 1,024 fresh nominal-mass flights, balanced across lateral-offset sign and
obliquity sign.

- At least 90% complete takeoff-to-gate success overall and in every declared stratum.
- The complete vehicle clears the inner aperture without annulus collision.
- The vehicle remains airborne and controlled after crossing.
- No ground recontact, invalid attitude or sustained stick/actuator saturation.
- Frozen vision and role-colour controls demonstrate that the gate image is causal.
- The Goal 1 hover, paired-marker and attitude suites remain passing.

Randomized mass/thrust effectiveness, roll/pitch-input ablation and multiple coloured
gates are later milestones. They must not weaken the nominal single-gate claim or be
silently included in its acceptance result.

## Evidence and artifact policy

Every bounded run writes its full checkpoint and traces below ignored `runs/` paths. A
small promoted or rejected diagnostic report, including configuration, seeds, data
splits, acceptance decisions and hashes, is committed under `artifacts/`. Large binaries
are committed only when required for a reproducible promoted milestone and must use Git
LFS. MaleCNS-derived artifacts retain the CC BY 4.0 attribution and transformation
notice described in [`data/README.md`](../data/README.md).

## Execution log

As of 2026-09-10, the 320×200 renderer, independent marker/camera sampling,
training-only teacher, detached physical DAgger rollout, native recurrent windows and
promotion suites are implemented. The initial mixed curriculum was rejected at update
50: it worsened marker-step hover and crossed the legacy-response safety threshold. See
[`artifacts/variable-height-hover-bounded-v1/`](../artifacts/variable-height-hover-bounded-v1/).

Two response-preserving bridge variants were then tested from the unchanged source. V1
reserved opposite wall/floor style combinations and failed to improve their response.
V2 balanced all four style combinations and added fixed training-support and fresh-height
matrices. At update 25, v2 improved training support only 3.57% while closed-loop altitude
RMSE regressed 27.4%, so its safety gate stopped training. Neither candidate was promoted;
the original source remains the selected controller. See
[`bridge-v1`](../artifacts/variable-height-visual-response-bridge-v1/) and
[`bridge-v2`](../artifacts/variable-height-visual-response-bridge-v2/).

The lesson-gradient and visual-signal propagation audit is now complete. Marker-height
information survives from the retina through central-brain and VNC populations to the
26 foreleg motor neurons. The rejected bridge did not fail because the marker signal
vanished: its principal failure was a source-global shift in common collective. At the
unchanged source, the squared common-throttle and replay-preservation losses have zero
first derivative, so they cannot oppose the first contrast-learning step. A family-RMS
parameter step of only `2e-5` improved the fixed visual contrast while already producing
0.482 normalized common-throttle error. See
[`artifacts/variable-height-bridge-gradient-signal-audit-v1/`](../artifacts/variable-height-bridge-gradient-signal-audit-v1/).

The authorized next experiment is therefore a bounded projected-Adam feasibility test,
not another unconstrained bridge. It retains the v2 all-style lessons and actor contract,
but projects each proposed optimizer displacement in an equal-family-RMS metric against
signed, per-style small- and medium-marker common-throttle Jacobian rows. It then applies
one scalar step cap and at most six backtracking trials. Every accepted update must pass
complete replay from zero; a fixed-burn-in Jacobian alone is never acceptance evidence.

Per-update family RMS is capped at `2e-5`, source-global family RMS at `1e-4`, normalized
common-throttle RMS at 0.02 per update and 0.05 from source, and maximum absolute
source-global common-throttle drift at 0.005 motor units. Legacy response, dynamic R/P/Y
and validity guards remain in force. The run is capped at 25 attempted updates and stops
after five consecutive rejected proposals. At update 25 it must improve the fixed
training-support contrast by at least 10% and a separate development-height matrix by at
least 5%, while marker-flight RMSE remains within 5% of source with no new ground or
invalid events. Passing establishes only safe local plasticity; it does not promote a
hover controller or consume the final held-out evidence.

That projected-Adam test is complete and rejected. Its preflight confirmed that the
Jacobian projection works: ordinary capped Adam broke five functional guards, whereas a
projected half step passed complete replay. In the endpoint-retaining replay, 12 of 22
proposals were accepted before five consecutive rejections stopped it. The fixed-bank
contrast NRMSE improved only 0.129%, from 0.738583 to 0.737629, while the source-family
radius reached `9.999e-5` of `1e-4`. No controller was promoted. See
[`artifacts/variable-height-projected-trust-region-v1/`](../artifacts/variable-height-projected-trust-region-v1/).

The next and final local-optimization diagnostic replaces minibatch Adam proposals with
a fixed, style/amplitude-balanced contrast gradient. It solves a small constrained
descent problem whose span contains that gradient, the common-output Jacobian rows and
the displacement from source, so source/step radii participate in direction selection
rather than merely rejecting outward proposals afterward. It keeps every v3 functional
limit and the same 25-attempt, 10% training-support and 5% development gates. A cheap
preflight at the source and v3 boundary must show a measurable feasible contrast direction
before training. Failure ends this local constrained family and triggers the already
documented responsibility-based native-anatomy/teacher-handoff tracks; it is not evidence
against visual information or native recurrence.

That final local preflight also stopped correctly. Fixed-bank constrained descent at the
untouched source retained 45.5% of the contrast descent and produced a safe 0.004073
NRMSE improvement in one quarter step. At the actual v3 boundary, every scale still
improved contrast, but none passed a new complete-replay bank: small-pair maximum common
drift remained 0.00578–0.00808 motor units against the 0.005 limit. Training did not
start and no parameters changed. See
[`artifacts/variable-height-constraint-aware-descent-v1/`](../artifacts/variable-height-constraint-aware-descent-v1/).

The global local-optimization family is now closed. The next bounded milestone returns to
the two control-theory-derived tracks in [`CONTROL_ARCHITECTURE.md`](CONTROL_ARCHITECTURE.md):
first audit where the native state represents visual height error, visual vertical motion
and tonic collective across independent trajectory banks; then train only supported
native routes under randomized progressive teacher handoff. Teacher control, decoded
probe values and phase remain training-only. The deployed actor still receives only RGB
and estimated roll/pitch and still drives all four axes through the two front legs.

### Preregistered native control-responsibility audit

The next run is a diagnostic-only, three-assay audit of the unchanged source. Current
height error uses a common recurrent prefix and identical vehicle pose while the physical
marker differs by plus or minus 5 or 10 cm. Vertical motion uses mirrored, smooth
camera-height histories ending at exactly the same pose and retinal image with endpoint
speeds of plus or minus 0.15 or 0.30 m/s. Self-generated command history uses those same
marker amplitudes to elicit different native throttle commands, executes the existing
foreleg and aircraft actuator plants in shadow, then supplies 0.26 seconds of identical
RGB and attitude. The last assay tests native efference/history retention; it is not a
hover-thrust estimate and does not supply leg state to the actor.

Each assay uses 64 development pairs and 128 unseen-style test pairs. Train-only sparse
linear probes may select at most 26 neurons in each disjoint anatomical partition, equal
to the complete motor interface size. Selection, scaling and ridge choice use development
data only; evaluation uses the independent bank and includes a separately selected
shuffled-label control. Representation requires test R2 at least 0.50, sign accuracy at
least 90%, and a paired-bootstrap R2 improvement over the constant predictor whose lower
95% bound is above zero. Endpoint retinal contrast must be exactly zero for the two
history assays.

Representation alone does not assign a control role. On a separate 16-pair causal bank,
each population is replaced once by its paired mean at the shared endpoint, after which
the untouched graph runs for 0.20 seconds. A native response is usable only if it has the
correct sign in at least 90% of pairs and reaches 10% of the conventional teacher's motor
contrast (or 10% retention of the prior native command). A partition is selectively
causal only if replacement removes at least 30% of that response with a paired 95%
confidence interval excluding zero while changing common motor output by at most 0.0025.
Identical-history and whole-state replacements check the intervention implementation.

If error and motion are represented and selectively used, proceed to anatomy-restricted
progressive teacher handoff. If they are represented but not used, train native routing
and readout without rebuilding the graph. A failed linear probe first triggers checks of
temporal retinal drive, saturation and probe sensitivity. FeCO becomes a controlled later
test only if physical leg state proves necessary; it is not added to rescue nominal hover.

That audit is complete and selects the second branch. Every internal partition decoded
height correction on the unseen-style bank with R2 0.964-0.989. More importantly, every
partition decoded vertical damping with R2 0.898-0.933 even though the endpoints had
identical pose and retinal input. The native output nevertheless used that history in the
wrong direction: its motion contrast was wrong-signed in all 128 pairs and measured
-0.625 of the teacher damping contrast. Height response remained correctly signed in all
pairs at 0.910 of teacher. No coarse partition passed the fixed selective-causality gate.
See
[`artifacts/variable-height-native-control-responsibility-audit-v1/`](../artifacts/variable-height-native-control-responsibility-audit-v1/).

The command-history time course supplies a concrete control explanation for throttle
hunting. After 0.26 seconds of identical input the native output retained 46.3% of the
prior command with the correct sign, but over the next 0.20 seconds it crossed zero and
reversed. This is useful recurrence, not a stable hover-thrust estimator. It neither
requires FeCO now nor rules out a later live/stale/shuffled FeCO experiment.

The next bounded experiment is native damping-route training, not stronger height
feedback and not progressive handoff yet. Before optimization, enumerate and freeze an
exact mask of existing at-most-two-synapse paths from descending or VNC interneurons to
the throttle antagonist motor pools; no other motor neuron may be an intermediate. Open
only those edge magnitudes and nonmotor intermediate biases. Keep motor biases, all time
constants, topology, transmitter signs and every other parameter fixed.

The preflight must differentiate the real recurrent unroll and show a finite-difference-
confirmed descent direction for the paired motion-contrast loss while full zero-state
replay preserves pair-common throttle, visual height response and dynamic R/P/Y. Do not
train a persistent velocity label under a frozen image. If feasible, permit at most 50
attempted balanced constraint-aware updates. Stop at update 25 unless motion-contrast
NRMSE falls at least 25%. A replay pass requires at least 90% correct damping sign and
teacher-aligned gain 0.5-1.5 on fresh motion pairs, height contrast within 10% of source,
source-global common-throttle RMS at most 0.0025, maximum drift at most 0.005 and all
existing attitude guards. This earns only a small nominal-mass closed-loop hover test;
checkpoint promotion still requires the full Goal 1 gate.
