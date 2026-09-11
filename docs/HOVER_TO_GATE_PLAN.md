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

The deterministic v1 route mask contains 12,314 edge magnitudes and 303 nonmotor
intermediate biases. Its selected-edge-index SHA-256 is
`cd0a11e986d30701cb3b08d4b404f253476ce8044cc250e0b01a10dff1c449a7`; its selected
bias-index SHA-256 is
`0b518b0411d8e39b9f4c56cfc8b68e057814a561ca4268ae16cf67cea777b593`. The mask
metric is the square root of mean selected-edge displacement squared plus mean
selected-bias displacement squared. Preflight steps cap each selected family at `2e-5`;
later optimization, if authorized by preflight, has a source-metric radius of `5e-4`.
The preflight tries the fixed scales 1, 1/2, 1/4, 1/8, 1/16 and 1/32 and requires at
least `1e-4` fixed-scale motion NRMSE improvement in addition to every functional guard.

That shallow-route preflight is complete and rejected. The target had 0.567 motor-unit
headroom, and the projected direction was both descending and finite-difference verified.
All functional checks passed at scale 1, including common-throttle RMS `4.57e-6`, maximum
common drift `8.73e-6` and height-response ratios within `5e-5` of source. Exact
common-output projection nevertheless shrank the selected-edge RMS from `2e-5` to
`2.657e-6`, and motion NRMSE improved only `3.016e-5` against the preregistered `1e-4`
floor. Damping sign remained wrong in every complete-replay sample. No update was retained
and the 50-update run did not start. See
[`artifacts/variable-height-native-damping-route-preflight-v1/`](../artifacts/variable-height-native-damping-route-preflight-v1/).

One revised preflight is authorized on the identical mask with identical loss and
acceptance thresholds. V1 imposed pointwise linearized equality on every pair-common
output, then left its projected direction far below the final step cap. V2 instead treats
common output as the bounded functional quantity it is: per-update RMS at most 0.001,
source-global RMS at most 0.0025 and maximum absolute drift at most 0.005 native motor
units. It constructs the raw equal-metric descent and the equality-projected direction
rescaled to the final `2e-5` selected-family cap, screens their fixed 0%, 25%, 50%, 75%
and 100% blends against the linearized inequalities and parameter bounds, then gives the
best predicted admissible direction the same full-replay and backtracking test. The
rescaled equality direction is always replayed as a sanity control.

This does not revise v1's rejection or lower its `1e-4` improvement threshold. V2 opens
no new parameters: the route hashes, frozen time constants and motor biases are
unchanged. A pass authorizes the already documented bounded training run; it does not
show that one step reversed damping or promote a controller. Failure closes this shallow
route/constraint family rather than triggering further post-hoc relaxation.

V2 passed. The best admissible direction was the raw selected-metric descent at the full
`2e-5` edge-RMS cap. Its predicted loss change was -0.006913 and finite replay measured
-0.006901. Motion NRMSE improved 0.002382, from 1.449651 to 1.447269. Full-replay
common-throttle RMS was 0.000426, maximum drift 0.000546, and both height contrasts
retained 0.9957 of source; all legacy and dynamic checks passed. The equality-rescaled
sanity direction passed separately. Parameters were restored exactly, and damping was
still wrong-signed after this single step, so this is authorization for the bounded run,
not a controller result. See
[`artifacts/variable-height-native-damping-route-bounded-preflight-v1/`](../artifacts/variable-height-native-damping-route-bounded-preflight-v1/).

The authorized v1 training run is resumable and capped at 50 attempted updates. Each
attempt averages gradients from two independently seeded balanced eight-pair motion
banks. It constructs the same inequality-screened direction family, then accepts a
backtracked step only if **each** of two separate fixed guard banks improves by at least
`1e-4` NRMSE and complete zero-state height/common, legacy and dynamic R/P/Y replay
passes. Nonfinite metrics or parameter displacements reject the step. The
selected-family step cap remains `2e-5`, the selected source-metric radius remains
`5e-4`, and every common-output bound from the v2 preflight remains unchanged. Five
consecutive rejected attempts stop the run.

Attempt 25 uses 64 fixed held-out milestone pairs (seed offset `+60000`). Unless their
motion NRMSE has fallen at least 25% from the unchanged source, training stops even if
every tiny step was locally valid. Terminal qualification uses a disjoint 64-pair cohort
(seed offset `+70000`): a candidate may advance to the small nominal-mass closed-loop
hover test only with at least 90% correct damping sign, teacher-aligned gain 0.5-1.5 and
every preservation gate still passing. Atomic resume state includes the complete
protocol and terminal decisions, preventing a restart from changing or bypassing gates.
The run's endpoint remains explicitly nonpromotional; only subsequent closed-loop
evidence can justify replacing the source.

Result: the frozen run in commit `6673469` stopped at the attempt-25 gate after 16
accepted updates. Milestone motion NRMSE improved only 1.117% (1.365113 to 1.349864),
far below 25%. On the disjoint terminal cohort it improved 1.168% (1.395011 to
1.378718), while correct damping sign remained 0% and aligned gain remained negative at
-0.552. Preservation passed, including height-response ratios near 0.970, but common-
throttle RMS reached 0.002491 of the 0.0025 source limit. This exact shallow route is
closed as insufficient: it weakens the wrong-signed response slightly but cannot reverse
it without consuming the tonic/height-control budget. No closed-loop test ran and no
checkpoint was promoted. See
[`artifacts/variable-height-native-damping-route-training-v1/`](../artifacts/variable-height-native-damping-route-training-v1/).

The next preregistered test restarts from the unchanged source and expands credit
assignment one targeted stage upstream. Let `I` be the 303 shallow-route intermediates
and `T` the seven throttle motors. The mask selects the 612 `descending_neuron` cells
with an existing output to `I` or `T`, then adds every existing synaptic edge ending at
those cells and their native biases. Deduplication produces 80,454 edge magnitudes and
883 biases: edge-index SHA-256
`a8d6118684cf64850b9e2d08f5f200c4501198fef6d4f5c52ec3aa349afa5aa3` and bias-index
SHA-256 `f2cedb5b38e80beb9c29bba190ac9881c4cc473ea2849329b1f449c80402bd9b`.
All presynaptic superclasses are admitted on those native afferents, so existing
optic/visual-projection-to-descending magnitudes may change; retinal mapping/input gains,
optic-neuron biases, all time constants, motor biases, signs and topology remain frozen.

To prevent the larger mask from silently receiving a larger trust region, edge and bias
RMS retain the shallow denominators 12,314 and 303. The one-step preflight otherwise
keeps the same damping bank, bounded common/height/attitude constraints, `2e-5` step cap,
`5e-4` later-training radius and backtracking sequence. Before a pass it additionally
requires a forward finite-difference directional derivative at scale 0.0625 to be finite,
negative and within 20% of autograd. Both the differentiated fixed-source-prefix bank and
a separately seeded complete zero-state native-recurrence bank must improve motion NRMSE
by at least `1e-4`, with every preservation gate passing. Improvement per common-output
drift is reported beside the same-seed shallow preflight. This tests selective
descending-signal recoding, not a claimed biological damping module; optimization beyond
one restored preflight step is not authorized until it passes.

Result: the upstream preflight passed at full scale and restored all parameters exactly.
Autograd and the scale-0.0625 finite difference agreed within 0.389%. Fixed-prefix and
complete zero-state motion NRMSE improved by 0.002654 and 0.002219, respectively, while
common-throttle RMS remained within bounds at 0.000577 and height contrast retained
about 0.9953 of source. Against the identical shallow fixed-prefix preflight, raw NRMSE
improvement was 11.4% larger but common drift was 35.5% larger, so improvement per unit
common RMS was worse (4.60 versus 5.59). This passes the local credit-assignment gates
but does not yet show better damping/collective separation. It authorizes only a
separately preregistered bounded training run; no controller was retained or promoted.
See
[`artifacts/variable-height-native-upstream-damping-route-preflight-v1/`](../artifacts/variable-height-native-upstream-damping-route-preflight-v1/).

The authorized upstream-training v1 keeps that exact mask and again restarts from the
preserved source. Its fixed denominators 12,314/303 govern not only reported step and
source caps but also the Riesz descent direction and common-output Jacobian projection.
This prevents the 80,454-edge/883-bias expansion from acquiring a larger or different
optimization metric. All frozen parameter families and all common/height/attitude limits
remain unchanged.

For direct comparison with shallow training, the schedule remains at most 50 attempts,
two independently seeded balanced eight-pair native-recurrence gradient banks per
attempt, and two fixed eight-pair native guard banks that must **each** improve by at
least `1e-4`. Five consecutive rejections stop the run. Attempt 25 still requires at
least 25% motion-NRMSE improvement on 64 held-out pairs. The new base seed `290941`
makes its milestone (`350941` onward) and terminal (`360941` onward) cohorts previously
unconsumed; terminal qualification still requires at least 90% correct sign, aligned
gain 0.5-1.5 and every preservation gate. Per-step improvement/common-RMS efficiency and
complete-native gain are diagnostic only, not post-hoc stopping criteria. Resume state
atomically freezes all protocol fields and terminal decisions. The endpoint remains
nonpromotional until a qualifying replay is followed by a successful small closed-loop
hover test.

Result: upstream training stopped at attempt 21 after five consecutive inadmissible or
non-improving proposals; ten updates were accepted. On its previously unconsumed
terminal cohort, motion NRMSE improved only 1.023% (1.380882 to 1.366758), correct sign
remained 0%, and aligned gain remained negative at -0.552. Preservation passed and
height responses retained about 0.973 of source, while common-throttle RMS reached
0.002432 of its 0.0025 limit and the fixed-denominator parameter metric used only
0.000134 of its 0.0005 radius. In the final screens, equality-heavy directions preserved
common output but increased damping loss; damping-descent directions exceeded the common
budget. This confirms that the expanded subspace still does not separate damping from
collective control under source preservation. No checkpoint was promoted and no
closed-loop test ran. Further anatomical expansion under this constraint is paused; the
next test must change the control decomposition. See
[`artifacts/variable-height-native-upstream-damping-route-training-v1/`](../artifacts/variable-height-native-upstream-damping-route-training-v1/).

Before changing the target or adding another anatomical region, common-anchor ablation
v1 tests the inferred bottleneck directly. It restarts from the original preserved
source and keeps the exact 80,454-edge/883-bias expanded mask, fixed 12,314/303 metric,
raw native-recurrence gradient banks, step/source radii, edge bounds and frozen parameter
families. The sole intervention is removal of every absolute-throttle-to-source
constraint: there is no common-output projection or screening, no common RMS/maximum
gate, and throttle is excluded from dynamic and legacy source preservation. Absolute
source-common drift and analytical-teacher throttle error are still recorded throughout
but are neither optimized nor gated.

Height contrast must remain within 10% of source. Dynamic roll, pitch and yaw must each
remain within the existing 0.05 normalized error. Legacy preservation drops throttle
from its old numerator without tightening the denominator:
`sqrt((roll² + pitch² + yaw²) / 4) <= 0.05`. Paired R/P/Y stays diagnostic-only, matching
the completed anchored runs. All measurements must be finite, every replay valid, motor
outputs within [-1, 1], each of the two fixed eight-pair guard banks improved by at least
`1e-4`, and the fixed parameter caps satisfied.

For a controlled paired comparison, the ablation reuses base seed `290941`, the same
gradient and guard cases, and the anchored run's terminal cohort (`360941` onward),
explicitly labelled a reused benchmark. It stops after at most 25 attempts or five
consecutive rejections and requires at least 25% benchmark motion-NRMSE improvement for
useful progress. Only if that gate passes is a previously unconsumed 64-pair cohort
(`370941` onward) evaluated for at least 90% correct damping sign and aligned gain
0.5-1.5. This is diagnostic only: no endpoint is promoted and no closed-loop handoff is
automatic.

Result: the ablation accepted 24 of 25 updates and allowed source-pair common-throttle
RMS to reach 0.011228, 4.49 times the removed 0.0025 limit. Motion NRMSE on the reused
benchmark still improved only 3.552% (1.380882 to 1.331832), damping sign remained 0/64,
and aligned gain remained negative at -0.5133. The run stopped at its attempt limit;
every attempt-25 scale violated the 0.9 minimum height-response ratio. The retained
small/medium ratios were only 0.90202/0.90013, and the source-relative parameter metric
reached 0.000437 of its 0.0005 radius. Thus the common anchor was operationally binding,
but removing it was insufficient by a wide margin: it is not the sole cause of the
wrong-sign damping failure. The useful-progress gate failed, so the fresh cohort was not
exposed, no checkpoint was promoted, and no closed-loop test ran. See
[`artifacts/variable-height-common-anchor-ablation-v1/`](../artifacts/variable-height-common-anchor-ablation-v1/).

The similar terminal attenuation of motion gain (to about 90.5% of source magnitude)
and visual-height response (to 90.0-90.2% of source) suggests a more specific failure:
the optimizer may merely weaken the existing position-feedback response and its delayed,
wrong-signed echo rather than learn an independent damping term. Because those numbers
come from different banks, this is a hypothesis to test, not a result to assume.

The next bounded test keeps the source initialization, expanded mask, fixed metric and
frozen parameter families, but replaces the paired-motion lesson with a matched 2x2
height-by-velocity assay. For each independently randomized scene and each balanced
amplitude pair (`|e|` in 0.05/0.10 m and `|v|` in 0.15/0.30 m/s), four smooth native
histories cross signed endpoint visual height error and endpoint vertical velocity.
Opposite-velocity histories within a height condition end at exactly the same pose and
image. Training, guard and evaluation cohorts use disjoint scene/trajectory combinations
and varied approach durations.

The four endpoint throttle outputs are decomposed into a transient marker-error `P`,
velocity `D`, common `C`, and height-by-velocity interaction component; the existing
analytical teacher supplies the corresponding training-only targets. `P` is only a
factorial name here, not a claim that the native response is a steady-state proportional
gain. Optimization descends only the normalized `D` error after projecting against the
matched `P` Jacobian from the **same scenes and histories** in the fixed source metric.
`C` remains unanchored and diagnostic, while the interaction is reported so cancellation
cannot masquerade as separation. Every candidate is checked by complete zero-state
replay, not accepted from the linear projection alone. The actor still sees only RGB and
roll/pitch and uses only native recurrence.

First run a restored one-step preflight. It must have a finite negative derivative,
finite-difference agreement within 20%, at least `1e-4` actual full-replay `D`-NRMSE
improvement, and matched `P` response within 10% of source in **every nondegenerate
scene**, not merely in the amplitude-group mean. Bound-active edge coordinates are frozen
and the post-bound linearized `P` change must remain at most 0.1% of source per scene.
All existing visual-height, RPY, validity, motor-bound and `5e-4` source-metric checks
also remain. A pass authorizes at most 25 attempts with two independent gradient banks
and two fixed guard banks; each guard bank must improve `D` NRMSE by at least `1e-4`, and
five consecutive rejections stop. At the attempt limit, useful progress requires at least
25% `D`-error reduction on an independent development cohort while retaining `P` within
10% of source and all other protections. Only then is a fresh cohort exposed for at least
90% correct damping sign and aligned gain 0.5-1.5. Replay alone cannot promote a
checkpoint. If the height-null preflight has no usable damping descent, or bounded
training again misses useful progress, close this local mask/metric family rather than
interpreting reduced wrong-sign amplitude as learned damping. Joint absolute
height-and-damping teaching would then be a separately declared broader redesign, not a
post-hoc continuation.

Result: the restored preflight passed at full scale. Its same-case objective/projection
replays agreed on the source `P` component within `3.73e-8`; bound-aware projection froze
696 edge coordinates and left a maximum linearized `P` change of only `3.56e-9` of source.
The projected damping derivative was -0.00021962 and its finite difference was
-0.00021124 (3.82% relative error). Actual damping NRMSE improved by 0.0001865 on the
fixed-prefix cases and 0.0001498 on the separately seeded complete-zero-state cases.
Every per-scene `P` ratio stayed within 0.99955-1.00032, existing height response retained
at least 0.99910 of source, and all non-throttle protections passed. Parameters were
restored exactly; no checkpoint or closed-loop test resulted. See
[`artifacts/variable-height-factorial-damping-route-preflight-v1/`](../artifacts/variable-height-factorial-damping-route-preflight-v1/).

The authorized bounded run uses new base seed `310949`. At each of at most 25 attempts,
two independently seeded four-scene factorial banks produce the averaged complete-native
`D` gradient and all eight same-case `P` rows. The bound-aware height-null direction is
accepted only if each of two fixed guard banks (`seed + 40000`) improves `D` NRMSE by at
least `1e-4`; every scene in those banks retains `P` within 10% of the unchanged source,
and all existing height, RPY, validity, motor-bound, `2e-5` step-family and `5e-4`
source-metric protections pass. Common throttle and the interaction component remain
diagnostic only. Five consecutive rejections stop the run; its complete protocol and
resume state are atomic.

At attempt 25, the independent 64-scene/256-history development cohort beginning at seed
`370949` must reduce factorial `D` NRMSE by at least 25% from source while retaining every
protection. Only then may the equally sized fresh cohort beginning at `380949` be exposed;
it requires at least 90% correct damping sign and teacher-aligned gain 0.5-1.5. Even a
fresh replay pass authorizes only a small nominal-mass closed-loop hover test and does not
itself promote a controller.

Result: all 25 attempted updates were accepted at full scale, showing that the same-case
height-null construction remains feasible over the complete bounded run. On the
independent development cohort, every per-scene `P` ratio remained in 0.98100-1.00578
and the existing small/medium height assays retained 0.98155/0.98207 of source. All RPY,
validity and motor-bound checks passed. Yet `D` NRMSE improved only 0.742% (0.594106 to
0.589698) against the 25% gate; damping sign stayed 0/64 and aligned gain only moved from
-0.3393 to -0.3300. The fixed source metric reached 0.0004214 of 0.0005, while unanchored
common-throttle drift reached 0.005645 RMS. Thus clean P/D separation prevents the prior
height-attenuation confound, but this local mask/metric family did not demonstrate useful
damping authority. It found safe local descent without producing a correctly signed
braking response; this is not evidence that the unrestricted native graph lacks one. The
fresh cohort was not exposed, no checkpoint was promoted, and no closed-loop test ran.
This family is now closed rather than extended or relaxed. See
[`artifacts/variable-height-factorial-damping-route-training-v1/`](../artifacts/variable-height-factorial-damping-route-training-v1/).

The next experiment is therefore a restored, full-native **learnability preflight**, not
another route expansion and not a hover-training run. It starts again from
`paired-dynamic-001`. Every existing native edge magnitude, every native bias including
the motor-pool biases, and every native time constant may change; topology, transmitter
signs, the retinal and attitude mappings, sensory gains, actor inputs and foreleg outputs
remain fixed. Edge magnitudes retain their [0, 8] projection and time constants retain
the controller's intrinsic 10-250 ms parameterization. There is no local route mask,
source-distance ball, height-null projection, added state or privileged actor input.

The preflight freezes eight training scenes at seed `320953` and eight disjoint
development scenes at `330953`; their separately cached dynamic-attitude banks use seeds
`320954` and `330954`. Each factorial scene has the same four signed height-by-velocity
branches and balanced magnitudes used by the factorial audit: 0.05/0.10 m marker error,
0.15/0.30 m/s vertical speed, and approach durations 15/18/22/25 policy steps. A neutral
five-step common prefix and a 25-step response are both differentiated from zero native
state; no source-generated neural state is injected or detached. Labels are applied at
one-indexed response steps 15, 20 and 25. At each label, the analytical teacher receives
the actual instantaneous camera height and the exact derivative of the complete
prescribed smooth trajectory, including its sinusoidal term and inactive interval; the
fly still receives only the rendered image and roll/pitch. Rendered inputs, attitudes and
labels are cached once and reused byte-for-byte by autograd, finite differences and
replay. The endpoint opposite-motion branches must again have bit-identical pose and
image.

At every labelled time, the four throttle outputs and teacher targets are Hadamard-
decomposed into common collective `C`, transient marker response `P`, visual damping `D`
and height-by-velocity interaction `I`. `D` is a bookkeeping name at intermediate times:
because opposite-motion branches then occupy different heights, their velocity-odd term
contains both position and velocity effects. Literal damping sign and aligned gain are
therefore scored only at the step-25 identical-image endpoint. `C` and `I` use fixed
motor-unit scales 0.05 and 0.01. `P` and `D` each use the RMS of their analytical-teacher
targets over the complete frozen training bank, floored at 0.01 motor units; those two
values are frozen before optimization and reused unchanged on development data. Three
dynamic attitude lessons replay the unchanged source's roll, pitch and yaw outputs at
scales 0.05, 0.05 and 0.04. The joint objective is the unweighted mean of these seven
separately normalized MSEs, with the three labelled horizons weighted equally and also
reported separately. Source replay is used for RPY because the source already passed the
attitude guard; no analytical attitude controller or privileged rate enters the actor or
target. Scenes are processed in deterministic microbatches and gradients are accumulated,
so batching cannot change the declared objective or turn GPU memory into an experimental
variable.

Before a learnability pass, teacher-against-teacher component error must be numerically
zero after a measured Hadamard reconstruction, source replay must be deterministic to
`1e-6`, and all labels and outputs must be finite. Every labelled analytical-teacher
command is also held for one second at the 100 Hz physics rate through the actual
foreleg/stick plant. Its measured RC must be within 0.02 of the target, remain bounded,
and retain the intended endpoint response signs. This checks command units, signs and
steady-state reachability, not time-varying tracking or closed-loop stability. The actual
proposed update must then pass a directional check. That update is one clipped Adam step
(global gradient-norm cap 1.0; edge/bias learning rate `1e-4`; raw-time-constant learning
rate `1e-6`; no weight decay). After edge-bound projection, its displacement is the
tested direction. A forward finite difference on the cached training bank at scale
0.0625 must be negative and agree with the autograd directional derivative within 20%.

The full step must reduce joint normalized MSE by at least `1e-4` on the fixed training
scenes. Each of training `C`, `P` and `D` NRMSE may be at most its own source baseline
plus 0.02; this permits a first joint step to trade a small component error without
mistaking unconstrained regression for learnability. Each dynamic RPY source NRMSE must
remain at most 0.05, and all outputs must remain finite and within [-1, 1]. The disjoint
development bank must independently reduce joint objective, with the same baseline-plus-
0.02 C/P/D, RPY, validity and motor-bound guards. Every parameter is then restored bit-
exactly; a pass retains no controller and authorizes only a separately preregistered,
at-most-200-update joint teacher-fitting diagnostic. That later diagnostic will have a
mandatory update-50 stop unless training endpoint `D` error has fallen at least 25% and
C/P have recovered to no worse than their source fidelity while RPY remains passing. It
will require disjoint-development damping sign, gain and component-error gates before any
small closed-loop teacher-handoff test. All fitting checkpoints remain nonpromotional.
Failure of this preflight sends work to the learning dynamics/parameterization audit,
not to more sensors, RL, a looser threshold or a post-hoc narrower claim.

Implementation smoke: after the complete protocol and code were frozen in commit
`63f4614`, a reduced four-scene run used the registered seed numbers and therefore
exposed them; the eventual eight-scene banks must not be described as unseen. No
threshold or optimizer setting changed afterward. The smoke passed cache identity,
determinism, finite-difference, teacher/foreleg and exact-restoration checks and reduced
joint MSE on both reduced banks, but failed the 0.05 RPY source-replay gate (pitch about
0.088 on both). This predicts a likely rejection of the exact first step without
prejudging the committed full run. It rejects neither full-native learnability nor a
different optimizer. No parameters were retained. See
[`artifacts/variable-height-full-native-joint-preflight-smoke-v1/`](../artifacts/variable-height-full-native-joint-preflight-smoke-v1/).

Result: the formal eight-scene preflight reproduced that failure. The actual displacement
was a valid descent direction (2.39% finite-difference error), and joint normalized MSE
fell 66.1% on training and 67.8% on the disjoint development bank. Most of the gain was
absolute collective calibration: `C` NRMSE fell from 2.383 to 0.937 and 2.696 to 1.145.
`P` improved slightly, while endpoint `D` remained wrong-signed in all eight scenes and
barely changed. The update failed only the RPY safety gate: training roll/pitch source
NRMSE reached 0.0519/0.0866 and development reached 0.0540/0.0859, versus 0.05. This
rejects the exact unprotected full-native Adam step, not joint learnability: the source-
replay loss is zero at the source and cannot shape the first displacement at first order.
All identity, teacher/foreleg, determinism, bound and exact-restoration checks passed.
No parameters were retained and no closed-loop test ran. See
[`artifacts/variable-height-full-native-joint-preflight-v1/`](../artifacts/variable-height-full-native-joint-preflight-v1/).

The next bounded learning-dynamics audit does not revise or rerun that failed preflight.
It restarts from the same source and deliberately reuses its exposed training and
development caches, teacher scales and regenerated Adam displacement. Before any new
claim, cache hashes, source replay, the full-scale displacement, objective and derivative
audit must reproduce the frozen result within reported numerical tolerance.

On the training bank, replay the **complete** displacement at the fixed scales 1, 1/2,
1/4 and 1/8, plus the unchanged source. Separately replay the edge-only, bias-only and
raw-time-constant-only portions at scale 1 with both other families restored. Those
single-family probes are diagnostic only and cannot become candidates or be combined
post hoc; their effects are not assumed to add linearly. Every replay reports C/P/D/I,
each RPY axis, endpoint damping sign/gain, motor bounds and actual family displacement.

A complete-displacement scale is training-feasible only if it retains the failed
preflight's exact gates: joint normalized MSE improves by at least `1e-4`; C, P and D
NRMSE are each no more than source plus 0.02; each RPY source NRMSE is at most 0.05; all
cached states and outputs are finite and valid; and motor outputs stay within [-1, 1].
The full direction must again be negative and pass the scale-0.0625 finite-difference
agreement limit of 20%. Select the largest training-feasible full scale, with no look at
development while choosing it, then evaluate that one scale only on development under
the same gates except that any positive joint-objective improvement suffices. A reported
D improvement is measurable only if its fixed-scale NRMSE reduction is at least `1e-4`,
above the replay floor.

If the selected scale passes both banks, the result establishes step-size overshoot for
this one update; it does not establish absence of longer-term task conflict. Only a pass
with measurable D-error reduction on **both** banks authorizes a separately preregistered
joint-fitting run using that step-control rule and the mandatory update-50 endpoint-D
improvement gate. If no scale passes, family probes guide a later individual-output RPY-
Jacobian projection protocol. If a scale passes without measurable D improvement, report
safe collective calibration only and do not automatically proceed to multi-step fitting.
If the selected scale fails development, stop without trying a smaller scale there. No
audit endpoint is retained or promoted and no closed-loop test follows directly.

Result: the run is formally an audit-control failure. Both factorial-cache hashes and
all registered scalar controls reproduced within tolerance, and source replay agreed to
`5.96e-8`, but independently regenerated source-driven CUDA attitude trajectories did
not reproduce their frozen byte hashes. The failed exact-hash gate is preserved rather
than revised after the result. Counterfactually applying the substantive gates, scale
1/2 was the largest training-feasible update and passed development safety. It reduced
the collective-dominated joint loss by 0.4582 on training and 0.5375 on development while
keeping RPY below 0.05. However, damping NRMSE improved by just 0.000263 on training and
worsened by 0.000466 on development; endpoint damping remained wrong-signed in every
scene. Edge-only updates supplied most of the collective gain but broke pitch safety,
bias-only changes were safe with only a tiny damping effect, and time-constant changes
were negligible at this learning rate. These family probes remain nonselective and
nonadditive diagnostics. No candidate was retained, no multi-step fitting was authorized,
and no closed-loop test ran. See
[`artifacts/variable-height-full-native-step-family-audit-v1/`](../artifacts/variable-height-full-native-step-family-audit-v1/).

The next experiment is a fresh-cache, full-native **endpoint-damping-only step audit**.
It tests the specific hypothesis that the previous joint gradient neglected visual
damping because common collective calibration dominated its normalized objective. It is
not a training run and does not revise or rerun the failed step-family audit. One
training bank at seed `340961` (attitude seed `340962`) and one disjoint development
bank at seed `350961` (attitude seed `350962`) are generated once, persisted as immutable
tensors below the ignored run directory, hashed, reloaded, and then reused exactly for
source replay, autograd, finite differences and every candidate evaluation.
Independently regenerated CUDA rollouts are not required to be byte-identical; the
protocol requires exact reuse of the persisted caches and numerical source replay within
`1e-6`.

Starting from `paired-dynamic-001`, all native edge magnitudes, biases and raw time
constants remain open under the same Adam learning rates, global norm cap and parameter
bounds as the joint preflight. Topology, transmitter signs, sensory mappings and gains,
actor inputs and front-leg output pools remain fixed. Native recurrence starts at zero
and the entire five-step prefix and 25-step response are differentiated. The objective
contains only endpoint `D`: the teacher-normalized squared error of the velocity-odd
throttle component at step 25, where opposite-motion branches have the exact same pose
and image. Its normalization is the RMS endpoint-D teacher target on the frozen training
bank, floored at 0.01 motor units, and is reused unchanged for development. Joint loss
and all intermediate-horizon components remain reporting and preservation metrics, not
optimization terms.

The actual bound-projected Adam displacement must have a negative endpoint-D directional
derivative and a scale-0.0625 finite difference agreeing within 20%. Replay its complete
displacement on training at the unchanged fixed descending scales 1, 1/2, 1/4 and 1/8.
Select the largest scale that improves endpoint-D NRMSE by at least 0.001 while keeping
aggregate C and P NRMSE, and each one's NRMSE at every labelled horizon, no worse than
their own source values plus 0.02; every RPY source NRMSE must remain at most 0.05, all
values finite and outputs within [-1, 1]. The interaction component and joint loss are
reported but do not gate the damping-specific claim. Evaluate only that selected scale
on development, with the same 0.001 endpoint-D improvement and preservation gates; do
not retry a smaller scale after development failure.

Every parameter is restored bit-exactly and no endpoint is retained. A pass shows only
that a safe, transferable first-order damping-directed step exists and authorizes a
separately preregistered bounded D-first fitting diagnostic. It is not a damping-capacity,
hover or flight claim. If the direction learns damping but fails RPY preservation, the
next justified diagnostic is an RPY-output-Jacobian-constrained displacement. If it fits
training but not development, projection would not address the demonstrated failure.

Result: all cache, identity, replay, teacher/foreleg, finite-difference and restoration
controls passed. Endpoint-D-only descent was clear and approximately linear: from a
source NRMSE of 1.45840, scales 1, 1/2, 1/4 and 1/8 improved it by 0.05516, 0.02759,
0.01374 and 0.00685. Even the full step kept roll/pitch/yaw source NRMSE at
0.01447/0.02259/0.00349, so RPY interference was not the limiting factor. No scale met
the C/P preservation gates. At scale 1/8, P passed, but aggregate C worsened by 0.02471
and its step-20/25 values worsened by 0.02483/0.03891, beyond 0.02. Thus the experiment
found a useful local damping gradient coupled primarily to common collective response.
It selected no candidate and consequently performed no development model evaluation.
No parameters were retained, no D-first fitting was authorized, and no closed-loop test
ran. See
[`artifacts/variable-height-full-native-endpoint-damping-step-audit-v1/`](../artifacts/variable-height-full-native-endpoint-damping-step-audit-v1/).

Before adding a Jacobian projection, perform one separately registered **small-step
extension** on the same immutable caches and damping-directed update. The nearly linear
training response predicts that an ordinary smaller step may satisfy the existing C/P
limits while retaining a damping improvement above the registered floor. Preserve the
failed original-grid verdict. Recreate the original bound-projected Adam displacement
without changing its objective or optimizer and require its baseline endpoint-D NRMSE,
full-scale endpoint result, directional derivative and parameter-family RMS values to
reproduce the frozen report within relative tolerance `1e-3` and absolute tolerance
`1e-7`. Persist that regenerated displacement, hash it, reload it, and use those exact
tensors thereafter.

Replay only scales 1/16 and 1/32 on the original training bank, in descending order.
Select the largest scale that improves endpoint-D NRMSE by at least 0.001 while passing
the identical aggregate and per-horizon C/P, RPY, finite-state and motor-bound gates.
If neither passes, stop and next preregister a C/P-output-Jacobian-constrained damping
direction; do not add RPY constraints because RPY is not the observed limiter. If one
passes, evaluate only that selection on the still-unexamined development outputs with
the identical gates and no smaller-scale fallback after development failure. Restore
all parameters bit-exactly and retain no endpoint.

A pass demonstrates only a locally feasible damping improvement, not independent C/D
control or sustained learnability. It authorizes a separately preregistered bounded
D-first fitting diagnostic whose cumulative C/P preservation is always measured against
the original source, so repeated individually safe steps cannot silently consume the
entire tolerance. No direct closed-loop or promotion claim follows.

Result: all controls passed and both smaller scales passed training, so 1/16 was selected
before development was examined. Endpoint-D NRMSE improved from 1.458401 to 1.454984 on
training and from 1.398940 to 1.396360 on the single development evaluation. All C/P,
RPY, validity and output-bound gates passed on both banks. The tightest constraint was
training endpoint C, which worsened by 0.019590 against the 0.02 allowance. Endpoint
damping remained wrong-signed in all scenes, although aligned gain moved in the correct
direction on both banks. This establishes a safe, transferable local damping step and
authorizes the bounded D-first fitting diagnostic; it does not establish independent
C/D control or useful damping yet. Parameters were restored exactly, nothing was
promoted, and no closed-loop test ran. See
[`artifacts/variable-height-full-native-endpoint-damping-small-step-audit-v1/`](../artifacts/variable-height-full-native-endpoint-damping-small-step-audit-v1/).

The authorized follow-up is a bounded, constraint-aware **D-first fitting diagnostic**,
not repeated 1/16 steps: that first step consumed 98% of the permitted training endpoint
C regression. Reuse the exact eight-scene training bank and now-exposed eight-scene
development bank, their source baselines, endpoint-D normalization, actor contract and
complete zero-state replay. Reserve an ungenerated 64-scene qualification cohort
beginning at seed `360971`; expose it only after a development-qualified endpoint exists.
The full-native parameter families and Adam settings remain unchanged. Endpoint-D
normalized MSE remains the only fitting objective; C/P/RPY are constraints rather than
weighted competing losses.

At each update, form the full-bank endpoint-D Adam displacement. Minimally modify it in
learning-rate-scaled Euclidean coordinates to satisfy the linearized *remaining*
cumulative C/P allowances relative to the original source. The eight inequality rows
are the gradients of squared normalized error for aggregate C and P and for C and P at
each of steps 15, 20 and 25; their limits are the squares of the corresponding original
source NRMSE plus 0.02. Solve the small dual nonnegative quadratic program to project the
proposal onto these half-spaces, and report primal violation, dual convergence and the
fraction of damping descent retained. Do not impose 48 individual-output equalities.
RPY remains forward-checked; add its analogous constraint row for an axis only when the
accepted current controller reaches NRMSE 0.04, because RPY has not been the observed
limiter. Parameter bounds are then applied and every directional and safety decision is
made from the actual post-bound displacement and replay, not the linear model alone.

Before fitting, the constrained proposal must pass a preflight on the complete training
bank: exact cache/source replay, teacher and foreleg/stick controls, finite values,
negative endpoint-D directional derivative, linearized constraint satisfaction, and a
scale-0.0625 endpoint-D finite difference agreeing within 20%. No update occurs if this
preflight fails. During fitting, backtrack each projected displacement at fixed scales
1, 1/2, 1/4, 1/8, 1/16 and 1/32. Accept the largest scale with actual endpoint-D NRMSE
improvement at least 0.001 and all cumulative aggregate/per-horizon C/P, RPY, validity
and motor-bound gates passing against the original source. Retain the Adam state only
with an accepted update. If no scale qualifies, restore both parameters and optimizer
and stop immediately; a deterministic rejected full-bank proposal is not retried.

Allow at most 200 accepted updates and evaluate development only every ten accepted
updates. At update 50, both training and development endpoint-D NRMSE must have improved
at least 25% from their original source values and all preservation gates must pass, or
stop. Thereafter the first scheduled checkpoint with training endpoint-D NRMSE at most
0.20, development at most 0.30, at least 90% correct endpoint damping sign on each bank,
teacher-aligned gain 0.5-1.5 on each, and every preservation gate passing is the sole
qualification candidate. If no such checkpoint exists by update 200, stop without a
fresh test. Checkpoints are ignored run artifacts and nonpromotional.

For the fresh qualification, stream eight independently generated eight-scene banks
with factorial seeds `360971` through `360978` and attitude seeds `370971` through
`370978`, comparing the candidate and original source on each same bank before
discarding its tensors. Across all 64 scenes require endpoint-D NRMSE at most 0.30, at
least 90% correct sign, gain 0.5-1.5, and independently measured aggregate/per-horizon
C/P, RPY, validity and motor bounds under the same source-relative limits. There is no
fallback checkpoint. Even a pass does not promote the fit; it only authorizes a
nominal-mass native closed-loop hover comparison against the source, including
frozen-vision controls. The source's large absolute C error means successful damping
fitting alone is not evidence of calibrated collective or stable hover.

Result: the fitting preflight stopped before update one. The eight-row dual QP converged,
left maximum linearized violation `1.50e-7`, and retained 74.1% of the endpoint-D descent.
The later edge [0, 8] clamp changed that direction enough to create post-bound linearized
MSE violations of 0.006976 for endpoint C and 0.000179 for step-20 P, so the frozen
post-bound projection gate failed. The actual scale-0.0625 replay was nevertheless safe,
improved endpoint-D NRMSE by 0.002529, and agreed with its derivative within 0.15%. This
supports a bound-aware projection correction but cannot pass the current preflight after
the fact. No update, resume state, development candidate, qualification or closed-loop
test resulted; parameters were restored exactly. See
[`artifacts/variable-height-full-native-d-first-fitting-preflight-v1/`](../artifacts/variable-height-full-native-d-first-fitting-preflight-v1/).

Repeat the same fitting protocol as a separately identified bound-aware run; preserve
the failed preflight above. Change only the inequality projection's handling of edge
bounds. Begin with no fixed coordinates. After each half-space solve, add every edge
whose proposed magnitude crosses [0, 8] to a monotonically growing active set and fix its
displacement to the actual boundary displacement, `0 - current` or `8 - current`—not
zero unless the edge already occupies that boundary. On every re-solve, remove those
coordinates from the free-coordinate metric and Gram matrix and add their complete
`J_bound * delta_bound` contribution to each constraint residual. Continue from the
original Adam proposal on all still-free coordinates for at most eight active-set rounds.

The active-set method is a bounded feasibility heuristic, not a claim of finding the
exact box-constrained optimum, because activated edges are never released. The preflight
passes only if no edge remains out of bounds, every final post-bound linear inequality is
within `1e-6`, a subsequent controller bound projection changes the displacement by at
most `1e-7`, and the existing damping derivative, scale-0.0625 finite-difference,
nonlinear replay, C/P/RPY, identity and restoration gates all pass. Failure at eight
rounds means this solver failed, not that no feasible damping direction exists. If the
preflight passes, continue directly into the otherwise unchanged frozen D-first fitting
and qualification protocol; do not change any threshold, data or actor input.

Result: the active set converged in three rounds after fixing 4,699 crossing edges at
their actual boundaries. The final direction had no box violation, kept post-bound
linearized constraint violation to `7.18e-7` below the `1e-6` limit, retained 73.97% of
damping descent, and passed the nonlinear scale-0.0625 replay with 0.002525 endpoint-D
NRMSE improvement and 0.18% finite-difference error. The preflight nevertheless failed
its separately frozen idempotence gate: re-materializing the controller parameters
changed the float32 displacement by `1.1902e-7`, just above `1e-7`. The limit is not
relaxed after observing it. No fitting update or resume state was created, parameters
returned exactly to source, and no development candidate or closed-loop test ran. See
[`artifacts/variable-height-full-native-d-first-bound-aware-preflight-v1/`](../artifacts/variable-height-full-native-d-first-bound-aware-preflight-v1/).

Make one separately registered numerical-canonicalization repeat; preserve both prior
preflight failures and keep every tolerance and substantive gate unchanged. After the
bound-aware active-set solve, materialize its proposed controller parameters exactly
once and make that float32 parameter tensor—not a repeatedly reconstructed float32
displacement—the authoritative full-step candidate. Apply `project_parameters()` a
second time directly to those authoritative parameters and require a maximum parameter
change at most `1e-7`. Report the coordinate responsible for the prior maximum
difference, its source/proposal/materialized values, local float32 ULP and difference in
ULPs. Do not iterate canonicalization until a check passes.

Derive the effective displacement from the authoritative candidate and current
parameters using float64 subtraction for the constraint and damping dot products.
Recompute all linear inequalities and damping descent from that effective direction.
Install a scale-1 trial by directly copying the authoritative parameter tensor, avoiding
another `source + (candidate - source)` round trip. For each smaller backtrack scale,
materialize exactly one float64 interpolation between the current and authoritative
parameter tensors, convert it once to the parameter dtype, apply the controller bounds,
and evaluate its actual displacement and nonlinear replay. The same procedure supplies
the scale-0.0625 finite difference.

The corrected repeat may enter the otherwise unchanged fitting protocol only after
parameter idempotence, linear feasibility, negative damping derivative, finite
difference and complete replay all pass. It creates no training or resume state before
then. This is a numerical representation correction only; it changes neither the actor
contract, objective, constraints, thresholds, optimizer, data nor qualification plan.

Result: canonicalization resolved the numerical control issue and the preflight passed.
The second direct parameter projection was exactly idempotent; the earlier maximum
difference was measured as 0.4996 local float32 ULP. Eight fitting updates were accepted
at scales 1/2, 1/2, 1/2, 1/8, 1/8, 1/16, 1/16 and 1/4. Training endpoint-D NRMSE fell
from 1.458400 to 1.376246 and aligned gain moved from -0.43297 to -0.35893, while sign
remained 0/8. At update 8, endpoint C was 1.68378794 against its source-relative limit
of 1.68379011, leaving only `2.17e-6` NRMSE slack. The update-9 linear tangent remained
feasible, but nonlinear replay exceeded endpoint C even at 1/32; that smallest trial
otherwise improved endpoint-D NRMSE by 0.001118 and passed every other gate. The run
therefore stopped after the first rejected deterministic proposal, before development
update 10. Its update-8 resume state is ignored and nonpromotional. No terminal, fresh,
closed-loop or promotion claim resulted. See
[`artifacts/variable-height-full-native-d-first-canonical-fitting-v1/`](../artifacts/variable-height-full-native-d-first-canonical-fitting-v1/).

Do not yet resume fitting or infer an unavoidable C/D tradeoff. Run one restored
**nonlinear feasibility-correction audit** from the exact ignored update-8 resume state
(SHA-256 `eb30a7c1d6f75ee65a6588dbe09475b66c172227170300248625baea1b15b148`).
First reproduce the frozen update-8 metrics, optimizer state and rejected update-9 Adam
proposal. Use only the update-9 1/16 trial as the starting candidate; its registered
training signature includes endpoint-D NRMSE 1.374014, endpoint C NRMSE 1.683883, pitch
NRMSE 0.044974 and actual endpoint-D improvement 0.002233 from update 8. Persist and hash
the reproduced starting parameter tensors before correction.

At that actual nonlinear candidate, compute fresh gradients of the same aggregate and
per-horizon C/P squared normalized errors, plus any RPY row activated at NRMSE 0.04.
Add one endpoint-D retention inequality. Solve a single minimum-norm correction in the
same learning-rate-scaled Euclidean metric with the established active-set boundary
materialization. Endpoint C must target NRMSE no greater than the original source plus
0.0199, placing it at least 0.0001 inside the unchanged outer source-plus-0.02 gate. All
other C/P and RPY constraints retain their original outer limits. Endpoint-D NRMSE must
remain at least 0.001 below update 8. Convert every NRMSE limit to its squared-error RHS
before projection.

Materialize the correction once at scales 1, 1/2, 1/4 and 1/8 in that fixed descending
order. Select the first scale whose actual complete replay satisfies the endpoint-C
interior target, the endpoint-D retention target, every original preservation/validity/
bound gate, linear feasibility and canonical parameter idempotence. There is one
correction solve and no retry with a newly linearized correction. Restore the update-8
parameters exactly and retain no corrected candidate. Failure does not automatically
authorize a D-sacrificing collective-restoration phase.

As a separate registered diagnostic in the same run, evaluate the fixed update-8
checkpoint once on the already-exposed development bank. Require endpoint-D NRMSE to
improve by at least 0.001 from the development source and all original C/P/RPY,
validity and motor-bound gates to pass. This evaluation cannot retroactively continue or
promote the stopped fit. The correction audit passes only if both its training
feasibility test and this update-8 development-transfer test pass. A pass authorizes a
separately registered corrected fitting protocol; it does not authorize closed-loop
hover or promotion.

Result: every immutable source, update-8, update-9 proposal and scale-1/16 starting-
candidate reproduction passed. The fixed update-8 controller also passed its separate
development diagnostic: endpoint-D NRMSE improved from 1.398940 to 1.336928, an
absolute improvement of 0.062012, while every original C/P/RPY, validity and motor-
output preservation gate passed. This is useful evidence that update 8 transfers, but it
does not establish adequate damping or authorize that checkpoint for closed-loop use.

The correction audit itself failed two frozen numerical controls. The two mathematically
equivalent endpoint-D gradient paths differed by `2.60e-5` at their worst coordinate,
above the registered combined absolute limit of `1.41e-5`, although their relative L2
difference of `8.37e-6` passed its `1e-5` limit. The continuous active-set correction
reached maximum linear violation `3.04e-11`; after conversion to authoritative float32
parameters, accumulated rounding raised that violation to `1.33e-5`, above `1e-6`.
The full correction's actual endpoint-C NRMSE was 1.68369424, only `4.14e-6` above the
source-plus-0.0199 interior target and still inside the original source-plus-0.02 outer
gate. It retained 0.002232 endpoint-D improvement from update 8 and passed every other
outer gate. The fixed thresholds are not relaxed after observing this result. Parameters,
Adam state and the resume file were restored exactly; no corrected candidate, fitting
authorization, closed-loop run or promotion resulted. See
[`artifacts/variable-height-full-native-d-first-nonlinear-correction-audit-v1/`](../artifacts/variable-height-full-native-d-first-nonlinear-correction-audit-v1/).

Run one separately registered **canonical guard-band correction repeat**. Reconstruct
the same hash-locked update-8 state, rejected update-9 proposal and scale-1/16 starting
candidate; preserve the failed audit and all of its measurements. At the actual starting
candidate, compute C/P and any activated RPY rows exactly as before. Generate the
authoritative endpoint-D retention row with the dedicated normalized endpoint-D
objective and `accumulated_endpoint_damping_gradient`, rather than the multi-loss row.
Still compute the multi-loss row once and report its agreement under the prior failed
absolute and relative limits, but make that duplicate comparison diagnostic only.

Replace the failed duplicate-gradient control with one directional finite-difference
control. Its fixed probe is the once-materialized interpolation one quarter of the way
from the starting candidate back toward update 8. Compare the authoritative endpoint-D
squared-error row's float64 dot product with the complete-replay endpoint-D squared-
error change over that displacement. Require both values and their comparison to be
finite, the direction to produce a measurable nonzero change, native parameter bounds
to hold, and relative disagreement no greater than 20%. Restore the exact starting
candidate before solving the correction.

Keep distinct solver and acceptance specifications. The one zero-reference, minimum-
norm, bound-aware solve uses an internal endpoint-C target of original source plus
0.0198, squared before projection. This supplies a numerical guard band; it does not
change the actual acceptance limit of source plus 0.0199 or the original outer limit of
source plus 0.02. The endpoint-D retention limit remains update-8 NRMSE minus 0.001,
squared, and all other constraints remain unchanged. Check the ideal continuous solve
against the stricter solver specifications. Check each canonical, post-materialization
linear displacement against separate acceptance specifications containing the unchanged
source-plus-0.0199 endpoint-C target, and apply all unchanged nonlinear D-retention,
C/P/RPY, validity, output and native-bound gates.

Use the same single fixed Jacobian, one correction solve, no relinearization and fixed
descending correction scales 1, 1/2, 1/4 and 1/8. Select the first actual passing scale,
repeat the fixed update-8 development-transfer diagnostic, then restore update-8
parameters and the complete optimizer state exactly. Retain no corrected candidate. A
pass authorizes only a separately registered corrected fitting protocol. If this one
canonical guard-band repeat fails, stop this correction route rather than adding a
residual-repair solve or changing a threshold after the result.

Result: the canonical guard-band repeat passed all controls. The authoritative endpoint-
D row's fixed directional finite difference was finite and measurable and disagreed by
only 0.0883%, below its 20% limit. The duplicate multi-loss row failed its preserved old
diagnostic limits again, as expected, but was not a gate. The ideal stricter-target solve
reached maximum linear violation `-9.18e-12`. Float32 materialization left `1.38e-5`
relative to the deliberately stricter solver boundary but `-1.67e-4` relative to the
unchanged acceptance specification, so the guard band worked without relaxing a task
threshold. Scale 1 was the first correction scale. Its actual endpoint-C NRMSE was
1.68359399 against the 1.68369011 acceptance limit; endpoint-D was 1.37401438, retaining
0.002232 improvement from update 8. All original outer gates, the repeated update-8
development diagnostic, canonical idempotence, bounds and exact restoration passed. No
candidate was retained and nothing was promoted. See
[`artifacts/variable-height-full-native-d-first-canonical-guard-band-audit-v1/`](../artifacts/variable-height-full-native-d-first-canonical-guard-band-audit-v1/).

Proceed with one separately registered **corrected D-first fitting run** beginning from
the exact update-8 resume state, without reopening or overwriting the stopped canonical
fit. Preserve its source/cache hashes, native recurrent actor and inputs, endpoint-D-only
Adam objective, parameter families, learning rates, gradient clipping, original C/P/RPY/
validity/output constraints, development and qualification banks, terminal gates and
no-automatic-promotion rule. The budget remains 200 total accepted updates, not 200 new
updates: start at accepted update 8, evaluate development at totals 10, 20 and so on,
apply the mandatory progress gate at total 50, and stop at total 200 if no terminal pass.

For each attempted update, take exactly one Adam moment/step update and form the same
canonical, bound-aware proposal. It must pass every existing numerical and projected-
direction control; a failed projection or other numerical control cannot be repaired.
Try the original backtrack scales in their fixed descending order and accept the first
ordinary passing candidate exactly as before. Retain the one pending Adam state when an
ordinary candidate is accepted.

Only if every ordinary backtrack fails may the fixed scale-1/16 candidate enter the
repair path. It is eligible only when its actual endpoint-D NRMSE improves by at least
0.001 from the current accepted controller, its actual displacement remains a damping
descent direction, and every ordinary finite/validity/native-bound/motor-output and
source-relative C/P/RPY gate passes except that endpoint common response at step 25 may
exceed its original nonlinear source-plus-0.02 limit. Any additional failure rejects the
proposal without repair.

At an eligible candidate, apply exactly the passing guard-band mechanism: C/P and
activated RPY refer to the original source, while endpoint-D retention requires at least
0.001 improvement from the current accepted controller. The authoritative D row comes
from the dedicated accumulated endpoint-D objective. Its fixed finite-difference probe
moves one quarter from the candidate back toward that current controller. Require the
same finite, measurable, bound, restoration and 20%-agreement controls. Report the
multi-loss D row under the prior limits as a non-gating diagnostic.

Use one zero-reference bound-aware minimum-norm correction with a fixed Jacobian, no
relinearization or second solve, endpoint-C solver target source plus 0.0198, actual
linear and nonlinear acceptance at source plus 0.0199, unchanged remaining outer
constraints, and correction scales 1, 1/2, 1/4 and 1/8. Accept the first passing scale.
Correction differentiation must not perform another Adam step. Retain the already-
pending single Adam state if repair succeeds; if eligibility, correction controls or all
correction scales fail, restore both pre-update parameters and optimizer exactly and stop.

Count repaired updates normally. Log whether each accepted update was ordinary or
repaired, both the proposal and correction scales, endpoint-C headroom, endpoint-D
improvement, and parameter/optimizer transaction checks. Resume artifacts remain ignored
and nonpromotional. Qualification remains first-terminal-checkpoint-only with no candidate
fallback. A fresh pass authorizes the already specified nominal-mass native closed-loop
hover comparison; fitting alone still cannot establish stable hover or flight.

Result: the corrected fitter accepted 12 additional updates, reaching total update 20.
Eight were ordinary updates; updates 9, 12, 14 and 18 used the fixed scale-1/16 proposal
and scale-1 guard-band correction. All corrected-run one-step Adam controls passed, as did
the pending-optimizer checks on every repaired update. Training endpoint-D NRMSE improved
from 1.458401 at the original source and 1.376246 at update 8 to 1.355903 at update 20.
Development endpoint-D NRMSE reached 1.334292 at update 10 and 1.321031 at update 20,
with every scheduled preservation gate passing. Sign fraction nevertheless remained zero
and gain remained negative, so this is not useful damping and authorizes no flight test.

Update 21 stopped before trial replay. The active-set projection's second round returned
an abnormal L-BFGS-B termination after round one had fixed 5,508 crossing edges. Its
continuous maximum linearized violation was `2.16e-7`, inside the `1e-6` residual limit,
but solver success was false and authoritative materialization left `3.87e-6` violation,
outside the limit. The frozen protocol requires both controls and forbids nonlinear repair
of a projection failure, so the proposal was rejected without retry. The stopped resume
retains update 20, restores the optimizer to its recorded pre-proposal hash, and cannot
silently continue. No terminal checkpoint, fresh qualification, closed-loop hover run or
promotion resulted. See
[`artifacts/variable-height-full-native-d-first-corrected-fitting-v1/`](../artifacts/variable-height-full-native-d-first-corrected-fitting-v1/).

Do not treat the update-21 stop as evidence that the actor, objective or constrained
route is exhausted, and do not resume the stopped run. Run one restored **FP64 projection
replay audit** from its exact update-20 controller and Adam state. Hash-lock the stopped
resume (SHA-256 `61d32fab3995599286f6eee3b30f24b3bea48042d2528392fdd1ca68e9bf60c4`)
and full report (SHA-256
`9b236fc6a6744b1b06984958ebc2c3ff3681cc45fc1fa6da73d1f96ae56b458b`).
First reproduce the registered update-20 metrics, rejected update-21 Adam proposal,
constraint specifications, two active-set rounds and failure measurements within the
existing reproduction tolerances. Persist hashes of the reconstructed raw displacement,
Jacobian rows, limits and failed round-two fixed set before comparing solvers. Generate
these tensors once only; every numerical variant must consume clones of the same frozen
tensors rather than regenerating gradients.

Replay the production projector unchanged as a control. Then change only the projection's
linear algebra to genuine float64: convert rows, raw displacement and current edge values
once before any product; perform Gram products, reductions, fixed-coordinate residual
contributions, displacement accumulation and final residuals in float64. Keep the same
learning-rate-scaled metric, inequalities, L-BFGS-B dual formulation and tolerances,
monotonic edge active set, actual boundary displacements and maximum eight rounds. Do not
override an unsuccessful solver result. Record the normalized Gram eigenvalues and rank
using relative eigenvalue cutoff `1e-12`, because the aggregate C/P rows are algebraically
dependent or nearly dependent on their three supervised-horizon rows.

At every active-set solve, report primal violation in original squared-error units, the
normalized dual projected-gradient residual and complementarity. Normalize the KKT
residual by `max(1, ||normalized_violation||_inf)`; for a lower-bounded dual coordinate use
the absolute gradient when its multiplier is positive and `max(-gradient, 0)` at zero.
The FP64 route requires L-BFGS-B success, original-unit primal violation at most `1e-6`
and normalized projected-gradient residual at most `1e-8`. Also run one fixed independent
SLSQP solve of each small frozen FP64 dual QP from zero, with analytic gradient,
nonnegative bounds, `ftol=1e-12` and at most 10,000 iterations. It is diagnostic only and
cannot supply or select a flight candidate.

Only if the primary FP64 active set passes those controls, canonicalize its final proposed
parameters once and apply the unchanged edge bounds, `1e-7` idempotence, post-materialized
`1e-6` linear limit and negative endpoint-D derivative gates. Replay the fixed scale-1/16
directional finite difference, requiring finite measurable D squared-error change and at
most 20% relative disagreement. Finally try the unchanged ordinary scales 1, 1/2, 1/4,
1/8, 1/16 and 1/32 in order and report whether the first candidate passes every original
nonlinear source-relative C/P/RPY, validity, motor-output and current-relative D-improvement
gate. Do not invoke the nonlinear repair path in this audit.

Restore update-20 parameters and complete Adam state exactly and leave the stopped resume
and report byte-for-byte unchanged. Run no optimizer update beyond the transactionally
reconstructed proposal, no development or fresh data, no closed-loop simulation and no
promotion. A pass authorizes only a separately registered genuine-FP64 projection
implementation; it does not retroactively accept update 21. A failure characterizes this
numerical route only, especially because zero displacement remains feasible for the
original preservation-plus-box constraints and the monotonic active-set heuristic may add
restrictions of its own.

Result: this audit failed its frozen production-reproduction control. The registered
production projection had stopped abnormally in active-set round two, whereas the newly
reconstructed production inputs completed round three and passed; proposal metrics still
matched within the existing numeric tolerances. This is evidence of solver-status
sensitivity across slightly different GPU reconstructions, not nondeterminism on identical
inputs. The failed control is not removed after observing it, so the audit does not
authorize an FP64 fitting implementation.

On this audit's one frozen tensor set, genuine FP64 projection nevertheless gave a useful
diagnostic. It completed three rounds with 5,524 fixed edges. All three 10-row normalized
Gram matrices had rank 8 and nullity 2. Primary L-BFGS-B and independent SLSQP reported
success; maximum original-unit primal violation was `5.31e-11` and maximum normalized
primary KKT residual was `3.04e-10`. Canonical idempotence, bounds, the post-materialized
linear gate at `1.30e-7`, negative D direction, and the scale-1/16 finite difference with
0.103% disagreement all passed. No ordinary nonlinear scale passed: scale 1/32 improved D
by 0.001130 but missed only endpoint C25's unchanged outer limit by about `1.30e-5`.
Repair was prohibited in this audit. Everything was restored exactly and no candidate,
development/fresh result, closed-loop run or promotion resulted. See
[`artifacts/variable-height-full-native-d-first-fp64-projection-audit-v1/`](../artifacts/variable-height-full-native-d-first-fp64-projection-audit-v1/).

Preserve that control failure and run one **frozen-input FP64 projection qualification**;
do not try to reproduce the historical `ABNORMAL` status again. From the same hash-locked
update-20 state, generate the raw update-21 Adam displacement, ordered constraint rows and
specifications, current parameters/bounds, and endpoint-D gradient exactly once. Before
any solve, persist all actual tensors—not only their hashes—to one ignored CPU tensor
archive; record its file and semantic hashes and reload it for the numerical trials. The
single reconstructed Adam transaction must advance every counter exactly once and be
restored at the end.

Run the complete genuine-FP64 active-set projection three times from an empty active set,
each time using fresh clones reloaded from that exact archive and retaining the same
eight-round bound. The production projector may also consume a clone, but its status is
diagnostic only. Each FP64 run must independently satisfy L-BFGS-B success, original-unit
primal violation at most `1e-6`, normalized projected-gradient residual at most `1e-8`,
native box bounds and active-set convergence. Dependent rows permit nonunique dual
multipliers, so compare primary **primal** results: require pairwise learning-rate-scaled
displacement difference divided by `max(1, displacement norm)` at most `1e-10`, and
relative difference in the complete bound-aware projection objective `0.5 * ||(projected
- raw) / learning_rate||^2` at most `1e-12`; do not gate on dual coefficients, iteration
counts or historical status strings.

An implementation review before any formal frozen-input run found a flaw in the initially
registered eigenspace-factor NNLS check. For rank-deficient Gram `G`, projecting the linear
term `v` into `range(G)` changes the nonnegative dual QP whenever `v` has a nullspace
component; differing aggregate and horizon limits can produce exactly that case. For
example, `G = ones(2, 2)` and `v = [2, 1]` has the correct multiplier sum 2, while the
factorized NNLS problem returns sum 1.5. KKT-gating that approximation would reject the
correct primary answer rather than independently validate it. This paragraph therefore
supersedes only that independent-check detail of commit `df0910d`, before looking at any
formal result.

Replace the prior SLSQP status comparison with an exhaustive active-support check of each
small FP64 dual QP (at most `2^10` supports). On every support, use Lawson-Hanson
nonnegative least squares on the original stationarity equations `G_SS * lambda_S = v_S`;
embed the result in the full dual vector and retain only finite solutions satisfying the
original full-QP normalized KKT residual at most `1e-8`. Report the normalized Gram rank,
nullity, and the violation component outside its retained eigenspace using the frozen
relative rank cutoff `1e-12`, but do not discard that component in the solve. Select the
KKT-feasible solution with the lowest original dual objective. Compare this independent
solution and the primary solution by Gram-induced primal correction distance normalized
by the primary correction norm, and by relative original dual objective; require at most
`1e-6` and `1e-8` respectively. These compare the unique primal effect rather than a
nonunique dual vector.

Canonicalize the first repeated FP64 result once and require the unchanged `1e-7`
idempotence, native bounds, `1e-6` post-materialized linear gate, negative endpoint-D
direction and scale-1/16 finite-difference controls. Do not require an ordinary nonlinear
candidate to pass and do not invoke repair: this audit qualifies only the numerical
projection implementation, not update 21. The prior ordinary replay already established
that the scale-1/16 candidate is in the separately tested C25-only repair case.

Restore parameters and full Adam state exactly, leave every source and failed-audit file
byte-for-byte unchanged, and use no development/fresh data, retained candidate, closed-loop
simulation or promotion. A pass authorizes only a separately registered one-step
FP64-projected-plus-existing-guard-band-corrected audit. If identical-input FP64
qualification fails, pause this numerical integration route without changing the actor,
sensors, runtime state or preservation thresholds.

Result: the frozen-input FP64 qualification passed. All three projections reloaded from
the one persisted CPU archive produced the same final displacement hash and complete
bound-aware primal objective, with zero pairwise numerical difference. Each fixed 5,508
then 16 edges and converged in round three with 5,524 fixed edges. Maximum original-unit
primal violation was `1.02e-10` and maximum normalized primary KKT residual was
`5.99e-10`. Every independent exhaustive-support NNLS check passed against the original
rank-deficient QP; maximum Gram-induced primal disagreement was `8.14e-9` and relative
objective disagreement was `6.14e-16`. The nonzero out-of-range violation components
confirmed that the pre-run protocol correction was material rather than cosmetic.

Canonical idempotence, native bounds, the `1.18e-7` post-materialization linear gate,
negative endpoint-D direction, and the scale-1/16 finite difference with 0.113%
disagreement all passed. The reconstructed Adam transaction advanced each parameter
counter exactly once, then parameters and Adam state were restored exactly. Every source
and prior failed-audit file remained byte-for-byte unchanged. No candidate, development
or fresh result, closed-loop run, or promotion resulted. See
[`artifacts/variable-height-full-native-d-first-fp64-frozen-input-audit-v1/`](../artifacts/variable-height-full-native-d-first-fp64-frozen-input-audit-v1/).

Run one restored **FP64-primary plus existing guard-band correction update-21 audit**.
Hash-lock the passing qualification report (SHA-256
`e1ead1779c568cacfcebda5df435175f505c9e7218592c37b1187faeaf929f25`), its frozen CPU
archive (SHA-256
`e228a3920c47f56a7eac6d1452f996d9709721f82536240d96dbe6e0c6e1830f`), and the stopped
update-20 resume (SHA-256
`61d32fab3995599286f6eee3b30f24b3bea48042d2528392fdd1ca68e9bf60c4`). Consume fresh
clones of the already-qualified archived current parameters, raw update-21 displacement,
ordered constraint rows/specifications, and endpoint-D gradient. Do not regenerate those
Jacobians and do not retry a historical solver status. Recompute the primary FP64
projection once from the archive and require its full controls plus exact agreement with
the qualified final displacement hash
`0fb352c318ab18efafe0bc2ff68991709aa2f14b79774f7c8e5475d432135e3f` and complete primal
objective `17920.27762329695` under the registered relative tolerance `1e-12`.

Reconstruct the pending Adam state from update 20 using the archived raw gradient and
unchanged clipping. Require exactly one counter increment for every parameter, exact
agreement with the archived raw displacement after native parameter projection, and the
qualified post-step optimizer hash
`6743cc01e41e01ccd7183b8981acff29eb5af22a72c3039bb9d234cfcbde42d6`. This is provenance
for the one transaction, not another gradient or optimizer update. Correction and all
candidate evaluation must leave that pending state unchanged.

Canonicalize the FP64 displacement and try the unchanged ordinary proposal scales 1,
1/2, 1/4, 1/8, 1/16 and 1/32 in descending order. Select the first candidate passing every
existing finite, idempotence, native-bound, motor-output, validity, source-relative C/P/RPY
and current-relative endpoint-D gate. Only if no ordinary scale passes may the fixed
scale-1/16 candidate enter repair. It must improve endpoint-D NRMSE by at least 0.001 from
update 20, remain a damping descent direction, and pass every ordinary gate except that
endpoint common response at step 25 is above its original source-plus-0.02 outer limit.
Any other failure, or any primary projection/numerical failure, rejects the transaction;
neither is repairable.

Apply the existing repair unchanged at that actual scale-1/16 candidate. Compute C/P and
activated RPY rows against the original source, use the authoritative dedicated endpoint-D
row with retention at least 0.001 better than update 20, and keep the duplicate multi-loss
D row diagnostic-only. Require the fixed finite-difference probe one quarter of the way
back toward update 20. Use one zero-reference bound-aware minimum-norm solve, one fixed
Jacobian, no relinearization or second correction, endpoint-C25 internal solver target
source plus 0.0198, actual acceptance source plus 0.0199, the unchanged outer gates, and
correction scales 1, 1/2, 1/4 and 1/8. Select the first scale passing every numerical,
canonical, bound, linear and actual nonlinear replay gate. Do not change the existing
repair arithmetic while qualifying the FP64 primary projector.

After training selection chooses exactly one ordinary or repaired candidate, evaluate
update 20 and that candidate on the fixed development bank. Require candidate endpoint-D
NRMSE improvement of at least 0.001 relative to update 20 and every original-source
development preservation gate. Development failure is terminal for this audit: do not try
another training candidate, run a fresh cohort or fall back to the source.

Finally restore update-20 parameters and complete Adam state exactly and leave the stopped
fit, qualification report and tensor archive byte-for-byte unchanged. Retain no runnable
continuation checkpoint, run no fresh data or closed-loop simulation, and make no
promotion. A pass authorizes only a separately registered FP64-primary corrected fitting
continuation from update 20 toward the existing total-update-50 milestone; it neither
accepts update 21 into the stopped run nor establishes useful damping or hover.

Result: the restored update-21 audit passed. The archived raw displacement, reconstructed
one-step Adam state, qualified FP64 displacement hash and complete primal objective all
matched exactly. No ordinary scale passed. Scale 1/16 improved training endpoint-D NRMSE
by 0.002262 and failed only endpoint-C25's outer gate, so it entered the unchanged repair.
The first correction scale, 1.0, passed every control and nonlinear gate. Its training
endpoint-D NRMSE was `1.35364211`, and endpoint-C25 was `1.68359399`, leaving `9.61e-5`
headroom to the source-plus-0.0199 acceptance boundary. The repair directional finite
difference disagreed by only 0.0580%, and correction made no additional Adam step.

The selected candidate also passed the one fixed development transfer. Development
endpoint-D NRMSE improved from update 20's `1.32103145` to `1.31926715`, a 0.001764
improvement, while every original-source preservation gate passed. Training and
development sign fraction remained zero and teacher-aligned gain remained negative, so
this is not yet useful damping. Update-20 parameters and Adam state were restored exactly;
all locked files were unchanged and no candidate, checkpoint, fresh cohort, closed-loop
run, or promotion resulted. See
[`artifacts/variable-height-full-native-d-first-fp64-corrected-step-audit-v1/`](../artifacts/variable-height-full-native-d-first-fp64-corrected-step-audit-v1/).

Proceed with one **FP64-primary corrected fitting continuation to total update 50** in a
new ignored run; do not resume or overwrite the stopped float32-projection run. Hash-lock
its update-20 report and resume, the passing frozen-input qualification and archive, and
the passing restored step-audit report (SHA-256
`8f76e9b5bd9cb1cecc782b207c3694b332926fb9f34b5be1fea8fd31b2deb6dd`). Preserve the
native recurrent actor, visual/roll-pitch inputs, foreleg outputs, endpoint-D-only Adam
objective, parameter families, learning rates, gradient clipping, source-relative
C/P/RPY/validity/output constraints, current-relative D improvement, development bank,
terminal criteria, and no-automatic-promotion rule.

Seed the new crash-safe resume from exact update 20. Reproduce accepted update 21 from the
frozen archive rather than regenerating its Jacobians: reconstruct the single pending Adam
state from the archived gradient, require the exact raw and FP64 displacement/objective
hashes, run the ordinary-then-repair selection unchanged, and require the accepted
parameter hash
`3d8d96f36944113ad942ebd1d30f6510998a3a337881b03e0ce3e6032e0f7308` plus registered
training metrics within the existing reproduction tolerances. Persist it as accepted
update 21 only after all transaction, projection, repair and reproduction controls pass.
The step audit's already passing development result is provenance, not an extra candidate
selection channel; scheduled continuation development evaluations remain fixed below.

For updates 22 onward, generate each current proposal's constraint specifications,
Jacobian rows and dedicated endpoint-D gradient exactly once from the fixed training bank.
Take exactly one Adam moment/step update and replace only the primary inequality projection
with the qualified FP64 implementation. Every free-coordinate round must require primary
solver success, original-unit primal violation at most `1e-6`, normalized KKT residual at
most `1e-8`, and the exhaustive-support independent check; the active set must converge
within eight rounds with native bounds. Canonicalize once and retain the existing
idempotence, post-materialization `1e-6` linear, negative-D direction, finite/validity and
optimizer-transaction controls. A projection or numerical-control failure is terminal for
this run and cannot enter repair or fall back to the old projector.

For each valid proposal, try ordinary scales 1, 1/2, 1/4, 1/8, 1/16 and 1/32 and accept
the first passing all original gates. Only the scale-1/16 C25-only case may use the existing
guard-band repair, unchanged: the same eligibility, authoritative D row and one-quarter
finite difference, one fixed Jacobian, one zero-reference correction solve, internal
source-plus-0.0198 C25 target, source-plus-0.0199 actual acceptance, correction scales 1,
1/2, 1/4 and 1/8, no relinearization or second solve, and no additional Adam step. Retain
the pending one-step Adam state only for an accepted ordinary or repaired candidate. Any
other rejection restores the pre-update controller and optimizer exactly and stops.

The phase budget ends at **50 total accepted updates**, not 50 new updates and not a reset
of the existing 200-update outer budget. Evaluate the unchanged fixed development bank at
accepted totals 30, 40 and 50, requiring original-source preservation. Apply the existing
terminal criteria at each scheduled checkpoint. At total 50 also apply the already
registered mandatory gate: endpoint-D NRMSE must have improved by at least 25% from the
original source on both training and development, with all preservation gates passing.
If that mandatory gate fails, stop this route at update 50 without relaxing it or testing
hover. If it passes without terminal success, record the result before registering any
continuation toward total update 200. If terminal criteria pass, retain only the first
terminal checkpoint and run the already specified fresh qualification before any
closed-loop hover authorization.

Persist resume state before the next attempted update and preserve the corrected fitter's
existing interruption semantics: a recorded rejected proposal or failed milestone cannot
silently retry, and an interrupted first qualification cannot select another checkpoint.
Keep all run tensors/checkpoints ignored; commit only compact reports. Do not introduce
external recurrence, state machines, engineered history, actor changes, fresh data outside
the terminal path, closed-loop simulation, or promotion in this fitting phase.

Result: the FP64-primary continuation stopped safely at attempted update 21, with total
accepted updates still 20. The frozen input, source controller, raw Adam displacement,
pending one-step Adam state, FP64 projected displacement and complete primal objective all
reproduced their registered hashes or values exactly. The ordinary scale-1/16 candidate
also reproduced its registered parameter hash exactly and was repair-eligible for the same
sole endpoint-C25 outer-gate failure.

The unchanged repair again passed at correction scale 1.0, with every numerical, bound,
linear and nonlinear safety gate passing. Its endpoint-D NRMSE was `1.35364258`, endpoint-
C25 NRMSE was `1.68359423`, and its registered training metric tree matched all 25 values
within the existing reproduction tolerance. Its tensor SHA-256 was nevertheless
`7d5f2b6bc03f2fa51de96aecee42d5128caa66a7acec9bb3cabba6b6c7eb987a`, not the exact
step-audit hash `3d8d96f36944113ad942ebd1d30f6510998a3a337881b03e0ce3e6032e0f7308`.
Because exact tensor identity was a preregistered gate, update 21 was rejected and the run
was not retried. The controller and optimizer were restored exactly to update 20, the
stopped resume was persisted, and there was no development selection, terminal checkpoint,
fresh qualification, closed-loop hover or promotion. This is a control failure of exact
reconstructed-repair tensor identity across GPU processes, not a damping, transfer or
safety failure. See
[`artifacts/variable-height-full-native-d-first-fp64-corrected-fitting-v1/`](../artifacts/variable-height-full-native-d-first-fp64-corrected-fitting-v1/).

Before another fitting continuation, run one **restored update-21 accepted-transaction
snapshot qualification** in a new ignored run. Hash-lock the update-20 report and resume,
the frozen primary archive and qualification report, the passing restored step-audit
report, and the stopped continuation report. Do not modify or retry either stopped fitting
run. Preserve the actor, objective, constraints, ordinary scales, repair arithmetic,
training/development banks and all no-promotion restrictions.

The producer process must restore exact update 20, reproduce the frozen primary proposal
and pending Adam transaction using all existing exact FP64 controls, and run the unchanged
ordinary-then-repair selection exactly once. A primary failure, a non-repair-eligible
ordinary result, a repair control failure or a rejected repaired candidate is terminal;
there is no retry, alternate candidate or relaxed hash comparison. Do not require equality
to either previously recomputed repaired tensor hash. Instead require all preregistered
repair controls, at least 0.001 training endpoint-D improvement from update 20, and every
existing training preservation gate. Evaluate only that selected candidate on the existing
fixed development bank, explicitly as reused development rather than fresh validation;
require at least 0.001 endpoint-D improvement from update 20 and every original-source
development preservation gate.

Only after those gates pass, write one complete accepted-transaction archive containing
the selected parameter tensors, exact pending Adam state, repair constraint rows, the
authoritative endpoint-D row, full correction direction, ordinary and corrected metrics,
all scalar specifications, and provenance. Record semantic hashes for every tensor tree
and a SHA-256 for the complete archive. This is an ignored run artifact, not a Git or Git
LFS payload. Restore the in-memory controller and optimizer exactly to update 20 after the
archive is written, and require every locked input to remain byte-for-byte unchanged.

A separate verifier process must load the archive without recomputing the primary proposal,
repair gradients, projection or candidate. Require exact candidate-parameter and pending-
optimizer semantic hashes, exact equality to the archived tensor trees after load, no
change to the archive's byte hash, and training plus fixed-development replay agreement
under the already-established numerical-tree tolerances. It must also restore its source
controller and optimizer exactly after replay. Any producer or verifier failure preserves
the diagnostic archive/report but authorizes nothing.

A passing producer and verifier authorize only a separately preregistered continuation
that loads this audited accepted update-21 transaction directly and begins fresh proposal
generation at update 22. It does not retroactively pass the stopped continuation, establish
useful damping, authorize closed-loop hover, change the total-update-50 mandatory gate, or
permit automatic promotion. CPU repair and a general parameter-distance tolerance are not
part of this audit: moving only the solver would not remove GPU gradient variation, while
the stored snapshot gives future processes an exact state boundary.

Result: the accepted-transaction snapshot qualification passed. The producer performed
the single permitted reconstruction and selected the unchanged scale-1/16 proposal plus
scale-1.0 repair. Its candidate parameter hash was
`223d3eda98a5c7c3280b28782d4f46ac5845c290e14ca11898749712cbba02e3`; the exact pending
Adam hash remained `6743cc01e41e01ccd7183b8981acff29eb5af22a72c3039bb9d234cfcbde42d6`.
Training endpoint-D NRMSE was `1.35364187`, endpoint-C25 was `1.68359447`, and every
repair and preservation control passed. Fixed-development endpoint-D NRMSE was
`1.31926703`, an improvement of 0.00176442 from update 20, with all preservation gates
passing.

The producer archived the complete transaction, repair rows and correction direction in
the ignored 224 MiB file whose SHA-256 is
`fb4c3118348eee63e9d0365c64123a459aa8327cd39a056ae5496318cd6f9f42`. In a separate
AIRA-confined process, the verifier loaded that archive without recomputing the primary or
repair. Candidate parameters and optimizer were exact, both training and fixed-development
replays matched all 25 registered values, the archive remained byte-for-byte unchanged,
and both processes restored update 20 exactly. The full report SHA-256 is
`c8f665732e2ebe84b0ee09fea8dee431e4e6ca021a8035bca1e9276d831413e5`.
Sign fraction is still zero and teacher-aligned gain remains negative, so this establishes
an exact continuation boundary rather than useful damping or hover. No fresh data,
closed-loop simulation or promotion occurred. See
[`artifacts/variable-height-full-native-d-first-fp64-update21-snapshot-audit-v1/`](../artifacts/variable-height-full-native-d-first-fp64-update21-snapshot-audit-v1/).

Proceed with one **snapshot-seeded FP64-primary corrected fitting continuation to total
update 50** in a new ignored run. Do not alter or resume either prior stopped continuation.
Hash-lock the snapshot qualification report (SHA-256
`c8f665732e2ebe84b0ee09fea8dee431e4e6ca021a8035bca1e9276d831413e5`), its producer
report (SHA-256 `429e592696d114d0d8d5df396a32df205aae12707085d36e043928d87cd1fa3a`),
and the complete transaction archive above, in addition to the existing update-20 source,
graph, checkpoint, immutable cache and FP64 qualification inputs.

Create the new crash-safe resume by loading the archived candidate parameters and pending
Adam state exactly as accepted total update 21. Require their semantic hashes to equal the
qualified snapshot hashes, append one provenance-only repaired update-21 history record,
and retain development history only through total update 20. Do not recompute update 21's
primary proposal, repair gradient, constraint rows, projection or candidate, and do not use
its fixed-development result as a selection channel. Before generating update 22, require
the loaded training metric tree to reproduce the archived candidate under the established
numerical tolerance, the source metric tree to reproduce, the controller and pending Adam
hashes to remain exact, and all locked files to remain unchanged. Failure is terminal and
must persist a stopped resume without attempting update 22.

For updates 22 onward, use the same fixed training bank and endpoint-D-only Adam objective.
Generate each proposal's constraint specifications, Jacobian rows and dedicated endpoint-D
gradient exactly once, take exactly one Adam update, and use only the qualified FP64
bound-aware primary projection. Require solver success, original-unit primal violation at
most `1e-6`, normalized KKT residual at most `1e-8`, exhaustive-support independent
agreement, active-set convergence within eight rounds, native bounds, canonical
idempotence, post-materialization linear violation at most `1e-6`, negative endpoint-D
direction, finite/valid outputs and exact optimizer transaction accounting.

Try the unchanged ordinary scales 1, 1/2, 1/4, 1/8, 1/16 and 1/32. Only the unchanged
scale-1/16 C25-only case may enter the existing guard-band repair with its current
eligibility, authoritative D row, finite-difference probe, fixed Jacobian, solver and
acceptance margins, correction scales and no-extra-Adam rule. A rejected proposal,
projection failure, numerical-control failure or failed scheduled development preservation
restores the pre-update controller and optimizer exactly, persists a stopped result and is
not retried.

The phase still ends at 50 total accepted updates. Evaluate the unchanged fixed development
bank at totals 30, 40 and 50. Apply the same terminal criteria at every scheduled point and
the existing mandatory 25% source-relative endpoint-D improvement on both training and
development at total 50. A mandatory-gate failure closes this route without hover. Terminal
success permits only the already registered fresh qualification; closed-loop hover remains
unauthorized until that passes. Keep all generated checkpoints, resumes and the 224 MiB
snapshot under ignored `runs/`, commit only compact reports, and make no automatic
promotion or actor-interface change.

Result: the snapshot boundary loaded exactly and its parameter, pending-Adam and two
25-value training metric-tree controls all passed. Fresh ordinary proposals at totals 22,
23 and 24 were accepted at scales 1/16, 1/32 and 1/32. Training endpoint-D NRMSE improved
from `1.35364187` at accepted update 21 to `1.34913635` at update 24. Sign fraction remained
zero and gain remained negative, so the result was not eligible for hover.

Attempted update 25 stopped at the preregistered solver-success gate before installing a
candidate. The first free-coordinate L-BFGS result returned `ABNORMAL`, even though its
original-unit primal violation was only `3.31e-11`, normalized projected-gradient KKT
residual was `9.05e-11`, and both the exhaustive-support and SLSQP references passed. The
round identified 5,506 edge lower-bound crossings but returned before adding them to the
active set; subsequent authoritative clamping of that unfinished first-round result caused
the reported `0.048034` common-step-25 violation. This is a solver-status control failure,
not evidence that the completed bound-aware QP is infeasible.

The failed update-25 transaction restored controller and Adam exactly to update 24 and
persisted a stopped resume (SHA-256
`5a500afb19910b0fb19186d575b0f75455d4b0f72b8f4e592663693697dc34b0`). No scheduled
development data beyond the inherited update-20 record were evaluated, and there was no
terminal checkpoint, fresh qualification, closed-loop hover or promotion. See
[`artifacts/variable-height-full-native-d-first-fp64-snapshot-corrected-fitting-v1/`](../artifacts/variable-height-full-native-d-first-fp64-snapshot-corrected-fitting-v1/).

Before any further fitting, run one restored, frozen-input **update-25 exhaustive-support
primary-solver audit** in a new ignored run. Hash-lock the stopped snapshot-seeded report
(SHA-256 `55aa8e0848d2b93b5493c2c6478b93f30485bd810520f0814b126a918aec1691`), its resume
above, the graph, source checkpoint and immutable cache. Load and verify the exact accepted
update-24 controller and Adam state, discard only the recorded rejected update-25 history
entry in the audit copy, reproduce the accepted-update-24 training metrics, generate the
update-25 constraint specifications, rows, endpoint-D gradient and raw one-step Adam
displacement exactly once, and archive those tensors before solving. The source run and
resume must remain untouched.

Use the existing exhaustive active-support Lawson-Hanson nonnegative solve as the primary
dual solution in every bound-aware round. Enumerate all supports (at most 1,024 for the ten
constraints), select the minimum-objective full-KKT-feasible solution with deterministic
support-mask tie-breaking, and apply it to the primal displacement. Retain SLSQP as an
independent diagnostic and L-BFGS only as a non-authoritative diagnostic; do not require
reproduction or success of the historical L-BFGS status. Require finite coefficients,
original-unit primal violation at most `1e-6`, normalized KKT residual at most `1e-8`,
independent objective and Gram-induced primal agreement, and completion of the unchanged
monotonic edge-bound active set within eight rounds.

Reload the frozen archive from CPU and run the complete exhaustive-primary projection three
times from an empty active set. Require exact active/fixed-set hashes per round and compare
the complete learning-rate-scaled primal displacement plus objective at the existing
`1e-10` and `1e-12` relative tolerances; do not require equality of non-unique dual
coefficients. On each repeat require native bounds, canonical idempotence, post-
materialization linear violation at most `1e-6`, negative endpoint-D derivative, exact
one-step Adam accounting and the existing finite-difference direction check.

Only if every numerical control passes, evaluate the unchanged six ordinary scales on the
fixed training bank as a diagnostic, with the unchanged source-relative preservation and
minimum endpoint-D-improvement gates. Do not invoke or modify nonlinear repair, do not
evaluate development or fresh data, and do not retain the candidate, optimizer step or any
controller mutation. Classify numerical failure separately from “projector qualified but
no ordinary scale passed.” A passing audit authorizes only a separately preregistered
continuation from the exact accepted update-24 boundary; it does not pass the stopped run,
waive the total-update-50 mandatory gate, authorize hover or permit promotion.

For this solver audit, `pass` means that source controls and exhaustive-primary numerical
qualification pass; the training-only ordinary-scale result is reported in the
classification but is not part of numerical qualification and does not install update 25.
This preserves visibility of a possible unchanged repair-eligible outcome without invoking
repair inside the solver audit.

Result: the frozen producer passed. It restored exact accepted update 24, reproduced both
25-value training metric trees, generated update 25 once and wrote the ignored 200 MiB
archive with SHA-256
`5fc3b908962986c43f511565296950ca06f0f04d80a9d2b3266c9dbb2fcc4f61` before any
projection. Its controller, one-step Adam transaction, final restoration and all source
hashes passed. The separate verifier loaded that archive three times with exact semantic
hashes and byte identity.

The exhaustive primary result was identical on all three reloads. In round 1 it examined
all 1,024 supports, found one full-KKT-feasible support (mask 840), achieved normalized KKT
residual `1.16e-17` and original-unit primal violation `5.03e-16`. The required unchanged
SLSQP reference reported success but did not meet the registered accuracy: normalized KKT
was `7.00e-7`, Gram-induced primal distance from primary was `9.34e-6`, and objective
relative difference was `8.72e-11`. The first two exceed their `1e-8` and `1e-6` limits.
All three audit repeats therefore failed identically in the first free-coordinate round,
before adding the 5,506 newly identified edge-bound crossings. The complete bound-aware
projection, post-projection controls and ordinary scales were not tested.

This is an independent-reference accuracy failure, not demonstrated primary-solver or QP
infeasibility. The controller and optimizer were restored exactly to accepted update 24;
no candidate or Adam step was retained, no development or fresh data were evaluated, and
there was no hover or promotion. The full report SHA-256 is
`7b9532798c7bb55ea8d27bd06dd026d6de4c54847a10c108120a22bfc0bfc47a`. See
[`artifacts/variable-height-full-native-d-first-fp64-update25-exhaustive-audit-v1/`](../artifacts/variable-height-full-native-d-first-fp64-update25-exhaustive-audit-v1/).

Run one final bounded **SLSQP-support-polished exhaustive-primary audit** in a new ignored
run. Hash-lock the failed exhaustive audit's report, producer report and frozen archive
above, plus the same graph, checkpoint, cache manifest, stopped update-24 report and resume.
Regenerate no constraint rows, gradients, Adam step, metrics or candidate. A separate
verifier must load the existing CPU archive and require all semantic and file hashes before
solving. Preserve the failed audit and its archive unchanged.

In each free-coordinate active-set round, keep the exhaustive-support primary arithmetic,
support enumeration, selection and gates unchanged. Run SLSQP once with the same zero
initialization, analytic gradient, nonnegative bounds, `ftol=1e-12` and 10,000-iteration
limit. Determine a polishing support only from that SLSQP result using
`lambda_i > 1e-10 * max(1, ||lambda||_inf)`. Do not use the exhaustive primary's support.
On that fixed support, perform exactly one direct FP64 SVD solve of
`G_SS lambda_S = v_S`, with relative singular-value cutoff `1e-12`; require the support
matrix to be full rank. Set all other dual coefficients to zero. There is no retry,
alternate threshold, regularization, support search, iterative refinement or fallback to
an unpolished solver.

Require the polished reference coefficients to be finite and nonnegative, its normalized
full-original-QP KKT residual at most `1e-8`, and the resulting original-unit primal
violation at most `1e-6`. Compare the polished reference to exhaustive primary using the
unchanged Gram-induced primal relative-distance limit `1e-6` and dual-objective relative-
difference limit `1e-8`; do not compare raw dual coefficients. Report unpolished SLSQP and
L-BFGS results as diagnostics only.

Apply that reference in every round of the unchanged monotonic edge-bound active-set loop,
with no more than eight rounds. Run three complete projections from empty active sets and
fresh CPU archive reloads; require every run to pass, identical active/fixed-set paths, and
the same `1e-10` learning-rate-scaled primal and `1e-12` objective repeat limits. Do not
materialize or evaluate an actor candidate, run finite-difference or nonlinear trials,
evaluate development/fresh data, or retain any state. Require exact final restoration and
unchanged locked inputs/archive.

If any control fails, classify the reason and pause this exhaustive-primary integration
route; do not rotate to another reference or relax a threshold. A pass qualifies only this
revised frozen-input numerical reference and permits a separately preregistered update-24
continuation. It does not retroactively pass either failed run, install update 25, authorize
hover, alter the total-update-50 gate or permit promotion.

Result: the SLSQP-support-polished reference audit passed. Three fresh reloads of the
existing frozen update-25 archive followed identical three-round active-set paths: 5,506
edges fixed in round 1, 15 more in round 2, and convergence with 5,521 fixed edges and zero
box violation in round 3. Pairwise learning-rate-scaled primal and objective differences
were all exactly zero.

Unchanged SLSQP selected support mask 840 from its own coefficients in every round. The
single fixed FP64 SVD solve found that four-coordinate support full rank each time. Against
the unchanged exhaustive primary, maximum Gram-induced primal distance was `4.22e-16` and
maximum objective relative difference was `1.98e-16`. Polished normalized KKT residuals
were at most `3.45e-17`, and polished original-unit primal violations were at most
`2.81e-16`. All locked inputs and the archive remained unchanged. The full report SHA-256
is `9abe6a4b433927e6136395dbe7abec869bce8daccd5b03284fc91cb076f0e075`.

No fly controller was instantiated, no candidate materialized, no metrics, gradients or
Adam state regenerated, no state retained, no development/fresh data used, and no hover or
promotion occurred. The pass qualifies only the polished numerical reference. See
[`artifacts/variable-height-full-native-d-first-fp64-update25-slsqp-polished-audit-v1/`](../artifacts/variable-height-full-native-d-first-fp64-update25-slsqp-polished-audit-v1/).

Proceed with one new **frozen-update-25 polished-reference corrected continuation to total
update 50**. Do not alter or resume the stopped snapshot continuation. Hash-lock its report
and accepted-update-24 resume, the failed exhaustive audit's producer report and frozen
update-25 archive, and the passing SLSQP-support-polished report (SHA-256
`9abe6a4b433927e6136395dbe7abec869bce8daccd5b03284fc91cb076f0e075`), in addition to
the graph, checkpoint and immutable cache manifest.

Create a new crash-safe resume by copying exact accepted update 24, removing only the
recorded rejected update-25 history entry, retaining development history only through
total update 20, and setting the new protocol active. Before attempting update 25, require
the loaded controller and Adam semantic hashes to equal
`299dae786c06b5a64b9d9d29a9e365697ea5a36333caac5eb7c1b3ea2ffbaeed` and
`5213d7ee914c97e402bc1108b6c4fcc7be385b952388981865d388ba3fc8574f`, and reproduce the
source and accepted-update-24 training metric trees under the established tolerance.

For update 25 only, load the frozen constraint specifications, rows, endpoint-D gradient,
raw displacement and pending optimizer-after state directly from the qualified archive.
Require every archive semantic hash, exact equality of its current parameters and
optimizer-before state to accepted update 24, and the original one-step counter transaction
from 24 to 25. Do not regenerate its constraints or gradient, call Adam again, or use
development. Recompute the qualified exhaustive-primary plus SLSQP-support-polished
three-round projection from the frozen inputs. Require all registered per-round gates, the
exact projected-displacement SHA-256
`56cf914132ffd85356d3c28fcccc1416e409ee9ed122e770820bffa4fcaeea44`, and complete
projection primal objective `18712.47294252837` within the existing `1e-12` relative
tolerance.

Materialize the authoritative update-25 proposal exactly once and require native bounds,
canonical idempotence, post-materialization linear violation at most `1e-6`, a negative
endpoint-D direction, and exact archived Adam accounting. Before nonlinear selection, run
the existing scale-1/16 directional finite-difference check on the fixed training bank and
require its existing sign, minimum-change, relative-error, idempotence and bound gates.
This checks the full projected direction; it does not require scale 1/16 itself to pass
nonlinear preservation.

Then run the unchanged ordinary scales in descending order. Only the unchanged scale-1/16
C25-only case may enter the existing nonlinear repair with its current eligibility,
authoritative D row, finite-difference probe, fixed Jacobian and correction scales. Do not
require any recomputed repair tensor to equal a tensor produced in another process. Accept
update 25 only if every numerical, nonlinear, preservation and optimizer gate passes;
otherwise restore exact update 24, persist a stopped resume and do not retry. Treat the
loaded optimizer-after state as the archived pending update-25 transaction, not as a newly
executed optimizer step. If accepted, persist the selected controller tensors and Adam
state immediately so interruption never requires reconstructing update 25.

From update 26 onward, generate each fixed-training proposal's constraints, native gradient
and single Adam transaction exactly once. Use the same exhaustive primary with the
qualified SLSQP-only support threshold and one-shot FP64 SVD polish in every active-set
round. Require all primary, polished-reference, active-set, canonical, bound, linear,
direction and transaction controls before the unchanged ordinary-then-eligible-repair
selection. Any failure restores the pre-update transaction and stops without retry.

Keep the existing scheduled development checks at total updates 30, 40 and 50, including
the crash-safe rollback of controller, Adam, training metrics and accepted count on failed
preservation. Keep the first-terminal fresh qualification path and the original mandatory
total-update-50 gate requiring at least 25% source-relative endpoint-D improvement on both
training and development. A failed mandatory gate closes the route without hover; no
closed-loop run or promotion is permitted without fresh qualification. All resumes,
checkpoints and large archives remain ignored; commit only compact reports.

Result: the polished continuation stopped safely at attempted update 30 with accepted update
29 retained. Frozen update 25 reproduced the registered displacement hash and objective
exactly, passed its full directional finite-difference and transaction controls, and was
accepted by the existing C25-only repair at proposal scale 1/16 and correction scale 1.
Updates 26 through 29 passed ordinarily at scale 1/32. Training endpoint-D NRMSE reached
`1.34239793`, a `7.95%` improvement relative to the original source, but endpoint-D sign
fraction remained zero and teacher-aligned gain remained negative (`-0.32804`).

Update 30's projection, post-materialization and one-step Adam controls passed. No nonlinear
scale passed: scale 1/32 missed only endpoint C25 by `3.12e-5`, while the scale-1/16 repair
was ineligible because P20 exceeded its source-plus-0.02 limit by `1.95e-6`. The rejected
controller and Adam transaction were restored exactly. There was no new development,
terminal checkpoint, fresh qualification, hover or promotion. The complete report SHA-256
is `48dedef889546a84916e04052ae7d6207bc6368159327e90025236b32455a3f7`; see
[`artifacts/variable-height-full-native-d-first-fp64-polished-corrected-fitting-v1/`](../artifacts/variable-height-full-native-d-first-fp64-polished-corrected-fitting-v1/).
This closes the source-preserving local-repair family. The narrow P20 miss does not establish
infeasibility or insufficient neural recurrence, and update 29 is not promoted.

The next bounded diagnostic is one **joint teacher-learning C/P/D experiment** from the
original native source, not from update 29. Retain the connectome topology, signs, input
mapping, recurrence, foreleg outputs, immutable training/development caches, opened native
parameter families, learning rates and global gradient cap. Initialize a fresh Adam state
and accepted-update counter at zero; do not import an optimizer transaction from any prior
fit. Before training, freeze each denominator from the original source on the fixed training
bank and minimize

\[
J=\frac{1}{3}\sum_{k\in\{C,P,D\}}
\frac{\operatorname{MSE}_k}{\max(\operatorname{MSE}_{k,\mathrm{source}},0.25^2)}.
\]

Use the existing normalized component errors. C and P each average scenes and the three
supervision horizons with equal weight; D remains endpoint-only. Report interaction
separately but do not add it to this objective. This experiment tests joint learning without
intermediate source ceilings. Because the objective also changes, it is not an isolated
causal test of the ceilings alone.

Remove the intermediate C/P source-ceiling projection rows and nonlinear source-ceiling
selection gates. Replace every endpoint-D-only descent, directional finite-difference and
per-step `0.001` D-improvement test with the corresponding test of J. For each attempt,
generate the J gradient once and execute exactly one fresh Adam transaction. Require the
authoritative materialized direction to have negative `grad(J) dot displacement`, the
existing scale-1/16 complete-replay finite difference to show the same negative sign within
the existing relative-error tolerance, and an accepted trial to reduce actual J from the
current candidate by at least `1e-4`. D may temporarily worsen before a milestone.

Retain only the existing activated RPY rows in the numerical projection, along with the
finite-state, actuator-output, native-bound, canonical-idempotence and transaction controls.
Use the qualified polished FP64 projection when an RPY row is active. With zero active RPY
rows, projection is the identity raw proposal before authoritative native-bound
materialization. Disable the C25 nonlinear repair entirely: every selected candidate is an
ordinary trial. Keep the established backtracking grid and numerical tolerances; do not add
smaller scales or tune the objective after seeing results. Selection uses the fixed training
bank only. Failure of a numerical control or absence of an admissible scale restores the
exact pre-attempt controller and Adam state, persists a stopped resume and is not retried.

Allow at most 50 accepted updates. At update 25, stop unless endpoint-D improves by at least
15% relative to the original source and aggregate C and P are each no worse than source. At
update 50, require endpoint-D improvement of at least 25%, correctly signed endpoint damping
on at least 50% of training scenes, and C and P at every horizon no worse than source. Only a
candidate passing all update-50 training gates receives one development evaluation, where
the same D-improvement, sign, per-horizon C/P, RPY, validity and output gates apply. Any
failure closes this fixed objective configuration with no retrospective weight adjustment or
alternate checkpoint selection. Persist each milestone decision before continuing. Persist
`development_started` before the sole development evaluation; an interruption closes the
run and cannot authorize another candidate, selection or development retry. A pass
authorizes only a subsequent learning experiment; it does not authorize hover, gate flight
or promotion.

Result: the joint teacher-learning run stopped at its preregistered update-25 milestone.
All 25 updates were accepted and reduced J from `1.000000` to `0.541016`. Aggregate common
NRMSE improved from `2.533490` to `0.294855`, height from `0.593303` to `0.548998`, and
endpoint-D from `1.458401` to `1.265771`. The endpoint-D change is a `13.208%` improvement,
short of the fixed 15% gate. The damping response also remained wrong-signed in all scenes:
correct-sign fraction was zero and teacher-aligned gain was still negative (`-0.254692`).
This is attenuation of the source anti-damping response, not learned vertical braking.

Final roll, pitch and yaw source NRMSE remained within the 0.05 limit at `0.039849`,
`0.049969` and `0.049507`. The stopped resume retained exact accepted update 25 and Adam
counters of 25. No development data, closed-loop hover or gate flight were evaluated and
no controller was promoted. The failed endpoint is audit-only and must not be continued.
See
[`artifacts/variable-height-full-native-joint-teacher-learning-v1/`](../artifacts/variable-height-full-native-joint-teacher-learning-v1/).

### Preregistered teacher-attitude-assisted native-throttle curriculum

The next experiment changes the training responsibility boundary rather than extending
the failed endpoint or merely increasing D's weight. It is the first bounded use of Track
B from [`CONTROL_ARCHITECTURE.md`](CONTROL_ARCHITECTURE.md). Restart from the original
native source with fresh Adam and accepted-update counter zero. Keep the full graph,
transmitter signs, RGB and roll/pitch mappings, native recurrence, front-leg motor pools,
foreleg/stick plant, parameter families, learning rates, gradient cap and native parameter
bounds unchanged. Do not load the failed joint endpoint or its optimizer.

During this diagnostic only, the analytical teacher supplies roll, pitch and yaw motor
commands while the native fly supplies throttle. Combine those four motor commands before
the actual foreleg/stick dynamics; do not inject RC commands after the sticks. The fly still
computes all four outputs but receives no teacher action, phase, target height, velocity,
impulse, position, timer or external recurrent state. Remove source-relative R/P/Y output
constraints because those axes do not control the training plant. Report their drift, but
do not claim this is full-native hover. Arm/disarm remains with the episode supervisor.

Before optimization, run a fixed 64-case positive control in which the same analytical
teacher supplies all four axes through the foreleg/stick plant. It must succeed in at least
95% of cases under the exact assisted-hover success definition below. This is an end-to-end
dynamic control, not merely a settled command-conversion check. Failure stops before any
gradient or final-data use.

Training is nominal-mass and uses 50 Hz policy frames with 100 Hz plant integration. Sample
marker height, starting camera height, wall range, texture phases/gains and illumination
within the declared training split. Balance stationary markers and unannounced plus/minus
0.20 m marker steps, plus no impulse and unannounced plus/minus 0.15 or 0.30 m/s
vertical-velocity impulses. Up-steps are restricted to 0.55-0.65 m before and 0.75-0.85 m
after the step; down-steps are 1.35-1.45 m before and 1.15-1.25 m after it. Match their
camera-height family to the marker family while sampling continuous camera height
independently inside that band. Thus every training command segment remains outside the
held-out 0.90-1.10 m marker band and the held-out family combinations, although marker-step
sign is necessarily associated with low/high family in this training support. Randomize
event time independently of scene and continuous height. Do not apply a supervised loss
until two new RGB frames have followed an impulse. Store physical state trajectories and
renderer nuisance variables, not privileged actor features; render ordinary 320x200 linear
RGB again during learning.

Every collected or evaluated episode is six seconds after an airborne, hover-stick-settled
initialization. Initial vertical speed is sign-balanced with magnitude sampled uniformly in
0-0.20 m/s. Exactly half of cases have no marker step, one quarter step up and one quarter
step down. Independently, half have no impulse and one quarter each have positive or negative
impulses; nonzero magnitudes balance 0.15 and 0.30 m/s. Marker-step time is uniform in
1.0-2.0 seconds and impulse time in 2.0-3.0 seconds. Training blocks contain 16 teacher
histories and, where enabled, 16 student histories. A learning update draws eight dense
histories—eight teacher cases in block one, then four of each history type—and eight paired
motion examples. Selection and 25-frame window positions follow the persisted optimizer RNG.

Each recurrent training example starts from native zero state. Replay its preceding RGB and
roll/pitch frames without gradient to reconstruct the fly's own state, detach that burn-in
state, then differentiate a 25-policy-step window. Freeze the same detached burn-in tensor
for the gradient, directional finite difference and ordinary scale trials of that update;
those numerical checks must not silently differentiate through a parameter-dependent
prefix. Separately, a selected candidate must replay the sampled histories again from zero
and pass the actual full-prefix objective gate. Plant transitions and trajectory collection
are detached; never backpropagate through the long aircraft rollout. Teacher-history
trajectories use teacher throttle as well as teacher attitude. Student-history trajectories
use teacher attitude and the current frozen fly throttle. The first 25 accepted updates use
teacher histories only. Updates 26-100 use an equal teacher-history/student-history mixture.
Regenerate and freeze both trajectory banks before updates 26, 51 and 76, so the later three
blocks are bounded DAgger rather than one permanently off-policy dataset.

Minimize the equal average of two losses. The dense term is native throttle-motor MSE to the
analytical teacher at every eligible differentiated frame, normalized by the frozen RMS of
the teacher's correction about nominal hover motor drive, floored at 0.01 motor units. The
motion term is paired throttle-contrast MSE on histories whose terminal RGB and roll/pitch
are exactly equal but whose preceding vertical motion is opposite, normalized by frozen
teacher-contrast RMS with the same floor. This term is evaluated at multiple registered
terminal history lengths, not only the old 25-step endpoint. Teacher state and simulator
state provide labels only; no decoded value is fed to the actor.

Freeze both denominators from block one's teacher and motion banks before update one and use
them for the entire run. Paired-motion examples balance terminal history lengths 15, 20 and
25 policy steps, with a common zero-state visual prefix and smooth mirrored camera histories
that end at identical pose, RGB and roll/pitch. The target is the teacher's signed throttle
contrast at that endpoint. The implementation must verify exact endpoint equality and the
teacher/foreleg sign convention before allowing a gradient. The four training motion banks
use only training wall/floor style pairings, height-error amplitudes 0.05 and 0.10 m, and
vertical-speed amplitudes 0.15 and 0.30 m/s. The midpoint bank uses the same amplitudes with
held-out wall/floor pairings. The final bank uses held-out pairings and new height-error
amplitudes 0.075 and 0.125 m with speed amplitudes 0.20 and 0.35 m/s.

For each attempted update, generate this objective's gradient once and execute exactly one
Adam transaction. Require finite gradients, recurrent state and outputs; native bounds;
canonical idempotence; a negative authoritative objective direction; and a fixed-burn-in
scale-1/16 finite difference with matching sign and at most 20% relative error. The gradient,
finite difference and scale trials use exactly the same frozen eight dense examples, eight
motion examples, window positions, detached burn-in tensors, eligibility masks, weights and
denominators; this sampled-minibatch objective is not called the whole block objective. Test
the unchanged scales 1, 1/2, 1/4, 1/8, 1/16 and 1/32 in descending order. A scale is accepted
only if both its fixed-burn-in sampled objective and its separate zero-state full-prefix
replay of those same sampled examples improve over their respective current-controller
values by at least `1e-4`. There is no projection, nonlinear repair, alternate objective,
smaller scale or second optimizer call.

A nonfinite value, invalid transaction, failed direction or failed finite-difference control
is fatal and stops immediately after exact controller/Adam restoration. A finite proposal
for which no ordinary scale passes is an ordinary rejected attempt; restore controller and
Adam, persist the already-advanced sampling RNG, and stop after five consecutive ordinary
rejections. Thus a resume neither retries a consumed minibatch nor silently converts a
numerical failure into backtracking. Persist controller, optimizer, RNG, current frozen banks
and counters atomically.

The budget is 100 accepted updates and at most 125 attempted updates. At update 50, evaluate
exactly one fixed 64-case development bank and a separate 64-pair equal-endpoint motion bank.
Continue only with at least 50% assisted-hover success and at least 50% correctly signed
motion pairs. Do not tune from this result. There is no update-25 evaluation and no
alternate-checkpoint selection.

Only accepted update 100 may consume the 128-case final bank and 128-pair final motion bank.
An assisted-hover episode succeeds only if it has no ground contact or invalid state and,
over its final two seconds, height RMSE is at most 0.10 m, vertical-speed RMS is at most
0.10 m/s and roll/pitch tilt RMS is at most 5 degrees. Require at least 90% success, at least
90% correct motion-response sign and teacher-aligned motion gain in 0.5-1.5. On the exact
same cases, keep legal roll/pitch live but freeze RGB at a target-aware frame. For marker-step
cases, use the first frame rendering the new marker position before the resulting action can
move the plant. For stationary-marker cases, freeze at 1.5 seconds; this precedes every
possible impulse. The paired live-minus-frozen success difference must have a 95% bootstrap
lower bound above zero.

For the 64-case positive-control and midpoint banks, split 32 cases over the two training
marker-height families, 16 over the unseen absolute marker band and 16 over the unseen
marker/camera-family combinations. The final bank uses 64 unseen-absolute-marker and 64
unseen-combination cases. Every bank balances the event strata above and samples fresh
continuous heights and renderer nuisances. The final motion bank uses held-out style
combinations and trajectory amplitudes not used by the four training motion banks.
For unseen-absolute-marker step cases, the pre-step marker lies in a training band and the
post-step target lies in 0.90-1.10 m; stationary cases remain in that held-out band. For
unseen-combination cases, every stationary or stepped marker segment remains in its training
marker family while the camera starts in the opposite family. No acceptance bank silently
relabels a training-support step as held out.

Use seed `380983` for the teacher positive control and `410983` for optimizer-side sampling.
The four frozen blocks use teacher-history seeds `381983`, `382983`, `383983`, `384983`;
blocks two through four use student-history seeds `392983`, `393983`, `394983`; and their
paired-motion seeds are `401983`, `402983`, `403983`, `404983`. Midpoint assisted-hover and
motion seeds are `420983` and `420984`. Once-only final assisted-hover and motion seeds are
`430983` and `430984`, with `440983` for paired bootstrap resampling. The frozen-vision
control reuses the exact final cases rather than sampling another bank. Report all concrete
seed assignments. Final data remain unopened unless update 100 is reached. A pass authorizes
only a separately preregistered reintegration/handoff of native attitude axes. It does not
promote a hover controller or authorize gate training.

This curriculum tests continuous feedback on the states the controller visits and removes
the nearly saturated source-R/P/Y preservation rows from the vertical-learning experiment.
It does not assert that those rows destroyed the old gradient: the polished projection
retained essentially all J descent. It also does not close Track A's native timescale and
multi-terminal temporal-credit questions, or the later controlled FeCO test.

Result: the teacher-attitude-assisted native-throttle run stopped safely at attempted update
20 with accepted update 18 retained. The 64-case teacher positive control passed at 100%
hover success, its foreleg motion control had the correct sign for all 24 pairs, and every
paired endpoint image was exactly equal. The source controller was wrong-signed on all 24
training motion pairs, with teacher-aligned gain `-0.029651`.

Eighteen updates were accepted; attempt 8 was one ordinary restored rejection. Attempt 20
had finite gradients, recurrent states, outputs and objectives, and its one pending Adam
transaction advanced all three counters from 18 to 19. The materialized proposal nevertheless
had positive current-objective direction, `grad(J) dot displacement = +0.961656`, so the
mandatory direction control failed before finite difference or ordinary scale selection.
Controller and Adam were restored, including the exact pre-attempt optimizer hash and counters
of 18. The run is closed and must not be resumed.

No midpoint or final data were generated, no assisted-hover claim was tested, and no native
attitude reintegration, full-native hover, gate flight or promotion was authorized. The full
report SHA-256 is `ca1562b8764174ac804185104eeb365fa2826241c7f292430d0a81700c177a34`;
see [`artifacts/variable-height-native-throttle-assisted-v1/`](../artifacts/variable-height-native-throttle-assisted-v1/).

### Preregistered accepted-18 optimizer attribution audit

Run one restored, training-only audit to determine whether Adam's first-moment inertia caused
the finite non-descent stop. Hash-lock the stopped report and resume above, accepted-18
controller hash `05cb3e44f66c8f19e6cc147c617a6496e8023edd9c97f07be99e0bd1859a46f8`,
optimizer hash `5b9df11d8396df8808af3fab1b916fa9d1b0a1b88031df2439e917152b52da0e`,
block-one teacher-history hash
`69993c6de95bfc150b4f2277a53c93da7aa2b6a94c9932d9f76c2893af3d1f62`, and
motion-bank hash `089bbb8cc72980ca6321f9919227cad5bda1a746041e64426a29455090be6fbd`.
Use only the exact failed attempt-20 sample, whose persisted sample-spec hash is
`0042747f8c2430f4024b69c0234aec4eb601c9b2524f90e144bfe6baefa7b175`.
Do not generate or inspect midpoint, final or other fresh data.

Reload accepted update 18 and its exact Adam state, reconstruct the same zero-state prefixes
and detached burn-ins, and generate the sampled objective gradient once. First reproduce the
reported finite current fixed-burn objective, full-prefix objective and original-Adam
materialized direction within `2e-5` absolute error. Compare that original transaction with
one counterfactual Adam transaction using `betas=(0, 0.999)`: retain the accepted-18 step
counters and second moments, while the next β1=0 update replaces rather than accumulates the
first moment. For each direction, report total and per-parameter-family `grad(J) dot
displacement` both before native bounds/canonicalization and after authoritative scale-one
materialization. This separates first-moment inertia from clipping and boundary effects.

For the β1=0 direction only, run the unchanged scale-1/16 fixed-burn finite difference and
require a negative measured and autograd direction, at least `1e-8` absolute measured change,
at most 20% relative error, finite recurrent states/outputs, native bounds and canonical
idempotence. Then test the unchanged scales 1, 1/2, 1/4, 1/8, 1/16 and 1/32 in descending
order on the exact same frozen sample. Require at least one scale to improve both the fixed-
burn and independent zero-state full-prefix objectives by `1e-4`. This audit selects nothing:
restore and hash-verify the exact controller and optimizer regardless of outcome, and retain
no candidate or optimizer state.

As descriptive training-only evidence, evaluate the original source and accepted-18
controller uniformly over all eligible frames of the frozen block-one teacher histories and
all 24 frozen motion pairs. Report teacher-target throttle error, accepted-18 source-output
drift normalized by the frozen dense denominator, and motion sign, gain and NRMSE overall and
by history length. These measurements are not generalization or hover evidence and are not
an alternate checkpoint selection.

The audit passes only if every identity/reproduction/restoration control passes, original
Adam remains a positive-direction proposal, and β1=0 passes its direction, finite-difference
and ordinary-scale gates. A pass authorizes only preregistration of a new run restarted from
the original source with `betas=(0, 0.999)` and otherwise unchanged curriculum, learning
rates, backtracks, budgets, midpoint/final seeds and gates. It does not authorize resuming
accepted update 18, evaluating its unopened data, assisted hover, native attitude
reintegration, gate flight or promotion. Failure closes this optimizer route.

Result: the attribution audit passed. It reproduced original Adam's materialized direction at
`+0.961657` and changed it to `-5.662634` with β1=0 while retaining the accepted-18 second
moments and exact 18-to-19 counter transition. The β1=0 scale-1/16 finite difference had
10.370% relative error and reduced the fixed objective by `0.317212`. Scale 1/2 passed both
objective gates, improving fixed-burn and zero-state full-prefix objectives by `0.558523` and
`0.494581`. Controller and optimizer hashes were restored exactly and no candidate was
retained.

Descriptively, accepted update 18 reduced frozen teacher-history throttle RMSE by 15.83%, from
`0.079912` to `0.067264`. Motion NRMSE improved only 0.51%, from `1.030300` to `1.025025`,
and all 24 training pairs remained wrong-signed. This is training-only evidence and does not
establish hover. No midpoint or final data were opened. The full report SHA-256 is
`cc4f66e168e4b6f2b1f7157c87584c24bfbfbf5eca579d8abdd4c3b13f53030c`; see
[`artifacts/variable-height-native-throttle-optimizer-attribution-v1/`](../artifacts/variable-height-native-throttle-optimizer-attribution-v1/).

### Preregistered source-restarted β1=0 assisted-throttle curriculum

Run one new teacher-attitude-assisted native-throttle experiment from the unchanged original
visual-hover source—not accepted update 18. Incorporate every actor, simulator, case-support,
teacher/student-history, paired-motion, detached-burn-in, objective, backtracking, numerical,
budget, midpoint, final, frozen-vision, bootstrap and interpretation rule from the curriculum
registered above. Change exactly one fitting choice: fresh Adam uses `betas=(0, 0.999)` rather
than `(0.9, 0.999)`. Both first and second moments start empty at the original source; β1=0
removes cross-update first-moment inertia while retaining Adam's within-run second-moment
coordinate scaling. Keep the same parameter families, gradient-norm cap and learning rates.

Reuse positive-control seed `380983`, training-history seeds `381983` through `384983`, student-
history seeds `392983` through `394983`, paired-motion seeds `401983` through `404983`, and
optimizer-sampling seed `410983` to make the optimizer change causally comparable. The update-
50 evaluation seeds `420983` and `420984`, update-100 seeds `430983` and `430984`, and bootstrap
seed `440983` remain valid because the stopped predecessor never generated those data. Do not
replace, preview or tune against them.

Record optimizer betas in every protocol manifest and resume, and reject a resume whose
param-group betas do not match `(0, 0.999)`. Retain the exact finite raw-gradient, recurrent-
state/output, bounds, canonical, one-transaction counter, materialized negative-direction and
scale-1/16 finite-difference controls. Test the same ordinary scales and require both fixed-
burn and zero-state full-prefix improvement by `1e-4`. Any numerical failure is terminal; a
finite no-scale proposal is an ordinary rejection, with the same five-consecutive and 125-
attempt limits. Preserve the same crash-atomic midpoint, final and terminal transitions.

Before opening formal seeds, run one complete disposable integration update using only the
existing `990983`, `990984` and `990985` nonformal seeds with β1=0, including CPU-canonical
resume round-trip validation. Then execute the formal run once at
`runs/variable-height-hover/native-throttle-assisted-beta1-zero-001`. The unchanged update-50
gate requires at least 50% assisted-hover success and 50% correctly signed motion pairs. The
unchanged update-100 gate requires at least 90% assisted-hover success, 90% correct motion sign,
motion gain 0.5-1.5 and a positive 95% live-minus-frozen bootstrap lower bound. A pass still
authorizes only a separately preregistered native-attitude reintegration experiment; it does
not itself promote a full-native hover controller or authorize gate training.

Result: the source-restarted β1=0 run stopped safely at attempted update 6 with accepted
update 5 retained. Its teacher hover and foreleg motion positive controls passed, every paired
endpoint image was exactly equal, and the unchanged source was again wrong-signed on all 24
training motion pairs. The first five attempts were accepted at scales 1/2, 1, 1/2, 1/2 and
1/4.

Attempt 6 had finite gradients, recurrence, outputs, objectives and projected parameters. Its
one β1=0 transaction advanced all three pending counters from 5 to 6 and remained a descent
direction: autograd and independently materialized values were `-0.785668` and `-0.785674`.
The mandatory scale-1/16 finite difference reduced the fixed objective from `1.046161` to
`1.020021`, but its measured derivative was `-0.418238`; the 46.77% relative error exceeded
the frozen 20% limit. This was therefore a fatal numerical-control failure before ordinary
scale trials. The transaction was discarded, and the stopped resume exactly retains the
pre-attempt optimizer hash and counters of 5.

No midpoint or final data were generated, no assisted-hover claim was tested, and no native
attitude reintegration, full-native hover, gate flight or promotion was authorized. The run
is closed and must not be resumed. The full report SHA-256 is
`1b951c0613ebe904cdb39186a053fcba72d5da7408170b73fac85693b915cc36`; see
[`artifacts/variable-height-native-throttle-assisted-beta1-zero-v1/`](../artifacts/variable-height-native-throttle-assisted-beta1-zero-v1/).

### Preregistered accepted-5 Taylor-convergence audit

Run one restored, training-only audit of the failed attempt-6 direction. Hash-lock the stopped
β1=0 report and resume at
`1b951c0613ebe904cdb39186a053fcba72d5da7408170b73fac85693b915cc36` and
`330fa83864a711a7630df8109f1a0550360b111f950e6572b25a4919b978acb3`; the accepted-5
controller and optimizer hashes are
`1c28c49770687e4db164e16b02eb65095ef32661dde24a7595626ac14e934e7b` and
`9ecbe922bcfc0bf5a6a14edc49836ac803af600363211bd75903aea7723d441f`. Require the
frozen block-zero teacher and motion hashes
`69993c6de95bfc150b4f2277a53c93da7aa2b6a94c9932d9f76c2893af3d1f62` and
`089bbb8cc72980ca6321f9919227cad5bda1a746041e64426a29455090be6fbd`, objective-scale
hash `53f780b8d08feeab98cf147f3ce3cbb31315eadfe8c7bc3a487fd5dd3526c810`, and exact
failed sample-spec hash `fd0ea38428fdff8be75421451537173ac8ba5066ac0ac9891c86d993a1c951e7`.
Create an exclusive, no-replay start marker before computation. Do not generate fresh cases or
open midpoint or final data.

Reconstruct the exact dense sample, motion examples and detached burn-ins at accepted update
5. Evaluate the unchanged fixed-burn baseline three times. The first replay is the single
authoritative `J0` for every objective change, derivative, residual and reproduction
comparison; the other two are used only to define replay noise as the maximum pairwise
absolute difference among the three objectives. Generate the sampled raw gradient once and
one β1=0 Adam proposal from the restored optimizer. Reproduce, within `2e-5` absolute
error, the recorded fixed-burn objective `1.0461612939834595`, full-prefix objective
`1.0461606103926897`, materialized direction `-0.7856744796120311`, scale-1/16 candidate
objective `1.0200214385986328`, and scale-1/16 measured direction
`-0.41823768615722656`. Require exact 5-to-6 pending counters, finite gradients, recurrence,
outputs and reports, and passing bounds and canonical controls.

Using that one proposal and those same fixed burn-ins, evaluate every scale `1/16`, `1/32`,
`1/64`, `1/128`, `1/256` and `1/512`; do not stop at the first passing scale. For each,
materialize the actual post-projection displacement and report dense and motion losses,
objective change, raw-gradient predicted change, measured and predicted directional
derivatives, the unchanged symmetric relative error, signed and absolute Taylor residual,
parameter-family displacement RMS, bounds/canonical controls, and recurrent/output
finiteness. Also report adjacent-scale Taylor-residual ratios and whether a residual reduction
is approximately quadratic, defined descriptively as a halving ratio from 1/8 through 1/2
when both residuals exceed the replay-noise floor.

A scale has local derivative agreement only if controls and all values are finite, predicted
and measured directions are both below `-1e-8`, relative error is at most 20%, and its absolute
objective change exceeds `max(1e-8, 10 * replay_noise)`. The audit finds local derivative
convergence consistent with finite-step curvature only if every identity, reproduction and
restoration control passes, scale 1/16 reproduces its registered failure, and two adjacent
scales among `1/32` through `1/512` have local derivative agreement. Restore and hash-verify
the accepted-5 controller and optimizer regardless of outcome; retain no candidate, optimizer
transaction or selected continuation scale. Write once to
`runs/variable-height-hover/native-throttle-beta1-zero-taylor-audit-001`.

A result consistent with finite-step curvature authorizes only preregistration of a new
source-restarted run that separates a sufficiently local derivative-check probe from ordinary
optimization-step selection. It does not retroactively pass or resume accepted update 5,
establish useful motion damping or hover, open evaluation data, authorize native attitude
reintegration or gate flight, or promote a controller. Above-noise nonconvergence instead
requires investigation of gradient/replay alignment before further training. If the smaller
objective changes do not clear the registered noise threshold, classify the audit as
noise-limited and inconclusive, not as evidence of a gradient defect.

Result: the accepted-5 Taylor-convergence audit passed every identity, reproduction,
numerical and restoration control. Three baseline replays measured maximum objective noise
of only `3.5763e-7`, setting the registered objective-change threshold to `3.5763e-6`. The
β1=0 transaction again advanced all pending counters from 5 to 6 and reproduced the full
materialized direction at `-0.785671`.

The scale-1/16 failure reproduced at 46.77% relative error, and scale 1/32 remained just over
the 20% limit at 23.33%. Scales 1/64, 1/128, 1/256 and 1/512 passed at 11.65%, 5.81%, 2.90%
and 1.41%; their objective changes all remained far above the noise threshold. The absolute
Taylor residual ratios after each step halving were `0.24946`, `0.24966`, `0.24951`,
`0.24942` and `0.24371`, so the residual decreased essentially quadratically across the
entire registered ladder. This is strong local derivative convergence consistent with
finite-step curvature, not retroactive acceptance of attempt 6.

Controller and optimizer hashes were restored exactly; no candidate, transaction or
continuation scale was retained, and no midpoint or final data were opened. This establishes
neither motion damping nor hover. The full report SHA-256 is
`0ee33e31f913a3de8e3b27934c1205706f4a417c7b3a33f12e190d4084930e50`; see
[`artifacts/variable-height-native-throttle-taylor-audit-v1/`](../artifacts/variable-height-native-throttle-taylor-audit-v1/).

### Preregistered source-restarted β1=0 derivative-ladder curriculum

Run one new teacher-attitude-assisted native-throttle experiment from the original visual-
hover checkpoint, not accepted update 5 or 18. Incorporate every actor, simulator, parameter,
β1=0 Adam, case-support, teacher/student-history, paired-motion, detached-burn-in, objective,
ordinary backtracking, budget, midpoint, final, frozen-vision, bootstrap and interpretation
rule from the registered β1=0 curriculum. Lock the passed Taylor-audit report at
`0ee33e31f913a3de8e3b27934c1205706f4a417c7b3a33f12e190d4084930e50`. Change only
the derivative-validation control described below; learning rates, gradient clipping,
optimizer transactions, ordinary candidate scales and acceptance thresholds remain unchanged.

For every attempted update, retain the exact sampled objective, raw gradient, one pending Adam
transaction and independently materialized full proposal. Evaluate the current fixed-burn
objective three times with the same sample and detached burn-ins. The first replay is the
authoritative `J0` used for every probe and ordinary-candidate comparison; the other two
measure replay noise, defined as the maximum pairwise objective difference. All three must be
finite. The pending full proposal must retain the existing finite negative-direction, exact
counter, parameter-bound and canonical-idempotence controls.

Probe the pending direction in the fixed order `1/16`, `1/32`, `1/64`, `1/128`, `1/256`,
`1/512`. At each scale, use the actual post-projection displacement and require finite
recurrence, outputs and reports, passing bounds/canonical controls, predicted and measured
directions below `-1e-8`, symmetric relative error at most 20%, and absolute objective change
greater than `max(1e-8, 10 * replay_noise)`. Continue until the first two adjacent probes pass;
then stop probing. A single passing probe is insufficient. Probe candidates are diagnostic
only: never retain one, use its scale to choose the optimizer step, or reuse its result as an
ordinary candidate evaluation.

Only after two adjacent probes pass, run the unchanged ordinary scales `1`, `1/2`, `1/4`,
`1/8`, `1/16`, `1/32` from largest to smallest. Retain the first scale that independently
improves both the fixed-burn objective and zero-state full-prefix objective by at least `1e-4`.
As before, a finite no-scale result is an ordinary rejection; any nonfinite recurrent state,
output, report, parameter-bound/canonical failure, invalid transaction or full-proposal
direction is a fatal numerical-control failure.

If no adjacent probe pair qualifies, restore the controller and optimizer and stop the run
without ordinary trials. Record `derivative_probe_numerical_failure` for any nonfinite or
invalid control; otherwise record `derivative_probe_noise_limited_inconclusive` when no two
adjacent smaller objective changes clear the registered noise threshold, or
`derivative_probe_above_noise_nonconvergence` when an adjacent above-noise pair exists but does
not meet local derivative agreement. Do not add smaller probes, weaken thresholds, resample or
retry during the run. Every failed attempt must preserve the established atomic transition,
sampling-RNG and exact controller/optimizer restoration rules.

Reuse positive-control seed `380983`, training-history seeds `381983` through `384983`,
student-history seeds `392983` through `394983`, paired-motion seeds `401983` through `404983`,
and optimizer-sampling seed `410983`. This reuse is intentional so the sole control change is
causally comparable. The update-50 seeds `420983` and `420984`, update-100 seeds `430983` and
`430984`, and bootstrap seed `440983` remain unopened. Before formal execution, run one
complete disposable ladder update and CPU-canonical resume round trip with only nonformal
seeds `990983`, `990984`, `990985`, including injected numerical-, noise- and above-noise-
failure rollback tests. Then execute the formal run once at
`runs/variable-height-hover/native-throttle-assisted-beta1-zero-ladder-001`.

The unchanged update-50 gate requires at least 50% assisted-hover success and 50% correctly
signed motion pairs. The unchanged update-100 gate requires at least 90% assisted-hover
success, 90% correct motion sign, motion gain 0.5–1.5 and a positive 95% live-minus-frozen
bootstrap lower bound. Passing still authorizes only a separately preregistered native-
attitude reintegration experiment; it does not itself promote full-native hover or authorize
gate training.

Result: the derivative-ladder control solved the numerical stopping problem but the formal
curriculum failed its fixed update-50 behavioral gate. All 53 attempts found two adjacent,
above-noise local derivative probes; 50 updates were accepted, three were ordinary restored
rejections, and no numerical or transaction guard failed. The stopped Adam counters are
exactly 50. This run is closed and must not be resumed.

Assisted hover succeeded in 23/64 cases (`0.359375`) against the fixed 50% midpoint gate.
There were no ground contacts or invalid states, mean final-window height RMSE was
`0.247881 m`, and vertical-speed RMS was `0.063173 m/s`. More importantly, motion-response
sign was wrong in all 64 equal-endpoint pairs at every 15-, 20- and 25-frame horizon.
Prediction RMS was `0.001636` motor units against target RMS `0.043659`, teacher-aligned gain
was `-0.024702`, and NRMSE was `1.025002`.

The accepted-step record exposes objective dominance rather than a derivative defect. Across
the 50 sampled accepted candidates, fixed-burn dense-loss improvement summed to `125.304640`
while motion-loss improvement summed to only `0.006764`; 21 accepted candidates actually
worsened their sampled motion loss. Full-prefix selection showed the same imbalance. These
within-step sums are not a shared-bank learning curve, but they demonstrate a severe selection
imbalance consistent with static/common throttle imitation overwhelming a much weaker
recurrent motion contrast. They do not by themselves prove that this imbalance caused the
terminal failure. This does not show inadequate fly recurrence; the next experiment must
first make visual damping a genuine optimization responsibility, then reintroduce
collective/height calibration rather than repeating this mixed loss.

Final data were never generated or opened. No native-attitude reintegration, full-native
hover, gate flight or promotion was authorized. The full report SHA-256 is
`549259e08db3dd1dc43afda0ea85b76f86fac115a5a28bb0aca6700dbccb9ed9`; see
[`artifacts/variable-height-native-throttle-assisted-beta1-zero-ladder-v1/`](../artifacts/variable-height-native-throttle-assisted-beta1-zero-ladder-v1/).

### Preregistered assisted native-throttle motion-only learnability curriculum

The next experiment tests the specific optimization-priority diagnosis; it does not resume
or tune the stopped update-50 controller. Restart from the original `paired-dynamic-001`
checkpoint with fresh Adam, beta values `(0, 0.999)` and accepted-update counter zero. Keep
the full graph, transmitter signs, 320x200 RGB and roll/pitch actor inputs, native MaleCNS
recurrence, front-leg motor outputs, parameter families, learning rates, gradient cap,
parameter bounds and projection unchanged. Add no velocity, accelerometer, target height,
phase, timer, decoded feature, external state or engineered history.

Optimize only the existing normalized equal-endpoint paired-motion contrast loss. There is
no dense imitation term, source-output anchor, height/common constraint or physical rollout
in this learnability experiment. This is deliberately not a larger arbitrary weight on D:
visual damping is the sole optimization responsibility. Common throttle, height response,
all four raw motor outputs and source-relative output drift are recorded but do not select a
candidate. The result cannot establish hover even if it passes.

This diagnostic changes both objective composition and motion sampling: it removes dense
teaching and replaces stochastic eight-pair samples with an exact complete bank. It therefore
tests whether that combined learning setup can expose native damping learnability; it cannot
uniquely attribute a pass or failure to loss competition versus sampling variance.

Generate one fixed 24-pair training bank at seed `450991`. Construct the exact full factorial
of marker-error magnitude 0.05/0.10 m, marker-error sign minus/plus, endpoint vertical-speed
magnitude 0.15/0.30 m/s and history length 15/20/25 policy frames, then apply one persisted
seeded permutation. Thus every combination occurs exactly once and every horizon contains
eight pairs. Use ordinary training wall/floor style combinations. Each pair starts with the
unchanged 25-frame neutral zero-state prefix, follows the existing smooth mirrored return
history and ends with bit-identical pose, RGB and roll/pitch under opposite signed endpoint
motion. The analytical teacher supplies only the target throttle contrast; it never supplies
an actor input or physical action.

Before the source replay, freeze the single normalization denominator as the RMS analytical-
teacher contrast over all 24 training pairs, floored at `0.01` motor units. Reuse that exact
scalar for every gradient, replay, derivative probe, ordinary candidate, milestone and
development evaluation. Never recompute it from candidate outputs or development targets.

Retain the existing motion trajectory for causal comparability. Its smooth-return function
has the declared continuous endpoint derivative but includes a fixed 5.5 cm excursion, so
the sampled last-frame displacement need not have exactly the labelled velocity magnitude.
Before optimization, report the finite-difference last-frame velocity/label ratio for every
case and require only finite values and matching nonzero sign; do not conceal or post-hoc
correct the magnitude mismatch. Also require finite source recurrence and outputs, exact
endpoint-image equality, a finite nonzero teacher target in every pair, and the established
teacher-to-foreleg sign control.

Every update uses all 24 training pairs in the same persisted order. Reconstruct each shared
neutral prefix from native zero state without gradient, detach it, differentiate every frame
of the 15-, 20- or 25-frame mirrored response, and accumulate the exact equally weighted
whole-bank loss before one Adam transaction. There is no case sampling RNG, minibatch
selection or bank regeneration. For candidate acceptance, independently replay all 24 cases
from zero state so an update must improve both its frozen-burn objective and its actual
full-prefix objective by at least `1e-4`.

Use the qualified derivative protocol unchanged: evaluate three identical current-objective
replays, define replay noise as their maximum pairwise difference, and test the pending
materialized direction at `1/16`, `1/32`, `1/64`, `1/128`, `1/256`, `1/512` until the first
two adjacent above-noise probes agree with the negative predicted derivative within 20%.
Those probes remain diagnostic and cannot select or supply an update. Then try the unchanged
ordinary scales `1`, `1/2`, `1/4`, `1/8`, `1/16`, `1/32` in descending order. Require finite
states, outputs, gradients and reports, a valid one-step Adam transaction, native bounds,
canonical idempotence and exact restoration on every rejected or terminal attempt. A finite
no-scale result stops on its first occurrence: because the bank, order, controller and Adam
state are fixed, retrying would reproduce the same proposal and risk selecting replay noise.
A failed direction, derivative ladder, transaction, bound, canonical or numerical control is
fatal.

Allow at most 100 accepted and 100 attempted updates. At accepted update 25, evaluate the
unchanged complete training bank and stop unless NRMSE has improved at least 25% relative to
its frozen source value and at least 50% of all pairs have the correct sign. Persist this
decision; there is no alternate-checkpoint selection. Continue only on a pass.

At update 100, require at least 90% correct sign overall and within each horizon, and require
teacher-aligned gain in 0.5-1.5 overall and within each horizon. Also require motion NRMSE to
have improved at least 50% relative to source, exact endpoint images, finite recurrence and
outputs, and legal motor bounds. Only a passing update-100 training result may generate the
single disjoint 24-pair bank at seed `460991`. That bank uses the same exact factorial
amplitudes and horizons but a separate permutation and held-out wall/floor style
combinations. It must independently pass the same sign, per-horizon gain, 50%-relative-NRMSE,
endpoint-equality, finiteness and output-bound gates relative to its own source replay.

The development result is first-terminal-checkpoint-only: record that evaluation has started
before generating it, never try another checkpoint or seed, and do not train from it. A pass
shows only that the present native visual/recurrent circuit can learn the required damping
sign when damping is its actual optimization priority. It authorizes a separately
preregistered staged teacher-attitude-assisted curriculum that preserves learned damping
while restoring collective and height calibration. A failure at update 25, update 100 or
development instead sends work to a temporal-credit/time-constant experiment, not a repeat
with a post-hoc loss weight. Neither outcome authorizes native-attitude reintegration,
full-native hover, gate flight or promotion.

Result: the motion-only experiment stopped safely before its update-25 milestone. The exact
24-combination bank, endpoint-image identity, source finiteness, teacher/foreleg sign and
sampled-trajectory sign controls passed. The registered trajectory limitation was sizeable:
last-frame displacement represented 1.27-2.93 times the declared continuous endpoint-speed
magnitude, although its sign was correct in every case.

All first 17 proposals passed the derivative ladder and were accepted at ordinary scale 1.
Full-prefix motion NRMSE improved only 2.92%, from `1.031749` to `1.001594`; correct-sign
fraction stayed 0/24 and aligned gain moved from `-0.030875` to `-0.001592`. Prediction RMS
fell 94.75%, from `0.002292` to `0.000120` motor units against an unchanged `0.043661` target.
Thus the optimizer erased the wrong-signed response rather than crossing zero to learn
braking. The descriptive height-sign response simultaneously collapsed from `0.038864` to
`0.001979`, while pair-common throttle drifted `0.270700` RMS from source. This is a
degenerate loss-reduction path, not useful native recurrence.

Attempt 18 remained finite and its pending Adam transaction and full proposal were valid.
The scale-1/16 derivative probe passed with 1.41% relative error and objective change
`-1.35303e-5`. The adjacent 1/32 change, `-6.13431e-6`, fell just below the registered
`6.91662e-6` replay-noise threshold; every smaller probe was also below threshold. Without
two adjacent above-noise passing probes the run correctly stopped as
`derivative_probe_noise_limited_inconclusive`. Controller and Adam were restored exactly,
with counters retained at 17.

The disjoint development bank was never generated. No staged curriculum, hover, native-
attitude reintegration, gate flight or promotion was authorized, and this endpoint must not
be resumed. The result rejects the tested combined motion-only/full-bank learning setup; it
does not establish that the fly lacks recurrence. Per the registered branch, the next work
is a source-restarted temporal-credit/time-constant diagnostic, not a post-hoc relaxation of
the noise threshold or continuation of the response-erasure endpoint. The full report
SHA-256 is `ad0f4f0524be5f9a769b74d890d8f8b6925ba25921bb283ac52b8089ddc22c5c`;
see [`artifacts/variable-height-native-throttle-motion-only-v1/`](../artifacts/variable-height-native-throttle-motion-only-v1/).

### Preregistered native neural-integration-rate audit

Before changing biological time constants or fitting another controller, test whether the
current discrete implementation itself creates the observed phase lag. The camera and flight-
command rates are 50 Hz, but the current actor also executes only one synchronous whole-graph
state update per camera frame. Consequently every connectome edge has at least one 20 ms
discrete delay, even though the initialized membrane time constant is about 21.38 ms and a
real recurrent circuit evolves continuously between camera samples. A multi-hop visual path
can therefore behave as an artificially delayed proportional controller. Internal CNS
integration is distinct from adding actor inputs, external memory or a faster flight command.

This is a source-only, no-learning numerical audit. Hash-lock the stopped motion-only report
and resume at `ad0f4f0524be5f9a769b74d890d8f8b6925ba25921bb283ac52b8089ddc22c5c` and
`04c4b4e0b598f37c3808a7740bad0160c2826d20584149c841694696d1d0e28d`, and require its
training-bank hash `0887ead1e9eef7755c47b51967f55bbde0e7203b17e76ea9f50456e97ca530fc`.
Load the original `paired-dynamic-001` controller, not accepted update 17. Reuse the exact
opened 24-pair bank and its frozen `0.043661270290613174` teacher-contrast scale. Generate no
new seed, case, development bank, trajectory or optimizer state.

Render and persist one immutable input cache from that bank: its 25-frame neutral prefix and
each complete opposite-motion image sequence, roll/pitch input and teacher target. Record a
semantic cache hash and require bit-identical terminal images within every pair. Every audit
condition consumes this same cache in the same case order. K=1 must reproduce the stopped
source's NRMSE `1.031749290796319`, aligned gain `-0.030875112861394882`, prediction RMS
`0.002292240969836712` and exact endpoint-image result within `2e-5` absolute error before
any other condition is interpreted.

Evaluate internal substep counts `K = 1, 2, 4, 8, 16`. For condition K, instantiate the
unchanged source controller with neural integration step `1/(50K)` seconds. Hold each 50 Hz
RGB and roll/pitch observation constant for K successive native recurrent updates, then emit
only the final native motor output for that camera frame. Advance no aircraft or external
state during substeps. Physical observations and prospective stick commands remain 50 Hz;
only integration of the fly's own 165,122-neuron recurrent state changes. Parameters,
topology, transmitter signs, sensory mappings, time constants and front-leg pools remain
bit-identical across conditions.

For every K, report all 24 predicted and teacher throttle contrasts; correct-sign fraction,
aligned gain, NRMSE and prediction RMS overall and by 15/20/25-frame horizon; all four motor-
axis ranges/RMS; pair-common throttle; recurrent/output finiteness; and endpoint identity.
Also record wall time and native recurrent updates so performance cost is explicit. Do not
choose K by its damping score.

Choose a numerically adequate rate only by adjacent refinement. For each K in `1, 2, 4, 8`,
compare its 24-value contrast vector and all 48 four-axis terminal motor outputs with 2K.
The K-to-2K refinement passes when contrast RMS difference divided by the frozen teacher
scale is at most 0.01, raw terminal-motor RMS difference is at most 0.005, all states and
outputs are finite and both conditions retain exact paired endpoints. The selected rate is
the smallest K whose adjacent refinement passes. If none through K=8 passes, this audit
selects no rate; a higher-order or more finely referenced solver must be separately declared.

The selected K, if any, is a numerical-fidelity result, not a behaviorally best checkpoint.
If K=1 is already adequate, reject integration coarseness as the explanation and proceed to
a separately preregistered time-constant/temporal-credit experiment. If only K>1 is adequate,
future training may use that fixed internal rate, subject to its own preregistration, even if
the unchanged source remains wrong-signed. A corrected source sign would be encouraging but
does not itself authorize training, hover or promotion. Restore/hash-verify the source after
every condition and at termination; retain no altered controller. This audit cannot authorize
assisted hover, native attitude, gate flight or promotion.

Result: the source-only audit completed with exact source restoration and finite recurrence,
outputs and metrics at every rate, but selected no numerically adequate K. The immutable
721 MiB input cache had bit-identical opposite-motion endpoint images. K=1 reproduced the
earlier source NRMSE, aligned gain and prediction RMS within `4.23e-8`, `4.10e-8` and
`8.15e-9`, respectively, passing the registered `2e-5` control.

The normalized contrast-vector RMS changes for K=1→2, 2→4, 4→8 and 8→16 were `0.115923`,
`0.068393`, `0.035531` and `0.015952`; all exceeded the `0.01` convergence limit. Four-axis
terminal-motor RMS changes were `0.017457`, `0.006837`, `0.004326` and `0.002647`, so the
last two passed their independent `0.005` limit. K=8→16 was therefore close but remained
nonconverged on the registered contrast criterion, and no rate was selected.

Descriptively, increasing internal rate strengthened rather than corrected the native
wrong-signed response. Correct-sign fraction remained 0/24 throughout. Aligned gain moved
from `-0.030875` at K=1 to `-0.114593`, `-0.168809`, `-0.197870` and `-0.210812`; NRMSE
worsened from `1.031749` to `1.224188`. This suggests—but does not yet numerically establish—
that the unchanged circuit's more continuous-time response is more strongly anti-damping.
It rules out treating K=2, 4, 8 or 16 as a post-hoc behavior fix.

No training execution, hover, gate flight or promotion was authorized. Per the registered
branch, the next numerical step is a separately preregistered finer reference or higher-order
solver, not selecting K=16 because its behavior or cost looks convenient. The full report
SHA-256 is `ded09b91a114d08039d5334a937e36812b69ba55183634ae4bd21ed9864baff7`;
see [`artifacts/variable-height-neural-integration-rate-audit-v1/`](../artifacts/variable-height-neural-integration-rate-audit-v1/).

### Preregistered continuous-CNS solver extension

Complete the numerical-fidelity branch without treating the increasingly wrong-signed
behavior as a selection signal. Hash-lock the terminal neural-rate report at
`ded09b91a114d08039d5334a937e36812b69ba55183634ae4bd21ed9864baff7`, its original
motion report/resume inputs, and the immutable RGB cache at physical SHA-256
`68ea98cc60a8969eb38c1bc62db698bb232ddb6ddaebac426a850ee31596ecb8` and semantic SHA-256
`43a576a2abc01283a7b5a72560e3c9ff921bfaaf3e5a481ac6715019843f7d0e`. Reuse the cached
24 cases, order, roll/pitch observations, targets and frozen `0.043661270290613174` scale.
Do not render, generate, reorder or optimize anything, and load the original
`paired-dynamic-001` source afresh for every condition.

First extend the exponential-Euler reference used by the completed audit. Evaluate K=32 and
compare its 24 throttle contrasts and 48 four-axis terminal outputs to the hash-locked K=16
result. Apply the same two limits: contrast RMS difference divided by the frozen teacher
scale at most `0.01`, terminal-motor RMS difference at most `0.005`, with finite state and
outputs, exact cached endpoints and exact source restoration. If K=16→32 passes, define K=32
as the fine reference and K=16 as an adequate exponential-Euler candidate. If it fails,
evaluate K=64 and apply the same test to K=32→64. A pass defines K=64 as the fine reference
and K=32 as the adequate candidate. If that also fails, stop with no reference or solver;
do not relax thresholds or inspect behavior to continue.

The continuous state equation represented by the substep limit is
`ds/dt = (5*tanh((R(tanh(s)) + b + u)/5) - s)/tau`, where `R`, `b`, `u` and `tau` are the
unchanged signed connectome recurrence, bias, cached sensory drive and native time constants.
Using the established fine exponential-Euler condition as the only reference, evaluate
classical fourth-order Runge-Kutta with one, two and four equal integration steps per 20 ms
camera frame (`RK4-M=1,2,4`, requiring 4, 8 and 16 recurrent graph evaluations). Hold the
sensory drive fixed within a camera frame; RK stage values are numerical intermediates, not
actor memory or inputs. Emit the native motor-pool readout only from the final state.

Compare every RK4 candidate directly with the fine reference using the same normalized-
contrast and raw terminal-motor limits, finiteness, endpoint and source-identity requirements.
Report all metrics required by the original audit for every newly evaluated condition,
including per-horizon results, all-axis ranges/RMS, pair-common throttle, wall time and both
solver steps and recurrent graph evaluations. Select the passing candidate with the fewest
graph evaluations per camera frame from the adequate coarse exponential-Euler condition and
the three RK4 candidates. On an evaluation-count tie, retain exponential Euler because it is
the already implemented update. Behavior scores, damping sign and runtime cannot select the
solver.

A selected solver establishes only a numerically adequate and computationally declared way
to evolve the unchanged native recurrent state at a 50 Hz sensor/action boundary. It may be
used by a separately preregistered temporal-credit/time-constant training experiment. It
does not authorize that training to execute, and it cannot authorize hover, gate flight or
promotion. A stronger wrong-signed converged response is diagnostically useful but is not a
failure of the numerical audit and must not be tuned away here.

Result: the extension passed and selected classical RK4 with one 20 ms step per camera frame.
K=16 exponential Euler passed against the K=32 fine reference with normalized contrast RMS
difference `0.008004` and all-axis terminal-motor RMS difference `0.001231`. K=64 therefore
remained unopened. RK4-M=1, M=2 and M=4 all passed directly against K=32; their normalized
contrast differences were `0.008310`, `0.007284` and `0.007373`, and motor differences were
`0.001152`, `0.001840` and `0.001843`. The registered evaluation-count rule selected RK4-M=1
at four recurrent graph evaluations per 50 Hz camera/action frame.

All source identities, endpoint equality and finiteness controls passed. Runtime was not a
selection input, but RK4-M=1 took `1.256 s` versus `11.980 s` for K=32 on the audit. It is
also four times cheaper in graph evaluations than adequate K=16 exponential Euler, making
backpropagation through the native recurrence materially more practical.

The converged descriptive response remained anti-damping: correct-sign fraction was 0/24,
with aligned gain `-0.217006` for K=32 and `-0.223272` for selected RK4-M=1. The source's
one-update-at-50-Hz dynamics were numerically coarse, but that coarseness did not cause the
sign failure. The next experiment therefore needs to train temporal credit/routing under the
selected accurate solver rather than expecting faster integration alone to repair control.

This result authorizes naming RK4-M=1 in a separately preregistered training experiment; it
does not authorize training execution, hover, gate flight or promotion. The full report
SHA-256 is `daeb4c000b3421a2d1d4d22890ccb900f654f1587748d035d88c4f884cff85a7`;
see [`artifacts/variable-height-continuous-cns-solver-v1/`](../artifacts/variable-height-continuous-cns-solver-v1/).

### Preregistered RK4 throttle-motor readout step audit

Test the control-responsibility hypothesis implied by the representation audit: keep the
visual/recurrent estimator fixed and adapt only its existing anatomical interface to the
throttle foreleg motors. Hash-lock the passing solver report at
`daeb4c000b3421a2d1d4d22890ccb900f654f1587748d035d88c4f884cff85a7`, the original
source checkpoint, stopped motion report/resume, and the immutable RGB cache at its recorded
physical and semantic hashes. Reuse all 24 cases, source-relative outputs, order and frozen
teacher scale. Generate no new scene, seed, trajectory, label or optimizer sample.

Use selected `RK4-M=1`: one classical fourth-order step over each 20 ms camera interval,
with four recurrent graph evaluations and one final native motor-pool readout. The actor still
receives only cached RGB and roll/pitch and retains only its MaleCNS state. No decoded motion,
external history, velocity, accelerometer, controller state or assistance is introduced.

The deterministic mask contains every existing edge whose postsynaptic node belongs to either
throttle antagonist output pool, plus the bias and raw time constant of each such motor node.
It contains 493 edge magnitudes and seven motor nodes. The sorted int64 edge-index SHA-256 is
`4978758242ecfc9e53e859e077c098821e5747027a96f2945d3020e65ef81f3d`; the node-index
SHA-256 is `69c46d204b312950eb04e4538d5802031803e301d5b28dc45e8adb2d35487831`,
and the corresponding body-ID SHA-256 is
`bcdbf3f61b91b16dd7b6792b5622bb990d234fa97d326ba1180df0dd39a24752`.
All other edge magnitudes, biases and time constants remain bit-identical to source. Existing
topology and transmitter signs remain fixed, edge magnitudes stay in `[0,8]`, and no new
readout parameter is added.

Compute one complete 24-pair endpoint-motion gradient from the source. Each case uses a
source-computed, detached 25-frame neutral prefix followed by a fully differentiated cached
response under RK4. The only objective is the frozen-scale squared error of native throttle
contrast; pair-common throttle and RPY outputs are preservation measurements, not weighted
losses. Accumulate the exact mean gradient in fixed bank order. Apply a fresh one-step Adam
transaction with `betas=(0,0.999)`, edge and bias learning rate `1e-4`, raw-time-constant
learning rate `1e-6`, epsilon `1e-8`, and global gradient-norm cap 1.0. Zero every gradient
outside the declared mask before clipping and require every unmasked parameter to remain
bit-identical.

Replay the fixed-burn objective three times; use the first value as authoritative and the
others only to measure maximum pairwise numerical noise. The bound-projected full proposal
must be finite and have negative raw-gradient directional derivative. On scale `1/16`, require
the measured and predicted derivatives to be below `-1e-8`, symmetric relative error at most
20%, and absolute objective change greater than `max(1e-8,10*replay_noise)`. This probe is a
numerical control and cannot select the ordinary step.

Evaluate the joint proposal at fixed ordinary scales `1, 1/2, 1/4, 1/8, 1/16, 1/32` in
descending order using both the detached source prefixes and complete zero-state prefixes.
Select the first scale that improves NRMSE by at least `0.001` in both evaluations, keeps
source-relative pair-common throttle drift at most `0.005` RMS and `0.01` maximum absolute,
keeps source-relative roll, pitch and yaw terminal-output drift at most `0.005` RMS and `0.01`
maximum absolute per axis, and retains finite recurrence/metrics, exact endpoints and outputs
within `[-1,1]`. Candidate comparisons use the same source output tensors, not regenerated
source runs. Report edge-, bias- and time-constant-only scale-1 replays as diagnostics, but
they cannot be selected or combined post hoc.

Restore and hash-verify the source after every replay and at termination; retain no candidate
or optimizer. A pass establishes only one safe local readout-directed step on the opened bank
and authorizes preregistration of a bounded last-hop fitting run with disjoint development and
qualification cohorts. Failure closes this exact last-hop/optimizer route and sends work to
an upstream native-routing or direct constrained-readout design. Neither outcome authorizes
training execution, closed-loop hover, gate flight or promotion.

Result: the audit passed every numerical, identity, mask, bound, endpoint, finiteness and
restoration control but selected no step. The full masked Adam proposal was a valid descent
direction (`-0.001430`), and its scale-1/16 measured and predicted changes agreed to 4.41%
above the replay-noise threshold. All gradients and parameter changes outside the 493-edge,
seven-neuron mask were exactly zero.

Scale 1 improved detached-prefix and zero-state full-prefix NRMSE by `0.000602`, short of the
registered `0.001` floor. That was its only failure. Pair-common throttle drift was just
`0.000428` RMS and `0.000544` maximum, and RPY drift remained at float32 noise scale. Smaller
steps gave nearly proportional, still sub-threshold improvements. The family-only diagnostics
assigned `0.000591` improvement to incoming edge magnitudes, `0.0000106` to motor bias, and
no measurable improvement to the registered motor-time-constant step.

This does not establish a harmful last-hop coupling: the mask had substantial safety headroom
but the exact Adam scaling was too weak for the preregistered useful-progress claim. The
failed scale is not retroactively accepted. Per the frozen branch, the next justified test is
a separately declared direct constraint-aware readout displacement or upstream routing, not
this exact optimizer repeated. No fitting, hover, gate flight or promotion was authorized.
The full report SHA-256 is
`508c54ebb00aa8419e66e540bbc7ca45bd0d254858e76eab590cc4f39dc878f7`;
see [`artifacts/variable-height-rk4-throttle-readout-step-v1/`](../artifacts/variable-height-rk4-throttle-readout-step-v1/).

### Preregistered direct readout trust-region extension

Preserve the failed scale-1 verdict and reconstruct its source-computed RK4 prefixes, complete
24-pair gradient, masked β1=0 Adam transaction and bound-projected pending parameters from
scratch. Hash-lock the failed report at
`508c54ebb00aa8419e66e540bbc7ca45bd0d254858e76eab590cc4f39dc878f7` and require exact
pending-parameter semantic SHA-256
`73d633cecf3fd0b42616142eb16eac931d9e41dac6158d640ab07f124353ec0d` and post-step
optimizer SHA-256 `0692f59a15c27ef9c3cff3a09dc6611246e0d1dcea8b051e9c32a97cd371976e`.
Reproduce its three fixed-prefix objectives, gradient objective and full directional derivative
within absolute `2e-5`, and its empty-to-one Adam counters exactly. Any reproduction failure
is terminal.

Treat the reconstructed displacement only as a direction in the declared 493-edge/seven-node
readout subspace. This is a new functional trust-region test, not retroactive acceptance of
the old scale. Materialize extrapolation scales `16, 8, 4, 2` in fixed descending order from
the original source, apply native edge bounds once, and evaluate actual detached-prefix and
zero-state full-prefix RK4 replay. Select the first scale that passes every unchanged readout
audit gate: at least `0.001` NRMSE improvement in both replays; pair-common throttle drift at
most `0.005` RMS and `0.01` maximum; per-axis RPY drift at most `0.005` RMS and `0.01`
maximum; exact endpoint equality; finite state, output and metrics; motor outputs in `[-1,1]`;
and exact unmasked parameters and replay restoration. Do not evaluate any scale at or below 1,
which has already failed, and do not interpolate an unregistered scale.

The trust-region result is determined by actual nonlinear functional replay, not Taylor
extrapolation. Report bound activations, masked-family displacement RMS and every preservation
margin at every tried scale. Restore and hash-verify the source and optimizer at termination;
retain no candidate. A pass authorizes only preregistration of a source-restarted bounded
last-hop fitting run whose per-update line search includes the complete combined ladder
`16,8,4,2,1,1/2,1/4,1/8,1/16,1/32` and always enforces the same source-relative output
constraints. Failure closes the last-hop readout route and moves upstream into native routing.
Neither result authorizes training execution, hover, gate flight or promotion.

Result: the extension stopped at its reproduction gate before opening any trust-region
scale. Its three source objectives reproduced within `3.58e-7`, gradient objective within
`4.77e-7`, and directional derivative within `6.37e-11`, all far inside the registered
`2e-5` scalar tolerance. The empty-to-one Adam counters and mask controls also passed.

The pending controller and optimizer did not reproduce byte-identically across CUDA
processes: their semantic hashes were `4c84cc…` and `5ae04a…` rather than the registered
`73d633…` and `0692f5…`. Tiny differences in the reported family-gradient RMS values are
consistent with GPU reduction nondeterminism, but the exact-hash gate is preserved. Scale
16, 8, 4 and 2 were never evaluated, so this result says nothing about their feasibility.

Source and optimizer restoration passed exactly. A corrected repeat must persist one
reconstructed direction and establish numerical plus functional equivalence to the prior
scale-1 replay; it must not simply delete the failed hash gate after the fact. No fitting,
hover, gate flight or promotion was authorized. The full report SHA-256 is
`03c4ee348d51c351611b425de184a3d64cf3211f1a75ddf1c6f9e50006dcfc5e`;
see [`artifacts/variable-height-rk4-readout-trust-region-v1/`](../artifacts/variable-height-rk4-readout-trust-region-v1/).

### Preregistered canonical readout trust-region repeat

Run one corrected repeat that preserves both preceding outcomes. Hash-lock the failed
trust-region report at
`03c4ee348d51c351611b425de184a3d64cf3211f1a75ddf1c6f9e50006dcfc5e` and the original
readout-step report at
`508c54ebb00aa8419e66e540bbc7ca45bd0d254858e76eab590cc4f39dc878f7`.
Reconstruct the same source, cache, mask, detached RK4 prefixes, complete-bank gradient and
one β1=0 Adam transaction once. Retain the existing `2e-5` scalar reproduction tolerance,
mask checks and empty-to-one counters, but do not require a floating-point CUDA reduction to
reproduce a prior process's parameter or moment bytes.

Before any new-scale evaluation, persist the once-reconstructed source parameters, pending
parameters and optimizer states below the ignored run directory. Hash them physically and
semantically, reload them on CPU, require bit-exact equality to the just-produced in-memory
tensors, and use only clones of that reloaded archive thereafter. This makes one canonical
direction inside the run and prevents different replays from silently regenerating it.

Evaluate old scale 1 once as a reproduction control. Against the hash-locked prior scale-1
record, require fixed-prefix and full-prefix NRMSE, gain, all 24 throttle contrasts, all 48×4
terminal motor outputs, pair-common drift and each RPY drift metric to agree within `2e-5`
absolute. Require the same exact mask, bounds, endpoint, finiteness and restoration controls.
This control cannot be selected and preserves its original failure below `0.001` improvement.
If either the scalar, archive or functional control fails, stop before opening a new scale.

Only after all controls pass, evaluate the unchanged trust-region ladder `16,8,4,2` in fixed
descending order from the canonical source/pending tensors. Apply the exact candidate and
selection gates registered for the failed extension and select the first actual passing
scale. Report bound activations and masked-family RMS. Restore and hash-verify source plus
the empty optimizer at termination and retain no candidate. A pass has the same narrow
meaning: it authorizes preregistration, not execution, of bounded last-hop fitting. Failure
closes this direct last-hop route and moves upstream. No result authorizes hover, gate flight
or promotion.
