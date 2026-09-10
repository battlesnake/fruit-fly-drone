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
this one update; it does not establish absence of longer-term task conflict. It authorizes
only a separately preregistered joint-fitting run using that step-control rule and the
mandatory update-50 endpoint-D improvement gate. If no scale passes, family probes guide
a later individual-output RPY-Jacobian projection protocol. If a scale passes without
measurable D improvement, report collective calibration only. No audit endpoint is
retained or promoted and no closed-loop test follows directly.
