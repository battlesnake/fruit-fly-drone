# Active target: more than 50% varied five-gate completion

Started 2026-09-12. The target is **the fly controller**, not a conventional teacher,
completing all five gates cleanly in more than half of held-out flights. It remains
unmet. A passing short imitation loss, one successful flight, or a perfect teacher
preflight does not meet it.

## Course and evaluation protocol

Use uninterrupted 30-second flights from the existing slightly perturbed airborne
initial states. Keep the present 0.62 m inner/0.76 m outer annuli and 0.09 m drone
clearance margin. No enlarged apertures, privileged actor observations, teacher
interventions or neural resets during evaluation.

The first gate retains its learned random distribution: distance 2.7–3.3 m, lateral
offset magnitude 0.61–0.78 m on either side, height 1.10 m, and 2–7° obliquity relative
to the initial bearing. Later gates independently vary:

- forward spacing: 0.9–1.5 m;
- lateral course increments: up to ±0.20 m, with deviation bounded ±0.50 m from the
  initial course line;
- height increments: up to ±0.08 m, within 0.90–1.30 m;
- gate yaw: independent ±15° perturbations around the first gate's orientation.

Paired episodes mirror course geometry while sharing height variation. An independent
yaw random stream keeps positions unchanged in matched angle ablations. This is a
broader forward-course family, not yet 180s, inverted manoeuvres or a typical full lap.
The end goal in [PLAN.md](PLAN.md) remains broader than this intermediate milestone.

Keep role colours and checkerboard backs. A clean result uses the all-gate,
both-direction rules: no illegal aperture traversals, ring contacts, ground contacts
or invalid states during the full evaluation window. Do not count eventual passes
after a failure as successful flights.

The actor still sees only 320×200 RGB at 125° HFOV and temporary roll/pitch inputs.
The full native recurrent state, forelegs, sticks and aircraft run continuously.

Small development banks may select exploratory candidates. Before claiming the goal,
evaluate the selected checkpoint on at least 128 fresh, unhinted episodes drawn from
the full distribution, with matched source results and both course sides represented.
Use separate seeds not used for training or checkpoint selection. Prefer a clear margin
above 50% and a second fresh bank over treating a marginal small-sample result as
reliable. A frozen-camera control should check whether flight actually depends on live
vision. Do not shrink the distribution after seeing failures.

## Evidence so far

The old 17/128 figure predates independent gate-yaw variation and the new clean-course
checks. New baselines from that same checkpoint:

| Bank | Distribution | Clean flights |
| --- | --- | ---: |
| seed 980983 | old position variation, corrected rules, 22 s | 2/32 |
| seed 990983 | full distribution above, 30 s | 0/32 |
| seed 1020983 | full distribution, ES development bank | 1/64 |
| seed 1050983 | full distribution, imitation development bank | 1/16 |

These are different banks, not evidence that the new rules alone caused a particular
drop. Crossing errors in the fresh baseline grow across the course, predominantly
laterally, with vertical drift too.

### Native-parameter search

`scripts/search_pragmatic_gate_course_es.py` searches 24 existing motor-pool bias and
incoming excitatory/inhibitory gain coordinates. No topology or actor interfaces change.
The fitness rewards clean-prefix gates and clean completion, penalizes failure, and
caps centering shaping at 0.2 gate-equivalents. Completion is the primary selection
criterion; first-gate retention is a tiebreaker, not a veto on more completed courses.

The first generation's 16 perturbations had no clean five-gate completion on their
eight-case training bank. The next generation produced one 1/8 candidate before the run
was intentionally interrupted to try the newly validated teacher. That partial result
was not development-validated and is not a promoted controller. The complete first
generation and its vectors are saved in
`runs/gate/pragmatic-five-gate-varied-motor-es-001/report.json`. This short probe does
not establish that ES cannot work.

The search now also supports `--candidate-batch`: independent parameter candidates
share a GPU batch through the existing anatomical motor-interface evaluator. Each
candidate keeps its own complete neural/leg/aircraft history. The 24 parameter values
are fixed for its entire flight, then compiled into ordinary native checkpoint weights
for standalone development evaluation. Tests compare batched recurrence with compiled
controllers and check that per-candidate scores preserve the full clean-course rules.

The six-generation batched search from native-best100 completed 48 candidate trials
and retained a standalone development result of 6/32, versus its 4/32 source, on seed
1110983. It did not exceed the subsequent imitation result. A generation-one trial
that scored 4/8 on its search bank scored only 5/32 when separately compiled and
checked; small search-bank successes are not sufficient for selection or the goal.

### Continuous-path teacher

`src/flydrone/course_teacher.py` implements a training-only cubic-Hermite course path
and full-state tracking controller. Shared tangents make the exit direction at one
gate depend on the following gate. Velocity damping and drag compensation drive the
same actual foreleg/stick/quad plant used by the fly. It requires monotonically
increasing-X courses; no imported controller code or model was executed.

`scripts/audit_pragmatic_course_teacher.py` checked both the path geometry and physical
tracking under the current all-gate rules:

| Bank | Clean flights | Mean completion | Mean / maximum crossing radius |
| --- | ---: | ---: | ---: |
| seed 1030983 | 64/64 | 15.11 s | 0.025 / 0.158 m |
| fresh seed 1037983 | 128/128 | 15.05 s | 0.028 / 0.207 m |
| same fresh bank, fixed world-X heading | 128/128 | 15.07 s | 0.014 / 0.064 m |

Neither bank had collisions, illegal traversals, ground contact or control saturation.
Maximum tilt in the fresh bank was 15.73°. A matched-state test with the same current
gate and opposite next-gate offsets confirms different pre-crossing actions and exit
tangents. These are **privileged teacher results, not fly successes**.

The fixed-heading variant retains closed-loop yaw stabilization and all four motor
channels. It changes teacher targets, not course geometry or the deployed actor: a
quadcopter can translate sideways without facing its velocity or each gate normal.
This is appropriate to the present forward courses, not a general teacher for reversals.

Raw records are under `runs/gate/continuous-course-teacher-001/`. Keep the visual-ribbon
and hint-fading idea in [RACING_ANTICIPATION.md](RACING_ANTICIPATION.md); it has not been
discarded or implemented in this trial.

### First four-axis imitation trial

`scripts/train_pragmatic_course_teacher.py` trains existing visual-to-motor paths with
full native recurrence and four-axis teacher targets. It imposes no static roll-sign
rule. The first trial trained 33,570 existing edges and 9,801 biases, with time constants
frozen: 200 teacher-driven updates followed by 100 updates gradually transferring
control to the student, 20-frame unrolls and four simultaneous episodes.

Action error on teacher trajectories improved, but autonomous development completion
was 0/16 at each 50-update check. The flight-selected checkpoint therefore stayed at
the original 1/16 baseline. First-gate crossing became faster but less accurate, and
some checkpoints became unstable. Lower imitation error was not flight progress.

The first run covered only eight course banks and reached fully native control only
on its final update. Two completed follow-ups used 200 genuinely student-driven,
teacher-labelled updates, starting from either the original or imitation checkpoint.
They also opened existing roll/pitch-to-motor paths (178,348 edges and 19,284 biases),
without adding observations. Both scored 0/16 at every development check and eventually
lost all first-gate passes. Neither was promoted. Reports are in
`runs/gate/pragmatic-course-native-dagger-{source,imitation}-001/`.

Large yaw error and unstable flight suggest a teacher/camera-heading mismatch and
overly disruptive updates, but these are hypotheses, not isolated causes. The next
bounded experiment starts from the preserved original with fixed-world-X teacher
heading, 50 updates at edge learning rate 1e-5, frozen biases/time constants, and only
the 33,570 visual-path edges. Native development flights every 25 updates decide whether
to extend it; improved teacher-action loss alone does not.

That pilot completed with **3/16 clean flights** at update 50, versus source 1/16 on
the same development bank. First-gate passes rose from 13/16 to 14/16 and mean clean
prefix from 1.94 to 2.63 gates. Update 25 scored 2/16. This is a small selection-bank
improvement, not a held-out reliability claim. Repeated FP32 evaluations can change a
borderline gate pass; one repeat of the same pilot checkpoint scored 4/16. Do not spend
the project on bitwise replay, but require larger fresh banks for the goal.

Reports/checkpoints are under `runs/gate/pragmatic-fixed-heading-imitation-001/`.
Two completed extensions compared 100 teacher + 100 handoff + 100 native updates at
1e-5 against 200 entirely native teacher-labelled updates at 3e-6, both with frozen
biases/taus and the same visual-path mask:

| Extension | Best development checkpoint | Final checkpoint |
| --- | ---: | ---: |
| teacher/handoff/native | update 100: 4/16 | update 300: 1/16 |
| fully native, gentler updates | update 100: 5/16 | update 200: 1/16 |

The selected native checkpoint retained 14/16 first-gate passes and reached an average
clean prefix of 3.25 gates. Its report is
`runs/gate/pragmatic-fixed-heading-native-001/report.json`. Later updates regressed,
so use `best-controller.pt`, not `latest-controller.pt`.

On a new 32-flight bank (seed 1110983), that checkpoint achieved **4/32**, versus the
original's **2/32** on identical courses. This is preliminary generalization evidence,
not proof that 31.25% from the small selection bank is its true success rate. That bank
is now a development bank for the six-generation batched motor search in
`runs/gate/pragmatic-fixed-heading-batched-es-001/`; do not reuse it as the final goal
holdout. The search retains the full geometry, all four native outputs and course rules.

### Direct-loss ablation

For a matched pair, direct MSE plus unit-weight pair-difference MSE equals common-mode
error squared plus five times differential-mode error squared. In native flight the
two members can also reach different phases, so contrast is not necessarily a clean
visual-steering lesson. Astra identified this as a possible contributor to shared
late-course lateral drift. `--contrast-weight 0` now permits a direct-only comparison;
unit tests verify the common/differential weighting. The default remains 1 for earlier
run recipes. A 100-update native-only trial starts from the preserved native-best100
at 3e-6 with all other plasticity restrictions unchanged, using seed 1130983 and the
32-flight development bank. This tests a hypothesis; no improvement is assumed.

The direct-only run reached 9/32 at update 50 (from its 3/32 repeated baseline), with
28/32 first-gate passes; update 75 scored 7/32 and update 100 scored 0/32. The best50
checkpoint is retained. Its successes are asymmetric: 8/16 negative-side and 1/16
positive-side courses, with no complete mirrored pair. A matched unit-contrast
control uses the same warm start, training seed, native-only stage, learning rate and
32-case development set; this is needed to distinguish loss weighting from the effect
of another training seed. Treat the 9/32 result as development selection, not fresh
validation or a demonstrated causal effect of the loss change. In fact, the matched
unit-contrast control reached **11/32 at update 50**, versus direct-only 9/32. Its
first-gate pass rate was 27/32 and mean prefix 2.81. The available comparison does not
support attributing the gain to removal of contrast; a further short training pass on
new courses is a competing explanation. The control also fell to 0/32 at update 100,
despite 30/32 first-gate passes. Its retained best50 scored 9/16 negative-side and 2/16
positive-side completions, with one complete mirrored pair.

A 64-flight check of best50 uses seed 1020983, outside this controller's training and
selection banks, where the original checkpoint previously scored 1/64. This bank was
an earlier ES development bank; do not describe it as globally never-used or recycle
it as the final goal holdout. Direct-only best50 achieved **10/64**, versus original
1/64. Its cumulative passes were 53, 40, 33, 21 and 10, with no ground contacts or
invalid states. Success was still asymmetric (9/32 negative-side, 1/32 positive-side),
and ring contact remained frequent. This supports some improvement beyond the small
selection set, but is nowhere near the >50% goal. The matched unit-contrast best50
also achieved **10/64**, all on the negative side (10/32 versus 0/32 positive). Its
cumulative passes were 53, 39, 30, 20 and 10. Thus neither loss variant established
superiority on this bank, and bilateral generalization is a clear remaining weakness.

`scripts/export_pragmatic_course_candidate.py` can compile any saved ES trial into a
normal controller for standalone checks, without replacing a selected checkpoint.
This also permits validation of a strong candidate between scheduled development
generations instead of losing it from consideration.

### Alternative retained: balanced replay with current-weight neural history

Astra identified a plausible weakness in chronological training: each optimizer update
sees the next 0.4 seconds of just four flights, and carries neural state produced by
slightly older weights. If the present extensions stall, collect a mixture of native
and teacher flights with weights frozen, then combine first-gate and later-gate windows
from different courses in each update. Replay each window's actual sensory prefix from
zero under the current weights before its differentiable unroll. Refresh native data
periodically. Replay buffers, course phase labels and teacher state remain training-only;
the deployed actor retains only its native recurrence. Measure stale-state prediction
error and gradient/update alignment before attributing failures to insufficient recurrence
or changing optimizer momentum. This alternative is now implemented in
`scripts/train_pragmatic_course_replay.py`; its first trial completed in
`runs/gate/pragmatic-phase-balanced-replay-001/`.

The next bounded recipe starts from the preserved unit-contrast best50, keeps contrast
1, edge LR 3e-6, the same mask/scales and frozen biases/taus, and runs 60 updates:

- Collect eight native mirrored pairs and four teacher-driven pairs on new training
  courses with weights fixed. Store small CPU physical-state histories, gate geometry,
  actual role indices, valid-before-failure flags and detached teacher targets. Re-render
  the recorded RGB; do not cache gigabytes of frames or reusable neural states.
- Each update combines a gate-one pair window and a later-gate pair window, cycling
  evenly through gates 2–5. Prefer native windows, falling back to teacher windows when
  both native branches lack that phase. Include approaches, gate transitions and
  pre-failure windows. Log phase coverage and fallback counts.
- Branches may start their windows at different times to align phase, but replay each
  complete original prefix from zero using the ten-frame warmup and current weights,
  without gradients. Differentiate 20 frames per window. Average both window losses
  and add the existing anchor once before one optimizer step.
- Evaluate at 0/20/40/60; refresh native collection after 20 and 40 using the retained
  controller. Preserve the best clean-course checkpoint, then check fresh courses and
  both sides. Do not extend an unhelpful chronological run merely because loss falls.

The initial collection contains eight native pairs on seed 1150983 and four teacher
pairs on seed 1250983. All five phases have recorded observations. Gate-five paired
windows initially require the teacher bank; actual source/phase/window-kind choices
are logged at every update. Early and late pair selections in this first implementation
are independent draws and can occasionally select the same course. Tests compare
unequal-start replay with separate original-prefix replays, check recomputation after
weight changes, and exclude invalid-tail windows. The actor input path is unchanged.

For this monotonic-X teacher, a training lesson ends after missing its expected aperture,
because the path tracker cannot recover a gate behind it. Evaluation still permits
exterior-plane manoeuvres. Loss and gradient checks prevent nonfinite optimizer updates.

The 60-update trial scored 4/32 at update 20 and 6/32 at update 40, restoring its retained
starting controller and clearing optimizer state before each new native collection.
Update 60 recovered to **11/32**, versus its repeated source baseline of 10/32. It
retained 28/32 first-gate passes, with mean clean prefix 3.125 and side completions
8/16 negative versus 3/16 positive. This one-flight development difference does not
establish an improvement. On the same seed-1020983 64-flight bank where its source
scored 10/64, replay best60 achieved **12/64**: 10/32 negative-side and 2/32 positive-side
completions, cumulative passes [53, 42, 32, 23, 12], and no ground/invalid events.
This is a small measured improvement with substantial remaining asymmetry, not a
reliable >50% controller. Neither bank is a final goal holdout.

### Rate-damped heading teacher ablation

The absolute world-X heading target may require unnecessary visual heading estimation
for these mild forward courses. The optional `rate-damped` teacher uses current yaw
to convert its desired world force into roll/pitch targets, and applies only body-yaw-rate
damping on its yaw axis. It does not zero or bypass the fly's yaw output. Ground-truth
yaw/rates remain privileged teacher data; no deployed inputs or memory are added.
This is not a general yaw-steering teacher for reversals or a full racing lap.

On the same 128 courses (seed 1037983), both world-X and rate-damped preflights completed
128/128 in about 15.065 s, with maximum crossing radius about 0.0645 m. Rate-damped
maximum heading excursion from launch was 0.632 degrees. A separate robustness probe
adding mirrored initial yaw offsets of 20 degrees and yaw rates of 30 degrees/s also
completed 128/128, with maximum heading excursion 1.058 degrees. The actual goal
distribution was not changed by this diagnostic.

Current and next gate centres were inside the camera frustum in all eligible frames
for both modes and the disturbance probe. This is explicitly a **centre-only geometric
proxy**, not rendered annulus visibility or an occlusion measurement. It includes only
frames before failure/fifth-gate passage and centres at least 0.5 m away. Reports are
`rate-damped-preflight.json`, `world-x-frustum-preflight.json` and
`rate-damped-offset-preflight.json` under `runs/gate/continuous-course-teacher-001/`.
These remain teacher results, not evidence of fly success.

`scripts/audit_course_teacher_targets.py` compares labels on identical unmodified fly
flights, split by gate and side. With native-best100 on 16 flights (seed 1160983),
normalized source-target RMS errors were:

| Teacher | Roll | Pitch | Yaw | Throttle |
| --- | ---: | ---: | ---: | ---: |
| world-X | 1.965 | 0.955 | 1.625 | 0.993 |
| rate-damped | 1.989 | 0.962 | 0.107 | 0.993 |

All four normalization scales are unchanged. Lower label error is explanatory only,
not a checkpoint selection or success measure. The resulting matched native-only
training comparison starts both branches from native-best100, using training seed
1130983, fresh Adam, LR 3e-6, contrast weight 1, 33,570 native visual edges and frozen
biases/time constants. Each runs 50 updates with checks at 0/25/50 on development seed
1110983 (32 flights). Only teacher heading mode differs. Runs are
`pragmatic-rate-damped-native-001` and `pragmatic-world-x-matched-native-001`.

Both matched branches scored 3/32 at update 25. At update 50, rate-damped scored
**6/32**, versus the world-X control's **8/32**, retaining 28/32 and 27/32 first-gate
passes respectively. Their baseline repeats were 4/32 and 3/32. This bounded test did
not establish a benefit from the easier yaw labels; do not extend it on loss alone.
The repeated world-X training outcome also differs from its earlier 11/32, so small
checkpoint-selection gains need independent behavioral checks, not reproducibility
claims or automatic promotion.

### Diagnostic retained: post-first-gate axis takeover

`scripts/audit_pragmatic_axis_takeover.py` tests whether the late-course roll errors
are primarily a roll-policy weakness or induced by the other control axes. Every
condition uses the same frozen replay best60 and starts fully natively. After the
first gate only, substitute rate-damped teacher motor targets on roll alone, the
other three axes alone, or all four axes, with a fully native baseline. Native neural
state still updates continuously from ordinary sensors. The teacher changes only the
physical motor drive; **assisted completions cannot count toward the goal**.

All four conditions retain the full 30-second window and unchanged course rules.
Report clean completion conditional on a clean first-gate passage, split by side,
and gate-by-gate lateral errors. Full takeover is a positive recovery control:
if it fails, do not attribute partial-takeover failures to a particular neural axis
before checking the teacher's recovery capability on native first-gate exit states.
The initial diagnostic uses 32 development courses, seed 1110983, in
`runs/gate/pragmatic-phase-balanced-replay-001/axis-takeover-32.json`.

The completed diagnostic strongly localizes the current weakness to roll control:

| Post-first-gate control | Clean completions / all 32 | Completions / 28 clean first-gate exits |
| --- | ---: | ---: |
| all native | 12/32 | 12/28 |
| teacher roll; native pitch/yaw/throttle | 27/32 | 27/28 |
| native roll; teacher pitch/yaw/throttle | 8/32 | 8/28 |
| teacher all four axes | 28/32 | 28/28 |

All conditions retained 28/32 clean first-gate passages. Positive-side conditional
success was 3/13 native, 13/13 with teacher roll, 0/13 with teacher other axes, and
13/13 with full teacher takeover. Thus the recovery positive control succeeds,
and replacing the other three axes does not rescue native roll. This motivates a
bounded roll-focused learning trial while preserving the source's other motor outputs;
it does not justify deploying privileged roll assistance or claiming the goal.

### Roll-focused replay with frozen-source preservation

The next 60-update trial is `pragmatic-late-roll-preservation-replay-001`, starting
from replay best60. `--supervision late-roll-preserve` selects only existing five-hop
visual-to-roll paths and explicitly removes any selected edges entering the other
three motor pools. All other edges, biases and time constants stay frozen. Upstream
recurrence can still influence other axes indirectly, so motor-input freezing is not
treated as sufficient preservation on its own.

A frozen copy of the starting fly runs during collection on the exact same sensory
history as the learner. Its four outputs are stored with the physical histories.
Before gate one, all four learning targets preserve those source outputs. Afterward,
only the roll target becomes the rate-damped teacher's roll; pitch/yaw/throttle still
target that same frozen source. No second reference network runs during gradient
replay, and no reference network or teacher is deployed.

Collection mixes eight fully native pairs with four roll-assisted pairs. In the
assisted bank the frozen source flies through gate one, then keeps its own three
non-roll outputs while teacher roll alone acts, matching the successful diagnostic.
Each update combines an early preservation window with a gate-2–5 learning window.
LR 3e-6, contrast 1, twenty-frame unrolls and 0/20/40/60 fully native checks remain
unchanged. Training seed is 1180983; assisted bank seed is 1280983; development seed
is 1110983. Native banks refresh under retained weights at updates 20 and 40, while
the preservation source stays fixed at the initial checkpoint.

The actual mask contains 19,286 edges, 7,056 path nodes and all six roll motor neurons;
none of these selected edges entered the other motor pools. The initial assisted bank
contains observations at every gate and one failed lesson out of eight; the initial
native bank contains thirteen failed lessons out of sixteen. The repeated autonomous
development baseline is 11/32. These collection counts are lesson coverage, not a new
autonomous evaluation result.

This bounded trial did not improve completion: update 20 tied 11/32 (29/32 first gate),
update 40 scored 10/32 (26/32 first gate), and update 60 scored 10/32 (28/32 first gate).
The starting checkpoint remained selected. Non-roll same-history errors stayed small,
but that preservation did not make the late roll correction sufficiently learnable in
these 60 short-window updates. Do not promote its latest checkpoint.

### Fast native roll-motor fit

The six roll motor neurons are **neural graph sinks**: they have no outgoing neural
edges, including no self/cross-motor edges, and receive no direct sensor injection.
Their 726 incoming synapses originate at 363 presynaptic neurons. Of those incoming
synapses, 616 belong to the current five-hop visual-to-roll mask. This permits a much
smaller training computation without adding a deployed readout or ignoring neural
feedback from the motor neurons.

`src/flydrone/motor_slice.py` computes the same six leaky motor states, fixed biases,
fixed time constants and sigmoid antagonist-pool means. A causal exponential filter
implements their membrane recurrence from zero, including the original warmup; it is
not a collection of independently reset one-frame predictions. The method rejects
graphs with outgoing motor edges or direct sensory injection into those motor cells.
Only fitted magnitudes of existing synapses are copied into an ordinary full-controller
checkpoint. Signs, topology, all other parameters and the deployed actor are unchanged.

`scripts/train_pragmatic_roll_motor_slice.py` collects eight native and eight roll-assisted
mirrored pairs from replay best60, storing source presynaptic activity on each actual
sensory history. Targets preserve the source through gate one and teach only late roll.
The last two whole pairs in each bank are held out from fitting; no frame-level split
leaks a flight between training and validation. The objective balances early/late
phases and both sides. The initial bounded run uses 200 Adam updates at 1e-3 on the
616 visual-path incoming synapses, with biases/time constants frozen.

The run is `pragmatic-roll-sink-fit-001`, using native collection seed 1190983 and
assisted seed 1200983. A fit is not flight success. Reconstruct the source outputs,
check the compiled candidate through full-network sensory replay, then evaluate fully
autonomous flights before considering promotion. The frozen-feature cache, teacher
labels and small training computation are never deployed as an external controller.

All exploratory checkpoints remain ignored under `runs/`; do not publish them as a new
best fly until full-flight results justify it. If a checkpoint is promoted, retain the
MaleCNS attribution and use Git LFS.

The motor-only fits did not improve autonomous completion. Run `pragmatic-roll-sink-fit-002`
added a reusable feature cache, full-network replay checks, an undeployed linear diagnostic
probe and separate selected/latest exports. Its held-out motor loss selected update 100,
but native development flights scored only 3/32; update 200 scored 6/32. A matched fit of
all 726 incoming roll synapses, instead of the 616 five-hop subset, also scored 6/32.
The corrections mainly traded negative-side success for positive-side success.

Static native-weight blends of 25%, 50% and 75% of the update-200 correction scored
9/32, 8/32 and 8/32 respectively on development seed 1110983. They add no runtime
selector, but still failed to beat the retained source's 11–12/32 on that same bank.
The retained source's separate 64-case check remains 12/64; none of these motor-only
experiments changes that best broader result. The unconstrained linear diagnostic
also failed to improve held-out motor error; this does not establish that the native
presynaptic activity contains no useful motion information.

### Targeted motion-module plasticity trial

A structural review with Astra found that the existing five-hop roll mask includes
only 360 of 6,861 T4 cells and none of 6,719 T5 cells. Just 523 of 53,231 existing
Mi1/Mi4/Mi9/Tm3→T4 and Tm1/Tm2/Tm4/Tm9→T5 input edges are trainable in that mask.
Those neurons and frozen edges still run in the full brain; the limitation is what
training may change, not which cells are simulated.

`scripts/train_pragmatic_course_motion_gains.py` tests one bounded alternative:
32 shared magnitude gains, one per presynaptic type and T4a–d/T5a–d target subtype.
Every matching existing edge in both eyes participates, without a hop cutoff. Gains
start at one, multiply the retained checkpoint's actual weights, and stay in [0.5,2].
Signs, topology, biases, time constants, downstream connections and motor readout are
frozen. The training parametrization is materialized into ordinary native edge weights
for checkpoint export; it is not a new deployed decoder or sensory channel.

Run `pragmatic-motion-shared-gains-002` starts from replay best60, with up to 30 updates
at gain LR 1e-3 and checks at 10/20/30. It reuses twenty-frame differentiated windows,
eight native pairs, four roll-assisted pairs, source early-roll/non-roll preservation,
and late rate-damped-teacher roll targets. Training seeds are 1180983/1280983; separate
validation collection seeds are 1210983/1310983, two mirrored pairs each. Fixed paired
validation windows cover the five gate phases. Continue only if late-roll RMSE falls
on both sides while early-roll RMSE stays at most 0.008; this small validation screen
is not a course success claim. Selection still uses unassisted complete development
flights on seed 1110983, with a first-gate preservation floor. Any promising candidate
then needs a fresh course check and frozen-camera control before promotion.

Attempt `001` was stopped before gradient updates to fix two review findings: split
diagnostic residuals by each frame's actual phase rather than its window's starting
phase, and enforce the first-gate floor as a hard selection condition rather than
a score tie-break. Its identity checkpoint and collection are not a trained result.

The corrected motion trial stopped at update 10: source and candidate both completed
12/32 development courses and passed gate one on 28/32. The split changed from
9 negative / 3 positive completions to 8 / 4; the development selector retained update
10 on that balance tie-break, not on improved total completion. Held-out late-roll
RMSE worsened slightly on both sides (0.033210→0.033244 and 0.019604→0.019639), so the
planned continuation screen stopped the trial. Do not promote it over the retained
source on this evidence. No gain reached its bounds; the learned range was about
0.9936–1.0070. This closes this small shared-gain trial, not all motion-circuit learning.

### Information near the motor neurons and an anticipation lesson

`scripts/audit_pragmatic_motor_parent_information.py` probes the already recorded
363 motor-parent activities in the sink-fit-002 cache. Linear probes fit six mirrored
pairs per bank and check two whole held-out pairs, with the same phase/side weighting
as the motor fit. Privileged quantities are diagnostic targets only; nothing is
injected into the actor or deployed as a decoder. The report is
`runs/gate/pragmatic-roll-sink-fit-002/parent-information-probe.json`.

Across ridge strengths 0.001–0.1, body-lateral-velocity RMSE is 0.033–0.043 m/s from
native parent activity versus 0.111–0.114 m/s from current roll/pitch alone. This is
useful predictive information beyond current attitude, not proof of causal visual
motion estimation. Current-gate-bearing prediction is much weaker: 0.349–0.365 rad
versus an attitude-only 0.418–0.419 rad. Hybrid roll-target error is 0.0123–0.0140;
the retained native source was already around 0.0126 on the weighted held-out task.
These recorded-flight correlations are not fresh autonomous course results.

The next idea, reviewed with Astra, is **matched-state anticipation supervision**:
hold the current gate and complete physical history fixed while placing the visible
next gate at two different lateral positions. Re-render the full prefix for each
branch, recompute its Hermite teacher, and teach the paired roll difference while
preserving source non-roll outputs. This specifically tests use of upcoming geometry
instead of confounding it with differing velocities or attitudes in mirrored flights.

`scripts/pragmatic_anticipation_lessons.py` constructs pairs of next-gate placements
0.30 m apart only where both satisfy the existing 0.20 m adjacent deviation-step and
0.50 m deviation limits around the **sloping launch-to-first-gate centreline**. These
are not bounds on raw world-Y differences. Other gate positions, angles and heights stay unchanged,
and requires the differentiated window to stay in the same current-gate phase.
Changed geometry is fixed from the beginning of each neural replay, not teleported
into an existing state. Source non-roll targets are recomputed on each branch's
actual modified sensory history. Whole base courses remain separated between fit
and diagnostic splits; the current pre-training audit does not claim learned flight.

The initial lesson audit rejected all twelve requested lessons because the new builder
incorrectly checked raw world-Y differences. This was corrected before any anticipation
training, with a sloped-centreline regression and a guard that the changed next gate
remains ahead throughout the copied history. Neither the sampler nor goal evaluation
was changed. The corrected audit is `anticipation-lesson-audit-v2.json` in the same run
directory. Counterfactual columns mean minus/plus next-gate placement; base course side
is recorded separately and must be balanced during learning.

The corrected audit produced all twelve requested lessons, six fit-side and six
held-out, with no rejections. Every pair has a visible image change. On held-out
lessons, teacher roll-contrast RMS ranges from 0.00099 to 0.00522 motor-drive units;
native contrast error ranges from 0.00067 to 0.00629. Thus upcoming geometry already
affects the fly's roll output, but not consistently in the teacher-required way.
This supports a learning experiment, not a claim of improved autonomous anticipation.

The next bounded training recipe is thirty updates from the retained replay source
on the existing 19,286 visual-to-roll edges at LR 3e-6, with biases and time constants
frozen. Mix an ordinary gate-one preservation window with one matched-state lesson,
cycling evenly over current gates 2–4 and both **base-course sides**. Keep current-weight
full-prefix replay and the twenty-frame differentiated window. Preserve source
non-roll outputs, use the actual Hermite teacher contrast sign rather than a fixed
left/right rule, and inspect held-out contrast alongside ordinary native development
flights at 10/20/30. Enforce the first-gate floor when selecting; do not continue past
this bound without a flight improvement. Any winner still requires a fresh full-course
check. The lesson builder is implemented and audited; this training integration has
now been added to `train_pragmatic_course_replay.py` as `--supervision late-roll-anticipation`.

Two matched ten-update screening trials are `pragmatic-anticipation-contrast1-001`
and `pragmatic-anticipation-contrast16-001`. Both start from retained replay best60,
use the same sink-fit-002 physical-history cache and lesson-selection seed 1260983,
and keep fresh Adam, LR 3e-6 and the existing roll-path mask. The last two whole pairs
in each cache bank are excluded from both the early-preservation and anticipation
gradient lessons. The remaining six pairs per bank supply the lessons. Fixed held-out
lessons use a separate selection stream, seed 1270983. Current-weight neural prefixes
are regenerated; there is no cached learner memory and no on-policy bank refresh
within this bounded comparison.

The sole difference between arms is the counterfactual **roll** contrast coefficient,
1 versus 16. The higher coefficient corresponds to a 0.005 rather than 0.02 motor-drive
contrast scale. Absolute motor errors, other-axis contrast and early-preservation
weights do not change. Early windows stay entirely before gate one; anticipation
windows cycle gates 2/3/4 with negative/positive base-course sides alternating, giving
five lessons on each side at the ten-update check. Their paired columns are two
next-gate placements, not two physical course sides.

Validation reports roll contrast error, pair-mean roll error, non-roll source error,
teacher alignment and the zero-contrast predictor's error. Because the source sometimes
has a worse contrast error than simply outputting no difference, an apparent reduction
alone is insufficient evidence of learned anticipation. Require beating the zero
baseline with positive teacher alignment before making that interpretation. Autonomous
development completion, with the hard first-gate floor, remains the checkpoint-selection
criterion. Extend only a promising arm to at most thirty total updates; neither this
contrast screen nor development selection establishes the >50% held-out goal.

Both ten-update screens finished without improvement. Contrast coefficient 1 produced
10/32 clean development completions, and coefficient 16 produced 11/32, against
12/32 for each matched source baseline. Both candidates passed the first gate on
29/32 versus source 28/32, so loss of initial approach was not the issue. Neither
candidate was selected; both best checkpoints remain update zero.

Held-out contrast error barely changed: the source's negative/positive-side RMSE
was 0.004090/0.004522; the two candidates were approximately 0.004091/0.004523 and
0.004090/0.004523. Neither beat a zero-contrast predictor on either side. The source's
pair-mean roll error against the full teacher was much larger, 0.021415/0.012915;
the trials mainly reduced that shared error slightly, not the response to upcoming
geometry. The matched source check is `source-validation-baseline.json` in the
contrast1 run directory. These results do not justify extending either arm.

One targeted follow-up, reviewed with Astra, is
`pragmatic-anticipation-source-mean-001`. It retains coefficient 16 and the same
source, seeds, lesson order, mask and ten-update bound, but uses
`--anticipation-mean-target source`. For each matched frame, the roll target is
`mean(frozen source roll) + teacher roll - mean(teacher roll)`. The true teacher's
paired difference is unchanged; the common steering command is anchored to the
existing fly instead of competing with the anticipation lesson. Targets outside the
native output range are rejected, not independently clipped. Non-roll preservation
and early lessons are unchanged. This is a training-label change only, not an actor
helper or decoder. Its validation pair-mean error now measures drift from the source;
the separate source-to-full-teacher mean error remains recorded. It may still fail
if preserving the source's mean preserves an inadequate steering policy.

The centered trial finished at 11/32 clean development completions versus its 12/32
source, with first-gate passes unchanged at 28/32. It retained update zero. Held-out
contrast RMSE changed only from 0.004089835/0.004521709 to
0.004087962/0.004521407: a very small reduction, still worse than zero contrast on
both sides and still negatively aligned with the teacher on the positive side.
Pair-mean drift from the frozen source was 0.00004250/0.00000709; non-roll RMSE was
0.00001648/0.00000400. No new controller is promoted.

These short, tiny-step trials are not a test of the network's ultimate learnability.
At the centered first update the raw gradient norm was about 1.247, and the proposed
update predicted only about a 0.05% improvement of the weighted lesson objective.
The trainer's `gradient_update_dot` is measured after clipping, so multiply by the
clipping factor when interpreting a raw first-order prediction.

`scripts/audit_pragmatic_anticipation_step.py` therefore tests one already-used
training lesson (bank 1190983, row 10, start 275), with targets held fixed. It scales
the same fresh Adam displacement by 0/1/10/100/1000, projects each proposal onto native
weight bounds independently, and restores the source after every test, including
exceptions. It compares the deliberately fixed source prefix state with complete
current-weight replay from zero, checking that both agree at scale zero. Contrast,
pair-mean drift, non-roll preservation, actual displacement and raw-gradient predicted
change are reported separately. This distinguishes inadequate update size from a
misleading truncated gradient without opening additional anatomy. It exports no
controller; improvement on one training lesson is neither generalization nor flight
success. The audit report is `step-audit.json` in the centered run directory.

The restored audit completed successfully, with the expected scale-zero agreement:

| Step multiplier | Fixed-prefix loss | Full-prefix loss | Full-prefix contrast RMSE |
| ---: | ---: | ---: | ---: |
| 0 | 0.374835 | 0.374835 | 0.006075 |
| 1 | 0.374643 | 0.374665 | 0.006074 |
| 10 | 0.372927 | 0.373173 | 0.006062 |
| 100 | 0.356220 | 0.346069 | 0.005837 |
| 1000 | 0.227499 | 0.124240 | 0.003396 |

Thus the small update was genuinely too small to change this lesson much, and
recomputing its complete neural history did not defeat descent. The 100× step reduced
the full-prefix objective by about 7.7%, with source pair-mean drift 0.000318 and
non-roll RMSE 0.000109. The 1000× probe moved those preserved quantities much more
(0.002995 and 0.000700), and still did not beat the zero-contrast predictor's 0.002518
RMSE or achieve positive teacher alignment. This is not a reason to deploy either
one-lesson perturbation or conclude that all truncated gradients are reliable.

The next bounded training run is `pragmatic-anticipation-source-mean-lr3em4-001`:
the same ten lessons, seeds, centered targets, mask and source, but LR 3e-4 rather
than 3e-6, with native development checks at updates 5 and 10. No stale diagnostic
prefix is used in training. The 1000× probe is not adopted as a learning rate.
Retain the ordinary first-gate selection floor and require actual flight improvement
before extending this run. The audit/replay refactor passed 509 regression tests;
tests do not establish course performance.

The larger-step run finished. Update 5 tied source completion at 12/32, improved
first-gate passes from 28/32 to 31/32, and shifted completions from 9 negative / 3
positive to 8 / 4. The selector retained update 5. Update 10 regressed to 9/32, with
30/32 first-gate passes, so the run will not be extended. Held-out negative-side
contrast error worsened to 0.004973 at update 5 and 0.005771 at update 10; positive-side
error fell slightly to 0.004479/0.004449 but remained negatively aligned. Neither
side beat zero contrast. A larger update can improve one lesson, but the balanced
training did not turn that into a higher development completion rate.

Because update 5 improved the initial approach and course-side balance, it was checked
on the existing separate 64-case bank, seed 1020983. It completed **15/64**, versus
source **12/64**, and passed the first gate on **60/64**, versus 53/64. Cumulative
gate passes changed from [53, 42, 32, 23, 12] to [60, 50, 41, 28, 15]; completion
split changed from 10 negative / 2 positive to 11 / 4. Neither evaluation had ground
contacts or invalid states. This is a modest improvement, not evidence of >50% success
or of consistent teacher-aligned anticipation.

That bank is independent of these anticipation fits, but was previously used in the
broader research; it is not a fresh final goal holdout. The report is
`independent-64.json` in the larger-step run directory. The second matched 64-case
bank, seed 2026091201, did **not** confirm higher completion: candidate **10/64**,
source **11/64**. First-gate improvement did repeat, 58/64 versus 52/64. Cumulative
passes were [58, 53, 37, 18, 10] versus [52, 42, 32, 17, 11], with no ground contacts
or invalid states for either. Reports are `fresh-64.json` and `fresh-source-64.json`;
geometry and rules are unchanged. Across the two banks the exploratory totals are
25/128 versus 23/128 completions, not convincing evidence of a new overall winner
and far below the goal. These two checks must not be relabelled as one preregistered
fresh 128-case final holdout.

Retain the earlier phase-balanced controller as the overall comparison/training
source. Preserve update 5 as a branch with replicated improvement through the first
two gates, not a demonstrated solution to the five-gate task. Do not extend the
centered imitation run: its larger changes expose incomplete later-course transfer,
and update 10 was worse. The short physical-tracking experiment below is the next
implementation step; no GPU training jobs from these trials remain running.

### Next control-responsibility experiment: physical lateral tracking

The design review with Astra identifies a more direct fallback than another motion
gain or latent-feature probe. The pragmatic imitation/replay gradients never pass
through the forelegs and aircraft: a better instantaneous motor label need not
produce a better trajectory. If the larger anticipation steps do not improve complete
flights, test **short closed-loop lateral-control learning** on the retained source.
The implementation and first physical-gradient preflight are now complete, as described
below. This does not replace the eventual racing or hint-fading goals.

- Keep the existing 19,286 visual-to-roll edges, fixed topology/signs, frozen biases
  and time constants, and the same deployed observations and motor interface.
- Use six clean native approach lessons covering current gates 2–4 and both base
  sides. Carry or reconstruct the actual foreleg/stick state from each native prefix;
  do not silently initialize the virtual sticks at a mid-flight lesson boundary.
  Recompute neural prefixes under current weights.
- Differentiate one second through the brain, rendered observations, physical
  forelegs and quad. All four native outputs remain live. Give the existing roll
  pathway the responsibility for reducing lateral position error along the Hermite
  path and terminal lateral-velocity error relative to its tangent. The path is a
  training target only. Retain early source-preservation lessons and source non-roll
  preservation; reject apparent improvements that materially reduce forward progress.
- Initially select windows ending before the next crossing. Soft annulus rendering
  supplies visual gradients, but visibility, checkerboard and role/event switches are
  not globally smooth. Check one proposed update with actual short-rollout finite
  differences before running ten updates and native course checks at 5/10. Use
  activation checkpointing if needed to fit the longer gradient window, without
  changing forward neural timing or adding external memory.

The success test remains uninterrupted full-distribution five-gate flight. Better
short-horizon tracking alone would justify a coverage/horizon investigation, not
promotion or a new deployed decoder. A training-only bearing/exit-line auxiliary on
existing descending neurons remains an alternative, but is lower priority than asking
the current controller directly whether its actions improve flight. The existing
velocity-information probe does not presently justify another motion-only trial.

`scripts/pragmatic_closed_loop.py` constructs fixed native physical-history lessons
from cache bank 1190983, excluding the last two held-out whole pairs. The three paired
lessons use current gates 2/3/4 and both base sides, selected with seed 1420983. They
end at least 0.15 m before the expected gate plane on their recorded source trajectories.
Neural prefixes are replayed from zero under current weights; actual foreleg states are
reconstructed from recorded source commands, with two physics ticks per frame and no
physical ticks during the ten-frame neural warmup. Cached actuator lag state is retained.

The loss differentiates fifty neural/visual frames and one hundred physical steps.
It combines mean squared lateral path error (0.5 m scale), terminal lateral velocity
error relative to `path dy/dx * current vx` (0.5 m/s scale), non-roll preservation,
and a forward-progress floor. Non-roll targets are the **fixed recorded native command
sequence**. They are not a detached teacher recomputed on each candidate's changing
images; that moving-target construction would invalidate a simple finite-difference
check of the preservation loss. All four actual actor outputs remain live.

#### Ground contact and trial acceptance

The full-course evaluator already permanently fails a flight at any physics step with
height at or below 0.03 m. Later recovery, or having passed every gate before the touch,
does not clear that failure. Course trials begin airborne. This is the simulator's
height-based contact proxy, not detailed tilted-body/propeller contact geometry. The
previous imitation objective was not directly a crash-penalized RL objective.

The new physical learner adds a smooth squared clearance cost below 0.30 m (weight 10)
and a 25-point penalty per failed short rollout, averaged across the paired batch. A
ground-contact flag is latched at every physical step. Candidate acceptance also rejects
**any** ground/ring/illegal-traversal/invalid event regardless of its scalar loss; touching
down and recovering cannot qualify. It requires at least 95% of recorded forward travel
and terminal forward speed, no more than 0.05 m altitude deterioration at the minimum
or endpoint, bounded non-roll error, and no gate crossing during this initially smooth
pre-crossing lesson. These are training-step guards, not changes to the full-course
evaluation rules. Tests explicitly cover touch-then-recover rejection and an upward
near-ground loss gradient.

#### Physical preflight and bounded training

`scripts/audit_pragmatic_closed_loop_step.py` verifies saved-action replay before any
gradient test. Its first attempt caught an implementation error: the shared imitation
bank's roll *teacher labels* were being used for boundary replay rather than its actual
recorded native commands. No gradient update ran in that attempt. The corrected check
reproduced all six physical-state components across all three paired lessons, including
actuator state. A regression now distinguishes source commands from teacher labels.

The completed audit is `runs/gate/pragmatic-closed-loop-step-001/report.json`. On its first
paired lesson, source tracking loss was 0.300102. A restored fresh-Adam proposal at
LR 1e-4 reduced fixed-prefix loss to 0.292742 and current-weight-prefix loss to 0.292094.
At 3e-4 these were 0.290879/0.290454; at 1e-3 they worsened to 0.415720/0.446585 because
terminal lateral velocity overshot. All tested steps met contact, altitude and progress
guards, so safety alone is insufficient selection evidence. First-order predictions
were optimistic even at small steps; actual finite changes, not a claim of globally
smooth rendering, justify the conservative training rate. The full one-second gradient
fits the available GPU without activation checkpointing. No controller was exported by
the restored audit.

`scripts/train_pragmatic_course_closed_loop.py` ran the bounded experiment in
`runs/gate/pragmatic-closed-loop-training-001`: ten updates, LR 1e-4, cycling the three
paired physical lessons and mixing an ordinary twenty-frame gate-one preservation
lesson each time. It tests fresh-prefix proposals at scales 1/0.3/0.1 and accepts only
an admissible decrease in both physical tracking and the combined objective. Rejected
proposals restore weights and optimizer state. Native 30-second development flights
on seed 1110983 select at updates 5 and 10 with the existing first-gate floor. The
new physical/safety helpers passed the 515-test regression suite before this training
run started.

The run finished without a promoted checkpoint. Six of ten proposals were accepted
on their short physical lessons, but full-course completion fell from source **12/32**
to **7/32** at update 5 and **9/32** at update 10. First-gate passes were 28/32,
24/32 and 27/32 respectively; update 5 also failed the first-gate selection floor.
Completion splits were source 9 negative / 3 positive, then 5 / 2 and 6 / 3.
All three full-flight checks had zero ground contacts and invalid states. The selector
retained update 0, and the phase-balanced controller remains the overall source.

This establishes that gradients through the real foreleg/aircraft model can improve
a short physical lesson, **not** that these six lessons improve whole-course flight.
One sampled twenty-frame early imitation window per update did not reliably preserve
the first approach, and later gates did not compensate for that regression. Do not
extend the same run or spend a fresh final holdout on it. A next physical-learning
experiment should investigate broader lesson coverage and preservation of the complete
early approach before increasing the number of updates; it must still earn promotion
on native full flights. The preserved anticipation branch remains available.

Astra's review identified a pre-crossing acceptance gap: a flight could cross the
expected gate's plane outside the annulus without changing its gate role. The helper
now separately latches expected-plane sign changes at every physics tick, including
a touch-and-return. This is deliberately stricter than full-course rules only for
these smooth training windows; exterior-plane misses remain legal in full evaluation.
The correction was made while the bounded run was already executing, so that run
used its original role-change-only crossing guard. There is no evidence it exploited
the gap, and no trained checkpoint was selected; its failed full-flight results stand.
Any future claimed short-window improvement must use the corrected guard. The final
suite passed **516 tests**, including this crossing regression and touch-then-recover
ground rejection. No GPU jobs from this experiment remain running.

### Whole-flight native physical training

The next trial changes the training trajectory and acceptance scope, not the actor.
`scripts/pragmatic_on_policy.py` runs uninterrupted native flights from launch, with
the ordinary ten-frame static neural warmup, 50 Hz RGB/brain and 100 Hz foreleg/quad
steps. Weights stay fixed throughout each complete 30-second training batch. Gradients
accumulate through fifty-frame physical segments, detaching neural, leg and all six
quad-state components between segments without resetting their numerical values.
One optimizer update follows the complete batch. Every new update starts fresh native
flights under the current weights; there are no teacher-driven or old-source physical
starts, and no externally supplied actor memory.

Astra recommended this bounded experiment over another replay-only diagnostic. It
addresses native state exposure, coverage of the whole first approach and reachable
later phases, and whole-flight acceptance together. It does not isolate which of those
changes caused any eventual improvement. Keep the existing 19,286 roll-path edges,
LR 1e-4, and frozen biases/time constants for this comparison.

The continuous objective averages squared lateral path error (0.5 m scale) plus
0.25 times squared tangent-relative lateral-velocity error (0.5 m/s scale), adds
0.05 times normalized non-roll preservation, and retains the smooth ground-clearance
loss throughout the full flight. Path geometry remains training-only. A nominal
pre-update native pass records actual executed commands and its clean-active time mask.
Both are frozen for the gradient pass and every proposed update's full native replay.
In particular, **earlier candidate failure cannot erase difficult tracking samples**.
Ground, ring, order, direction and invalid-state failures remain latched after recovery
or completion. Exterior-plane misses are legal, exactly as in full-course evaluation;
this is not the pre-crossing-only training screen used in the previous experiment.

`scripts/train_pragmatic_course_on_policy.py` tests restored proposal scales 1/0.3/0.1
on the same full training flights. Acceptance requires lower continuous objective,
nondecreasing clean-first counts on both sides, clean-prefix count and completions,
and no increase in any latched failure category. A failed proposal restores both
weights and optimizer state. These small training-bank guards do not prove generalization;
ordinary 32-case native development evaluation still selects checkpoints.

The initial run is `runs/gate/pragmatic-on-policy-50-001`, starting from the retained
phase-balanced source for two updates. Each update samples two mirrored pairs on seed
1520983 plus the update index minus one, processed as two-episode microbatches to bound
GPU memory. Assess native development after update 2 before considering an extension
to at most ten updates, with checks at 5/10. Three consecutive rejected updates or a
material clean-first development regression stops a run. If this approach stalls, a
matched hundred-frame truncation is the next horizon comparison, not a new anatomical
expansion. The six new focused tests cover unchanged numerical state across truncation,
native motor gradients, frozen tracking masks, all failure-category guards, legal
exterior misses, and primary-evaluator agreement on contact after the last gate.
The complete regression suite passed **522 tests**. Astra's read-only implementation
review found no substantive blocker. The script deliberately aborts on exceptions;
it is not an in-process retry/resume trainer. Exception-safe restoration of pending
Adam state would be required before adding that recovery behavior.

The first update exposed an acceptance-objective mismatch. Its full-scale proposal
improved the four native training flights from 1 to 2 clean completions and from 11
to 17 clean-prefix gates, while retaining all four clean first-gate passes. Latched
failures/ring contacts fell from 3 to 1, wrong-order episodes from 1 to 0, with no
ground, invalid or wrong-direction events. Nevertheless, the original continuous-only
rule rejected it because tracking/preservation loss increased from 0.181763 to
0.189587. Scale 0.3 was accepted: 2 completions, 15 prefix gates, loss 0.133474.
The original run keeps that recorded decision; its executing code is not changed.

After reviewing these complete metrics, Astra recommended reconstructing and testing
the one nominated full-scale proposal before trying a longer gradient horizon. The
trainer now offers an explicit `--acceptance-mode flight-first`: prioritize clean
completions, then clean prefix, and only then lower continuous loss, while retaining
the existing nondecreasing prefix/first-gate and failure-category guards. This is a
prospective training selection change, not a relaxation of flight evaluation or proof
that a four-flight result generalizes. The planned matched one-update reconstruction
uses the same source, seed 1520983, fifty-frame chunks and LR 1e-4; its native 32-case
development result will decide whether further validation is warranted. Do not turn
development into a search over every backtracking proposal.

The original two-update run finished: update 2 accepted scale 0.3 on its new training
bank, preserving 1/4 completions and 12 prefix gates while reducing the continuous
loss from 0.216407 to 0.200745. Full native development nevertheless fell to **7/32**
completions versus source **12/32**, despite clean-first passes improving from 28/32
to 29/32. The source remains selected (update 0). This rules out loss of first-gate
competence as the explanation for this particular regression, but does not isolate
the later-course failure cause. Do not extend this continuous-only run.

The nominated full-scale reconstruction is now running separately as
`runs/gate/pragmatic-on-policy-flight-first-001`, with the explicit flight-first
option and a one-update bound. Its source and training seed match the original first
update. The full revised regression suite passed **523 tests** before it started.

That matched reconstruction finished without reproducing the original full-scale
training result. Its nominal flights again had 1/4 completions and 11 prefix gates;
the fresh gradient proposal produced **1/4 completions and 14 prefix gates**, with
continuous loss 0.180655 → 0.189099. Flight-first selection accepted the prefix gain,
but native development was again **7/32**, with 29/32 clean-first passes. The selector
retained the source. These are a matched recipe and seed, **not** a byte-identical
recovery of the earlier unexported 2/4, 17-prefix proposal. The small-bank gain was
not reproduced; do not claim that exact earlier parameter vector was evaluated.
Do not divert into FP64/bitwise certification or extend this unsuccessful run.

Both whole-flight trials preserved first-gate competence better than the earlier
short-replay trial, but neither improved later-course completion. The original
two-update crossing summaries had larger lateral errors at gates two and four while
vertical errors did not increase; later-gate summaries are conditional on reaching
those gates, not a matched per-episode causal analysis. The next comparison concerns
longer gradient credit assignment versus broader native-course sampling, retaining
the source, full course distribution and deployed sensor/motor interface.

#### Broader native-course gradient batch

Astra recommends addressing coverage first: the repeated training-to-development
reversal makes a four-course batch a more immediate concern than extending the same
four courses' temporal horizon. This is a hypothesis, not proof of overfitting.
The next bounded run is `pragmatic-on-policy-broader-001`: restart the retained
source, keep fifty-frame chunks, the 19,286-edge mask, LR 1e-4 and flight-first
acceptance, but accumulate **eight mirrored pairs (16 episodes)** before clipping
and making one Adam update. Pair microbatches share the same fixed weights until
the complete training batch finishes.

Use a predetermined rotation through four eight-pair banks (32 training pairs),
seeds 1620983–1620986, with no source-success-based case selection and no development
cases. Six updates maximum revisit the first two banks after the first rotation.
Native development checks are at 2/4/6; existing rejection and first-gate stopping
rules remain. Report gate-phase exposure and clean prefix by side, because a larger
launch batch does not guarantee enough clean late-gate training examples.

The trainer's `--save-trial-proposals` option saves each actual evaluated parameter
proposal as an ordinary, explicitly training-only checkpoint before restoring or
accepting it. These local ignored artifacts are not automatically promoted, published
or all tested on development. They avoid relying on a numerically identical rerun to
recover a promising proposal. Before spending a fresh 128-case comparison, seek at
least four additional development completions over source, with neither side worse
and first-gate competence preserved. The original full-distribution >50% goal and
fresh-holdout completion requirements remain unchanged. If broader sampling still
repeatedly helps training but harms development, compare a hundred-frame gradient
horizon under this broader setup.
The broader run has started; **524 regression tests passed** beforehand. Both smaller
whole-flight runs are terminal, and no trained checkpoint from them was promoted.
Its repeated source development baseline is 11/32, with 28/32 clean-first passes;
the earlier runs obtained 12/32 from the same source. Retain the run's actual matched
baseline when judging its gains. Astra's read-only review of the rotation, per-side
reporting, proposal persistence and development schedule found no blocker.

#### Tracking-reference validity after an exterior miss

The broader run was intentionally interrupted **before its first optimizer update**.
The nominal 16 flights had 2 completions, 28 clean-prefix gates and 10 clean-first
passes (7 negative / 3 positive), with no ground or invalid states. Two negative-side
flights legally missed their expected gate outside the annulus, then kept flying far
beyond it without a course failure. Their current-gate phases lasted 815 frames at
gate five and 1,204 at gate two. The forward-only Hermite reference kept teaching
lateral tracking beyond those uncleared gates, although it specifies no recovery
back through them. With one-second truncated gradients, much of that tail cannot
credit the earlier approach that needed correction.

The two affected **pairs** account for about 84.4% of the total nominal tracking loss
(pair losses 3.239694 and 5.161709; eight-pair mean 1.244633). That includes their
valid prefixes; it is not a measured post-miss-only percentage. No positive-side
flight supplied gate-five frames. The partial evidence is preserved locally in
`runs/gate/pragmatic-on-policy-broader-001/interrupted-objective-diagnosis.json`.
Three pair-gradient logs completed before the interrupt; no optimizer update or
trained proposal from this run was applied/exported. Its source checkpoint is retained.

Astra recommends fixing this objective-coverage issue before spending further updates
or comparing horizons. `fly_course` now maintains a separate nominal tracking-validity
latch. It retains the first expected crossing's frame, but stops monotonic-path targets
on subsequent frames if that crossing did not cleanly pass the gate. Actual physics,
rendered role changes, all 30 seconds of flight, safety penalties and clean-course
evaluation continue unchanged. Exterior misses remain legal. The nominal eligibility
mask remains frozen across gradient/proposal evaluations: an earlier candidate miss
cannot delete additional training samples. Logs distinguish actual phase occupancy
from eligible tracking exposure and measure excluded post-miss tracking loss directly.
The post-miss diagnostic describes the current trajectory: it equals excluded loss
for nominal passes, but need not be excluded in a candidate using a frozen mask.

The next run will restart from the retained source with this single objective correction
and the same broader banks, mask, rate, six-update limit and 2/4/6 development checks.
Nominal per-update metrics are now saved before gradients, so another interrupted
trial will retain its evidence without reconstruction from console output.

The corrected run is `runs/gate/pragmatic-on-policy-valid-tracking-001`.
The final regression suite passed **525 tests**, and Astra reviewed the separation
between reference eligibility and flight legality. Its nominal first bank again has
2/16 completions, 28 prefix gates and first-side counts [7, 3]. On this same nominal
pass, eligible tracking loss is 0.254000 and excluded post-miss loss is 1.001583,
about **79.8%** of their combined value. Thus the problematic tail dominance is now
measured directly rather than inferred from whole-pair losses. These are unchanged
source flights, not improved control. The per-update nominal record is saved as
`nominal-u001.json`; the corrected gradient/update trial is still running.

A read-only timing check provides context for a later horizon test. Source membrane
time constants are about 21 ms (all-node range 20.18–22.19 ms), not hundreds of ms.
But the existing virtual foreleg's 0→0.1 roll-stick step reaches 90% at 0.60 s and
settles within 2% at 0.76 s; actual stick positions are 0.0308 at 0.20 s, 0.0818 at
0.50 s and 0.1011 at 1.00 s. Aircraft attitude and position respond after that motion.
This supports testing two-second rather than one-second gradient credit if needed;
it does not establish that horizon length caused the failures. Neither neural timing
nor the foreleg/quad plant is changed by this check.

#### Fixed motor-wiring metadata cache

The motor-pool reduction was reading sixteen fixed pool-boundary scalars from the
GPU on every neural frame. `ConnectomeController` now caches those integer ranges
on the host while retaining the same ordered gathers, means and antagonist
subtractions. This is topology metadata, not actor memory or a new control path.
The persisted offset buffer remains authoritative: direct and nested checkpoint
loads refresh the cache automatically. Explicit in-place topology edits must call
`refresh_motor_pool_ranges()`; ordinary device transfers need no refresh.

A short isolated RTX 5080 benchmark alternated old/cached/cached/old implementations
for 100 full brain-forward frames at batch two, after ten warmup frames. Times were
3.423 / 1.015 / 1.225 / 3.991 ms per frame. The benchmark shared the GPU with the
ongoing training run and is **not** an end-to-end training speed measurement.
Motor outputs matched in that check. Regression tests additionally compare outputs
and state gradients on unequal pools, prohibit per-frame scalar reads, and exercise
direct/nested checkpoint loads and explicit metadata refresh. The full suite passed
**529 tests**. No checkpoint format, precision, neural dynamics, physical dynamics or
learned weights change. The already-running corrected training process loaded the
old implementation; leave it uninterrupted and use the cache in subsequent processes.

#### Corrected update-one outcome and next comparison

The first corrected update completed in 904.6 s including the initial development
baseline. All three proposals were rejected; weights and optimizer state were restored.
The 16-course training bank produced:

| Proposal scale | Clean completions | Clean prefix gates | First passes, negative / positive | Ring contacts | Wrong-order episodes |
| --- | ---: | ---: | --- | ---: | ---: |
| Source nominal | 2 | 28 | 7 / 3 | 12 | 2 |
| 1.0 | 2 | 31 | 7 / 6 | 13 | 4 |
| 0.3 | 2 | 32 | 7 / 5 | 13 | 3 |
| 0.1 | 1 | 26 | 7 / 3 | 13 | 2 |

Ground, invalid and wrong-direction counts remained zero. The larger steps improved
early passes but added collisions and illegal traversals; the rejection is a real
flight tradeoff, not just a disagreement with the continuous objective. The smallest
step also lost a full completion. No proposal is promoted. The preset broader run
continues to bank 1620984, with its first post-training development check at update 2.

Astra's renewed design review recommends a matched **100-frame / two-second gradient
horizon** as the next comparison if this run does not improve full flights. Under
50-frame truncation, an action has at most one second, and on average roughly half a
second, of downstream gradient credit before a boundary; much of the physical effect
arrives after the measured 0.60 s foreleg response plus aircraft response. Keep the
retained starting checkpoint, eight-pair batches, four-bank rotation, 19,286-edge mask,
LR 1e-4, full-distribution geometry and all acceptance guards fixed. Compare late-gate
outcomes and held-out development, not just tracking loss. First finish the running
bounded trial; do not interrupt it for this comparison or change its parameters.

Longer credit cannot supply positive-side late-gate experience absent from the native
flights. Report eligible phase exposure by side in both runs. Missing exposure or a
failed horizon trial would not prove insufficient anatomical capacity. Opening more
wiring or adding privileged representation targets remains a later branch, not a
simultaneous change in this comparison. The >50% fresh-evaluation goal remains unmet.

The update-one case-level check strengthens the reason to keep those guards. Nominal
and gradient reflight agree on all sixteen clean-prefix counts before any optimizer
step. Each larger proposal loses the source's clean negative-side episode 6 (five
gates become three), gains a different negative-side completion, and turns episode
8's legal exterior miss into a ring contact. Thus the added collisions are not merely
the cost of reaching farther on previously failed positive-side courses. No exact
floating-point replay claim or additional reproducibility work is needed for this
observed tradeoff.

Update two's nominal bank (1620984) is now recorded: the unchanged source completes
3/16, clears the first gate in 13/16 with side counts [7, 6], and accumulates 40 clean
prefix gates. There are eleven ring-contact episodes, one wrong-order episode, and
no ground, invalid or wrong-direction events. Eligible gate-five exposure is 539
negative-side frames and **zero positive-side frames**, following [394, 0] in bank
one. Positive gate-four exposure increases from 189 to 446 frames. These are thirty-two
predetermined source flights, not evidence that the learner has improved; do not
replace difficult cases with source-success-selected courses. Update-two gradients
have completed; full update/development results are still pending.

#### One nominated rejected proposal, before changing the guard

Update-two scale 1 is a different case from update one. Its **actual saved**
`trial-u002-s1.pt` retains all three source completions, increases clean prefix from
40 to 47, and first-side counts from [7, 6] to [7, 7]. No individual episode loses
any clean-prefix gates. Positive-side episodes 1, 3, 7, 9 and 11 improve respectively
from 0→1, 3→4, 2→3, 1→4 and 1→2 clean gates. Negative-side episodes 0 and 10 change
from exterior misses to ring contacts without a prefix gain; positive-side episode
3 changes a ring hit at gate four into four clean passes followed by an exterior
miss at gate five. Total ring-contact/failure count rises 11→12 and wrong-order
count 1→2, with ground/invalid/wrong-direction still zero. The current guard correctly
rejects it under its declared rules; do not retroactively change that run's acceptance.

This reveals a possible training barrier between missing outside a gate and learning
to clear its aperture. Astra recommends nominating **only this checkpoint** for one
matched, ordinary 32-case native development comparison after the live run ends,
before spending on the hundred-frame comparison or redesigning acceptance. Re-evaluate
the retained source alongside it with the same current implementation, full geometry,
30 s tails and seed 1110983. Do not screen every rejected proposal afterward. Training
prefix gains are not evidence of transfer; retain the existing development-gain and
side/first-retention requirements before spending a fresh evaluation bank.

If that check is genuinely promising, consider a separate, bounded training-only
prefix-expansion mode: preserve every previously clean episode through the full tail,
require nondecreasing clean prefix per episode with a strict gain somewhere, preserve
per-episode ground/invalid protections, and allow additional ring contacts only in
previously incomplete episodes. New collisions still make those episodes absolute
failures in full-course scoring. Do not silently extend the exception to wrong-order
events: first inspect whether the extra violation occurred after an existing terminal
failure or introduced a new pre-failure gate skip. This mode is **not implemented or
enabled**. If the nominated proposal does not transfer, proceed with the already
specified matched 100-frame gradient-horizon comparison instead.

#### Development repeat is not a trained improvement

Update two has now finished: scales 1 / 0.3 / 0.1 all retain three training
completions, with prefixes 47 / 42 / 40, ring counts 12 / 11 / 10 and wrong-order
counts 2 / 2 / 3. All are rejected. The ensuing development repeat scores **13/32**
(10 negative / 3 positive) versus the launch baseline's 11/32 (8 / 3), both with
28/32 clean-first passes. This is **not a learned gain**: a CPU comparison of every
persisted controller tensor confirms that the saved `best-controller.pt` has exactly
the same controller state as the retained source. No weight update has been accepted.

The running trainer's old selector labelled that better repeat `selected_update=2`.
Treat it as an unchanged-source repeat, not a new trained controller. Future runs now
track actual accepted parameter changes and the last controller version evaluated;
repeating an unchanged controller cannot select a new best checkpoint. Repeat metrics
remain recorded and explicitly labelled. This fixes selection bookkeeping without
changing FP32 arithmetic, gradients, acceptance rules, dynamics or observations, and
does not alter the already-running process. All **533 regression tests pass**.
The nominated *changed* `trial-u002-s1.pt` remains a separate pending development
comparison; it is not the unchanged `best-controller.pt` just described. The broader
run has advanced to its third predetermined bank.

Bank three (1620985) supplies the missing side coverage without selecting cases by
success: the unchanged source completes 3/16 (1 negative / 2 positive), passes all
sixteen first gates, and has 483 negative / 477 positive eligible gate-five frames.
Thus the predetermined rotation does provide late-gate training on both sides; the
zero positive exposure in the first two banks must not be generalized to the whole
run. Its third gradient/proposal update is in progress.

#### Reconsider the training constraint, not the clean-flight definition

Astra's broader review notes that requiring every discontinuous component statistic
to improve or stay flat on each small training batch can prevent learning. For
example, bank three's 16/16 first-gate baseline makes even one lost first pass a veto,
regardless of any increase in complete courses. Zero accepted updates alone are not
evidence that the brain cannot learn this task. Preserving the source and judging
separate native development flights protects useful behavior more directly than
requiring every intermediate training statistic to improve monotonically.

Keep the running experiment and nominated-checkpoint comparison unchanged. A later
bounded six-to-ten-update alternative could retain finite-state, bounded-step and
strong ground protections, but rank proposals by one outcome-oriented score combining
clean completion, clean prefix and failure penalties, with tracking loss secondary.
Ring/order errors would still be penalized and invalidate clean completion, without
each category independently vetoing every optimizer step. Pure continuous-loss-only
acceptance is not the proposed remedy: it can also disagree with actual flight gains.
The development first-gate heuristic must not permanently veto a substantial increase
in complete-course success. No relaxed mode is implemented or enabled yet.

The saved update-two diagnostics do **not** prove that its extra wrong-order event
is only post-crash behavior. It occurs in pair five, where negative-side episode 10
loses valid tracking at frame 274 but first fails at frame 382; the later ring contact
does not identify which event failed it first. A useful candidate would require an
explicit event-sequence check before any exception for post-failure order violations.

#### A simpler reference may suffice for this course family

A CPU-only geometric check found that a horizontal line from launch XY through the
first gate clears all five annuli in 128/128 courses on the existing teacher bank
1037983, and 16/16 in each of training banks 1620983–1620985. The 0.50 m course-deviation
limit is slightly smaller than the 0.53 m clean radius; this family need not demand
substantial anticipatory turns. This geometric check starts at first-gate height and
is not a physical or learned flight result.

A subsequent **privileged four-axis teacher** comparison used the sampled airborne
aircraft and initial foreleg/stick states, all original varied gates,
50 Hz commands, 100 Hz physical steps and all 30 seconds of latched safety checks:

| Training reference | Clean flights | Maximum crossing radius | 95th-percentile radius |
| --- | ---: | ---: | ---: |
| Curved reference through all gates | 128/128 | 0.06446 m | 0.03769 m |
| Straight XY line through the first gate | 128/128 | 0.51052 m | 0.35337 m |

Both keep world-X heading and have no ring, illegal, ground or invalid events. The
straight reference uses launch, the first gate and a synthetic reference point 20 m
farther along that XY line at first-gate height; the original five physical gates do
not change. The reference still smoothly handles the sampled initial altitude offset.
No fly controller ran in this CPU comparison. Records are in the local ignored
`straight-line-teacher-audit.json` under the corrected run. This is a previously used
teacher bank, **not** a fresh goal holdout, and no controller was trained or exported.

The worst straight-line crossing has only about 0.0195 m radial margin, which native
altitude/speed errors could consume. After the nominated checkpoint test, Astra
recommends a matched **curved versus straight roll-only takeover** before changing the
training reference: let the native controller clear gate one, replace only later
roll, and keep native pitch/yaw/throttle, live images and uninterrupted neural state.
Compare clean full-flight completion, both sides, clearance and roll-command effort.
If straight guidance retains the rescue with less steering, a simpler training-only
reference is a reasonable alternative to unnecessary curved-path tracking.

Any line slope, desired heading or stored teacher reference remains outside the actor;
retained intent must emerge inside the native recurrence. Keep the agreed course
distribution unchanged, and do not call simple line-following proof of anticipation
or general racing. Native five-gate success, live visual correction, and later
genuinely turning-course competence are separate claims. This result motivates a
simpler learning target, not a broader claim or more anatomical parameters.

#### Corrected short-horizon run finished; nominated proposal did not transfer

`pragmatic-on-policy-valid-tracking-001` finished normally after 3,048.76 s, stopping
at three consecutive rejected updates. All nine scale proposals were rejected and
**no parameter update was accepted**. Its final unchanged-source development repeat
was 12/32. The old process's `selected_update=2` / 13/32 record remains the previously
documented unchanged-source selection artifact, not learned progress.

The one nominated, actually changed `trial-u002-s1.pt` was then evaluated alongside
the retained source on the same ordinary 32-case development bank (1110983), using
the current implementation, all original varied gates and the complete 30 s flight:

| Controller | Clean courses | Negative / positive | Clean first passes | Clean prefix gates |
| --- | ---: | --- | ---: | ---: |
| Retained source | 12/32 | 9 / 3 | 28/32 | 101 |
| Nominated update-two scale-one proposal | 10/32 | 8 / 2 | 31/32 | 92 |

Ring-contact episodes rose from 19 to 20 and wrong-order episodes from two to three;
ground and invalid counts stayed zero. Better first-gate performance did not transfer
to better complete courses. The proposal is **not promoted**, no fresh holdout is
spent on it, and the other rejected proposals will not be screened for a lucky result.
The matched records are `nominated-u002-s1-development-{source,candidate}.json` in
the same ignored run directory.

#### Curved reference retained after the roll-only comparison

The matched roll-only diagnostic is complete on seed 1110983. The source drives all
four axes through gate one; only subsequent roll comes from the privileged teacher.
Native pitch, yaw, throttle, live RGB and the full continuous neural/physical state
remain in use. All five actual gates, clearance rules and 30 s tails are unchanged.

| Teacher roll reference | Clean courses | Negative / positive | Clean first passes | Ring-contact episodes |
| --- | ---: | --- | ---: | ---: |
| Curved | 27/32 | 14 / 13 | 28/32 | 4 |
| Straight | 24/32 | 11 / 13 | 28/32 | 7 |

Both have zero ground/invalid/wrong-direction events and one wrong-order episode.
Straight guidance loses three negative-side completions, so its narrower geometric
margin matters when the other axes remain native. Mean absolute roll-motor effort
is 0.004405 versus 0.005140 for curved guidance; RMS is 0.007566 versus 0.008316.
These effort summaries include frames whose current gate is two through five,
including post-failure tails still in those phases. Their differing occupancy means
they are descriptive, not a matched clean-flight energy comparison. Neither result
is native five-gate success. Full records are `straight-roll-takeover-audit.json`.

Keep the simpler reference as a documented alternative, but retain curved tracking
for the next native trial. Astra concurs: reduced steering has not earned replacing
the reference after losing three completions.

#### Matched two-second physical credit trial started

The next bounded run is `runs/gate/pragmatic-on-policy-valid-tracking-100-001`, with
100-frame / two-second gradient chunks instead of 50-frame / one-second chunks.
The source checkpoint, 19,286-edge plasticity mask, LR 1e-4, eight mirrored training
pairs, seeds 1620983–1620986 in four-bank rotation, six-update limit, development
checks at 2/4/6 and all acceptance guards remain unchanged. Total-flight loss
normalization remains 1/1,500; truncation changes gradient credit, not numerical
state, actor timing or deployed memory. Actual proposals remain saved locally.
The already-tested motor-metadata cache and unchanged-controller selection fix are
also present; neither deliberately changes controller arithmetic or learned state.

The reason to try this is the measured 0.60 s foreleg response plus subsequent
aircraft motion, not evidence that longer horizons already improve learning. Judge
the run by admissible whole-flight gains and independent native development, not
just gradient magnitude or tracking loss. If it again produces only rejected steps,
an explicitly separate outcome-oriented acceptance experiment is the next branch,
not more horizon extensions or an immediate expansion of anatomy. The >50% fresh
native five-gate objective remains unmet.

#### Optional outcome-based update acceptance prepared, not enabled

While the matched 100-frame comparison runs, a separate opt-in
`--acceptance-mode outcome` has been implemented and reviewed by Astra. It uses the
training score `5 * clean completions + clean-prefix gates - 2 * failed episodes`,
with continuous loss only breaking score ties. Prefix credit already stops at the
first latched failure; a completion requires all 30 seconds to remain clean. Ring
contacts and illegal traversals still fail a flight and are separately reported,
but their counts no longer each veto an otherwise better aggregate training score.
The rule does not add another ring/order penalty on top of the failure penalty.

Every proposed update must have **zero ground-contact and zero invalid episodes**,
and finite objectives. This holds even if its nominal bank already has such events;
the nominal safety condition is recorded, not used to redraw easier cases. The
physical near-ground loss, all course rules, full-flight tails, bounded optimizer
steps, starting anatomy and actor interfaces remain unchanged. No teacher is added
to the actor. The existing `continuous` and `flight-first` modes retain their rules.

This score is explicitly a training proxy: several partial-course improvements can
outweigh losing a completion. Development selection remains clean-completion-first,
then favors the worse course side, and still requires genuinely changed weights.
In outcome mode, a first-gate regression alone does not veto a better completion
result or stop the bounded experiment. The retained source remains available, and
the normal fresh native evaluation is still required before claiming the goal.
Avoiding every gate can also score above crashing immediately; monitor stalled or
exterior-miss flights instead of treating score improvements as successful navigation.

The full suite passes **540 tests**, including physical event sequences that hit a
ring, skip a gate or traverse backwards before eventually passing all five. These
recoveries retain zero clean prefix/completion, and a ground touch after all five
passes removes the completion bonus. Tests also cover hard ground/invalid rejection,
score ties, finite losses, completion-first development and unchanged-source repeats.
Outcome runs carry the distinct `native-whole-flight-tbptt-outcome-v1` experiment label.

This mode is **not enabled in the live matched comparison**, whose first nominal
bank has the same 2/16 completions and 28 prefix gates as the 50-frame comparison.
Its longer backward passes are running within the RTX 5080's memory capacity; no
optimizer proposal has yet been assessed. If that bounded comparison also fails to
produce transferable progress, start a separately named source-restarted outcome
trial, keeping the 100-frame horizon, curved reference, banks and other settings
fixed. Do not reinterpret rejected proposals from the current run as accepted steps.

#### Two-second comparison: first update rejected at every step size

The 100-frame run's first update finished at 984.68 s including its baseline and
nominal flights. No weights were accepted; the run has advanced to the next
predetermined bank, 1620984. Its source nominal has 2/16 clean courses, 28 clean
prefix gates and first-side counts [7, 3]. The actual saved proposals give:

| Step scale | Clean courses | Clean prefix | First passes, negative / positive | Ring contacts | Wrong-order episodes |
| --- | ---: | ---: | --- | ---: | ---: |
| 1.0 | 2/16 | 31 | 7 / 5 | 13 | 3 |
| 0.3 | 2/16 | 31 | 7 / 5 | 13 | 2 |
| 0.1 | 1/16 | 26 | 7 / 3 | 13 | 2 |

The source has twelve ring-contact and two wrong-order episodes. Ground, invalid
and wrong-direction counts stay zero for every proposal. At both larger scales,
source-clean episode 6 falls from five to three prefix gates while episode 12
improves to a full completion; this is not preservation of both source successes.
The smallest scale retains only source-clean episode 2. Continuous loss rises from
0.253734 to 0.258254 / 0.263311 / 0.268916. This first bank therefore does not show
better native flight from the longer gradient window. No proposal is promoted or
sent to a fresh holdout. Keep the later banks and development checks unchanged.

The pre-clipping gradient norm is 4.81256, versus 0.26844 in the corrected 50-frame
first update. A larger gradient is not evidence of more useful credit: both runs
reject every proposal on this bank. This observation does not by itself identify
neural or physical instability, and does not justify changing the live trial's rate.

#### Later throughput option, without changing either current comparison

Astra's read-only audit finds no cross-episode coupling in the controller or plant.
A later opt-in implementation could batch the sixteen **no-gradient** nominal/trial
flights, while keeping each gradient microbatch at two episodes. Slice the batched
motor/mask traces by episode for gradients, keeping their existing 1/8 scale. Use
the same batched execution for both nominal and candidate acceptance scores.
Preserve individual prefix, failure, timing and loss records; the current B2
`clean_prefix_by_side` ceases to identify individuals at B16. Phase means must use
summed loss numerators and frame counts, not averages of pair means.

The tiny detached motor/mask trace can also stay on GPU until a rollout ends,
avoiding two host copies per frame without retaining a neural computation graph.
These changes are not implemented or enabled. A representative-bank comparison
would need to check gross B2/B16 behavior and the sliced-reference gradient replay;
ordinary FP32 differences near crossing boundaries do not require byte identity.
Since gradient work remains pairwise, any total speedup will be limited—not sixteen
times faster. Keep the active horizon run and first outcome-mode comparison
unchanged; this is a later throughput option, not evidence of improved flying.

The small trace-copy optimization has now been implemented independently of the
unimplemented B16 batching option. `fly_course` keeps only detached motor and mask
tensors on their execution device, stacks them after the complete rollout, and then
returns the same CPU trace interface. That moves two explicit host copies per frame
to two per rollout (3,000 to two for a 30 s trace), without changing controller,
physics, loss, gradient truncation, batch size or acceptance arithmetic. The active
100-frame process already loaded the old implementation and remains unchanged;
future processes can use this diagnostic-storage optimization without changing the
experimental recipe. No end-to-end speedup is claimed yet. All **542 tests pass**,
including tracing with and without backward passes, detached outputs, host copies
only after the last frame, and preserved trainable gradients.

#### Next small diagnostic: neutral roll after the first gate

Astra recommends one inexpensive diagnostic after the current GPU run ends, before
spending on the outcome-mode fallback: compare native flight, curved teacher roll,
and **zero roll-motor difference** after gate one on the same 32 development cases.
Keep native pitch/yaw/throttle, live RGB, continuous neural/physical state and all
30 s of scoring. Record completion conditional on a clean first gate by side, plus
the usual gate crossing errors and safety events. The prepared local script is
`neutral-roll-takeover-audit.py` in the 100-frame run directory; it has not run yet.

This is not redundant with straight teacher guidance, which still actively corrects
the flight. Neutral motor drive lets the foreleg/stick relax toward center and the
acro controller damp roll rate; it does **not** command wings-level attitude and
may retain the existing bank and lateral motion. A rescue would suggest that the
native late-roll commands are harmful and motivate a simpler training-only target.
A failure rejects this neutral intervention, not every constant/open-loop policy
and not all alternatives to feedback steering. The external gate-one switch is
diagnostic only: it cannot be deployed or counted toward the native completion goal.
The original courses, source checkpoint and fresh-evaluation requirement stay fixed.

#### Two-second update two learns on its bank, but does not yet transfer

Update two's scale-one proposal is the first **accepted** step in the longer-credit
run. On bank 1620984, clean completions increase 3/16→5/16, clean-prefix gates
40→46, and first-side counts [7, 6]→[7, 7]. Ring-contact/failure episodes fall
11→9; wrong-order stays at one, with ground/invalid/wrong-direction all zero.
All three source-clean episodes (4, 8, 12) remain clean. Episode 0 improves from
three gates to five and positive-side episode 9 from one to five. Two previously
incomplete episodes lose one prefix gate each. This is an accepted aggregate flight
improvement under the existing rules, not a retrospective relaxation of them.

Continuous loss rises 0.210187→0.217283; the declared flight-first mode correctly
prioritizes the actual completion gain. The pre-clipping gradient norm is 6.05916.
This result differs from the 50-frame run's rejected update-two proposal, which
kept three completions but added failure/order events. The longer horizon can
produce an admissible training improvement on this bank; that is not yet transfer.

The actual changed controller's ordinary 32-case development check gives **8/32**
clean courses (5 negative / 3 positive), versus the retained source's 12/32 (9 / 3).
Clean-first passes rise 28→29, but prefix gates fall 101→91 and ring-contact episodes
rise 19→22. Wrong-order stays at two; ground and invalid counts remain zero.
The selector correctly keeps update zero as best and labels this as a genuinely
changed controller, not an unchanged-source repeat. No fresh holdout is spent.

At 1,619.23 s elapsed, the bounded run continues into bank 1620985 from the accepted
weights, as planned. The first-gate stop condition is not triggered. Subsequent
banks must establish whether this early training gain develops into broader native
flight improvement. Neither the >50% goal nor the value of outcome-mode fallback
has been settled by this one accepted step.

#### Third bank supplies both-side late experience; another smaller step accepted

On bank 1620985, the update-two controller passes all sixteen first gates and
completes 2/16 full courses, one on each side. It has 39 clean-prefix gates, fourteen
ring-contact episodes and three wrong-order episodes, with ground/invalid/direction
counts zero. Eligible tracking frames by gate are [2565, 1130, 670, 416, 362] on the
negative side and [2654, 1283, 716, 621, 330] on the positive side. Both sides now
supply gate-five training; no easier courses have been substituted.

Update three's scale-one proposal loses a completion (1/16, prefix 38) and is
rejected. Scale 0.3 is accepted: 2/16 completions, prefix 41, all sixteen first
passes, thirteen ring-contact episodes and one wrong-order episode. Ground, invalid
and wrong-direction counts stay zero. It retains positive-side completion 9, loses
negative-side completion 12 and gains negative-side completion 2. This is an
aggregate improvement, not preservation of every prior successful flight.
Continuous loss rises 0.215057→0.234027, while the flight-first rule favors the two
additional clean-prefix gates and reduced failure counts. No development check was
scheduled at update three. At 2,309.73 s elapsed, the run continues to the fourth
predetermined bank with two genuinely accepted parameter updates; the selected best
remains the original source.

The first three nominal banks allocate respectively 61.56%, 50.37% and 38.10% of
eligible tracking loss to gate-one approach, versus 7.57%, 3.10% and 5.13% to gate
five. These are measured **loss contributions**, not gradient norms or proof of
which phase drives learning. They come from the logged phase loss means weighted
by eligible frame counts. No phase weighting or objective changes are made here.

#### Bounded fixed-input sensitivity check does not justify a precision detour

Astra ran one CPU-only FP32 check on the retained source: two mirrored initial
scenes, ten static warmup steps, then a single approximately 1e-6-RMS membrane
perturbation in one copy of each brain, followed by ten seconds of identical fixed
RGB and roll/pitch inputs. No aircraft physics or gate events run. Peak roll-motor
differences are 5.66e-7 and 1.79e-7, at least 8,800 times smaller than a 0.005
reference correction. All other motor-axis differences also remain below 4.2e-7.
Identical unperturbed copies have zero motor differences. Some neuronal differences
grow without producing material motor amplification.

This provides no evidence of control-scale motor sensitivity to that single tiny
state perturbation at those operating points. It does not establish closed-loop
stability, cover moving images or repeated perturbations, or identify event
thresholds as the cause of replay differences. No further sensitivity investigation
is planned now; keep judging native flight transfer. The 33.21 s check is recorded
in `cpu-fixed-input-membrane-sensitivity.json` under the 100-frame run. Controller
files and the live GPU experiment were untouched.

#### Fourth accepted-bank check: transfer worsens despite local prefix gains

Update four uses the fourth predetermined bank, seed 1620986. Its scale-one proposal
keeps three clean completions out of sixteen, increases clean-prefix gates 39→41,
and improves first-side counts [7, 5]→[7, 6]. Ring-contact/failure episodes remain
twelve and wrong-order episodes fall four→two; ground, invalid and wrong-direction
counts remain zero. The clean cases change from [0, 8, 10] to [0, 6, 8]. Continuous
loss rises 0.184865→0.238931 and the pre-clipping gradient norm is 13.6570. The
declared flight-first rule accepts the prefix gain; this is not evidence that the
tracking loss or every individual flight improved.

Development transfer deteriorates further: **5/32** clean courses (4 negative /
1 positive), compared with update two's 8/32 and the retained source's 12/32.
First-gate passes remain 29/32, but clean-prefix gates fall to 79, ring-contact
episodes rise to 26 and wrong-order episodes to four. Ground and invalid counts
remain zero. At 2,974.43 s elapsed the selector still retains the actual original
source (update zero), and the bounded run continues for its two remaining updates.
No fresh holdout is consumed and no native improvement is claimed.

Three accepted parameter updates now argue against rejection guards being the
sole obstacle. After the final check, retain the planned neutral-roll diagnostic
before deciding whether another outcome-mode run is worthwhile. A further
training-target alternative is a current-gate, body-relative bearing/velocity
teacher, avoiding a world-coordinate path label that can depend on already passed
gates. This is a hypothesis, not an implemented or proven replacement. It still
requires motion estimation from visual history, must bound near-plane commands,
and may introduce harmful target jumps at gate transitions. First test teacher
feasibility and roll-only assistance with the other three axes native; assisted
success would not satisfy the goal. Keep the curved reference and all current
course/scoring rules unchanged until comparative evidence supports a change.

The local-feedback alternative is now prepared, but **has not been flight-tested**:
`current_gate_roll_motor` in `src/flydrone/course_teacher.py`, exercised through
`scripts/audit_pragmatic_current_gate_roll.py`. It rotates relative gate position
and velocity into the current yaw-aligned horizontal frame, requests a bounded
intercept velocity, applies velocity damping plus nominal drag compensation, and
uses the existing rate/foreleg mapping. The forward-distance denominator is floored
at 0.5 m and velocity/acceleration are bounded. Behind the gate or after completion,
it requests lateral braking. This is an approximate collision-course law, not an
exact constant-bearing guarantee and not an externally decoded actor feature.

The diagnostic replaces only roll after gate one, keeps the other three neural
outputs and neural state live, and uses ordinary full-30-s evaluation. The evaluator
still constructs its usual path object, but the new target never reads it. True
geometry, velocity, roll rate and the gate-index intervention remain teacher-only
privileges, so its result cannot count toward the native goal. Thirteen focused
tests pass, including mirrored/global-heading invariance, bounded near-plane
behavior and actual two-second lateral braking through the foreleg/quad dynamics.
Those checks establish basic implementation behavior, not gate-flight success.

Astra recommends activating this diagnostic only if neutral roll does not already
provide a strong rescue. A useful initial screen is about 25/32 clean assisted
courses, at least 80% conditional completion after a clean first gate on each side,
and no ground/invalid events. This is a predeclared diagnostic screen, not a new
goal threshold. If it preserves most of the curved teacher's rescue, consider a
source-restarted local-feedback learning trial before merely relaxing acceptance
rules. The current six-update process is unchanged.

#### Mass-distribution correction

The standard mirrored-course sampler sets `mass_scale` to ones, and the multi-gate
sampler retains that value. Current native course results therefore use **nominal
mass**, with randomized initial aircraft pose/rates/height and varied gate geometry;
they do not establish robustness to mass variation. The earlier straight-line
teacher note and its generated JSON described randomized masses without supporting
mass measurements. That characterization is withdrawn above; the archived JSON is
not rewritten. The new CPU teacher preflight will explicitly record the actual mass
range. This corrects the description, not the benchmark: no actor inputs, mass
distribution, course geometry, safety checks or completion threshold change.

#### Current-gate roll passes teacher-only physical feasibility

The bounded CPU preflight on the already-used 128-case teacher bank 1037983 clears
all five gates in **128/128** cases for both conditions: curved all-axis teacher,
and curved teacher until the first actual pass followed by current-gate roll with
curved-teacher pitch/yaw/throttle. Both sides are 64/64, all 640 gate passes receive
clean-prefix credit, and there are no ring, wrong-order, wrong-direction, ground or
invalid failures over the full 30 s. First-gate passage ticks match between the
conditions. Initial cases are unchanged, mass scale is exactly one (35 g), commands
run at 50 Hz and the same foreleg/quad dynamics at 100 Hz. No gains were tuned.

Across all crossings, current-gate roll gives radial mean / p95 / maximum of
0.02231 / 0.06540 / 0.11181 m, versus 0.01438 / 0.03810 / 0.06444 m for the curved
teacher. Both have substantial clearance within the 0.53 m clean radius. The check
took 8.67 s on CPU and loaded no neural actor or renderer; it establishes feasibility
with privileged control of the other three axes, **not native fly success** and not
the requested fresh holdout. The generated record is
`current-gate-roll-cpu-preflight.json` under the 100-frame run directory.

Keep the neutral-roll control first when the current GPU process ends. If that
does not strongly rescue flight, the local-roll/native-other-three comparison now
has a physically feasible teacher and remains the next useful diagnostic.

#### Update five rejects two completion gains for an order-count regression

Returning to training seed 1620983, the current controller completes 3/16 courses
(negative-side episodes 2, 10, 12), with 32 prefix gates, first-side counts [7, 4],
eleven failed/ring-contact episodes and three wrong-order episodes. Ground, invalid
and wrong-direction counts are zero. Scale one loses a completion (2/16) and adds
two failed/ring-contact episodes, so it is rejected.

Scales 0.3 and 0.1 each preserve the three clean episodes and add negative-side
episode zero, giving **4/16**, prefix 34 and unchanged first-side and failure/ring
counts. Both are rejected solely because wrong-order episodes increase three→four.
Continuous loss is 0.213296 and 0.208499 respectively, against nominal 0.213783.
The existing outcome-mode proxy would improve 25→32 for either proposal, but the
running flight-first experiment correctly retains its declared rules. No independent
development check is made for these rejected proposals; all completions on this
bank remain negative-side, and no transfer or post-failure-only explanation is
assumed. The raw gradient norm is 2.40874.

At 3,924.12 s elapsed, weights and optimizer state are restored to their pre-update
values and the process proceeds to its sixth/final update. The selected source is
still update zero. These proposals keep the outcome-mode fallback scientifically
relevant, but do not overturn the next-action priority: neutral-roll control, then
local roll with native other axes if needed, before another long learning run.

#### Optional simpler targets are available for the existing replay learner

`train_pragmatic_course_replay.py` now accepts `--roll-teacher curved|neutral|current-gate`,
with the existing curved target still the default. Non-curved choices require
`--supervision late-roll-preserve` and a frozen source; incompatible all-teacher or
anticipation modes are rejected. Native and roll-assisted collection can therefore
use the same target law that was screened by the diagnostic. Before the first gate,
all four labels preserve the source; afterward, only roll changes. The other three
curved-teacher outputs are discarded in favor of source outputs before any
non-curved assisted flight is driven.

The selected law is recorded in collection and checkpoint metadata. Current-weight
neural-prefix replay, sensor inputs, graph mask, first-gate/source preservation,
physical forelegs and full native evaluation remain unchanged. All **550 tests pass**,
including label-only axis replacement and rejection of unsupported collection modes.
No new learning job has started: activation still depends on the neutral/current-gate
assistance results, and the ongoing sixth native physical-gradient update is untouched.

Astra's review highlights that the older 60-update preservation replay restored the
retained source and cleared Adam at unsuccessful development checks 20 and 40. It
was therefore three short attempts, not necessarily sixty continuous fitting steps.
The trainer now has opt-in `--keep-latest-training-weights`, which leaves best-export
selection intact but avoids that automatic training rollback. Its default behavior
is unchanged. `--check-replay-fit` also reports motor RMSE by phase, side and axis on
five fixed paired windows, with current-weight neural-prefix recomputation. These
are explicitly examples from the training collections, **not held-out validation**.
The added fitting/reporting changes pass 26 focused regression tests.

If neutral or current-gate assistance meets the declared rescue screen, the proposed
learning pilot restarts from the retained source with the same 19,286-edge mask,
frozen bias/time constants, **20 updates at LR 1e-5 and 20-frame unrolls**, checks at
10/20, latest-training-weight continuation and fixed replay-fit reporting. Start
with training seed 1690983 and assisted seed 1790983; development stays the already
used 1110983 bank. This changes optimization as well as teacher labels, so it is a
practical pilot, not a teacher-label-only causal comparison. The foreleg lag does
not by itself justify longer action-imitation unrolls, since replay detaches physics.

Continue toward sixty updates only if late-roll fitting improves on both sides
while early/all-other-axis preservation and native flight remain useful. If fitting
barely moves, inspect optimization before rejecting the teacher idea. If imitation
improves while native flights deteriorate, stop this branch rather than extending
the same replay blindly. Native-first window selection still makes assisted lessons
a fallback; this known coverage limitation is unchanged for the first pilot. No new
learning trial or altered deployment has been launched yet.
