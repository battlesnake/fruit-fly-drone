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

#### Two-second physical-gradient run finishes without native transfer

The sixth/final update returns to seed 1620984 with 4/16 clean courses (2 per side),
45 prefix gates, first-side counts [7, 8], ten failed/ring-contact episodes and no
wrong-order, wrong-direction, ground or invalid events. Its scale 1 / 0.3 / 0.1
proposals give respectively 3 / 2 / 3 completions and 44 / 45 / 47 prefix gates; all
lose a completion and are rejected. The first two add one ring/failure episode and
one wrong-order episode; the smallest keeps the nominal failure counts. The raw
gradient norm is 20.0866. No sixth-update weights are retained.

Final development is **6/32** clean (5 negative / 1 positive), first passes 29/32,
prefix gates 81, ring-contact episodes 25 and wrong-order episodes four, with ground
and invalid counts zero. This repeats the unchanged update-four controller, and the
report correctly labels it as such; the one-case difference from its prior 5/32
check is not a learning gain. **The retained source remains selected at 12/32.**

The process completed normally after 4,971.12 s (about 83 minutes), with three
accepted updates out of six. Longer physical credit produced admissible training
steps but did not improve varied-course development. No fresh holdout is consumed,
no goal success is claimed, and the run is not extended. The prepared matched
native/curved-teacher/neutral-roll audit was launched after this GPU process exited;
all conditions use the original retained source and full 30-second scoring.

#### Neutral roll fails the matched native-other-axes comparison

The completed 32-case audit on development seed 1110983 gives:

| Post-first-gate roll control | Clean courses | Negative / positive | Ground / invalid episodes |
| --- | ---: | ---: | ---: |
| Native, unchanged source | 14/32 | 10 / 4 | 0 / 0 |
| Curved teacher | 27/32 | 14 / 13 | 0 / 0 |
| Zero roll-motor difference | 2/32 | 2 / 0 | 4 / 4 |

All conditions pass the first gate cleanly in 28/32 cases (15 negative, 13 positive).
Conditional full completion after the first gate is 50.0%, 96.4% and 7.1%
respectively. Native / curved / neutral clean-prefix counts are 102 / 139 / 52;
ring-contact episodes are 17 / 4 / 19 and wrong-order episodes 2 / 1 / 1. Wrong
direction is zero throughout. Ground and invalid counts can overlap and must not
be added as distinct failures. Each condition retains all 30 s of scoring and the
same native other three axes, live visual inputs and continuous neural state.

The unchanged source's 14/32 here is a repeat measurement, not a newly learned
improvement over the earlier 12/32. Neutral drive is decisively not a useful target
for the proposed learning pilot on this bank. This rejects that intervention, not
every possible open-loop policy. The matched curved rescue remains strong. The
record is `neutral-roll-takeover-audit.json` under the 100-frame run; the three
conditions took about 90 s total. The current-gate roll/native-other-three diagnostic
was launched only after this process exited, with no simultaneous GPU training.

#### Current-gate steering retains the rescue with native other axes

The subsequent current-gate roll diagnostic achieves **26/32** clean courses,
**13/16 on each side**, with the fly controlling pitch/yaw/throttle throughout.
Clean first passes stay 28/32; conditional completions are 13/15 negative (86.7%)
and 13/13 positive (100%). It records 137 clean-prefix gates, five ring-contact
episodes, one wrong-order episode and no wrong-direction, ground or invalid events.
All 30 s are scored. The matched curved teacher gives 27/32 and neutral roll 2/32.
The 32.00 s diagnostic is recorded in `current-gate-roll-native-other-axes.json`
under the 100-frame run directory.

This meets the predeclared assisted screen: approximately 25/32, at least 80%
conditional completion per side and no ground/invalid episodes. A roll target
without the Hermite path or remembered launch line retains most of the curved
teacher rescue. It remains privileged control, **not learned fly success**: the
teacher reads current-gate geometry and motion, and its post-first-gate intervention
is external. The result supports trying to learn the simpler target; it does not
establish that the native neural circuit can yet reproduce it.

The GPU pilot `pragmatic-current-gate-roll-replay-001` has now started from the
original retained source using the documented 20-update / LR 1e-5 / 20-frame recipe,
checks at 10/20, eight native pairs, four assisted pairs and the documented seeds.
It enables `--supervision late-roll-preserve --roll-teacher current-gate`,
`--teacher-heading-mode rate-damped`, `--keep-latest-training-weights` and
`--check-replay-fit`. The completed physical-gradient experiment and both diagnostic
processes exited before this launch. No fresh goal holdout has been spent; the goal
remains unproven until unassisted performance transfers to fresh varied courses.

#### First local-target learner barely fits and does not improve native flight

The 20-update pilot finishes normally after 279.23 s, retaining the original source
at **12/32**. Update ten gives 9/32 clean (5 negative / 4 positive), first passes
26/32, prefix gates 89, ring episodes 22 and wrong-order episodes three. Update
twenty still gives **9/32** (8 negative / 1 positive), first passes 25/32, prefix
gates 81, ring episodes 21 and wrong-order episodes six. Ground and invalid counts
remain zero. Training weights were allowed to continue; no interim rollback erased
the first ten updates. This branch is not extended.

On the fixed training examples, aggregate late-roll RMSE across gates 2–5 starts at
0.023013 negative / 0.047809 positive. It changes to 0.022635 / 0.048126 at update ten
and 0.021847 / 0.048951 at twenty: a 5.07% negative-side reduction but a 2.39%
positive-side increase at the end. Individual phases are mixed, so this is not a
uniform improvement. Early-source roll preservation error rises to 0.001822 /
0.001779; the maximum fixed-window non-roll RMSE reaches 0.000694. These are motor
differences on training examples, not flight-coordinate errors or held-out scores.
The result is insufficient two-sided fitting, not evidence that a well-learned
local teacher has already failed to transfer.

Only two of twenty late windows came from assisted flight; native-first sampling
made the physically successful teacher trajectories fallback-only. The next bounded
pilot addresses that coverage limitation along with optimization strength. It does
not change the target law, network, physical plant or native success definition.

#### Stronger, explicitly mixed replay pilot

The prepared ten-update source restart uses LR **1e-4**, twenty-frame unrolls and
native checks at **5/10**. Each update assigns half its window-loss weight to early
source preservation, one quarter to native late examples and one quarter to assisted
late examples, with equal direct-loss weighting of the two sides. In this mode early
windows are entirely before the first gate; unlike the legacy transition windows,
they cannot include post-pass teacher targets.

Late sampling prefers a mirrored pair from the requested bank. If none exists, it
uses independent negative/positive examples at the same gate phase. If a native side
is absent, only that side is substituted from the assisted bank. The source and row
of each side, substitutions and actual **direct-loss** source weights are logged;
the additional pair-difference term is logged separately and is not attributed as
a controlled mirrored-geometry experiment. No bank is enlarged or resampled until
it happens to contain every desired case. Assisted means valid pre-failure windows,
not a filter for complete-course successes; full native evaluation still owns the
30-second safety requirement.

`--balanced-late-replay --freeze-replay-collections` recollects the same original
eight native / four assisted pairs once, at seeds 1690983 / 1790983, from the retained
source. These are the same course seeds, **not a claim of byte-identical reuse of
the previous trajectories**. The banks then stay fixed and are saved as a tensor-only
`source-replay.pt` artifact for reuse. Nine fixed fitting checks cover early preservation
and both requested banks at each late phase, with fresh current-weight neural prefixes.
All **556 tests pass**, including independent-course selection, missing-side fallback,
mixed-history padding and strictly pre-first-gate preservation.

Continue beyond ten updates only if late fitting moves meaningfully on both sides
(roughly 10% RMSE reduction as an initial screen), individual phases remain sensible,
and early/other-axis preservation plus native flight avoid substantial deterioration.
This is an optimization-and-data-mixture pilot, not a learning-rate-only attribution.
The retained source and fresh varied-course goal validation remain protected.

The pilot is running as `pragmatic-current-gate-mixed-replay-001`, with
`--keep-latest-training-weights` enabled alongside both collection flags. Its actual
baseline is 12/32 clean (9 negative / 3 positive), 28 first passes, 100 clean-prefix
gates, 19 ring-contact episodes, two wrong-order episodes and no ground/invalid
episodes. The native bank contains 943 recorded frames from 16 episodes; the assisted
bank contains 1,010 from eight episodes. Native gate-five active frames split
498 negative / zero positive, whereas the assisted split is 629 / 327. Consequently
the gate-five native-late slot keeps its native negative row and substitutes only
the positive row. That update's direct loss weights are 0.5 early native,
0.125 late native and 0.375 late assisted; the separately logged contrast term
couples the sides. The nine fixed fitting checks and tensor-only replay cache have
been created. These collection counts are not full-flight success results.

#### Mixed pilot completes without two-sided fitting or flight improvement

The process finishes normally after 284.63 s. Update five ties the source's 12/32
clean courses, with an 8 negative / 4 positive split, 27 first passes, 95 clean-prefix
gates, 20 ring-contact episodes and two wrong-order episodes. The experiment's local
selector saves that checkpoint on its side-balance tie-break, **not an increase in
total completion**. Update ten drops to 6/32 (5 negative / 1 positive), 23 first
passes, 78 prefix gates, 24 ring-contact episodes and five wrong-order episodes.
Both checks have zero ground/invalid episodes. The global retained source is unchanged;
neither pilot export is promoted and no fresh goal holdout is used.

Across the eight fixed late fitting windows (equal window MSE, then square root),
negative / positive roll RMSE is 0.021278 / 0.028311 at source, 0.021041 / 0.027875
at update five and 0.020071 / 0.028533 at ten. Final reductions are 5.68% negative
and **-0.78% positive**: the latter worsens. Early-roll RMSE reaches 0.001336 /
0.001986, and maximum non-roll RMSE across the nine checks reaches 0.001715. These
remain same-history motor errors, not held-out flights. This misses the declared
two-sided fitting screen while native flight deteriorates, so the identical recipe
is not extended. Explicit assisted sampling plus this higher learning rate did not
solve the fitting problem in ten updates; it does not establish a capacity limit.

A read-only structural check counts 19,286 trainable existing visual-to-roll edges
at five hops, 1,097,500 at seven, 2,491,540 at nine and 2,620,304 at twelve, always
excluding edges into the other motor pools. This large jump is a reason to measure
training capacity deliberately rather than casually widening a supposedly small
mask. The full neural graph already runs in every case; these counts concern
plasticity only. The CPU diagnostic completed separately under AIRA in about four
CPU seconds, without another GPU job.

#### Fixed-lesson fitting comparison: five-hop versus seven-hop plasticity

The next experiment deliberately separates repeated fitting from changing lesson
samples. `scripts/audit_pragmatic_fixed_replay_fit.py` restores the same nine recorded
windows from the mixed pilot's `source-replay.pt` and `report.json`, including the
gate-five cross-bank substitution. **Every optimizer update** accumulates the complete
objective: 0.5 early-source loss, 0.25 mean requested-native late loss and 0.25 mean
assisted late loss. Each of the eight late windows therefore has weight 0.0625.
The existing contrast term and motor normalization are unchanged. Both arms use the
five-hop edge count (19,286) as the anchor-loss denominator, so increasing the mask
does not silently weaken the penalty per changed synapse.
These are deliberately fitted examples, not a held-out performance test.

Two arms restart from the original retained checkpoint with fresh Adam at LR 1e-4,
twenty differentiated frames, and up to 100 updates each. The only between-arm
change is five-hop versus seven-hop trainable existing visual-to-roll paths. The
latter opens 1,097,500 edges rather than 19,286: a substantial change in optimization
freedom, not new wiring, neurons or actor inputs. Biases, membrane time constants,
signs, topology and direct inputs to the other motor pools stay fixed. Every lesson
recomputes its complete current-weight neural prefix before differentiation; stale
source neural states are not reused.

Checks every 25 updates (25/50/75/100) report the nine fitting errors, early/non-roll
preservation and the usual fully native 32-case development flights over all 30 s. There is no
development-driven rollback or automatic checkpoint promotion. An arm stops early
at a check only if aggregate late-roll RMSE falls at least 50% on **each** training
side; otherwise it reaches the 100-update cap. Nonfinite computation stops the run.
Preservation losses remain in the objective, but their degradation is recorded rather
than used to veto this diagnostic fitting experiment. All **568 tests pass** after
adding fixed-lesson identity, mixed-source history, weighting, two-sided finite-screen
and equal-per-edge anchor tests.

If five hops fit, repeated coherent fitting was an immediate missing ingredient;
an anatomical capacity limit is not established. If only seven hops fit, that supports
broader plasticity under this optimizer. If neither fits, objective conflict and
truncated recurrent credit remain possibilities; do not infer absent visual information.
Twenty neural ticks allow gradients through a seven-edge feedforward route, but
arbitrary long-memory learning is not tested by this comparison.

Unused cached whole pairs cannot supply a balanced late transfer check: the two
unused assisted positive rows both fail gate one and contain zero later active frames;
native positive gate-five coverage is absent bank-wide. If fitting reaches the screen,
collect one predeclared new source-native/source-roll-assisted bank, without resampling
until coverage looks favourable, and test the nominated fit on that new sensory history.
Seek at least 20% late-roll RMSE improvement on both transfer sides before a larger
native-flight validation. The deferred implementation is recorded below; no transfer
collection has run.
Fresh full-distribution goal validation remains separate and requires a genuinely
promising autonomous controller; overfitting these lessons cannot satisfy the goal.

The first launch, `pragmatic-fixed-bank-mask-fit-001`, exited before completing its
source fitting measurement and before any optimizer update. The restored report
records still contained axis/side labels that `replay_fit_summary` supplies itself,
causing a duplicate-key exception. The loader now removes those regenerated fields,
and a regression test exercises a saved record through the actual summary wrapper.
The empty attempt directory is retained; the corrected run uses a new `-002` directory.

The corrected `pragmatic-fixed-bank-mask-fit-002` process has now completed its source
measurements and its first five-hop optimizer update. The source again scores 12/32
native clean courses. Restored fixed-window late-roll RMSE is 0.021278 / 0.028311,
consistent with the mixed pilot's original examples; early-roll errors are below
3e-7 and maximum non-roll error is 3.3e-6. This is an ordinary FP32 reconstruction
check, not a byte-exact criterion. The first complete nine-window training objective
is 0.803150 before the update. All **569 tests pass** with the report-restoration
regression included, and the fixes are pushed. The five-hop arm is active; the
seven-hop source restart follows in the same single GPU process. The next native
flight/fit check is at update 25. No improvement is claimed from startup or training loss.

#### Deferred new-course replay-transfer check

`scripts/audit_pragmatic_replay_transfer.py` is ready, with **578 tests passing** in
the full suite. It has not collected data or used the GPU. The sole active GPU job
remains the fixed-bank comparison. At five-hop update 14 its complete fixed objective
is 0.682768 versus 0.803150 initially; this is before the first fit/native check and
is not a two-sided or autonomous improvement claim.

After a saved checkpoint meets the 50% fitting reduction on each side, nominate its
fit directory, hop budget and update. The tool rejects an unfitted or nonfinite
nomination. It collects exactly eight source-native pairs at seed **2026091303** and
eight source-roll-assisted pairs at **2026091304**, once each, with the same varied
course geometry and local roll teacher. It does not resample for favourable coverage.
These seeds are allocated to diagnostic transfer histories, not future goal holdouts.
Only run this after the current GPU process exits; no parallel GPU collection is needed.

Source and candidate replay every recorded frame from zero neural state and ten
warmup frames, maintaining separate continuous neural state. Inactive or unscored
intervals mask the score, not neural updates. Labels preserve the source before the
first gate and on pitch/yaw/throttle, with current-gate roll labels afterward. All
four motor errors are reported by bank, gate phase and side; the recorded physical
trajectory is identical for both actors. This is fixed-input replay transfer, not
closed-loop candidate flight. In particular, collection eligibility ends at source
lesson failures and is not the full 30-second clean-flight criterion.

A bank/phase/side group qualifies for aggregation only with at least **20 active
frames from two distinct episodes**. Thinner or empty groups remain explicitly
reported. The available groups have equal MSE weight, with identical weights and
source-derived masks for both actors. Every late phase/side must have a qualifying
group in at least one bank, and each bank kind must have qualifying late observations
on both sides. Missing native positive gate-five data is not imputed from another
group or concealed by dropping the coverage check.

The summary reports combined, native-only and assisted-only reductions. At least
20% combined reduction on both sides, complete coverage and no native-only regression
permits expanded native validation. A combined gain with worse native-bank errors is
flagged as assisted-state-only improvement instead. No checkpoint is automatically
promoted, and passing this diagnostic cannot establish the above-50% flight goal.

#### Five-hop fixed-bank check at update 25

The first saved check reduces aggregate late-roll RMSE from 0.021278 / 0.028311 to
**0.017791 / 0.025499**, or **16.39% negative / 9.93% positive**. Almost all fixed
late windows improve; the assisted gate-five positive window worsens slightly
(0.004475 to 0.004545). Early-roll preservation RMSE is 0.001744 / 0.002221 and the
maximum non-roll RMSE is 0.001859. The full objective before update 25 is 0.611370,
versus 0.803150 before update one. These remain deliberately fitted training examples.

Native development flight regresses to **6/32 clean** (2 negative / 4 positive)
versus the source's 12/32 (9 / 3). Clean first-gate passes increase to **30/32**
(15 / 15) from 28/32 (15 / 13), but clean-prefix passes fall from 101 to 88 and
ring-contact episodes rise from 19 to 26. There are two wrong-order episodes in
both, and zero wrong-direction, ground or invalid episodes. All 30 s are scored.
The completion loss is therefore not explained by fewer first-gate passes; changed
gate-entry state versus errors in subsequent control are not causally separated.

`hops-5-update-25.pt` is an explicitly diagnostic checkpoint, not promoted. The
two-sided 50% fitting screen is not met, so the new-course replay-transfer check
remains unrun and no fresh goal holdout is used. Per the predeclared experiment,
training continues without development-driven rollback toward the next check at
50 and the cap at 100. The seven-hop arm has not yet started. This experiment asks
whether repeated fitting can reproduce the lessons, not whether partial fitting is
already safe to substitute for the retained autonomous actor.

#### Five-hop fixed-bank check at update 50

Repeated fitting continues to reduce the measured errors: late-roll RMSE is now
**0.016171 / 0.023721**, a **24.00% negative / 16.21% positive** reduction from source.
Every late-window/side error is below its source value, although some improvements
are small. This is still far short of the two-sided 50% fitting threshold. Early-roll
preservation errors rise to **0.003158 / 0.003795** and maximum non-roll error to
**0.003120**. The complete objective before update 50 is 0.516149, versus 0.803150
before update one. No fitting statistic here is held out.

Full native development remains **6/32 clean**, now **zero negative / six positive**,
versus two / four at update 25 and nine / three for the retained source. First-gate
passes are 29/32 (15 negative / 14 positive), clean-prefix gates 78, ring-contact
episodes 24 and wrong-order episodes two. Wrong-direction, ground and invalid counts
are all zero. Thus improved fixed-example fitting has not produced an overall
autonomous gain, and the loss of all negative-side completions is not hidden by the
positive-side improvement.

`hops-5-update-50.pt` remains diagnostic-only. The existing process continues to the
next interval check at 75 and the 100-update cap, unless it first meets the declared
fitting screen. The seven-hop source restart is still pending. No candidate is
promoted, no new-course replay-transfer collection is launched and no fresh goal
holdout is used. The original autonomous checkpoint remains unchanged.

#### Unified current-gate roll rule: physical feasibility and deferred matched test

The current replay labels preserve source roll before the first gate and teach the
local gate-relative roll rule afterward. That imposed supervision boundary might
make learning harder; it is a hypothesis, not an established explanation for the
fitting/flight gap. A simpler alternative is to teach the same local roll rule from
the first physical command, while preserving source pitch/yaw/throttle throughout.
This still requires visual gate identification and motion inference inside the fly.
No external phase state or privileged teacher inputs would be deployed in the actor.

Before considering new learning, the existing CPU physical teacher audit now accepts
`--roll-teacher current-gate`, applied from the first frame including post-completion
braking. Two matched preflights used **128 cases / 64 pairs at seed 1037983**, the same
full varied geometry, nominal **35 g** mass, rate-damped curved-teacher pitch/yaw/throttle,
50 Hz commands through the actual foreleg plant and 100 Hz physics, scoring all 30 s:

- Original curved roll: **128/128 clean**, 64/64 per side; maximum gate-passage radial
  offset 0.06444 m; mean completion time 15.065 s.
- Current-gate roll from the start: **128/128 clean**, 64/64 per side; maximum radial
  offset 0.10968 m; mean completion time 15.098 s.

Both pass all 640 gates with zero ring, illegal-traversal, ground or invalid episodes.
CPU runtimes were 7.19 s / 5.95 s. These are **all-axis teacher-controlled physics**,
not fly completions or evidence of learnability. The seed is a reused teacher diagnostic
bank, not a fresh goal holdout. Reports are ignored local artifacts under
`runs/gate/pragmatic-fixed-bank-mask-fit-002/`, named
`curved-teacher-cpu-preflight.json` and `current-gate-from-start-cpu-preflight.json`.

The next diagnostic is prepared in `audit_pragmatic_current_gate_roll.py` with
`--compare-starts`. **Do not launch it while the fixed-bank fitting process is live.**
Once that process exits, it compares the retained source in three conditions on the
same 32 development cases at seed 1110983: native, local roll after gate one, and local
roll from the first physical command. All share the same initial aircraft/foreleg
state, ten neural-only warmup frames, continuous native neural updates, native
pitch/yaw/throttle and full 30-second scoring including the completion tail.

The conservative predeclared from-start screen is at least **26/32 clean**, at least
**12/16 per side**, at least **28/32 clean first-gate passes**, zero ground/invalid
episodes and at most **one paired clean completion lost** versus the matched after-first
condition. Paired losses/gains and net count change are reported separately: a
failure solely on paired losses would reject this conservative screen, not prove the
unified teacher ineffective. The old `all_gate_success_episode_indices` reports raw
passes and is deliberately retained; a new `clean_course_success_episode_indices`
reports full-episode clean flights for this comparison. Existing clean success scores
already enforce the full failure rule and are not relaxed.

Passing assistance would justify a bounded unified-label training trial, not count
as autonomous success. The active five/seven-hop fitting experiment and its labels
remain unchanged. No matched GPU start comparison or unified-label learning has run.
All **589 tests pass**, including first-command intervention timing, unchanged native
PYT, neural-only warmup, continuous recurrence, completion-tail intervention, matched
initial states, and the distinction between raw passages and clean paired successes.
Astra's read-only review found no blocking design or implementation issue.

#### Five-hop fixed-bank check at update 75

Late-roll fitting RMSE falls to **0.014426 / 0.021964**, reductions of **32.20% /
22.42%** from source, still below the two-sided 50% fitting screen. Early-roll
preservation RMSE rises to 0.003978 / 0.004682 and maximum non-roll RMSE to 0.003617.
The complete objective before update 75 is 0.427314.

Native development deteriorates to **5/32 clean** (one negative / four positive),
versus source 12/32. Clean first-gate passes fall to **22/32** (11 / 11); clean-prefix
gate passes total **53**, ring-contact episodes **24**, wrong-order episodes **three**,
and wrong-direction, ground and invalid episodes **zero**. All 30 s are scored.
Unlike the previous checks, this checkpoint also loses substantial first-gate capture.
Improved fixed-input imitation remains insufficient for autonomous improvement.

`hops-5-update-75.pt` stays diagnostic-only. The same live process continues toward
the 100-update cap and the independent seven-hop source restart. No promotion,
new-course replay-transfer collection, matched start comparison or fresh goal
holdout is triggered by this result. The retained autonomous source is unchanged.

#### Cached first-gate label-boundary check

A CPU-only check recomputed the local roll teacher on the existing native seed
1690983 and roll-assisted seed 1790983 physical histories, without collecting new
data or replaying a brain. Before the first gate, local-versus-source roll-target
RMSE is **0.02741 / 0.03136** on the native bank (2,359 / 2,467 active frames), and
**0.02222 / 0.03144** on the assisted bank (1,325 / 1,311 frames). Units are motor
antagonist differences, not stick angles. The proposed unified rule therefore
changes supervision meaningfully before gate one; it is not just a different label.

For the 15 native and six assisted first-gate transitions with both adjacent frames
eligible, the RMS target jump changes from **0.02369 to 0.01940** on native histories
but **0.01159 to 0.01541** on assisted histories when applying local roll throughout.
Thus removing the policy boundary does **not** universally smooth gate-crossing
commands. Gate identity still changes at the crossing. This check supports testing
one consistent supervision rule, not claiming that a discontinuity caused the current
learning failures. Recomputed late labels match the cached local teacher within
2.24e-8 in this CPU check. The small report is retained locally as
`pragmatic-fixed-bank-mask-fit-002/first-gate-label-boundary-cpu.json`; no training,
checkpoint selection or goal evaluation is changed.

Astra's broader design review keeps the order: finish the mask comparison, test the
from-start intervention, then try unified motor labels if assistance passes. The
previous geometry-auxiliary idea is retained as a **conditional fallback**, not a
new running branch: use existing descending/VNC premotor neurons within two edges
upstream of the roll motors; fit and then freeze a training-only linear bearing
head (`sin(theta)`, `cos(theta)`), and compare unified motor training alone against
the same training plus a small bearing loss (e.g. 0.1 times source-normalized MSE).
Keep native motor supervision active and deploy neither the head nor its outputs.
An auxiliary would be useful only if better geometry coding also improves both
sides' native roll predictions on unused whole-course histories and then actual
unassisted flights. Better probe accuracy alone is not a flight solution. This
fallback is not implemented or scheduled ahead of the simpler unified-rule test.

Scheduling revision: the source-only matched start comparison now runs alongside
the fixed-bank fitting job, superseding the earlier wait-until-exit instruction.
Before launch, the RTX 5080 used **2,497 / 16,303 MiB** with 19% utilization at the
sampled instant. Separate processes/models and ample VRAM permit this short no-grad
diagnostic without changing training parameters, simulated timing or score rules;
wall-clock timings may be slower and are not a controlled performance benchmark.
No second training branch is started. The comparison uses the original source and
the predeclared 32 cases/screen, writing `current-gate-start-matched-gpu.json` in the
same ignored run directory. Both job handles are monitored separately; no inference
output is substituted into the fitting process.

#### Matched GPU start comparison: promising assistance, failed preservation screen

The three-condition comparison completed in **145.03 s**, with the fit process still
running independently. Observed combined GPU memory was 4,101 / 16,303 MiB. The matched
source checkpoint, 32 development cases (seed 1110983), inputs, physics and full 30 s
failure rules were unchanged:

| Roll condition | Clean courses | Negative / positive | Clean first gate | Clean-prefix gates |
| --- | ---: | ---: | ---: | ---: |
| All native | 12/32 | 9/16 / 3/16 | 28/32 | 101 |
| Local roll after first | 26/32 | 13/16 / 13/16 | 28/32 | 137 |
| Local roll from start | 27/32 | 11/16 / 16/16 | 32/32 | 152 |

All conditions have zero ground, invalid and wrong-direction episodes. Ring-contact
episodes are 19 / 5 / 5 and wrong-order episodes 2 / 1 / 0. The from-start condition
loses three previously clean episodes (indices 6, 18, 22; all negative), while gaining
four (9, 10, 13, 23). Thus the **predeclared conservative screen fails** both the
12-negative requirement and the at-most-one paired-loss requirement. Its saved flag
remains false; the positive result is not retroactively labelled a pass.

With from-start roll, all 32 episodes pass gates 1–3; three stop at gate four and two
at gate five, all on the negative side. Mean absolute lateral crossing offsets stay
around 0.020–0.033 m, while mean absolute vertical offsets grow from 0.094 m at gate
one to 0.249 m at gate five. These aggregate observations suggest a remaining vertical
tracking issue but do not establish each failed episode's cause. No throttle change
or roll-teacher retuning is justified solely by that aggregate. All assisted outcomes
remain privileged-teacher results, **not learned fly completion or fresh holdout data**.

Astra's design review recommends an **explicitly exploratory** unified-label pilot
despite the preservation-screen failure: 27/32 overall and 32/32 first capture provide
a credible roll-learning target, without establishing preservation of the older rescue.
This is a learning experiment, not adoption of the assisted controller or a weakened
goal test. Another teacher-certification round is not required before testing learning.

`audit_pragmatic_fixed_replay_fit.py --roll-labels unified` is prepared, not launched.
After the original mask comparison finishes, choose its better-fitting mask by the
best saved worst-side late-roll reduction (prefer five hops on a tie) and restart the
original source. Reuse the same nine physical histories/windows, optimizer, learning
rate 1e-4, 0.5 early / eight times 0.0625 late weighting, 20-frame gradients, 25-update
checks and 100-update cap. Replace only pre-first roll labels with the current-gate
teacher; retain source PYT and all existing later roll targets. No new collection or
neural coding auxiliary is combined with this first trial.

Early and late teacher errors are reported separately. For unified labels, an early
fitting stop requires at least 50% improvement on both sides in **both** early and
late errors; otherwise run to the cap. This is not a native promotion criterion.
The legacy late-preservation mode is unchanged, and the old-label replay-transfer
tool rejects unified checkpoints rather than silently testing against mismatched
targets. A source-beating autonomous result is still required before larger native
validation; better fitting alone does not authorize promotion.

The actual-cache CPU preflight changes the early supervised roll by RMS **0.012212 /
0.012643** and changes no supervised later roll or PYT label. Every physical-history,
gate, role and window-start object is reused. All **592 tests pass**, including the
new relabelling and transfer-label guards, and Astra's read-only code review found no
blocking issue. The original fitting process retains its already-loaded code and
labels; the opt-in mode does not alter that running experiment.

#### Five-hop arm complete; seven-hop source restart is live

The five-hop arm reaches its **100-update cap without meeting the fitting screen**.
Late-roll RMSE ends at **0.013863 / 0.020161**, reductions of **34.85% / 28.79%**.
Early source-preservation error is 0.004774 / 0.005248 and maximum non-roll RMSE is
0.003750. The complete objective before update 100 is 0.364817, versus 0.803150 at
source. These nine training examples remain incompletely fitted under this setup;
that alone does not establish a hard anatomical capacity limit.

Native update-100 development is **7/32 clean** (one negative / six positive), with
**20/32 clean first gates** (seven / thirteen), 57 clean-prefix gates, 25 ring-contact
episodes and two wrong-order episodes. Wrong-direction, ground and invalid episodes
remain zero. The source's 12/32 completion and 28/32 first capture are not beaten by
any of this arm's four checks. Its checkpoint remains diagnostic-only and no transfer
or goal holdout is triggered.

The same process has independently reloaded source weights, removed the five-hop
gradient hook and created fresh Adam state for **seven hops: 1,097,500 selected edges
and 92,640 nodes**. Its first objective is 0.803149, back at the source scale rather
than continued from the five-hop endpoint. It uses the same fixed examples, weighting,
anchor denominator and 25/50/75/100 checks. The first seven-hop optimizer update is
verified complete; the unified-label trial remains prepared but unlaunched pending
the comparison's results.

#### Faster fixed-window training without a new learning objective

A bounded source-only GPU probe evaluated the prepared unified-label objective and
its gradients with no optimizer updates. The early lesson stays separate at weight
0.5. Consecutive late lessons can be batched because all have weight 0.0625, twenty
supervised frames, and adjacent negative/positive rows. For a group of G lessons,
the batched mean loss multiplied by `G * 0.0625` preserves their summed contribution,
including the within-pair contrast loss. Each row retains its own complete neural
prefix and waits without state advancement after reaching its own window start.

The warmup-completed, CUDA-synchronized probe ran alongside the existing fit process:

| Late lessons per batch | Groups per update | Update-equivalent time | Peak allocated GPU MiB | Gradient relative L2 difference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9 | 43.58 s | 1,362 | reference |
| 2 | 5 | 28.78 s | 2,264 | 2.71e-6 |
| 4 | 3 | 16.15 s | 4,176 | 4.89e-5 |

The source objective is 0.881805980 individually and 0.881808646 with four late
lessons per batch (about 3e-6 relative difference). Gradient norms are 3.7779045 and
3.7779324. The roughly **2.7x observed speedup** is a practical shared-GPU measurement,
not an isolated hardware benchmark or promise of identical timings in later runs.
Peak reserved memory in the four-lesson probe is 6,762 MiB; these figures belong to
the probe process, not combined whole-GPU use. Its FP32 scalar cosine calculation
wandered slightly above one and is not used as evidence; loss, norm and relative-L2
comparisons support the batching decision. No trained checkpoint was produced.

The existing fitter now accepts `--late-windows-per-group 1|2|4`, defaulting to the
original one. The planned unified-label pilot will use four. All groups accumulate
before one anchor contribution, one gradient-clipping operation and one Adam update.
Original individual-window fitting measurements remain unchanged at checks. Batched
reports list the actual group indices/weights and group losses rather than inventing
per-window training losses from a group mean. This is numerically equivalent FP32
batching, not a byte-identical execution claim.

All **598 tests pass**, including retained pair ordering, histories, weights and
summed recurrent loss/gradient comparisons. Astra's code review found no blocking
issue. The temporary probe and raw results are under
`/home/mark/tmp/fly-over50-20260912/replay_batch_probe.py` and `.json`. The live original
mask comparison continues with its already-loaded one-window implementation; no
optimizer state, target or runtime configuration in that process is changed.

#### Seven-hop update 25; unified-label pilot launched from source

The broader mask's first check reduces late-roll RMSE to **0.011885 / 0.013786**,
or **44.14% / 51.31%** below source. Early preservation errors are 0.004475 / 0.005562,
maximum non-roll RMSE is 0.003701, and the pre-update objective is 0.189393. This is
substantially better fixed-window fitting than any five-hop check, but only one side
has met the two-sided 50% threshold.

Native flight is **5/32 clean** (zero negative / five positive), with **13/32 clean
first gates** (two / eleven), 38 clean-prefix gates, 19 ring-contact episodes and one
wrong-order episode. Wrong-direction, ground and invalid counts are zero. Better
fitting has again failed to preserve the source's autonomous control, especially
negative-side first-gate capture. `hops-7-update-25.pt` is not promoted and cannot
yet nominate the old-label transfer audit.

The mask choice for the isolated unified-label pilot is now settled under the
already stated rule: seven hops' worst-side fitting improvement is **44.14%**, greater
than the completed five-hop arm's best **28.79%**. Future seven-hop checks cannot
erase its best saved result. Consequently, waiting for the rest of that arm is no
longer necessary to choose the pilot mask. This revises scheduling, not the selection
rule or the goal criteria.

The new `pragmatic-unified-current-gate-fixed-fit-001` process has been launched with:

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/audit_pragmatic_fixed_replay_fit.py \
  --checkpoint runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt \
  --cache runs/gate/pragmatic-current-gate-mixed-replay-001/source-replay.pt \
  --fitting-report runs/gate/pragmatic-current-gate-mixed-replay-001/report.json \
  --output-dir runs/gate/pragmatic-unified-current-gate-fixed-fit-001 \
  --hop-budgets 7 --updates 100 --interval 25 --learning-rate 1e-4 \
  --roll-labels unified --late-windows-per-group 4
```

It restarts the **original source**, not the poorly flying update-25 checkpoint.
The same nine recorded windows, their weights and source PYT targets are retained;
only early roll supervision changes. The tested grouping reduces dispatch overhead
without changing the summed learning objective. The old fit process continues its
seven-hop arm independently with unchanged labels, optimizer and runtime settings.
Before the new launch the GPU used 2,497 / 16,303 MiB; the measured batching probe
supports running both within available memory. Each has its own job handle and
output directory. No new course collection, teacher assistance in native evaluation,
fresh goal holdout or automatic controller promotion is introduced.

The unified pilot has completed its source measurements and first optimizer update.
Source native development is **12/32 clean**. Late source fitting RMSE is
0.021278 / 0.028311 and early local-teacher RMSE is **0.012212 / 0.012643**, consistent
with the prior CPU label-change check; maximum non-roll error is 3.3e-6. The first
grouped objective is **0.881808** and gradient norm 3.77836. Three group losses use
weights 0.5 / 0.25 / 0.25, representing all original nine lessons. This confirms
startup and the intended training target, not learned improvement. The first
unified fit/native check is at update 25. Both process handles remain live.

#### Fixed-window endpoints: improved fitting, failed autonomous control

The original mask comparison has finished normally. The seven-hop arm stops at
update 50 because both late-roll fitting reductions exceed 50%: **53.21% / 72.62%**,
with RMSE **0.009955 / 0.007753**. Early preservation RMSE is 0.001094 / 0.006633 and
maximum non-roll RMSE is 0.003003. This establishes useful fitting capacity, not
control: autonomous development is **0/32 clean**, four clean first gates (zero
negative / four positive), eleven clean-prefix gates, twelve ring-contact episodes
and five wrong-order episodes. Ground, invalid and wrong-direction counts are zero.
The source remains selected; this endpoint is diagnostic only.

The isolated unified-label pilot is also failing in flight:

| Update | Late-roll reduction, negative / positive | Early-roll reduction, negative / positive | Clean courses | Clean first gates | Ground / invalid episodes |
| --- | --- | --- | ---: | ---: | --- |
| 25 | 39.52% / 46.05% | -9.41% / 37.88% | 2/32 (0 / 2) | 15/32 (10 / 5) | 0 / 0 |
| 50 | 51.30% / 70.50% | 2.22% / 70.90% | 0/32 | 0/32 | 4 / 4 |

At update 25 there are 27 clean-prefix gates, 24 ring-contact episodes and five
wrong-order episodes. At update 50 those counts are zero, three and zero. Neither
check has wrong-direction events. Update-50 late RMSE is 0.010361 / 0.008351;
early local-teacher RMSE is 0.011941 / 0.003679 and maximum non-roll RMSE is 0.005891.
These early errors are **teacher-fitting** errors, not preservation errors. The
pilot remains bounded at 100 updates and has not met its two-sided early threshold.
No checkpoint is promoted. Fitting improvements cannot override flight failures.

#### New-course fixed-input transfer: some learning, insufficient generalization

The completed old-label seven-hop update-50 checkpoint was eligible for the already
prepared transfer audit. `pragmatic-fixed-replay-transfer-001` collected its two
reserved seeds once: 2026091303 source-native and 2026091304 source-roll-assisted,
eight mirrored pairs each. These seeds are now used diagnostic courses, not future
goal holdouts. Both actors replay the same entire physical histories with independent
current neural states; this is not a candidate-driven autonomous flight test.

All phase/side/source coverage requirements were met. Late-roll RMSE reductions are:

| Input-history source | Negative | Positive |
| --- | ---: | ---: |
| Source-native | 19.01% | 25.01% |
| Source-roll-assisted | -11.35% | 36.86% |
| Equal-group combined | 11.17% | 27.42% |

The two-sided 20% combined screen fails. No expanded native validation is nominated.
On native pre-first-gate histories, source preservation error is approximately zero
while the candidate's roll error is **0.017537 / 0.015719**. Corresponding assisted
history errors are **0.018191 / 0.017042**. This demonstrates substantial early-policy
drift outside the one fitted preservation window. Late learning transfers somewhat,
but neither the fixed-input screen nor actual autonomous performance is adequate.

#### The fixed early lesson covers very little of the approach

A CPU-only inspection of the original recorded history and local teacher found:

| Course side | Sole supervised early interval | First-gate pass | Fraction of pre-first active frames | Mean local roll target |
| --- | --- | --- | --- | ---: |
| Negative | 1.34–1.74 s | 6.04 s | 6.62% | -0.02970 |
| Positive | 4.24–4.64 s | 6.12 s | 6.54% | -0.02147 |

These are native seed 1690983 rows 0/1, starts 67/212, twenty supervised frames each.
Both windows have negative mean roll targets despite opposite course sides; they
sample different stages of the manoeuvre. There is no direct launch supervision,
and fourteen other native episodes are absent from the fixed early lesson. In the
first half-second the local roll means are approximately +0.00018 / +0.01896, whereas
source roll means are +0.04915 / -0.02076. Roll output is a rate-control motor drive,
not an angle: these signs alone do not diagnose a wrong teacher.

This is evidence of narrow coverage, not proof that coverage explains every failure.
In particular, the unified pilot also struggles to fit its negative-side early
window. Nevertheless, making a new all-flight roll policy from one 400 ms example
per side is not an adequate test of the overall approach.

#### Next bounded learning trial: refreshed whole-approach roll imitation

Astra recommends broadening learner-state coverage before adding geometry auxiliaries
or switching to PPO. The new `train_pragmatic_course_coverage.py` reuses the existing
collector, recurrent replay, renderer, physics and native evaluator. It does not
introduce a deployed controller, mode machine, extra observation or memory buffer.

The pilot restarts the original source, with **three rounds of ten updates**:

- Each round collects eight fresh learner-native pairs and four fresh roll-assisted
  pairs, with learner weights fixed throughout collection. Assisted flight uses
  local current-gate roll from the first command and frozen-source PYT. Native
  collection uses all four learner outputs; its labels use local roll and the
  frozen source evaluated on those same actual sensory histories.
- Every update has four equal-weight paired windows: native early, assisted early,
  native late and assisted late. Early sampling covers launch (start zero) and all
  three thirds of eligible approach starts. The final third is
  called **late approach**, not assumed to precede a successful crossing. Late
  phases rotate gates two through five. Missing native sides may use assisted
  examples, explicitly recorded as substitutions; no courses are redrawn to obtain
  successes. Both signs and all phases are reported without relabeling assistance
  as native exposure.
- All four pairs are batched with pair adjacency preserved. Each has its own full
  current-weight neural prefix and twenty supervised frames. No cached neural
  states are reused. The same seven-hop mask, fixed signs/topology, fixed biases and
  time constants, 1e-4 learning rate, single clipping/Adam step and five-hop-count
  anchor denominator remain. The actor still advances at 50 Hz, physics at 100 Hz.
- Collections refresh under the latest learner each round, without restoring a
  previous checkpoint between rounds. Updates four and nine also sample a retained
  older collection when available, preserving some earlier exposures. Before/after
  fitting is measured on identical per-round diagnostic windows, separated by actual
  source, phase, side and axis. These are training examples, not goal holdouts.
- Full 30-second, unassisted 32-case development checks happen every round. Retain
  the checkpoint with most clean completions, then best worst-side count, then
  prefix progress. Do not reject a better overall/better-balanced policy merely
  because one side declines. All saved checkpoints remain diagnostic pending fresh
  validation; the original source is retained if none improves.

Training seeds are 2026091310/11, 2026091330/31 and 2026091350/51. Development remains
the reused seed 1110983. A practical reason to nominate a checkpoint for a fresh
128-case balanced evaluation is at least four additional clean development
completions, with zero ground/invalid episodes; this is not the goal itself. The
goal still needs **more than 64/128 clean autonomous completions on fresh varied
courses**, with both sides represented and no teacher assistance. If refreshed
native-state fitting improves across rounds without flight improvement, do not
extend imitation indefinitely: outcome-based recurrent PPO is the next design
candidate, not an already implemented solution.

All **616 tests pass**, including from-start takeover versus native action isolation,
unchanged source PYT, stage sampling, pair-preserving diagnostic batching, and a
mocked multi-round run verifying fresh collection under latest weights, frozen
reference weights, retained older exposures and native-only checkpoint selection.
These tests validate the training plumbing, not successful flight. At this entry
the coverage pilot is implemented but has not yet launched.

Astra's pre-launch review caught two sampling gaps, both corrected: selecting only
time zero for launch left most of the first approach third unsupervised, and older
bank updates four/eight aliased the four-phase cycle, excluding some fresh later-gate
lessons. Explicit first-third sampling and older updates four/nine now retain all
early stages and all later phases in every ten-update fresh round. Regression tests
check the actual schedule and full eligible-start coverage. Shorter custom rounds
also avoid replacing the only fresh occurrence of a phase.

#### Coverage pilot launched; final fixed-window run still bounded

The reviewed trainer is committed as `ccba208` and pushed to master. The new
`pragmatic-whole-approach-coverage-001` job is live with this command:

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/train_pragmatic_course_coverage.py \
  --checkpoint runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt \
  --output-dir runs/gate/pragmatic-whole-approach-coverage-001 \
  --rounds 3 --updates-per-round 10 --learning-rate 1e-4
```

Its full native development baseline reproduces **12/32 clean**. The first fresh
native collection, seed 2026091310, records sixteen episodes and has active-frame
counts by gate/side of [2631,2646], [1170,959], [948,650], [445,658], [0,467]. Thus
there is no negative-side fifth-gate training exposure in this bank; the sampler
must use explicitly labeled assistance there. Fifteen episodes end training
eligibility at a failure/missed crossing. These collection counts are not full-tail
autonomous evaluation results, and no replacement courses are sampled.

The independent unified fixed-window run is still approaching its original
100-update limit. Its update-75 late reductions are **65.80% / 72.07%**, early
reductions **41.15% / 68.68%**, maximum non-roll RMSE 0.006220. Native development
is again **0/32 clean**, with no first-gate passes and **22/32 ground/invalid
episodes**. Ring, wrong-order and wrong-direction counts are zero. This remains a
failed controller despite improved fitting; the new coverage pilot starts from
the source, not this checkpoint. Both jobs have separate weights, optimizer state
and output directories. Verified simultaneous GPU usage was 7,117 / 16,303 MiB
during the coverage baseline, leaving headroom for the measured eight-row replay
batch. No claim is made about isolated benchmark speed under shared GPU load.

#### Coverage round one: 1/32 clean, source retained

The first ten optimizer updates completed with the intended stage rotation and
explicit negative-side fifth-gate substitution. The eight-episode assisted bank,
seed 2026091311, has active-frame gate/side counts [1256,1302], [673,720], [649,675],
[557,569], [423,610]; one episode ends training eligibility at a failure/miss.
Again, these are collection statistics, not full-tail autonomous successes.

The full native update-10 check is **1/32 clean** (one negative / zero positive),
with one clean first-gate pass, five clean-prefix gates and **18 ring-contact
episodes**. Ground, invalid, wrong-order and wrong-direction counts are zero.
The source's 12/32 remains selected. `update-10.pt` is diagnostic, not promoted.

On the identical per-round fitting probes, roll RMSE changes are:

| Actual history source / phase | Negative before → after | Positive before → after |
| --- | --- | --- |
| Native early | 0.019976 → 0.021889 | 0.030983 → 0.024503 |
| Native late | 0.027033 → 0.026253 | 0.023519 → 0.031499 |
| Assisted early | 0.019166 → 0.016986 | 0.026506 → 0.020762 |
| Assisted late | 0.020091 → 0.009956 | 0.011917 → 0.022221 |

These are equal-window RMS aggregates by **actual** source; the negative late
groups contain three native and five assisted windows because the missing native
fifth-gate probe is counted as assisted. Other groups contain four windows per
side. Non-roll source agreement starts near numerical zero, as expected. After
training, aggregate non-roll RMSE across these groups ranges approximately
0.00023–0.00208. Early negative fitting and positive late fitting have worsened:
this is not evidence of uniformly successful imitation, much less flight mastery.

The bounded pilot continues to its second round without restoring old weights.
Its fresh learner-native seed 2026091330 has gate/side active-frame counts
[2230,2928], [109,0], [167,0], [178,0], [209,0]. All sixteen episodes eventually
lose collection eligibility, and there are no positive-side later-gate histories.
Later positive lessons must therefore come from explicitly marked assistance or
retained older banks. The refresh is exposing the changed learner's failure
distribution; it does not justify claiming progress in autonomous completions.

#### Unified fixed-window trial finished at its cap

The independent unified trial has exited normally after 100 updates (2,013.68 s
total). Final late-roll RMSE is **0.006580 / 0.007410**, reductions **69.08% / 73.83%**.
Early local-roll RMSE is **0.004680 / 0.006458**, reductions **61.68% / 48.92%**;
maximum non-roll RMSE is 0.004511. It finishes at the update cap, not the two-sided
early/late fitting threshold. Being close to that fitting threshold is not a
reason to extend this unsuccessful branch.

The final native check is **0/32 clean**, zero first-gate/prefix passes and **29/32
ground/invalid episodes**, with no ring or illegal-traversal events. No candidate
is promoted and no goal holdout is used. The refreshed coverage pilot is now the
only live training job; its twenty optimizer updates have completed and its second
full-flight development check is pending. The source remains selected.

#### Refreshed coverage pilot finished: source still wins

All three rounds completed normally in **600.38 s**. The native development
trajectory is not a successful learning result:

| Check | Clean courses, negative / positive | Clean first gates, negative / positive | Clean-prefix gates | Ring-contact episodes | Ground / invalid |
| --- | --- | --- | ---: | ---: | --- |
| Source | 12/32 (9 / 3) | 28/32 (15 / 13) | 101 | 19 | 0 / 0 |
| Update 10 | 1/32 (1 / 0) | 1/32 (1 / 0) | 5 | 18 | 0 / 0 |
| Update 20 | 3/32 (0 / 3) | 8/32 (0 / 8) | 30 | 25 | 0 / 0 |
| Update 30 | 2/32 (2 / 0) | 24/32 (16 / 8) | 50 | 27 | 0 / 0 |

Update 20 has two wrong-order episodes; update 30 has none. Neither has
wrong-direction events. First-gate control partly recovers by update 30, but later
flight does not. The selected update remains **zero**, the original source. No
fresh goal validation is warranted by these candidates.

The later rounds also do not establish uniformly improved fitting on refreshed
native histories. In round two, native early roll RMSE improves by 40.46% / 7.81%,
but the available negative native-late error approximately doubles; positive
native-late examples are absent. In round three, native early errors improve by
20.99% / 26.98%, but available positive native-late error worsens by 8.49%; negative
native-late examples are absent. Assisted late improvements remain asymmetric:
round two -27.61% / +54.28%, round three +23.26% / -20.21%. These comparisons use
the identical before/after per-round examples and actual source labels. They do
not compare different banks as if they were the same test.

The next experiment should target whole-flight outcomes rather than extend this
same imitation pilot. Recurrent PPO remains an option, but the repository's legacy
PPO trainer targets an older assisted single-gate actor and is not a drop-in for
the full five-gate brain. A lower-cost alternative is the existing native motor
parameter search: its previous six-generation trial started from a 4/32 source,
not the current 12/32 source. Searching fixed native weights provides temporally
coherent exploration through the real foreleg dynamics, without external noise
state in the deployed actor. Any selected candidate still needs standalone,
full-duration native development and then fresh varied-course validation.

#### Bounded outcome-based search from the stronger source

The next pilot uses the existing 24-coordinate native motor search, not a new
controller or an extension of the failed replay fit. It starts from
`pragmatic-phase-balanced-replay-001/best-controller.pt`. All eight opposing motor
pools may adjust intrinsic biases and excitatory/inhibitory incoming gains, so
roll and PYT can co-adapt to the actual flight outcome. The native topology,
transmitter signs, foreleg plant and deployed sensory interface remain unchanged.
This deliberately differs from the roll-only seven-hop imitation mask; it is a
small whole-flight parameter search, not a controlled comparison of the masks.

Budget: **six generations, four antithetic directions (eight candidates) each**, two
candidates per GPU batch, eight varied mirrored training pairs per generation,
full 30-second flights. Bias perturbation scale is 0.0025, incoming log-gain scale
0.05, update factor 0.5 and sigma decay 0.98, retaining the existing search defaults.
Parameter vectors are fixed throughout each flight and compiled into ordinary
checkpoint weights for standalone development checks.

An opt-in `--course-refresh-generations 1` gives six distinct training seeds,
2026091370 through 2026091375; the legacy default remains two generations per bank.
Source and candidates use the same complete gate distribution and scoring. Every
second generation checks its updated search centre and generation winner on the
reused 32-case development seed 1110983. Clean completions remain the primary
selection criterion. The source stays retained unless a standalone candidate
improves, and a development improvement alone is not a goal result. The pilot also
has a 20-minute between-generation time cap, not an automatic extension condition.

The search now refuses existing output directories and overlapping training/
development seeds. **617 tests pass**, including the refresh schedule and existing
batched-versus-compiled native recurrence and per-candidate clean-course scoring
tests. No new model download, privileged actor input, teacher assistance or
deployed exploration memory is introduced. At this entry the pilot is prepared;
its flight evidence will be recorded separately after launch.

#### Pre-search ground-penalty correction and revised launch

The first search process (`pragmatic-phase-balanced-motor-es-001`) reproduced the
12/32 source baseline, then was intentionally stopped by SIGINT during its first
candidate batch. No generation or candidate result was completed or recorded.
Its files are retained; nothing is overwritten or promoted. This was a deliberate
correction, not a timeout interpreted as a stopped job.

Astra agrees with searching all 24 coordinates from this stronger source before
building a new PPO implementation, but identified a missing training penalty:
the existing search fitness charged ground/invalid events only the generic -2
failure term. Clean-course scoring already disqualified such flights, but that
was not a strong explicit search penalty.

The revised search now ranks candidates and chooses its training winner by:

`course_race_fitness - 25 * ground_or_invalid_rate`

The full and compact evaluators report the same per-candidate latched union of
ground contact or invalid state. A ground touch that also sets invalid is charged
once by this **additional** cost, not twice; the ordinary -2 failure term remains.
Even an unsafe flight with five recorded passes cannot make up its -25 cost from
gate progress. Candidates with any ground or invalid episode in standalone
development cannot replace the source. Ring/order events retain the existing
failure handling; no new per-component monotonicity constraints are introduced.
Full-duration clean-course success definitions are unchanged.

All **619 tests pass**, including union accounting, strong unsafe penalties and
separation of batched candidates. Astra's review found no blocking issue. Before
any completed search generation, the planned direction budget was increased to
**eight antithetic directions (sixteen candidates) per generation**, retaining six
generations, eight mirrored training pairs and two candidates per GPU batch. The
between-generation time cap is correspondingly **40 minutes**. It may overrun that
limit by the duration of the already-started generation; it is not a hard watchdog.
The per-generation course refresh remains one and the same training seed range is
retained, explicitly including the partly sampled seed from the interrupted run.

The corrected restart uses a separate output directory:

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/search_pragmatic_gate_course_es.py \
  --checkpoint runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt \
  --output-dir runs/gate/pragmatic-phase-balanced-motor-es-002 \
  --generations 6 --directions 8 --candidate-batch 2 --training-pairs 8 \
  --development-pairs 16 --development-interval 2 --course-refresh-generations 1 \
  --seed 2026091370 --development-seed 1110983 --maximum-minutes 40 \
  --ground-invalid-cost 25
```

If this bounded search produces no useful native improvement, the next design
candidate remains recurrent PPO. Astra notes an important constraint: the seven-hop
roll mask also changes native PYT through shared recurrence. A roll-only policy
likelihood would therefore omit other executed commands that depend on the updated
weights. A proper PPO pilot must account for all four executed actions, with
foreleg-compatible exploration and a training-only privileged critic. Neither an
external controller nor an exploration-memory process belongs in the deployed actor.

#### Corrected outcome search is live

The corrected process is running in `pragmatic-phase-balanced-motor-es-002`.
Its unchanged-source development check is **11/32 clean**, 28/32 first gates and
100 clean-prefix gates, with zero ground/invalid episodes. This differs by one
clean flight and one prefix gate from the earlier 12/32 source runs. Record this
variation rather than calling a one-case change learned progress; no FP64 or
byte-exact reproducibility detour is introduced. A nominee should clearly exceed
the historical source range before fresh matched evaluation.

The first four generation-one candidates have respectively 0, 0, 1 and 0 clean
completions out of sixteen training cases. Candidate two has eight ground/invalid
episodes. Its unpenalized mean race fitness is -0.170741 and its search fitness
is **-12.670741**, confirming the additional `25 * 8/16` penalty in the real batched
flight path. The other three have no ground/invalid episodes. These are exploratory
training outcomes, not standalone development or fresh goal-validation results.

The process is verified live, using 2,288 / 16,303 MiB of GPU memory at the sampled
instant. First standalone candidate development checks are scheduled after
generation two. No controller has been promoted; the retained-source decision and
the full varied-course >50% goal remain unchanged.

#### Generation one complete; prepare a bounded PPO fallback without changing the search

The first generation's updated centre scores **4/16 clean training flights**,
14/16 first-gate passes, 47 clean-prefix gates and zero ground/invalid episodes on
seed 2026091370. Its fitness is 2.985954. The best sampled candidate has only 2/16
clean training completions. These counts do not establish improvement over source
on that training bank, because this generation did not run a matched source
control there; standalone development remains scheduled after generation two.
The six-generation outcome search continues with unchanged settings.

While it runs, a small CPU-only preparation module,
`scripts/pragmatic_correlated_exploration.py`, checks the exploration mathematics
and the existing foreleg plant's response. It is **not a PPO trainer**, and it is
not imported into the deployed controller or the active search.

For the proposed fallback, use fixed four-axis latent noise scales and an AR(1)
residual. Let `u` be the saved pre-tanh latent command and `mu` the current-weight
`atanh(native_motor)` output. At a true episode start the conditional mean is `mu`
and standard deviation is `sigma`. Subsequently they are
`mu_t + rho * (u_(t-1) - mu_(t-1))` and `sigma * sqrt(1-rho^2)`. This accounts for
the past command instead of treating correlated samples as independent Gaussian
draws. Both current and previous means must receive gradients. This formulation
follows the history-conditioned autoregressive-policy construction; our first
pilot would fix the noise parameters rather than learn them.
[Autoregressive Policies, equations 9–11](https://arxiv.org/abs/1903.11524)

Our implementation stores and scores the latent command, sends `tanh(u)` to the
forelegs, and provides a stable tanh-Jacobian calculation. For old/new policy
ratios that identical Jacobian cancels, so integration should subtract latent log
densities directly. Sum the four axis log densities before making one PPO ratio
per command; do not clip axes independently or multiply an entire flight's ratios
into one clipping term. With fixed covariance, conditional KL is one half of the
sum of squared conditional-mean changes divided by conditional variances. Clipped
surrogate optimization and KL monitoring follow the usual PPO framework.
[PPO](https://arxiv.org/abs/1707.06347)

Integration constraints from Astra's review, not yet implemented as a trainer:

- Preserve the native 50 Hz brain/command rate and 100 Hz physical forelegs. A
  correlation time of 0.6 seconds means `rho = exp(-0.02/0.6) = 0.967216`, not a
  slower brain or a held native command.
- No noise during the ten static neural warmup frames. The first physical
  command draws a stationary residual. Neither a gate transition nor a TBPTT
  boundary restarts noise. At a replay chunk boundary, recompute the preceding
  mean with gradient; earlier prefix detachment still makes the recurrent
  gradient a truncated approximation, not exact full-flight backpropagation.
- Use nonzero exploration on all four actions because shared trainable pathways
  can change PYT as well as roll. Keep advantages and behavior log probabilities
  fixed during optimization. Fixed noise entropy has no actor-gradient benefit.
- A separate training-only critic may use physical and actuator state, course
  geometry, failures, remaining time, previous residual and current native means.
  Do not feed the just-drawn innovation into a state-value baseline. These compact
  features do not fully identify the entire neural state. Undiscounted complete-
  flight reward-to-go minus a detached baseline is a simple first choice; any
  later GAE must carry credit across TBPTT chunks and respect the full clean tail.
- Monitor deterministic mean displacement as well as conditional KL. A small
  average conditional KL under strongly correlated noise does not establish good
  deterministic flight. Check the `atanh` boundary clamp and zero-noise equivalence
  on actual native outputs before training. Full autonomous evaluation has no
  noise sampler, action-history buffer, critic or privileged observations.

#### CPU foreleg exploration bandwidth measurement

The probe uses the retained source's actual HoverConfig, 256 independent foreleg
instances for 30 seconds, 50 Hz commands and 100 Hz physical updates, initialized
at hover-stick equilibrium. It uses the same random innovations across conditions,
four stationary latent standard deviations **[0.006, 0.002, 0.001, 0.0025]**, and
excludes the first two seconds from measurement. It loads configuration only:
there is no brain forward pass, quad flight, camera, course or optimizer.

| Noise correlation time | Measured latent roll RMS | Measured roll RC perturbation RMS | Versus independent noise |
| --- | ---: | ---: | ---: |
| Independent, 0 s | 0.005991 | 0.004201 | 1.00x |
| 0.1 s | 0.005961 | 0.012451 | 2.96x |
| 0.3 s | 0.005944 | 0.017784 | 4.23x |
| 0.6 s | 0.005930 | 0.020257 | 4.82x |

The comparable input RMS but much larger physical output supports using coherent
exploration with this foreleg plant. No stick saturation occurred in this isolated
probe, which does **not** establish safe flight or a suitable final exploration
amplitude. A subsequent PPO pilot would still need source-flight amplitude checks.
The small raw report is local at
`/home/mark/tmp/fly-over50-20260912/foreleg-ar1-exploration-probe.json`.

Astra's code review found no blocking mathematics or equilibrium-initialization
issue. All **627 tests pass**: new checks compare the conditional sequence density
and gradients with a dense temporal Gaussian, verify the previous-mean derivative,
stationary first sample, stable squash handling and joint four-axis KL. No third-
party implementation was downloaded or copied; the linked papers are references,
not newly redistributed datasets or model weights.

One implementation idea to preserve for that fallback: replay complete episode
microbatches in chronological order with weights fixed throughout each microbatch.
Accumulate gradients over short detached neural chunks, then make one optimizer
step after the complete microbatch. This avoids replaying the entire prefix again
for every short window, while still producing current-weight recurrent states.
Each chunk needs one overlapping previous-mean evaluation with gradient for the
AR likelihood. A new optimizer step must be followed by a new from-zero episode
replay, never continuation from stale recurrent state. This is a proposed memory/
throughput tradeoff to benchmark, not a claim that a full PPO training loop exists.

#### First standalone outcome-search checks: no improvement

Generation two of `pragmatic-phase-balanced-motor-es-002` completed its standalone
32-case development checks. The updated centre achieved **2/32 clean** (0 negative,
2 positive), 28 first gates and 75 clean-prefix gates; 28 episodes hit a ring and
three incurred wrong-order events. The generation's best training candidate
achieved **6/32 clean** (3/3), 29 first gates and 85 prefix gates, with 21 ring-contact
episodes and two wrong-order episodes. Both had zero ground/invalid episodes.
Neither replaced the same-run source, which remains **11/32 clean** (8/3), 28 first
gates and 100 prefix gates. Development seed 1110983 is repeatedly reused and is
not fresh goal validation.

Generation three's updated centre subsequently achieved 1/16 clean training
completions, 13 first gates and 40 prefix gates on new training seed 2026091372,
with zero ground/invalid episodes. There is no standalone generation-three check.
The search was verified live in generation four; its six-generation budget and
settings are unchanged. The >50% varied-course goal is still unmet.

#### Prepared full-flight exploration calibration

`scripts/calibrate_pragmatic_course_exploration.py` implements the next bounded
prerequisite if the outcome search finishes without a useful candidate. This is
**not PPO and does not update or save controller weights**. It loads the retained
source and compares six conditions on one new **32-case training bank**, course
seed 2026091391 and noise seed 2026091392:

1. Unmodified native motor commands.
2. Zero-noise `tanh(atanh(clamped_native_motor))` control.
3. Independent latent noise at the CPU probe's original four-axis scales.
4. AR noise with 0.6-second correlation and one-quarter of those scales.
5. The same AR noise at half scale.
6. The same AR noise at full scale.

The complete current five-gate geometry, 320x200/125-degree camera, RGB plus
roll/pitch inputs, 50 Hz native recurrence and 100 Hz plant remain unchanged. All
conditions use the same initial courses, ten noise-free static warmup frames, and
full 30-second flights, including the post-completion tail. The first physical
command gets a stationary residual. Innovations match across noisy conditions by
episode and command index, remain independent across rows, and never restart at
gates or failures. Residuals are measured against each noisy flight's **own**
native command, not against commands from a different baseline trajectory.

A read-only evaluator observer records position, actual RC and latched failure
flags after each physical command interval. Paired trajectory/stick differences
use only times before either flight's first failure, so arbitrary post-crash
motion does not establish useful exploration. Overall safety/completion metrics
still cover all 30 seconds. Command statistics explicitly include the full tail.
No observed simulator state is fed into the native actor or noise sampler.

Nominate the **largest AR scale** satisfying this coarse screen, rather than the
scale with the largest chance success count: zero ground and invalid episodes,
at least half the deterministic clean completions (and at least one), at least
75% of its clean-first passes, no more than five percentage points additional
stick saturation, roll-RC difference RMS at least 1e-4 and some position-axis RMS
difference at least 1 mm. These are training-calibration thresholds, not a safety
claim or evidence that learning works. The white-noise condition is a bandwidth
control, not a competing deployment controller.

Nomination also requires the zero-noise control to show no native-output clamps,
maximum local command discrepancy at most 1e-6, no additional ground/invalid
episodes, and clean/clean-first rates within the larger of two episodes or ten
percentage points of baseline. This permits ordinary small run variation without
introducing FP64 or byte-exact validation. Report all conditions even if these
checks fail; do not automatically authorize a PPO run when nothing qualifies.

All **640 tests pass**. New tests cover untouched warmup, stationary initialization,
continuous correlated residuals around changing native means, independent rows,
episode reset, zero-noise clamp accounting, copied read-only trajectory records,
failure-excluded differences and nomination criteria. Existing full/compact flight
scoring remains consistent with and without the observer. At this entry the
calibration is prepared but **not launched**; the existing search owns the GPU.
Astra's implementation review found no launch blocker. A reported nomination is
provisional until the calibration report has `status: complete`; partial reports
must not trigger learning.

Planned command after a non-improving terminal search:

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/calibrate_pragmatic_course_exploration.py \
  --checkpoint runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt \
  --output runs/gate/pragmatic-course-exploration-calibration-001/report.json \
  --pairs 16 --seed 2026091391 --noise-seed 2026091392
```

This calibration bank is consumed training data, never a later fresh held-out
goal bank. The deterministic deployed controller has no sampler, exploration
history or critic. No external implementation, model or dataset was downloaded
or added to git for this preparation.

#### Generation-four result and recurrent policy-gradient preparation

The outcome search's generation-four standalone checks both achieved **0/32
clean completions**. Its centre passed 27 first gates and accumulated 63 clean-
prefix gates, with 26 ring-contact episodes and two wrong-order episodes. The
generation winner passed 29 first gates and accumulated 45 prefix gates, with
19 ring-contact episodes, one wrong-order and one wrong-direction episode.
Both had zero ground/invalid episodes. The same-run 11/32 source remains selected;
the six-generation process was verified live in generation five. No fresh native
validation or controller promotion is warranted by these results.

While that bounded search continues, `scripts/pragmatic_recurrent_policy_gradient.py`
implements the core replay operation for a possible PPO pilot. It is **not a
trainer**: no collector, critic, optimizer loop or GPU learning run has been added.
It replays a complete episode microbatch from zero neural state under current
weights, uses ten noise-free/no-gradient warmup frames, accumulates joint four-axis
clipped policy gradients over short neural chunks and leaves optimizer steps to
the caller. The recorded old density must be the unsquashed latent density and
is checked against the old conditional distribution with ordinary FP32 tolerance.

Each chunk after the first re-evaluates the preceding frame with gradient from
its detached input state. This keeps both current and previous native means in
the AR conditional gradient without altering the numerical neural trajectory or
retaining full-flight activations. These are extra **training replay** forwards,
not extra physical brain ticks. A new optimizer step requires a new full episode
replay; there are no reusable stale neural states. Gradient truncation remains an
approximation. Losses normalize by the whole microbatch's valid command count;
if several calls are accumulated before one step, an explicit gradient scale
provides their valid-count weights. Diagnostics include joint conditional KL
mean/p99/max, clipping, all-axis deterministic mean displacement and native clamps.

Astra caught and re-reviewed an important robustness fix: masking a sample only
after likelihood arithmetic does not prevent NaN padding from contaminating
gradients. All stored replay values, including inactive rows, must now be finite;
observations and actor outputs/state are checked before each chunk's backward
pass, and accumulated gradients must be finite before successful return. A caller
must discard partial gradients after an error. Tests include poisoned inactive
latents/observations, incorrect behavior densities and gradient scaling. The
overlap test checks that chunking preserves the feedforward policy gradient,
whereas recurrent tests check continuous values without claiming exact full-
history gradients. All **648 tests pass**, with no remaining review blocker.

For the later collector, preserve the current outcome score through incremental
rewards: +1 per clean-prefix gate, +0.2/5 times its centering credit, -2 once on
first failure, -25 once on the first ground-or-invalid union event, and +5 only
at the final 30-second command if the whole course remains clean. Ring/order
failure alone must **not** terminate training collection: a later ground strike
still incurs its penalty. At ground/invalid, include the causing command, then
make the remaining row absorbing with zero reward and no bootstrap; no clean
success or further prefix credit can survive that event. This is return-equivalent
to the existing full-tail score. Candidate evaluation still runs its unchanged
full 30 seconds. The helper supplies undiscounted reward-to-go with tests for
ring-then-ground and delayed clean-tail credit, but actual collection is not yet
implemented. Exploration calibration and a measured GPU replay cost remain the
next prerequisites, not a reason to claim PPO or improved flight already exists.

#### Six-generation outcome search finished without improvement

The corrected search terminated normally after all six generations in **1,961.79
seconds**. Generation six's centre scored **1/32 clean** (1 negative / 0 positive),
28 first gates and 65 prefix gates, with 28 ring-contact episodes and two wrong-
order episodes. Its generation winner scored **4/32 clean** (1/3), 25 first gates
and 73 prefix gates, with 26 ring-contact episodes and four wrong-order episodes.
Both had zero ground/invalid/wrong-direction episodes. The selected parameter
vector is exactly zero: the retained checkpoint is still the unchanged source,
at 11/32 in this run's reused development bank. No fresh validation is nominated.

This rejects the tested run, not all outcome learning or all motor-coordinate
search. In generations one through five, respectively 12, 12, 15, 16 and 14 of
16 candidates had zero clean training completions. There was no same-bank source
training control, so perturbation scale, changing course difficulty and centre
drift are confounded. Astra recommends allowing at most one **source-centred scale
screen** at 0.1x and 0.25x the original perturbations, with fixed matched antithetic
directions and a source control on the same new courses, before abandoning this
small search family. Do not move the centre during that screen. A brief ES follow-
up is justified only by meaningful matched improvement while preserving useful
competence; mere preservation or another destructive screen should close that
branch. This is a future option, not another launched search or a claim of a
proven neural-capacity problem.

After verifying the previous process's successful terminal exit, the documented
six-condition action-noise calibration was launched at
`runs/gate/pragmatic-course-exploration-calibration-001/report.json`, using the
unchanged source and the planned seeds 2026091391/2026091392. It is the only new
GPU job. No condition, threshold, course geometry or flight duration was changed
in response to the search result. Its results must be recorded separately; the
launch itself is not evidence for useful exploration or improved flight.

#### Full-flight correlated exploration calibration completed

The calibration terminated normally after **198.88 seconds** and its report has
`status: complete`. All six conditions used the same 32 new varied training cases,
the unchanged source weights and full 30-second flight tails. **None of the 192
flights had ground contact, an invalid state or stick saturation.** Gate-ring and
wrong-order failures remain common; this is not general safe racing.

| Condition | Clean completions (negative / positive) | Clean first gates | Clean-prefix gates | Ring-contact episodes | Wrong-order episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native baseline | 3/32 (2 / 1) | 28 | 79 | 28 | 4 |
| Zero-noise latent roundtrip | 3/32 (2 / 1) | 28 | 79 | 28 | 4 |
| Independent noise, full scale | 5/32 (5 / 0) | 26 | 74 | 24 | 7 |
| AR, quarter scale | 3/32 (3 / 0) | 27 | 76 | 27 | 6 |
| AR, half scale | 4/32 (2 / 2) | 26 | 60 | 26 | 11 |
| AR, full scale | 5/32 (3 / 2) | 24 | 68 | 22 | 8 |

No condition had a wrong-direction episode. The source's 3/32 on this new bank
is weaker than its repeatedly reused 11–12/32 development score. This bank is
now calibration/training data, not a fresh final-goal test, and reinforces the
need for broader untouched native validation before any completion claim.

The zero-noise control passes: no native values were clamped, the maximum local
command roundtrip discrepancy was 8.94e-8, and the principal outcome counts match.
The numerical trajectories are not byte-identical: before either flight failed,
lateral position difference RMS was 0.0166 m and roll RC difference RMS 0.00221.
This ordinary closed-loop variation is recorded, not used to launch an FP64 audit.

All three AR scales pass the predefined coarse survival screen, so the nominated
setting is the largest, **full scale**, with four-axis stationary latent standard
deviations **[0.006, 0.002, 0.001, 0.0025]**, correlation time **0.6 s** and
`rho = 0.9672161005`. Its measured latent residual RMS is approximately
[0.005903, 0.002015, 0.001011, 0.002457]. Before either matched flight failed, its
position difference RMS was [0.135, 0.401, 0.080] m and RC difference RMS
[0.0776, 0.0287, 0.00379, 0.00376]. These latter differences include the native
controller's responses along diverging trajectories; they are **not** an isolated
actuator transfer-gain measurement. They are substantially above the zero-noise
control's drift and establish physically effective exploration on this bank.

The noisy 5/32 outcome is **not learned improvement**: weights were untouched,
the sampler is training-only, and this was not a held-out deterministic evaluation.
The >50% varied-course goal remains unmet. This supplies a practical exploration
setting for the planned outcome-learning pilot; actual GPU recurrent-gradient
throughput and the collection/critic/training loop still remain to be implemented
or measured. The single bounded source-centred parameter-scale screen described
above remains an option before committing to that larger training run; neither
branch is allowed to treat these noisy flights as deployed fly success.

#### Final bounded motor-coordinate scale screen

The next test implements Astra's suggested scale check in
`scripts/audit_pragmatic_motor_search_scale.py`. It is fixed at **four Gaussian
24-coordinate directions**, each evaluated with both signs at **0.1x and 0.25x**
the previous bias/log-gain perturbation scales. Directions are shared across
scales; all sixteen vectors remain centred on the unchanged source. There is no
optimizer or centre update. Original scales are 0.0025 for bias and 0.05 for
incoming log gain. The same graph, signs, sensor interface, foreleg physics,
320x200/125-degree camera and full 30-second varied five-gate scoring remain.

Course seed **2026091393** supplies 32 matched training cases, independent of the
previous action-noise calibration. Direction seed is **2026091394**. The source
gets its own standalone control on this bank. Candidate pairs share a 64-row GPU
batch, with independent neural/physical state for every flight. Existing compact
metrics retain per-candidate clean/first/prefix/failure/ground/invalid and fitness;
detailed ring/order diagnostics are recorded on standalone source/nominee checks.
The extra -25 ground/invalid union penalty is still reported, and any ground or
invalid episode makes a candidate ineligible.

Nominate at most **one** candidate that adds at least **four clean completions**
over the matched source. Completion is primary, with side balance and prefix as
tie-breakers, not a per-side nonregression veto. That nominee must reproduce the
four-case gain when its vector is compiled into ordinary native weights and run
standalone on the same training bank. If it fails, do not try a runner-up. Only
a confirmed nominee receives matched standalone source/nominee checks on reused
development seed 1110983. Require the same four-case development gain and zero
ground/invalid episodes before authorizing a brief smaller-scale follow-up.

No nominee, failed confirmation or inadequate development gain closes this limited
motor-coordinate branch and sends work to the prepared outcome-learning pilot.
All tested vectors and outcomes are retained in the report; only a development-
qualified candidate gets a controller file. Four cases is a coarse experimental
screen, not statistical proof, promotion or the >50% goal. A useful nominee still
needs further training and/or fresh varied-course native validation.

Planned command:

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/audit_pragmatic_motor_search_scale.py \
  --checkpoint runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt \
  --output-dir runs/gate/pragmatic-motor-scale-screen-001 \
  --seed 2026091393 --direction-seed 2026091394 --development-seed 1110983
```

All **658 tests pass**, including the fixed antithetic directions, gain/safety
criteria and mocked complete control flow for no nominee, failed confirmation,
failed development and successful development. Astra's implementation review
found no launch blocker. The fixed screen has now been launched with the command
above; launch is not a result and does not change the retained native baseline.

#### Scale screen closed; real recurrent outcome-gradient replay is practical

`pragmatic-motor-scale-screen-001` completed normally in **431.5 seconds**. The
matched source achieved **6/32 clean (5 negative / 1 positive)**. The eight 0.1x
candidates achieved [6, 8, 4, 7, 6, 5, 5, 7] clean flights; the eight 0.25x
candidates achieved [6, 2, 4, 3, 9, 8, 5, 2]. All had zero ground/invalid episodes.
The best, index 12, reached **9/32 (8 / 1)**, below the predeclared **10/32**
nomination threshold. No standalone confirmation, development check or model
export was authorized. This closes the limited 24-coordinate motor search;
the unchanged source remains retained. These are training-bank results, not
fresh goal validation.

The next implementation is `pragmatic_policy_rollout.py`: actual full-flight
collection of physical observation histories, four-axis unsquashed AR latents,
behavior means/densities, rewards and Monte Carlo returns. Actor inputs remain
RGB and roll/pitch only. A separate 76-feature privileged critic may use physical
state, gate geometry/role, time and previous exploration residual for training;
its features are sampled **before** the current innovation and never enter the
actor. Ground/invalid includes the causing command and then freezes that row to
finite pre-contact padding. Ring/order failure alone does not terminate a row,
so later ground costs remain visible. Clean completion earns its +5 only at
the end of all 30 seconds, not when gate five is first crossed.

Two real GPU probes used the same four training episodes, course seed
**2026091395**, noise seed **2026091396**. Both collected 30 seconds; neither took
an optimizer step or changed weights. The 100-frame prefix gradient took **1.33 s**.
The complete **1,500-frame replay took 18.57 s for 6,000 command samples**, with
**2,524 MiB peak reserved GPU memory**. Mean joint conditional KL against the
unchanged behavior policy was **5.85e-8**, p99 **7.89e-7**, ratio mean **0.999997**,
and all selected gradients were finite. Full collection took **25.38 s** and
produced zero clean completions, two first-gate passes and zero ground/invalid
events on these four cases. It is a cost/correctness probe, not a flight result.
Neural values remain recurrent across the full flight; gradients still truncate
at 20 frames, with preceding-frame overlap for the AR conditional derivative.

Reports are `pragmatic-policy-replay-probe-001/report.json` and
`pragmatic-policy-replay-probe-002/report.json`. The first report inherited an old
mask metadata label mentioning teacher preservation; no teacher was actually
used. The driver and second report label the outcome-gradient supervision
correctly. Reports and models remain ignored local experiment artifacts.

#### First bounded native PPO pilot

The new `train_pragmatic_course_ppo.py` and `pragmatic_policy_optimization.py`
implement the first actual outcome-learning loop, starting from the unchanged
phase-balanced source. Astra reviewed the design and emphasized fixed behavior
histories, globally frozen advantages and current-weight recurrent replay.

- **Three rounds**, each collecting **32 fresh varied training episodes**
  (16 mirrored pairs), seeds **2026091400–2026091402**, with noise seeds
  **2026091410–2026091412**. Full 30-second flights and existing geometry/rules.
- Fixed AR exploration: stationary latent sigma **[.006, .002, .001, .0025]**,
  time constant **.6 s**. Exploration is removed during native evaluation/export.
- Train only the **1,097,500 existing seven-hop visual-to-roll-path magnitudes**;
  topology, transmitter signs, biases and time constants remain fixed. Existing
  magnitude bounds [0,8] remain; no teacher targets or source-distance penalty.
- At most **two Adam proposals per round**, LR **1e-6**. Accumulate all eight
  four-episode microbatches before one step, weighting by valid command counts;
  then mask and clip the combined gradient norm at 1. Twenty-frame TBPTT affects
  gradients, not returns or numerical recurrent history.
- Round one uses zero baseline. Later rounds predict with the previously fitted
  separate **76→128→128→1 critic**. Normalize advantages once over the whole
  round's valid samples and freeze them before fitting this round's critic to
  full Monte Carlo returns (five epochs, batch 1024, Adam 3e-4). Critic input
  normalization is initialized from round one and frozen throughout the pilot.
- After every proposal, replay all recorded behavior histories under the new
  weights. Both proposals are compared against the **round-start behavior**.
  Mean joint conditional KL >.005 stops further proposals after an otherwise
  accepted step. Mean KL >.01, true global p99 KL >.10, or any-axis native latent
  mean displacement RMS >.5 stationary sigma rejects that step and restores
  **both parameters and Adam state**. Nonfinite loss/state/gradient/likelihood
  rejects and stops the pilot. Global p99 uses individual valid KL samples,
  never an average of microbatch quantiles.
- Standalone deterministic source and after-round development use 32 cases on
  reused seed **1110983**. Rank clean completions first, side balance then prefix
  only as tie-breakers; zero ground/invalid is required for selection. Continue
  training from the latest accepted weights, not the development winner.
  A gain of at least **four clean development flights** nominates a candidate
  for further work/fresh validation. Smaller improvements can be retained as
  exploratory checkpoints but do not authorize a goal claim.

The pilot reports accepted steps, KL, native mean movement and actual native
flight outcomes. If six proposals hardly move or all are rejected, that is an
optimization-scale finding, not evidence that PPO or outcome learning cannot
work. A critic or lower replay loss is not success. The objective still requires
**above 50% on at least 128 fresh varied, unassisted native flights**, with matched
source and both course sides represented; this reused development bank cannot
satisfy it.

Launch command after implementation tests and advisor review:

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/train_pragmatic_course_ppo.py \
  --checkpoint runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt \
  --output-dir runs/gate/pragmatic-course-ppo-001 \
  --seed 2026091400 --noise-seed 2026091410 --development-seed 1110983
```

Implementation verification: **676 tests pass**. New coverage checks collector
reward/evaluator agreement, terminal-causing commands and finite padding,
pre-innovation critic features, full-round gradient weighting, exact global KL
quantiles, frozen advantages/critic normalization, magnitude projection, trust
rejection with parameter **and Adam** rollback, and complete trainer selection.
Astra found a source-alias selection issue during review; it is fixed and tested:
policy revision advances only for an accepted nonzero parameter change, and a
repeat evaluation of unchanged weights cannot update the selected checkpoint or
manufacture a nomination. Ruff and whitespace checks pass.

#### Preserve these post-pilot diagnostic options

Two pilot proposals have worsened the **full-history fixed-data surrogate** after
their Adam step, including one that passed the KL bounds. This motivates a
directional check before extending training or blaming sparse rewards. Finish
the bounded pilot first; do not alter its settings mid-run.

On one fixed training rollout, freeze observations, behavior actions/densities
and advantages. Compare the truncated gradient prediction `g dot d` with actual
full-history surrogate changes along the **materialized Adam displacement** `d`
at a small, fixed scale ladder. An ordinary repeated zero-step replay supplies
the noise floor; no FP64/byte-exact audit. A nonnegative prediction points first
to optimizer/momentum; a negative prediction that agrees at smaller steps points
to curvature/update size. Persistent disagreement above replay variability makes
the short recurrent gradient window a leading suspect, not a proven cause.

If indicated, test full recurrent backpropagation with activation checkpointing.
Carry neural state **and previous latent mean without detaching**, include the
ten warmup frames in differentiation, sum the complete fixed-data PPO surrogate,
then backward once. This is a derivative through the recurrent policy, **not**
through flight physics, and introduces no deployed state. Use pure replay
functions with immutable chunk indices and do not capture all rendered images in
closures. The installed Torch 2.12 development build exposes non-reentrant
checkpointing; the [official checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html)
describes recomputing intermediates to save memory and warns that changed
recomputation behavior can invalidate gradients. Memory, cost and long-horizon
gradient growth still need a bounded GPU test; this option is not implemented.

If updates become well behaved but native improvement remains unclear, compare
source/candidate with noise off/on on one matched **training-only** course bank,
using common AR innovations. Noisy-only improvement suggests an exploration-to-
deployment gap; training improvement without development improvement suggests
coverage/generalization. Neither improving suggests optimization or noisy return
estimation before a new anatomy change. A stronger outcome critic and GAE are
later variance-reduction options, not changes to make during this pilot.

Teacher-derived potential shaping remains an option but must be described
honestly: with gamma=1 and terminal potential zero, shaped reward-to-go is
`G(t) - Phi(state(t))`, a state baseline rather than additional long-term reward
information. Non-potential centering/speed/path rewards genuinely change the
objective and can reward lingering or post-miss path tracking. Preserve that
idea for a demonstrated reward problem; do not relax ordered-gate or ground rules.

#### First PPO pilot complete: operational, no completion-rate improvement

`runs/gate/pragmatic-course-ppo-001/report.json` is **complete**; the process exited
normally after **914.8 seconds**. The tested implementation is commit `76d90db`,
pushed to master. All three fresh training collections and four standalone
development evaluations ran the complete 30-second course. **No ground or invalid
episodes occurred** in any of them. The pilot made three proposals, accepted two,
and stopped the second proposal in every round under its predeclared trust rules.

| Native development | Clean (negative / positive) | Clean first gates | Clean-prefix gates |
| --- | --- | --- | --- |
| Same-run source | 13/32 (10 / 3) | 28 | 102 |
| Round 1, rejected step / unchanged source | 12/32 (9 / 3) | 28 | 101 |
| Round 2, one accepted step | 9/32 (6 / 3) | 28 | 94 |
| Round 3, two accepted steps | 13/32 (9 / 4) | 28 | 97 |

The unchanged round-one repeat is not a learned change. Round three ties the
source's clean count and wins only the side-balance tie-breaker, so the experiment
retains it as an **exploratory** `best-controller.pt` (also its last controller).
It gains **zero** clean flights, fails the four-extra-flight nomination threshold,
and has **not** received fresh goal validation. The globally retained source is
not replaced by this tie. The varied-course >50% objective remains unmet.

The fresh noisy training batches scored **4/32 (4 / 0)**, **2/32 (2 / 0)** and
**8/32 (5 / 3)**. They are different course banks, so their increasing/decreasing
counts are not matched evidence of learning or of noisy-to-native transfer.
They used 48,000 valid command samples each. Four-episode full-flight gradient
microbatches took about 18–19 s; whole-round backward plus post-step trust replay
took **233, 232 and 229 s**. No extra runtime actor features or memory were added.

| Proposal | Mean conditional KL | Global p99 KL | Decision | Replay loss before → after |
| --- | --- | --- | --- | --- |
| Round 1 | .008214 | .140834 | Reject; restore weights and Adam | .0000374 → .001575 |
| Round 2 | .008400 | .025068 | Accept; stop round | .0000165 → .000445 |
| Round 3 | .007845 | .061407 | Accept; stop round | .00000175 → .000855 |

Round one exceeds the .10 p99 bound. Rounds two and three pass rejection bounds
but exceed the .005 mean-KL early-stop threshold. Native latent mean displacement
RMS stays below .04 stationary sigma on every axis; this is not the rejection
cause. Ordinary unchanged-policy replay differences are reported (before-update
mean KL .000161, .000286 and .00000596), not hidden or pursued into a byte-exact
reproducibility project.

All three actual Adam steps worsen the full-history fixed-data PPO surrogate,
despite finite gradients and two accepted trust checks. This does **not** prove
TBPTT is responsible: finite-step curvature, the materialized optimizer direction
and replay variability must be separated. It does prioritize the bounded
directional test described above over extending this configuration, changing
anatomy or adding denser rewards. No pilot GPU process remains running.
