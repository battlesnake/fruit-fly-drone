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
or changing optimizer momentum. This alternative is not yet implemented.

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

For this monotonic-X teacher, a training lesson ends after missing its expected aperture,
because the path tracker cannot recover a gate behind it. Evaluation still permits
exterior-plane manoeuvres. Loss and gradient checks prevent nonfinite optimizer updates.

All exploratory checkpoints remain ignored under `runs/`; do not publish them as a new
best fly until full-flight results justify it. If a checkpoint is promoted, retain the
MaleCNS attribution and use Git LFS.
