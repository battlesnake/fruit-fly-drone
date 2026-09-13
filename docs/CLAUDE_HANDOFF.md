# Claude handoff — 2026-09-13

## Stop instruction and current objective

The user requested: finish the current experiment, record findings and future plans,
then stop because Codex quota is nearly exhausted. **Do not automatically launch
another run from the historical plans below.** This handoff is for a deliberate
resumption by Claude/the user. The goal is not complete and has not been declared
blocked.

The intermediate goal is **more than 50% clean completion of varied five-gate
courses by the native fly controller**. The eventual goal is a typical racing lap
from FPV video through front-leg/stick outputs. Neither is solved. A privileged
teacher's success, better imitation loss, or a development-bank win is not a fly
success on fresh courses.

**Stopped after normal completion of all three rounds** of
`pragmatic-whole-approach-full-prefix-001` (exit 0, 828.53 s). Native clean flights
were **3/32, 2/32, 5/32**, versus source **12/32**; all had zero ground/invalid
episodes. The source remains selected at update 0. No fresh validation was
nominated or launched, and no successor experiment is scheduled. A host process
check found no remaining project training/search/evaluation job.

## Start here, not at an old plan's “next experiment”

1. Read this handoff and the final section of [FIVE_GATE_OVER50.md](FIVE_GATE_OVER50.md).
   That file is the complete chronological five-gate ledger: earlier prospective
   instructions are often superseded by later negative results.
2. Inspect the final local report in
   `runs/gate/pragmatic-whole-approach-full-prefix-001/report.json` and the retained
   source below. Reports/checkpoints under `runs/` are local and Git-ignored.
3. Choose one bounded, evidence-driven next intervention. The last several training
   families did not improve autonomous flight; do not resume the last optimizer
   sweep or promote a diagnostic checkpoint merely because it is newer.

The earlier [hover-to-gate ledger](HOVER_TO_GATE_PLAN.md),
[visual-hover record](VISUAL_HOVER.md), and [README results](../README.md#current-results)
retain the earlier findings. [PLAN.md](PLAN.md),
[CONTROL_ARCHITECTURE.md](CONTROL_ARCHITECTURE.md), and
[RACING_ANTICIPATION.md](RACING_ANTICIPATION.md) retain the broader design and the
user's requested ideas; do not discard the progressive teacher/hint curriculum.

## Retained local assets

| Asset | Path relative to repository |
| --- | --- |
| Retained controller; start future comparisons here | `runs/gate/pragmatic-phase-balanced-replay-001/best-controller.pt` |
| Full graph used by that controller | `data/derived/full-visual-connectome-v1.npz` |
| Last experiment report and diagnostic checkpoints | `runs/gate/pragmatic-whole-approach-full-prefix-001/` |
| Last full-prefix auxiliary experiment | `runs/gate/pragmatic-premotor-bearing-full-prefix-001/` |

Retained controller SHA-256:
`953b1d0f2ee1543a0beb11adaa02163a6111e48994672db355b8b1262e69de7d`.
Graph SHA-256:
`8c6ba28d149e9ac4a5223c5919657a1734f2c2cac9828c51114fd0174a383665`.
Both were rechecked at handoff and match the recorded assets.

The retained source completed **12/64** on seed 1020983 and **11/64** on seed
2026091201. The former bank was already used elsewhere in the research; the latter
was fresh at that comparison, but both are now consumed. These roughly 17–19%
results are more informative than repeatedly selecting on development seed
1110983, where source repeats score about 11–13/32. The current matched source is
12/32. Ordinary FP32 closed-loop differences can flip borderline passes; the user
explicitly prioritizes working flight over byte-exact/FP64 certification.

Do not replace the retained source with the auxiliary trial's local 14/32 candidate:
it gained only two reused-development flights, below the nomination rule, and has
no fresh validation. Earlier README 17/128 five-gate results predate the current
independent-yaw and clean-flight rules.

## What actually runs

- **Full traced MaleCNS graph:** 165,122 neurons, 2,749,407 edges; 2,197 mapped
  photoreceptors driven from RGB. Synapse counts are anatomical initialization,
  not pretrained physiological weights. Dynamics, signs and sensory mapping are
  modeling assumptions, documented in the source/provenance records.
- **Actor input:** 320×200 RGB, 125° horizontal FOV, plus roll and pitch angles.
  No acceleration, angular-rate vector, yaw, position, velocity, mass, gate index,
  target bearing, optical-flow decoder, external recurrent memory or flight-mode
  state machine is supplied to the policy. Neural recurrence persists throughout
  each flight. Bookkeeping/privileged labels stay in training and scoring.
- **Output:** identified bilateral front-leg motor pools drive abstract two-axis
  leg/gimbal dynamics; measured Mode-2 stick positions give roll/pitch/yaw/throttle.
  This is not yet an articulated FlyGym fly moving a physical transmitter.
- **Timing:** one neural update per 50 Hz video frame; forelegs/aircraft at 100 Hz.
  Ten static observation warmup ticks initialize neural state before each flight.
  The current behavior-first runner does **not** propagate multiple neural hops
  per video frame. Faster neural integration is a retained design question, not
  something already delivered by the training replay/checkpoint code.
- **Plant:** headless PyTorch 3D renderer and simplified six-degree-of-freedom
  quad dynamics. Sticks request body angular rates and collective thrust; an
  ordinary **proportional body-rate loop**, not a full PID, produces bounded torque.
  Gravity, drag and actuator lag are modeled. There is no deployed auto-level,
  altitude hold, position hold, teacher takeover or waypoint autopilot.
- **Limits:** lumped thrust/torque, not individual rotors/mixer or PX4/AirSim
  fidelity. Euler-angle dynamics and a 70° tilt validity bound do not establish
  inverted/aerobatic operation. Current gate trials start airborne around
  1.02–1.18 m at fixed 35 g mass; they do not test integrated takeoff-to-course.
- **Scene:** differentiated surfaces, world-coordinate texture, relative gate
  colors (current green, next red, later blue, passed black), checkerboard backs.
  Passed black rings remain physical obstacles. Gate yaw varies; arbitrary gate
  roll/pitch and typical hairpins/reversals are not in this intermediate family.

The old compact 498-neuron hover milestone had 100% takeoff and 92.5% loose height
hold over 40×20 s trials, with 0.188 m final-window RMSE. Do not attribute it to the
full graph or call it precise hover. Full-graph behavior-first hover achieved
50/128 strict six-second airborne holds, 0.138 m RMSE, no ground/invalid states;
frozen vision reduced strict success to 6.25%. Full-graph single-gate flight
achieved 50/64 clean passes, 25/32 on each side; frozen vision reduced it to 24/64.
These are real but narrower feasibility results, not reliable five-gate racing.

## Evaluation and ground-contact rules

Use uninterrupted **30 s**, including the tail after the fifth gate. All five
gates must be crossed forward in order, with no ring contact, illegal aperture
traversal, ground contact or invalid state anywhere in that window. Crossing the
expected gate plane outside the ring is a recoverable miss, not a pass or an
automatic evaluation failure.

Keep inner/outer gate radii **0.62/0.76 m** and drone clearance **0.09 m**. The first
gate is at X 2.7–3.3 m, lateral magnitude 0.61–0.78 m, height 1.10 m, obliquity 2–7°.
Later spacing is 0.9–1.5 m, lateral increments up to ±0.20 m with deviation bounded
±0.50 m **relative to the sloping initial course line**, height increments up to
±0.08 m within 0.90–1.30 m, and independent yaw jitter ±15°. Mirror both sides.

The ground proxy is aircraft **center Z ≤0.03 m**, checked at 100 Hz. It can miss
tilted propeller/body contact; this is not full collision geometry. Current tests
start airborne, with no takeoff exemption. Native evaluation permanently
disqualifies a ground-contact flight, even after all gates were passed.

In outcome PPO, the ground-or-invalid union incurs −25 once, in addition to the
first-failure −2, versus +1 per clean-prefix gate and +5 for a full clean course
at the end. The row then absorbs without bootstrap. **The final experiment is
supervised imitation, not PPO:** it uses motor-label losses, excludes invalid/
ground-contact training windows, and uses clean native flight for selection.
Do not claim that its gradient directly contains the PPO ground penalty.

The coverage collector also conservatively ends training eligibility at missed
expected-plane crossings. That historical cutoff is not the evaluator's rule;
its `failed_lessons` bookkeeping is not a count of full-flight failures.

Development seed **1110983 is heavily reused**. Require at least four additional
clean development flights and zero ground/invalid events before nominating a
larger test. A goal claim requires at least **128 fresh full-distribution native
flights, more than 64 clean**, matched source results, both sides, and preferably
a margin/second bank plus a frozen-camera control. Choose genuinely unused seeds
after checking the ledger and local reports. Do not silently simplify the course.

The evaluator's defaults are two aligned gates, not this goal. For future use
only, supply every relevant geometry flag (replace placeholders; not a scheduled job):

```sh
aira confine --memory-reserve 8G -- .venv/bin/python scripts/evaluate_pragmatic_two_gate_zero_shot.py \
  --checkpoint PATH_TO_SELECTED_CHECKPOINT \
  --pairs 64 --seconds 30 --observation-warmup-steps 10 --seed UNUSED_SEED \
  --layout variable --gates 5 --gate-back-pattern checkerboard \
  --yaw-jitter-degrees 15 --spacing-min .9 --spacing-max 1.5 \
  --lateral-step-min 0 --lateral-step-max .2 --lateral-deviation-limit .5 \
  --height-step-min 0 --height-step-max .08 --height-min .9 --height-max 1.3 \
  --output NEW_REPORT_PATH
```

Do not add `--teacher`, enlarged radii, or hidden actor observations. A separate
`--frozen-vision` comparison freezes the camera after 0.5 s.

## Recent learning findings

The complete reports and commands are in [FIVE_GATE_OVER50.md](FIVE_GATE_OVER50.md).
The following table is a navigation summary, not fresh-goal evidence. All listed
auxiliary native checks had zero ground/invalid episodes.

| Experiment suffix under `runs/gate/pragmatic-` | Source clean /32 | Rounds 1 / 2 / 3 | Conclusion |
| --- | ---: | ---: | --- |
| `premotor-bearing-001` | 12 | 11 / 9 / 10 | No native improvement |
| `premotor-bearing-002` | 12 | 10 / 8 / 8 | Stronger learning rate did not help |
| `premotor-bearing-centered-001` | 12 | 12 / 13 / 10 | Mean compensation did not yield a meaningful nominee |
| `premotor-bearing-full-prefix-001` | 12 | 14 / 10 / 12 | +2 best; no meaningful nominee; formulation closed |
| `whole-approach-full-prefix-001` | 12 | 3 / 2 / 5 | No improvement; source retained; stopped for handoff |

Important conclusions and boundaries:

- A privileged full-state continuous-path teacher completed **128/128**; a local
  current-gate roll teacher from the first command, with native pitch/yaw/throttle,
  achieved **27/32**, versus 12/32 all-native. The plant/control interface can do
  the task, but these are assisted diagnostics, not learned fly policies.
- Short recurrent gradient truncation was a real problem in a policy-gradient
  audit: short/full gradient cosine .01055 and opposite first-order loss-change
  signs for the old update. Correct full-history gradients were implemented and
  tested. **Correcting credit assignment alone did not solve flight.**
- Sink-only outcome learning tried Adam, steepest, natural, failure-aware and
  box-constrained natural updates. None produced a meaningful native nominee.
  Do not reopen that optimizer-only branch with another damping/step-size sweep.
- Frozen-head bearing learning at 329 direct premotor neurons changed 11,781
  external incoming edges, with 2,200 internal edges frozen. Stronger learning,
  mean compensation and full-prefix gradients did not establish robust two-sided
  spatial improvement or native flight gains. Close this particular auxiliary
  formulation, not the claim that all fly regions lack useful information.
- Probes found useful velocity correlations in motor-parent activity but weaker
  bearing estimates. Decodability is not proof of causally accessible control.
- Earlier whole-flight physical training used different Hermite tracking losses
  and detached neural boundaries. It is not the same as the final experiment's
  full-history motor imitation on fixed recorded sensory trajectories.

### What the last experiment changes

Launched from committed/pushed **`3f37f81`**, from the retained source, for exactly
**3 rounds ×10 updates**, LR 1e-4. Each round refreshes eight native pairs and four
roll-assisted pairs. Four equally weighted paired lessons per update cover native/
assisted early/later phases, with explicit assisted substitutions where needed.
Roll labels use the current-gate teacher from the first command; pitch/yaw/throttle
labels come from the frozen original source on those same histories. Those labels
do not guarantee unchanged learner pitch/yaw/throttle outputs.

The seven-hop plastic mask contains **1,097,500 edges /92,640 nodes**, reaches 2,168
selected photoreceptors and six roll motor neurons. Biases, time constants, edge
signs and topology stay frozen. The shared helper differentiates all ten warmup
ticks and every preceding sensory frame. Twenty-frame activation checkpoints
are memory boundaries, **not gradient detaches**. Only each selected twenty-frame
lesson carries the motor loss; physical trajectories remain fixed training data.

One pair per microbatch preserves the same four-pair objective and contrast loss;
the anchor is accumulated once, then one clip/Adam/projection step occurs. There
is no auxiliary head or alternating PPO in this run. All deployed outputs remain
native foreleg outputs. The exact launch command is at the end of the ledger.

Per-round `fit_before`/`fit_after` windows are **training examples**, not held-out
generalization. Use their actual source/phase/side fields: fallback samples must
not be mislabeled native. Loss improvement must be compared with actual flight.

All 30 updates had finite losses/gradients and changed native weights. Round-three
first-gate passes improved to **31/32** versus source 28/32, but only **5/32**
completed all gates, with **27 ring-contact episodes** versus source 19. Thus loss
of first-gate competence is not the explanation for that final regression. Some
sampled motor fits improved, others worsened; no evidence supports extending this
recipe unchanged. The ledger records each phase/side fit and full-flight result.
The run's `best-controller.pt` retains source weights with new run metadata;
`update-30.pt` is the failed final learner, not the checkpoint to deploy.

## Future options — untested, not scheduled

First reassess why fitting does not transfer to closed-loop behavior. Another
long optimizer sweep is not the default. Preserve these distinct ideas:

1. **Denser full-history direct supervision:** the new gradient reaches the whole
   prefix, but only twenty frames per selected window carry labels. A dense
   full-trajectory motor objective with explicit phase/side weighting might make
   better use of collected teacher labels. This is a tentative Codex hypothesis,
   not an identified cause, implemented change, or Astra-reviewed plan. Check its
   overlap with earlier dense/physical trials before choosing it; retain native
   observations/outputs and compare actual flight, not just action RMSE.
2. **Representation location/objective:** consider visual projection or other
   anatomically supported pathways if the current premotor site is limiting.
   Establish accessible information and useful causal influence; do not assign
   “EKF” or spatial-state functionality from a region's name or connectivity alone.
   The control-theory decomposition and progressively withdrawn teacher idea are
   retained in [CONTROL_ARCHITECTURE.md](CONTROL_ARCHITECTURE.md).
3. **Visible teacher path and anticipation:** relative gate colors were intended
   to make anticipation easier, not just gate identification. A training-only
   ribbon/direction hint or recorded continuous teacher path, progressively faded,
   remains in [RACING_ANTICIPATION.md](RACING_ANTICIPATION.md). Deployment must keep
   course state/memory in the neural network, not add an external pilot.
4. **Leg bandwidth:** a preserved, untried plant comparison scales spring 25→100,
   motor strength 100→400 and damping 8→16, retaining static stick mapping and
   damping ratio 0.8 while doubling ideal natural frequency. This changes the
   benchmark plant and requires matched controls; it is not a shortcut to claiming
   improvement on the unchanged task.
5. **Neural timing and fidelity later:** more internal neural steps per camera
   frame, video-only ablation, articulated legs, integrated takeoff, better contact
   geometry and a higher-fidelity simulator remain open. Each changes assumptions;
   avoid bundling them into one unidentifiable experiment.

## Implementation and operational notes

- Shared full-prefix replay: `scripts/train_pragmatic_course_replay.py`.
  Bounded final trainer: `scripts/train_pragmatic_course_coverage.py`.
  Prior auxiliary: `scripts/train_pragmatic_premotor_bearing.py` and
  `scripts/pragmatic_premotor_bearing.py`.
- Outcome learning: `scripts/train_pragmatic_course_ppo.py` and
  `scripts/pragmatic_recurrent_policy_gradient.py`. Native evaluator:
  `scripts/evaluate_pragmatic_two_gate_zero_shot.py`. Teacher:
  `src/flydrone/course_teacher.py`. Plant/legs: `src/flydrone/hover.py`.
  Geometry/scoring: `src/flydrone/gate_course.py`.
- At code commit **3f37f81**, **831 tests passed in 10.57 s**, with Ruff and
  whitespace checks clean. Tests cover independent full-unroll/warmup gradients,
  unequal prefixes, pair-microbatch equivalence and native safe selection. Astra
  reviewed the implementation and found no launch blocker. This is not evidence
  that the learning objective works. No code changes are needed merely to hand off.
- Machine: WSL2, RTX 5080, existing `.venv`, headless FP32. Prefix potentially
  heavy commands with `aira confine --`; recent GPU jobs use
  `--memory-reserve 8G` (host RAM scope, not a VRAM reservation), CPU tests 4G.
  Training shares the user's GPU; honor explicit pause/resume requests.
- Use task-specific scratch under `/home/mark/tmp/`, e.g.
  `/home/mark/tmp/fly-over50-20260912/`, never `/tmp` or loose `~/tmp` files.
  Use `apply_patch` for edits, preserve unrelated changes, and avoid destructive
  cleanup. A quiet process is not a completed one: verify terminal exit.
- Repository: `/home/mark/claude/fly`, branch `master`, origin
  `https://github.com/battlesnake/fruit-fly-drone`. The user explicitly authorized
  commit/push to this repository. Do not publish ignored checkpoints/data by
  accident. A new clone will not contain local `runs/` or derived assets.
- Official data/provenance/checksums: [data/README.md](../data/README.md).
  Official paper bundle and licenses: [papers/README.md](../papers/README.md).
  Papers use Git LFS; source tables (~1.9 GB) stay ignored and can be restored with
  `scripts/fetch-malecns-v1.0.sh`. Respect CC BY attribution and other applicable
  licenses. No downloaded pretrained racing model was executed in these trials.
- The user permitted occasional Astra advice; existing advisor was
  `astra_rl_advisor`. No new consultation or GPU experiment is required for this
  stop/handoff. Do not spend the remaining quota on reproducibility perfection.
