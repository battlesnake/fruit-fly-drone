# Anticipation, gate direction and a racing teacher

Design update, 2026-09-12. The goal is still video → recurrent fly nervous system →
front legs → virtual sticks → acro flight controller. Temporary attitude inputs remain
as documented in [the main plan](PLAN.md). No course index, route planner, teacher state
or extra recurrent controller is added to the deployed actor.

## What the colours are for

Relative gate colours are intended to teach anticipation, not just target selection.
The current gate stays green, the following gate red, and later gates blue; a correct
crossing advances these roles and the passed gate becomes black. The present palette
has three entries, with all more-distant ranks sharing blue. It does not yet uniquely
encode an arbitrary number of future ranks. Keep that horizon explicit in experiments.

The desired behaviour is to choose an approach and **exit velocity through the current
aperture** that sets up the following gate. The current staged teacher uses the current
gate only; more imitation of that teacher will not explicitly teach a multi-gate racing
line. Changing its targets is more promising than merely increasing imitation updates.

Use matched course pairs with identical aircraft state and first gate, but place the
next visible gate left/right or higher/lower. A successful anticipatory controller should
change its approach or exit steering before the first crossing, when that is physically
useful. Also measure the subsequent crossing: an early steering difference alone is not
proof of a better racing line. Avoid imposing a roll-sign rule; braking and setting up
the next gate can legitimately require steering away from the current gate's bearing.

## Front/back and course rules — implemented

The RGB renderer can retain a solid-colour front annulus and paint a coarse, gate-local
checkerboard on its back, preserving the role hue. The aperture remains empty. Front is
the side from which a legal crossing travels along the gate's positive normal. Passed
gates remain black on both sides and retain their physical geometry.

Enable this on the sequence evaluator or course imitation trainer with
`--gate-back-pattern checkerboard`. The setting is saved in new checkpoints; the default
for old checkpoints remains `solid` to avoid silently changing their observation task.

The sequence evaluator and main course lessons now check every gate, both directions,
in crossing-time order within each physics step:

- Only the current gate's clean forward aperture traversal advances the course.
- Backwards or out-of-order aperture traversals, including re-traversing passed gates,
  are illegal. An event satisfying both conditions incurs one traversal penalty.
- Ring contact counts for future and passed gates, from either side.
- Crossing the infinite gate plane outside the ring is legal. Circling around a gate is
  needed for several real track elements and must not itself be punished.

The selection score is currently +1 per valid gate before the first failure, −1 per
illegal traversal, −2 per ring collision, and −2 once for ground contact/invalid state.
Passes in the same physics step as a failure are conservatively excluded from prefix
progress, and later recovery passes remain diagnostic only. These are provisional
selection weights, not tuned RL rewards. Main course
imitation lessons latch an illegal event or collision as failure; the existing isolated
gate-one replay retains its older one-gate termination rules. The imitation objective
itself has not become RL.
Late-gate lesson resets now start in a clearance-checked gap between adjacent gates;
the old fixed 1.40 m approach could start behind a passed gate on the 1 m-spaced courses.
More complex layouts should obtain resets by flying a teacher prefix instead.

Selection retains the gate-one competence floor, then prioritizes clean paired and
individual course completion, followed by penalized gate progress. Reports distinguish
`clean_course_success_rate` from raw `all_gates_pass_rate`, and expose direction/order
violations and event penalties. Clean completion also requires no ground contact or
invalid state over the full evaluation window, including its post-pass tail. The older
`strict_all_gate_success_rate` additionally retains legacy tilt, saturation, terminal
clearance and missed-plane restrictions; use the new clean metric for circling courses.
Old published results have not been retrospectively re-evaluated under these new rules.

These are swept-centre/plane checks with a radial drone-clearance margin, not a realistic
swept-body collision solver. The world still has yaw-only, zero-thickness annuli. Flags,
poles, hurdles, gate thickness and full-attitude aerobatics need additional simulation
work. Do not claim all eight racing elements are supported yet.

## Teacher and hint curriculum — next experiment, not implemented yet

1. Build a continuous, feasible path through two or three shallow turning/height-varying
   gates. Constrain clearance through each aperture and the desired exit direction;
   simply averaging gate centres can cut through a ring. Match thrust limits, rate
   limits, motor response and the actual foreleg/stick plant. Require the teacher to
   complete these courses before distilling it.
2. Record its images and front-leg-compatible control targets along full trajectories.
   Include on-policy student recoveries relabelled by the teacher (DAgger-style), not
   only perfect demonstrations. Carry native recurrence across the useful anticipation
   interval and retain gate-one lessons. Teacher/critic may use ground truth; actor may
   not. Evaluate coordinated four-axis adaptation, not just the current roll-only mask.
3. Optionally render a short, world-space direction arrow or path ribbon covering the
   next few gates, depth-tested against the scene. Mix hinted and unhinted episodes from
   the beginning, then fade hints completely before selecting a controller. This is
   visual scaffolding, not a hidden waypoint vector. Compare against no-hint imitation
   so we learn whether it actually helps.
4. Select on hint-free uninterrupted flights on held-out layouts. Report clean
   completion, violations, clearance, time and the paired anticipation intervention.
   Keep simpler course replay to detect regressions. If the teacher provides a useful
   starting policy but closed-loop errors persist, fine-tune with recurrent RL using
   these gate events, completion/time rewards and a training-only privileged critic.

A route that is entirely hidden or out of the camera's field of view can be ambiguous.
Do not require different actions for identical visual histories just because the teacher
knows an unseen course. The fly may use its own recurrence to remember previously seen
gates; training should initially make the relevant next gate observable.

Start with a shallow three-gate slalom or bend. Then extend to carousel/180-degree turns
and composed laps. The supplied GetFPV Learn PDF (Ervin Liao, 2020-08-21, *Drone Racing –
The Eight Common Track Elements*) covers simple gates, 180s, slaloms, Split-S, corkscrews,
carousels, ladders and hurdles. In particular, corkscrews/stacked ladders may require
circling back for another crossing in the same direction; they are not simply alternate
forward/backward passages. This motivates explicit face markings and aperture-based
rules. No copy of that copyrighted PDF or its diagrams is added here.

## Reputable teacher sources checked

The following are primary sources linked by the University of Zurich Robotics and
Perception Group. No external repository, executable or checkpoint has been installed
or run for this update.

- [Time-Optimal Planning for Quadrotor Waypoint Flight](https://github.com/uzh-rpg/rpg_time_optimal)
  (Foehn, Romero, Scaramuzza, Science Robotics 2021): official example planner, GPL-3.0,
  with CSV trajectory export. This is the most concrete starting reference for offline
  multi-gate path generation. Its published requirements pin an old scientific-Python
  stack, so do not install them over this project's environment. A trajectory optimizer
  is not a closed-loop teacher: we still need a tracker and recovery demonstrations in
  our dynamics, with aperture/direction constraints checked independently.
- [RPG MPC / perception-aware MPC](https://github.com/uzh-rpg/rpg_mpc): official quadrotor
  tracking controller, GPL-3.0-or-later, based on ACADO and a ROS-era stack. Useful
  control-design reference; integration is not turnkey for our headless Torch plant.
- [Swift supplementary data](https://zenodo.org/records/7955278) (Kaufmann, 2023): the
  official deposit advertises recorded drone positions from races/time trials and
  pseudocode. It does not advertise a runnable champion-policy checkpoint. Recorded
  paths can illustrate line choice, but are not automatically feasible demonstrations
  for our dynamics or newly sampled courses. Check deposit/file reuse terms before
  importing or redistributing data; no archive was downloaded in this update.
- [Bootstrapping RL with Imitation for Vision-Based Agile Flight](https://rpg.ifi.uzh.ch/bootstrap-rl-with-il/index.html)
  (Xing et al., CoRL 2024): the authors report a privileged-state RL teacher → visual
  imitation → visual RL curriculum for gate sequences. This supports the proposed
  training strategy, not a claim that the same results transfer to a fly connectome.

Before adoption, pin the chosen source revision, review code and dependencies, retain
license/provenance records, and use an isolated, resource-confined teacher environment.
Prefer inspected plain trajectory data; do not load unknown pickle-based model objects
or run convenience installers blindly. Official provenance reduces risk but is not a
malware-free guarantee. Do not copy GPL implementation into differently licensed files
without resolving the licensing implications. Any large redistributable artifacts must
use LFS; exploratory runs remain ignored.

## Verification for this update

`aira confine --memory-reserve 8G -- .venv/bin/python -m pytest -q` passed 461 tests.
New cases cover role-preserving back-face patterns, both traversal directions, physical
crossing order, dark/future gate collisions, exterior-plane manoeuvres, touch-and-return,
clean-versus-eventual completion, failure-aware selection and short-spacing lesson resets.
Targeted lint checks passed. No new trained-flight result is claimed by these tests.
