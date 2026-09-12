# Variable-height, variable-line five-gate improvement

This behavior-first experiment improves the full-MaleCNS actor on five uninterrupted
annular gates whose heights and relative lateral positions change. The result is still
an early feasibility result, not a reliable racing controller.

The deployed actor boundary is unchanged: 320×200 linear RGB at 125° horizontal field
of view plus roll and pitch enter the connectome; the front-leg motor pools move two
abstract forelegs and therefore the four virtual transmitter sticks. All 165,122 neural
states, both forelegs, the sticks, and the aircraft remain continuous for the entire
flight. Gate index is used by the renderer to make the current gate green, the next red,
later gates blue, and passed gates black. It is never given to the actor.

## What changed

Gates 2–5 are 0.9–1.1 m apart. Each can move laterally by up to 0.10 m relative to the
previous course perturbation, bounded 0.25 m from the first-gate centreline, and in height
by up to 0.05 m, bounded to 0.95–1.25 m. Mirrored members share the same height course and
see mirrored lateral courses. Across the two fresh evaluation banks, sampled heights
covered the complete 0.95–1.25 m range and lateral deviations covered 0.0004–0.25 m.

The useful curriculum change was to stop waiting for rare native flights to reach every
late gate. Training-only rollouts cycle reset positions across gates 2–5 and run short
teacher-driven recovery segments. A separate gate-one rollout replays the frozen source
on every update. These reset positions and gate indices select training examples only;
final evaluation always starts before gate one and is fully native and uninterrupted.

Only 19,286 existing visual-to-roll synapses within five anatomical hops were trainable.
Signs and topology were retained; every other edge, all biases, and all time constants
were frozen. A 60-update run at learning rate 3e-5 selected update 30, where the held-out
development bank improved from 2/16 to 4/16 complete flights with unchanged first-gate
pass rate.

## Fresh variable-course result

Two untouched banks compare source and candidate on identical courses:

| Bank | Controller | Gate 1 | Gate 2 | Gate 3 | Gate 4 | All 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| seed 952983 | source | 51/64 | 41/64 | 27/64 | 16/64 | 7/64 |
| seed 952983 | candidate | 49/64 | 37/64 | 27/64 | 19/64 | 10/64 |
| seed 959983 | source | 47/64 | 36/64 | 22/64 | 14/64 | 6/64 |
| seed 959983 | candidate | 47/64 | 33/64 | 22/64 | 11/64 | 7/64 |
| combined | source | 98/128 | 77/128 | 49/128 | 30/128 | 13/128 |
| combined | candidate | 96/128 | 70/128 | 49/128 | 30/128 | **17/128** |

Five-gate completion rose from 10.2% to 13.3%, a 3.1 percentage-point or 31% relative
increase. Candidate completions were present on both mirrored sides (7 negative and 10
positive, versus 6 and 7 for the source). First-gate completion declined by 2/128 while
gate-four reach was unchanged; the gain comes from converting more late-course attempts,
not from an easier first gate. There were no ground contacts or invalid simulations.

On the exact earlier straight-course bank (seed 940983), the same candidate improved from
4/64 to 9/64 complete flights. Its cumulative crossings were 50, 35, 24, 18 and 9 versus
the source's 53, 42, 32, 19 and 4. This is useful retention evidence, though the losses at
early gates show that the next training objective should protect full-course reach more
directly.

## Interpretation and next design iteration

This clears the immediate milestone: the five-gate rate is higher on genuinely variable
height/lateral courses and on the original straight course. It also identifies a limit of
the present approach. The staged teacher takes roughly 40 seconds to negotiate these
close gates because it repeatedly aligns and advances, while successful native flights
finish within the 22-second window. More teacher imitation would likely teach the wrong
flight style.

The next promising branch is therefore trajectory-level optimization of the existing
native actor—evolution strategies or recurrent RL over a tightly restricted anatomical
parameter subset—using cumulative gates and all-five completion directly. The balanced
late-gate curriculum remains useful for coverage, and control-theory teachers remain
useful for recovery and diagnostics, but neither should dictate the deployed trajectory.

The 77 MiB candidate checkpoint remains in the ignored exploratory path
`runs/gate/pragmatic-five-gate-variable-balanced-001/best-controller.pt`, SHA-256
`6679044e84b52c90ff336a3fb4fbc96ca649b2118b7ec689715ad2f13e81c10c`. It must be added
through Git LFS, with the MaleCNS CC BY 4.0 attribution retained, if later promoted.
