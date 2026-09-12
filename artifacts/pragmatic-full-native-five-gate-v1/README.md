# Pragmatic full-network five-gate proof

This behavior-first result shows the same uninterrupted full-MaleCNS actor flying through
five physical annular gates. It uses the existing two-gate checkpoint without further
training. All 165,122 neural states, both abstract forelegs, four virtual-stick states and
the aircraft state remain continuous for the whole flight.

The actor receives only 320×200 linear RGB at 125° horizontal field of view plus roll and
pitch. It emits roll, pitch, yaw and throttle through identified front-leg motor pools and
the two forelegs. It receives no gate index, crossing bit, waypoint, target geometry,
position, velocity, accelerometer, timer, external history or external state machine.

Course bookkeeping belongs to the renderer: the current gate is green, the next is red,
later gates are blue, and passed gates become black. The beginner course is deliberately
straight, with the original 0.62 m inner-radius annuli spaced 0.9–1.1 m apart. Thus the
fifth plane is 3.6–4.4 m after the first, comparable to the earlier two-gate test's final
distance.

## Fresh result

On 32 new mirrored pairs (64 flights, seed 940983, 22 seconds), cumulative clean crossings
were 53, 42, 32, 19 and 4. The four complete five-gate flights included three negative-side
and one positive-side course. There were no ground contacts, invalid states or saturated
sticks. An earlier 16-pair bank produced 6/32 complete flights, also on both sides.

| Measurement | Live RGB | RGB frozen after 0.5 s |
| --- | ---: | ---: |
| Gate 1 passes | 53/64 | 23/64 |
| Gate 2 passes | 42/64 | 16/64 |
| Gate 3 passes | 32/64 | 10/64 |
| Gate 4 passes | 19/64 | 6/64 |
| All 5 gates pass | 4/64 | 5/64 |
| Negative / positive all-gate passes | 3 / 1 | 5 / 0 |
| Complete mirrored pairs | 0/32 | 0/32 |

The frozen result is a warning about this easy collinear geometry: a few trajectories can
coast through several nearby apertures without useful updated vision. It does not erase
the live behavior proof, but it means this particular completion-rate comparison is not a
clean vision ablation. The earlier single-gate paired result—50/64 live, balanced 25/32 per
side, versus no positive-side frozen passes—remains the causal steering control.

This milestone proves possibility, not reliable racing. The fly starts airborne, the
course has no turn, fresh completion is 6.25%, and failures accumulate with distance. The
next task is a state-dependent between-gate centering/braking response, followed by wider
spacing and shallow turns.

No new checkpoint is stored for this result. It reuses the ignored 77 MB checkpoint at
`runs/gate/pragmatic-two-gate-aligned-roll-path-001/best-controller.pt`, SHA-256
`e099fb4f87a81ea7565d0c6bc693a3d236ef27b06df003d23dccd2ea5efa1099`. If promoted, that
file must use Git LFS and retain the MaleCNS CC BY 4.0 attribution.
