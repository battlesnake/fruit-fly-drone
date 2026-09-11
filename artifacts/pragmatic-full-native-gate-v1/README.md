# Pragmatic full-network annular-gate proof

This is a compact record of a behavior-first experiment, not a formally promoted or
fully reproducible milestone. It demonstrates that the recurrent full-MaleCNS actor can
use an ordinary FPV stream to steer the simulated quadcopter through an off-axis annular
gate on either side.

The deployed actor receives 320×200 linear RGB at 125° horizontal FOV plus estimated
roll and pitch. All 165,122 neurons and 2,749,407 anatomical edges remain in the recurrent
network. Roll, pitch, yaw and throttle come only from the two front-leg motor pools and
the measured virtual-stick positions. It receives no gate bearing, mask, range, waypoint,
velocity, accelerometer, mass, previous action, phase, external history or external state
machine. A 0.2-second pre-arm observation lets activity propagate through the initially
zero neural state while the aircraft and sticks remain at their airborne trim.

Training used a privileged visual-servo teacher only as a label. The selected final stage
updated 19,286 existing edges on anatomical paths of at most five hops from 1,327
photoreceptors to all six roll motor neurons. Every other edge, every neuron bias and every
time constant was frozen. The deployed flight itself contains no teacher or gate geometry.

## Fresh result

One untouched batch sampled 32 physically matched left/right pairs (64 flights) at seed
620983. Initial altitude varied from 1.02–1.18 m, forward gate range from 2.7–3.3 m,
lateral displacement magnitude from 0.61–0.78 m, and gate obliquity from 2–7°. The clean
aperture radius after vehicle clearance was 0.53 m, so flying straight could not pass.
Mass was nominal and the aircraft began airborne with settled sticks.

| Measurement | Live RGB | RGB frozen after 0.5 s |
| --- | ---: | ---: |
| Clean gate passes | 50/64 (78.1%) | 24/64 (37.5%) |
| Strict completed flights | 38/64 (59.4%) | 16/64 (25.0%) |
| Both members of pair pass | 18/32 (56.2%) | 0/32 |
| Negative-offset passes | 25/32 (78.1%) | 24/32 (75.0%) |
| Positive-offset passes | 25/32 (78.1%) | 0/32 |
| Ground contacts / invalid flights | 0 / 0 | 0 / 0 |
| Mean gate-plane crossing radius | 0.385 m | 0.776 m |

The exactly balanced live side result and frozen-camera collapse on one side are strong
evidence of continuing visual steering, rather than a fixed open-loop launch maneuver.
They are not evidence of robust racing: this test starts airborne, contains one gate, uses
a deliberately narrow geometry distribution, and remains below the formal 90% target.

The 77 MB exploratory checkpoint remains at the ignored path
`runs/gate/pragmatic-visual-roll-path-001/best-controller.pt` and has SHA-256
`35956cae1ba37d976fc237cf55d54a483aa14791e9b391d23a9b196552505076` on this
machine. It is deliberately not committed as an ordinary Git object; if promoted later,
it must be added with Git LFS and the MaleCNS CC BY 4.0 attribution and transformation
notice described in [`data/README.md`](../../data/README.md).

The next behavior-first test keeps neural, aircraft and foreleg state uninterrupted across
two gates. The just-passed gate goes dark, the current gate takes the learned target colour,
and the following gate takes a fixed secondary colour. Gate identity and course progress
remain simulator/rendering state and are never actor inputs.
