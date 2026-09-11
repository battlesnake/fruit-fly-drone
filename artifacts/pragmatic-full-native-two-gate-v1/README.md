# Pragmatic full-network two-gate proof

This behavior-first experiment extends the full-MaleCNS single-gate controller to two
role-coloured annular gates without resetting any deployed state. The recurrent state of
all 165,122 neurons, the two forelegs, the virtual sticks, and the aircraft remain
continuous across both crossings. The actor sees only 320×200 RGB plus roll and pitch and
still emits all four flight axes through the front legs.

Course bookkeeping is outside the actor, just like physical race timing. Before the first
crossing, the current gate is green and the next gate is red. Once the aircraft crosses,
the passed gate becomes black and the second gate becomes green. No gate index, crossing
bit, waypoint, detector output, bearing, range, velocity, accelerometer, history buffer or
external state machine enters the actor.

For the first sequence proof the gates are deliberately easy: the second gate lies on the
same flight line, 3.8–4.2 m beyond the first. Whole courses are mirrored left/right. The
first gate retains the single-gate distribution. A harder shallow S-turn is implemented
but remains a later curriculum step.

## Fresh result

On 32 fresh mirrored pairs (64 flights, seed 630983, 22 seconds), the untouched single-gate
actor cleared both gates in 9 flights and one complete pair. A short imitation stage using
a training-only geometry teacher selected update 75. The resulting actor cleared both in
13 flights and three complete pairs, while first-gate passes rose from 56 to 58. Both
mirrored directions completed the sequence, and neither actor produced a ground contact or
invalid flight.

| Measurement | Single-gate source | Two-gate candidate |
| --- | ---: | ---: |
| First gate passes | 56/64 (87.5%) | 58/64 (90.6%) |
| Both gates pass | 9/64 (14.1%) | 13/64 (20.3%) |
| Both members of pair pass both | 1/32 (3.1%) | 3/32 (9.4%) |
| Negative mirrored course passes both | 2/32 (6.2%) | 4/32 (12.5%) |
| Positive mirrored course passes both | 7/32 (21.9%) | 9/32 (28.1%) |
| Ground contacts / invalid flights | 0 / 0 | 0 / 0 |

This proves that an uninterrupted full-network actor can complete a colour-switched gate
sequence. It does not establish reliable racing: four fifths of flights still fail at the
second gate, and the first course has no bend. The next useful work is to improve aligned
gate-two tracking, then introduce the already implemented shallow S-turn rather than
expanding the evaluation bureaucracy.

Only the same five-hop anatomical photoreceptor-to-roll-motor edge set was trainable. The
selected checkpoint changed 19,186 of those edges by at most 0.005933; all unselected
edges, neuron biases, and time constants changed by exactly zero. The 77 MB checkpoint is
kept under ignored `runs/` storage and has SHA-256
`e099fb4f87a81ea7565d0c6bc693a3d236ef27b06df003d23dccd2ea5efa1099`. It must use
Git LFS and retain the MaleCNS CC BY 4.0 attribution if later promoted.
