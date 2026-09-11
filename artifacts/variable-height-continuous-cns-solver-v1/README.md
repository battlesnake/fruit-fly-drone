# Continuous-CNS solver selection

This is the compact terminal record of the numerical extension registered in commit
`2f3e46b` and implemented in `be0c1de`. It reused the exact immutable visual-motion cache
from the preceding rate audit and loaded the original `paired-dynamic-001` source afresh for
every condition. It performed no rendering, case generation, learning or controller change.

Exponential Euler at K=16 passed against the newly evaluated K=32 fine reference. The
24-value throttle-contrast difference was `0.008004` of the frozen teacher scale (limit
`0.01`), and the 48×4 terminal-motor RMS difference was `0.001231` (limit `0.005`). K=64
therefore remained unopened, as preregistered.

All three classical RK4 candidates also passed directly against the K=32 reference. RK4-M=1,
M=2 and M=4 used 4, 8 and 16 recurrent graph evaluations per 20 ms camera frame. Their
normalized contrast errors were `0.008310`, `0.007284` and `0.007373`; their all-axis motor
RMS errors were `0.001152`, `0.001840` and `0.001843`. The selection rule used only numerical
accuracy and graph-evaluation count, so RK4-M=1 was selected. It holds the observation fixed
during its four stages and adds no actor input, history or state outside the 165,122-neuron
MaleCNS recurrence.

This is consequential for feasible training: the selected method is numerically adequate by
the fixed thresholds with one eighth of the graph evaluations used by the K=32 reference and
one quarter of the adequate K=16 exponential update. On this audit RK4-M=1 took 1.26 seconds,
versus 11.98 seconds for K=32, although runtime was not a selection criterion.

The converged behavioral result also clarifies the control problem. Correct damping sign was
0/24 for every condition. The K=32 reference had aligned gain `-0.2170`, and selected RK4-M=1
had `-0.2233`, against a correctly signed target. Thus the one-update-at-50-Hz implementation
was numerically coarse, but integration coarseness did not cause the sign error. The native
source circuit genuinely needs temporal-credit/routing training; more accurate integration
alone cannot turn it into a vertical damper.

This result permits the selected solver to be named in a separately preregistered training
experiment. It does not authorize that training to execute and does not authorize hover,
gate flight or promotion.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/native-continuous-cns-solver-001/report.json` (SHA-256
`daeb4c000b3421a2d1d4d22890ccb900f654f1587748d035d88c4f884cff85a7`).
