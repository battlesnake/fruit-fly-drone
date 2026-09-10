# Variable-height projected trust-region v1

This non-promotional experiment tested whether common-collective drift could be removed
from the all-style visual-height bridge. Each ordinary Adam proposal was projected in an
equal-family-RMS metric against eight signed common-throttle Jacobian rows: one dynamic
window for each of four wall/floor styles at both 5 cm and 10 cm marker separations. A
single scalar capped each family RMS step at `2e-5`; up to six backtracking trials then
had to pass complete replay from zero against source and previous controllers.

The mechanism worked locally. The preflight reproduced the source within `1.20e-6`.
Ordinary capped Adam worsened the fixed-bank contrast and violated common-throttle,
legacy and dynamic-replay guards. Projection changed the fixed-bank first-order contrast
derivative from `+4.42e-4` to `-6.14e-5`, reduced the linearized common residual below
`2.97e-7`, and produced a safe half-scale step.

It did not produce enough learning. A replay retaining the boundary endpoint accepted 12
of 22 attempted updates. The fixed-bank contrast NRMSE changed from 0.738583 to 0.737629,
only a 0.129% improvement against the preregistered 10% requirement. The last accepted
point used 99.99% of the
`1e-4` source-radius budget; small-pair common NRMSE, maximum common drift and dynamic
throttle error were also close to their limits. Five consecutive later proposals failed
even at `1/32`, so the run stopped early. No final selection suite was run. The endpoint
was retained only in ignored run storage for the next diagnostic; it was not promoted.

This establishes that Jacobian projection can protect the native controller locally. It
does not show that the connectome lacks visual information or recurrence, and it does not
justify weakening safety limits. The next diagnostic replaces minibatch Adam with a
fixed balanced contrast direction and solves the source-radius and functional constraints
together; it first tests feasibility at the source and this run's boundary direction.

Compact measurements are in [`report.json`](report.json). The full ignored report is
`runs/variable-height-hover/trust-region-001/report.json`.
