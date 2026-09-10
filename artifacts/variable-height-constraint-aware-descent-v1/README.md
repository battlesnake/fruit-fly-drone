# Variable-height constraint-aware descent v1

This preflight replaced minibatch Adam with the equal-family-metric steepest descent of
one fixed, balanced visual-height contrast bank. Sixteen signed common-throttle Jacobian
rows constrained the direction. At the v3 boundary, the source-radius tangent was added
to the small solve and the candidate was projected exactly into the source ball before
the usual complete-replay checks.

The source result is promising but insufficient to authorize this run. A quarter step
from the untouched source safely reduced fixed-bank contrast NRMSE by 0.004073, well above
the `1e-5` replay-noise floor. It retained 45.5% of the unconstrained first-order descent
and moved only `3.30e-6` in the source-family metric.

At the retained v3 endpoint, the constrained direction still predicted descent and every
trial improved the fixed bank. None passed the independent complete-replay guard. The
small-pair maximum common-throttle drift was 0.00578–0.00808 motor units across trial
scales, above the 0.005 limit; shrinking toward zero exposed that the v3 endpoint itself
did not generalize its common-output constraint to this new bank. The preflight therefore
stopped before training, with no parameter change or promotion.

This closes the current global local-optimization family. It does not refute visual
information, recurrence, or feasible motion at the original source. The next work should
use the control-responsibility audit and progressive teacher-handoff tracks to identify
and train native visual-error, motion/damping and foreleg-actuation pathways without
allowing unrestricted global parameters to spend the collective-calibration budget.

Compact measurements are in [`report.json`](report.json). The full ignored report is
`runs/variable-height-hover/constrained-descent-preflight-001/report.json`.
