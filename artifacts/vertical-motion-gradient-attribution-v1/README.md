# Proposal-25 vertical-motion gradient attribution v1

The disposable, training-only audit completed successfully and reproduced the
archived proposal-25 controller. None of the four preregistered whole-bank Adam
trials passed: every step size improved the aggregate loss but worsened both
`ON_down` and `OFF_down`. A larger effective batch is therefore not authorized.

The stratum gradients are strongly opposed within each pathway (`ON` cosine
`-0.8245`; `OFF` cosine `-0.9738`). Nevertheless, the solver returned a finite
unit vector whose independently calculated directional derivatives are negative
for the full objective and all four strata. The frozen protocol classifies that
geometry as inconclusive because SLSQP exited with status 8, so this is not a
certified direction and no controller is retained.

The response attribution also shows that the aggregate training improvement was
dominated by removal of common bias and stationary response, while signed vertical
direction remained small or inconsistent. This supports testing the frozen
candidate vector directly before changing the objective or expanding anatomy.

The compact [`report.json`](report.json) records the terminal decision, controls,
gradient geometry, fixed Adam trials, and hashes of the complete ignored report.
Development and acceptance data remained sealed. This result does not authorize
motion routing, hover, gate flight, promotion, or resuming the stopped trainer.
