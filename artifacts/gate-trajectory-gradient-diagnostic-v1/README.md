# Full-horizon trajectory-gradient diagnostic v1

This is a **rejected training-method experiment**, not a controller checkpoint. It asks
whether a complete eight-second flight loss can be differentiated reliably through the
recurrent connectome, measured foreleg sticks, renderer, and quadcopter plant.

The structural checks passed. Two plants given identical teacher commands produced exactly
the same positions and velocities, the ordinary actor call matched its frozen deployment
equivalent, all parameter gradients were finite, and perturbing the first roll command had
a measurable effect on late trajectory loss. No recurrent state was detached and no
parameter changed during the rollout.

The numerical derivative check failed decisively. Edge and bias gradient norms reached
`1.37e17` and `6.93e16`. Analytic directional derivatives were about `1e16`, while central
finite differences were tens to hundreds over all preregistered perturbation scales. Even
late-loss sensitivity to the first roll command disagreed by about fifteen orders of
magnitude. The harness therefore stopped without applying an optimizer update.

This does not mean the graph lacks recurrence or hysteresis. It means naive 800-step
backpropagation is catastrophically ill-conditioned around this closed-loop trajectory.
Subsequent training should use complete-flight rollout scores without differentiating
through the history.

[`report.json`](report.json) is the compact committed record. The full ignored report hash
is included for audit. These small source and JSON files do not require Git LFS. The graph
continues to follow the upstream MaleCNS CC BY 4.0 terms documented in
[`data/README.md`](../../data/README.md).
