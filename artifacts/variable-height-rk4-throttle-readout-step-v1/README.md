# RK4 throttle-motor readout step audit

This is the compact terminal record of the source-restored last-hop audit registered in
commit `757048e` and implemented in `536ef12`. It used the selected RK4-M=1 solver and the
same immutable 24-pair RGB cache. The only plastic parameters were the 493 existing signed
edges entering the seven throttle antagonist motor neurons, plus those neurons' seven biases
and time constants. No new readout, input, state or connection was introduced.

Every numerical control passed. Three fixed-prefix objectives differed by at most `5.96e-7`.
The one β1=0 Adam proposal had directional derivative `-0.001430`; its scale-1/16 measured
and predicted objective changes were `-9.341e-5` and `-8.938e-5`, agreeing to 4.41%. All
gradients outside the mask were exactly zero, all unmasked parameters remained bit-identical,
and candidate loading, bounds, recurrence, endpoints and source restoration passed.

No registered ordinary scale reached the required `0.001` NRMSE improvement. The largest,
scale 1, improved detached-prefix and complete-prefix NRMSE by `0.000602` each. It passed
every preservation gate with large headroom: pair-common throttle drift was only `0.000428`
RMS and `0.000544` maximum against limits `0.005` and `0.01`; RPY changes were at float32
noise scale. Smaller candidates produced almost exactly proportional improvements.

The family diagnostics localize the effect. At scale 1, edge magnitudes supplied
`0.000591` of the full-prefix improvement. Bias supplied `0.0000106`, and the registered
time-constant step produced no measurable improvement. All were nonselective diagnostics.

The formal classification is therefore `no_safe_local_readout_step`: “safe” here includes
the preregistered minimum useful improvement, not merely stability. The result does not show
harmful output coupling; it shows that this exact one-step Adam scaling was too weak to meet
the declared progress floor. Per the frozen branch, the exact last-hop/optimizer route is
closed. A new experiment must use an upstream routing design or a separately declared direct
constraint-aware readout step, not retroactively pass scale 1.

No candidate or optimizer was retained, and no fitting, hover, gate flight or promotion was
authorized. Compact measurements are in [`report.json`](report.json). The complete ignored
report is `runs/variable-height-hover/native-rk4-throttle-readout-step-001/report.json`
(SHA-256 `508c54ebb00aa8419e66e540bbc7ca45bd0d254858e76eab590cc4f39dc878f7`).
