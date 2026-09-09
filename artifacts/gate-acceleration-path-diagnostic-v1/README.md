# Recurrent acceleration-path search diagnostic v1

This is a **rejected selective-plasticity experiment**, not a promoted controller. It asks
whether complete recurrent paths from the existing body-Z acceleration inputs to the
existing throttle motor pools can improve light-mass flights without sacrificing the
heavy-mass flights already solved by `gate-motor-interface-es-v1`.

The search changed only the magnitudes of 282 existing signed edges lying on paths of at
most four hops. Biases, time constants, topology, transmitter signs, every other edge,
retinal mapping, leg mechanics, and quad dynamics remained fixed. The deployed candidate
would have used only the persistent connectome state: there was no mass input, history
feature, scaler, estimator, clock, planner, or external actor state. Thirteen selected
edges initially had zero magnitude; their exploration perturbation measurably changed
early throttle-stick motion, so the experiment was not effectively limited to nonzero
edges.

Thirty-two antithetic directions were evaluated on 32 common-seed, exactly matched
light/heavy flight cases for each of 60 generations. The generation-20 continuation rule
passed. The selected held-out candidate improved light success by 9.38 percentage points
and reduced heavy success by 1.56 points on 256 matched flights.

The fresh final audit did not meet the preregistered promotion threshold. On 1,024 matched
flights, light success rose from 4.88% to 12.89%, an 8.01-point gain with a paired 95%
interval of 5.65–10.36 points. Heavy success fell from 91.60% to 90.62%, within the
two-point noninferiority margin, and overall success rose from 48.24% to 51.76%. The
required light gain was at least ten points. A separate ordinary balanced evaluation
improved from 47.95% to 50.88% and reduced mean gate-plane radial error from 0.674 m to
0.614 m.

The candidate was 1.07 points better with live acceleration than constant 1g, but swapping
the acceleration traces within each matched light/heavy geometry pair changed no net
successes. It therefore does not demonstrate mass-specific acceleration adaptation. No
checkpoint is promoted, and this four-hop mask should not be widened automatically.

[`report.json`](report.json) is the compact committed record. The exact 282-value rejected
search vector and its edge-index mapping are retained in
[`candidate-vector.json`](candidate-vector.json); it is explicitly not a controller
checkpoint. The full ignored report hash is included for audit. These small source and
JSON files do not require Git LFS. The graph continues to follow the upstream MaleCNS CC
BY 4.0 terms documented in [`data/README.md`](../../data/README.md).
