# Teacher-assisted motor-interface ES diagnostic v1

This bounded experiment optimized actual 12-second flight return instead of teacher
action error. It split the promoted controller's 24 native motor-interface parameters by
their target pools: 18 roll/pitch/yaw gains and biases were searched while the mass-free
teacher supplied throttle, and six throttle gains and biases were searched independently
while the teacher supplied roll/pitch/yaw. Core weights and all native time constants were
frozen. Assisted results were training-only and could never be promoted.

The steering stage passed its fixed generation-20 gate and completed 30 generations. On
256 fixed diverse cases, the selected safe steering vector raised assisted success from
53.9% to 75.4% and the worst mass/lateral stratum from 10.2% to 53.1%, while preserving
the source's 97.7% negative-lateral result. This shows useful steering control is available
within the small readout family when throttle is competent.

The throttle stage failed its fixed gate and stopped at generation 20. Its best safe
vector moved success only from 35.9% to 39.1%; light-mass and worst-stratum success stayed
at exactly zero while heavy-mass success rose from 71.9% to 78.1%. The six static
throttle-pool gains/biases therefore did not produce the required state-dependent control.
Because both assisted stages had to pass, native merge and joint polish did not run and
nothing was promoted.

The rejected vectors and search states are retained in [`archive.pt`](archive.pt), with
the selected zero deployment vector in [`selected-vector.json`](selected-vector.json) and
the complete protocol in [`report.json`](report.json). Re-run the exact code at commit
`0190b3f` with:

```bash
scripts/run_gate_assisted_motor_es.sh \
  --output-dir runs/gate/assisted-motor-es-v1
```
