# Assisted recurrent-routing PPO diagnostic v1

This bounded experiment restarted from the preserved source controller and optimized
complete 12-second assisted-flight outcomes. PPO could change only the same 125 existing
fixed-sign recurrent-routing magnitudes, bounded to `[0,8]`; topology, biases, time
constants, and every other magnitude remained frozen. The candidate's native recurrent
state controlled throttle throughout. A parallel frozen source supplied steering for the
first 0.5 seconds and the privileged reserve teacher supplied steering afterwards. Only
scalar throttle motor drive was explored, so this is not a native coupled-flight test.

The numerical preflights passed: source forward parity was within `5.97e-8`, all discrete
assisted-evaluator outcomes matched exactly, the scalar likelihood directional derivative
matched finite differences, and unchanged-policy replay had exact KL `7.54e-13`. Every
truncated burn-in audit passed; mean exact KL was at most `4.45e-7`. PPO's largest accepted
post-update KL was 0.00134, well below the 0.01 cap.

The fixed development source scored 38.28% overall, 0% light mass, 76.56% heavy mass, and
0% in its worst mass/lateral stratum. At iteration 5, the selected diagnostic vector
scored 39.06%, 0%, 78.12%, and 0%, respectively. Iteration 10 regressed to 35.94%
overall and 71.88% heavy while light and the worst stratum remained 0%. The iteration-10
continuation gate therefore stopped the run. Across the ten stochastic training rollouts,
independent per-step throttle exploration produced no light-mass success in 320 light
episodes.

This rejects the tested independent-noise, complete-flight PPO configuration, not native
recurrence or the 125-edge circuit in general. The next no-learning diagnostic should test
whether temporally correlated throttle exploration can produce useful light-mass
trajectories before another PPO budget is spent. Forward progress was 1.0 because every
episode reached the gate plane; terminal height alignment is not gate-opening accuracy.

The fresh 1,024-case final set was not consumed. Nothing was compiled, merged, or
promoted. The complete protocol and logs are in [`report.json`](report.json); the two
validation vectors are in [`archive.pt`](archive.pt), and the rejected diagnostic
selection is in [`candidate-vector.json`](candidate-vector.json). Re-run the code
introduced at commit `71b9ab2`, including the preflight tolerance calibration at
`036780c`, with:

```bash
scripts/run_gate_recurrent_routing_ppo.sh \
  --output-dir runs/gate/recurrent-routing-ppo-v1
```
