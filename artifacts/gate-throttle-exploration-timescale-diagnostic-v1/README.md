# Throttle exploration-timescale diagnostic v1

This no-learning audit tests whether the failed recurrent-routing PPO used throttle noise
that changed too quickly to discover viable light-mass trajectories. It evaluates the
unchanged source on the same 256 development cases under deterministic throttle, two
fixed seeds of independent per-step latent noise, and the same two innovation seeds
filtered by a stationary AR(1) process. Both noisy modes have latent standard deviation
0.03; correlated noise has a 0.2-second correlation time. Frozen-source steering handles
the first 0.5 seconds and the analytical reserve handles steering thereafter.

The temporal intervention worked mechanically. Motor perturbation RMS was approximately
0.0247 in both noisy modes, while the measured foreleg-throttle trace differed from the
deterministic trace by only about 0.0061 RMS under independent noise and 0.032--0.034
under correlated noise. These trace differences are complete closed-loop effects, not an
isolated actuator-filter measurement, because trajectories diverge and simulation traces
continue after classified failure.

It did not produce useful exploration. Deterministic success was 0% light and 76.56%
heavy. Independent-noise heavy success was 77.34% and 76.56% across the two seeds, with
0% light success in both. Correlated-noise heavy success fell to 67.97% and 67.19%, while
light success again remained 0% in both. All episodes crossed the gate plane, but light
cases crossed roughly 2.09 m vertically off-centre under every mode; the gate's clean
radius is only 0.53 m.

The preregistered continuation gate required at least 10% light success under correlated
noise in both seeds, with heavy success no more than 10 percentage points below matched
independent noise. Both heavy checks narrowly passed at a 9.375-point drop, but both light
checks failed. The tested 125-edge PPO/exploration family is therefore paused; correlated-
noise PPO was not built.

No actor parameter changed, no fresh final case was consumed, and nothing was compiled or
promoted. Correlated noise was diagnostic-only external state, not actor memory. Full
metrics and protocol are in [`report.json`](report.json). Re-run commit `295a282` with the
resource-limit adjustment at `7fe554d` using:

```bash
scripts/run_gate_throttle_exploration_timescale_audit.sh \
  --output-dir runs/gate/throttle-exploration-timescale-v1
```
