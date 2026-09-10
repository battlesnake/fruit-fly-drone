# Frozen axis-takeover factorial diagnostic v1

This no-training diagnostic localized the remaining closed-loop failure. On 512 fresh,
matched diverse cases, both the promoted source and rejected dense-DAgger update 200 owned
the first 0.5 seconds. The mass-free reserve teacher then replaced either throttle motor
commands, roll/pitch/yaw motor commands, all four axes, or nothing before the physical
foreleg/stick plant. Native recurrent state continued from actual observations without a
reset. None of these diagnostic interventions is eligible for promotion.

For the source controller, native success was 19.9%. Reserve throttle alone raised it to
56.6% (51.6% light, 61.7% heavy), while reserve steering alone reached 36.9% but retained
zero light-mass successes. Neither intervention met 90% in every stratum. Full reserve
control achieved 512/512, with 7.95 cm mean crossing radius and no collision or miss. The
failure is therefore coupled across throttle and steering, though throttle is the larger
single-axis-group rescue.

The rejected update-200 controller scored zero natively, but full reserve takeover still
recovered 512/512. Its first half-second is therefore not irrecoverably damaging; the
collapse lies in its later native control. See [`report.json`](report.json) for every
stratum and paired success interval. Re-run the exact audit at commit `6393ad8` with:

```bash
scripts/run_gate_axis_takeover_factorial.sh \
  --output-dir runs/gate/axis-takeover-factorial-v1
```
