# Multi-time distillation target preflight

This diagnostic stopped before optimizer update 1. The proposed target combined the
promoted controller's roll, pitch, and yaw actions with the older mass oracle's throttle
action. On 64 balanced, deliberately diverse gate cases after a 0.5-second promoted-policy
prefix, it achieved 87.5% success overall and separately on both mass halves. The fixed
preflight requirement was 90% for every one of those three rates, so no training ran and
no checkpoint was produced.

The stop exposed a teacher-quality issue, not a limitation of the fly graph. See
[`report.json`](report.json) for the machine-readable record.

Reproduce the stopped protocol with:

```bash
scripts/run_gate_multitime_distillation.sh \
  --output-dir runs/gate/multitime-distillation-v1
```
