# Full-native endpoint-damping step audit v1

This is the formal result of the fresh-cache protocol frozen in commit
`87ef766d4b90919c30d952ab0c8316a652a1fdce` and implemented in commit
`16b6b171ebb36de42a5b97b8e669308846e25472`. It differentiated the complete
native recurrence from zero state while optimizing only the literal endpoint
velocity-odd throttle response. All native edge magnitudes, biases and time
constants were open; actor inputs, topology, signs, sensory mappings and
front-leg outputs were unchanged.

All experimental controls passed. The four newly seeded caches were generated
once, persisted under the ignored run directory, hashed, reloaded, and reused
without regeneration. Persisted file and tensor hashes matched, endpoint
opposite-motion images were identical, source replay agreed to `5.96e-8`, the
teacher and foreleg/stick positive controls passed, and the bound-projected
directional derivative agreed with its finite difference within 0.24%.

The damping-only direction was effective and safe for attitude. Endpoint-D
NRMSE improved monotonically from 1.45840 by 0.05516, 0.02759, 0.01374 and
0.00685 at scales 1, 1/2, 1/4 and 1/8. Even the full step kept roll/pitch/yaw
source NRMSE at 0.01447/0.02259/0.00349. Endpoint damping remained wrong-signed
at this one-step scale, but its teacher-aligned gain moved in the correct
direction.

No registered scale passed all training preservation gates. At scale 1/8,
height response passed, but aggregate common-collective NRMSE increased by
0.02471 and its step-20/25 values increased by 0.02483/0.03891, beyond the
fixed 0.02 tolerance. Because training selected no candidate, no development
model evaluation occurred. The development tensors remain available only for
the next separately frozen test; this run did not use them to choose a scale.

This result is evidence that the native graph has a locally useful visual
damping gradient. It is not yet evidence that damping can be changed while
preserving collective response, nor a hover or flight result. No parameters
were retained or promoted, no bounded D-first fitting was authorized, and no
closed-loop test ran. Compact measurements are in [`report.json`](report.json).
The complete ignored report is
`runs/variable-height-hover/full-native-endpoint-damping-step-audit-001/report.json`
(SHA-256 `b8191ccff3e9ee4afd048627c79c7e2f988b1ec3437ba13ab1916cf0e28930aa`).
