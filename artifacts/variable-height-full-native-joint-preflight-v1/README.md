# Full-native joint learnability preflight v1

This is the formal eight-scene result of the restored one-step protocol implemented in
commit `63f4614`. It reopened all 2,749,407 native edge magnitudes, all 165,122 biases
and all 165,122 native time constants while preserving topology, transmitter signs,
retinal/attitude mappings, sensory gains, actor inputs and the foreleg outputs. The
student was differentiated from zero state through the complete cached prefix and
response. The registered seed numbers had already appeared in the documented reduced
smoke, so these banks are fixed and disjoint from each other but are not described as
unseen.

The implementation and learnability checks were strong. Teacher-factorial reconstruction
passed below `1e-6`; opposite-motion endpoint images were exactly equal; source replay
differed by at most `5.96e-8`; constant teacher commands passed through the actual
foreleg/stick plant within 0.00891 RC; and every parameter was restored bit-exactly. The
actual post-bound Adam displacement was descending, with its forward finite difference
agreeing with autograd within 2.39%.

The full step reduced joint normalized MSE from 1.04061 to 0.35287 on training and from
1.25586 to 0.40401 on development. Most of this was common-collective calibration:
`C` NRMSE fell from 2.383 to 0.937 and 2.696 to 1.145. `P` improved slightly and `D`
barely changed. Endpoint damping stayed wrong-signed in all eight scenes; aligned gain
only moved from -0.451 to -0.430 on training and -0.328 to -0.314 on development.

The preregistered preflight nevertheless failed, solely and decisively, because the
candidate disturbed attitude output. Training roll/pitch source NRMSE became
0.05190/0.08658 and development roll/pitch became 0.05401/0.08591, above the 0.05
per-axis limit. This is consistent with source-replay preservation having zero first
derivative at the exact source: it could screen the first joint step but could not shape
it. It does not show that the full native graph cannot learn joint control. It rejects
this exact unprotected Adam displacement and sends the next work to a separately frozen
learning-dynamics/parameterization audit.

No candidate was retained or promoted and no closed-loop hover ran. Compact measurements
are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-joint-preflight-001/report.json` (SHA-256
`e35a6ad20497ee3f1f14202a199257a138a92dde56973f509ec8dd92ad3fd4a9`).
