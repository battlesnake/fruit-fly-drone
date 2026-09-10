# Canonical D-first correction guard-band audit v1

This is the formal result of the canonical guard-band repeat frozen in commit
`4a51b39` and implemented in commit `ce6f560`. It preserved the prior failed
audit, reconstructed the same immutable update-8 and rejected update-9 state,
and changed only the endpoint-D gradient authority/validation and the internal
continuous-solver C target.

All reproduction and control gates passed. The dedicated endpoint-D objective
supplied the authoritative retention row. Its quarter-step directional finite
difference was measurable and finite and agreed with autograd to 0.0883%, well
inside the preregistered 20% limit. The duplicate multi-loss row retained its
old diagnostic limits and failed them again; as frozen before this run, that
comparison was reporting-only and did not replace the directional validation.

The one continuous active-set solve used the stricter source-plus-0.0198
endpoint-C target and reached maximum ideal linear violation `-9.18e-12`.
Float32 materialization left `1.38e-5` violation relative to that deliberately
stricter solver target, but `-1.67e-4` relative to the unchanged source-plus-
0.0199 acceptance specification. Canonical idempotence and native bounds passed.

Scale 1 was the first correction scale and passed complete replay. Endpoint-C
NRMSE was 1.68359399 against the 1.68369011 acceptance limit. Endpoint-D NRMSE
was 1.37401438, retaining 0.002232 improvement from update 8. Every original
outer C/P/RPY, validity and motor-output gate passed. The fixed update-8
development diagnostic also passed again with endpoint-D NRMSE 1.336928 versus
1.398940 at source.

The audit restored update-8 parameters, complete Adam state and the immutable
resume file exactly and retained no corrected candidate. It authorizes only a
separately registered corrected fitting protocol; it does not promote a
checkpoint or authorize closed-loop hover. Compact measurements are in
[`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-canonical-guard-band-audit-001/report.json`
(SHA-256 `40c34d00e8ab5c69558a3d385197ab8ef0170c8a2bd076c09f75ed2ee22c019e`).
