# Full-native endpoint-damping small-step audit v1

This is the formal result of the two-scale extension frozen in commit
`f293999a134889e867e4f8c47bb5a20573acdb2e` and implemented in commit
`73049a18c47873bf19b19525b1a80d45fbbb3db3`. It preserved the failed verdict
of the preceding audit and tested only 1/16 and 1/32 of the same full-native,
endpoint-damping-directed Adam displacement.

All controls passed. The original immutable cache hashes and registered scalar
signature reproduced; the regenerated displacement was persisted, hashed,
reloaded and verified bit-exactly before use. Source replay agreed to
`7.45e-8`, teacher and foreleg/stick controls passed, and parameters were
restored exactly.

Both smaller scales passed training, so the preregistered largest scale, 1/16,
was selected. It improved endpoint-D NRMSE from 1.458401 to 1.454984 on
training and from 1.398940 to 1.396360 on the single development evaluation.
All aggregate and per-horizon common/height limits, RPY limits, validity and
motor bounds passed on both banks. The tightest margin was training endpoint
common NRMSE: it increased 0.019590 against the source-plus-0.02 limit.

Endpoint damping was still wrong-signed in all scenes after this one small
step; aligned gain moved from -0.43297 to -0.42992 on training and from
-0.37881 to -0.37645 on development. The result therefore demonstrates a
safe, transferable local damping improvement—not correct damping, independent
C/D control, sustained learning, hover or flight.

No candidate was retained or promoted and no closed-loop test ran. The pass
authorizes only a separately preregistered bounded D-first fitting diagnostic
whose cumulative preservation is measured against the original source.
Compact measurements are in [`report.json`](report.json). The complete ignored
report is
`runs/variable-height-hover/full-native-endpoint-damping-small-step-audit-001/report.json`
(SHA-256 `a4f5b1ceb47ee7f65b0fae3664a52ce8fa2ed5bc06a765a87099793d4572b2d5`).
