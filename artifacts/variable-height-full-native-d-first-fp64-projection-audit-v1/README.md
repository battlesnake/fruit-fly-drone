# FP64 projection replay audit v1

This is the formal result of the restored FP64 projection audit frozen in
commit `44448fe`, implemented in `6b61a85`, and corrected before the formal run
in `f4a3b93`. It reconstructed update 21 from the stopped update-20 controller,
generated one set of proposal/Jacobian tensors, and restored the complete source
state without retaining a candidate.

The audit failed its preregistered production-reproduction control. The
historical production projection stopped abnormally in round two; the
regenerated production projection instead completed three rounds and passed.
Its proposal signature otherwise reproduced within the registered numeric
tolerances. This demonstrates termination sensitivity across slightly different
GPU reconstructions, not nondeterminism on identical frozen tensors. The failed
control is preserved and the audit does not authorize an FP64 fitter.

The FP64 result is useful but diagnostic-only. It completed the same 5,508 then
16 edge activations and converged in round three with 5,524 fixed edges. Each
10-row normalized Gram matrix had rank 8 and nullity 2. Primary L-BFGS-B and
independent SLSQP both reported success. Original-unit primal residuals were at
most `5.31e-11`, and normalized primary KKT residuals were at most `3.04e-10`,
well within the registered limits. Canonicalization then passed idempotence,
bounds, a `1.30e-7` post-materialization linear residual, negative D direction,
and a 0.103% finite-difference disagreement.

No ordinary nonlinear scale passed. At scale 1/32, endpoint-D NRMSE improved by
0.001130, but endpoint-C at step 25 was 1.68380272 against the unchanged outer
limit of approximately 1.68378975, a miss of about `1.30e-5`; every other
ordinary gate passed there. The nonlinear repair was deliberately not invoked.

Parameters and Adam state were restored exactly, both stopped-fit source files
remained byte-for-byte unchanged, and no candidate, development/fresh result,
closed-loop run or promotion resulted. Compact measurements are in
[`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-projection-audit-001/report.json`
(SHA-256 `8e92a28ba2c67f76ee8d08426ccc0651e915007111085772e3c81ecfabf0ca80`).
