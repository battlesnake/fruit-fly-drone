# Accepted update-21 transaction snapshot qualification v1

This is the compact record of the two-process audit registered in `44f39c5` and implemented
in `0acf7c5`. Its producer reconstructed update 21 once, ran the unchanged FP64-primary
ordinary-then-repair path, required fixed-development transfer, and archived the complete
accepted transaction. Its verifier then loaded that archive in a separate process without
recomputing the primary or repair.

The audit passed. The selected scale-1/16 plus scale-1.0-repair candidate has parameter
SHA-256 `223d3eda...`; pending Adam remained exactly `6743cc01...`. Training endpoint-D
NRMSE was `1.35364187`; fixed-development endpoint-D was `1.31926703`, improving update 20
by 0.00176442 with every preservation gate passing.

The 224 MiB ignored snapshot has file SHA-256 `fb4c311...`. All eight recorded tensor-tree
hashes verified exactly after load, including candidate parameters, optimizer state, repair
rows, authoritative D row and correction direction. Training and development each replayed
all 25 metric values within the existing tolerance. The archive and all locked inputs were
unchanged, and controller plus optimizer were restored exactly in both processes.

This authorizes only a separately registered continuation that loads this exact accepted
update-21 boundary and generates fresh proposals beginning at update 22. Sign fraction is
still zero and gain remains negative; the audit did not run hover, use fresh data or promote
a controller. The large snapshot remains under ignored `runs/` and is not committed or
placed in Git LFS.

Compact measurements are in [`report.json`](report.json). The complete ignored report is
`runs/variable-height-hover/full-native-d-first-fp64-update21-snapshot-audit-001/report.json`
(SHA-256 `c8f665732e2ebe84b0ee09fea8dee431e4e6ca021a8035bca1e9276d831413e5`).
