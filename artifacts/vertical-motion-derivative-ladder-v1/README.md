# Vertical T4/T5 derivative ladder v1

The preregistered float32 derivative ladder reproduced the v1 preflight exactly, but
did not satisfy its all-six-probe qualification rule. Five probes qualified at two
adjacent resolved scales. `T4c_tau` agreed in sign at every scale and was within the
2% error limit at steps `0.008` and `0.002`, but those passing scales were not adjacent.

The optimizer gate therefore remained closed: no update ran, no candidate was retained,
and no training, hover, gate flight, or promotion is authorized. Source/local identity
was restored exactly, development and acceptance remained unopened, and peak CUDA
reserved memory was 5.12 GiB.

The compact [`report.json`](report.json) preserves the per-scale derivative evidence and
hashes the complete ignored terminal report and start marker.
