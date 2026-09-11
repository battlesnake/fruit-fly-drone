# Vertical T4/T5 commissioning preflight v1

The first committed, one-shot preflight stopped at its finite-difference gate. Exact
source identity passed over all eight normal and literal-reversal cases with zero maximum
difference. Every gain and bias probe passed the fixed 2% gradient-error limit; T4c and
T5c tau probes measured 10.31% and 2.84%, respectively, so the optimizer step was not
opened and no candidate was retained.

This result also establishes that the complete 165,122-node K32 backward pass is practical
on the RTX 5080: all 24 gradients were finite and nonzero, source/local identity was
restored, and peak CUDA reserved memory was only 2.77 GiB against the 14 GiB limit.

The compact [`report.json`](report.json) records the decisive values and hashes the complete
ignored run report and start marker. The failure authorizes no training, hover, gate flight
or promotion.
