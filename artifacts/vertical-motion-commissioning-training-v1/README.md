# Vertical T4/T5 commissioning training v1

The preregistered production-FP32 run completed 25 accepted proposals and then
stopped at its mandatory whole-training-bank gate. The equal-weight 24-batch loss
fell from `0.804611` to `0.476532` (40.77%), exceeding the required 25% reduction.
However, the separately protected `ON_up` and `OFF_down` direction hinges worsened,
so the all-strata rule correctly rejected the candidate.

The numerical K32/K64 gate did not run, development and acceptance pixels remained
unopened, and no vertical-motion module was retained. The unchanged source was
restored exactly and peak CUDA reserved memory was 2.805 GiB. Motion-to-DN/VNC
routing, hover, gate flight and promotion remain unauthorized.

The compact [`report.json`](report.json) preserves the full-bank decision, component
changes, terminal parameters and hashes of the complete ignored terminal artifacts.
The terminal parameters are evidence for failure analysis only, not a retained
candidate or an authorized controller.
