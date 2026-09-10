# Variable-height bridge gradient/signal audit v1

This diagnostic distinguishes visual signal loss from functional interference in the
rejected all-style response bridge. It compares the unchanged paired-dynamic source with
bridge-v2 update 25 on two independent balanced gradient batches, two physical marker
displacements, official MaleCNS superclass partitions, and two non-committing parameter
steps. No actor parameter was saved or promoted.

Marker information survives the native graph. At the source, 5 cm and 10 cm marker
contrasts produced throttle sensitivities of 0.102 and 0.111 motor units per metre. The
rejected candidate increased them to 0.115 and 0.126. After 0.5 seconds, normalized state
contrast was present in visual, central, VNC and output-motor populations; it did not
collapse before the readout or cancel between the two throttle pools. The injected
retinal marker signal was weaker than independent scene nuisance (marker/nuisance RMS
ratios 0.34 and 0.66), but it was repeatable and scaled with displacement.

The failure is common-collective drift. At identical static pose, marker and history,
the four styles produced source throttle outputs from 0.143 to 0.150; update 25 shifted
that range to 0.180–0.189. A family-RMS-normalized negative contrast gradient of only
`2e-5` RMS improved fixed-burn-in contrast NRMSE, but immediately caused 0.482 normalized
common-throttle error and 0.322 dynamic throttle-source error. Its R/P/Y errors were much
smaller. A `1e-4` step made common-throttle error roughly 2.0 and lost the full-history
contrast gain.

At the source, common-throttle, legacy and dynamic preservation gradients are numerical
zero compared with the contrast gradient (roughly `1e-5` versus `2.5` global norm).
Consequently, adding their squared losses cannot protect the first step. At the rejected
candidate those gradients become large and mostly align, but only after collective
calibration has moved. Undefined source-reference cosines are not interpreted as
compatibility evidence.

The next intervention should therefore be a functional trust region: project the actual
Adam proposal against signed per-style common-throttle output Jacobians, then backtrack
it against complete-replay common-throttle, legacy-response and dynamic R/P/Y constraints.
Any proposal that violates closed-loop calibration is rejected without advancing the
optimizer. It should not add visual gain, freeze guessed anatomical modules or change the
actor interface.

Compact measurements are in [`report.json`](report.json). The full ignored diagnostic is
`runs/variable-height-hover/bridge-gradient-audit-001/report.json`.
