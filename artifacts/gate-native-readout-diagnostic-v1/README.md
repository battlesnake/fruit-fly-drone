# Native anatomical readout diagnostic v1

This is a **rejected readout experiment**, not a flight checkpoint. It tests whether the
launch-derived state retained by the native recurrent motif can be converted into a
calibrated throttle correction by one fixed anatomical readout.

The frozen update-180 encoder was chosen by its held-out neutral-suffix ordering. The fit
used all 37 existing transmitter-signed edges from 19 neurons into the two throttle motor
pools, averaging neurons equally within each pool. One learned, time-independent
antagonist bias inside those motor pools was also allowed to cancel a shared offset. The
same readout had to work from reset through 1.5 seconds under clean live sensing, small
accelerometer noise, and an intervention that replaced all inputs with neutral values
after 0.75 seconds. Exact mass constructed the training target only.

The bias settled at `0.04662`, inside its ±`0.2` bound, but the held-out live correction
slope fell from `0.197` at 0.75 seconds to `0.008` at 1.5 seconds. Overall R² was `0.240`
and normalized RMSE was `0.872`. Even the linearized replay of the fitting data failed,
so post-fit recurrent feedback is not the main cause. The preregistered screen rejected
the readout and no flight evaluation or candidate promotion was performed.

[`report.json`](report.json) is a compact committed record. The full reports and retained
selection checkpoints stay under ignored `runs/`; their hashes are included for audit.
These small source and JSON files do not require Git LFS. The derived graph follows the
upstream MaleCNS CC BY 4.0 terms recorded in [`data/README.md`](../../data/README.md).
