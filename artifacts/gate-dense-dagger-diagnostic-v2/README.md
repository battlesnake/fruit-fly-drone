# Corrected-weight dense DAgger diagnostic v2

This controlled follow-up changed dense DAgger v1's throttle normalization and midpoint
guard, not its actor, graph, initialization, optimizer, windows, cases, seeds, or replay
schedule. Throttle used the post-takeover RMS correction between the mass-free teacher and
the immutable promoted-controller shadow (`0.07875`) instead of the full hover-command RMS
(`0.38798`). A qualifying midpoint now required five-point gains both overall and on the
light-mass half, with no more than a five-point heavy-mass loss.

The expert again flew 64/64 cases, full student replay stayed within `2.4e-7`, and all
three native parameter families passed central finite-difference checks over the exact
100-step training window. Nevertheless, update 50 reached only 19.5% success versus the
19.9% source baseline. Update 100 fell to 11.3%, so block-end selection restored the
source. Mixed expert/student training then reached zero successes at updates 150 and 200.
The bounded run stopped and selected unchanged update 0; nothing was promoted.

Fixed expert-history diagnostics make the result more specific. At update 50, light-mass
throttle MAE improved slightly in the 4--5 s crossing window (`0.0860` to `0.0818`) and
5--6 s window (`0.1068` to `0.1030`), but worsened in the 0.5--1.5 s takeover window
(`0.0910` to `0.0940`). Thus correcting throttle weighting alone did not align dense
action imitation with successful closed-loop flight.

The rejected state is retained in [`selected-vector.json`](selected-vector.json), the
validation snapshots in [`archive.pt`](archive.pt), and the complete protocol, gradient
audit, replay audit, and time/mass diagnostics in [`report.json`](report.json). Re-run the
exact v2 code at commit `f8299e8` with:

```bash
scripts/run_gate_dense_dagger.sh \
  --output-dir runs/gate/dense-dagger-v2
```
