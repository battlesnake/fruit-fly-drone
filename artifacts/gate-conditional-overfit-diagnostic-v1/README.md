# Conditional endpoint overfit diagnostic v1

This directory records a deliberately tiny representability test. It is not a deployable
controller and does not count toward the gate-flight goal; no checkpoint is present.

Eight distinct annular-gate geometries were paired at exact mass scales 0.92 and 1.08.
The promoted controller alone drove each 0.75-second trajectory while a frozen mass-aware
native controller supplied action labels in shadow. Starting from the promoted checkpoint,
all existing edge magnitudes, neuron biases, and native membrane time constants were
trained through the complete 75-step recurrent prefix. Equal, independently normalized
losses on light-minus-heavy throttle contrast and pair-mean throttle prevented an
unconditional shared trim from satisfying the objective.

- `report.json` contains both 250-update learning-rate runs, full-prefix finite-difference
  audits, training and disjoint-geometry results, and the exact identical-input control.
  SHA-256: `dbbeddf7ffb699cb52a95b696e49a2ff4796c2ca2003d01174650c7137759ea7`.
- `candidate-vector.json` stores the diagnostic parameters selected from learning rate
  0.0003 at update 180. Its parameter-vector SHA-256 is
  `8b24183012f3fa86639d461f155fbaab352fcd89b51da08f5bf28ee0bad9212e`.
- `archive.pt` stores the best snapshot from each learning-rate run. SHA-256:
  `f337b3b5309c9292436f07ba06342a5a09c55e758e00df85e435e0e6d2fabf0f`.

On the eight training pairs, normalized throttle-contrast error reached 2.12% and pair-mean
error reached 4.60%, both below the predeclared 10% threshold. Giving both members of each
pair identical complete sensor prefixes produced exactly identical four-axis outputs, so
batch position and labels did not leak into the actor. On eight separately generated gate
geometries without more training, contrast error was 5.71% and mean error 17.84% at 0.75
seconds. The untrained 0.50-second endpoint failed.

The result establishes that the native recurrent connectome can learn a sensor-conditioned
light/heavy action difference. It narrows the failed full-flight distillation to temporal
credit assignment and generalization: the next experiment should train normalized pair
contrast and mean objectives at multiple times through their complete recurrent prefixes.
