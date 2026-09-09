# Full-network oracle-distillation diagnostic v1

This directory preserves a rejected experiment that exposed every existing edge
magnitude, neuron bias, and native membrane time constant in the 1,138-neuron recurrent
connectome. It is evidence, not a deployable controller: no `candidate.pt` is present and
the promoted `gate-motor-interface-es-v1` checkpoint remains unchanged.

The frozen, mass-aware native oracle supplied training labels while the student alone
drove the foreleg sticks and quadcopter. Seventy-five percent of each training window came
from current-student histories; the remaining 25% came from clean oracle-driven histories
as a stabilizer. Incoming recurrent state was reconstructed from reset under the current
student, then held fixed while differentiating each 50-step window. The deployed actor
would still have received only current FPV, roll/pitch, body-Z specific force, and its own
persistent connectome state—never mass, timestamps, stacked observations, teacher state,
or an external recurrent module.

- `report.json` records the protocol, causal and numerical audits, all 100 updates,
  validation snapshots, a fresh 1,024-flight evaluation, sensor controls, and failed
  promotion checks. SHA-256:
  `6693b9de9d8efdea193684d6a66c1ab27f2a7989eedcd3f0fa63ca2b712b3de2`.
- `candidate-vector.json` stores the exact selected 4,469 edge magnitudes, 1,138 biases,
  and 1,138 raw time constants. Its parameter-vector hash is
  `d36a1bc1f637897958f6ec7646ddb6af963c5a9bd3fec1b905cb4a62483693ef`.
- `archive.pt` retains the ten validation parameter snapshots. SHA-256:
  `75c7478633eb40e3b9b9ca278b4670a63f650506f06df9e2ed0ffba61f28ecda`.

All weight-, bias-, and time-constant-only finite differences passed at 25 and 50 recurrent
steps on early and late windows. Generating oracle labels versus disabling them produced
bit-identical CPU student trajectories, and the frozen teacher succeeded from 95.3% to
100% when taking over student flights between zero and two seconds.

The run nevertheless learned a nearly mass-independent throttle shift. On fresh student
histories, the selected candidate's matched light/heavy throttle slopes were only 0.024,
-0.0007, and 0.008 at 0.75, 1.5, and 3 seconds. Fresh 1,024-flight success changed from
47.46% overall (3.91% light, 91.02% heavy) to 43.07% (50.78% light, 35.35% heavy).
Constant-1g and matched-pair acceleration swapping were indistinguishable from live input.
The action-fidelity, heavy non-degradation, overall improvement, and causal-acceleration
checks all failed, so no checkpoint was promoted.
