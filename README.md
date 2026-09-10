# Fly connectome quadcopter pilot

This is a research workspace for a **connectome-constrained artificial pilot**. Its
recurrent core follows a selected subgraph of the MaleCNS v1.0 fruit-fly connectome.
Activity in identified front-leg motor-neuron pools moves two virtual Mode-2 transmitter
sticks, and the measured stick positions become the four FPV acro commands: roll, pitch,
yaw, and throttle.

## What this project is—and is not

MaleCNS is a static wiring diagram. Its published connection “weights” are counts of
detected synapses between segments, not measured electrophysiological strengths or
pretrained artificial-network weights. The source data do not provide neuron dynamics,
receptor-specific signs, delays, plasticity, or a runnable fly mind.

A successful result here would show that an artificial controller constrained by fly
anatomy can be trained for a simulated flight task. It would not show that a biological
fly knows how to fly a quadcopter.

## Quick start

After cloning, install Git LFS and restore the LFS-managed paper bundle:

```bash
git lfs pull
```

Then restore or verify the official MaleCNS v1.0 source tables:

```bash
scripts/fetch-malecns-v1.0.sh
```

Read [the project plan](docs/PLAN.md) for the architecture and experiment roadmap. The
current quantitative results are in the [hover milestone](docs/HOVER_MILESTONE.md) and
[annular-gate milestone](docs/GATE_MILESTONE.md).

## Repository guide

- [`docs/PLAN.md`](docs/PLAN.md) — architecture, simulator decision, curriculum,
  experiments, safety, and milestones.
- [`docs/HEADLESS_SPIKE.md`](docs/HEADLESS_SPIKE.md) — reproducible WSL2/MJWarp
  acceptance tests and initial RTX 5080 results.
- [`docs/HOVER_MILESTONE.md`](docs/HOVER_MILESTONE.md) — the first passing
  connectome-to-forelegs-to-sticks takeoff and visual-height hover result.
- [`docs/GATE_MILESTONE.md`](docs/GATE_MILESTONE.md) — the first recorded annular-gate
  traversal and its larger evaluation, including the unmet acceptance criteria.
- [`data/README.md`](data/README.md) — the exact MaleCNS snapshot, provenance,
  licensing, checksums, and intentionally omitted downloads.
- `data/raw/malecns-v1.0/` — the local Feather tables, ignored by Git because the
  selected core is about 1.9 GB.
- [`papers/README.md`](papers/README.md) — the LFS-managed paper bundle and primary
  reading list.
- [`scripts/fetch-malecns-v1.0.sh`](scripts/fetch-malecns-v1.0.sh) — the resumable,
  checksum-verifying official-data downloader.

## Current results

### Visual hover: milestone passed

The first differentiable surrogate experiment passed its goal-level acceptance test. In
40 randomized 20-second episodes it achieved:

- 100% lift-off;
- 92.5% successful visual-height hold;
- 0.188 m final-ten-second altitude RMSE;
- 3.66° roll/pitch RMS;
- no ground recontacts.

A paired mid-flight test compared a changed world-space height band with an unchanged
continuation from identical physical and recurrent state. The band-induced height
difference had the correct direction in every trial. Freezing the initial FPV image
reduced goal success to zero, providing evidence that the visual signal is necessary.

This result uses a compact, auditable **498-neuron MaleCNS subgraph** and abstract
bilateral two-axis foreleg/stick kinematics. It is not an articulated fly-leg result, a
FlyGym result, or an independent MuJoCo transfer result.

### Annular gate: working checkpoint, milestone not yet passed

A denser **1,122-neuron MaleCNS subgraph** can lift off and traverse a 1.24 m
inner-diameter annular gate. The gate centre begins 0.8 m off-axis and its plane is 20°
oblique to the launch displacement.

The current checkpoint succeeds on **40.5% of 1,024 held-out trials**, with randomized
side, obliquity sign, and vehicle mass. Freezing the first FPV frame reduces success to
zero. This is a genuine vision-dependent traversal result, but it remains below the 90%
goal threshold and the broader random-geometry task is harder still.

A follow-up direct-sensor checkpoint adds eight identified `wind_gravity` cells carrying
body-Z accelerometer feedback through MaleCNS paths to the throttle leg pools. A bounded
complete-flight search over only 109 existing pathway edges raises the best result to
**45.6%**, versus 40.5% when the same controller is held at a constant 1g. Disabling
either signed sensor channel hurts performance, but swapping traces between vehicle
masses does not; this is useful recurrent feedback, not demonstrated mass identification
or a solved gate task. See the [annular-gate milestone](docs/GATE_MILESTONE.md) for the
controls and limitations.

Several diagnostics sharpen the next step. Adding a four-cell throttle-stick
proprioception route improved a fresh balanced suite to 49.4%, but constant and shuffled
position controls retained essentially all of the gain, so that checkpoint is rejected
as evidence of position sensing. In contrast, an explicitly privileged two-parameter
trim driven by the simulator's exact mass achieved 1024/1024, while shuffled mass labels
achieved only 103/1024. Delaying that oracle until 0.5, 0.75, or 1.0 seconds still met the
90% threshold, establishing a short but usable calibration window.

The accelerometer history contains enough information: a held-out diagnostic decoded
normalized mass from the first 0.25 seconds with R²=0.983. More importantly, an
18-neuron native recurrent motif retained mass ordering to 1.5 seconds after all visual,
attitude, and acceleration inputs were neutralized at 0.75 seconds (Pearson r=0.921 at
the last training checkpoint). That is genuine internal hysteresis with no external
history feature, clock, estimator, or actor state machine. A follow-up fit used all 37
real signed edges into the throttle motor pools plus one fixed internal antagonist bias;
held-out gain still collapsed from slope 0.197 at 0.75 seconds to 0.008 at 1.5 seconds.
The readout was rejected before flight evaluation. Thus the graph has demonstrated memory,
but this hand-selected mass-code scaffold is not a usable controller. The evidence is
retained under
[`gate-proprio-diagnostic-v1`](artifacts/gate-proprio-diagnostic-v1/) and
[`gate-mass-oracle-v1`](artifacts/gate-mass-oracle-v1/), with a compact recurrence record
under [`gate-recurrence-diagnostic-v1`](artifacts/gate-recurrence-diagnostic-v1/) and its
[`native-readout follow-up`](artifacts/gate-native-readout-diagnostic-v1/).

A complete eight-second differentiable unroll was also rejected before training. Its
forward simulation and early-to-late causal path were valid, but analytic parameter
gradients exploded to about `1e17` and disagreed with measured directional derivatives by
roughly fifteen orders of magnitude. The result rules out naive full-flight backpropagation
at this checkpoint, not recurrent control itself; the next optimization stage uses
complete-flight rollout scores without differentiating through history. See the compact
[`trajectory-gradient diagnostic`](artifacts/gate-trajectory-gradient-diagnostic-v1/).

That rollout route has now produced the best native checkpoint. A 24-parameter mirrored
evolution strategy adjusted only shared biases and signed incoming-edge gains at the eight
existing motor pools, then folded those values into the graph. On 1,024 fresh paired
flights it raised success from 43.0% to **49.5%** (paired 95% CI for the gain: 4.9–8.2
percentage points), improved both lateral sides, and reduced mean gate-plane radial error
from 0.748 m to 0.663 m. Frozen vision scored 4.9%. Constant-1g acceleration scored 50.0%,
however, so this is improved static native calibration—not learned mass inference—and it
still falls well short of the 90% goal. See
[`gate-motor-interface-es-v1`](artifacts/gate-motor-interface-es-v1/).

A follow-up rollout search exposed all 282 signed edges on existing acceleration-to-
throttle paths of at most four hops while keeping every bias, time constant, and other
edge frozen. On 1,024 fresh matched flights it improved light-mass success by 8.0 points
(paired 95% CI 5.7–10.4), kept heavy-mass performance within one point, and raised overall
success from 48.2% to 51.8%. It missed the preregistered ten-point light-mass threshold,
however, and swapping acceleration traces within each matched mass pair caused no net
success change. The candidate was therefore rejected: it neither replaces the 49.5%
promoted controller nor demonstrates mass-specific acceleration adaptation. See the
[`recurrent acceleration-path diagnostic`](artifacts/gate-acceleration-path-diagnostic-v1/).

A recurrent-PPO follow-up then trained the 198 existing inputs and 26 biases at the native
front-leg motor interface while carrying the connectome's own state through every flight.
The simulator and privileged critic were training-only; the actor gained no history buffer
or extra recurrent network. Numerical replay and gradient audits passed, but the run
stopped at iteration 10: its best light-mass gains traded away heavy-mass performance. On
1,024 new matched flights, the selected safe snapshot was essentially flat overall
(47.7% to 47.6%), with light success up 1.6 points and heavy success down 1.8. Constant-1g
and pair-swapped controls again ruled out an acceleration-adaptation claim, so no checkpoint
was promoted. See the
[`recurrent-PPO diagnostic`](artifacts/gate-recurrent-ppo-diagnostic-v1/).

A full-network follow-up then distilled the verified mass-aware native oracle into all
4,469 edge magnitudes, 1,138 neuron biases, and 1,138 native time constants. The oracle
only labeled current-student histories; it never drove the student's plant, and no mass,
timer, history stack, or added recurrent module entered the actor. The optimization was
numerically sound but learned a shared throttle offset rather than accelerometer-conditioned
state: light success rose from 3.9% to 50.8% while heavy success fell from 91.0% to 35.4%.
Constant-1g and matched-pair acceleration controls were indistinguishable from live input,
so the candidate was rejected. See the
[`full-network oracle-distillation diagnostic`](artifacts/gate-full-network-oracle-diagnostic-v1/).

A smaller representability audit isolated the failure. Training through the complete
75-step native recurrent prefix on eight exact light/heavy pairs reduced normalized
throttle-contrast error to 2.1% and pair-mean error to 4.6%; identical sensor prefixes
produced exactly identical outputs. Contrast error remained 5.7% on eight disjoint gate
geometries, although the untrained 0.5-second endpoint failed. Native recurrence can learn
the conditional action, so the next iteration targets multi-time, full-prefix credit
assignment rather than adding engineered history. See the
[`conditional-overfit diagnostic`](artifacts/gate-conditional-overfit-diagnostic-v1/).

The first multi-time run then stopped before update 1 because its spliced target—promoted
steering plus the older oracle's throttle—reached only 87.5% on a broader set of balanced
gate geometries, below the fixed 90% teacher threshold. Recalibrating privileged mass bias
for the promoted controller confirmed a throttle-only ceiling: 79.6% over 1,024 fresh
flights, with 100% success on negative offsets but only 59.2% on positive offsets. A
steering-aware analytical teacher was therefore selected instead. Its visual/accelerometer
variant with exact training-only mass achieved **100%** overall, on both mass halves, both
lateral sides, and both obliquity signs over 1,024 new flights after a 0.5-second promoted
prefix. These are teacher diagnostics, not fly-controlled results. See the
[`stopped multi-time preflight`](artifacts/gate-multitime-preflight-diagnostic-v1/),
[`promoted-oracle calibration`](artifacts/gate-promoted-oracle-diagnostic-v1/), and
[`analytical-teacher audit`](artifacts/gate-analytic-teacher-v1/).

The first full analytical distillation run exposed a sharper issue: it learned the
0.75-second mass-conditioned throttle contrast but not the 0.5-second contrast, where its
prediction remained essentially zero. It stopped at the fixed first-round fidelity gate,
without a promoted checkpoint. Because a mass-free visual/accelerometer reserve teacher
also flies the same diverse distribution perfectly, a new 1,024-flight preflight froze
that observation-compatible teacher at 100% in every declared stratum. The next run will
distill it without privileged mass labels; near-zero target contrasts are normalized on a
fixed 0.01 motor-command scale. See the
[`exact-mass distillation diagnostic`](artifacts/gate-multitime-analytic-exact-mass-diagnostic-v1/)
and [`mass-free teacher preflight`](artifacts/gate-analytic-teacher-reserve-v1/).

That mass-free run verified the diagnosis: its tiny 0.5-second throttle contrast was fit
to 2.8% of the fixed actuator scale. But sampling only one horizon per update caused
strong cross-time interference. Middle-horizon throttle means improved while the
5-second mean error grew to 4.24 normalized RMSE, and roll fidelity remained poor. It too
stopped after round one without flight evaluation or promotion. See the
[`sampled-horizon diagnostic`](artifacts/gate-multitime-reserve-sampled-diagnostic-v1/).

A bounded joint-horizon capacity audit then supervised all seven endpoints from each
complete recurrent replay and raised the steering weight. Its worst training margin fell
from 18.75 to 5.03 in 150 updates and disjoint holdout improved to 4.52, but it still
failed the fixed fit gate. This audit had inherited the earlier exact-mass overfit vector,
whose wrong-sign 0.75-second reserve target made it a poor initialization; it was never a
promoted controller. The result and rejected vector are preserved in the
[`joint multi-time diagnostic`](artifacts/gate-joint-multitime-overfit-diagnostic-v1/).

Repeating that audit from the promoted source controller—while holding every other input,
including the regularization reference, fixed—improved the initial margin from 18.75 to
5.00. After 150 updates it reached 4.37 on training and 4.05 on disjoint holdout, but
1--5-second contrast and roll fidelity still failed. Exact seven-endpoint imitation is
therefore not being extended post hoc; the controlled record is in the
[`source-initialized joint diagnostic`](artifacts/gate-joint-multitime-source-init-diagnostic-v1/).

Dense successful-trajectory imitation removed the sparse-endpoint objective. Every native
parameter received 1-second truncated recurrent gradients from 64 perfect mass-free
teacher flights, followed by current-student DAgger histories. FP32 collection replay and
directional finite differences passed, but the first run improved fixed-suite success only
from 19.9% to 21.1% and lost all light-mass successes. It stopped at update 200.

A controlled repeat corrected throttle's loss scale from the full hover-command RMS
(`0.388`) to the teacher-minus-source correction RMS (`0.0787`) and added fixed signed
throttle diagnostics by mass and time. The best eligible result was then the unchanged
source: update 50 reached 19.5%, update 100 fell to 11.3%, and mixed replay later produced
only misses. Correcting action weighting alone therefore did not make offline teacher
imitation improve closed-loop flight. Neither run was promoted. See the
[`dense DAgger v1`](artifacts/gate-dense-dagger-diagnostic-v1/) and
[`corrected-weight v2`](artifacts/gate-dense-dagger-diagnostic-v2/) diagnostics.

A frozen 2×4 axis-takeover audit then separated the causal roles. On 512 new cases,
replacing only throttle after 0.5 seconds raised the source from 19.9% to 56.6%; replacing
only roll/pitch/yaw reached 36.9% and still failed every light-mass case. Full mass-free
reserve takeover achieved 100% in every stratum. The rejected v2 update-200 prefix was
also fully recoverable by the complete takeover. Thus the remaining failure requires
coupled steering and throttle learning, while the first half-second is not the obstacle.
See the [`axis-takeover factorial diagnostic`](artifacts/gate-axis-takeover-factorial-v1/).

A teacher-assisted complete-flight evolution strategy then tested the existing 24-value
motor readout directly. With teacher throttle, the 18 steering gains/biases raised fixed
validation success from 53.9% to 75.4% and the worst mass/lateral stratum from 10.2% to
53.1%. With teacher steering, however, all six-parameter throttle candidates retained
zero light-mass success; the best safe result moved only from 35.9% to 39.1%. The fixed
gate stopped the run before native merge. This narrows the next search to recurrent
acceleration-to-throttle circuitry rather than a wider static output trim. See the
[`assisted motor-interface diagnostic`](artifacts/gate-assisted-motor-es-diagnostic-v1/).

That recurrent search then varied all 282 signed edge magnitudes on four-hop-or-shorter
accelerometer-to-throttle paths while a teacher supplied only steering. Fixed validation
improved from 39.8% to 44.1%, but the gain was again entirely heavy-mass: light-mass
success stayed at zero and early throttle remained nearly mass-invariant. The
preregistered generation-20 gate stopped the run before its causal final, and no weights
were merged or promoted. See the
[`assisted acceleration-path diagnostic`](artifacts/gate-assisted-acceleration-path-es-diagnostic-v1/).

A frozen-history routing audit then separated latent information from the native throttle
mapping. A linear probe of all 93 path neurons predicted the required throttle correction
very accurately, and a probe restricted to the 19 return sources plus seven throttle
motor states narrowly passed every held-out time/mass gate. The exact 37-edge fixed-sign
native mapping did not pass. Acceleration ablation strongly changed the wider path probe
but barely changed the return-source prediction, so useful information reaches the output
boundary without establishing that acceleration is its carrier there. See the
[`throttle-routing diagnostic`](artifacts/gate-throttle-routing-diagnostic-v1/).

A paired FP64 trust-region fit then tested whether the failed native readout was merely
under-optimized. Both legal fixed-sign starts fully converged to the same solution, but
still missed the held-out thresholds (worst-group NRMSE 0.366; 46.8% improvement over a
constant). Allowing the same edges to reverse sign for diagnosis only passed at 0.235 and
57.4%, reversing 15 of 37 signs. This implicates the fixed signs of that final readout and
redirects work toward a different legal anatomical route; no sign-relaxed weights can be
deployed. See the
[`readout-constraint diagnostic`](artifacts/gate-throttle-readout-constraint-diagnostic-v1/).

[`showcase.mp4`](artifacts/gate-v1/showcase.mp4) records one complete traversal and shows
both schematic forelegs moving the virtual transmitter sticks. Neither the gate nor hover
checkpoint has yet been transferred to an independent simulator.

## Implementation roadmap

1. Ingest and profile the Feather graph; join connection endpoints to annotations and
   neurotransmitter predictions without modifying raw counts.
2. Define fixed mappings from FPV pixels and inertial values into identified sensory
   neurons, and from identified front-leg motor neurons into simulated joints.
3. Build a headless one-gate environment and a privileged-state teacher used only for
   training; an episode supervisor owns arm/disarm.
4. Tether a biomechanical fly over a virtual Mode-2 transmitter and make measured
   fore-tarsus/stick motion—not a separate learned decoder—the quadcopter action.
5. Train explicitly selected visual-to-motor MaleCNS subgraphs and compare them with
   matched generic and rewired baselines.
6. Once the CPU loop is correct, batch physics and FPV rendering with MJWarp; then expand
   the graph, add multi-gate role changes, AirSim visual evaluation, and validated
   Betaflight SITL.

## Data, artifacts, and licences

- Project source code is licensed under [GNU GPL v2](LICENSE).
- The official MaleCNS dataset is licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Any redistribution or
  derivative must credit the dataset creators, link the licence, and identify changes;
  see [`data/README.md`](data/README.md) for the canonical source and attribution.
- The large raw Feather tables are deliberately excluded from Git. The fetch script
  downloads them from the official release and verifies their recorded checksums.
- The five mirrored research papers are CC BY 4.0 and retain their authors’ copyright.
  They are tracked through Git LFS; their citations and official records are listed in
  [`papers/README.md`](papers/README.md).
- Small reports, derived connectome subgraphs, trained checkpoints, trajectories, and
  videos under `artifacts/` are outputs of this project. They are **not** official
  Google/Janelia neural-network weights or unmodified source data.

## Primary sources

- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [HHMI Janelia MaleCNS project](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [MaleCNS v1.0 downloads and licence](https://male-cns.janelia.org/download/)
- [Peer-reviewed MaleCNS paper](https://doi.org/10.1016/j.cell.2026.08.015)
- [Femoral chordotonal proprioception circuits](https://doi.org/10.1038/s41467-025-59302-3)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
