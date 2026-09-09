# MaleCNS v1.0 data snapshot

This directory contains the compact, analysis-ready core of the complete male
*Drosophila melanogaster* central nervous system connectome released by HHMI Janelia,
Google Research, Cambridge, and collaborators.

Release: **MaleCNS v1.0**, 8 June 2026. The publication reports 166,691 annotated
neurons across brain, optic lobes, neck connective, and ventral nerve cord. The local
tables include all qualifying segments, so annotation/table row counts need not equal the
paper's neuron count.

## Locally downloaded tables

| File | Rows | Key columns | Upstream size |
| --- | ---: | --- | ---: |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 151,856,684 | `body_pre`, `body_post`, `weight` (synapse count) | 1,051,241,946 B |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 211,577 | `bodyId`, `status`, `type`, `class`, sides, cross-dataset IDs | 14,483,314 B |
| `body-neurotransmitters-male-cns-v1.0.feather` | 1,835,518 | `body`, transmitter labels and prediction confidences | 43,282,834 B |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 88,384,522 | `body`, `pre`, `post`, `status_fine`, `synweight` | 778,062,826 B |
| `male-cns-v1.0.neuroglancer.json` | — | Official scene and volume-source manifest | 60,253 B |

Files live under `raw/malecns-v1.0/` and are ignored by Git. Restore or verify them with
`../scripts/fetch-malecns-v1.0.sh` from the repository root.

The first hover graph is deterministically derived with
`scripts/build_hover_connectome.py`.  Working outputs under `derived/` are ignored; the
small exact graph used for the passing result, plus its raw-file hashes and selection
manifest, are committed under `artifacts/hover-v1/`.

These bulk tables intentionally include far more than the 166,691 curated neurons (for
example, unproofread or non-neuronal segments). Building the simulated neuron graph must
therefore be an explicit, audited join/filter—not an assumption that every table row is a
neuron or that the 151.9 million graph rows equal the paper's headline synapse count.

## Intentionally not mirrored yet

The point-level tables are reproducible but unnecessary for initial graph simulation:

- synapse points: 12.7 GB;
- synaptic partner locations: 6.8 GB;
- per-T-bar transmitter predictions: 2.7 GB;
- skeletons, meshes, segmentation, and EM volumes: from many GB to multi-terabyte scale.

Add these only for a defined spatial-analysis need. Their canonical bucket paths and
formats are documented on the official download page.

## Provenance and license

- Dataset/download documentation: https://male-cns.janelia.org/download/
- Project page: https://www.janelia.org/project-team/flyem/male-cns-connectome
- Neuroglancer scene: `gs://flyem-male-cns/v1.0/male-cns-v1.0.json`
- Paper: Berg et al., *Sexual dimorphism in the complete connectome of the Drosophila
  male central nervous system*, *Cell* (2026),
  https://doi.org/10.1016/j.cell.2026.08.015
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Credit the dataset creators, link the license, and indicate transformations in any derived
artifact. This repository does not alter the raw files.
