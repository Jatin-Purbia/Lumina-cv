# nta-zerodce

Noise- and Temporal-Aware Zero-DCE: a research implementation for low-light photos,
hostel-room videos, classroom recordings, and webcam enhancement.

This is the independent Python CV/backend workspace in [Lumina CV](../README.md).
The empty Next.js demo shell lives in [web](../web/README.md). Package name and
research interfaces remain nta_zerodce; the repository folder is now core/.

## Thirty-second pitch

Estimate per-pixel curves instead of a high-light reference image, following
[Zero-DCE](docs/REFERENCES.md#r1). Add five switchable project extensions to
study noise amplification, edge/colour distortion, bright-source overexposure,
and video flicker. Estimate curves on a smaller frame for the eventual webcam
demo. The documented model, loss, flow, data, training and inference contracts are implemented. CPU numerical tests and a generated-data training/checkpoint/enhancement/evaluation cycle are executable. Research validation remains pending; see [implementation status](docs/IMPLEMENTATION.md).

## Features

| Extension | Config flag / CLI | Main contracts |
| --- | --- | --- |
| Noise estimation and optional residual denoise | noise_aware / --noise; denoise / --denoise | [noise_branch](src/nta_zerodce/models/noise_branch.py), [denoiser](src/nta_zerodce/models/denoiser.py), [noise losses](src/nta_zerodce/losses/noise.py) |
| Reliable edge preservation | edge / --edge | [edge loss](src/nta_zerodce/losses/edge.py) |
| Local chromaticity consistency | chroma / --chroma | [chroma loss](src/nta_zerodce/losses/chroma.py) |
| Adaptive exposure and highlight protection | adaptive_exp / --adaptive-exp | [adaptive exposure](src/nta_zerodce/losses/adaptive_exposure.py), [highlight](src/nta_zerodce/losses/highlight.py) |
| Flow-based temporal training and curve blending | temporal / --temporal | [temporal loss](src/nta_zerodce/losses/temporal.py), [video enhancer](src/nta_zerodce/inference/video_enhancer.py) |

Denoise implies noise_aware. Edge/chroma/adaptive/temporal can each run alone;
sigma is then zero. The --light variant changes convolution type, while
--curve-scale controls inference estimator resolution; neither is a sixth extension.

## Install

Python 3.10+; CPU is sufficient for tests and small smoke runs. On Colab/Kaggle, retain the
platform's compatible torch build; select CUDA explicitly for training on a compatible installation.

```sh
cd core
python -m venv .venv
# POSIX: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e ".[dev]"
python -m pre_commit install
make lint typecheck test docs-check
```

GNU Make is required for make targets. On Windows without Make, run the same
recipes directly: python -m ruff check ., python -m ruff format --check .,
python -m mypy, python -m pytest, and python -m pytest --collect-only -q
tests/test_config_schema.py. make setup installs the editable package and dev
tools, including nbstripout. No dataset/checkpoint is installed.

## Quickstart contracts

All five commands execute. Training requires curated local images; enhancement and evaluation require a compatible trusted local checkpoint. Missing data and incompatible settings fail with explicit errors. Training writes checkpoints/last.pt by default; use separate run/checkpoint directories for each experiment. A nonempty metrics log requires --resume or a new run directory.

```sh
python scripts/train.py --help
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/full.yaml --noise --denoise --edge --chroma --adaptive-exp --temporal --video-dir data/videos/train
python scripts/ablate.py --matrix configs/ablation_matrix.yaml --output runs/ablations
python scripts/evaluate.py --config configs/full.yaml --checkpoint checkpoints/last.pt --output runs/eval.csv
python scripts/enhance.py --config configs/full.yaml --checkpoint checkpoints/last.pt --input data/unpaired --output runs/enhanced --curve-scale 0.5
python scripts/enhance.py --config configs/full.yaml --checkpoint checkpoints/last.pt --webcam 0 --light --curve-scale 0.5
python scripts/make_synthetic_data.py --config configs/full.yaml --input data/SICE/part1 --output data/synthetic
```

Use --no-noise / --no-edge etc. for explicit false overrides. Absent flags preserve
YAML. Light must match the checkpoint; do not switch it on for a baseline checkpoint.
To use real pairs set temporal.pair_source to real in YAML; --video-dir
sets only the directory. See [INTERFACES](docs/INTERFACES.md) for precedence.

## Repository map

| Path | Purpose |
| --- | --- |
| [docs](docs/) | Architecture, method, contracts, protocol and milestones |
| [configs](configs/) | Complete baseline/full, three-seed matrix, path example |
| [models](src/nta_zerodce/models/) | Estimators, curves, composition, factory |
| [losses](src/nta_zerodce/losses/) | Pure tensor objectives and registry contracts |
| [data](src/nta_zerodce/data/) | Datasets, affine/noise synthesis, preprocessing |
| [flow](src/nta_zerodce/flow/) | DIS estimation, differentiable warp, reliability masks |
| [inference](src/nta_zerodce/inference/) | Image/video/webcam contracts and IO |
| [metrics](src/nta_zerodce/metrics/) | Fidelity, noise, colour, clipping and temporal metrics |
| [training](src/nta_zerodce/training/) | Schema, gated loop, checkpoint and CSV contracts |
| [utils](src/nta_zerodce/utils/) | Seed/device/registry contracts |
| [scripts](scripts/) | Executable train/enhance/evaluate/ablation/synthetic commands |
| [tests](tests/) | Structural audits and executable numerical/workflow checks |
| [notebook](notebooks/colab_quickstart.ipynb) | Colab command outline, no execution/output |
| [report](report/outline.md) | Final report outline tied to docs/figures/tables |

## Status badges table

Plain text status labels deliberately replace coloured badges.

| Area | Status | Evidence |
| --- | --- | --- |
| Docs and signatures | PRESENT | [ARCHITECTURE](docs/ARCHITECTURE.md), [METHOD](docs/METHOD.md), [INTERFACES](docs/INTERFACES.md) |
| Packaging, lint, static types, CLI help | CHECKABLE | [pyproject](pyproject.toml), [CI](../.github/workflows/ci.yml) |
| Local doc links, complete config keys | ACTIVE | [schema tests](tests/test_config_schema.py), make docs-check |
| Config validation / registry / checkpoint IO | IMPLEMENTED | Typed validation, strict state compatibility, RNG and manifest hashes |
| Model/loss/data/flow/inference/metric/training behavior | IMPLEMENTED | [numerical and workflow tests](tests/test_runtime.py), [TESTING](docs/TESTING.md) |
| Trained weights, measured metrics, FPS | ABSENT | [EXPERIMENTS](docs/EXPERIMENTS.md), [REAL_TIME](docs/REAL_TIME.md) contain empty templates |

## Honest limitations

The baseline is a planned independent reproduction, not validated parity.
Zero-reference enhancement cannot recover absent photons or clipped detail.
Proposed synthetic noise in display RGB approximates camera processing poorly;
noise prediction may confuse texture with noise. Flow can fail in dark scenes,
disocclusions and flicker; blending can ghost or lag. Proxy metrics can reward
oversmoothing or dimming. No quality, robustness, speed or real-time result has
been established. See [EXPERIMENTS](docs/EXPERIMENTS.md) for controls and threats.

## Reading and contribution

[DATA](docs/DATA.md) and [download guide](scripts/download_data.md) define manual
acquisition; [TESTING](docs/TESTING.md) lists future assertions;
[ROADMAP](docs/ROADMAP.md) defines M0..M6; [CONTRIBUTING](docs/CONTRIBUTING.md)
defines how to add an extension. [GLOSSARY](docs/GLOSSARY.md) defines terms.

## Licence and citation

Our original work is [MIT](LICENSE). The Zero-DCE reference is CC BY-NC 4.0 and
has not been copied; read [NOTICE](NOTICE.md) before importing any code/assets.
Data licences remain separate. Use [CITATION.cff](CITATION.cff) for this project
and cite the original paper and applicable datasets from [REFERENCES](docs/REFERENCES.md).

## Runtime details

Unpaired training never reads paired high-light targets. Image training holds out a deterministic 10% of scene IDs with split seed 17, independent of the model seed. Curate low-light exposures before training: the loader cannot identify enhanced composites automatically. Flat image roots use filenames as scene IDs; use scene directories for multiple exposures. Images smaller than the configured crop fail explicitly. Evaluation preserves original size; inference requires H,W >=16.

Runs save resolved YAML, split/provenance JSON, raw and weighted loss CSV, gradient norms and atomic checkpoints. --resume requires the same resolved configuration and dataset content; continuation starts after the completed checkpoint epoch. Inference can change curve_scale while keeping checkpoint architecture and feature flags compatible. Only load checkpoints you trust.

For explicit unpaired or video evaluation, pass --input to evaluate.py. A video input is a supported video file or a directory containing videos and no images. Without --input, evaluation requires paired low/high directories and also processes the unpaired directory if present. CSV records include sample IDs, mask coverage and input clipping. Videos preserve source FPS and omit audio. Webcam requires an available camera and GUI; these hardware paths have not been exercised here.

The ablation command validates all 13 documented flag combinations before running each configured seed and writing independent training/evaluation outputs. It is a real training schedule, not a dry-run command. Synthetic export stores PNG previews, unquantized tensor pairs, analytic flows/masks and sampled affine/noise/flicker metadata in pair.pt, plus source hashes and resolved settings.
