# nta-zerodce

Noise- and Temporal-Aware Zero-DCE: low-light enhancement for photos, hostel-room
videos, classroom recordings, and webcam streams, with training code, inference CLIs
and an HTTP API.

This is the independent Python CV/backend workspace in [Lumina CV](../README.md).
The Next.js demo lives in [web](../web/README.md). The Python package is nta_zerodce;
the repository folder is core/.

## Thirty-second pitch

Estimate per-pixel curves instead of a high-light reference image, following
[Zero-DCE](docs/REFERENCES.md#r1). Add five switchable project extensions to
study noise amplification, edge/colour distortion, bright-source overexposure,
and video flicker. Estimate curves on a smaller frame for the eventual webcam
demo. Everything described here is implemented and tested on CPU with synthetic data.
No model has been trained on real datasets in this repository, so no quality, speed or
robustness claim is made (see Honest limitations).

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

Python 3.10+; CPU is sufficient for all checks and the smoke run. On Colab/Kaggle, retain
the platform's compatible torch build; use `--device cuda` for real training.

```sh
cd core
python -m venv .venv
# POSIX: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -e ".[dev]"
python -m ruff check . && python -m ruff format --check .
python -m mypy
python -m pytest
```

No dataset/checkpoint is installed.

## Quickstart

Without any dataset (synthetic images, three CPU epochs, then enhancement of the test images):

```sh
python scripts/make_synthetic_data.py --output .smoke/data/train --count 32 --size 96 --seed 1
python scripts/make_synthetic_data.py --output .smoke/data/test --count 4 --size 96 --seed 2
python scripts/train.py --config configs/smoke.py --run-name smoke
python scripts/enhance.py --checkpoint .smoke/checkpoints/smoke/final.pt --input .smoke/data/test --output .smoke/out --device cpu
```

With real data (see [DATA](docs/DATA.md) for the expected layout):

```sh
python scripts/train.py --help
python scripts/train.py --config configs/baseline.py
python scripts/train.py --config configs/full.py --noise --denoise --edge --chroma --adaptive-exp --temporal --video-dir data/videos/train
python scripts/ablate.py --matrix configs/ablation_matrix.py --output runs/ablations
python scripts/evaluate.py --config configs/full.py --checkpoint checkpoints/full.pt --output runs/eval.csv
python scripts/enhance.py --config configs/full.py --checkpoint checkpoints/full.pt --input data/unpaired --output runs/enhanced --curve-scale 0.5
python scripts/enhance.py --config configs/full.py --checkpoint checkpoints/full.pt --webcam 0 --light --curve-scale 0.5
python scripts/make_synthetic_data.py --config configs/full.py --input data/SICE/part1 --output data/synthetic
```

Serve a checkpoint over HTTP (details, limits and deployment in [API](docs/API.md)):

```sh
LUMINA_CHECKPOINT=checkpoints/full.pt python -m nta_zerodce.api   
curl -F file=@dark.png http://127.0.0.1:8000/api/v1/enhance -o enhanced.png
```

Use --no-noise / --no-edge etc. for explicit false overrides. Absent flags preserve
YAML. Light must match the checkpoint; do not switch it on for a baseline checkpoint.
To use real pairs later set temporal.pair_source to real in YAML; --video-dir
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
| [inference](src/nta_zerodce/inference/) | Image/video/webcam enhancement and IO |
| [api](src/nta_zerodce/api/) | FastAPI server ([API](docs/API.md)) |
| [metrics](src/nta_zerodce/metrics/) | Fidelity, noise, colour, clipping and temporal metrics |
| [training](src/nta_zerodce/training/) | Schema, gated loop, checkpoint and CSV contracts |
| [utils](src/nta_zerodce/utils/) | Seed/device/registry contracts |
| [scripts](scripts/) | Train, ablate, evaluate, enhance and synthetic-data CLIs |
| [tests](tests/) | Unit, integration and end-to-end tests ([TESTING](docs/TESTING.md)) |

## Status badges table

Plain text status labels deliberately replace coloured badges.

| Area | Status | Evidence |
| --- | --- | --- |
| Docs and signatures | PRESENT | [ARCHITECTURE](docs/ARCHITECTURE.md), [METHOD](docs/METHOD.md), [INTERFACES](docs/INTERFACES.md) |
| Packaging, lint, static types, CLI help | CHECKED | [pyproject](pyproject.toml) |
| Models, losses, data, flow, metrics, training, inference, API | IMPLEMENTED, TESTED | [tests](tests/), `python -m pytest` |
| Local doc links, config keys, annotation audit | ACTIVE | [schema tests](tests/test_config_schema.py) |
| Training on real datasets, ablation results | NOT RUN | needs SICE/LOL data and GPU time ([ROADMAP](docs/ROADMAP.md) M6) |
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
acquisition; [TESTING](docs/TESTING.md) lists what is verified;
[ROADMAP](docs/ROADMAP.md) defines M0..M6; [CONTRIBUTING](docs/CONTRIBUTING.md)
defines how to add an extension. [GLOSSARY](docs/GLOSSARY.md) defines terms.

## References

Cite the original paper and applicable datasets from [REFERENCES](docs/REFERENCES.md). The Zero-DCE
reference is CC BY-NC 4.0 and has not been copied. Data licences remain separate.
