# nta-zerodce

Noise- and Temporal-Aware Zero-DCE: low-light enhancement for images, video and webcam streams,
with training code, command-line tools and an HTTP API. Part of [Lumina CV](../README.md); the
web app is in [web](../web/README.md).

The model estimates per-pixel brightness curves instead of needing a bright reference image,
and adds five switchable extensions:

| Flag | CLI | Purpose |
| --- | --- | --- |
| `noise_aware`, `denoise` | `--noise`, `--denoise` | noise estimation, optional residual denoiser |
| `edge` | `--edge` | reliable edge preservation |
| `chroma` | `--chroma` | local colour consistency |
| `adaptive_exp` | `--adaptive-exp` | adaptive exposure, highlight protection |
| `temporal` | `--temporal` | flow-based temporal consistency for video |

`--light` selects a lighter network and `--curve-scale` lowers the estimation resolution; neither
is an extension.

## Install

Python 3.10+. Use `--device cuda` for real training.

```sh
cd core
python -m pip install -e ".[dev]"
python -m pytest
```

## Usage

```sh
python scripts/train.py --config configs/photo.py --run-name photo_target06
python scripts/ablate.py --matrix configs/ablation_matrix.py --output runs/ablations
python scripts/evaluate.py --checkpoint checkpoints/lol/final.pt --output runs/eval.csv
python scripts/enhance.py --checkpoint checkpoints/lol/final.pt --input data/unpaired --output runs/enhanced
python scripts/benchmark_inference.py --checkpoint checkpoints/lol/final.pt --input ../public --output runs/inference_benchmark.json
LUMINA_CHECKPOINT=checkpoints/lol/final.pt python -m nta_zerodce.api
```

Run any script with `--help` for all options.

## Layout

| Path | Content |
| --- | --- |
| `src/nta_zerodce/` | models, losses, data, flow, metrics, training, inference, api |
| `configs/` | `photo.py` (10-epoch photo experiment), `baseline.py`, `full.py`, `ablation_matrix.py` |
| `scripts/` | train, evaluate, enhance, ablate, make_synthetic_data, benchmark_inference, prepare_photo_validation, check_photo_regressions |
| `tests/` | unit, integration and end-to-end tests |
| `docs/` | [ARCHITECTURE.md](docs/ARCHITECTURE.md): design, math, API, data, implementation, training, and measured results |

Trained weights are included under `checkpoints/` (see [checkpoints/README.md](checkpoints/README.md)
for what each one was trained on and its known failure modes). The extension ablation matrix
has not been run. See [photo training](docs/ARCHITECTURE.md#still-photo-training-and-checkpoint-promotion) for
checkpoint promotion, and [improvement notes](docs/ARCHITECTURE.md#settings-for-natural-looking-photos) for measured inference tradeoffs.
