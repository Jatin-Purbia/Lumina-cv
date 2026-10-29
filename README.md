# Lumina CV

Noise- and Temporal-Aware Zero-DCE for low-light images and video, with independent Python and web workspaces.

| Workspace | Responsibility | Status |
| --- | --- | --- |
| [core](core/README.md) | Models, losses, data, optical flow, training, evaluation and inference | Implemented; covered by CPU numerical and local workflow tests |
| [web](web/README.md) | Future Next.js demo | Intentionally blank App Router shell |

The core implements the baseline and the five documented extensions: noise/denoise, edge preservation, chromaticity, adaptive exposure/highlights, and temporal consistency. The light estimator and curve-resolution option are also available. No HTTP API or web integration is specified or included.

## Core

Use Python 3.10+ and install from the independent workspace:

```sh
cd core
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
python scripts/train.py --help
```

Acquire and curate datasets manually using [DATA](core/docs/DATA.md), then edit the complete YAML paths in core/configs. Relative paths resolve against core/, including when a command is launched elsewhere. No dataset or trained weights are bundled.

```sh
python scripts/train.py --config configs/baseline.yaml
python scripts/enhance.py --config configs/baseline.yaml --checkpoint checkpoints/last.pt --input data/unpaired --output runs/enhanced
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/last.pt --output runs/evaluation.csv
```

See [core usage](core/README.md), [architecture](core/docs/ARCHITECTURE.md), [interfaces](core/docs/INTERFACES.md), and [implementation status](core/docs/IMPLEMENTATION.md).

## Web

Use Node.js 22 LTS:

```sh
cd web
npm ci
npm run dev
npm run typecheck
npm run build
```

The home page at http://localhost:3000 intentionally renders nothing. UI design, API selection and browser enhancement integration remain deferred as specified by the original scaffold.

## Research status

Executable code and synthetic smoke checks do not establish enhancement quality, baseline parity, noise calibration on real cameras, or real-time speed. The 13-row, three-seed research protocol, held-out comparisons and report require actual datasets and training; their results tables remain empty.
