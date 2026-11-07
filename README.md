# Lumina CV

Low-light image enhancement with a Noise- and Temporal-Aware Zero-DCE model, a FastAPI
server and a Next.js demo.

| Workspace | Responsibility | State |
| --- | --- | --- |
| [core](core/README.md) | PyTorch model, losses, data, flow, training, inference CLIs, HTTP API | Implemented and tested (about 2,060 tests); no trained weights or benchmark results are shipped |
| [web](web/README.md) | Next.js 16 demo: upload, enhance, before/after slider | Implemented and tested |

```text
Lumina-CV/
  core/                     Independent Python package nta_zerodce
    src/nta_zerodce/         models, losses, data, flow, metrics, training, inference, api
    configs/ docs/ tests/ scripts/
  web/                      Next.js + React + TypeScript
    app/                     page and the runtime /api/v1 proxy route handler
    components/ hooks/ lib/ tests/
```

## Try it end to end

Needs Python 3.10+ and Node.js 22. No dataset is required: the smoke run trains a tiny model
on procedurally generated images, so the output demonstrates the pipeline, not image quality.

```sh
# 1. Train a tiny model on synthetic data and enhance a few images (CPU, a few minutes)
cd core
python -m pip install -e ".[dev]"
python scripts/make_synthetic_data.py --output .smoke/data/train --count 32 --size 96 --seed 1
python scripts/make_synthetic_data.py --output .smoke/data/test --count 4 --size 96 --seed 2
python scripts/train.py --config configs/smoke.py --run-name smoke
python scripts/enhance.py --checkpoint .smoke/checkpoints/smoke/final.pt --input .smoke/data/test --output .smoke/out --device cpu      # writes .smoke/checkpoints/smoke/final.pt

# 2. Serve the model
LUMINA_CHECKPOINT=.smoke/checkpoints/smoke/final.pt python -m nta_zerodce.api   # http://127.0.0.1:8000

# 3. Open the demo (second terminal)
cd ../web
npm ci
npm run dev                              # http://localhost:3000
```

The page proxies `/api/v1/*` to the server named by `LUMINA_API_URL` (default
`http://127.0.0.1:8000`), read at request time. To use a real model, train with
`core/configs/full.py` on your own data (see [core README](core/README.md)) and point
`LUMINA_CHECKPOINT` at the resulting `final.pt`.

## Checks

```sh
cd core && python -m ruff check . && python -m ruff format --check . && python -m mypy && python -m pytest
cd web  && npm run typecheck && npm run lint && npm test && npm run build
```

An API container image is built from [core/Dockerfile](core/Dockerfile); see
[API docs](core/docs/API.md) for configuration, limits and deployment notes.

## Separation

The web workspace owns the browser UI and never imports Python or runs enhancement. The core
owns all image processing and exposes it through the documented HTTP contract
([API](core/docs/API.md)); the two are developed, installed and tested independently.
