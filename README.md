# Lumina CV

Low-light image enhancement with a Noise- and Temporal-Aware Zero-DCE model, a FastAPI
server and a Next.js web app.

```text
core/   Python package (PyTorch model, training, inference, HTTP API) - see core/README.md
web/    Next.js app: upload an image, enhance it, compare before/after - see web/README.md
```

## Run locally

Needs Python 3.10+ and Node.js 22.

```sh
# 1. API server (set LUMINA_CHECKPOINT to a trained final.pt)
cd core
python -m pip install -e ".[dev]"
LUMINA_CHECKPOINT=path/to/final.pt python -m nta_zerodce.api   # http://127.0.0.1:8000

# 2. Web app (second terminal)
cd web
npm ci
npm run dev                                                    # http://localhost:3000
```

The web app proxies `/api/v1/*` to the server in `LUMINA_API_URL` (default
`http://127.0.0.1:8000`).

## Train a model

```sh
cd core
python scripts/train.py --config configs/full.py
```

Configs live in `core/configs/` as Python files. See [core/README.md](core/README.md) for data
layout and options, and [core/docs/API.md](core/docs/API.md) for the API and Docker image
([core/Dockerfile](core/Dockerfile)).

## Tests

```sh
cd core && python -m pytest
cd web  && npm test
```
