# Lumina CV

Low-light image enhancement with a Noise- and Temporal-Aware Zero-DCE model, a FastAPI
server and a Next.js web app.

```text
core/   Python package (PyTorch model, training, inference, HTTP API) - see core/README.md
web/    Next.js app: upload an image, enhance it, compare before/after - see web/README.md
```

## Run locally

Needs Python 3.10+ and Node.js 22.

### 1. Backend (API server)

One-time setup, from the repo root:

```sh
cd core
python -m venv .venv
python -m pip install -e ".[dev]"      # run after activating the venv (below)
```

Activate the virtual environment (skip if you use conda/base and already installed the package):

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux / Git Bash
source .venv/bin/activate        # Git Bash on Windows: source .venv/Scripts/activate
```

Start the server with the path to a trained checkpoint. The server listens on
http://127.0.0.1:8000 and keeps running until you press Ctrl+C.

```powershell
# Windows PowerShell
$env:LUMINA_CHECKPOINT = "path\to\final.pt"
python -m nta_zerodce.api
```

```sh
# macOS / Linux / Git Bash
LUMINA_CHECKPOINT=path/to/final.pt python -m nta_zerodce.api
```

Check it is up: open http://127.0.0.1:8000/api/v1/health (`"model_loaded": true` means the
checkpoint loaded). Without `LUMINA_CHECKPOINT` the server still starts, but image enhancement
answers 503.

### 2. Web app

In a second terminal, from the repo root:

```sh
cd web
npm ci
npm run dev                      # http://localhost:3000
```

The web app proxies `/api/v1/*` to the server in `LUMINA_API_URL` (default
`http://127.0.0.1:8000`).

## Train a model

```sh
cd core
python scripts/train.py --config configs/full.py
```

Configs live in `core/configs/` as Python files. See [core/README.md](core/README.md) for data
layout and options, and [core/docs/ARCHITECTURE.md](core/docs/ARCHITECTURE.md) for the API and Docker image
([core/Dockerfile](core/Dockerfile)).

## Tests

```sh
cd core && python -m pytest
cd web  && npm test
```
