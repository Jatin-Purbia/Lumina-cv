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

Steps 1-3 are one-time setup; run them from the repo root.

**Step 1. Go to `core` and create a virtual environment**

```sh
cd core
python -m venv .venv
```

**Step 2. Activate it** (skip steps 1-2 if you use conda `base`)

```powershell
# Windows PowerShell (the leading .\ is required)
.\.venv\Scripts\Activate.ps1
# If it says scripts are disabled, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```sh
# macOS / Linux / Git Bash
source .venv/bin/activate        # Git Bash on Windows: source .venv/Scripts/activate
```

**Step 3. Install the package**

```sh
python -m pip install -e ".[dev]"
```

**Step 4. Start the server** (every time, with the venv activated)

Set the path to a trained checkpoint and start the server. It listens on
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
