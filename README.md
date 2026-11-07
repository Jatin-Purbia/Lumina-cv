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

**Step 2. Activate it** (conda `base` users can skip this step and the `python -m venv` line, but still `cd core`)

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

**Step 4. Start the server**

```powershell
$env:LUMINA_CHECKPOINT = "C:/Desktop/Lumina-CV/core/checkpoints/full/final.pt"
python -m nta_zerodce.api
```

A demo checkpoint trained on synthetic low-light images (40 epochs, CPU) is at that path on this machine; it is not stored in git and is not a quality model. Train your own as described under "Train a model" below. On macOS/Linux use
`export LUMINA_CHECKPOINT=core/checkpoints/full/final.pt` instead of the first line.
The API runs at http://127.0.0.1:8000 (stop with Ctrl+C).

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
python scripts/train.py --config configs/full.py --run-name full     # writes checkpoints/full/final.pt
```

Configs live in `core/configs/` as Python files. See [core/README.md](core/README.md) for data
layout and options, and [core/docs/ARCHITECTURE.md](core/docs/ARCHITECTURE.md) for the API and Docker image
([core/Dockerfile](core/Dockerfile)).

## Tests

```sh
cd core && python -m pytest
cd web  && npm test
```
