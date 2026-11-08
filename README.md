# Lumina CV

Low-light image enhancement with a Noise- and Temporal-Aware Zero-DCE model, a FastAPI
server and a Next.js web app.

```text
core/   Python package (PyTorch model, training, inference, HTTP API) - see core/README.md
web/    Next.js app: upload an image, enhance it, compare before/after - see web/README.md
```

Requirements: Python 3.10+, Node.js 22. Commands below are for Windows PowerShell; on
macOS/Linux use `source .venv/bin/activate` to activate and `export NAME=value` instead of
`$env:NAME = "value"`.

The repo includes a trained model, `core/checkpoints/lol/final.pt` (about 1 MB, trained 10 epochs
on the LOL dataset), so you can skip steps 2 and 3 and go straight to step 4. Steps 2-3 download
the data and retrain it yourself. The server never downloads data or trains by itself. Only use
checkpoints you trust: they are loaded with PyTorch's pickle-based format.

## 1. Set up the backend (once)

```powershell
cd core
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The leading `.\` is required. If PowerShell says scripts are disabled, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Keep the venv activated for every
`python` command below.

## 2. Download the training data (once)

```powershell
python scripts/download_lol.py
```

This downloads the public [LOL dataset](https://daooshee.github.io/BMVC2018website/) (about 350 MB
from a HuggingFace mirror) into `core/data/LOL/`: 485 low-light training images and 15 held-out
low/normal pairs for evaluation. Check the dataset's terms before use (research and education);
it is not stored in git. Running the script again does nothing if the data is already there.

## 3. Train the model (once)

```powershell
python scripts/train.py --config configs/full.py --no-temporal --device cpu --epochs 10 --run-name lol
```

- Takes about 20 to 30 minutes on a CPU. Use `--device cuda` if you have a GPU.
- **Keep `--epochs` small (10).** On this data, longer training makes the model brighten
  everything too much: by epoch 15 ordinary photos start to show blown-out white patches, and by
  epoch 25 they wash out completely. Ten epochs gives natural, moderate enhancement. It
  under-brightens the very darkest scenes (the LOL test set) rather than ruining normal photos.
- `--no-temporal` skips the video-consistency part, which the photo app does not use and which
  doubles the training time. Drop it to train the full model.
- Output: `core/checkpoints/lol/final.pt`, plus logs in `core/runs/lol/` (`metrics.csv`,
  `val_metrics.csv`). Other options: `python scripts/train.py --help`. Configs are Python files in
  `core/configs/`.

## 4. Start the API server

In the same terminal (venv activated, inside `core`):

```powershell
copy .env.example .env
python -m nta_zerodce.api
```

`.env` (gitignored) lists every checkpoint in `checkpoints/` with one `LUMINA_CHECKPOINT=` line
active and the rest commented out. To serve a different model, move the `#` to another line and
restart the API. The default is `checkpoints/lol/final.pt`. A variable already set in your shell
(`$env:LUMINA_CHECKPOINT = "..."`) takes precedence over `.env`. Without any checkpoint the API
starts but `/enhance` answers 503.

The API runs at http://127.0.0.1:8000 until you press Ctrl+C. Check it:
http://127.0.0.1:8000/api/v1/health should show `"model_loaded": true`.

If it fails with "only one usage of each socket address", another program uses port 8000: set
a different port with `$env:LUMINA_PORT = "8001"` and use the same address in step 5.

## 5. Start the web app

In a second terminal, from the repo root:

```powershell
cd web
npm ci
npm run dev
```

If the API is not on port 8000, first run `$env:LUMINA_API_URL = "http://127.0.0.1:8001"`
(matching `LUMINA_PORT`). Open http://localhost:3000, choose a photo, press Enhance and drag the
divider to compare. Below the image, an **Analysis** section measures what changed (brightness,
shadows, contrast, noise, color, tonal range, clipping), draws a brightness histogram and compares
the result with plain brightening at the same overall brightness. It is computed in your browser,
so it needs no extra server. Photos larger than about 2 megapixels are shrunk in the browser before
upload; the API limits are listed in [core/docs/ARCHITECTURE.md](core/docs/ARCHITECTURE.md).

## Command-line tools (optional)

Run from `core` with the venv activated:

```powershell
# enhance a folder of images
python scripts/enhance.py --checkpoint checkpoints/lol/final.pt --input path\to\images --output runs\enhanced

# metrics on the held-out LOL pairs (PSNR, SSIM, angular error)
python scripts/evaluate.py --config configs/full.py --checkpoint checkpoints/lol/final.pt --output runs/eval.csv

# ablation matrix (13 variants x 3 seeds; list the runs without training)
python scripts/ablate.py --matrix configs/ablation_matrix.py --output runs/ablations --dry-run
```

## Tests

```powershell
cd core; python -m pytest        # about 2,060 tests, CPU, no data or network needed
cd ..\web; npm test
```

More detail on the design, API and limits: [core/docs/ARCHITECTURE.md](core/docs/ARCHITECTURE.md)
and [core/docs/IMPLEMENTATION.md](core/docs/IMPLEMENTATION.md).
