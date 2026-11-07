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

The app needs a trained model (`final.pt`). Steps 1-3 create one; the server (step 4) cannot
enhance images without it. The server never downloads data or trains by itself.

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
python scripts/train.py --config configs/full.py --no-temporal --device cpu --epochs 60 --run-name lol
```

- Takes roughly an hour or more on a CPU (about 1.5 minutes per epoch on the machine this was
  developed on). With a GPU use `--device cuda` and more epochs (the config default is 100).
- `--no-temporal` skips the video-consistency part, which the photo app does not use and which
  doubles the training time. Drop it to train the full model.
- Output: `core/checkpoints/lol/final.pt` (plus `epoch_XXXX.pt` every 5 epochs), and logs in
  `core/runs/lol/` (`metrics.csv`, `val_metrics.csv`). Resume with
  `--resume checkpoints/lol/epoch_0030.pt`.
- Other options: `python scripts/train.py --help`. Configs are Python files in `core/configs/`.

## 4. Start the API server

In the same terminal (venv activated, inside `core`):

```powershell
$env:LUMINA_CHECKPOINT = "checkpoints/lol/final.pt"
python -m nta_zerodce.api
```

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
divider to compare. Photos larger than about 2 megapixels are shrunk in the browser before upload;
the API limits are listed in [core/docs/ARCHITECTURE.md](core/docs/ARCHITECTURE.md).

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
