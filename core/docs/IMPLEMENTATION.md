# Implementation

What is implemented, how to run it, and its limits. Design and math are in
[ARCHITECTURE](ARCHITECTURE.md).

## Components

- **Models**: seven-conv DCE-Net and depthwise light variant, eight curve iterations,
  noise head, residual denoiser.
- **Losses**: four baseline losses plus edge, chroma, adaptive exposure, highlight, noise,
  invariance, fidelity and temporal, with detached targets and masks.
- **Data**: sorted image/video manifests, EXIF-aware RGB IO, shared crops, per-channel
  heteroscedastic noise, analytic affine pairs with flow and flicker.
- **Flow**: bidirectional DIS, differentiable pixel-coordinate warp, reliability masks,
  histogram scene-cut detection.
- **Training**: Adam, fixed scene split, loss CSV logging, atomic versioned checkpoints,
  optimizer/RNG resume, rejection of incompatible state.
- **Inference**: image, video and webcam enhancers; checkpoint loader.
- **Metrics**: PSNR, Gaussian-window SSIM, angular error, flat-region noise proxy, clipping,
  masked temporal error.
- **API**: FastAPI server with upload limits and JSON errors. **Web**: Next.js upload and
  before/after demo with a runtime API proxy.

## Commands

Run from `core/` after `python -m pip install -e ".[dev]"`.

```sh
python scripts/train.py --config configs/baseline.py
python scripts/train.py --config configs/full.py --resume checkpoints/full_s17/epoch_0010.pt
python scripts/ablate.py --matrix configs/ablation_matrix.py --output runs/ablations
python scripts/evaluate.py --config configs/full.py --checkpoint checkpoints/full.pt --output runs/eval.csv
python scripts/enhance.py --config configs/full.py --checkpoint checkpoints/full.pt --input data/unpaired --output runs/enhanced
python scripts/enhance.py --config configs/full.py --checkpoint checkpoints/full.pt --webcam 0 --light --curve-scale 0.5
python scripts/make_synthetic_data.py --input data/LOL/train_low --output data/synthetic
LUMINA_CHECKPOINT=checkpoints/full.pt python -m nta_zerodce.api
```

Every training flag has an explicit pair (`--noise/--no-noise`, `--edge/--no-edge`, ...); an
explicit `--no-x` always overrides the config. Run `--help` on any script for all options.

## Artifacts

- Training writes `config.yaml`, `split.json`, `provenance.json`, `metrics.csv`, `last.pt` and
  periodic epoch checkpoints. Resume requires the identical resolved config and unchanged
  dataset hashes.
- Checkpoint format v1 holds model, optimizer, resolved config, RNG state and provenance;
  unknown versions, extra keys and architecture mismatches are rejected.
- Evaluation CSVs keep per-sample IDs and mask coverage.
- Synthetic export stores preview PNGs, `pair.pt` (unquantized frames, flows, masks, affine
  matrix, flicker, noise coefficients) and a JSON source/config record.

## Testing

```sh
python -m pytest                      # about 2,060 tests, CPU only, no data or network
python -m ruff check . && python -m ruff format --check . && python -m mypy
cd ../web && npm test && npm run typecheck && npm run lint && npm run build
```

Tests build tiny synthetic images, videos and checkpoints in temp directories. They cover
numerical identities and gradients, curve and affine conventions, temporal state and resets,
checkpoint and RNG round trips, a 30-step baseline overfit, an end-to-end train, enhance,
evaluate cycle (CLI and API produce identical pixels), the API contract and error codes, and
config schema and ablation matrix validation. Known failures on Windows: path handling in
`test_resolve_path`, plus `test_malformed_multipart_is_invalid_image` and
`test_scripts_run_as_programs`; `tests/test_metrics.py` needs a NumPy build compatible with
the installed SciPy/OpenCV.

## Limits

- **Train briefly.** On LOL (`configs/full.py`, no temporal, CPU) the model over-brightens as
  training continues: held-out LOL PSNR rises (10.7 at epoch 10, 13.5 at 15, 16.6 at 40) but a
  moderately dark photo is blown out from epoch 15 and washed out by 25. The shipped recipe uses 10
  epochs. A more varied training set (adding normal-light and darkened copies) did not fix it.
  Fixing this properly needs loss/weight tuning (for example a lower exposure target) or a
  validation set of ordinary photos for early stopping; neither has been done.

- No trained weights, benchmark results or ablation results are included. No quality or
  real-time claim is made; the 30 fps / 720p budget (capture 3 ms, resize 2, estimation 10,
  DIS 8, warp and curves 4, denoise 3, output 3) is a design target to be measured.
- Loss weights and thresholds are untuned starting values. Synthetic noise is a Gaussian
  approximation; added-noise `sigma` is not a camera-noise label.
- The loader cannot infer exposure labels or permissions: curate low-light inputs and
  scene/session splits. Images smaller than the training crop fail. Training needs `H, W`
  divisible by 16; inference needs at least 16.
- Checkpoints use trusted-local PyTorch serialization; never load untrusted files. CUDA
  grid-sampling backward can be nondeterministic.
- The API has no authentication; add TLS, authentication and rate limiting in front of it.

## Metrics for evaluation

PSNR and SSIM (data range 1, 11x11 Gaussian window, sigma 1.5) and angular error on paired
LOL; noise proxy, clipping fraction and temporal error on unpaired and video sets. Report
mean and standard deviation over the three seeds, with held-out failure cases. Do not tune on
the LOL evaluation split.
