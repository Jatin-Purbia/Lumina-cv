# Architecture

This is the single documentation guide for architecture, implementation, training, and measured settings.

- [System and model](#system)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Implementation and limits](#implementation)
- [Photo training and checkpoint promotion](#still-photo-training-and-checkpoint-promotion)
- [Measured settings and experiments](#settings-for-natural-looking-photos)

Technical overview of the `nta_zerodce` package (`core/`) and how the web app connects to it.
The method is a Noise- and Temporal-Aware extension of Zero-DCE: a small CNN predicts
per-pixel curve parameters that are applied iteratively to the input image, trained without
paired ground truth.

## System

```text
browser --> web (Next.js, /api/v1/* proxy) --> core API (FastAPI) --> Enhancer --> model
                                                core CLIs (train / enhance / evaluate / ablate)
```

- `web/` owns the UI only. It never runs enhancement; it proxies `/api/v1/*` to
  `LUMINA_API_URL` at request time.
- `core/` owns all image processing: model, losses, data, optical flow, metrics, training,
  inference, HTTP API.

## Model

Conventions: RGB `float32`, `[B,3,H,W]`, values in `[0,1]`. `Y(x) = mean(R,G,B)`.

```text
x --> NoiseBranch --> sigma [B,1,H,W]            (noise_aware, else sigma = 0)
concat(x, sigma) --> DCE-Net --> A [B,24,H,W]    (eight RGB curve maps, tanh, in [-1,1])
z = x_8,  x_{i+1} = x_i + A_i * x_i * (1 - x_i)  (i = 0..7)
o = clamp(z - 0.1*tanh(Denoiser(concat(z, sigma*G))), 0, 1)   (denoise, else o = z)
```

- **DCE-Net**: seven 3x3 convolutions, 32 channels, ReLU, symmetric skip concatenations
  (h3+h4, h2+h5, h1+h6), tanh head. No normalization or downsampling. The `light` variant uses
  depthwise-separable hidden layers and `curve_scale` to estimate curves at reduced
  resolution; curves are bilinearly upsampled and applied at full resolution.
- **Curve**: endpoints 0 and 1 are fixed; `A = 0` is identity.
- **NoiseBranch**: 3 conv layers (RGB-16-16-1), sigmoid output scaled to `[0, 0.2]`.
- **Denoiser**: two depthwise/pointwise 16-channel blocks and an RGB residual head,
  conditioned on `sigma * G`, where `G` is a detached local gain
  `clamp(avgpool5(Y(z)) / avgpool5(Y(x)), 1, 8)`.

## Losses

Baseline (Zero-DCE reference-style reductions), weights in brackets:

| Loss | Definition | Weight |
| --- | --- | --- |
| spa | squared change of neighbouring 4x4-pooled contrast between input and output | 1 |
| exp | mean `(P16(Y(o)) - 0.6)^2` | 10 |
| col | `sqrt(d_RG^4 + d_RB^4 + d_GB^4)` of channel-mean differences | 5 |
| tv | squared finite differences of `A` | 200 |

Extensions, each behind a feature flag:

| Flag | Loss | Idea | Weight |
| --- | --- | --- | --- |
| `noise_aware` | noise, invariance | Synthetic Poisson-Gaussian-style noise (`var = a*x + b^2`) gives a known `sigma` target; the noisy output must match a no-grad clean-input teacher | 20, 10 |
| `denoise` | fidelity | Noisy output stays close to the teacher's raw (pre-denoise) output | 1 |
| `edge` | edge | Sobel gradient of output follows gain-scaled input gradient, weighted by `clamp((d_in - 2*sigma)/d_in, 0, 1)` | 20 |
| `chroma` | chroma | Output chromaticity `c = v/sum(v)` follows locally averaged input chromaticity | 20 |
| `adaptive_exp` | adaptive_exp, highlight | Replaces `exp` with a per-region target that rises for bright and falls for noisy regions; penalizes brightening already-bright pixels | 10, 20 |
| `temporal` | temporal | L1 between output and warped previous output on pixels where flow is reliable | 10 |

Targets, masks, `sigma` and gain used inside losses are detached. Total loss is the weighted
sum of enabled terms; Adam, coupled weight decay, gradient clipping.

## Temporal processing

- **Flow**: OpenCV DIS, bidirectional. Reliability mask = forward-backward consistency AND
  photometric agreement AND in-bounds. Training pairs are real video frames or analytic affine
  pairs with known flow and exposure flicker.
- **Inference** (`VideoEnhancer`): keeps previous RGB, smoothed curves and output. A scene cut
  (half-L1 distance of 32-bin histograms > 0.3), first frame or size change resets state.
  Otherwise `A_bar = (1 - 0.7*m) * A_t + 0.7*m * warp(A_bar_prev)`, then curves are applied to
  the current full-resolution frame. Optional output blending is off by default.
- A still-image forward never uses flow or state.

## Package layout

| Module | Responsibility | May depend on |
| --- | --- | --- |
| `models` | DCE-Net, noise branch, denoiser, curve application, factory | torch |
| `losses` | pure tensor loss functions and registry | torch, flow warp |
| `data` | image/video datasets, crops, noise synthesis, synthetic pairs | io, flow |
| `flow` | DIS estimator, differentiable warp, reliability masks | torch, OpenCV (estimator only) |
| `inference` | image/video/webcam enhancers, image IO, checkpoint loader | models, flow |
| `metrics` | PSNR, SSIM, angular error, noise proxy, clipping, temporal error | torch |
| `training` | config schema, trainer, loop step, checkpoints, ablation, evaluation | all of the above |
| `api` | FastAPI app, settings, image decode/encode, error mapping | inference |
| `cli`, `utils` | shared argparse layer; seed, device, registry | stdlib |

Rules: models never import data, trainer, optimizer or losses; losses never read files, infer
flow or move devices; only `flow/estimator` calls OpenCV optical flow; scripts contain
argument parsing only.

## Configuration

Experiments are plain Python files in `configs/` that define a `CONFIG` dict (`baseline.py`,
`full.py`) or a `MATRIX` dict (`ablation_matrix.py`). A YAML file is also accepted by
`--config`. The schema (`training/config.py`) is strict: unknown or missing keys fail, flags
have dependency checks (for example `denoise` requires `noise_aware`), and precedence is
defaults < config file < explicit CLI flags. Each run saves its resolved `config.yaml`.

Ablation rows (`ablation_matrix.py`, three seeds 17/29/43, 13 rows): `baseline`, five
single-extension rows, `noise_denoise`, `full`, and five leave-one-out rows.

## HTTP API

Base path `/api/v1`. No authentication: run behind the web proxy or a reverse proxy.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | `{status, model_loaded, device, version}` |
| `GET /options` | resolved flags, default `curve_scale`, limits, accepted types |
| `POST /enhance` | multipart `file` (PNG/JPEG) and optional `curve_scale` in (0, 1]; returns an 8-bit RGB PNG with `X-Lumina-Size` and `X-Lumina-Elapsed-Ms` |

Errors are always `{"error": {"code", "message"}}` with fixed messages: `invalid_image` 400,
`too_large` 413, `unsupported_type` 415, `bad_option` 422, `model_not_loaded` 503,
`server_busy` 503 (with `Retry-After: 1`),
`not_found` 404, `method_not_allowed` 405, `internal_error` 500. Every response carries
`X-Request-Id` and `Cache-Control: no-store`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LUMINA_CHECKPOINT` | unset | trusted checkpoint; unset starts without a model (`/enhance` answers 503) |
| `LUMINA_CONFIG` | unset | optional config; must match the checkpoint |
| `LUMINA_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |
| `LUMINA_MAX_UPLOAD_BYTES` | 20 MiB | compressed file limit |
| `LUMINA_MAX_PIXELS` | 2,097,152 | decoded pixel limit (memory is about 1.4 KB per pixel on CPU) |
| `LUMINA_MAX_SIDE` | 4096 | longest side |
| `LUMINA_DEFAULT_CURVE_SCALE` | from config | used when a request omits `curve_scale` |
| `LUMINA_MAX_CONCURRENCY` | 1 | simultaneous upload/decode/infer/encode jobs; excess requests receive `server_busy` before body parsing |
| `LUMINA_CORS_ORIGINS` | empty | comma-separated origins or `*` |
| `LUMINA_HOST`, `LUMINA_PORT` | `127.0.0.1`, `8000` | bind address |

The model loads once at startup; a bad checkpoint or config exits with code 3, a bad variable
with code 2. One process holds one model; extra uvicorn workers multiply memory. Uploads stay
in memory, and logs contain only method, path, status, time and request id. Admission is
bounded before multipart parsing so waiting uploads cannot accumulate in an unbounded queue.
Health/options requests do not consume enhancement slots. `Retry-After` is exposed through
CORS and forwarded by the web proxy. The API extras require Starlette 0.46+ for the parser's
`max_part_size` and `spool_max_size` interface.

## Data

Local layout (paths set in the `paths` section of the config, relative to `core/`):

```text
data/LOL/train_low/<scene>/low.png          training images (unpaired objective; python scripts/download_lol.py)
data/LOL/eval15/{low,high}/<pair>.png       paired evaluation only, never trained on
data/unpaired/{DICM,LIME,MEF,NPE}/...       qualitative evaluation
data/videos/{train,val,test}/<session>/<clip>.mp4
```

Split by scene/session, not by image, and keep LOL evaluation disjoint from training. Check
each dataset's terms before use; none are redistributed. Training stores split hashes and run provenance in the checkpoint payload;
the validation preparation script records image hashes in its manifest.

## Docker

`core/Dockerfile` builds the API image: installs the package, copies `src` and `configs`,
runs `python -m nta_zerodce.api`. Mount the checkpoint and set `LUMINA_CHECKPOINT`.

## References

1. Guo et al., Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement, CVPR 2020.
2. Li-Chongyi/Zero-DCE reference repository (not copied; license CC BY-NC 4.0).
3. Li, Guo, Loy, Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation
   (Zero-DCE++), TPAMI. The `light` variant is inspired by it, not a reproduction.
4. Cai, Gu, Zhang, Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images
   (SICE), TIP 2018.
5. Wei et al., Deep Retinex Decomposition for Low-Light Enhancement (LOL), BMVC 2018.
6. Kroeger et al., Fast Optical Flow using Dense Inverse Search (DIS), ECCV 2016.
7. Immerkaer, Fast Noise Variance Estimation, CVIU 1996.
8. Lai et al., Learning Blind Video Temporal Consistency, ECCV 2018.
9. Wang et al., Image Quality Assessment: From Error Visibility to Structural Similarity,
   TIP 2004.

---

## Implementation

What is implemented, how to run it, and its limits. Design and math are in
[architecture](#system).

### Components

- **Models**: seven-conv DCE-Net and depthwise light variant, eight curve iterations,
  noise head, residual denoiser.
- **Losses**: four baseline losses plus edge, chroma, adaptive exposure, highlight, noise,
  invariance, fidelity and temporal, with detached targets and masks.
- **Data**: sorted image/video manifests, EXIF-aware RGB IO, shared crops, per-channel
  heteroscedastic noise, analytic affine pairs with flow and flicker.
- **Flow**: bidirectional DIS, differentiable pixel-coordinate warp, reliability masks,
  histogram scene-cut detection.
- **Training**: Adam, fixed scene split, loss CSV logging (plus per-step `brightness`/`clipped`
  over-brightening diagnostics), atomic versioned checkpoints, optimizer/RNG resume, rejection of
  incompatible state.
- **Inference**: image, video and webcam enhancers; checkpoint loader.
- **Metrics**: PSNR, Gaussian-window SSIM, angular error, flat-region noise proxy, clipping,
  masked temporal error.
- **API**: FastAPI server with upload limits and JSON errors. **Web**: Next.js upload, before/after slider and an in-browser Analysis section (brightness, shadows, contrast, noise, saturation, tonal richness, clipping, histogram, comparison with matched gamma brightening; `web/lib/analysis.ts`), with a runtime API proxy.

### Commands

Run from `core/` after `python -m pip install -e ".[dev]"`.

```sh
python scripts/train.py --config configs/photo.py --run-name photo_target05
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

### Artifacts

- Training writes `config.yaml`, `metrics.csv`, `val_metrics.csv`, periodic `epoch_XXXX.pt`
  checkpoints and `final.pt` after the last epoch (see `training/trainer.py`). Resume requires
  the identical resolved config and unchanged dataset split hashes, both embedded in the
  checkpoint's provenance rather than written as separate files.
- Checkpoint format v1 holds model, optimizer, resolved config, RNG state and provenance (split
  hashes, run name, device, step counters, elapsed time, git/package versions) in one payload;
  unknown versions, extra keys and architecture mismatches are rejected.
- Evaluation CSVs keep per-sample IDs and mask coverage.
- Synthetic export stores preview PNGs, `pair.pt` (unquantized frames, flows, masks, affine
  matrix, flicker, noise coefficients) and a JSON source/config record.

### Testing

```sh
python -m pytest                      # about 2,060 tests, CPU only, no data or network
python -m ruff check . && python -m ruff format --check . && python -m mypy
cd ../web && npm test && npm run typecheck && npm run lint && npm run build
```

Tests build tiny synthetic images, videos and checkpoints in temp directories. They cover
numerical identities and gradients, curve and affine conventions, temporal state and resets,
checkpoint and RNG round trips, a 30-step baseline overfit, an end-to-end train, enhance,
evaluate cycle (CLI and API produce identical pixels), the API contract and error codes, and
config schema and ablation matrix validation. CI runs the complete suite, including API and
end-to-end tests, on Python 3.10 and 3.12, and runs strict type checking on 3.12. The CI CPU
thread limits prevent small synthetic tests from oversubscribing the runner. In restricted
Windows environments pytest's temporary-directory permissions may require running outside
the sandbox; permission errors should not be reported as model regressions.

### Limits

- **Train briefly, and watch the brightening-drift diagnostics.** On LOL (`configs/full.py`, no
  temporal, CPU) the model over-brightens as training continues: held-out LOL PSNR rises (10.7 at
  epoch 10, 13.5 at 15, 16.6 at 40, with the checkpoints shipped under `checkpoints/lol_dark_only/`
  trained with `loss.exposure_target=0.6`) but a moderately dark photo is blown out from epoch 15
  and washed out by 25. The shipped `lol/final.pt` recipe uses 10 epochs as the workaround. A more
  varied training set (adding normal-light and darkened copies) did not fix it.

  Two changes address this problem. `loss.exposure_target`'s research default was lowered
  from `0.6` to `0.5` (`training/config.py`, `configs/baseline.py`, `configs/full.py`) as a
  proposed mitigation. A completed photo experiment now shows that this is not a uniform
  improvement (see below), so the recommended photo recipe retains 0.6. Separately,
  `train_step`/`validation_step` (`training/loop_step.py`) now log `brightness` (mean output
  intensity) and `clipped`
  (near-saturated pixel fraction) on every step, so the drift this limit describes is visible in
  `metrics.csv`/`val_metrics.csv` during a run instead of only being noticed afterward on a demo
  photo. This is monitoring, not a fix: the loss and validation split are still LOL-only, so
  nothing here does early stopping against ordinary photos. That still needs a held-out set of
  ordinary photos and a stopping rule keyed on `clipped`/`brightness` rather than LOL PSNR; neither
  has been done.

  `configs/photo.py` now supplies a separate 10-epoch, non-temporal experiment with a checkpoint
  every epoch. Evaluation reports input/output brightness and contrast, and the preservation
  gate rejects per-image regressions using explicitly supplied limits. See
  [Photo training and checkpoint promotion](#still-photo-training-and-checkpoint-promotion) for the held-out validation preparation and promotion
  process. A seed-17, 10-epoch target-0.5 experiment has now been completed: it improved
  normal-input SSIM slightly but lost low-light SSIM and visible detail, so it was not
  promoted. `photo.py` retains target 0.6; `photo_target05.py` records the alternative.
  See [Measured settings and experiment results](#settings-for-natural-looking-photos) for the full comparison and remaining validation needs.

- Trained weights are included under `checkpoints/` (see `checkpoints/README.md`); controlled
  extension-ablation results are not included. Inference measurements and their limits are in
  [Measured settings and experiment results](#settings-for-natural-looking-photos). No general quality or real-time claim is made; the 30 fps / 720p
  budget (capture 3 ms, resize 2, estimation 10, DIS 8, warp and curves 4, denoise 3, output 3)
  is a design target to be measured.
- Loss weights and thresholds are untuned starting values. Synthetic noise is a Gaussian
  approximation; added-noise `sigma` is not a camera-noise label.
- The loader cannot infer exposure labels or permissions: curate low-light inputs and
  scene/session splits. Images smaller than the training crop fail. Training needs `H, W`
  divisible by 16; inference needs at least 16.
- Checkpoints use trusted-local PyTorch serialization; never load untrusted files. CUDA
  grid-sampling backward can be nondeterministic.
- The API has no authentication; add TLS, authentication and rate limiting in front of it.

### Metrics for evaluation

PSNR and SSIM (data range 1, 11x11 Gaussian window, sigma 1.5) and angular error on paired
LOL; noise proxy, clipping fraction and temporal error on unpaired and video sets. Report
mean and standard deviation over the three seeds, with held-out failure cases. Do not tune on
the LOL evaluation split.

---

## Still-photo training and checkpoint promotion

`configs/photo.py` is a conservative **photo recipe**, not a claim of optimal
quality. It uses the existing full photo architecture and losses, disables the video
objective, limits the run to 10 epochs, and saves every epoch. The shipped checkpoint
remains unchanged. It retains the shipped exposure target of 0.6. The baseline/full
presets keep their original experiment schedules.

Run from `core/`, with the Python environment active:

```powershell
python scripts/train.py --config configs/photo.py --run-name photo_target06
python scripts/evaluate.py --checkpoint checkpoints/photo_target06/epoch_0005.pt --output runs/photo_target06/epoch05_eval.csv
python scripts/evaluate.py --checkpoint checkpoints/photo_target06/final.pt --output runs/photo_target06/final_eval.csv
```

Use a new run name for each experiment: a fresh training run owns and resets that run's
logs and checkpoint files. Never reuse `lol` or `lol_dark_only` when testing a new recipe.

### Validate the exposure-target change

The shipped weights were trained with exposure target 0.6. We tested a 10-epoch, seed-17
alternative at 0.5: it passed the illustrative preservation limits and slightly improved
normal-input SSIM, but substantially reduced low-light PSNR/SSIM and visible detail. It
was not promoted. Results are in [Measured settings and experiment results](#settings-for-natural-looking-photos).

`configs/photo_target05.py` preserves that alternative; the two photo recipes differ
only in `loss.exposure_target` (0.5 versus 0.6). Use distinct run names for new comparisons.
Keep the training data, scene split, seed, augmentation, optimizer, and schedule identical.
Use checkpoint metadata/config files to record the exact recipe. A tiny synthetic smoke
run proves that training works; it does not establish better photographic quality.

Before training or selecting checkpoints, curate an independent set of ordinary photos
under `data/unpaired/ordinary/`, including moderately dark scenes, well-lit regions,
skin tones, saturated colors, fine texture, and bright highlights. Separate scenes from
the training data and keep the same manifest for every candidate. Do not use the 15 LOL
evaluation images to tune settings and then present those scores as unbiased test results.

### Promotion checks

Evaluate each candidate and the shipped checkpoint on the same images. Keep per-image
rows, not just dataset averages. Evaluation includes `mean_brightness`, `contrast_std`,
their `_input` counterparts, and output clipping with the input clipping value. These
measure exposure drift and contrast collapse even when no pixels reach the clipping
threshold. The CLI summary reports the finite sample count and excluded non-finite
samples, so missing noise-mask measurements cannot silently inflate sample counts.

Review before/after images as well: brightness/contrast are descriptive statistics, not
quality scores. Check that already well-lit areas stay plausible, shadows retain detail,
color stays natural, and denoising does not erase texture. Report PSNR/SSIM and color error
on paired data alongside these checks. Set any acceptable regression limits before
viewing candidate results, and repeat promising settings across the existing seeds
(17, 29, 43) before making a general quality claim. If a candidate fails these checks,
retain the shipped weights instead of automatically deploying the final epoch.

### Reproducible preservation checks with the existing LOL archive

When a local LOL ZIP is available (the repository's existing archive is
`data/_dl/lol.zip`), prepare the normal-light counterparts of the trainer's fixed
held-out scenes:

```powershell
python scripts/prepare_photo_validation.py
python scripts/evaluate.py --checkpoint checkpoints/lol/final.pt --unpaired runs/improvements/ordinary_validation --output runs/improvements/reference.csv
python scripts/evaluate.py --checkpoint checkpoints/photo_target05_20261109/final.pt --unpaired runs/improvements/ordinary_validation --output runs/improvements/candidate.csv
```

The preparation script excludes all train-scene IDs, reads only exact normal-light
archive members, and records split and PNG hashes in `manifest.json`. It refuses to
replace different files or mix unrelated images into this directory. With the original
485-scene dataset this yields 48 scenes: the trainer uses `round(0.1 * 485)`.
These normal-light scenes test preservation within LOL; they do not replace a separate
ordinary-photo collection. Use `--archive` when the local ZIP is elsewhere. Preparation
does not download a dataset.

`evaluate.py --unpaired` changes only the unpaired directory. Configured paired/video
sets are still evaluated when present; the gate's exact `--dataset unpaired` filter
uses only the flat preservation folder from these commands.

Compare each candidate against the reference using per-image limits chosen in advance:

```powershell
python scripts/check_photo_regressions.py --reference runs/improvements/reference.csv --candidate runs/improvements/candidate.csv --dataset unpaired --max-brightness-increase 0.05 --max-contrast-drop 0.02 --max-clipping-increase 0.01
```

The example limits allow +0.05 mean intensity, -0.02 intensity standard deviation,
and +0.01 clipped fraction relative to the reference. They are illustrative operational
limits, not validated universal settings. Supply your own limits based on the intended
use, before inspecting candidate results. Every input image is checked individually, so
an improved average cannot conceal a badly washed-out scene. Different sample sets,
input statistics, missing measurements, duplicate rows, or non-finite values fail
validation. The gate exits 1 on regressions and 2 on invalid input; it never changes
the deployed checkpoint. Passing still requires visual color/detail review.

---

## Settings for natural-looking photos

The priority is natural exposure and preserved detail. Keep the shipped
`checkpoints/lol/final.pt`, full-resolution curve estimation (`curve_scale=1`), and one
enhancement slot until a replacement passes held-out preservation and visual checks.
Neither a lower training exposure target nor a faster inference scale is automatically
a photographic improvement. The existing trained weights have not been replaced.

### Measured inference tradeoffs

Measured on 9 November 2026 using the six committed `public/low-light-*.png` samples,
the shipped checkpoint, CPU PyTorch 2.15.0, one warmup and three timed runs per image
and scale. Results are the mean of each image's median time; decode, IO and transfers
are excluded. The CPU was shared with other checks, so times are approximate local
measurements, not a hardware-independent speed claim. The JSON files record checkpoint
SHA-256, image paths, dimensions, runtime versions and all timing samples.

| CPU threads | Curve scale | Time (ms/image) | Speedup over scale 1 | Mean absolute drift (0-255) | Largest pixel/channel drift (0-255) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1544.2 | 1.00x | 0 | 0 |
| 1 | 0.5 | 428.0 | 3.61x | 0.443 | 75.95 |
| 1 | 0.25 | 224.5 | 6.88x | 0.765 | 90.67 |
| 4 | 1 | 1250.2 | 1.00x | 0 | 0 |
| 4 | 0.5 | 425.0 | 2.94x | 0.443 | 75.95 |
| 4 | 0.25 | 165.4 | 7.56x | 0.765 | 90.67 |

Drift is measured against the same checkpoint at scale 1. It is not error against a
reference photo and does not establish which result looks better. Small average drift
coexists with substantial local differences, so preserve scale 1 for the detail-focused
default. Reduced scales remain explicit speed/quality options. Thread counts also need
measurement on the deployment CPU rather than being hardcoded from this desktop.

Raw measurements: [one thread](benchmarks/inference_threads1.json) and
[four threads](benchmarks/inference_threads4.json).

To repeat from `core/` (choose a new output path for every run):

```powershell
python scripts/benchmark_inference.py --checkpoint checkpoints/lol/final.pt --input ../public --output runs/inference_threads1.json --threads 1
python scripts/benchmark_inference.py --checkpoint checkpoints/lol/final.pt --input ../public --output runs/inference_threads4.json --threads 4
```

### Training and checkpoint review

The photo preset uses 10 epochs, saves every epoch, and disables the video objective.
It retains exposure target 0.6 from the shipped weights. The tested 0.5 alternative is
preserved in `configs/photo_target05.py`. A fresh run must use a new name so training
cannot reset a shipped run's files.

Evaluation now includes per-image input/output brightness and contrast, plus clipping.
Non-finite values are excluded from means and counted separately. The regression gate
checks per-image changes against explicit limits rather than allowing a dataset average
to hide individual failures. See [Photo training and checkpoint promotion](#still-photo-training-and-checkpoint-promotion) for commands,
the deterministic 48-scene normal-light preservation set, and promotion requirements.
The preservation set is held out within LOL; independent everyday photos and manual
texture/color review are still necessary before claiming general improvement.

The preservation check was exercised on all 48 normal-light validation scenes using
the existing epoch-10 and epoch-15 checkpoints at full resolution. Both checkpoints'
stored validation-split hashes match the prepared manifest. Input mean brightness was
0.4927; mean output brightness rose from 0.5542 (epoch 10) to 0.5771 (epoch 15).

For a demonstration, absolute limits of 0.05 brightness increase, 0.02 contrast drop,
and 0.01 clipping-fraction increase flagged four per-image regressions: excess clipping
on scenes 126 and 53, and excess brightening on scenes 598 and 769. A self-comparison of
epoch 10 passed. These illustrative limits were not used to tune training or establish
universal acceptance criteria; real promotion limits must be agreed before evaluating
new candidates. This check supports retaining the shipped weights rather than blindly
switching to the later checkpoint.

Evidence: [comparison and failures](benchmarks/preservation_epoch15_vs10.json),
[validation manifest](benchmarks/preservation_manifest.json),
[epoch-10 measurements](benchmarks/ordinary_epoch10.csv), and
[epoch-15 measurements](benchmarks/ordinary_epoch15.csv). These files include measurements
and image hashes, not redistributed dataset images.

### Completed exposure-target experiment

The `photo_target05_20261109` run completed 10 epochs / 540 updates in about 15.9 minutes
on CPU (four threads), seed 17, with 437 training scenes and the same 48-scene validation
split as the shipped model. The only learning-objective change was exposure target
0.6 to 0.5; the schedule stopped at epoch 10 and saved every epoch. All logged values
were finite, and validation-crop clipping stayed zero. The complete checkpoint is
`checkpoints/photo_target05_20261109/final.pt`; the shipped weights were preserved.

Both models were evaluated at curve scale 1 on the same 48 normal-light preservation
photos and 15 paired low-light photos, followed by full-image and central-detail review.

| Measurement | Shipped target 0.6 | Candidate target 0.5 |
| --- | --- | --- |
| Normal-photo SSIM against input (preservation) | 0.9353 | 0.9464 |
| Normal-photo contrast std (input: 0.1803) | 0.1612 | 0.1700 |
| Normal-photo mean brightness (input: 0.4927) | 0.5542 | 0.5555 |
| Normal-photo color angular error against input | 1.669 degrees | 1.735 degrees |
| Normal-photo clipped fraction | 0.01034 | 0.01101 |
| Low-light PSNR against reference | 10.682 dB | 9.507 dB |
| Low-light SSIM against reference | 0.5057 | 0.4154 |
| Low-light color angular error | 7.767 degrees | 8.242 degrees |

The candidate passed the pre-recorded preservation limits, but this did not establish
an overall improvement. It preserved normal-input structure somewhat better, while
average brightness did not improve, color error increased, and low-light enhancement
lost substantial structure and visible detail. The visual comparisons also showed
weaker shadow lift. We therefore **did not promote it** and retained target 0.6 in the
recommended photo recipe. Target 0.5 remains a recorded experiment, not a quality fix.

This is a one-seed exploratory comparison within LOL, not evidence of a universal best
setting or independent everyday-photo generalization. Further tuning needs independent
ordinary-photo scenes and seed replication rather than repeatedly selecting on eval15.

Evidence: [complete comparison](benchmarks/photo_target05_comparison.json),
[reference rows](benchmarks/photo_target05_reference.csv),
[candidate rows](benchmarks/photo_target05_candidate.csv),
[training log](benchmarks/photo_target05_training.csv),
[validation log](benchmarks/photo_target05_validation.csv), and the
[recorded experiment plan](benchmarks/photo_target05_20261109_plan.json).
Local visual comparisons are under `runs/improvements/photo_*comparison.png` and
`photo_normal_detail_crops.png`; dataset images are not copied into these docs.

### Application reliability

- Enhancement admission occurs before multipart parsing. Saturation returns
  `503 server_busy` and `Retry-After: 1`, with no in-memory upload waiting queue.
- Health/options remain available during inference, and browser options refresh after
  health polls. Stale responses cannot restore obsolete limits after a restart.
- Browser analysis describes saturation, fine variation, entropy and near-white share
  without presenting them as proof of color fidelity, denoising or highlight safety.
- Gamma baselines report whether their brightness target was actually reached.
- CI includes API/end-to-end tests and strict Python type checking, with a Python
  3.10/3.12 runtime matrix. Docker uses the CPU-only PyTorch index and honors its port
  setting in the health check.
- API dependency minimums include Starlette 0.46.0 (the parser's `spool_max_size` API;
  see its [release notes](https://www.starlette.io/release-notes/#0460-february-22-2025))
  and python-multipart 0.0.18.
