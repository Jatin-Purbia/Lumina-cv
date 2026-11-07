# Architecture

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
| `LUMINA_MAX_CONCURRENCY` | 1 | simultaneous decode/infer/encode jobs |
| `LUMINA_CORS_ORIGINS` | empty | comma-separated origins or `*` |
| `LUMINA_HOST`, `LUMINA_PORT` | `127.0.0.1`, `8000` | bind address |

The model loads once at startup; a bad checkpoint or config exits with code 3, a bad variable
with code 2. One process holds one model; extra uvicorn workers multiply memory. Uploads stay
in memory, and logs contain only method, path, status, time and request id.

## Data

Local layout (paths set in the `paths` section of the config, relative to `core/`):

```text
data/SICE/part1/<scene>/<exposure>.jpg      training images (unpaired objective)
data/LOL/eval15/{low,high}/<pair>.png       paired evaluation only, never trained on
data/unpaired/{DICM,LIME,MEF,NPE}/...       qualitative evaluation
data/videos/{train,val,test}/<session>/<clip>.mp4
```

Split by scene/session, not by image, and keep LOL evaluation disjoint from training. Check
each dataset's terms before use; none are redistributed. Training records split and content
hashes in `split.json` and `provenance.json`.

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
