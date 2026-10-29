# Implementation status

The original M0 scaffold is now an executable core implementation. The web home route remains intentionally blank; no API server or browser integration has been added.

## Implemented

- Seven-convolution baseline and depthwise light estimator, eight plus-sign curve iterations, full-resolution curve upsampling, bounded noise head and residual denoiser.
- Four baseline losses and the five documented feature groups, with detached teachers/targets/gain/masks, adaptive exposure replacement, finite scalar logging and gradient clipping.
- Strict typed YAML validation and explicit-false CLI precedence, sorted image/video manifests, EXIF-aware RGB IO, shared crops, per-channel heteroscedastic noise and analytic affine pairs.
- DIS bidirectional flow, pixel-coordinate differentiable warping, current-domain reliability masks, histogram scene cuts, detached curve/output history and stream/shape resets.
- Adam training with a fixed scene split, content-aware provenance, loss CSVs, atomic versioned checkpoints, optimizer/RNG resume and incompatible-state rejection.
- All five CLI commands: training, image/video/webcam enhancement, paired/unpaired/video evaluation, fixed-matrix ablations and local synthetic export.
- PSNR, Gaussian-window SSIM, angular error, input-selected flat-region noise proxy, clipping and masked temporal error.

## Verification and limits

The CPU suite exercises generated low-light data, numerical identities and gradients, affine conventions, temporal state, checkpoint/RNG roundtrips, a 30-step tiny baseline overfit and a local train/save/load/CLI-enhance/evaluate cycle. It also roundtrips a short encoded video. See [TESTING](TESTING.md) for commands and test-specific settings.

These checks do not satisfy the real-data exit criteria of M1–M5. No curated SICE/LOL data, pretrained or research-trained weights, real-camera calibration, held-out visual grids, GPU performance measurements, camera/display validation, 39-run ablation results, or completed research report are supplied. The [experiment tables](EXPERIMENTS.md) and [latency tables](REAL_TIME.md) remain empty. No baseline parity, quality improvement, or real-time claim is made.

The loader cannot infer exposure labels or permissions: curate low-light inputs and scene/session splits before training. A directory with one scene has no held-out scene; use multiple scenes for research. Relative paths resolve from core/. Images smaller than the configured training crop fail explicitly. Checkpoints use trusted-local PyTorch serialization; never load untrusted files. CUDA grid-sampling backward may be nondeterministic despite seeded execution.

## Runtime artifacts

Training writes `config.yaml`, `split.json`, `provenance.json`, `metrics.csv`, `last.pt`, and periodic epoch checkpoints to the configured local directories. Resume requires the identical resolved experiment and unchanged dataset hashes. Separate run/checkpoint directories prevent mixing objectives or seeds.

Synthetic export stores preview PNGs, `pair.pt` with unquantized frames, analytic flows/masks, affine matrix, flicker and applied noise coefficients, and a JSON source/config record. Evaluation CSVs retain individual sample IDs and mask coverage; aggregation and report conclusions require the full experimental protocol.
