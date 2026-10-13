# Experiments and reporting protocol

No runs, measurements or trained checkpoints exist. Tables below remain empty
until imported from run CSVs; do not insert hypothetical results. All added
objectives in [METHOD](METHOD.md) are project proposals. Baseline attribution:
[R1](REFERENCES.md#r1), [R2](REFERENCES.md#r2).

## Ablations

| Run | Noise | Denoise | Edge | Chroma | Adaptive | Temporal |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | off | off | off | off | off | off |
| noise_only | on | off | off | off | off | off |
| edge_only | off | off | on | off | off | off |
| chroma_only | off | off | off | on | off | off |
| adaptive_only | off | off | off | off | on | off |
| temporal_only | off | off | off | off | off | on |
| noise_denoise | on | on | off | off | off | off |
| full | on | on | on | on | on | on |
| loo_noise | off | off | on | on | on | on |
| loo_edge | on | on | off | on | on | on |
| loo_chroma | on | on | on | off | on | on |
| loo_adaptive | on | on | on | on | off | on |
| loo_temporal | on | on | on | on | on | off |

Exact rows are in [ablation_matrix.yaml](../configs/ablation_matrix.yaml). There
is no denoise-only row because it is invalid. Noise extension alone uses
noise_only, and noise_denoise isolates the additional residual head. Leave-one-out
removes all parts of an extension; loo_noise removes both heads. Temporal training
and inference switch together initially; report additional train-vs-inference
analysis only after a reviewed protocol/config change.

## Fixed schedule and controls

Seeds 17,29,43; scene split seed 17 fixed across runs. Epochs 100, crop 256,
batch 8, Adam lr 1e-4, wd 1e-4, global grad-norm clip .1. No scheduler or mixed
precision in the initial protocol. Same data order per seed across compatible
runs, same held-out scenes, same augmentation/coefficient distributions.
Train on SICE low images without high-light references. Initial temporal source
is synthetic to make all rows reproducible without private footage; conduct
the real-video experiment separately with temporal.pair_source=real and record
the manifest, sampled pair counts and effective number of optimizer steps.
Free Colab/Kaggle interruption changes are handled by checkpoint state, not by
changing the schedule. No automatic claim of equal compute: temporal adds forwards.

Record resolved YAML, schema/version, git commit, seed, device/software versions,
split hashes, elapsed time, epoch/step and raw/weighted loss terms. For metrics,
write rows with run, seed, dataset, sample_id, metric, value, mask_coverage and
input_clipping where applicable; aggregate by scene then seed, not by pixels.

## Metrics and direction

```text
MSE=mean_RGBHW((o-ref)^2); PSNR=10*log10(1/MSE), data_range=1
SSIM=((2*mu_x*mu_y+C1)*(2*cov_xy+C2)) /
     ((mu_x^2+mu_y^2+C1)*(var_x+var_y+C2))
C1=.01^2; C2=.03^2; Gaussian window 11x11 sigma=1.5
angle=acos(clamp(dot(o,ref)/(norm(o)*norm(ref)+eps),-1,1))*180/pi
clipping=mean(1[Y(o)>=.99])
temporal_error=sum(mask*|o_t-warp(o_prev,b)|)/(3*sum(mask)+eps)
coverage=mean(mask)
Immerkaer K=[[1,-2,1],[-2,4,-2],[1,-2,1]]
sigma_hat=sqrt(pi/2)*sum(|Y(x)*K|)/(6*valid_pixel_count)
```

| Metric | Use | Better direction | Caveat |
| --- | --- | --- | --- |
| RGB PSNR | paired LOL | higher | registered target, no per-image alignment/tone fit |
| RGB SSIM | paired LOL | higher | average channel values, crop valid window border; [R10](REFERENCES.md#r10) |
| RGB angular error, degrees | paired LOL reliable reference | lower | exclude norm<=eps; not perceptual colour distance |
| Flat-region sigma proxy | own/unpaired | lower conditionally | same input-derived flat mask; texture bias [R8](REFERENCES.md#r8) |
| Near-clipped luminance fraction | all sets | lower conditionally | dim images score well; report exposure and input clipping |
| Flow-aligned RGB L1 | held-out videos | lower | can favour blurry outputs; use fixed input flow/masks |
| Mask coverage | videos | descriptive, no optimum | expose rejected motion/darkness |
| Median and p95 frame latency | webcam/held-out clips | lower | fixed resolution/device and synchronized GPU timing |

For Immerkaer use valid 3x3 convolution responses wholly inside an input-selected
flat mask, fixed across methods. Define flat selection before M2 using input
gradient threshold from edge reliability; report chosen threshold and coverage.
No-reference metrics are proxies, not quality certificates. NIQE is deliberately
not promised because it is outside the listed API/dependency contract.

## Empty results tables

| Run | Seeds | LOL PSNR mean +/- std | LOL SSIM mean +/- std | Angle mean +/- std | CSV source |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

| Run | Set | Noise proxy | Clipping | Temporal L1 | Coverage | CSV source |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

| Run | Device | Resolution | Curve scale | Median ms | p95 ms | FPS | CSV source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## Qualitative figure plan

F1: architecture/flow convention diagram from ARCHITECTURE.
F2: fixed SICE/LOL and unpaired photo grid: input, baseline, singles, full,
paired target where available; same zoom and no selective exposure adjustment.
F3: noisy flat patch, edge patch, chroma map, highlight crop; fixed source IDs.
F4: temporal adjacent frames, space-time strip, difference maps with masks,
cut/reset example and moving-object ghosting failures.
F5: latency-quality tradeoff from measured curve_scale/light sweep, same device.
Show representative failures as well as successes; preserve filenames and settings.

## Statistics and validity

Three independent training seeds: report per-seed scene-averaged metrics, then
sample mean +/- sample std across the three seeds. No unqualified significance
claim from n=3. For comparisons use matched scenes/seeds; bootstrap by scene or
video session, not adjacent frames, and label any intervals exploratory. Separate
image metrics from video metrics and synthetic from real capture results.

Threats include data overlap in repackaged sets, different scene distributions,
imperfect paired alignment, synthetic noise mismatch, JPEG artifacts, pseudo-clean
teacher error, flow/mask bias, metric preference for smoothing/dimming, correlated
video samples, tuned weights selected on evaluation, and uncontrolled GPU timing.
Baseline profile differences mean results cannot be advertised as exact reference
parity. Record effective training compute and compare identical evaluation settings.
