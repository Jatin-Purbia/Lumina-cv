# Interfaces and schema

## Global tensor and device contract

Float32 RGB BCHW [0,1] is canonical; dataset items omit B and collation adds it.
Curve maps are BCHW with 24 iteration-major RGB channels in [-1,1]; sigma is
B1HW in [0,.2]. Model always returns enhanced, enhanced_raw, A and sigma. Disabled
noise gives exact zero sigma; disabled denoise makes enhanced equal enhanced_raw.
No implicit device transfer inside model/loss/warp. All participating tensors share
device/dtype. Metrics return per-sample tensors; losses return scalar tensors.
Only IO, DIS CPU adapter and explicit orchestration may transfer. Inference uses
eval/no_grad; training keeps student graphs and detaches teachers/masks/targets/gain.

## Flow contract

```text
flow=f(a,b): [B,2,H,W], x displacement then y, in pixels of full-resolution image
a(p) approximately b(p+flow(p))
warp(b,flow) samples b into a's coordinates
backward=f(current,previous); warp(previous,backward) is previous in current domain
grid=(2*(p_x+dx)/(W-1)-1,2*(p_y+dy)/(H-1)-1), align_corners=True
```

Warp uses bilinear zero-padded sampling and returns an in-bounds B1HW mask.
Forward/backward flow is estimated independently; negation is not a general
inverse. DIS returns CPU detached vectors rescaled by full/small width and height.
Mask tests are evaluated in current coordinates. See [METHOD](METHOD.md) extension
5 and [TESTING](TESTING.md) affine cases. Map upsampling uses align_corners=False;
this does not alter the warp convention.

## Data and orchestration context

ImageDataset items: image (3,H,W), optional reference (3,H,W). VideoPairDataset
items: image1,image2 (3,H,W), scalar index; real flow is computed by orchestration
via the flow adapter. Synthetic batches contain image1,image2, forward_flow,
backward_flow, mask; training noise path additionally keeps clean RGB and known
target_sigma. train_step context supplies image, enhanced, A, sigma, gain, targets,
noisy_enhanced, teacher_enhanced, teacher_raw, target_sigma, previous_output,
current_output, backward_flow and mask as required by active loss metadata.
Do not rename model output keys silently; registry aliases are explicit context keys.

## CLI and checkpoint contracts

All scripts provide --config, --noise, --denoise, --edge, --chroma, --adaptive-exp,
--temporal, --light, --curve-scale, --video-dir, --seed and --device. Boolean flags
also have --no-* forms. --noise maps to flags.noise_aware; --adaptive-exp to
flags.adaptive_exp; --curve-scale to model.curve_scale; --video-dir to
paths.video_dir; remaining feature flags use underscores in YAML. Precedence:
schema defaults < complete YAML < explicitly supplied CLI. None is absent, not
false. Resolve/revalidate before construction. --webcam and --input conflict;
enhancement/evaluation need checkpoints at runtime; --help has no data dependency.
train --resume restores compatible optimizer/RNG; ablate --matrix expands base
config/seed/flag rows; evaluate --output writes CSV; make_synthetic_data --output
writes local private samples. All five scripts are implemented; see [Implemented CLI behaviour](#implemented-cli-behaviour).

Checkpoint format v1 contains model state, optional optimizer, completed epoch,
resolved config/flags, schema_version, RNG states and provenance (git/environment
and split hashes). Reject unknown format/schema versions, noise/light/shape mismatch,
and unexpected keys. Load trusted local checkpoints on CPU then explicitly move.
Loss-only flag changes are also rejected during exact resume; a future deliberate
fine-tune path must document compatible architecture and changed objective.

## Schema bounds and units

Records below are dataclasses; generated constructors store fields, but validate,
Config load/save/override helpers are implemented in training/config.py. Never interpret constructing Config
as validated. YAML baseline/full includes every record field; paths example and
ablation matrix have specialized schemas described after the table.

| Section | Units and validation |
| --- | --- |
| schema_version | integer exactly 1; reject unknown fields at every nesting level |
| flags | booleans exactly; denoise implies noise_aware; all extension combos otherwise valid |
| model | positive widths, iterations exactly 8, sigma_max in (0,1], residual_scale in (0,1], curve_scale in (0,1], gain_min>=1 and <=gain_max, eps>0 |
| weights | all finite nonnegative scalars; adaptive exp uses weights.exp |
| loss | patch sizes positive and divide crop; target/thresholds in [0,1]; low<high; noise scales/eps>0; penalties>=0; odd smoothing/local kernels, positive Gaussian sigma |
| noise | a,b>=0 and ordered min/max; probability [0,1]; sigma_max >=sqrt(a_max+b_max^2); b is std |
| synthetic | rotation degrees/translation pixels >=0; scales/flicker >0 and ordered ranges |
| flow | method exactly dis, preset medium initially; scale (0,1]; alpha,beta>=0, beta squared-pixel units; photometric threshold [0,1]; align_corners=True |
| temporal | blends [0,1], cut threshold [0,1], pair_source synthetic or real, frame_gap>=1 |
| training | lr>0, batch/epochs/workers/log intervals valid integers, crop>=16 divisible by patches, optimizer Adam, wd>=0, clip>0, nonnegative seed, explicit device supported |
| paths | nonempty local paths; resolve against core workspace root, required roots checked at execution not --help; train/eval disjoint manifests |

Reject NaN/infinity and wrong YAML scalar types (including bool as int). Numeric
integer literals may be converted to float only where explicitly scalar-float
fields allow it. For temporal real pairs require an existing video directory;
synthetic needs only training image paths. Do not require local paths during a
pure serialization round trip. Configuration validates these dependencies before
training/inference; CLI errors should eventually be actionable ValueErrors.

### flags: FeatureFlags

| Field | Type | Default |
| --- | --- | --- |
| noise_aware | bool | False |
| denoise | bool | False |
| edge | bool | False |
| chroma | bool | False |
| adaptive_exp | bool | False |
| temporal | bool | False |
| light | bool | False |

### model: ModelConfig

| Field | Type | Default |
| --- | --- | --- |
| channels | int | 32 |
| iterations | int | 8 |
| noise_channels | int | 16 |
| denoise_channels | int | 16 |
| sigma_max | float | 0.2 |
| residual_scale | float | 0.1 |
| curve_scale | float | 1.0 |
| gain_window | int | 5 |
| gain_min | float | 1.0 |
| gain_max | float | 8.0 |
| eps | float | 1e-6 |

### weights: LossWeights

| Field | Type | Default |
| --- | --- | --- |
| spa | float | 1.0 |
| exp | float | 10.0 |
| col | float | 5.0 |
| tv | float | 200.0 |
| edge | float | 20.0 |
| chroma | float | 20.0 |
| highlight | float | 20.0 |
| noise | float | 20.0 |
| invariance | float | 10.0 |
| fidelity | float | 1.0 |
| temporal | float | 10.0 |

### loss: LossConfig

| Field | Type | Default |
| --- | --- | --- |
| spatial_patch | int | 4 |
| exposure_patch | int | 16 |
| exposure_target | float | 0.6 |
| adaptive_low | float | 0.55 |
| adaptive_high | float | 0.85 |
| adaptive_noise_scale | float | 0.05 |
| adaptive_noise_penalty | float | 0.15 |
| highlight_threshold | float | 0.8 |
| chroma_window | int | 5 |
| reliability_low | float | 0.05 |
| reliability_high | float | 0.2 |
| reliability_noise | float | 2.0 |
| edge_smooth_sigma | float | 0.8 |
| edge_kernel | int | 5 |
| edge_noise_multiplier | float | 2.0 |
| eps | float | 1e-6 |

### noise: NoiseConfig

| Field | Type | Default |
| --- | --- | --- |
| a_min | float | 0.0 |
| a_max | float | 0.01 |
| b_min | float | 0.0 |
| b_max | float | 0.02 |
| probability | float | 1.0 |

### synthetic: SyntheticConfig

| Field | Type | Default |
| --- | --- | --- |
| rotation_degrees | float | 3.0 |
| translation_pixels | float | 8.0 |
| scale_min | float | 0.98 |
| scale_max | float | 1.02 |
| flicker_min | float | 0.9 |
| flicker_max | float | 1.1 |

### flow: FlowConfig

| Field | Type | Default |
| --- | --- | --- |
| method | str | "dis" |
| preset | str | "medium" |
| scale | float | 0.5 |
| fb_alpha | float | 0.01 |
| fb_beta | float | 0.5 |
| photometric_threshold | float | 0.1 |
| align_corners | bool | True |

### temporal: TemporalConfig

| Field | Type | Default |
| --- | --- | --- |
| curve_blend | float | 0.7 |
| output_blend | float | 0.0 |
| histogram_bins | int | 32 |
| scene_cut_threshold | float | 0.3 |
| pair_source | str | "synthetic" |
| frame_gap | int | 1 |

### training: TrainingConfig

| Field | Type | Default |
| --- | --- | --- |
| lr | float | 1e-4 |
| batch_size | int | 8 |
| crop_size | int | 256 |
| epochs | int | 100 |
| optimizer | str | "Adam" |
| adam_beta1 | float | 0.9 |
| adam_beta2 | float | 0.999 |
| adam_eps | float | 1e-8 |
| weight_decay | float | 1e-4 |
| grad_clip | float | 0.1 |
| seed | int | 17 |
| workers | int | 2 |
| device | str | "auto" |
| checkpoint_every | int | 5 |
| log_every | int | 10 |

### paths: DataPaths

| Field | Type | Default |
| --- | --- | --- |
| train_images | str | "data/SICE/part1" |
| eval_low | str | "data/LOL/eval15/low" |
| eval_high | str | "data/LOL/eval15/high" |
| unpaired | str | "data/unpaired" |
| video_dir | str | "data/videos/train" |
| run_dir | str | "runs" |
| checkpoint_dir | str | "checkpoints" |

### Specialized YAML

data_paths_example.py has one paths mapping with exactly the DataPaths fields;
it is a merge example, not a complete training Config. ablation_matrix.py has
base_config (local complete YAML), seeds (unique nonnegative integer list) and
runs (unique name, complete FeatureFlags mapping). Every run inherits all base
hyperparameters and overrides only flags plus seed. Reject duplicates, unknown
fields, invalid dependencies and missing baseline/full/single/leave-one-out rows.

## Structural public types

ModelOutput is a TypedDict with enhanced, enhanced_raw, A, sigma tensors as above.
VideoState stores optional previous_image, previous_A, previous_output; None means
reset, B=1 only, detached tensors on model device. LossSpec is immutable with
callable (tensor loss), flag (name, not-flag expression or None), default_weight (nonnegative float),
required_inputs (ordered context names). LOSS_REGISTRY is keyed by name and
stores twelve default LossSpec records referencing pure loss functions; no loss executes
at import. Dynamic registration is implemented in utils/registry.py.
LOSS_CONTRACTS provides the complete declarative planned names/gates/weights/inputs.
Registry[T] provides unique typed lookup; Config and its nested records are typed
metadata, not computation. For every executable class, constructor/forward and
other method contracts are listed below. Package __init__ files expose namespaces
without eager imports or registry/model side effects.

## Every public callable and class method

The following inventory gives the exact pre-format signature and detailed contract
for each source API. Source formatter may wrap lines without changing signatures.
All are implemented and tested; signatures below were written at M0, so where source and this
inventory differ the source and its docstrings win (additions are listed in
[Implemented additions](#implemented-additions)).

### models/dce_net

Source: [models/dce_net.py](../src/nta_zerodce/models/dce_net.py)

Class `DCENet`: Estimate eight RGB curve maps using seven symmetric skip convolutions.

Constructor and forward contracts are below.

```python
def DCENet.__init__(self, in_channels: int=3, channels: int=32, iterations: int=8) -> None:
```

```text
Construct the curve estimator.

Args:
    in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
    channels: Positive hidden width, default 32.
    iterations: Number of updates; fixed to 8 in this project.

Returns:
    No value; registers layers.

Method:
    METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

```

```python
def DCENet.forward(self, image: Tensor) -> Tensor:
```

```text
Estimate eight RGB curve maps using seven symmetric skip convolutions.

Args:
    image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

Returns:
    A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

Method:
    METHOD baseline network, with identical skips for the light variant.

```

Class `LightDCENet`: Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

Constructor and forward contracts are below.

```python
def LightDCENet.__init__(self, in_channels: int=3, channels: int=32, iterations: int=8) -> None:
```

```text
Construct the curve estimator.

Args:
    in_channels: 3 for baseline or 4 for noise-aware RGB+sigma.
    channels: Positive hidden width, default 32.
    iterations: Number of updates; fixed to 8 in this project.

Returns:
    No value; registers layers.

Method:
    METHOD baseline architecture; 3x3 stride 1, symmetric concatenations, tanh head.

```

```python
def LightDCENet.forward(self, image: Tensor) -> Tensor:
```

```text
Estimate curves with a project depthwise-separable variant; not a Zero-DCE++ reproduction.

Args:
    image: (B,in_channels,H,W) float32; RGB in [0,1], optional sigma >=0.

Returns:
    A: (B,24,H,W) float32 in [-1,1], iteration-major RGB ordering.

Method:
    METHOD baseline network, with identical skips for the light variant.

```

```python
def apply_curves(image: Tensor, A: Tensor, iterations: int=8) -> Tensor:
```

```text
Apply iterative light-enhancement curves.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    A: (B,3*iterations,H,W) in [-1,1], iteration-major RGB.
    iterations: Positive count, default 8.

Returns:
    Enhanced RGB (B,3,H,W) in [0,1] on the same device.

Method:
    Zero-DCE [R1], METHOD baseline: x_next = x + A*x*(1-x).

```

### models/noise_branch

Source: [models/noise_branch.py](../src/nta_zerodce/models/noise_branch.py)

Class `NoiseBranch`: Predict one per-pixel noise standard deviation shared across RGB. See METHOD extension 1.

```python
def NoiseBranch.__init__(self, channels: int=16, sigma_max: float=0.2) -> None:
```

```text
Construct the auxiliary head.

Args:
    channels: Positive hidden width.
    sigma_max: Positive bound in normalized RGB units.

Returns:
    No value; future implementation registers convolutional layers.

Method:
    METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

```

```python
def NoiseBranch.forward(self, image: Tensor) -> Tensor:
```

```text
Predict one per-pixel noise standard deviation shared across RGB.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.

Returns:
    sigma (B,1,H,W) in [0,sigma_max], same device.

Method:
    METHOD extension 1: bounded sigmoid sigma; synthetic target sqrt(a*meanRGB(x)+b^2).

```

### models/denoiser

Source: [models/denoiser.py](../src/nta_zerodce/models/denoiser.py)

Class `ResidualDenoiser`: Remove a bounded residual conditioned on estimated amplified noise. See METHOD extension 1.

```python
def ResidualDenoiser.__init__(self, channels: int=16, residual_scale: float=0.1) -> None:
```

```text
Construct the auxiliary head.

Args:
    channels: Positive hidden width.
    residual_scale: Positive bound in normalized RGB units.

Returns:
    No value; future implementation registers convolutional layers.

Method:
    METHOD extension 1: clamp(z - residual_scale*tanh(head([z,sigma*gain])),0,1).

```

```python
def ResidualDenoiser.forward(self, enhanced_raw: Tensor, sigma: Tensor, gain: Tensor) -> Tensor:
```

```text
Remove a bounded residual conditioned on estimated amplified noise.

Args:
    enhanced_raw: RGB (B,3,H,W) in [0,1].
    sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
    gain: Detached (B,1,H,W) in [1,8].

Returns:
    enhanced (B,3,H,W) in [0,1].

Method:
    METHOD extension 1: clamp(z - residual_scale*tanh(head([z,sigma*gain])),0,1).

```

### models/nta_zerodce

Source: [models/nta_zerodce.py](../src/nta_zerodce/models/nta_zerodce.py)

Class `ModelOutput`: All outputs are full-resolution float32, on the input device.

enhanced/enhanced_raw: RGB (B,3,H,W) in [0,1]. A: (B,24,H,W) in [-1,1].
sigma: (B,1,H,W) in [0,0.2]; exact zeros when noise_aware is disabled.
METHOD composition; keys are always present, including in baseline mode.

Class `NTAZeroDCE`: Unified flag-controlled enhancement module; METHOD composition.

No training, dataset, file IO or optical-flow dependencies are allowed here.

```python
def NTAZeroDCE.__init__(self, config: Config) -> None:
```

```text
Construct the flag-controlled composition.

Args:
    config: Validated schema from training/config.py; denoise implies noise_aware.

Returns:
    No value; future implementation owns curve/noise/denoise submodules.

Method:
    ARCHITECTURE composition; flags gate construction.

```

```python
def NTAZeroDCE.forward(self, image: Tensor, A_override: Tensor | None=None) -> ModelOutput:
```

```text
Estimate and apply curves, then optionally denoise.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    A_override: Optional full-resolution (B,24,H,W) in [-1,1] from temporal smoothing.

Returns:
    ModelOutput with enhanced, enhanced_raw, A, sigma; baseline enhanced equals raw.

Method:
    METHOD composition and temporal inference; override bypasses curve estimation only.

```

### models/factory

Source: [models/factory.py](../src/nta_zerodce/models/factory.py)

```python
def build_model(config: Config) -> NTAZeroDCE:
```

```text
Build a model from validated feature flags.

Args:
    config: Validated Config; light changes convolution type, noise changes input channels.

Returns:
    Uninitialized-for-training NTAZeroDCE; caller chooses device.

Method:
    ARCHITECTURE flag matrix and METHOD architecture.

```

### losses/spatial

Source: [losses/spatial.py](../src/nta_zerodce/losses/spatial.py)

```python
def spatial_loss(image: Tensor, output: Tensor, patch_size: int=4) -> Tensor:
```

```text
Preserve local intensity contrast.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    patch_size: Non-overlapping region size, default 4.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Baseline L_spa; valid four-neighbour region pairs, squared absolute differences. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/exposure

Source: [losses/exposure.py](../src/nta_zerodce/losses/exposure.py)

```python
def exposure_loss(output: Tensor, target: float=0.6, patch_size: int=16) -> Tensor:
```

```text
Control baseline patch exposure.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    target: Desired mean RGB intensity in [0,1], default 0.6.
    patch_size: Non-overlapping region size, default 16.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Baseline L_exp; reference-profile squared error; paper absolute variant documented. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/color

Source: [losses/color.py](../src/nta_zerodce/losses/color.py)

```python
def color_loss(output: Tensor) -> Tensor:
```

```text
Penalize unequal global RGB channel means.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Baseline L_col = mean_batch sqrt(d_RG^4+d_RB^4+d_GB^4). See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/smoothness

Source: [losses/smoothness.py](../src/nta_zerodce/losses/smoothness.py)

```python
def smoothness_loss(A: Tensor) -> Tensor:
```

```text
Regularize curve maps spatially.

Args:
    A: (B,24,H,W) in [-1,1], H,W>=2.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Baseline L_tv = 2*(mean(dx(A)^2)+mean(dy(A)^2)). See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/adaptive_exposure

Source: [losses/adaptive_exposure.py](../src/nta_zerodce/losses/adaptive_exposure.py)

```python
def adaptive_targets(image: Tensor, sigma: Tensor, target: float=0.6, low: float=0.55, high: float=0.85, noise_scale: float=0.05, noise_penalty: float=0.15, patch_size: int=16) -> Tensor:
```

```text
Compute detached noise-aware patch exposure targets.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
    target/low/high: [0,1] constants with low<high.
    noise_scale: Positive normalization, default 0.05.
    noise_penalty: Target reduction, default 0.15.
    patch_size: Pool size, default 16.

Returns:
    Detached patch targets (B,1,H//patch_size,W//patch_size) in [0,1].

Method:
    Extension 4 target formula and smoothstep in METHOD. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

```python
def adaptive_exposure_loss(output: Tensor, targets: Tensor, patch_size: int=16) -> Tensor:
```

```text
Match adaptive patch exposure targets.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    targets: Detached (B,1,H//patch_size,W//patch_size) in [0,1].
    patch_size: Non-overlapping pool size, default 16.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 4; mean squared pooled luminance error, replaces L_exp. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/highlight

Source: [losses/highlight.py](../src/nta_zerodce/losses/highlight.py)

```python
def highlight_loss(image: Tensor, output: Tensor, threshold: float=0.8) -> Tensor:
```

```text
Protect already bright input pixels.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    threshold: Input luminance threshold in [0,1], default 0.8.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 4; masked positive luminance increase squared; empty mask returns graph-connected zero. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/chroma

Source: [losses/chroma.py](../src/nta_zerodce/losses/chroma.py)

```python
def chroma_loss(image: Tensor, output: Tensor, sigma: Tensor, window: int=5, eps: float=1e-06, reliability_low: float=0.05, reliability_high: float=0.2, reliability_noise: float=2.0) -> Tensor:
```

```text
Preserve reliable local chromaticity.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
    window: Odd local averaging width, default 5.
    eps: Positive division guard.
    reliability_low/high: Luminance ramp endpoints, low<high.
    reliability_noise: Sigma multiplier subtracted from input luminance.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 3; masked L1 distance of RGB/sum to detached local input chromaticity. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/edge

Source: [losses/edge.py](../src/nta_zerodce/losses/edge.py)

```python
def edge_loss(image: Tensor, output: Tensor, sigma: Tensor, gain: Tensor, smooth_sigma: float=0.8, kernel_size: int=5, noise_multiplier: float=2.0, eps: float=1e-06) -> Tensor:
```

```text
Match noise-reliable gain-scaled input edges.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    sigma: Nonnegative float32 tensor (B,1,H,W), standard deviation in RGB units.
    gain: Detached (B,1,H,W) in [1,8].
    smooth_sigma: Positive Gaussian standard deviation in pixels.
    kernel_size: Odd Gaussian width, default 5.
    noise_multiplier: Positive edge reliability threshold, default 2.
    eps: Positive gradient/division guard.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 2; normalized Sobel magnitude of output against gain*smoothed input gradient. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/noise

Source: [losses/noise.py](../src/nta_zerodce/losses/noise.py)

```python
def noise_loss(predicted_sigma: Tensor, target_sigma: Tensor) -> Tensor:
```

```text
Supervise the per-pixel noise branch.

Args:
    predicted_sigma: (B,1,H,W) in [0,0.2].
    target_sigma: Detached (B,1,H,W), synthetic unclipped standard deviation.

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 1; mean absolute standard-deviation error. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

```python
def invariance_loss(noisy_output: Tensor, teacher_output: Tensor) -> Tensor:
```

```text
Enforce enhancement invariance to independent added noise.

Args:
    noisy_output: RGB (B,3,H,W) in [0,1], retains graph.
    teacher_output: Detached RGB (B,3,H,W) in [0,1].

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 1; mean |E(x+n)-stopgrad(E(x))|. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

```python
def fidelity_loss(output: Tensor, teacher_output: Tensor) -> Tensor:
```

```text
Anchor synthetic denoising to clean-input curve output.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    teacher_output: Detached clean-input enhanced_raw (B,3,H,W) in [0,1].

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 1; L1 synthetic fidelity; requires denoise, uses no real high-light GT. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/temporal

Source: [losses/temporal.py](../src/nta_zerodce/losses/temporal.py)

```python
def temporal_loss(previous_output: Tensor, current_output: Tensor, backward_flow: Tensor, mask: Tensor) -> Tensor:
```

```text
Penalize reliable flow-aligned frame differences.

Args:
    previous_output/current_output: RGB (B,3,H,W) in [0,1].
    backward_flow: (B,2,H,W), pixel displacement current->previous, x then y.
    mask: Detached reliable weights (B,1,H,W) in [0,1].

Returns:
    Scalar float32 nonnegative differentiable Tensor on the input device.

Method:
    Extension 5; masked L1 current vs warp(previous,backward_flow); no detach on outputs. See docs/METHOD.md; baseline attribution [R1,R2], extensions are project proposals.

```

### losses/registry

Source: [losses/registry.py](../src/nta_zerodce/losses/registry.py)

Class `LossSpec`: Registration contract; INTERFACES loss graph and METHOD weight table.

callable: Tensor-only differentiable loss; inputs specified by required_inputs.
flag: Config activation expression (flag or not flag), or None for unconditional baseline terms.
default_weight: Nonnegative scalar starting weight, not a tuned result.
required_inputs: Ordered names from the loop-step context, including scalar options.
Returns an immutable metadata record; no loss execution at construction.

```python
def register_loss(name: str, flag: str | None, default_weight: float, required_inputs: tuple[str, ...]) -> Callable[[LossCallable], LossCallable]:
```

```text
Register one loss through a decorator.

Args:
    name: Unique registry key.
    flag: Schema activation flag, or None.
    default_weight: Nonnegative scalar.
    required_inputs: Ordered tensor/context argument names.

Returns:
    Decorator preserving the callable and installing one LossSpec.

Method:
    CONTRIBUTING extension checklist; reject duplicate keys and unknown flags.

```

```python
def build_losses(flags: dict[str, bool], weights: dict[str, float]) -> dict[str, LossSpec]:
```

```text
Resolve active loss specifications.

Args:
    flags: Validated feature map, including denoise=>noise_aware.
    weights: Full nonnegative weight map; adaptive_exp inherits exp weight.

Returns:
    Active names and immutable specs; exp and adaptive_exp mutually exclusive.

Method:
    METHOD total objective; missing runtime tensor inputs must be rejected.

```

### training/config

Source: [training/config.py](../src/nta_zerodce/training/config.py)

Class `FeatureFlags`: FeatureFlags fields and defaults; see INTERFACES config table.

Args:
    noise_aware: bool; default False. Units and bounds in INTERFACES.
    denoise: bool; default False. Units and bounds in INTERFACES.
    edge: bool; default False. Units and bounds in INTERFACES.
    chroma: bool; default False. Units and bounds in INTERFACES.
    adaptive_exp: bool; default False. Units and bounds in INTERFACES.
    temporal: bool; default False. Units and bounds in INTERFACES.
    light: bool; default False. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `ModelConfig`: ModelConfig fields and defaults; see INTERFACES config table.

Args:
    channels: int; default 32. Units and bounds in INTERFACES.
    iterations: int; default 8. Units and bounds in INTERFACES.
    noise_channels: int; default 16. Units and bounds in INTERFACES.
    denoise_channels: int; default 16. Units and bounds in INTERFACES.
    sigma_max: float; default 0.2. Units and bounds in INTERFACES.
    residual_scale: float; default 0.1. Units and bounds in INTERFACES.
    curve_scale: float; default 1.0. Units and bounds in INTERFACES.
    gain_window: int; default 5, odd stride-one local pooling width.
    gain_min: float; default 1.0. Units and bounds in INTERFACES.
    gain_max: float; default 8.0. Units and bounds in INTERFACES.
    eps: float; default 1e-6. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `LossWeights`: LossWeights fields and defaults; see INTERFACES config table.

Args:
    spa: float; default 1.0. Units and bounds in INTERFACES.
    exp: float; default 10.0. Units and bounds in INTERFACES.
    col: float; default 5.0. Units and bounds in INTERFACES.
    tv: float; default 200.0. Units and bounds in INTERFACES.
    edge: float; default 20.0. Units and bounds in INTERFACES.
    chroma: float; default 20.0. Units and bounds in INTERFACES.
    highlight: float; default 20.0. Units and bounds in INTERFACES.
    noise: float; default 20.0. Units and bounds in INTERFACES.
    invariance: float; default 10.0. Units and bounds in INTERFACES.
    fidelity: float; default 1.0. Units and bounds in INTERFACES.
    temporal: float; default 10.0. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `LossConfig`: LossConfig fields and defaults; see INTERFACES config table.

Args:
    spatial_patch: int; default 4. Units and bounds in INTERFACES.
    exposure_patch: int; default 16. Units and bounds in INTERFACES.
    exposure_target: float; default 0.6. Units and bounds in INTERFACES.
    adaptive_low: float; default 0.55. Units and bounds in INTERFACES.
    adaptive_high: float; default 0.85. Units and bounds in INTERFACES.
    adaptive_noise_scale: float; default 0.05. Units and bounds in INTERFACES.
    adaptive_noise_penalty: float; default 0.15. Units and bounds in INTERFACES.
    highlight_threshold: float; default 0.8. Units and bounds in INTERFACES.
    chroma_window: int; default 5. Units and bounds in INTERFACES.
    reliability_low: float; default 0.05. Units and bounds in INTERFACES.
    reliability_high: float; default 0.2. Units and bounds in INTERFACES.
    reliability_noise: float; default 2.0. Units and bounds in INTERFACES.
    edge_smooth_sigma: float; default 0.8. Units and bounds in INTERFACES.
    edge_kernel: int; default 5. Units and bounds in INTERFACES.
    edge_noise_multiplier: float; default 2.0. Units and bounds in INTERFACES.
    eps: float; default 1e-6. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `NoiseConfig`: NoiseConfig fields and defaults; see INTERFACES config table.

Args:
    a_min: float; default 0.0. Units and bounds in INTERFACES.
    a_max: float; default 0.01. Units and bounds in INTERFACES.
    b_min: float; default 0.0. Units and bounds in INTERFACES.
    b_max: float; default 0.02. Units and bounds in INTERFACES.
    probability: float; default 1.0. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `SyntheticConfig`: SyntheticConfig fields and defaults; see INTERFACES config table.

Args:
    rotation_degrees: float; default 3.0. Units and bounds in INTERFACES.
    translation_pixels: float; default 8.0. Units and bounds in INTERFACES.
    scale_min: float; default 0.98. Units and bounds in INTERFACES.
    scale_max: float; default 1.02. Units and bounds in INTERFACES.
    flicker_min: float; default 0.9. Units and bounds in INTERFACES.
    flicker_max: float; default 1.1. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `FlowConfig`: FlowConfig fields and defaults; see INTERFACES config table.

Args:
    method: str; default "dis". Units and bounds in INTERFACES.
    preset: str; default "medium". Units and bounds in INTERFACES.
    scale: float; default 0.5. Units and bounds in INTERFACES.
    fb_alpha: float; default 0.01. Units and bounds in INTERFACES.
    fb_beta: float; default 0.5. Units and bounds in INTERFACES.
    photometric_threshold: float; default 0.1. Units and bounds in INTERFACES.
    align_corners: bool; default True. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `TemporalConfig`: TemporalConfig fields and defaults; see INTERFACES config table.

Args:
    curve_blend: float; default 0.7. Units and bounds in INTERFACES.
    output_blend: float; default 0.0. Units and bounds in INTERFACES.
    histogram_bins: int; default 32, normalized scene-cut histogram resolution.
    scene_cut_threshold: float; default 0.3. Units and bounds in INTERFACES.
    pair_source: str; default "synthetic". Units and bounds in INTERFACES.
    frame_gap: int; default 1. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `TrainingConfig`: TrainingConfig fields and defaults; see INTERFACES config table.

Args:
    lr: float; default 1e-4. Units and bounds in INTERFACES.
    batch_size: int; default 8. Units and bounds in INTERFACES.
    crop_size: int; default 256. Units and bounds in INTERFACES.
    epochs: int; default 100. Units and bounds in INTERFACES.
    optimizer: str; default "Adam". Units and bounds in INTERFACES.
    adam_beta1: float; default .9, first-moment decay in [0,1).
    adam_beta2: float; default .999, second-moment decay in [0,1).
    adam_eps: float; default 1e-8, positive denominator stability.
    weight_decay: float; default 1e-4. Units and bounds in INTERFACES.
    grad_clip: float; default 0.1. Units and bounds in INTERFACES.
    seed: int; default 17. Units and bounds in INTERFACES.
    workers: int; default 2. Units and bounds in INTERFACES.
    device: str; default "auto". Units and bounds in INTERFACES.
    checkpoint_every: int; default 5. Units and bounds in INTERFACES.
    log_every: int; default 10. Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `DataPaths`: DataPaths fields and defaults; see INTERFACES config table.

Args:
    train_images: str; default "data/SICE/part1". Units and bounds in INTERFACES.
    eval_low: str; default "data/LOL/eval15/low". Units and bounds in INTERFACES.
    eval_high: str; default "data/LOL/eval15/high". Units and bounds in INTERFACES.
    unpaired: str; default "data/unpaired". Units and bounds in INTERFACES.
    video_dir: str; default "data/videos/train". Units and bounds in INTERFACES.
    run_dir: str; default "runs". Units and bounds in INTERFACES.
    checkpoint_dir: str; default "checkpoints". Units and bounds in INTERFACES.

Returns:
    Immutable schema record; no tensor computation.

Method:
    Project configuration contract, not a paper algorithm.

Class `Config`: Root schema; nested sections below, schema_version=1.

Args: Nested immutable records with no tensor payloads.
Returns: Serializable experiment specification; call validate before execution.
Method: INTERFACES config schema and EXPERIMENTS fixed protocol.

```python
def Config.validate(self) -> None:
```

```text
Validate schema bounds and cross-field constraints.

Args:
    self: Config with every YAML field resolved; reject unknown fields and nonfinite values.

Returns:
    None on success; ValueError for invalid flags, bounds, paths or schema version.

Method:
    INTERFACES validation checklist; denoise implies noise_aware; temporal pair source must be valid.

```

```python
def load_config(path: Path, overrides: dict[str, object] | None=None) -> Config:
```

```text
Load, override and validate a YAML experiment.

Args:
    path: UTF-8 YAML path, resolved relative to the core workspace root.
    overrides: Dotted field names to explicit CLI values; None leaves YAML unchanged.

Returns:
    Validated immutable Config; no model or dataset construction.

Method:
    INTERFACES precedence defaults < YAML < explicit CLI; reject unknown keys.

```

```python
def save_config(config: Config, path: Path) -> None:
```

```text
Serialize the resolved experiment schema.

Args:
    config: Validated Config.
    path: UTF-8 YAML destination; parent creation is caller responsibility.

Returns:
    None; writes all fields, flags and schema_version.

Method:
    INTERFACES round-trip/checkpoint contract; no implicit path rewriting.

```

### data/noise_synth

Source: [data/noise_synth.py](../src/nta_zerodce/data/noise_synth.py)

```python
def sample_noise(image: Tensor, a: Tensor, b: Tensor, generator: Generator | None=None) -> tuple[Tensor, Tensor]:
```

```text
Synthesize Gaussian-approximated Poisson-Gaussian noise.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    a/b: Nonnegative (B,1,1,1); b is read-noise std, not variance.
    generator: Device-matched RNG or None; do not reset global seed.

Returns:
    (clipped noisy RGB, sigma_target (B,1,H,W)); targets are pre-clipping.

Method:
    METHOD extension 1: n=sqrt(a*x+b^2)*epsilon; sigma_target=sqrt(a*meanRGB(x)+b^2).

```

### data/image_dataset

Source: [data/image_dataset.py](../src/nta_zerodce/data/image_dataset.py)

Class `ImageDataset`: Unpaired SICE training and paired LOL evaluation; DATA split rules. Items use unbatched tensors; DATA manifest contract.

```python
def ImageDataset.__init__(self, root: Path, crop_size: int=256, paired_root: Path | None=None) -> None:
```

```text
Index the dataset without loading it eagerly.

Args:
    root: Existing source directory; images or videos according to dataset type.
    crop_size: Positive multiple of 16; train only.
    paired_root: Optional aligned high-reference root for image evaluation only.

Returns:
    No value; future implementation records stable sorted paths and split manifests.

Method:
    DATA layouts; image eval preserves size, video pairs use gap=1 by default.

```

```python
def ImageDataset.__len__(self) -> int:
```

```text
Count indexed samples.

Args:
    self: Indexed dataset.

Returns:
    Nonnegative sample count; empty training roots must fail during initialization.

Method:
    DATA deterministic index, no sampling during length queries.

```

```python
def ImageDataset.__getitem__(self, index: int) -> dict[str, Tensor]:
```

```text
Decode one indexed sample.

Args:
    index: Integer in [0,len(self)); reject out of range.

Returns:
    Image: image (3,H,W), optional reference; video: image1/image2 (3,H,W), index scalar.

Method:
    INTERFACES dataset batch keys; RGB [0,1] CPU, shared pair geometry.

```

### data/video_pairs

Source: [data/video_pairs.py](../src/nta_zerodce/data/video_pairs.py)

Class `VideoPairDataset`: Adjacent real-video pairs; flow is delegated exclusively to flow/estimator. Items use unbatched tensors; DATA manifest contract.

```python
def VideoPairDataset.__init__(self, root: Path, crop_size: int=256, paired_root: Path | None=None, frame_gap: int=1) -> None:
```

```text
Index the dataset without loading it eagerly.

Args:
    root: Existing source directory; images or videos according to dataset type.
    crop_size: Positive multiple of 16; train only.
    paired_root: Must be None for this real-video dataset; no high-light references.
    frame_gap: Positive source-frame separation, default 1.

Returns:
    No value; future implementation records stable sorted paths and split manifests.

Method:
    DATA layouts; image eval preserves size, video pairs use gap=1 by default.

```

```python
def VideoPairDataset.__len__(self) -> int:
```

```text
Count indexed samples.

Args:
    self: Indexed dataset.

Returns:
    Nonnegative sample count; empty training roots must fail during initialization.

Method:
    DATA deterministic index, no sampling during length queries.

```

```python
def VideoPairDataset.__getitem__(self, index: int) -> dict[str, Tensor]:
```

```text
Decode one indexed sample.

Args:
    index: Integer in [0,len(self)); reject out of range.

Returns:
    Image: image (3,H,W), optional reference; video: image1/image2 (3,H,W), index scalar.

Method:
    INTERFACES dataset batch keys; RGB [0,1] CPU, shared pair geometry.

```

### data/synthetic_pairs

Source: [data/synthetic_pairs.py](../src/nta_zerodce/data/synthetic_pairs.py)

```python
def make_pair(image: Tensor, config: Config, generator: Generator | None=None) -> dict[str, Tensor]:
```

```text
Create synthetic temporal supervision.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    config: Validated synthetic/noise/flow constants.
    generator: Device-matched RNG; independent draws for each frame.

Returns:
    image1/image2 RGB (B,3,H,W); forward/backward_flow (B,2,H,W); mask (B,1,H,W).

Method:
    METHOD extension 5: affine inverse coordinate maps, exposure flicker and independent noise; DATA generation.

```

### data/transforms

Source: [data/transforms.py](../src/nta_zerodce/data/transforms.py)

```python
def normalize_rgb(image: Tensor) -> Tensor:
```

```text
Convert decoded RGB bytes to normalized float32.

Args:
    image: RGB uint8 (3,H,W) in [0,255]; BGR conversion must happen at decoding.

Returns:
    RGB float32 (3,H,W) in [0,1], same device.

Method:
    INTERFACES IO range contract; no gamma or white-balance correction.

```

```python
def paired_crop(first: Tensor, second: Tensor, crop_size: int=256, generator: Generator | None=None) -> tuple[Tensor, Tensor]:
```

```text
Crop both frames using one sampled origin.

Args:
    first/second: Matching RGB (3,H,W) in [0,1].
    crop_size: Positive multiple of 16, <=min(H,W).
    generator: Device-matched RNG.

Returns:
    Two aligned RGB (3,crop_size,crop_size) tensors.

Method:
    DATA shared geometry; crop before computing flow, or transform cached flow consistently.

```

### flow/estimator

Source: [flow/estimator.py](../src/nta_zerodce/flow/estimator.py)

```python
def estimate_flow(first: Tensor, second: Tensor, config: FlowConfig) -> Tensor:
```

```text
Estimate source-to-target pixel displacement.

Args:
    first/second: CPU RGB float32 (B,3,H,W) in [0,1], matching size.
    config: DIS method/preset, flow scale in (0,1].

Returns:
    CPU float32 flow (B,2,H,W), dx,dy in original pixels; detached.

Method:
    INTERFACES: first(p) ~ second(p+flow(p)); DIS [R6]. Resize vectors by full/small dimensions.

```

```python
def estimate_bidirectional(first: Tensor, second: Tensor, config: FlowConfig) -> tuple[Tensor, Tensor]:
```

```text
Estimate forward and backward flow.

Args:
    first/second: CPU RGB (B,3,H,W) in [0,1].
    config: Validated DIS settings.

Returns:
    (first->second, second->first), each CPU (B,2,H,W) in original pixel units.

Method:
    METHOD extension 5; compute both independently, never negate forward flow.

```

### flow/warp

Source: [flow/warp.py](../src/nta_zerodce/flow/warp.py)

```python
def warp(source: Tensor, flow: Tensor, align_corners: bool=True) -> tuple[Tensor, Tensor]:
```

```text
Sample a target-domain source into the flow origin coordinates.

Args:
    source: Float32 (B,C,H,W), arbitrary C, same device as flow.
    flow: Float32 (B,2,H,W), origin->source displacement dx,dy in pixels.
    align_corners: True by default; fixed in training and inference.

Returns:
    (warped source (B,C,H,W), valid mask (B,1,H,W)); bilinear, zero padding.

Method:
    INTERFACES: warp(b,f(a,b))(p)=b(p+f(p)); METHOD normalized grid formula.

```

### flow/masks

Source: [flow/masks.py](../src/nta_zerodce/flow/masks.py)

```python
def reliability_mask(forward_flow: Tensor, backward_flow: Tensor, previous: Tensor, current: Tensor, alpha: float=0.01, beta: float=0.5, photometric_threshold: float=0.1) -> Tensor:
```

```text
Compute a current-coordinate reliable-pixel mask.

Args:
    forward_flow/backward_flow: (B,2,H,W), prev->current and current->prev.
    previous/current: RGB (B,3,H,W) in [0,1].
    alpha: Nonnegative relative FB threshold.
    beta: Nonnegative threshold in squared pixels.
    photometric_threshold: Mean-RGB residual threshold in [0,1].

Returns:
    Detached mask (B,1,H,W), float32 0 or 1; combines FB, bounds and photometric tests.

Method:
    METHOD extension 5; compare backward + warp(forward,backward), in current coordinates.

```

```python
def is_scene_cut(previous: Tensor, current: Tensor, threshold: float=0.3, bins: int=32) -> bool:
```

```text
Detect a scene cut before reusing temporal state.

Args:
    previous/current: One RGB frame (1,3,H,W) in [0,1], same size.
    threshold: Half-L1 normalized histogram threshold in [0,1].
    bins: Number of mean-RGB histogram bins, integer >=2, default 32.

Returns:
    True if normalized mean-RGB bins-bin histogram distance exceeds threshold.

Method:
    METHOD extension 5; conservative project heuristic, not a learned cut detector.

```

### inference/enhancer

Source: [inference/enhancer.py](../src/nta_zerodce/inference/enhancer.py)

```python
def enhance_image(model: NTAZeroDCE, image: Tensor, curve_scale: float=1.0) -> ModelOutput:
```

```text
Enhance RGB using inference-only model execution.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    model: Eval-mode module on the image device.
    curve_scale: In (0,1]; estimate sigma/A at reduced size and resize maps.

Returns:
    Full-resolution ModelOutput; no gradients, RGB in [0,1].

Method:
    REAL_TIME downsample estimator then apply eight curves to original RGB; no resized final RGB.

```

### inference/video_enhancer

Source: [inference/video_enhancer.py](../src/nta_zerodce/inference/video_enhancer.py)

Class `VideoState`: One stream's detached state; METHOD temporal inference.

previous_image/output: Optional RGB (1,3,H,W) in [0,1].
previous_A: Optional full-resolution curve (1,24,H,W) in [-1,1].
All share device and dimensions; None denotes new stream or cut/reset.
Returns a state record, not an executed enhancer. No cross-stream sharing.

Class `VideoEnhancer`: Sequential one-stream enhancement; caller resets between videos.

```python
def VideoEnhancer.__init__(self, model: NTAZeroDCE, config: Config) -> None:
```

```text
Construct a video enhancer with empty state.

Args:
    model: Loaded eval-mode model.
    config: Validated temporal, curve-scale and flow settings.

Returns:
    No value; future implementation owns VideoState.

Method:
    ARCHITECTURE inference graph; temporal flag false is frame-independent.

```

```python
def VideoEnhancer.reset(self) -> None:
```

```text
Clear history on stream changes or detected cuts.

Args:
    self: Enhancer associated with one stream.

Returns:
    None; clear all three prior tensors.

Method:
    METHOD scene cut handling; resolution changes also force reset.

```

```python
def VideoEnhancer.enhance_frame(self, image: Tensor) -> ModelOutput:
```

```text
Enhance one frame with optional reliable curve blending.

Args:
    image: RGB (1,3,H,W) in [0,1] on model device.

Returns:
    Full-resolution ModelOutput; update detached previous state.

Method:
    METHOD backward current->previous flow, mask-aware convex A blend and optional output blend.

```

### inference/webcam

Source: [inference/webcam.py](../src/nta_zerodce/inference/webcam.py)

```python
def run_webcam(model: NTAZeroDCE, config: Config, camera_id: int=0) -> None:
```

```text
Run a real-time webcam demonstration.

Args:
    model: Loaded eval-mode model on config device.
    config: Validated inference/flow settings.
    camera_id: Nonnegative OpenCV capture index.

Returns:
    None; release capture/window resources on exit or interruption.

Method:
    REAL_TIME budget/profiling; sequential VideoEnhancer state and RGB/BGR boundary.

```

### inference/io

Source: [inference/io.py](../src/nta_zerodce/inference/io.py)

```python
def read_image(path: Path) -> Tensor:
```

```text
Decode one image into RGB.

Args:
    path: Existing PNG/JPEG path; reject grayscale/alpha ambiguity or explicitly convert to RGB.

Returns:
    CPU RGB float32 (1,3,H,W) in [0,1].

Method:
    INTERFACES IO; ignore EXIF orientation only if documented; planned policy applies orientation.

```

```python
def write_image(image: Tensor, path: Path) -> None:
```

```text
Encode one normalized RGB result.

Args:
    image: RGB float32 (1,3,H,W) in [0,1]; may be on GPU.
    path: PNG/JPEG output path; create parent only in CLI runner.

Returns:
    None; detach, move to CPU and quantize once at save boundary.

Method:
    INTERFACES IO; round(255*x), PNG preferred for evaluation.

```

```python
def read_video(path: Path) -> Iterator[Tensor]:
```

```text
Stream RGB frames in presentation order.

Args:
    path: Existing video path; no frame resampling unless requested.

Returns:
    Iterator of CPU RGB (1,3,H,W) in [0,1]; close decoder on exhaustion/error.

Method:
    DATA video protocol; preserve original timestamps in downstream metadata.

```

```python
def write_video(frames: Iterator[Tensor], path: Path, fps: float) -> None:
```

```text
Encode sequential enhanced RGB frames.

Args:
    frames: RGB (1,3,H,W) in [0,1], constant dimensions.
    path: Output video path.
    fps: Positive source frame rate.

Returns:
    None; no audio copying promised at M0; release encoder on error.

Method:
    INTERFACES IO; timing and video-metric quantization caveats.

```

### metrics/fidelity

Source: [metrics/fidelity.py](../src/nta_zerodce/metrics/fidelity.py)

```python
def psnr(output: Tensor, reference: Tensor) -> Tensor:
```

```text
Measure peak signal-to-noise ratio.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    reference: Registered RGB (B,3,H,W) in [0,1], same device.

Returns:
    Per-image (B,) dB values; +infinity on exact equality.

Method:
    EXPERIMENTS: 10*log10(1/MSE); higher is better.

```

```python
def ssim(output: Tensor, reference: Tensor) -> Tensor:
```

```text
Measure structural similarity.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    reference: Registered RGB (B,3,H,W) in [0,1].

Returns:
    Per-image (B,) SSIM, usually [-1,1]; fixed RGB averaging policy.

Method:
    EXPERIMENTS: Gaussian 11x11 sigma=1.5, K1=.01 K2=.03, data_range=1; [R10].

```

### metrics/noise

Source: [metrics/noise.py](../src/nta_zerodce/metrics/noise.py)

```python
def noise_estimate(image: Tensor, flat_mask: Tensor | None=None) -> Tensor:
```

```text
Estimate flat-region Gaussian noise standard deviation.

Args:
    image: RGB float32 tensor (B,3,H,W) in [0,1]; no implicit device transfer.
    flat_mask: Optional (B,1,H,W) binary valid region; no reliable pixels => NaN.

Returns:
    Per-image (B,) normalized RGB-unit sigma estimates.

Method:
    EXPERIMENTS: Immerkaer kernel and calibration [R8]; texture causes bias.

```

### metrics/colour

Source: [metrics/colour.py](../src/nta_zerodce/metrics/colour.py)

```python
def angular_error(output: Tensor, reference: Tensor, eps: float=1e-06) -> Tensor:
```

```text
Measure RGB angular error on reliable reference pixels.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    reference: RGB (B,3,H,W) in [0,1], aligned.
    eps: Positive norm guard; exclude near-black vectors.

Returns:
    Per-image (B,) mean degrees, lower better; empty reliable set => NaN.

Method:
    EXPERIMENTS acos(clamp(dot/(norms),-1,1)); not perceptual Delta-E.

```

### metrics/exposure

Source: [metrics/exposure.py](../src/nta_zerodce/metrics/exposure.py)

```python
def clipping_fraction(output: Tensor, threshold: float=0.99) -> Tensor:
```

```text
Measure near-saturated luminance fraction.

Args:
    output: RGB float32 tensor (B,3,H,W) in [0,1], on the input device.
    threshold: In [0,1], default .99.

Returns:
    Per-image (B,) fraction [0,1], lower conditionally better.

Method:
    EXPERIMENTS: mean(Y(output)>=threshold); report input clipping alongside output.

```

### metrics/temporal

Source: [metrics/temporal.py](../src/nta_zerodce/metrics/temporal.py)

```python
def warping_error(previous: Tensor, current: Tensor, backward_flow: Tensor, mask: Tensor) -> Tensor:
```

```text
Measure per-pair flow-aligned temporal L1 error.

Args:
    previous/current: RGB (B,3,H,W) in [0,1].
    backward_flow: Current->previous (B,2,H,W) pixel units.
    mask: Detached (B,1,H,W) in [0,1].

Returns:
    Per-pair (B,) normalized L1; empty masks => NaN for metrics, lower better.

Method:
    EXPERIMENTS temporal metric; also report mask coverage to expose flow rejection.

```

### training/trainer

Source: [training/trainer.py](../src/nta_zerodce/training/trainer.py)

```python
def train(config: Config, resume: Path | None=None) -> Path:
```

```text
Run the configured training schedule.

Args:
    config: Validated Config with dataset paths and gated objectives.
    resume: Optional checkpoint; exact flag/schema compatibility required.

Returns:
    Final checkpoint Path; persist resolved config, seed and metrics CSV.

Method:
    EXPERIMENTS 100 epochs, Adam, fixed seeds; loop_step owns updates.

```

### training/loop_step

Source: [training/loop_step.py](../src/nta_zerodce/training/loop_step.py)

```python
def train_step(model: NTAZeroDCE, batch: dict[str, Tensor], optimizer: Optimizer, config: Config) -> dict[str, float]:
```

```text
Perform exactly one gated optimization step.

Args:
    model: Train-mode model on batch device.
    batch: image (B,3,H,W) in [0,1]; temporal fields as INTERFACES specifies.
    optimizer: Adam bound to model parameters.
    config: Validated flags/weights and gradient norm clip.

Returns:
    Detached Python scalars: total and every loss name, finite for valid batches.

Method:
    METHOD objective; teacher/masks/gain/targets detached, all active student paths differentiable.

```

### training/checkpoint

Source: [training/checkpoint.py](../src/nta_zerodce/training/checkpoint.py)

```python
def save_checkpoint(path: Path, model: NTAZeroDCE, optimizer: Optimizer, config: Config, epoch: int) -> None:
```

```text
Save reproducible training state.

Args:
    path: Checkpoint destination.
    model/optimizer: Current state owners.
    config: Fully resolved validated Config.
    epoch: Completed zero-based epoch index >=0.

Returns:
    None; future file contains model/optimizer states, config, flags, schema_version, epoch and RNG states.

Method:
    TESTING flag/config round-trip; atomic write and explicit checkpoint format version 1.

```

```python
def load_checkpoint(path: Path, model: NTAZeroDCE, optimizer: Optimizer | None=None, expected_config: Config | None=None) -> Config:
```

```text
Load compatible model and optional optimizer state.

Args:
    path: Trusted local checkpoint; no network loading.
    model: Destination architecture on explicit caller device.
    optimizer: Optional resume optimizer.
    expected_config: If set, reject flag/shape/schema mismatches.

Returns:
    Restored resolved Config; require exact checkpoint architecture flags.

Method:
    INTERFACES checkpoints; portable CPU map_location then explicit caller transfer.

```

### training/logging

Source: [training/logging.py](../src/nta_zerodce/training/logging.py)

```python
def log_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
```

```text
Append one stable-schema CSV log record.

Args:
    path: Run-local CSV destination.
    step: Nonnegative global step.
    metrics: Stable named finite scalar values; missing evaluations use NaN explicitly.

Returns:
    None; include run/seed through run directory and resolved config.

Method:
    EXPERIMENTS reporting; do not silently mix train/eval columns.

```

### utils/seed

Source: [utils/seed.py](../src/nta_zerodce/utils/seed.py)

```python
def seed_everything(seed: int, deterministic: bool=True) -> None:
```

```text
Seed Python, NumPy and PyTorch random sources.

Args:
    seed: Nonnegative integer.
    deterministic: Request deterministic algorithms with explicit unsupported-op failures.

Returns:
    None; future implementation seeds CPU/CUDA and documents worker seeding.

Method:
    EXPERIMENTS [17,29,43]; determinism does not guarantee identical cross-device results.

```

### utils/device

Source: [utils/device.py](../src/nta_zerodce/utils/device.py)

```python
def resolve_device(requested: str='auto') -> device:
```

```text
Resolve a requested compute device.

Args:
    requested: auto, cpu or cuda[:index]; explicit unavailable CUDA must fail.

Returns:
    torch.device; auto chooses available CUDA else CPU.

Method:
    INTERFACES device rules; OpenCV flow remains CPU then caller transfers.

```

### utils/registry

Source: [utils/registry.py](../src/nta_zerodce/utils/registry.py)

Class `Registry`: Typed unique-name lookup; no tensor or paper-specific computation.

```python
def Registry.__init__(self) -> None:
```

```text
Construct an empty typed registry.

Args:
    self: Generic Registry[T].

Returns:
    No value; future implementation owns a string-to-T mapping.

Method:
    CONTRIBUTING registration convention.

```

```python
def Registry.register(self, name: str, value: T) -> None:
```

```text
Register one unique name.

Args:
    name: Nonempty unique identifier.
    value: Metadata or callable of generic type T.

Returns:
    None; ValueError on duplicates.

Method:
    CONTRIBUTING fail-fast duplicate policy.

```

```python
def Registry.get(self, name: str) -> T:
```

```text
Retrieve registered metadata.

Args:
    name: Previously registered identifier.

Returns:
    T; KeyError for unknown name.

Method:
    CONTRIBUTING explicit lookup.

```

## Script run boundaries

### scripts/train.py

[train](../scripts/train.py)

```python
def run(args: argparse.Namespace) -> None:
```

```text
Execute the train command after resolving validated settings.

Args:
    args: Parsed CLI namespace; explicit overrides use None when absent.

Returns:
    None; delegation described below.

Method:
    INTERFACES CLI: load_config -> trainer.train; --video-dir overrides paths.video_dir.

```

### scripts/evaluate.py

[evaluate](../scripts/evaluate.py)

```python
def run(args: argparse.Namespace) -> None:
```

```text
Execute the evaluate command after resolving validated settings.

Args:
    args: Parsed CLI namespace; explicit overrides use None when absent.

Returns:
    None; delegation described below.

Method:
    INTERFACES CLI: load_config/build_model/load_checkpoint -> paired/unpaired/video metrics -> CSV.

```

### scripts/ablate.py

[ablate](../scripts/ablate.py)

```python
def run(args: argparse.Namespace) -> None:
```

```text
Execute the ablate command after resolving validated settings.

Args:
    args: Parsed CLI namespace; explicit overrides use None when absent.

Returns:
    None; delegation described below.

Method:
    INTERFACES CLI: matrix inheritance -> seed/flag overrides -> trainer.train and evaluate.

```

### scripts/enhance.py

[enhance](../scripts/enhance.py)

```python
def run(args: argparse.Namespace) -> None:
```

```text
Execute the enhance command after resolving validated settings.

Args:
    args: Parsed CLI namespace; explicit overrides use None when absent.

Returns:
    None; delegation described below.

Method:
    INTERFACES CLI: load_config/build_model/load_checkpoint -> image IO, VideoEnhancer or run_webcam.

```

### scripts/make_synthetic_data.py

[make_synthetic_data](../scripts/make_synthetic_data.py)

```python
def run(args: argparse.Namespace) -> None:
```

```text
Execute the make_synthetic_data command after resolving validated settings.

Args:
    args: Parsed CLI namespace; explicit overrides use None when absent.

Returns:
    None; delegation described below.

Method:
    INTERFACES CLI: load_config -> dataset index -> make_pair/sample_noise -> local generated output.

```

Additional schema bounds: model.gain_window is positive odd; temporal.histogram_bins
is an integer >=2; training.adam_beta1/adam_beta2 lie in [0,1), adam_eps>0.
These constants are explicit in both complete YAMLs and are untuned defaults.

## Implemented additions

APIs added during implementation (M1-M5) that the M0 inventory above does not list.

| Area | API | Contract |
| --- | --- | --- |
| models | `NTAZeroDCE.estimate_maps(image) -> (A, sigma)` | Curve maps and noise map at the input resolution; no curve applied. |
| models | `NTAZeroDCE.apply_maps(image, A, sigma) -> ModelOutput` | Applies given (possibly resized) maps; `forward` is `apply_maps(*estimate_maps(x))`. |
| models | `compute_gain(...)` | Adaptive-exposure gain used for highlight protection (`gain_window`, `gain_min`, `gain_max`). |
| inference | `estimate_full_res_maps(model, image, curve_scale)` | Estimates at `curve_scale`, bilinearly upsamples A and sigma, no gradients. |
| inference | `enhance_image(model, image, curve_scale)` | Full-resolution `ModelOutput`; equals `forward` at `curve_scale == 1`. |
| inference | `load_inference_model(checkpoint, config_path=None, overrides=None, device="auto")` | `(model, config, device)`; the checkpoint's stored config is the default. |
| inference | `VideoEnhancer` diagnostics | Per-frame scene-cut, flow-reliability and blend statistics for temporal curve smoothing. |
| inference | `run_webcam(...)` | Capture/enhance/display loop with FPS and latency overlay; camera and window are injectable for tests. |
| training | `save_checkpoint(path, model, optimizer, config, epoch, *, provenance=None)` | Atomic write, format v1, RNG state captured for exact resume. |
| training | `read_checkpoint`, `checkpoint_config`, `load_checkpoint_full -> LoadedCheckpoint`, `load_checkpoint` | Loads with `weights_only=True`; `LoadedCheckpoint(config, epoch, provenance)`. |
| training | `config_to_dict`, `config_from_dict`, `override_config` | Plain-dict round trip and dotted-key overrides with full re-validation. |
| training | `prepare_batch` | Builds the loss context (images, pair fields, flow, masks) from a dataset batch. |
| cli | `nta_zerodce.cli` | Shared flag/override parsing and `main_guard` exit codes: 2 usage, 1 runtime, 130 interrupt. |
| api | `nta_zerodce.api.create_app(Settings)` | FastAPI app; see [API](API.md). |

## Implemented CLI behaviour

- `scripts/train.py`: writes `runs/<run>/` (config snapshot, per-epoch CSV logs) and
  `checkpoints/<run>/epoch_NNNN.pt` plus `final.pt`; `--resume` is bit-exact for model,
  optimizer and RNG; `--epochs` shortens smoke runs.
- `scripts/evaluate.py`: CSV with the `EVAL_COLUMNS` of `training/evaluation.py`, one row
  per image plus an aggregate row; temporal metrics need held-out videos.
- `scripts/ablate.py`: expands the three-seed matrix, trains and evaluates each row, and
  writes a combined CSV; rows are resumable.
- `scripts/enhance.py`: image, folder (layout preserved), video (temporal smoothing when
  `flags.temporal`; audio is not copied) or `--webcam`; output is PNG.
- `scripts/make_synthetic_data.py`: procedural or source-based noisy images, or `--pairs`
  synthetic temporal pairs, each with a JSON sidecar; deterministic for a seed and refuses to
  overwrite without `--overwrite`.
- `python -m nta_zerodce.api`: HTTP server configured through `LUMINA_*` variables.
