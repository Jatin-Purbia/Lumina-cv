# Testing strategy

Run checks from core/ after installing the editable package with dev dependencies:

```sh
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
python -m pytest --collect-only -q tests/test_config_schema.py
```

No dataset, checkpoint, GPU or camera is required. Tests generate local images and a short video; OpenCV video encoding must be available. In a restricted Windows runner, use an accessible temporary directory or an authorized process outside the sandbox if pytest's private temporary-directory permissions are inaccessible.

| Test file | Coverage |
| --- | --- |
| [curves](../tests/test_curve_math.py) | Identity, plus/minus sign, endpoints, bounds, fixed-map monotonicity and gradients |
| [models](../tests/test_model_shapes.py) | Baseline/full/light, scaled curve estimation, output shapes/ranges, disabled branches and branch gradients |
| [losses](../tests/test_losses_signatures.py) | Scalar/device/student-gradient contracts, baseline gray diagnostics, detached adaptive targets, graph-connected zero masks, known synthetic variance |
| [flow](../tests/test_flow_conventions.py) | Translation direction/pixel units, analytic affine inverse, bounds, reliability masks, identical-frame DIS, temporal reset behavior |
| [flags/checkpoints](../tests/test_flags_roundtrip.py) | YAML/CLI explicit-false precedence, adaptive replacement gate, strict checkpoint keys/flags, optimizer/RNG restoration |
| [schema](../tests/test_config_schema.py) | Wrong types, nonfinite/bounds/dependency rejection, malformed matrices, complete YAML keys, local Markdown links and loss metadata |
| [runtime](../tests/test_runtime.py) | 30-step baseline brightness improvement, full synthetic pair update, train/save/load/CLI-enhance/CSV evaluation, image pairing, stable log schema, video roundtrip |

The baseline smoke test uses a fixed pair of 32x32 low-light tensors and a test-specific Adam learning rate .003 for 30 updates. It requires exposure error to improve by at least 20%; it does not require each loss to decrease every step. Test model widths are four channels to keep CPU runtime small. Research configurations retain their original 32-channel width, lr 1e-4 and 100 epochs.

The old stub-only AST requirement and placeholder xfails have been retired. Collection still checks repository Markdown targets and complete config/ablation metadata. Local checks do not establish a hosted CI result, research quality, real-camera noise calibration, or GPU determinism. PyTorch grid_sample backward on CUDA is nondeterministic; deterministic mode warns for unsupported operations. Camera/display operation, real moving-scene DIS quality, CUDA/MPS and long training remain unverified.
