# Testing strategy

Run Python checks from core/. The repository-root CI workflow runs independent core
(Python 3.10 and 3.12), web and Docker-build jobs. The frontend's npm checks do not
run Python training or require datasets/checkpoints.

## What is verified

`make lint typecheck test docs-check` runs ruff (lint and format), strict mypy over source,
scripts and tests, and the full pytest suite (about 2,060 tests on CPU; CUDA-only tests skip
without a GPU). No dataset, checkpoint or network access is required: tests build tiny
synthetic images, videos and checkpoints in temporary directories.

| Layer | Test modules |
| --- | --- |
| Math and models | [curve_math](../tests/test_curve_math.py), [model_shapes](../tests/test_model_shapes.py), [losses_signatures](../tests/test_losses_signatures.py), [flow_conventions](../tests/test_flow_conventions.py), [metrics](../tests/test_metrics.py) |
| Data and IO | [data](../tests/test_data.py), [io](../tests/test_io.py), [utils](../tests/test_utils.py) |
| Config and flags | [config_schema](../tests/test_config_schema.py), [flags_roundtrip](../tests/test_flags_roundtrip.py) |
| Training | [training](../tests/test_training.py), [checkpoint](../tests/test_checkpoint.py), [scripts_training](../tests/test_scripts_training.py) |
| Inference | [inference](../tests/test_inference.py), [scripts_inference](../tests/test_scripts_inference.py) |
| HTTP API | [api](../tests/test_api.py) |
| End to end | [e2e](../tests/test_e2e.py): train CLI, checkpoint, enhance CLI and API give identical pixels |

`make smoke` is the same pipeline as a shell recipe (synthetic data, a three-epoch CPU
training run, enhancement, output check). The web workspace has its own vitest suite
(`npm test` in web/) covering the API client, error mapping, the proxy route handler and
the page components; its CI job also runs typecheck, lint, build and `npm audit`.

`make docs-check` collects [config_schema](../tests/test_config_schema.py), whose top-level
assertions verify every local Markdown link in README/docs, that YAML configs carry exactly
the dataclass keys, the 13-row/three-seed ablation matrix, and that every source function is
annotated. Link anchors and external URLs are outside the local file check. Directories
count as valid local paths; runtime data/checkpoint examples in fenced commands are
intentionally not required to exist.

Passing tests establish that the code matches its documented contracts, not that the model
reaches any quality target: no trained weights, benchmark metrics or FPS figures are
claimed. See [EXPERIMENTS](EXPERIMENTS.md) and [REAL_TIME](REAL_TIME.md).

## Known-affine test recipe

Create a textured coordinate-grid RGB image at 64x80; translate by (+3,-2) and
also test a rotation about centre. Use exact analytic flow, not estimated DIS, for
the deterministic warp assertion. Compare away from padding with bilinear tolerance
1e-5. For rotation, build inverse maps and prove warp(forward,backward)+backward
is near zero on reliable samples. The backward field lives in current coordinates;
negating arbitrary forward values must not pass. DIS accuracy is a separate
tolerance-based adapter test on a visible synthetic texture.

## Tiny overfit and smoke tests (M1+)

Use a fixed seeded batch of two 32x32 synthetic low-light images on CPU, no real
data or downloads. Over 20..50 fixed optimizer steps check finite loss/gradients,
nonzero curve-head gradient and improvement of exposure error against the initial
value, with a declared tolerance; do not demand monotonically decreasing each
loss. Use test-specific lr only when recorded explicitly; this is a sanity test,
not a training-quality experiment. Smoke a baseline train/save/load/enhance cycle,
then one full pair update at M5. Assert restored flags and comparable outputs.

## Mask, precision and state checks

Test all-zero masks return graph-connected zero without NaN; metric empty masks
return NaN/coverage zero. Test noise sigma supervision with a=b=0 and known positive
variance away from clipping. Test detached adaptive targets so loss cannot lower
targets by inflating sigma. Cut/new stream/changed dimensions must clear previous
RGB/A/output; temporal false must equal framewise inference. Check CPU and optional
CUDA with float32 and tolerance chosen per operation, not bit equality across devices.

## Check commands and completion

```sh
make setup
make lint typecheck test docs-check
python scripts/enhance.py --help
make smoke
```

Third-party torch imports are treated as opaque by mypy (`follow_imports = skip`) while every
project argument and return stays strictly checked. mypy targets Python 3.12 because numpy 2.5
stubs use 3.12 syntax; runtime compatibility with 3.10 is enforced by ruff's `target-version`
and the 3.10 CI job. The hosted CI workflow has not been observed running from this
repository yet.
