# Testing strategy

Run Python checks from core/. The repository-root CI workflow runs an independent
web job for the empty Next.js workspace. The frontend's npm typecheck/build do not
run Python training or require datasets/checkpoints.

## M0 verification

Packaging, ruff lint/format, strict project mypy, CLI help and collection-time structural
audits are executable. All behavioral test functions are strict xfail and raise
NotImplementedError; xfail catches only that exception, not arbitrary failures.
An xfailed plan is not evidence of tested algorithm behavior. CI runs on CPU;
no dataset or GPU is required. make docs-check collects the schema test module,
whose top-level assertions verify every local Markdown link in README/docs.
Directories count as valid local paths; runtime data/checkpoint examples in fenced
commands are intentionally not required to exist. All referenced repository files
use Markdown links so the audit can see them. Link anchors/external URL availability
are outside the local file check.

Active collection audits also verify complete YAML keys against dataclass defaults,
valid ablation denoise/noise dependency, 13 rows/three seeds, and that every source
and CLI function is a detailed docstring followed only by a TODO raise. Dataclass,
TypedDict and registry declarations are structural interfaces, not implemented
algorithm bodies. CLI argparse is module-level wiring. Documentation link errors
fail collection even with --collect-only; no additional tooling file was needed.

## Future assertions by file

| Test file | Assertions to implement |
| --- | --- |
| [model_shapes](../tests/test_model_shapes.py) | baseline/full/light shapes, channel ordering, finite ranges, no transfers, full-resolution maps at curve_scale<1, branch gradients |
| [curve_math](../tests/test_curve_math.py) | A=0 identity, 0/1 fixed points, .25 with +/-.5 sign examples, boundedness, fixed-A monotonicity, finite gradients |
| [losses_signatures](../tests/test_losses_signatures.py) | scalar/device/gradient contracts, baseline constant-gray identities, masks, no detached student, mutual exposure gate |
| [flow_conventions](../tests/test_flow_conventions.py) | known translation and nontranslation affine, inverse direction, x/y scaling, warp bounds, FB mask coordinates |
| [flags_roundtrip](../tests/test_flags_roundtrip.py) | explicit-false CLI precedence, schema/YAML/checkpoint flags, invalid denoise and incompatible architecture rejection |
| [config_schema](../tests/test_config_schema.py) | all keys/types/bounds, nonfinite rejection, scale/patch restrictions, unknown-field rejection, same ablation inheritance |

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
```

Once a behavior exists, remove its matching xfail and replace the raise with real
assertions. Never broaden xfail to hide bugs. M1..M6 introduces implementation;
the M0 stub-only AST audit must then be intentionally retired or narrowed alongside
the status table and roadmap update. CI green at M0 guarantees scaffold integrity,
not training/enhancement correctness. The remote workflow still needs an actual
GitHub push/run before a hosted success can be claimed.

At M0, mypy treats third-party torch imports as opaque and permits inheritance
from their nn.Module/Dataset bases; every project argument/return is still checked
under strict settings. Restore full torch type analysis when computational bodies
are implemented. YAML parsing used by audits has a scoped missing-stubs override.
