# Implementation plan (M0 to working end-to-end system)

Branch: feat/end-to-end-implementation. Goal: turn the documented scaffold into a
tested, working system: train on image data (synthetic fallback), enhance images
and video, serve over HTTP, and drive it from the Next.js demo.

The normative contracts stay [METHOD](METHOD.md), [INTERFACES](INTERFACES.md) and
[ARCHITECTURE](ARCHITECTURE.md). If an implementation needs a contract change, the
agent records it in its report and the lead updates the docs, never silently.

## Ground rules for every agent

1. Edit only the files you own (tables below). Shared files (pyproject.toml,
   Makefile, CI, README, docs) belong to the lead; report change requests.
2. Do not run git commit or push. The lead commits after each phase.
3. Implement the exact signatures in INTERFACES. Keep docstrings; remove the TODO
   raises. Keep ruff (line length 100), strict mypy, and Python 3.10 compatibility.
4. Replace every strict-xfail contract test you own with real assertions. Never
   broaden or keep an xfail to hide a bug.
5. Tests are CPU-only, deterministic (seeded), small (images at most 64x64 unless
   stated), and need no downloads or datasets.
6. Source comes from the published mathematics in METHOD only. Do not copy any
   Zero-DCE reference code (see NOTICE).
7. Run from core/ with python. Your own tests plus ruff and
   mypy on your files must pass before you report.

## Phase 1 (parallel, disjoint files)

| Agent | Owns | Delivers |
| --- | --- | --- |
| P1-A models and baseline losses | models/*, losses/{spatial,exposure,color,smoothness,adaptive_exposure,highlight,chroma,edge,noise,registry}.py, tests/test_model_shapes.py, tests/test_curve_math.py, tests/test_losses_signatures.py | DCENet, LightDCENet, apply_curves, NoiseBranch, ResidualDenoiser, NTAZeroDCE, build_model, all pure losses, register_loss/build_losses |
| P1-B data and IO | data/*, inference/io.py, tests/test_data.py, tests/test_io.py | transforms, noise synthesis, synthetic temporal pairs with analytic flow, image and video datasets, image/video IO |
| P1-C flow, temporal, metrics | flow/*, losses/temporal.py, metrics/*, tests/test_flow_conventions.py, tests/test_metrics.py | DIS adapter, warp, reliability mask, scene cut, temporal loss, PSNR/SSIM/angular/noise/clipping/warping metrics |
| P1-D config and utils | training/config.py, utils/*, tests/test_flags_roundtrip.py, tests/test_config_schema.py, new scripts/_cli.py | validation, YAML load/save with CLI override precedence, seed/device/registry, shared CLI flag resolution used by all scripts |

## Phase 2 (parallel after phase 1 merges)

| Agent | Owns | Delivers |
| --- | --- | --- |
| P2-E training | training/{loop_step,trainer,checkpoint,logging}.py, scripts/{train,ablate,evaluate}.py, tests/test_training.py | gated train step, trainer with resume, checkpoint format v1, CSV logging, ablation and evaluation drivers |
| P2-F inference | inference/{enhancer,video_enhancer,webcam}.py, scripts/{enhance,make_synthetic_data}.py, tests/test_inference.py | curve-scale inference, stateful video enhancer with reset/cut/blend, webcam loop, enhance and synthetic-data CLIs |
| P2-G API server | new src/nta_zerodce/api/*, tests/test_api.py | FastAPI app: health, enhance image, options; checkpoint via env var; size and type limits |
| P2-H web demo | web/* | upload, options, before/after view, download, error states; vitest unit tests; typecheck, lint and build clean |

## HTTP contract (shared by P2-G and P2-H)

Base path /api/v1. Server default http://127.0.0.1:8000; the web app reaches it
through a same-origin proxy (Next.js rewrite from /api/v1/* to the server), so the
browser never needs CORS. Server address comes from LUMINA_API_URL (web) and
LUMINA_CHECKPOINT, LUMINA_CONFIG, LUMINA_DEVICE (server).

- GET /api/v1/health returns JSON {status: "ok", model_loaded: bool, device: str,
  version: str}. Always 200 when the process is up.
- GET /api/v1/options returns JSON {flags: {...resolved flags...}, curve_scale: float,
  max_upload_bytes: int, max_side: int, accepted_types: ["image/png","image/jpeg"]}.
- POST /api/v1/enhance multipart/form-data: file (PNG or JPEG), optional
  curve_scale (float in (0,1]). Returns the enhanced image as image/png with headers
  X-Lumina-Elapsed-Ms and X-Lumina-Size (WxH). Errors are JSON {error: {code, message}}:
  400 invalid_image, 413 too_large, 415 unsupported_type, 422 bad_option,
  503 model_not_loaded.

## Phase 3 (lead)

Integrate: train a tiny model on synthetic images, enhance through CLI and API,
run the web build against the API, add Dockerfile, CI updates, docs and status
table updates, an independent review agent, final full check run, push the branch.

## Acceptance

make lint typecheck test docs-check green; web typecheck, lint, test, build green;
end-to-end smoke (train, checkpoint, enhance CLI, API request) green; docs describe
only what exists, with no invented measurements.
