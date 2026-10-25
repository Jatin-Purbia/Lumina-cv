# HTTP API

The API server wraps the still-image enhancer for the Next.js demo. The contract (paths,
headers, status codes, error shape) is defined in [PLAN](PLAN.md) ("HTTP contract"); this
page documents what the code in `src/nta_zerodce/api/` does. The model and flags themselves
are described in [METHOD](METHOD.md) and [INTERFACES](INTERFACES.md).

**There is no authentication.** The server is meant to sit behind the demo's same-origin
proxy (a Next.js rewrite from `/api/v1/*`) on a trusted network. Do not expose it to the public
internet without a reverse proxy that provides authentication and rate limiting.

## Running

```bash
cd core
LUMINA_CHECKPOINT=path/to/checkpoint.pt python -m nta_zerodce.api
```

`python -m nta_zerodce.api` reads the `LUMINA_*` variables below, then serves with uvicorn on
`LUMINA_HOST:LUMINA_PORT` (default `127.0.0.1:8000`). Startup problems are reported and the
process exits with a non-zero code:

- an invalid environment variable: exit code 2 and a message naming the variable;
- a checkpoint or config that cannot be loaded (missing file, corrupt file, a config that does
  not match the checkpoint, unavailable CUDA device): the lifespan raises `StartupError`
  naming `LUMINA_CHECKPOINT` or `LUMINA_CONFIG`, and uvicorn exits with code 3.

The model is loaded once during startup, never per request. Without `LUMINA_CHECKPOINT` the
server still starts: `/health` reports `model_loaded: false` and `/enhance` answers 503.
Setting `LUMINA_CHECKPOINT` never falls back silently to "no model".

The `nta_zerodce.api.create_app(settings)` factory can also be used directly (for example
`uvicorn --factory nta_zerodce.api.app:create_app`, which reads `Settings.from_env()`). The
interactive `/docs` and `/openapi.json` pages are disabled.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `LUMINA_CHECKPOINT` | unset | Trusted local checkpoint (format v1). Unset or empty starts without a model. |
| `LUMINA_CONFIG` | unset | Optional experiment YAML. Its architecture flags and shapes must match the checkpoint; when unset the config stored in the checkpoint is used. Without a checkpoint it only supplies the flags and `curve_scale` shown by `/options`. |
| `LUMINA_DEVICE` | `auto` | `auto`, `cpu`, `cuda` or `cuda:N`. An unavailable CUDA device fails startup when a checkpoint is configured. |
| `LUMINA_MAX_UPLOAD_BYTES` | `20971520` | Largest accepted image file in bytes. |
| `LUMINA_MAX_SIDE` | `4096` | Largest accepted width or height in pixels (at least 16, and `MAX_SIDE`² must not exceed Pillow's `Image.MAX_IMAGE_PIXELS`). |
| `LUMINA_DEFAULT_CURVE_SCALE` | `model.curve_scale` of the config | Value in (0, 1] used when a request omits `curve_scale`. |
| `LUMINA_MAX_CONCURRENCY` | `1` | Number of requests that may decode, run the model and encode at the same time. Further requests wait. |
| `LUMINA_CORS_ORIGINS` | empty (CORS off) | Comma-separated browser origins such as `http://localhost:3000`, or `*`. The demo uses a same-origin proxy and needs none. |
| `LUMINA_HOST` | `127.0.0.1` | Bind address for `python -m nta_zerodce.api`. |
| `LUMINA_PORT` | `8000` | Bind port for `python -m nta_zerodce.api`. |

Empty or whitespace-only values count as unset. Integers must be plain digits. Any other value
fails at startup with a message naming the variable.

## Endpoints

All paths are under `/api/v1`.

### `GET /health`

Always 200 while the process is up.

```json
{"status": "ok", "model_loaded": true, "device": "cpu", "version": "0.0.1"}
```

`device` is the resolved torch device when a model is loaded, otherwise the requested
`LUMINA_DEVICE` string. `version` is the installed package version (`unknown` if not installed).

### `GET /options`

```json
{
  "flags": {"noise_aware": true, "denoise": true, "edge": true, "chroma": true,
            "adaptive_exp": true, "temporal": true, "light": false},
  "curve_scale": 1.0,
  "max_upload_bytes": 20971520,
  "max_side": 4096,
  "accepted_types": ["image/png", "image/jpeg"]
}
```

`flags` are the resolved feature flags of the loaded model's config (library defaults when no
checkpoint is configured). `curve_scale` is the default applied when a request omits the field.

### `POST /enhance`

`multipart/form-data` with:

| Field | Required | Meaning |
| --- | --- | --- |
| `file` | yes | Exactly one PNG or JPEG file part. Its part `Content-Type` must be `image/png` or `image/jpeg`. |
| `curve_scale` | no | Plain decimal number in (0, 1] (an exponent such as `5e-1` is allowed). Omitted means the server default. |

Other form fields are ignored (at most 16 fields and 4 file parts are accepted per request).

On success the response is `200` with an 8-bit RGB PNG body (no metadata) and:

| Header | Value |
| --- | --- |
| `Content-Type` | `image/png` |
| `Cache-Control` | `no-store` |
| `X-Lumina-Size` | `WxH` of the output, which equals the size of the decoded input after EXIF rotation |
| `X-Lumina-Elapsed-Ms` | Integer milliseconds of server-side work: transfer to the device, model inference and PNG encoding. It excludes reading the upload, decoding it and any wait for a free concurrency slot. |

Processing: the upload is held in memory only (never written to disk). The header is checked
against the size limits before pixels are decoded; EXIF orientation is applied; any mode is
converted to 8-bit RGB with transparency composited on black (16-bit and floating-point
grayscale are rejected); values are `byte / 255` as in `read_image`. The tensor is enhanced
with `enhance_image` under `torch.inference_mode` in a worker thread, quantized once with
`round(255 * x)` and PNG-encoded.

### Cross-cutting headers

Every response carries `X-Request-Id` (the request's `X-Request-Id` header when it matches
`[A-Za-z0-9._-]{1,128}`, otherwise a generated 32-character hex id), `X-Content-Type-Options:
nosniff`, and `Cache-Control: no-store` unless the endpoint set one.

## Errors

Every non-2xx response, including framework-generated ones, is JSON with exactly this shape:

```json
{"error": {"code": "invalid_image", "message": "The uploaded file is empty."}}
```

Messages are fixed strings: they never contain file names, server paths or exception text.

| Status | `code` | When |
| --- | --- | --- |
| 400 | `invalid_image` | Corrupt or truncated PNG/JPEG; bytes that are not an image (including a text file sent as `image/png`); declared type contradicts the content (for example JPEG bytes labelled `image/png`); empty file; width or height below 16; 16-bit or floating-point grayscale; missing or duplicated `file` field, a `file` field that is not a file part, a non-multipart body, a malformed multipart body or invalid `Content-Length`. |
| 413 | `too_large` | Request body larger than `LUMINA_MAX_UPLOAD_BYTES` plus 64 KiB of multipart allowance (rejected from `Content-Length` before reading, or as soon as the streamed body exceeds it); file larger than `LUMINA_MAX_UPLOAD_BYTES`; width or height above `LUMINA_MAX_SIDE`; a Pillow decompression-bomb error. |
| 415 | `unsupported_type` | Declared type is not `image/png` or `image/jpeg` (including a missing type); or the bytes are a real but unsupported image such as GIF, WebP, TIFF or BMP. |
| 422 | `bad_option` | `curve_scale` is not a plain decimal in (0, 1]: `0`, `1.5`, `abc`, `nan`, `inf`, `-1`, empty, repeated, etc.; also framework request-validation failures. |
| 503 | `model_not_loaded` | `/enhance` while no checkpoint is loaded. Checked before the body is read. |
| 404 | `not_found` | Unknown path. |
| 405 | `method_not_allowed` | Wrong method for a known path; the `Allow` header is kept. |
| 500 | `internal_error` | Unexpected failure (the details are only in the server log). |

Ambiguity decisions: a request with no usable `file` field answers 400 `invalid_image` (the
same as `web/scripts/mock-api.mjs`), not 422. A declared type that disagrees with the content
answers 400, except when the content is a recognised unsupported format, which answers 415.
When several problems apply the first check in this order wins: model loaded, body size,
multipart structure and file presence, file size, `curve_scale`, declared type and magic bytes,
then decoding (side limits, integrity).

## Examples

```bash
curl -s http://127.0.0.1:8000/api/v1/health
curl -s http://127.0.0.1:8000/api/v1/options

# Enhance a PNG; -D - prints the response headers, the PNG goes to out.png
curl -sS -D - -o out.png -F "file=@input.png" http://127.0.0.1:8000/api/v1/enhance

# JPEG with a reduced curve-estimation scale (curl derives the type from the extension;
# otherwise force it with ;type=image/jpeg)
curl -sS -o out.png -F "file=@photo.jpg;type=image/jpeg" -F curve_scale=0.5 \
  http://127.0.0.1:8000/api/v1/enhance

# Error example (415): the body is JSON
curl -s -F "file=@anim.gif" http://127.0.0.1:8000/api/v1/enhance
# {"error":{"code":"unsupported_type","message":"..."}}
```

## Logging

Each request writes one line to the `nta_zerodce.api.access` logger, for example:

```text
method=POST path=/api/v1/enhance status=200 elapsed_ms=30 request_id=c22dc2cfbe8d443586f8821ea7ff98e3
```

Only the method, path (no query string), status, elapsed milliseconds and request id are logged:
never payloads or uploaded file names. `python -m nta_zerodce.api` configures INFO-level
logging to stderr and disables uvicorn's own access log.

## Limits and resource use

- Upload bytes are counted while the body is streamed; at most `LUMINA_MAX_UPLOAD_BYTES` plus
  64 KiB are buffered (in memory) per in-flight request. Many simultaneous connections can
  still add up, so limit connections in the reverse proxy.
- Decoded pixels are bounded by `LUMINA_MAX_SIDE`² (and Pillow's decompression-bomb guard).
  Inference memory grows with the pixel count and the model; choose `LUMINA_MAX_SIDE` and
  `LUMINA_MAX_CONCURRENCY` for the machine. Requests beyond the concurrency limit wait; they do
  not fail.
- `LUMINA_MAX_UPLOAD_BYTES` bounds the compressed file only; the side limit bounds the pixels.

## Deployment notes

- One server process holds one model. `python -m nta_zerodce.api` runs a single uvicorn
  process. Starting several uvicorn workers (for example `uvicorn --factory
  nta_zerodce.api.app:create_app --workers N`) loads one model copy per worker, multiplying
  memory use, and `LUMINA_MAX_CONCURRENCY` then applies per worker. Prefer one process and
  raise `LUMINA_MAX_CONCURRENCY` only after checking memory.
- The checkpoint is a trusted local file; never point `LUMINA_CHECKPOINT` at untrusted input.
- Put the server behind the demo's same-origin proxy (set `LUMINA_API_URL` in the web app).
  For anything reachable by others add TLS, authentication and rate limiting in a reverse proxy.
- Graceful shutdown: on SIGINT/SIGTERM uvicorn finishes in-flight requests (up to 10 seconds),
  then the lifespan releases the model.
- Tests: `pytest tests/test_api.py` (CPU, random-initialised checkpoint, no downloads).
