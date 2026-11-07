# Lumina CV web demo

Single-page Next.js 16 (App Router, React 19, strict TypeScript) UI for the Lumina CV
low-light enhancement API. Plain CSS, system fonts, no analytics and no external
network requests; the browser only calls same-origin `/api/v1/...` URLs.

## What it does

- Choose or drag in a PNG or JPEG. Type and size are checked in the browser before
  upload, using `accepted_types` and `max_upload_bytes` from `GET /api/v1/options`
  (fallback: PNG/JPEG and 20 MB).
- An estimation-scale control (default, 0.25, 0.5, 1) appears only when `/options`
  reports `curve_scale`. Choosing a preset sends it as the `curve_scale` form field;
  "Default" sends nothing.
- `POST /api/v1/enhance` runs with a loading state and a Cancel button. The result is
  shown in a before/after slider (drag, or focus it and use Left/Right/Up/Down,
  Shift+arrows, PageUp/PageDown, Home, End), with the server-reported elapsed time
  (`X-Lumina-Elapsed-Ms`), output size (`X-Lumina-Size`) and a PNG download.
- API error codes (`invalid_image`, `too_large`, `unsupported_type`, `bad_option`,
  `model_not_loaded`) and network failures each get their own message.
- A status badge in the header reads `GET /api/v1/health` every 15 s and when the
  window regains focus: model loaded, server up but no model, or unreachable.
- Light and dark themes follow `prefers-color-scheme`.

## Run with the real API

Start the Python server from `core` (see the core docs), then:

```sh
cd web
npm ci
LUMINA_API_URL=http://127.0.0.1:8000 npm run dev   # the URL shown is the default
```

Open http://localhost:3000. The route handler `app/api/v1/[...path]/route.ts` proxies
`/api/v1/health`, `/options` and `/enhance` to `${LUMINA_API_URL}/api/v1/...`.
`LUMINA_API_URL` is read **per request** (default `http://127.0.0.1:8000`), so one
production build runs against any server: set it in the environment of `next start`
(or the container), not at build time.

Request and response bodies are streamed, so uploads up to the API's own limit (20 MiB)
are not truncated. The proxy rejects bodies over 21 MiB, times out after 120 s, forwards
only an allow-list of headers, and answers 502 (`upstream_unavailable`) or 504
(`upstream_timeout`) as JSON when the API cannot be reached. The page shows both as
"Cannot reach the enhancement server".

## Run with the mock API (no Python)

```sh
npm run mock-api   # terminal 1: http://127.0.0.1:8000
npm run dev        # terminal 2
```

`scripts/mock-api.mjs` implements the contract but does not enhance. It echoes an
uploaded PNG back unchanged (so the slider shows identical images), answers 415 for
JPEG, and answers 400 for other bytes. Set `MOCK_MODEL_LOADED=0` to simulate a
server without a model; `MOCK_API_PORT` / `MOCK_API_HOST` change the listen address.
Details are in the file header.

## Scripts

| Script | Purpose |
| --- | --- |
| `npm run dev` | Next dev server |
| `npm run build` / `npm start` | Production build and server |
| `npm run typecheck` | `next typegen` then `tsc --noEmit` |
| `npm run lint` | ESLint (`eslint-config-next`, core-web-vitals + TypeScript), zero warnings allowed |
| `npm test` | Vitest + Testing Library on jsdom |
| `npm run mock-api` | Dependency-free mock of the HTTP API |

## Layout

- `app/` layout, thin `page.tsx`, `globals.css`
- `components/` `EnhanceApp` (page flow), `CompareSlider`, `FilePicker`, `ScalePicker`,
  `ResultSummary`, `StatusIndicator`
- `hooks/` `useServerInfo` (health polling and options), `useObjectUrl` (revokes blob URLs)
- `lib/` `api.ts` (typed client), `errors.ts` (`ApiError` and user messages), `validate.ts`, `types.ts`
- `tests/` Vitest specs; `scripts/mock-api.mjs`

Use Node.js 22 LTS. Dependency versions are pinned and `package-lock.json` is tracked.
The Python package lives in [core](../core/README.md).

## Known limitations

- Only one image at a time; no batch mode, history or persistence.
- Enhancement runs on the server; the page makes no claim about quality or speed.
- `next.config.ts` uses Next 16 experimental options (`proxyClientMaxBodySize`,
  `proxyTimeout`) that may be renamed in later Next releases.
