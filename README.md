# Lumina CV

Separate workspaces for the future web demo and the Noise- and Temporal-Aware
Zero-DCE core.

| Workspace | Responsibility | Current state |
| --- | --- | --- |
| [web](web/README.md) | Next.js demo UI | Empty App Router shell; home page renders nothing |
| [core](core/README.md) | Python CV/backend logic, training and inference | Existing documented, typed research stubs |

```text
Lumina-CV/
  web/                     Next.js + React + TypeScript
    app/layout.tsx         HTML shell
    app/page.tsx           Empty page; UI will be built later
  core/                    Independent Python package
    src/nta_zerodce/        Models, losses, data, flow, inference and training
    configs/ docs/ tests/ scripts/ notebooks/ report/
  .github/workflows/ci.yml  Separate core and web checks
```

## Web development

Use Node.js 22 LTS and npm. The minimal shell follows the
[Next.js App Router installation guide](https://nextjs.org/docs/app/getting-started/installation).

```sh
cd web
npm ci
npm run dev
```

The home page at http://localhost:3000 is intentionally blank.

```sh
npm run typecheck
npm run build
npm start
```

## Core development

The Python package remains named nta_zerodce; only its repository folder changed
from nta-zerodce/ to core/. All source functions remain TODO stubs.

```sh
cd core
python -m pip install -e ".[dev]"
make lint typecheck test docs-check
```

See [core documentation](core/docs/ARCHITECTURE.md) for the ML architecture and
[interfaces](core/docs/INTERFACES.md) for tensor/config contracts. Core commands
resolve configs and local data paths relative to core/.

## Separation

The web workspace owns the future demo screens and browser interaction. The core
owns CV processing and backend behavior. Develop and install each independently;
web does not import Python source or run enhancement logic. A future HTTP API will
connect the two once its endpoints and serving framework are chosen. No API server,
upload UI or enhancement integration is implemented in this scaffold.
