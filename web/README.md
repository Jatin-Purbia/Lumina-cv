# Web demo shell

Empty Next.js App Router + TypeScript workspace for the future Lumina CV demo.
The home route renders no visible content. No screens, components, API routes or
backend integration have been added.

```sh
npm ci
npm run dev
```

Open http://localhost:3000. Start implementing the UI in [app/page.tsx](app/page.tsx).
[app/layout.tsx](app/layout.tsx) provides only the required HTML/body shell.

```sh
npm run typecheck
npm run build
npm start
```

Use Node.js 22 LTS. Dependency versions are pinned and package-lock.json is tracked.
The independent Python CV/backend package is in [core](../core/README.md). Future
integration will call its HTTP API; the serving contract is intentionally deferred.
