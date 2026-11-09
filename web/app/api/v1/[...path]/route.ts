/**
 * Runtime proxy from same-origin /api/v1/* to the Lumina API server.
 *
 * A route handler is used instead of a next.config rewrite because rewrites are resolved at
 * BUILD time, which would bake LUMINA_API_URL into the build. This reads it per request,
 * streams request and response bodies (no buffering or truncation), forwards only an
 * allow-list of endpoints and headers, and turns an unreachable upstream into a JSON 502.
 */

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DEFAULT_API_URL = "http://127.0.0.1:8000";
const UPSTREAM_TIMEOUT_MS = 120_000;
/** Upload limit (20 MiB) plus multipart framing; the API enforces the exact limit itself. */
const MAX_REQUEST_BYTES = 21 * 1024 * 1024;

/** Allowed endpoint -> allowed method. Anything else is answered here and never forwarded. */
const ROUTES: Readonly<Record<string, "GET" | "POST">> = {
  health: "GET",
  options: "GET",
  enhance: "POST",
};
const REQUEST_HEADERS = ["content-type", "content-length", "accept", "x-request-id"] as const;
const RESPONSE_HEADERS = [
  "retry-after",
  "content-type",
  "content-length",
  "cache-control",
  "x-lumina-elapsed-ms",
  "x-lumina-size",
  "x-request-id",
] as const;

type Method = "GET" | "POST";

interface RouteContext {
  params: Promise<{ path: string[] }>;
}

/** Upstream base URL, read at request time so one build can run against any server. */
export function upstreamBase(): string {
  const value = process.env.LUMINA_API_URL?.trim();
  return (value ? value : DEFAULT_API_URL).replace(/\/+$/, "");
}

function errorResponse(status: number, code: string, message: string): Response {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

async function proxy(request: Request, context: RouteContext, method: Method): Promise<Response> {
  const { path } = await context.params;
  const endpoint = path.length === 1 ? path[0] : undefined;
  const allowedMethod = endpoint === undefined ? undefined : ROUTES[endpoint];
  if (endpoint === undefined || allowedMethod === undefined) {
    return errorResponse(404, "not_found", "No such endpoint.");
  }
  if (allowedMethod !== method) {
    return errorResponse(405, "method_not_allowed", `Use ${allowedMethod} for this endpoint.`);
  }

  const declaredLength = Number(request.headers.get("content-length") ?? "0");
  if (Number.isFinite(declaredLength) && declaredLength > MAX_REQUEST_BYTES) {
    return errorResponse(413, "too_large", "The upload is larger than the server accepts.");
  }

  const headers = new Headers();
  for (const name of REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers,
    redirect: "manual",
    cache: "no-store",
    signal: AbortSignal.any([request.signal, AbortSignal.timeout(UPSTREAM_TIMEOUT_MS)]),
  };
  if (method === "POST") {
    init.body = request.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${upstreamBase()}/api/v1/${endpoint}`, init);
  } catch (cause) {
    const timedOut = (cause as { name?: unknown } | null)?.name === "TimeoutError";
    return timedOut
      ? errorResponse(504, "upstream_timeout", "The enhancement server took too long to answer.")
      : errorResponse(502, "upstream_unavailable", "The enhancement server could not be reached.");
  }

  const responseHeaders = new Headers();
  for (const name of RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value !== null) responseHeaders.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export async function GET(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context, "GET");
}

export async function POST(request: Request, context: RouteContext): Promise<Response> {
  return proxy(request, context, "POST");
}
