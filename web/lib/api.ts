import { ApiError, isAbortError, isKnownCode } from "./errors";
import type { EnhanceResult, HealthResponse, OptionsResponse } from "./types";

/** All calls are same-origin; Next.js rewrites /api/v1/* to the Lumina API server. */
export const API_BASE = "/api/v1";

async function request(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (cause) {
    if (isAbortError(cause)) throw cause;
    throw new ApiError({ kind: "network", status: null, code: null, cause });
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Turns a non-2xx response into a typed ApiError using the {error:{code,message}} body. */
async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  const detail = isRecord(body) && isRecord(body.error) ? body.error : null;
  if (detail === null) {
    return new ApiError({ kind: "unexpected", status: response.status, code: "unknown" });
  }
  const code = isKnownCode(detail.code) ? detail.code : "unknown";
  const serverMessage = typeof detail.message === "string" ? detail.message : undefined;
  return new ApiError({ kind: "api", status: response.status, code, serverMessage });
}

async function getJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await request(`${API_BASE}${path}`, { method: "GET", cache: "no-store", signal });
  if (!response.ok) throw await toApiError(response);
  try {
    return await response.json();
  } catch (cause) {
    throw new ApiError({ kind: "unexpected", status: response.status, code: "unknown", cause });
  }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const data = await getJson("/health", signal);
  if (!isRecord(data) || typeof data.model_loaded !== "boolean") {
    throw new ApiError({ kind: "unexpected", status: 200, code: "unknown" });
  }
  return {
    status: typeof data.status === "string" ? data.status : "ok",
    model_loaded: data.model_loaded,
    device: typeof data.device === "string" ? data.device : "unknown",
    version: typeof data.version === "string" ? data.version : "unknown",
  };
}

export async function getOptions(signal?: AbortSignal): Promise<OptionsResponse> {
  const data = await getJson("/options", signal);
  if (!isRecord(data)) throw new ApiError({ kind: "unexpected", status: 200, code: "unknown" });
  const options: OptionsResponse = {};
  if (isRecord(data.flags)) options.flags = data.flags;
  if (typeof data.curve_scale === "number" && Number.isFinite(data.curve_scale)) {
    options.curve_scale = data.curve_scale;
  }
  if (typeof data.max_upload_bytes === "number" && data.max_upload_bytes > 0) {
    options.max_upload_bytes = data.max_upload_bytes;
  }
  if (typeof data.max_side === "number") options.max_side = data.max_side;
  if (Array.isArray(data.accepted_types)) {
    options.accepted_types = data.accepted_types.filter((t): t is string => typeof t === "string");
  }
  return options;
}

/** Parses "WxH" from X-Lumina-Size. */
export function parseSize(value: string | null): { width: number; height: number } | null {
  if (value === null) return null;
  const match = /^(\d+)x(\d+)$/i.exec(value.trim());
  if (match === null) return null;
  return { width: Number(match[1]), height: Number(match[2]) };
}

export function parseElapsed(value: string | null): number | null {
  if (value === null || value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

export interface EnhanceParams {
  file: File;
  /** Omit to let the server use its default. */
  curveScale?: number | null;
  signal?: AbortSignal;
}

export async function enhance({ file, curveScale, signal }: EnhanceParams): Promise<EnhanceResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  if (curveScale !== undefined && curveScale !== null) form.append("curve_scale", String(curveScale));

  const response = await request(`${API_BASE}/enhance`, { method: "POST", body: form, signal });
  if (!response.ok) throw await toApiError(response);

  const type = response.headers.get("Content-Type") ?? "";
  if (!type.toLowerCase().startsWith("image/png")) {
    throw new ApiError({ kind: "unexpected", status: response.status, code: "unknown" });
  }
  let blob: Blob;
  try {
    blob = await response.blob();
  } catch (cause) {
    if (isAbortError(cause)) throw cause;
    throw new ApiError({ kind: "network", status: response.status, code: null, cause });
  }
  return {
    blob,
    elapsedMs: parseElapsed(response.headers.get("X-Lumina-Elapsed-Ms")),
    size: parseSize(response.headers.get("X-Lumina-Size")),
  };
}

