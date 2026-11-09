/** Response of GET /api/v1/health. */
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
}

/** Response of GET /api/v1/options. Only the fields the UI reads are required. */
export interface OptionsResponse {
  flags?: Record<string, unknown>;
  curve_scale?: number;
  max_upload_bytes?: number;
  max_side?: number;
  max_pixels?: number;
  accepted_types?: string[];
}

/** Successful result of POST /api/v1/enhance. */
export interface EnhanceResult {
  blob: Blob;
  /** Server-side elapsed time from X-Lumina-Elapsed-Ms, if present and numeric. */
  elapsedMs: number | null;
  /** Output size from X-Lumina-Size ("WxH"), if present and well formed. */
  size: { width: number; height: number } | null;
}

/** Error codes defined by the HTTP contract. */
export const KNOWN_ERROR_CODES = [
  "invalid_image",
  "too_large",
  "unsupported_type",
  "bad_option",
  "model_not_loaded",
  "server_busy",
] as const;

export type KnownErrorCode = (typeof KNOWN_ERROR_CODES)[number];
