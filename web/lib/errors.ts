import { KNOWN_ERROR_CODES, type KnownErrorCode } from "./types";

export type ApiErrorKind =
  /** The server answered with a JSON error body from the contract. */
  | "api"
  /** The server answered, but not with a usable contract response. */
  | "unexpected"
  /** No answer at all (server down, proxy failure, offline). */
  | "network";

export interface ApiErrorInit {
  kind: ApiErrorKind;
  status: number | null;
  code: KnownErrorCode | "unknown" | null;
  serverMessage?: string;
  cause?: unknown;
}

/** Typed error thrown by the API client. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly code: KnownErrorCode | "unknown" | null;
  readonly serverMessage: string | undefined;

  constructor(init: ApiErrorInit) {
    super(init.serverMessage ?? `${init.kind} error`, { cause: init.cause });
    this.name = "ApiError";
    this.kind = init.kind;
    this.status = init.status;
    this.code = init.code;
    this.serverMessage = init.serverMessage;
  }
}

export function isKnownCode(value: unknown): value is KnownErrorCode {
  return typeof value === "string" && (KNOWN_ERROR_CODES as readonly string[]).includes(value);
}

export interface UserFacingError {
  title: string;
  message: string;
}

/** Maps any thrown value to a specific, actionable message for the person using the page. */
export function describeError(error: unknown): UserFacingError {
  if (!(error instanceof ApiError)) {
    return {
      title: "Something went wrong",
      message: "The image could not be enhanced. Try again, or pick a different image.",
    };
  }
  if (error.kind === "network") {
    return {
      title: "Cannot reach the enhancement server",
      message:
        "The request did not get an answer. Check that the Lumina API is running and that LUMINA_API_URL points to it, then try again.",
    };
  }
  switch (error.code) {
    case "server_busy":
      return {
        title: "The enhancement server is busy",
        message: "Another image is being processed. Wait a moment, then try again.",
      };
    case "model_not_loaded":
      return {
        title: "The model is not loaded yet",
        message:
          "The server is running but has no model checkpoint loaded. Start it with a checkpoint (LUMINA_CHECKPOINT) and try again.",
      };
    case "invalid_image":
      return {
        title: "The file is not a readable image",
        message: "The server could not decode this file. Try re-exporting it as a PNG or JPEG.",
      };
    case "too_large":
      return {
        title: "The image is too large",
        message: "The server rejected this upload for its size. Use a smaller or more compressed image.",
      };
    case "unsupported_type":
      return {
        title: "This file type is not supported",
        message: "Only PNG and JPEG images can be enhanced.",
      };
    case "bad_option":
      return {
        title: "The enhancement setting was rejected",
        message: "The server did not accept the selected estimation scale. Choose another value or use the default.",
      };
    default:
      return {
        title: error.status === null ? "Unexpected response" : `Server error (HTTP ${error.status})`,
        message:
          error.serverMessage ??
          "The server returned a response this page does not understand. Try again in a moment.",
      };
  }
}

/** True for the AbortError raised when a fetch is cancelled via AbortSignal. */
export function isAbortError(value: unknown): boolean {
  return typeof value === "object" && value !== null && "name" in value && value.name === "AbortError";
}
