import { afterEach, describe, expect, it, vi } from "vitest";
import { enhance, getHealth, getOptions, parseElapsed, parseSize } from "@/lib/api";
import { ApiError, describeError } from "@/lib/errors";

function mockFetch(response: Response | Error | DOMException): ReturnType<typeof vi.fn> {
  const fn = vi.fn(() => (response instanceof Response ? Promise.resolve(response) : Promise.reject(response)));
  vi.stubGlobal("fetch", fn);
  return fn;
}

function errorResponse(status: number, code: string, message = "server says no"): Response {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const file = new File([new Uint8Array([1, 2, 3])], "dark.png", { type: "image/png" });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getHealth / getOptions", () => {
  it("requests the relative same-origin URLs", async () => {
    const fn = mockFetch(Response.json({ status: "ok", model_loaded: true, device: "cpu", version: "1.2.3" }));
    const health = await getHealth();
    expect(fn).toHaveBeenCalledWith("/api/v1/health", expect.objectContaining({ method: "GET" }));
    expect(health).toEqual({ status: "ok", model_loaded: true, device: "cpu", version: "1.2.3" });
  });

  it("rejects a health body without model_loaded as unexpected", async () => {
    mockFetch(Response.json({ status: "ok" }));
    await expect(getHealth()).rejects.toMatchObject({ kind: "unexpected" });
  });

  it("reads only well-typed fields from /options", async () => {
    const fn = mockFetch(
      Response.json({
        flags: { a: 1 },
        curve_scale: 0.5,
        max_upload_bytes: 1234,
        max_side: 2048,
        accepted_types: ["image/png", 7],
      }),
    );
    const options = await getOptions();
    expect(fn).toHaveBeenCalledWith("/api/v1/options", expect.anything());
    expect(options).toEqual({
      flags: { a: 1 },
      curve_scale: 0.5,
      max_upload_bytes: 1234,
      max_side: 2048,
      accepted_types: ["image/png"],
    });
  });

  it("maps a fetch rejection to a network error", async () => {
    mockFetch(new TypeError("Failed to fetch"));
    await expect(getHealth()).rejects.toMatchObject({ kind: "network", status: null });
  });
});

describe("enhance", () => {
  it("posts multipart with file and curve_scale and parses headers", async () => {
    const fn = mockFetch(
      new Response(new Uint8Array([9, 9, 9, 9]), {
        status: 200,
        headers: { "Content-Type": "image/png", "X-Lumina-Elapsed-Ms": "42.6", "X-Lumina-Size": "640x480" },
      }),
    );
    const result = await enhance({ file, curveScale: 0.5 });

    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/v1/enhance");
    expect(init.method).toBe("POST");
    const form = init.body as FormData;
    expect((form.get("file") as File).name).toBe("dark.png");
    expect(form.get("curve_scale")).toBe("0.5");

    expect(result.elapsedMs).toBe(42.6);
    expect(result.size).toEqual({ width: 640, height: 480 });
    expect(result.blob.size).toBe(4);
  });

  it("omits curve_scale when not set", async () => {
    const fn = mockFetch(new Response(new Uint8Array([1]), { headers: { "Content-Type": "image/png" } }));
    await enhance({ file, curveScale: null });
    const form = (fn.mock.calls[0] as unknown as [string, RequestInit])[1].body as FormData;
    expect(form.has("curve_scale")).toBe(false);
  });

  it("tolerates missing or malformed metadata headers", async () => {
    mockFetch(
      new Response(new Uint8Array([1]), {
        headers: { "Content-Type": "image/png", "X-Lumina-Elapsed-Ms": "soon", "X-Lumina-Size": "big" },
      }),
    );
    const result = await enhance({ file });
    expect(result.elapsedMs).toBeNull();
    expect(result.size).toBeNull();
  });

  it.each([
    [400, "invalid_image"],
    [413, "too_large"],
    [415, "unsupported_type"],
    [422, "bad_option"],
    [503, "model_not_loaded"],
  ])("maps HTTP %i %s to a typed ApiError", async (status, code) => {
    mockFetch(errorResponse(status, code));
    const error = await enhance({ file }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ kind: "api", status, code, serverMessage: "server says no" });
  });

  it("maps an unknown error code and a non-JSON error body", async () => {
    mockFetch(errorResponse(500, "kaboom"));
    await expect(enhance({ file })).rejects.toMatchObject({ kind: "api", status: 500, code: "unknown" });

    mockFetch(new Response("<html>Bad gateway</html>", { status: 502 }));
    await expect(enhance({ file })).rejects.toMatchObject({ kind: "network", status: 502 });
  });

  it("rejects a 200 response that is not a PNG", async () => {
    mockFetch(new Response("hello", { headers: { "Content-Type": "text/plain" } }));
    await expect(enhance({ file })).rejects.toMatchObject({ kind: "unexpected" });
  });

  it("maps a network failure", async () => {
    mockFetch(new TypeError("Failed to fetch"));
    await expect(enhance({ file })).rejects.toMatchObject({ kind: "network", code: null });
  });

  it("passes AbortError through untouched", async () => {
    mockFetch(new DOMException("aborted", "AbortError"));
    await expect(enhance({ file })).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("header parsers", () => {
  it("parses sizes and elapsed values", () => {
    expect(parseSize("10x20")).toEqual({ width: 10, height: 20 });
    expect(parseSize(null)).toBeNull();
    expect(parseElapsed("0")).toBe(0);
    expect(parseElapsed("-1")).toBeNull();
    expect(parseElapsed("")).toBeNull();
  });
});

describe("describeError", () => {
  const make = (init: ConstructorParameters<typeof ApiError>[0]) => describeError(new ApiError(init));

  it("gives each contract error its own message", () => {
    const messages = (["invalid_image", "too_large", "unsupported_type", "bad_option", "model_not_loaded"] as const).map(
      (code) => make({ kind: "api", status: 400, code }),
    );
    expect(new Set(messages.map((m) => m.title)).size).toBe(5);
    expect(messages[4].message).toContain("LUMINA_CHECKPOINT");
  });

  it("explains network failures and falls back for unknown errors", () => {
    expect(make({ kind: "network", status: null, code: null }).title).toMatch(/cannot reach/i);
    expect(make({ kind: "api", status: 500, code: "unknown", serverMessage: "boom" })).toEqual({
      title: "Server error (HTTP 500)",
      message: "boom",
    });
    expect(describeError(new Error("x")).title).toBe("Something went wrong");
  });
});
