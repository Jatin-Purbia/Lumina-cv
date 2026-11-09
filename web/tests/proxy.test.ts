import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GET, POST, upstreamBase } from "../app/api/v1/[...path]/route";

const ctx = (...path: string[]) => ({ params: Promise.resolve({ path }) });

describe("api proxy route", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("LUMINA_API_URL", "http://api.test:9000/");
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("reads LUMINA_API_URL at request time and trims trailing slashes", () => {
    expect(upstreamBase()).toBe("http://api.test:9000");
    vi.stubEnv("LUMINA_API_URL", "http://other:1");
    expect(upstreamBase()).toBe("http://other:1");
    vi.stubEnv("LUMINA_API_URL", "  ");
    expect(upstreamBase()).toBe("http://127.0.0.1:8000");
  });

  it("forwards GET and passes through allow-listed response headers only", async () => {
    fetchMock.mockResolvedValue(
      new Response('{"status":"ok"}', {
        status: 200,
        headers: { "content-type": "application/json", "set-cookie": "a=b", "x-lumina-size": "3x4" },
      }),
    );
    const res = await GET(new Request("http://web/api/v1/health"), ctx("health"));
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://api.test:9000/api/v1/health");
    expect(res.status).toBe(200);
    expect(res.headers.get("x-lumina-size")).toBe("3x4");
    expect(res.headers.get("set-cookie")).toBeNull();
  });

  it("streams POST bodies with duplex half and forwards the content type", async () => {
    fetchMock.mockResolvedValue(new Response("png", { status: 200 }));
    const req = new Request("http://web/api/v1/enhance", {
      method: "POST",
      body: "payload",
      headers: { "content-type": "multipart/form-data; boundary=x", cookie: "secret=1" },
    });
    await POST(req, ctx("enhance"));
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit & { duplex?: string };
    expect(init.method).toBe("POST");
    expect(init.duplex).toBe("half");
    expect(new Headers(init.headers).get("content-type")).toBe("multipart/form-data; boundary=x");
    expect(new Headers(init.headers).get("cookie")).toBeNull();
  });

  it("preserves the overload response and retry hint", async () => {
    fetchMock.mockResolvedValue(Response.json({ error: { code: "server_busy", message: "Busy" } }, {
      status: 503, headers: { "retry-after": "1" },
    }));
    const response = await POST(new Request("http://web/api/v1/enhance", { method: "POST", body: "upload" }), ctx("enhance"));
    expect(response.status).toBe(503);
    expect(response.headers.get("retry-after")).toBe("1");
    expect((await response.json()).error.code).toBe("server_busy");
  });

  it("answers 502 JSON when the upstream is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    const res = await GET(new Request("http://web/api/v1/health"), ctx("health"));
    expect(res.status).toBe(502);
    expect((await res.json()).error.code).toBe("upstream_unavailable");
  });

  it("answers 504 JSON on upstream timeout", async () => {
    fetchMock.mockRejectedValue(new DOMException("timed out", "TimeoutError"));
    const res = await GET(new Request("http://web/api/v1/health"), ctx("health"));
    expect(res.status).toBe(504);
    expect((await res.json()).error.code).toBe("upstream_timeout");
  });

  it("rejects unknown paths, wrong methods and oversized uploads without forwarding", async () => {
    expect((await GET(new Request("http://web/x"), ctx("admin"))).status).toBe(404);
    expect((await GET(new Request("http://web/x"), ctx("health", "x"))).status).toBe(404);
    expect((await GET(new Request("http://web/x"), ctx("enhance"))).status).toBe(405);
    const big = new Request("http://web/api/v1/enhance", {
      method: "POST",
      body: "x",
      headers: { "content-length": String(30 * 1024 * 1024) },
    });
    expect((await POST(big, ctx("enhance"))).status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
