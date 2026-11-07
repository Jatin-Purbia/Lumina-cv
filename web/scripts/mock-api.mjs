#!/usr/bin/env node
/**
 * Mock of the Lumina CV HTTP API (contract: core/docs/ARCHITECTURE.md, "HTTP contract").
 * Dependency-free; for developing the web UI without the Python server.
 *
 * LIMITATIONS (read before relying on this):
 *  - It does NOT enhance anything. Without an image library it cannot brighten
 *    pixels, so POST /api/v1/enhance echoes the uploaded bytes back unchanged as
 *    image/png, and only when the upload really is a PNG (checked by magic bytes).
 *    The before/after slider will therefore show two identical images.
 *  - JPEG uploads (and PNG-labelled files that are not PNG) are rejected with a
 *    415 unsupported_type JSON error, because the mock cannot convert them.
 *    Bytes that are neither PNG nor JPEG get 400 invalid_image.
 *  - X-Lumina-Elapsed-Ms is the mock's own handling time, not model time.
 *  - /options values (limits, curve_scale) are fixed placeholders.
 *
 * Environment:
 *  MOCK_API_PORT      listen port (default 8000)
 *  MOCK_API_HOST      listen host (default 127.0.0.1)
 *  MOCK_MODEL_LOADED  set to "0" to report model_loaded=false and answer 503
 *                     model_not_loaded on /enhance
 *
 * Run: npm run mock-api   (then LUMINA_API_URL defaults to http://127.0.0.1:8000)
 */
import http from "node:http";

const PORT = Number(process.env.MOCK_API_PORT ?? 8000);
const HOST = process.env.MOCK_API_HOST ?? "127.0.0.1";
const MODEL_LOADED = process.env.MOCK_MODEL_LOADED !== "0";
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const JPEG_MAGIC = Buffer.from([0xff, 0xd8, 0xff]);

function sendJson(res, status, body, headers = {}) {
  const payload = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": payload.length,
    "Cache-Control": "no-store",
    ...headers,
  });
  res.end(payload);
}

function sendError(res, status, code, message) {
  sendJson(res, status, { error: { code, message } });
}

/** Reads the request body, rejecting with {tooLarge:true} past MAX_UPLOAD_BYTES (+1 MiB framing). */
function readBody(req) {
  const limit = MAX_UPLOAD_BYTES + 1024 * 1024;
  return new Promise((resolve, reject) => {
    const declared = Number(req.headers["content-length"] ?? 0);
    if (declared > limit) {
      req.resume();
      reject({ tooLarge: true });
      return;
    }
    const chunks = [];
    let total = 0;
    req.on("data", (chunk) => {
      total += chunk.length;
      if (total > limit) {
        req.destroy();
        reject({ tooLarge: true });
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

/** Minimal multipart/form-data parser: returns [{name, filename, data}]. */
function parseMultipart(body, contentType) {
  const match = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType ?? "");
  if (!match) return null;
  const delimiter = Buffer.from(`--${match[1] ?? match[2]}`);
  const parts = [];
  let start = body.indexOf(delimiter);
  while (start !== -1) {
    const next = body.indexOf(delimiter, start + delimiter.length);
    if (next === -1) break;
    // Part = bytes between the delimiter's CRLF and the CRLF preceding the next delimiter.
    const part = body.subarray(start + delimiter.length + 2, next - 2);
    const headerEnd = part.indexOf("\r\n\r\n");
    if (headerEnd !== -1) {
      const headers = part.subarray(0, headerEnd).toString("utf8");
      const disposition = /content-disposition:[^\r\n]*/i.exec(headers)?.[0] ?? "";
      const name = /\bname="([^"]*)"/i.exec(disposition)?.[1];
      const filename = /\bfilename="([^"]*)"/i.exec(disposition)?.[1];
      if (name !== undefined) parts.push({ name, filename, data: part.subarray(headerEnd + 4) });
    }
    start = next;
  }
  return parts;
}

async function handleEnhance(req, res) {
  const started = performance.now();
  if (!MODEL_LOADED) {
    req.resume();
    return sendError(res, 503, "model_not_loaded", "No model checkpoint is loaded (mock started with MOCK_MODEL_LOADED=0).");
  }
  let body;
  try {
    body = await readBody(req);
  } catch (error) {
    if (error && error.tooLarge) {
      return sendError(res, 413, "too_large", `Upload exceeds ${MAX_UPLOAD_BYTES} bytes.`);
    }
    return sendError(res, 400, "invalid_image", "The request body could not be read.");
  }

  const parts = parseMultipart(body, req.headers["content-type"]);
  const file = parts?.find((p) => p.name === "file" && p.filename !== undefined);
  if (!file || file.data.length === 0) {
    return sendError(res, 400, "invalid_image", "Expected a multipart field named 'file' containing an image.");
  }
  if (file.data.length > MAX_UPLOAD_BYTES) {
    return sendError(res, 413, "too_large", `Upload exceeds ${MAX_UPLOAD_BYTES} bytes.`);
  }

  const scaleField = parts.find((p) => p.name === "curve_scale");
  if (scaleField) {
    const text = scaleField.data.toString("utf8").trim();
    const value = Number(text);
    if (text === "" || !Number.isFinite(value) || value <= 0 || value > 1) {
      return sendError(res, 422, "bad_option", "curve_scale must be a number in (0, 1].");
    }
  }

  const isPng = file.data.subarray(0, PNG_MAGIC.length).equals(PNG_MAGIC);
  const isJpeg = file.data.subarray(0, JPEG_MAGIC.length).equals(JPEG_MAGIC);
  if (isJpeg) {
    return sendError(res, 415, "unsupported_type", "The mock API only echoes PNG uploads; it cannot convert JPEG.");
  }
  if (!isPng) {
    return sendError(res, 400, "invalid_image", "The upload is not a PNG or JPEG image.");
  }
  // PNG IHDR: width and height are big-endian uint32 at byte offsets 16 and 20.
  if (file.data.length < 24) {
    return sendError(res, 400, "invalid_image", "The PNG is truncated.");
  }
  const width = file.data.readUInt32BE(16);
  const height = file.data.readUInt32BE(20);

  res.writeHead(200, {
    "Content-Type": "image/png",
    "Content-Length": file.data.length,
    "Cache-Control": "no-store",
    "X-Lumina-Elapsed-Ms": String(Math.round(performance.now() - started)),
    "X-Lumina-Size": `${width}x${height}`,
  });
  res.end(file.data);
}

const server = http.createServer((req, res) => {
  const path = new URL(req.url ?? "/", "http://localhost").pathname;

  if (path === "/api/v1/health") {
    if (req.method !== "GET") return sendError(res, 405, "method_not_allowed", "Use GET.");
    return sendJson(res, 200, { status: "ok", model_loaded: MODEL_LOADED, device: "mock", version: "mock" });
  }
  if (path === "/api/v1/options") {
    if (req.method !== "GET") return sendError(res, 405, "method_not_allowed", "Use GET.");
    return sendJson(res, 200, {
      flags: {},
      curve_scale: 1.0,
      max_upload_bytes: MAX_UPLOAD_BYTES,
      max_side: 4096,
      accepted_types: ["image/png", "image/jpeg"],
    });
  }
  if (path === "/api/v1/enhance") {
    if (req.method !== "POST") return sendError(res, 405, "method_not_allowed", "Use POST.");
    handleEnhance(req, res).catch(() => sendError(res, 400, "invalid_image", "The request could not be processed."));
    return undefined;
  }
  return sendError(res, 404, "not_found", `No route for ${path}.`);
});

server.listen(PORT, HOST, () => {
  console.log(`Lumina mock API listening on http://${HOST}:${PORT}/api/v1 (model_loaded=${MODEL_LOADED})`);
  console.log("Echoes PNG uploads unchanged; does not enhance. See the file header for limits.");
});
