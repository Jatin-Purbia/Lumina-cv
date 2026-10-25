import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_UPLOAD_BYTES,
  enhancedFileName,
  formatBytes,
  validateFile,
} from "@/lib/validate";

const png = (size: number, name = "a.png") => ({ name, size, type: "image/png" });

describe("validateFile", () => {
  it("accepts PNG and JPEG within the default limit", () => {
    expect(validateFile(png(1024))).toBeNull();
    expect(validateFile({ name: "a.jpg", size: 10, type: "image/jpeg" })).toBeNull();
  });

  it("accepts a file exactly at the limit and rejects one byte over", () => {
    expect(validateFile(png(DEFAULT_MAX_UPLOAD_BYTES))).toBeNull();
    const failure = validateFile(png(DEFAULT_MAX_UPLOAD_BYTES + 1));
    expect(failure?.code).toBe("too_large");
    expect(failure?.message).toContain("20.0 MB");
  });

  it("uses the limit and types supplied by /options", () => {
    const rules = { maxBytes: 1000, acceptedTypes: ["image/png"] };
    expect(validateFile(png(1001), rules)?.code).toBe("too_large");
    expect(validateFile({ name: "a.jpg", size: 10, type: "image/jpeg" }, rules)?.code).toBe("unsupported_type");
  });

  it("rejects unsupported and unknown types and names the accepted ones", () => {
    const failure = validateFile({ name: "a.gif", size: 10, type: "image/gif" });
    expect(failure?.code).toBe("unsupported_type");
    expect(failure?.message).toContain("PNG or JPEG");
    expect(validateFile({ name: "noext", size: 10, type: "" })?.code).toBe("unsupported_type");
  });

  it("rejects empty files", () => {
    expect(validateFile(png(0))?.code).toBe("empty");
  });
});

describe("helpers", () => {
  it("formats byte sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });

  it("derives the download name", () => {
    expect(enhancedFileName("street.jpg")).toBe("street-enhanced.png");
    expect(enhancedFileName("a.b.png")).toBe("a.b-enhanced.png");
    expect(enhancedFileName("noext")).toBe("noext-enhanced.png");
    expect(enhancedFileName(".png")).toBe(".png-enhanced.png");
  });
});
