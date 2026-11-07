import { describe, expect, it } from "vitest";
import { fitDimensions } from "../lib/resize";

describe("fitDimensions", () => {
  it("returns null when the image already fits", () => {
    expect(fitDimensions(1920, 1080, 2 * 1024 * 1024)).toBeNull();
  });

  it("shrinks a 3000x2002 photo under the pixel limit and keeps the aspect ratio", () => {
    const out = fitDimensions(3000, 2002, 2 * 1024 * 1024, 4096);
    expect(out).not.toBeNull();
    const { width, height } = out as { width: number; height: number };
    expect(width * height).toBeLessThanOrEqual(2 * 1024 * 1024);
    expect(width * height).toBeGreaterThan(2 * 1024 * 1024 * 0.98);
    expect(Math.abs(width / height - 3000 / 2002)).toBeLessThan(0.01);
  });

  it("honours the longest-side limit", () => {
    const out = fitDimensions(5000, 100, 10_000_000, 4096);
    expect(out).toEqual({ width: 4096, height: 81 });
  });
});
