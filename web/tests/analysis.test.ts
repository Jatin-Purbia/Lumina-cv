import { describe, expect, it } from "vitest";
import {
  analyze,
  analyzePair,
  darkMaskOf,
  describe as describeChange,
  gammaPixels,
  matchGamma,
  type Pixels,
} from "../lib/analysis";

function solid(width: number, height: number, rgb: [number, number, number]): Pixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = rgb[0];
    data[i + 1] = rgb[1];
    data[i + 2] = rgb[2];
    data[i + 3] = 255;
  }
  return { data, width, height };
}

function noisy(width: number, height: number, level: number, amplitude: number): Pixels {
  const p = solid(width, height, [level, level, level]);
  let seed = 12345;
  for (let i = 0; i < p.data.length; i += 4) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const n = ((seed % 2001) / 1000 - 1) * amplitude;
    p.data[i] = p.data[i + 1] = p.data[i + 2] = Math.round(level + n);
  }
  return p;
}

describe("analyze", () => {
  it("measures a flat gray image", () => {
    const p = solid(16, 16, [100, 100, 100]);
    const m = analyze(p, darkMaskOf(p));
    expect(m.mean).toBeCloseTo(100, 5);
    expect(m.contrast).toBeCloseTo(0, 5);
    expect(m.noise).toBeCloseTo(0, 5);
    expect(m.saturation).toBeCloseTo(0, 5);
    expect(m.entropy).toBeCloseTo(0, 5);
    expect(m.histogram.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });

  it("reports saturation for a colored image and clipping for white", () => {
    const red = solid(8, 8, [200, 0, 0]);
    expect(analyze(red, darkMaskOf(red)).saturation).toBeCloseTo(1, 5);
    const white = solid(8, 8, [255, 255, 255]);
    expect(analyze(white, darkMaskOf(white)).clipped).toBeCloseTo(100, 5);
  });

  it("reads more noise from a noisy image than from a flat one", () => {
    const flat = solid(32, 32, [90, 90, 90]);
    const grain = noisy(32, 32, 90, 12);
    expect(analyze(grain, darkMaskOf(grain)).noise).toBeGreaterThan(analyze(flat, darkMaskOf(flat)).noise + 3);
  });

  it("averages the originally dark pixels only", () => {
    const original = solid(4, 4, [10, 10, 10]);
    const enhanced = solid(4, 4, [80, 80, 80]);
    expect(analyze(enhanced, darkMaskOf(original)).darkMean).toBeCloseTo(80, 5);
  });
});

describe("gamma", () => {
  it("brightens with gamma below one and keeps 0 and 255", () => {
    const p = solid(2, 2, [64, 64, 64]);
    expect(gammaPixels(p, 0.5).data[0]).toBeGreaterThan(64);
    expect(gammaPixels(solid(2, 2, [0, 255, 0]), 0.4).data.slice(0, 3)).toEqual(new Uint8ClampedArray([0, 255, 0]));
  });

  it("finds the gamma that reaches a target mean", () => {
    const p = noisy(32, 32, 60, 20);
    const g = matchGamma(p, 120);
    const out = analyze(gammaPixels(p, g), darkMaskOf(p));
    expect(out.mean).toBeGreaterThan(115);
    expect(out.mean).toBeLessThan(125);
  });
});

describe("analyzePair and describe", () => {
  it("compares an enhancement against matched plain brightening", () => {
    const original = noisy(32, 32, 30, 10);
    const enhanced = noisy(32, 32, 90, 4);
    const a = analyzePair(original, enhanced);
    expect(a.enhanced.mean).toBeGreaterThan(a.original.mean);
    expect(a.baselines).toHaveLength(2);
    expect(a.baselines[0].metrics.mean).toBeGreaterThan(a.original.mean);
    expect(a.insights.length).toBeGreaterThanOrEqual(4);
    expect(a.insights.join(" ")).toMatch(/brighter/);
    expect(a.insights.join(" ")).toMatch(/Fine pixel variation is \d+% lower/);
    expect(a.insights.join(" ")).toContain("lost detail");
  });

  it("notes when shadows are lifted more than matched plain brightening", () => {
    const flat = solid(8, 8, [30, 30, 30]);
    const mask = darkMaskOf(flat);
    const original = analyze(flat, mask);
    const enhanced = { ...analyze(solid(8, 8, [90, 90, 90]), mask), mean: 100 };
    const matched = { ...analyze(solid(8, 8, [60, 60, 60]), mask), mean: 100 };
    const lines = describeChange(original, enhanced, matched);
    expect(lines.join(" ")).toContain("originally dark pixels are brighter");
    expect(lines.join(" ")).not.toContain("bright areas are left alone");
  });

  it("warns when highlights clip and calls a tiny lift tiny", () => {
    const original = solid(8, 8, [100, 100, 100]);
    const blown = solid(8, 8, [255, 255, 255]);
    const lines = describeChange(analyze(original, darkMaskOf(original)), analyze(blown, darkMaskOf(original)), analyze(original, darkMaskOf(original)));
    expect(lines.some((l) => l.startsWith("Warning"))).toBe(true);

    const dim = solid(8, 8, [10, 10, 10]);
    const same = describeChange(analyze(dim, darkMaskOf(dim)), analyze(dim, darkMaskOf(dim)), analyze(dim, darkMaskOf(dim)));
    expect(same[0]).toMatch(/barely lifted/);
  });

  it("does not infer color fidelity from unchanged saturation", () => {
    const result = analyzePair(solid(8, 8, [200, 0, 0]), solid(8, 8, [0, 200, 0]));
    expect(result.insights.join(" ")).not.toMatch(/color is preserved/i);
    expect(result.insights.join(" ")).toContain("does not establish whether hues were preserved");
  });

  it("labels unreachable gamma targets without claiming matched brightness", () => {
    const result = analyzePair(solid(8, 8, [0, 0, 0]), solid(8, 8, [100, 100, 100]));
    expect(result.brightnessMatched).toBe(false);
    expect(result.baselines[0].label).toContain("closest available");
    expect(result.baselines[0].note).toContain("target not reached");
    expect(result.baselines[0].metrics.mean).toBe(0);
    expect(result.insights.join(" ")).not.toContain("same overall brightness");
  });

  it("does not invent a shadow sample or rule out individual channel clipping", () => {
    const result = analyzePair(solid(8, 8, [200, 0, 0]), solid(8, 8, [255, 0, 0]));
    expect(result.original.darkCount).toBe(0);
    expect(result.insights[0]).toContain("shadow lift cannot be measured");
    expect(result.insights.join(" ")).toContain("does not detect clipping of individual color channels");
    expect(result.insights.join(" ")).not.toContain("No meaningful highlight clipping");
  });

  it("describes darker shadows and a zero-to-positive lift without a misleading ratio", () => {
    const darker = analyzePair(solid(8, 8, [20, 20, 20]), solid(8, 8, [5, 5, 5]));
    expect(darker.insights[0]).toContain("became darker");
    const lifted = analyzePair(solid(8, 8, [0, 0, 0]), solid(8, 8, [100, 100, 100]));
    expect(lifted.insights[0]).toContain("brightened from 0 to 100");
    expect(lifted.insights[0]).not.toContain("x brighter");
    const unchangedBlack = analyzePair(solid(8, 8, [0, 0, 0]), solid(8, 8, [0, 0, 0]));
    expect(unchangedBlack.insights[0]).not.toContain("became darker");
  });
});
