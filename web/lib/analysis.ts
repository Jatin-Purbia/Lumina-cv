/** Image statistics used by the Analysis panel. Pure functions over RGBA pixel buffers. */

export interface Pixels {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface Metrics {
  /** Mean brightness, 0-255 (mean of R, G, B). */
  mean: number;
  /** Mean brightness of the pixels that were dark in the original image, 0-255. */
  darkMean: number;
  /** Standard deviation of brightness, 0-255 (global contrast). */
  contrast: number;
  /** Mean absolute Laplacian of brightness; a noise and fine-texture proxy. */
  noise: number;
  /** Mean HSV saturation, 0-1. */
  saturation: number;
  /** Shannon entropy of the brightness histogram in bits (tonal richness). */
  entropy: number;
  /** Share of pixels with brightness above 98% of white, 0-100. */
  clipped: number;
  /** Share of pixels below 10% brightness (crushed shadows), 0-100. */
  crushed: number;
  /** Normalized brightness histogram (sums to 1). */
  histogram: number[];
}

export const HISTOGRAM_BINS = 64;
/** Pixels below this brightness in the original image count as "dark" (about 12% of white). */
export const DARK_LEVEL = 0.12 * 255;

function brightness(p: Pixels): Float32Array {
  const count = p.width * p.height;
  const out = new Float32Array(count);
  for (let i = 0, j = 0; i < count; i += 1, j += 4) {
    out[i] = (p.data[j] + p.data[j + 1] + p.data[j + 2]) / 3;
  }
  return out;
}

/** Marks the pixels that are dark in `original`. */
export function darkMaskOf(original: Pixels): Uint8Array {
  const y = brightness(original);
  const mask = new Uint8Array(y.length);
  for (let i = 0; i < y.length; i += 1) mask[i] = y[i] < DARK_LEVEL ? 1 : 0;
  return mask;
}

/** Computes every metric. `darkMask` must have one entry per pixel (from the original image). */
export function analyze(p: Pixels, darkMask: Uint8Array): Metrics {
  const { width, height, data } = p;
  const count = width * height;
  const y = brightness(p);

  let sum = 0;
  let darkSum = 0;
  let darkCount = 0;
  let clipped = 0;
  let crushed = 0;
  let saturation = 0;
  const bins = new Array<number>(HISTOGRAM_BINS).fill(0);
  for (let i = 0, j = 0; i < count; i += 1, j += 4) {
    const v = y[i];
    sum += v;
    if (darkMask[i] === 1) {
      darkSum += v;
      darkCount += 1;
    }
    if (v > 0.98 * 255) clipped += 1;
    if (v < 0.1 * 255) crushed += 1;
    const hi = Math.max(data[j], data[j + 1], data[j + 2]);
    const lo = Math.min(data[j], data[j + 1], data[j + 2]);
    saturation += (hi - lo) / (hi + 1e-6);
    bins[Math.min(HISTOGRAM_BINS - 1, Math.floor((v / 256) * HISTOGRAM_BINS))] += 1;
  }
  const mean = sum / count;

  let variance = 0;
  for (let i = 0; i < count; i += 1) variance += (y[i] - mean) ** 2;

  let lap = 0;
  let lapCount = 0;
  for (let row = 1; row < height - 1; row += 1) {
    for (let col = 1; col < width - 1; col += 1) {
      const i = row * width + col;
      lap += Math.abs(4 * y[i] - y[i - 1] - y[i + 1] - y[i - width] - y[i + width]);
      lapCount += 1;
    }
  }

  const histogram = bins.map((b) => b / count);
  let entropy = 0;
  for (const prob of histogram) if (prob > 0) entropy -= prob * Math.log2(prob);

  return {
    mean,
    darkMean: darkCount > 0 ? darkSum / darkCount : mean,
    contrast: Math.sqrt(variance / count),
    noise: lapCount > 0 ? lap / lapCount : 0,
    saturation: saturation / count,
    entropy,
    clipped: (clipped / count) * 100,
    crushed: (crushed / count) * 100,
    histogram,
  };
}

/** Applies out = 255 * (in / 255) ** gamma to R, G and B (a plain brightness boost for gamma < 1). */
export function gammaPixels(p: Pixels, gamma: number): Pixels {
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v += 1) lut[v] = Math.round(255 * (v / 255) ** gamma);
  const data = new Uint8ClampedArray(p.data.length);
  for (let i = 0; i < p.data.length; i += 4) {
    data[i] = lut[p.data[i]];
    data[i + 1] = lut[p.data[i + 1]];
    data[i + 2] = lut[p.data[i + 2]];
    data[i + 3] = 255;
  }
  return { data, width: p.width, height: p.height };
}

/** The gamma whose output has (about) the given mean brightness; found by bisection on the channel histogram. */
export function matchGamma(p: Pixels, targetMean: number): number {
  const counts = new Float64Array(256);
  for (let i = 0; i < p.data.length; i += 4) {
    counts[p.data[i]] += 1;
    counts[p.data[i + 1]] += 1;
    counts[p.data[i + 2]] += 1;
  }
  const total = (p.data.length / 4) * 3;
  const meanFor = (g: number): number => {
    let s = 0;
    for (let v = 0; v < 256; v += 1) s += counts[v] * 255 * (v / 255) ** g;
    return s / total;
  };
  let lo = 0.1;
  let hi = 3;
  for (let step = 0; step < 24; step += 1) {
    const mid = (lo + hi) / 2;
    // A smaller gamma is brighter.
    if (meanFor(mid) > targetMean) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

export interface Comparison {
  label: string;
  note: string;
  metrics: Metrics;
}

export interface Analysis {
  original: Metrics;
  enhanced: Metrics;
  baselines: Comparison[];
  matchedGamma: number;
  insights: string[];
}

const num = (v: number, digits = 0): string => v.toFixed(digits);
const pct = (v: number): string => `${Math.abs(Math.round(v * 100))}%`;

/** Plain-language findings, derived only from the measured numbers. */
export function describe(original: Metrics, enhanced: Metrics, matched: Metrics): string[] {
  const out: string[] = [];

  const darkRatio = enhanced.darkMean / Math.max(original.darkMean, 1);
  if (darkRatio < 1.15) {
    out.push(
      `Shadows were barely lifted: dark areas went from ${num(original.darkMean)} to ${num(enhanced.darkMean)} (out of 255).`,
    );
  } else {
    out.push(
      `Dark areas are ${num(darkRatio, 1)}x brighter (${num(original.darkMean)} to ${num(enhanced.darkMean)} out of 255); overall brightness went from ${num(original.mean)} to ${num(enhanced.mean)}.`,
    );
  }

  if (matched.darkMean > 0 && enhanced.darkMean > matched.darkMean * 1.08) {
    out.push(
      `At the same overall brightness, shadows are lifted more than plain brightening would manage (${num(enhanced.darkMean)} against ${num(matched.darkMean)}), so bright areas are left alone.`,
    );
  }

  const noiseChange = (enhanced.noise - original.noise) / Math.max(original.noise, 1e-6);
  if (noiseChange < -0.08) {
    out.push(`Noise and fine grain are ${pct(noiseChange)} lower than the original.`);
  } else if (noiseChange > 0.08) {
    out.push(`Noise and fine grain are ${pct(noiseChange)} higher than the original (brightening amplified them).`);
  } else {
    out.push("Noise and fine grain are about the same as the original.");
  }

  const satRatio = enhanced.saturation / Math.max(original.saturation, 1e-6);
  const matchedRatio = matched.saturation / Math.max(original.saturation, 1e-6);
  if (satRatio >= 0.93) {
    out.push(`Color is preserved (saturation ${pct(satRatio)} of the original).`);
  } else {
    out.push(`Colors look ${pct(1 - satRatio)} less saturated than the original.`);
  }
  if (matchedRatio < satRatio - 0.08) {
    out.push(
      `A plain brightness boost to the same overall level would have washed out color more (saturation ${pct(matchedRatio)} of the original).`,
    );
  }

  const clippedChange = enhanced.clipped - original.clipped;
  if (clippedChange > 2) {
    out.push(`Warning: ${num(clippedChange, 1)} percentage points more of the image is near-white (highlights clipped).`);
  } else {
    out.push("No meaningful highlight clipping was introduced.");
  }

  const contrastChange = enhanced.contrast - original.contrast;
  if (contrastChange < -4) {
    out.push(`Global contrast dropped by ${num(-contrastChange)} (${num(original.contrast)} to ${num(enhanced.contrast)}).`);
  } else if (contrastChange > 4) {
    out.push(`Global contrast rose by ${num(contrastChange)} (${num(original.contrast)} to ${num(enhanced.contrast)}).`);
  }

  const entropyChange = enhanced.entropy - original.entropy;
  if (entropyChange < -0.15) {
    out.push(`Some tonal variety was smoothed away (tonal richness ${num(original.entropy, 2)} to ${num(enhanced.entropy, 2)} bits).`);
  } else if (entropyChange > 0.15) {
    out.push(`More tonal variety is visible (tonal richness ${num(original.entropy, 2)} to ${num(enhanced.entropy, 2)} bits).`);
  }

  return out;
}

/** Full analysis of an original/enhanced pair of equally sized pixel buffers. */
export function analyzePair(original: Pixels, enhanced: Pixels): Analysis {
  const mask = darkMaskOf(original);
  const originalMetrics = analyze(original, mask);
  const enhancedMetrics = analyze(enhanced, mask);
  const matchedGamma = matchGamma(original, enhancedMetrics.mean);
  const matched = analyze(gammaPixels(original, matchedGamma), mask);
  const strong = analyze(gammaPixels(original, 0.4), mask);
  return {
    original: originalMetrics,
    enhanced: enhancedMetrics,
    matchedGamma,
    baselines: [
      {
        label: "Plain brightening, same overall brightness",
        note: `gamma ${matchedGamma.toFixed(2)}`,
        metrics: matched,
      },
      { label: "Strong plain brightening", note: "gamma 0.40", metrics: strong },
    ],
    insights: describe(originalMetrics, enhancedMetrics, matched),
  };
}
