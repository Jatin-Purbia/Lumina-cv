import type { Pixels } from "./analysis";

/** Longest side used for analysis; keeps the statistics fast and compares both images at one scale. */
export const ANALYSIS_MAX_SIDE = 800;

export function analysisSize(
  width: number,
  height: number,
  maxSide = ANALYSIS_MAX_SIDE,
): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/** Decodes an image (honouring EXIF rotation) and returns its pixels, drawn at `size` when given. */
export async function loadPixels(blob: Blob, size?: { width: number; height: number }): Promise<Pixels> {
  const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  try {
    const target = size ?? analysisSize(bitmap.width, bitmap.height);
    const canvas = document.createElement("canvas");
    canvas.width = target.width;
    canvas.height = target.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (context === null) throw new Error("canvas is not available");
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, target.width, target.height);
    const image = context.getImageData(0, 0, target.width, target.height);
    return { data: image.data, width: target.width, height: target.height };
  } finally {
    bitmap.close();
  }
}
