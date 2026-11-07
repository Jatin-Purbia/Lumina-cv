/** Server default for the largest accepted width x height (LUMINA_MAX_PIXELS). */
export const DEFAULT_MAX_PIXELS = 2 * 1024 * 1024;

/**
 * Size that keeps the aspect ratio and fits both limits, or null when the image already fits.
 * The result never exceeds either limit (dimensions are floored).
 */
export function fitDimensions(
  width: number,
  height: number,
  maxPixels: number,
  maxSide?: number,
): { width: number; height: number } | null {
  let scale = Math.min(1, Math.sqrt(maxPixels / (width * height)));
  if (maxSide !== undefined) scale = Math.min(scale, maxSide / Math.max(width, height));
  if (!(scale < 1)) return null;
  let w = Math.max(1, Math.floor(width * scale));
  let h = Math.max(1, Math.floor(height * scale));
  while (w * h > maxPixels && (w > 1 || h > 1)) {
    w = Math.max(1, w - 1);
    h = Math.max(1, Math.floor((h * w) / (w + 1)));
  }
  return { width: w, height: h };
}

/**
 * Returns the file unchanged when it fits the server's pixel limits, otherwise a downscaled PNG.
 * Any browser failure returns the original file so the server can give its own error.
 */
export async function fitToPixelLimit(file: File, maxPixels: number, maxSide?: number): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const target = fitDimensions(bitmap.width, bitmap.height, maxPixels, maxSide);
    if (target === null) {
      bitmap.close();
      return file;
    }
    const canvas = document.createElement("canvas");
    canvas.width = target.width;
    canvas.height = target.height;
    const context = canvas.getContext("2d");
    if (context === null) {
      bitmap.close();
      return file;
    }
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, target.width, target.height);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (blob === null) return file;
    const dot = file.name.lastIndexOf(".");
    const base = (dot > 0 ? file.name.slice(0, dot) : file.name) || "image";
    return new File([blob], `${base}.png`, { type: "image/png" });
  } catch {
    return file;
  }
}
