export const DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
export const DEFAULT_ACCEPTED_TYPES: readonly string[] = ["image/png", "image/jpeg"];

export interface ValidationRules {
  maxBytes: number;
  acceptedTypes: readonly string[];
}

export const DEFAULT_RULES: ValidationRules = {
  maxBytes: DEFAULT_MAX_UPLOAD_BYTES,
  acceptedTypes: DEFAULT_ACCEPTED_TYPES,
};

export interface ValidationFailure {
  code: "empty" | "unsupported_type" | "too_large";
  message: string;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function typeLabel(types: readonly string[]): string {
  const names = types.map((t) => t.replace(/^image\//, "").toUpperCase());
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} or ${names[names.length - 1]}`;
}

/** Returns null when the file may be uploaded, otherwise the reason it may not. */
export function validateFile(
  file: Pick<File, "name" | "size" | "type">,
  rules: ValidationRules = DEFAULT_RULES,
): ValidationFailure | null {
  if (file.size === 0) {
    return { code: "empty", message: `${file.name} is empty. Choose an image that contains data.` };
  }
  if (!rules.acceptedTypes.includes(file.type)) {
    return {
      code: "unsupported_type",
      message: `${file.name} is not a supported image. Choose a ${typeLabel(rules.acceptedTypes)} file.`,
    };
  }
  if (file.size > rules.maxBytes) {
    return {
      code: "too_large",
      message: `${file.name} is ${formatBytes(file.size)}. The limit is ${formatBytes(rules.maxBytes)}.`,
    };
  }
  return null;
}

/** Output file name for a download, e.g. "street.jpg" -> "street-enhanced.png". */
export function enhancedFileName(original: string): string {
  const dot = original.lastIndexOf(".");
  const stem = dot > 0 ? original.slice(0, dot) : original;
  return `${stem || "image"}-enhanced.png`;
}
