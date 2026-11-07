"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface ManagedObjectUrl {
  url: string | null;
  /** Revokes the previous URL and, for a Blob, creates and returns a new one. */
  replace: (blob: Blob | null) => string | null;
}

/**
 * Owns at most one object URL. URLs are created in event handlers (never during
 * render), revoked when replaced, and revoked when the component unmounts.
 */
export function useObjectUrl(): ManagedObjectUrl {
  const [url, setUrl] = useState<string | null>(null);
  const current = useRef<string | null>(null);

  const replace = useCallback((blob: Blob | null): string | null => {
    if (current.current !== null) URL.revokeObjectURL(current.current);
    current.current = blob === null ? null : URL.createObjectURL(blob);
    setUrl(current.current);
    return current.current;
  }, []);

  useEffect(
    () => () => {
      if (current.current !== null) {
        URL.revokeObjectURL(current.current);
        current.current = null;
      }
    },
    [],
  );

  return { url, replace };
}
