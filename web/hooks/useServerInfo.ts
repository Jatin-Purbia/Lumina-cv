"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, getOptions } from "@/lib/api";
import type { HealthResponse, OptionsResponse } from "@/lib/types";

export type HealthState =
  | { phase: "checking" }
  | { phase: "ready"; health: HealthResponse }
  | { phase: "no-model"; health: HealthResponse }
  | { phase: "unreachable" };

export const HEALTH_INTERVAL_MS = 15_000;

/**
 * Polls /health on a timer and whenever the window regains focus or becomes
 * visible. Loads /options once the server is reachable (and again after a failure).
 */
export function useServerInfo(intervalMs: number = HEALTH_INTERVAL_MS): {
  health: HealthState;
  options: OptionsResponse | null;
  refresh: () => void;
} {
  const [health, setHealth] = useState<HealthState>({ phase: "checking" });
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const optionsLoaded = useRef(false);
  const inFlight = useRef<AbortController | null>(null);
  const optionsFlight = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    getHealth(controller.signal)
      .then((h) => {
        if (controller.signal.aborted) return;
        setHealth(h.model_loaded ? { phase: "ready", health: h } : { phase: "no-model", health: h });
        if (!optionsLoaded.current) {
          optionsLoaded.current = true;
          const optionsController = new AbortController();
          optionsFlight.current = optionsController;
          getOptions(optionsController.signal)
            .then((o) => {
              if (!optionsController.signal.aborted) setOptions(o);
            })
            .catch(() => {
              // Allow the next health check to retry /options.
              optionsLoaded.current = false;
            });
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setHealth({ phase: "unreachable" });
      });
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, intervalMs);
    const onVisible = (): void => {
      if (document.visibilityState === "visible") refresh();
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", onVisible);
      inFlight.current?.abort();
      optionsFlight.current?.abort();
    };
  }, [refresh, intervalMs]);

  return { health, options, refresh };
}
