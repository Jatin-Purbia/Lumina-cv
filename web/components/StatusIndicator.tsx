import type { ReactElement } from "react";
import type { HealthState } from "@/hooks/useServerInfo";

const COPY: Record<HealthState["phase"], { label: string; tone: "ok" | "warn" | "down" | "wait" }> = {
  checking: { label: "Checking server", tone: "wait" },
  ready: { label: "Model loaded", tone: "ok" },
  "no-model": { label: "Server up, no model loaded", tone: "warn" },
  unreachable: { label: "Server unreachable", tone: "down" },
};

export function StatusIndicator({ health }: { health: HealthState }): ReactElement {
  const { label, tone } = COPY[health.phase];
  const detail =
    health.phase === "ready" || health.phase === "no-model"
      ? `Device ${health.health.device}, version ${health.health.version}`
      : undefined;
  return (
    <p className={`status status--${tone}`} title={detail} data-testid="health-status">
      <span className="status__dot" aria-hidden="true" />
      <span>
        <span className="sr-only">Server status: </span>
        {label}
      </span>
    </p>
  );
}
