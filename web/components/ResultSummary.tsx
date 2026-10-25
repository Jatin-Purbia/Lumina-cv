import type { ReactElement } from "react";
import { formatBytes } from "@/lib/validate";

export interface ResultSummaryProps {
  elapsedMs: number | null;
  size: { width: number; height: number } | null;
  bytes: number;
  downloadUrl: string;
  downloadName: string;
}

export function ResultSummary({ elapsedMs, size, bytes, downloadUrl, downloadName }: ResultSummaryProps): ReactElement {
  return (
    <section className="result" aria-labelledby="result-title">
      <h2 id="result-title" className="panel__heading">
        Enhanced image
      </h2>
      <dl className="facts">
        <div>
          <dt>Server time</dt>
          <dd>{elapsedMs === null ? "Not reported" : `${Math.round(elapsedMs)} ms`}</dd>
        </div>
        <div>
          <dt>Output size</dt>
          <dd>{size === null ? "Not reported" : `${size.width} × ${size.height} px`}</dd>
        </div>
        <div>
          <dt>File size</dt>
          <dd>{formatBytes(bytes)}</dd>
        </div>
      </dl>
      <a className="btn btn--primary" href={downloadUrl} download={downloadName}>
        Download PNG
      </a>
    </section>
  );
}
