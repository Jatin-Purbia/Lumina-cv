"use client";

import type { DragEvent, ReactElement } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { CompareSlider } from "./CompareSlider";
import { FilePicker } from "./FilePicker";
import { ResultSummary } from "./ResultSummary";
import { ScalePicker } from "./ScalePicker";
import { StatusIndicator } from "./StatusIndicator";
import { useObjectUrl } from "@/hooks/useObjectUrl";
import { useServerInfo } from "@/hooks/useServerInfo";
import { enhance } from "@/lib/api";
import { DEFAULT_MAX_PIXELS, fitToPixelLimit } from "@/lib/resize";
import { ApiError, describeError, isAbortError, type UserFacingError } from "@/lib/errors";
import {
  DEFAULT_ACCEPTED_TYPES,
  DEFAULT_MAX_UPLOAD_BYTES,
  enhancedFileName,
  formatBytes,
  validateFile,
} from "@/lib/validate";

interface ResultMeta {
  elapsedMs: number | null;
  size: { width: number; height: number } | null;
  bytes: number;
}

export function EnhanceApp(): ReactElement {
  const { health, options, refresh } = useServerInfo();
  const source = useObjectUrl();
  const result = useObjectUrl();

  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState<ResultMeta | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<UserFacingError | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const [dragging, setDragging] = useState(false);
  const [curveScale, setCurveScale] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const rules = useMemo(
    () => ({
      maxBytes: options?.max_upload_bytes ?? DEFAULT_MAX_UPLOAD_BYTES,
      acceptedTypes:
        options?.accepted_types !== undefined && options.accepted_types.length > 0
          ? options.accepted_types
          : DEFAULT_ACCEPTED_TYPES,
    }),
    [options],
  );

  useEffect(() => () => abortRef.current?.abort(), []);

  const selectFile = (candidate: File): void => {
    if (busy) return;
    const failure = validateFile(candidate, rules);
    if (failure !== null) {
      setError({ title: "This file cannot be used", message: failure.message });
      setAnnouncement(failure.message);
      return;
    }
    source.replace(candidate);
    result.replace(null);
    setFile(candidate);
    setMeta(null);
    setError(null);
    setAnnouncement(`${candidate.name} selected, ${formatBytes(candidate.size)}.`);
  };

  const runEnhance = async (): Promise<void> => {
    if (file === null || busy) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setError(null);
    setAnnouncement("Enhancing image.");
    try {
      const upload = await fitToPixelLimit(
        file,
        options?.max_pixels ?? DEFAULT_MAX_PIXELS,
        options?.max_side,
      );
      if (controller.signal.aborted) return;
      const out = await enhance({ file: upload, curveScale, signal: controller.signal });
      if (controller.signal.aborted) return;
      result.replace(out.blob);
      setMeta({ elapsedMs: out.elapsedMs, size: out.size, bytes: out.blob.size });
      setAnnouncement("Enhancement finished. Use the divider to compare the images.");
    } catch (caught) {
      if (controller.signal.aborted || isAbortError(caught)) return;
      const described = describeError(caught);
      setError(described);
      setAnnouncement(`${described.title}. ${described.message}`);
      if (caught instanceof ApiError && (caught.kind === "network" || caught.code === "model_not_loaded")) {
        refresh();
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setBusy(false);
      }
    }
  };

  const cancel = (): void => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setAnnouncement("Enhancement canceled.");
  };

  const onDrag = (event: DragEvent<HTMLDivElement>): void => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    if (event.type === "dragleave") {
      if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
    } else {
      setDragging(!busy);
    }
  };

  const onDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped !== undefined) selectFile(dropped);
  };

  const showScale = typeof options?.curve_scale === "number";
  const resultUrl = result.url;
  const sourceUrl = source.url;
  const hasResult = resultUrl !== null && sourceUrl !== null && meta !== null;
  const acceptedLabel = rules.acceptedTypes.map((t) => t.replace("image/", "").toUpperCase()).join(" or ");

  return (
    <div className="shell">
      <header className="masthead">
        <p className="wordmark">Lumina CV</p>
        <StatusIndicator health={health} />
      </header>

      <main className="main">
        <div className="intro">
          <h1 className="intro__title">Bring the detail back to dark photos</h1>
          <p className="intro__lede">
            Add a low-light photo, enhance it, then drag the divider to compare it with the original.
          </p>
        </div>

        <div className="workspace">
          <div
            className={`stage${dragging ? " is-dragging" : ""}${file === null ? " stage--empty" : ""}`}
            onDragEnter={onDrag}
            onDragOver={onDrag}
            onDragLeave={onDrag}
            onDrop={onDrop}
            aria-busy={busy}
          >
            {file === null || sourceUrl === null ? (
              <div className="stage__empty">
                <p className="stage__prompt">Drop an image here</p>
                <p className="stage__sub">
                  {acceptedLabel}, up to {formatBytes(rules.maxBytes)}
                </p>
                <FilePicker accept={rules.acceptedTypes} label="Choose image" onPick={selectFile} variant="primary" />
              </div>
            ) : hasResult ? (
              <CompareSlider key={resultUrl} beforeSrc={sourceUrl} afterSrc={resultUrl} />
            ) : (
              <div className="stage__preview">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={sourceUrl} alt={`Preview of ${file.name}`} className="stage__img" />
              </div>
            )}
            {busy ? (
              <div className="stage__busy" aria-hidden="true">
                <span className="spinner" />
                <span>Enhancing</span>
              </div>
            ) : null}
          </div>

          <aside className="panel" aria-label="Enhancement controls">
            {file !== null ? (
              <section aria-labelledby="file-title">
                <h2 id="file-title" className="panel__heading">
                  Image
                </h2>
                <p className="file">
                  <span className="file__name">{file.name}</span>
                  <span className="file__size">{formatBytes(file.size)}</span>
                </p>
              </section>
            ) : (
              <p className="panel__empty">Controls appear once you add an image.</p>
            )}

            {file !== null && showScale && typeof options?.curve_scale === "number" ? (
              <ScalePicker
                defaultValue={options.curve_scale}
                value={curveScale}
                onChange={setCurveScale}
                disabled={busy}
              />
            ) : null}

            {file !== null ? (
              <div className="actions">
                <button type="button" className="btn btn--primary" onClick={() => void runEnhance()} disabled={busy}>
                  {busy ? "Enhancing…" : hasResult ? "Enhance again" : "Enhance"}
                </button>
                {busy ? (
                  <button type="button" className="btn btn--quiet" onClick={cancel}>
                    Cancel
                  </button>
                ) : (
                  <FilePicker accept={rules.acceptedTypes} label="Replace image" onPick={selectFile} />
                )}
              </div>
            ) : null}

            {file !== null && (health.phase === "no-model" || health.phase === "unreachable") ? (
              <p className="field__hint">
                {health.phase === "no-model"
                  ? "The server reports that no model is loaded, so enhancing will fail until one is."
                  : "The server is not answering health checks, so enhancing will fail until it is back."}
              </p>
            ) : null}

            {error !== null ? (
              <div className="alert" role="alert">
                <p className="alert__title">{error.title}</p>
                <p>{error.message}</p>
              </div>
            ) : null}

            {hasResult && file !== null ? (
              <ResultSummary
                elapsedMs={meta.elapsedMs}
                size={meta.size}
                bytes={meta.bytes}
                downloadUrl={resultUrl}
                downloadName={enhancedFileName(file.name)}
              />
            ) : null}
          </aside>
        </div>

        <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
          {announcement}
        </div>
      </main>

      <footer className="footnote">
        <p>Images are sent only to the Lumina API server configured for this deployment.</p>
      </footer>
    </div>
  );
}
