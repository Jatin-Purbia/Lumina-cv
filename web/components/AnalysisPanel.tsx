"use client";

import type { ReactElement } from "react";
import { useEffect, useState } from "react";
import { analyzePair, HISTOGRAM_BINS, type Analysis, type Metrics } from "@/lib/analysis";
import { loadPixels } from "@/lib/pixels";

export interface AnalysisPanelProps {
  original: Blob;
  enhanced: Blob;
}

type State = { phase: "loading" } | { phase: "error" } | { phase: "ready"; analysis: Analysis };

const f0 = (v: number): string => v.toFixed(0);
const f1 = (v: number): string => v.toFixed(1);
const f2 = (v: number): string => v.toFixed(2);

interface Row {
  label: string;
  hint: string;
  read: (m: Metrics) => string;
  value: (m: Metrics) => number;
  /** Which direction counts as an improvement, or null when it depends on the image. */
  better: "up" | "down" | null;
}

const ROWS: Row[] = [
  {
    label: "Overall brightness",
    hint: "Mean brightness, 0 (black) to 255 (white).",
    read: (m) => f0(m.mean),
    value: (m) => m.mean,
    better: null,
  },
  {
    label: "Shadow brightness",
    hint: "Brightness of the pixels that were dark in the original.",
    read: (m) => f0(m.darkMean),
    value: (m) => m.darkMean,
    better: "up",
  },
  {
    label: "Contrast",
    hint: "Spread of brightness across the image.",
    read: (m) => f1(m.contrast),
    value: (m) => m.contrast,
    better: null,
  },
  {
    label: "Noise and grain",
    hint: "Fine pixel-to-pixel variation; lower is cleaner.",
    read: (m) => f1(m.noise),
    value: (m) => m.noise,
    better: "down",
  },
  {
    label: "Color saturation",
    hint: "How colorful the image is; plain brightening fades color.",
    read: (m) => f2(m.saturation),
    value: (m) => m.saturation,
    better: null,
  },
  {
    label: "Tonal richness",
    hint: "Entropy of the brightness histogram, in bits.",
    read: (m) => f2(m.entropy),
    value: (m) => m.entropy,
    better: null,
  },
  {
    label: "Clipped highlights",
    hint: "Share of near-white pixels; lower is safer.",
    read: (m) => `${f1(m.clipped)}%`,
    value: (m) => m.clipped,
    better: "down",
  },
];

function change(row: Row, before: Metrics, after: Metrics): { text: string; tone: "good" | "bad" | "flat" } {
  const a = row.value(before);
  const b = row.value(after);
  const diff = b - a;
  const rel = a === 0 ? 0 : diff / a;
  const flat = Math.abs(rel) < 0.03 && Math.abs(diff) < 0.5;
  const text = flat ? "no change" : `${diff > 0 ? "+" : ""}${Math.abs(rel) >= 0.01 && a !== 0 ? `${Math.round(rel * 100)}%` : f1(diff)}`;
  if (flat || row.better === null) return { text, tone: "flat" };
  return { text, tone: (row.better === "up") === diff > 0 ? "good" : "bad" };
}

function Histogram({ original, enhanced }: { original: number[]; enhanced: number[] }): ReactElement {
  const width = 256;
  const height = 90;
  const peak = Math.max(...original, ...enhanced, 1e-9);
  const path = (values: number[]): string =>
    values
      .map((v, i) => `${i === 0 ? "M" : "L"}${((i + 0.5) / HISTOGRAM_BINS) * width},${height - (v / peak) * (height - 4)}`)
      .join(" ");
  return (
    <figure className="hist">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Brightness histogram of the original and enhanced image"
        preserveAspectRatio="none"
      >
        <line x1="0" y1={height} x2={width} y2={height} className="hist__axis" />
        <path d={path(original)} className="hist__line hist__line--before" />
        <path d={path(enhanced)} className="hist__line hist__line--after" />
      </svg>
      <div className="hist__scale" aria-hidden="true">
        <span>dark</span>
        <span>brightness</span>
        <span>bright</span>
      </div>
      <figcaption className="hist__legend">
        <span className="hist__key hist__key--before">Original</span>
        <span className="hist__key hist__key--after">Enhanced</span>
      </figcaption>
    </figure>
  );
}

export function AnalysisPanel({ original, enhanced }: AnalysisPanelProps): ReactElement {
  const [state, setState] = useState<State>({ phase: "loading" });

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const after = await loadPixels(enhanced);
        const before = await loadPixels(original, { width: after.width, height: after.height });
        if (!cancelled) setState({ phase: "ready", analysis: analyzePair(before, after) });
      } catch {
        if (!cancelled) setState({ phase: "error" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [original, enhanced]);

  return (
    <section className="analysis" aria-labelledby="analysis-title">
      <h2 id="analysis-title" className="analysis__title">
        Analysis
      </h2>
      {state.phase === "loading" ? <p className="analysis__note">Measuring the images…</p> : null}
      {state.phase === "error" ? (
        <p className="analysis__note">The analysis could not be computed in this browser.</p>
      ) : null}
      {state.phase === "ready" ? <AnalysisBody analysis={state.analysis} /> : null}
    </section>
  );
}

function AnalysisBody({ analysis }: { analysis: Analysis }): ReactElement {
  const { original, enhanced, baselines, insights } = analysis;
  return (
    <>
      <div className="analysis__grid">
        <div>
          <h3 className="analysis__sub">What changed</h3>
          <ul className="insights">
            {insights.map((line) => (
              <li key={line} className={line.startsWith("Warning") ? "insights__warn" : undefined}>
                {line}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="analysis__sub">Brightness distribution</h3>
          <Histogram original={original.histogram} enhanced={enhanced.histogram} />
        </div>
      </div>

      <h3 className="analysis__sub">Measurements</h3>
      <div className="table-wrap">
        <table className="metrics">
          <thead>
            <tr>
              <th scope="col">Measure</th>
              <th scope="col">Original</th>
              <th scope="col">Enhanced</th>
              <th scope="col">Change</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => {
              const delta = change(row, original, enhanced);
              return (
                <tr key={row.label}>
                  <th scope="row">
                    {row.label}
                    <span className="metrics__hint">{row.hint}</span>
                  </th>
                  <td>{row.read(original)}</td>
                  <td>{row.read(enhanced)}</td>
                  <td className={`metrics__change metrics__change--${delta.tone}`}>{delta.text}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3 className="analysis__sub">Compared with plain brightening</h3>
      <div className="table-wrap">
        <table className="metrics metrics--compact">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">Overall</th>
              <th scope="col">Shadows</th>
              <th scope="col">Noise</th>
              <th scope="col">Color</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Original</th>
              <td>{f0(original.mean)}</td>
              <td>{f0(original.darkMean)}</td>
              <td>{f1(original.noise)}</td>
              <td>{f2(original.saturation)}</td>
            </tr>
            <tr className="metrics__ours">
              <th scope="row">Lumina (this result)</th>
              <td>{f0(enhanced.mean)}</td>
              <td>{f0(enhanced.darkMean)}</td>
              <td>{f1(enhanced.noise)}</td>
              <td>{f2(enhanced.saturation)}</td>
            </tr>
            {baselines.map((b) => (
              <tr key={b.label}>
                <th scope="row">
                  {b.label}
                  <span className="metrics__hint">{b.note}</span>
                </th>
                <td>{f0(b.metrics.mean)}</td>
                <td>{f0(b.metrics.darkMean)}</td>
                <td>{f1(b.metrics.noise)}</td>
                <td>{f2(b.metrics.saturation)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="analysis__note">
        Measured in your browser on this one image, at reduced size. There is no reference photo, so these are
        descriptive measurements, not a quality score. Plain brightening is a per-channel gamma curve applied to the
        original.
      </p>
    </>
  );
}
