"use client";

import type { CSSProperties, KeyboardEvent, PointerEvent, ReactElement } from "react";
import { useCallback, useRef, useState } from "react";

export const SLIDER_STEP = 2;
export const SLIDER_BIG_STEP = 10;

export interface CompareSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeAlt?: string;
  afterAlt?: string;
  /** Initial divider position, 0 (all enhanced) to 100 (all original). */
  initial?: number;
}

function clamp(value: number): number {
  return Math.min(100, Math.max(0, value));
}

/**
 * Before/after comparison. The original is revealed on the left of the divider and
 * the enhanced image on the right. The divider is a keyboard-operable slider.
 */
export function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeAlt = "Original photo",
  afterAlt = "Enhanced photo",
  initial = 50,
}: CompareSliderProps): ReactElement {
  const [position, setPosition] = useState(clamp(initial));
  const frame = useRef<HTMLDivElement | null>(null);
  const handle = useRef<HTMLDivElement | null>(null);
  const dragging = useRef(false);

  const moveTo = useCallback((clientX: number) => {
    const el = frame.current;
    if (el === null) return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return;
    setPosition(clamp(((clientX - rect.left) / rect.width) * 100));
  }, []);

  const onPointerDown = (event: PointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    dragging.current = true;
    if (typeof event.currentTarget.setPointerCapture === "function") {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    moveTo(event.clientX);
    handle.current?.focus({ preventScroll: true });
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>): void => {
    if (dragging.current) moveTo(event.clientX);
  };

  const endDrag = (): void => {
    dragging.current = false;
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const step = event.shiftKey ? SLIDER_BIG_STEP : SLIDER_STEP;
    let next: number | null = null;
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowDown":
        next = position - step;
        break;
      case "ArrowRight":
      case "ArrowUp":
        next = position + step;
        break;
      case "PageDown":
        next = position - SLIDER_BIG_STEP;
        break;
      case "PageUp":
        next = position + SLIDER_BIG_STEP;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = 100;
        break;
      default:
        return;
    }
    event.preventDefault();
    setPosition(clamp(next));
  };

  const rounded = Math.round(position);
  const style = { "--pos": `${position}%` } as CSSProperties;

  return (
    <div
      ref={frame}
      className="compare"
      style={style}
      role="group"
      aria-label="Before and after comparison"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {/* Blob URLs cannot use next/image optimisation. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="compare__img" src={afterSrc} alt={afterAlt} draggable={false} />
      <div className="compare__before">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="compare__img compare__img--overlay" src={beforeSrc} alt={beforeAlt} draggable={false} />
      </div>
      <span className="compare__tag compare__tag--before">Original</span>
      <span className="compare__tag compare__tag--after">Enhanced</span>
      <div className="compare__line" aria-hidden="true" />
      <div
        ref={handle}
        className="compare__handle"
        role="slider"
        tabIndex={0}
        aria-label="Comparison divider"
        aria-orientation="horizontal"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={rounded}
        aria-valuetext={`${rounded}% original, ${100 - rounded}% enhanced`}
        onKeyDown={onKeyDown}
      >
        <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false">
          <path d="M9 6 3 12l6 6M15 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    </div>
  );
}
