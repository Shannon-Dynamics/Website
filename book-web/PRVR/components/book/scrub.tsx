'use client';

import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useExplorableState, type Role } from '@/lib/explorable/store';

export interface ScrubProps {
  /** Namespaced key a figure reads, e.g. "ch05.sensorSigma". */
  id: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  /** Appended after the number, e.g. "m" or "°". */
  unit?: string;
  /** Tints the number with the estimation palette so it matches the figure. */
  role?: Role;
  /** Decimal places; inferred from `step` when omitted. */
  precision?: number;
  /** Describes the quantity for assistive technology. */
  label?: string;
}

/**
 * A number in a sentence that the reader can drag.
 *
 * The prose stays readable if nobody touches it — it is simply the value the
 * text claims — but a reader who wants to ask "what if the sensor were worse?"
 * can answer it without leaving the paragraph. Dragging updates the figure
 * that shares this key.
 *
 * Keyboard: arrows step, shift+arrow steps by ten, Home/End jump to the range.
 * That matters here: a control that only works with a mouse is not a control,
 * it is a decoration.
 */
export function Scrub({
  id,
  value: initial,
  min,
  max,
  step,
  unit,
  role,
  precision,
  label,
}: ScrubProps) {
  const inferredStep = step ?? (max - min) / 100;
  const digits =
    precision ?? Math.min(3, Math.max(0, -Math.floor(Math.log10(inferredStep || 0.01))));

  const [value, setValue] = useExplorableState(id, initial);
  const [dragging, setDragging] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const startX = useRef(0);
  const startValue = useRef(initial);
  const describedBy = useId();

  const clamp = useCallback(
    (v: number) => {
      const snapped = Math.round(v / inferredStep) * inferredStep;
      return Math.min(max, Math.max(min, snapped));
    },
    [inferredStep, min, max],
  );

  // Pointer capture on the element itself would lose the drag when the value
  // re-renders, so the move/up handlers live on the window for the duration.
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: PointerEvent) => {
      // One full range sweep is ~260 px, which feels right for a word-sized target.
      const delta = ((e.clientX - startX.current) / 260) * (max - min);
      setValue(clamp(startValue.current + delta));
    };
    const onUp = () => setDragging(false);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [dragging, clamp, max, min, setValue]);

  useEffect(() => {
    if (!dragging) return;
    const previous = document.body.style.cursor;
    document.body.style.cursor = 'ew-resize';
    document.body.style.userSelect = 'none';
    return () => {
      document.body.style.cursor = previous;
      document.body.style.userSelect = '';
    };
  }, [dragging]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const big = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      setValue(clamp(value + inferredStep * big));
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      setValue(clamp(value - inferredStep * big));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setValue(min);
    } else if (e.key === 'End') {
      e.preventDefault();
      setValue(max);
    }
  };

  const pct = ((value - min) / (max - min)) * 100;

  return (
    <span
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-valuenow={Number(value.toFixed(digits))}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-label={label ?? id.split('.').pop()}
      aria-describedby={describedBy}
      onPointerDown={(e) => {
        e.preventDefault();
        startX.current = e.clientX;
        startValue.current = value;
        setDragging(true);
      }}
      onKeyDown={onKeyDown}
      data-scrubbing={dragging || undefined}
      className="pr-scrub"
      style={
        {
          '--pr-scrub-fill': `${pct}%`,
          ...(role ? { '--pr-scrub-ink': `var(--pr-${role})` } : {}),
        } as React.CSSProperties
      }
    >
      <span className="pr-scrub-value tabular-nums">{value.toFixed(digits)}</span>
      {unit ? <span className="pr-scrub-unit">{unit}</span> : null}
      <span id={describedBy} className="sr-only">
        Draggable value. Use the arrow keys to adjust, shift for larger steps.
      </span>
    </span>
  );
}
