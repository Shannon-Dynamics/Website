'use client';

import type { ReactNode } from 'react';
import { Pause, Play, RotateCcw, SkipForward, Shuffle } from 'lucide-react';

/* -------------------------------------------------------------------------- */
/* Transport                                                                   */
/* -------------------------------------------------------------------------- */

export interface TransportProps {
  playing: boolean;
  onToggle: () => void;
  onStep?: () => void;
  onReset?: () => void;
  onReseed?: () => void;
  seed?: number;
  tick?: number;
  speed?: number;
  onSpeed?: (v: number) => void;
}

/**
 * The standard control bar under every animated widget.
 *
 * Order is deliberate: play/pause first (the only control most readers touch),
 * then step, then the destructive-ish reset and re-roll. The seed is shown, not
 * hidden, because reproducibility is part of what the book is teaching.
 */
export function Transport({
  playing,
  onToggle,
  onStep,
  onReset,
  onReseed,
  seed,
  tick,
  speed,
  onSpeed,
}: TransportProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-fd-border px-3 py-2 text-sm">
      <button
        type="button"
        onClick={onToggle}
        aria-label={playing ? 'Pause simulation' : 'Play simulation'}
        className="inline-flex items-center gap-1.5 rounded-sm border border-fd-border bg-fd-card px-2.5 py-1 font-ui text-xs font-medium transition-colors hover:bg-fd-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-primary"
      >
        {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
        {playing ? 'Pause' : 'Play'}
      </button>

      {onStep ? (
        <IconButton onClick={onStep} label="Step one iteration">
          <SkipForward className="size-3.5" />
        </IconButton>
      ) : null}
      {onReset ? (
        <IconButton onClick={onReset} label="Reset to the start">
          <RotateCcw className="size-3.5" />
        </IconButton>
      ) : null}
      {onReseed ? (
        <IconButton onClick={onReseed} label="Re-roll the random seed">
          <Shuffle className="size-3.5" />
        </IconButton>
      ) : null}

      <div className="ml-auto flex items-center gap-3 font-mono text-[0.7rem] text-fd-muted-foreground tabular-nums">
        {onSpeed && speed !== undefined ? (
          <label className="flex items-center gap-1.5">
            <span className="eyebrow">speed</span>
            <input
              type="range"
              min={2}
              max={60}
              step={1}
              value={speed}
              onChange={(e) => onSpeed(Number(e.target.value))}
              className="h-1 w-20"
              aria-label="Simulation speed in steps per second"
            />
          </label>
        ) : null}
        {tick !== undefined ? <span>t={tick}</span> : null}
        {seed !== undefined ? <span>seed={seed}</span> : null}
      </div>
    </div>
  );
}

function IconButton({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="inline-flex items-center justify-center rounded-sm border border-fd-border bg-fd-card p-1.5 text-fd-muted-foreground transition-colors hover:bg-fd-accent hover:text-fd-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-primary"
    >
      {children}
    </button>
  );
}

/* -------------------------------------------------------------------------- */
/* Parameter controls                                                          */
/* -------------------------------------------------------------------------- */

export interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  /** Rendered after the numeric value, e.g. "m" or "rad". */
  unit?: string;
  /** Tie this parameter to a role color so the reader connects it to the figure. */
  role?: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
  format?: (v: number) => string;
  help?: string;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  unit,
  role,
  format,
  help,
}: SliderProps) {
  const shown = format ? format(value) : value.toFixed(step >= 1 ? 0 : 2);
  return (
    <label className="flex flex-col gap-1" title={help}>
      <span className="flex items-baseline justify-between gap-2">
        <span
          className="font-ui text-[0.72rem] font-medium"
          style={role ? { color: `var(--pr-${role})` } : undefined}
        >
          {label}
        </span>
        <span className="font-mono text-[0.7rem] text-fd-muted-foreground tabular-nums">
          {shown}
          {unit ? <span className="ml-0.5 opacity-70">{unit}</span> : null}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-full"
        aria-label={label}
      />
    </label>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
  role,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  role?: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2 font-ui text-[0.72rem] font-medium">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="size-3.5 accent-fd-primary"
      />
      <span style={role && checked ? { color: `var(--pr-${role})` } : undefined}>{label}</span>
    </label>
  );
}

export function ButtonRow({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center gap-2">{children}</div>;
}

export function ActionButton({
  onClick,
  children,
  emphasis = false,
}: {
  onClick: () => void;
  children: ReactNode;
  emphasis?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        emphasis
          ? 'rounded-sm bg-fd-primary px-2.5 py-1 font-ui text-xs font-medium text-fd-primary-foreground transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-primary'
          : 'rounded-sm border border-fd-border bg-fd-card px-2.5 py-1 font-ui text-xs font-medium transition-colors hover:bg-fd-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-primary'
      }
    >
      {children}
    </button>
  );
}

/** A panel of parameter controls beside or beneath a canvas. */
export function ControlPanel({
  children,
  columns = 2,
  title,
}: {
  children: ReactNode;
  columns?: 1 | 2 | 3;
  title?: string;
}) {
  const cols = columns === 1 ? 'sm:grid-cols-1' : columns === 3 ? 'sm:grid-cols-3' : 'sm:grid-cols-2';
  return (
    <div className="border-t border-fd-border px-3 py-3">
      {title ? <p className="eyebrow mb-2">{title}</p> : null}
      <div className={`grid grid-cols-1 gap-x-5 gap-y-2.5 ${cols}`}>{children}</div>
    </div>
  );
}
