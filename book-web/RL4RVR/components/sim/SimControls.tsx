'use client';

import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { cn } from '@/lib/utils';

/** Transport controls shared by every animated simulation in the book. */
export function SimControls({
  playing,
  onPlayPause,
  onStep,
  onReset,
  speed,
  onSpeedChange,
  disabled,
  children,
}: {
  playing: boolean;
  onPlayPause: () => void;
  onStep?: () => void;
  onReset: () => void;
  speed?: number;
  onSpeedChange?: (v: number) => void;
  disabled?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={onPlayPause}
        disabled={disabled}
        className="flex items-center gap-1.5 rounded-md bg-series-1 px-2.5 py-1.5 text-[12px] font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-40"
      >
        {playing ? <Pause size={13} /> : <Play size={13} />}
        {playing ? 'Pause' : 'Play'}
      </button>

      {onStep && (
        <button
          type="button"
          onClick={onStep}
          disabled={disabled || playing}
          className="flex items-center gap-1.5 rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink disabled:opacity-40"
        >
          <SkipForward size={13} />
          Step
        </button>
      )}

      <button
        type="button"
        onClick={onReset}
        className="flex items-center gap-1.5 rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
      >
        <RotateCcw size={13} />
        Reset
      </button>

      {onSpeedChange && speed !== undefined && (
        <label className="ml-1 flex items-center gap-1.5 text-[11.5px] text-ink-muted">
          Speed
          <input
            type="range"
            min={1}
            max={60}
            value={speed}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
            className="h-1 w-20 accent-[var(--series-1)]"
            aria-label="Simulation speed"
          />
        </label>
      )}

      {children}
    </div>
  );
}

/** Labeled slider with a live value read-out — the standard parameter control. */
export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  format,
  hint,
  className,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
  className?: string;
}) {
  return (
    <label className={cn('block', className)}>
      <span className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[11.5px] font-medium text-ink-secondary">{label}</span>
        <span className="tabular text-[11.5px] font-semibold text-ink">
          {format ? format(value) : value.toFixed(2)}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-full accent-[var(--series-1)]"
      />
      {hint ? <span className="mt-0.5 block text-[10.5px] text-ink-muted">{hint}</span> : null}
    </label>
  );
}

/** Segmented control for mutually exclusive modes. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <div>
      {label ? (
        <span className="mb-1 block text-[11.5px] font-medium text-ink-secondary">{label}</span>
      ) : null}
      <div
        className="inline-flex rounded-md border border-hairline p-0.5"
        role="group"
        aria-label={label}
      >
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            aria-pressed={value === o.value}
            className={cn(
              'rounded px-2 py-1 text-[11.5px] font-medium transition-colors',
              value === o.value
                ? 'bg-series-1 text-white'
                : 'text-ink-secondary hover:bg-surface-sunken hover:text-ink',
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Panel wrapper for a simulation: title bar, controls row, body. */
export function SimPanel({
  title,
  subtitle,
  id,
  controls,
  caption,
  children,
}: {
  title: string;
  subtitle?: string;
  /** Stable widget ID, e.g. `ch05-gpi-dashboard` — referenced from prose. */
  id?: string;
  controls?: React.ReactNode;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <figure
      id={id}
      className="my-7 overflow-hidden rounded-xl border border-hairline bg-surface"
      style={{ scrollMarginTop: '5rem' }}
    >
      <div className="border-b border-hairline px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h4 className="text-[13.5px] font-semibold tracking-tight text-ink">{title}</h4>
          {id ? (
            <code className="font-mono text-[10px] uppercase tracking-wide text-ink-muted">
              {id}
            </code>
          ) : null}
        </div>
        {subtitle ? (
          <p className="mt-0.5 text-[12px] leading-snug text-ink-muted">{subtitle}</p>
        ) : null}
      </div>
      {controls ? (
        <div className="border-b border-hairline bg-surface-sunken px-4 py-2.5">{controls}</div>
      ) : null}
      <div className="p-4">{children}</div>
      {caption ? (
        <figcaption className="border-t border-hairline px-4 py-2.5 text-[12.5px] leading-relaxed text-ink-muted">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
