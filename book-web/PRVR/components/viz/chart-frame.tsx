/**
 * Shared chrome for every chart in the book: the sized frame, the legend, the
 * tooltip card, the color-scale key, and the screen-reader table twin.
 *
 * None of it needs state, so it stays a plain shared component and is pulled
 * into whichever bundle imports it.
 */

import type { CSSProperties, ReactNode } from 'react';
import type { BookColors, BookRole } from '@/lib/chart-theme';
import { BOOK_ROLE_ORDER } from '@/lib/chart-theme';

/** Tiny class joiner — the project has no clsx dependency and needs none. */
export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ');
}

/**
 * Series color assignment. A declared `role` always wins, so blue keeps meaning
 * "prior" across every figure in the book; role-less series fall back to the
 * caller's palette, then to the reserved five in fixed order. More than five
 * unroled series is a signal to facet, not to invent a sixth hue.
 */
export function resolveSeriesColors<T extends { id: string; role?: BookRole }>(
  series: readonly T[],
  bookColors: BookColors,
  overrides?: readonly string[],
): Map<string, string> {
  return new Map(
    series.map((serie, index) => [
      serie.id,
      serie.role
        ? bookColors[serie.role]
        : (overrides?.[index] ?? bookColors[BOOK_ROLE_ORDER[index % BOOK_ROLE_ORDER.length]]),
    ]),
  );
}

export interface PivotableSeries<X extends string | number> {
  id: string;
  data: readonly { x: X; y: number }[];
}

/**
 * Turns per-series point lists into the index-major form bar charts and table
 * twins need, keeping first-seen index order unless asked to sort.
 */
export function pivotSeries<X extends string | number>(
  series: readonly PivotableSeries<X>[],
  options?: { sort?: boolean },
): { indices: X[]; valueAt: (index: X, seriesId: string) => number | undefined } {
  const indices: X[] = [];
  const seen = new Set<X>();
  const bySeries = new Map<string, Map<X, number>>();

  for (const serie of series) {
    const values = new Map<X, number>();
    for (const point of serie.data) {
      values.set(point.x, point.y);
      if (!seen.has(point.x)) {
        seen.add(point.x);
        indices.push(point.x);
      }
    }
    bySeries.set(serie.id, values);
  }

  if (options?.sort) {
    indices.sort((a, b) =>
      typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b)),
    );
  }

  return { indices, valueAt: (index, seriesId) => bySeries.get(seriesId)?.get(index) };
}

/* -------------------------------------------------------------------------- */

export interface ChartFrameProps {
  /** Rendered height of the plot in pixels, axis bands included. */
  height: number;
  /** The chart itself. */
  children: ReactNode;
  legend?: ReactNode;
  /** Table twin: every plotted value, reachable without seeing the chart. */
  table?: ReactNode;
  caption?: ReactNode;
  className?: string;
}

export function ChartFrame({ height, children, legend, table, caption, className }: ChartFrameProps) {
  // `not-prose` keeps chapter typography off the legend list and the table twin.
  return (
    <figure className={cx('not-prose m-0 flex w-full flex-col gap-2.5', className)}>
      <div className="w-full" style={{ height }}>
        {children}
      </div>
      {legend}
      {table}
      {caption ? (
        <figcaption className="font-ui text-[0.78rem] leading-snug text-fd-muted-foreground">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

/* -------------------------------------------------------------------------- */

export interface LegendItem {
  id: string;
  label?: string;
  color: string;
  /** `line` for series drawn as strokes, `dot` for point clouds and nodes. */
  shape?: 'line' | 'dot' | 'square';
}

/**
 * Identity never rides on color alone: the swatch carries the hue, the label
 * stays in a text token so it is legible at any lightness.
 */
export function ChartLegend({ items, className }: { items: readonly LegendItem[]; className?: string }) {
  if (items.length === 0) return null;
  return (
    <ul className={cx('flex flex-wrap items-center gap-x-4 gap-y-1.5', className)}>
      {items.map((item) => (
        <li
          key={item.id}
          className="flex items-center gap-1.5 font-ui text-[0.72rem] text-fd-muted-foreground"
        >
          <LegendSwatch color={item.color} shape={item.shape ?? 'line'} />
          {item.label ?? item.id}
        </li>
      ))}
    </ul>
  );
}

function LegendSwatch({ color, shape }: { color: string; shape: NonNullable<LegendItem['shape']> }) {
  const style: CSSProperties = { background: color };
  if (shape === 'dot') {
    return <span aria-hidden className="size-2 shrink-0 rounded-full" style={style} />;
  }
  if (shape === 'square') {
    return <span aria-hidden className="size-2.5 shrink-0 rounded-[2px]" style={style} />;
  }
  return <span aria-hidden className="h-0.5 w-3.5 shrink-0 rounded-full" style={style} />;
}

/* -------------------------------------------------------------------------- */

export interface TooltipRow {
  label: string;
  value: string;
  color?: string;
}

/** The one tooltip surface every chart uses. */
export function TooltipCard({ title, rows }: { title?: ReactNode; rows: readonly TooltipRow[] }) {
  return (
    <div className="rounded-md border border-fd-border bg-fd-popover px-2.5 py-1.5 shadow-lg">
      {title ? <div className="eyebrow mb-1 text-[0.62rem]">{title}</div> : null}
      <table className="border-separate border-spacing-0">
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              {row.color ? (
                <td className="pr-1.5 align-middle">
                  <span
                    aria-hidden
                    className="block size-2 rounded-full"
                    style={{ background: row.color }}
                  />
                </td>
              ) : null}
              <td className="pr-3 font-ui text-[0.72rem] whitespace-nowrap text-fd-muted-foreground">
                {row.label}
              </td>
              <td className="font-mono text-[0.75rem] tabular-nums text-fd-foreground">{row.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export interface ScaleLegendProps {
  /** Left-to-right color stops of the ramp being explained. */
  stops: readonly string[];
  min: string;
  max: string;
  mid?: string;
  label?: string;
}

/** Continuous color key for heatmaps — a gradient rail with its end values. */
export function ScaleLegend({ stops, min, max, mid, label }: ScaleLegendProps) {
  return (
    <div className="flex flex-col gap-1">
      {label ? <span className="eyebrow text-[0.62rem]">{label}</span> : null}
      <div
        aria-hidden
        className="h-1.5 w-full max-w-64 rounded-full"
        style={{ backgroundImage: `linear-gradient(to right, ${stops.join(', ')})` }}
      />
      <div className="flex w-full max-w-64 justify-between font-mono text-[0.68rem] tabular-nums text-fd-muted-foreground">
        <span>{min}</span>
        {mid ? <span>{mid}</span> : null}
        <span>{max}</span>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export interface ChartDataTableProps {
  caption: string;
  columns: readonly string[];
  rows: readonly (readonly (string | number)[])[];
}

/**
 * The WCAG-clean twin of a chart. Visually hidden — sighted readers have the
 * plot and its tooltips — but it keeps every value reachable, so no number in
 * the book is gated behind hovering a mark.
 */
export function ChartDataTable({ caption, columns, rows }: ChartDataTableProps) {
  return (
    <table className="sr-only">
      <caption>{caption}</caption>
      <thead>
        <tr>
          {columns.map((column) => (
            <th key={column} scope="col">
              {column}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {row.map((cell, cellIndex) =>
              cellIndex === 0 ? (
                <th key={cellIndex} scope="row">
                  {cell}
                </th>
              ) : (
                <td key={cellIndex}>{cell}</td>
              ),
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
