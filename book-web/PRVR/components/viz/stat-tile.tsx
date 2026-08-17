import { curveMonotoneX, line as d3Line } from 'd3-shape';
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';

import { ROLE_VAR, formatNumber, type BookRole } from '@/lib/chart-theme';
import { cx } from './chart-frame';

export interface StatTileProps {
  label: string;
  value: number | string;
  unit?: string;
  /** Tints the tile's rule with the book color for that meaning. */
  role?: BookRole;
  /** Signed change since the previous step. Direction reads from the icon. */
  trend?: number;
  /** Names what the trend is measured against, e.g. "since resample". */
  trendLabel?: string;
  /** Recent history, oldest first. Drawn as a hairline sparkline. */
  sparkline?: readonly number[];
  /** Fixed decimals for numeric values; omit to let the value pick. */
  precision?: number;
  className?: string;
}

const SPARK_WIDTH = 76;
const SPARK_HEIGHT = 22;
const SPARK_PAD = 2.5;

/**
 * The atom of the book's dashboards: one labelled number, optionally with the
 * direction it is moving and the shape of how it got there.
 *
 * Values are tabular-figured on purpose. These readouts are wired to running
 * simulations, and proportional digits make a live number jitter sideways as it
 * changes — the one place where equal-width digits beat proportional ones.
 */
export function StatTile({
  label,
  value,
  unit,
  role,
  trend,
  trendLabel,
  sparkline,
  precision,
  className,
}: StatTileProps) {
  const display = typeof value === 'number' ? formatNumber(value, precision) : value;
  const accent = role ? ROLE_VAR[role] : undefined;
  const spark = buildSparkline(sparkline);

  return (
    <div
      className={cx(
        'not-prose flex flex-col gap-1.5 rounded-md border border-fd-border border-l-2 bg-fd-card px-3 py-2.5',
        className,
      )}
      style={accent ? { borderLeftColor: accent } : undefined}
    >
      <span className="eyebrow">{label}</span>

      <p className="m-0 flex items-baseline gap-1.5">
        <span className="font-mono text-2xl leading-none font-medium tabular-nums text-fd-foreground">
          {display}
        </span>
        {unit ? <span className="font-ui text-xs text-fd-muted-foreground">{unit}</span> : null}
      </p>

      {trend !== undefined || spark ? (
        <div className="flex items-end justify-between gap-2">
          {trend !== undefined ? <Trend value={trend} label={trendLabel} precision={precision} /> : <span />}
          {spark ? (
            <svg
              width={SPARK_WIDTH}
              height={SPARK_HEIGHT}
              viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
              aria-hidden
              className="shrink-0 overflow-visible"
            >
              <path
                d={spark.path}
                fill="none"
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ stroke: accent ?? 'var(--color-fd-muted-foreground)', opacity: 0.75 }}
              />
              <circle
                cx={spark.lastX}
                cy={spark.lastY}
                r={2}
                style={{ fill: accent ?? 'var(--color-fd-foreground)' }}
              />
            </svg>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/**
 * Direction is carried by the glyph, not by color: in estimation "up" is
 * neither good nor bad, and the five data hues are spoken for.
 */
function Trend({ value, label, precision }: { value: number; label?: string; precision?: number }) {
  const Icon = value > 0 ? ArrowUpRight : value < 0 ? ArrowDownRight : Minus;
  const word = value > 0 ? 'up' : value < 0 ? 'down' : 'unchanged';
  const magnitude = formatNumber(Math.abs(value), precision);

  return (
    <span className="inline-flex items-center gap-0.5 font-mono text-[0.7rem] tabular-nums text-fd-muted-foreground">
      <Icon className="size-3 shrink-0" aria-hidden />
      <span className="sr-only">{word} </span>
      {magnitude}
      {label ? <span className="ml-1 font-ui not-italic">{label}</span> : null}
    </span>
  );
}

function buildSparkline(values: readonly number[] | undefined) {
  if (!values || values.length < 2) return null;

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min)) return null;

  const span = max - min || 1;
  const toX = (index: number) =>
    SPARK_PAD + (index / (values.length - 1)) * (SPARK_WIDTH - 2 * SPARK_PAD);
  const toY = (value: number) =>
    SPARK_HEIGHT - SPARK_PAD - ((value - min) / span) * (SPARK_HEIGHT - 2 * SPARK_PAD);

  const path = d3Line<number>()
    .x((_, index) => toX(index))
    .y((value) => toY(value))
    .curve(curveMonotoneX)(values);

  if (!path) return null;

  return {
    path,
    lastX: toX(values.length - 1),
    lastY: toY(values[values.length - 1]),
  };
}
