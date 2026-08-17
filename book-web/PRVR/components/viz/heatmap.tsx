'use client';

import { useCallback, useMemo } from 'react';
import type { ReactNode } from 'react';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import type { ComputedCell, TooltipProps } from '@nivo/heatmap';

import {
  adjustLightness,
  formatNumber,
  mixColors,
  readableInk,
  relativeLuminance,
  useChartAnimation,
  useChartTheme,
  useChartTokens,
} from '@/lib/chart-theme';
import { ChartDataTable, ChartFrame, ScaleLegend, TooltipCard } from './chart-frame';

interface Cell {
  x: string;
  y: number | null;
}

export interface HeatMapProps {
  /** Row-major values; every row must have the same length. */
  matrix: readonly (readonly number[])[];
  rowLabels?: readonly string[];
  colLabels?: readonly string[];
  /**
   * Signed data (correlations, innovations) gets two opposed hues around a
   * neutral zero; unsigned magnitude gets one hue, light to dark.
   */
  diverging?: boolean;
  height?: number;
  /** Names the quantity in the tooltip and the color key. */
  valueLabel?: string;
  precision?: number;
  showValues?: boolean;
  square?: boolean;
  ariaLabel?: string;
  caption?: ReactNode;
}

/** Fraction of the ramp spent reaching the base hue; the rest deepens it. */
const HUE_STOP = 0.7;
/** Cell budgets past which labels crowd and the hidden table stops helping. */
const LABEL_LIMIT = 81;
const TABLE_LIMIT = 400;

export function HeatMap({
  matrix,
  rowLabels,
  colLabels,
  diverging = false,
  height = 320,
  valueLabel = 'value',
  precision,
  showValues,
  square,
  ariaLabel,
  caption,
}: HeatMapProps) {
  const theme = useChartTheme();
  const tokens = useChartTokens();
  const animate = useChartAnimation();

  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;

  const rowNames = useMemo(
    () => Array.from({ length: rows }, (_, index) => rowLabels?.[index] ?? String(index)),
    [rows, rowLabels],
  );
  const colNames = useMemo(
    () => Array.from({ length: cols }, (_, index) => colLabels?.[index] ?? String(index)),
    [cols, colLabels],
  );

  const { min, max } = useMemo(() => {
    let low = Number.POSITIVE_INFINITY;
    let high = Number.NEGATIVE_INFINITY;
    for (const row of matrix) {
      for (const value of row) {
        if (!Number.isFinite(value)) continue;
        if (value < low) low = value;
        if (value > high) high = value;
      }
    }
    return Number.isFinite(low) ? { min: low, max: high } : { min: 0, max: 1 };
  }, [matrix]);

  /**
   * The ramp is built from the book's own hues rather than a d3 scheme, and
   * mixed in Oklab so the middle of the scale carries the value it should.
   * Both ends are anchored on the *surface*, so the same code produces a
   * light-to-dark ramp on paper white and a dark-to-light one at night.
   */
  const ramp = useMemo(() => {
    const surface = tokens.surface;
    const surfaceLuminance = relativeLuminance(surface);
    const beyond = (hue: string) =>
      adjustLightness(hue, surfaceLuminance > relativeLuminance(hue) ? -0.16 : 0.16);

    const towards = (from: string, hue: string, t: number) =>
      t <= HUE_STOP
        ? mixColors(from, hue, t / HUE_STOP)
        : mixColors(hue, beyond(hue), (t - HUE_STOP) / (1 - HUE_STOP));

    if (diverging) {
      const zero = mixColors(surface, tokens.truth, 0.16);
      const extent = Math.max(Math.abs(min), Math.abs(max)) || 1;
      return (value: number) => {
        const t = Math.max(-1, Math.min(1, value / extent));
        // Warm for positive, cool for negative, neutral at zero — the midpoint
        // must read as "nothing", so it is the only gray on the scale.
        return towards(zero, t >= 0 ? tokens.prediction : tokens.prior, Math.abs(t));
      };
    }

    const floor = mixColors(surface, tokens.prior, 0.05);
    const span = max - min || 1;
    return (value: number) => towards(floor, tokens.prior, Math.max(0, Math.min(1, (value - min) / span)));
  }, [tokens, diverging, min, max]);

  // Inferred, not annotated: nivo's `HeatMapSerie` carries an extra-props
  // parameter that defaults to `Record<string, never>` and rejects `{ id, data }`.
  const data = useMemo(
    () =>
      rowNames.map((name, rowIndex) => ({
        id: name,
        data: colNames.map<Cell>((column, colIndex) => ({
          x: column,
          y: matrix[rowIndex]?.[colIndex] ?? null,
        })),
      })),
    [rowNames, colNames, matrix],
  );

  const colorOf = useCallback(
    (cell: Pick<ComputedCell<Cell>, 'value'>) =>
      cell.value === null ? mixColors(tokens.surface, tokens.truth, 0.08) : ramp(cell.value),
    [ramp, tokens],
  );

  const Tooltip = useMemo(
    () =>
      function HeatMapTooltip({ cell }: TooltipProps<Cell>) {
        return (
          <TooltipCard
            title={`${cell.serieId} · ${cell.data.x}`}
            rows={[{ color: cell.color, label: valueLabel, value: cell.formattedValue ?? '—' }]}
          />
        );
      },
    [valueLabel],
  );

  const legendStops = useMemo(() => {
    const extent = Math.max(Math.abs(min), Math.abs(max)) || 1;
    return Array.from({ length: 9 }, (_, index) => {
      const t = index / 8;
      return ramp(diverging ? (t * 2 - 1) * extent : min + t * (max - min));
    });
  }, [ramp, diverging, min, max]);

  const cells = rows * cols;
  const labelled = showValues ?? cells <= LABEL_LIMIT;
  const format = useCallback((value: number) => formatNumber(value, precision), [precision]);

  // Row labels sit outside the plot, so the left gutter has to fit the longest.
  const longestRowName = rowNames.reduce((longest, name) => Math.max(longest, name.length), 0);
  const marginLeft = Math.min(132, 20 + longestRowName * 7);

  const description =
    ariaLabel ??
    `Heat map of a ${rows} by ${cols} matrix of ${valueLabel}, ranging from ${format(min)} to ${format(max)}.`;

  return (
    <ChartFrame
      height={height}
      caption={caption}
      legend={
        <ScaleLegend
          label={valueLabel}
          stops={legendStops}
          min={format(diverging ? -Math.max(Math.abs(min), Math.abs(max)) : min)}
          mid={diverging ? '0' : undefined}
          max={format(diverging ? Math.max(Math.abs(min), Math.abs(max)) : max)}
        />
      }
      table={
        cells > 0 && cells <= TABLE_LIMIT ? (
          <ChartDataTable
            caption={description}
            columns={['', ...colNames]}
            rows={rowNames.map((name, rowIndex) => [
              name,
              ...colNames.map((_, colIndex) => {
                const value = matrix[rowIndex]?.[colIndex];
                return value === undefined ? '—' : format(value);
              }),
            ])}
          />
        ) : null
      }
    >
      {/* The second parameter is the serie's extra props; nivo defaults it to
          `Record<string, never>`, which rejects a plain `{ id, data }` serie. */}
      <ResponsiveHeatMap<Cell, Record<never, never>>
        defaultWidth={720}
        defaultHeight={height}
        data={data}
        theme={theme}
        colors={colorOf}
        emptyColor={mixColors(tokens.surface, tokens.truth, 0.08)}
        margin={{ top: 34, right: 16, bottom: 16, left: marginLeft }}
        forceSquare={square ?? rows === cols}
        // A 2px ring of the surface color is what separates neighbouring cells.
        borderWidth={2}
        borderColor={tokens.surface}
        borderRadius={2}
        enableLabels={labelled}
        valueFormat={format}
        labelTextColor={(cell) => readableInk(cell.color, tokens.canvasInk, tokens.canvasBg)}
        axisTop={{ tickSize: 0, tickPadding: 8 }}
        axisRight={null}
        axisBottom={null}
        axisLeft={{ tickSize: 0, tickPadding: 8 }}
        hoverTarget="rowColumn"
        activeOpacity={1}
        inactiveOpacity={0.45}
        tooltip={Tooltip}
        animate={animate}
        motionConfig="stiff"
        role="img"
        ariaLabel={description}
      />
    </ChartFrame>
  );
}
