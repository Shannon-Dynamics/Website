'use client';

import { useCallback, useMemo } from 'react';
import type { ReactNode } from 'react';
import { ResponsiveBar } from '@nivo/bar';
import type { BarDatum, BarTooltipProps, ComputedDatum } from '@nivo/bar';

import {
  formatNumber,
  useBookColors,
  useChartAnimation,
  useChartTheme,
  type BookRole,
} from '@/lib/chart-theme';
import {
  ChartDataTable,
  ChartFrame,
  ChartLegend,
  TooltipCard,
  pivotSeries,
  resolveSeriesColors,
} from './chart-frame';

export interface BarChartSeries {
  id: string;
  /** Takes the book's reserved color for that meaning. */
  role?: BookRole;
  /** One entry per bin; `x` is the bin, `y` its mass. */
  data: readonly { x: string | number; y: number }[];
}

export interface BarChartProps {
  series: readonly BarChartSeries[];
  xLabel: string;
  yLabel: string;
  height?: number;
  /** Colors for series that declare no `role`, in series order. */
  colors?: readonly string[];
  legend?: boolean;
  layout?: 'vertical' | 'horizontal';
  /** Beliefs are compared, not summed — grouped is the default for a reason. */
  groupMode?: 'grouped' | 'stacked';
  maxValue?: number | 'auto';
  valueFormat?: (value: number) => string;
  ariaLabel?: string;
  caption?: ReactNode;
  margin?: Partial<{ top: number; right: number; bottom: number; left: number }>;
}

/** Index column of the pivoted rows; `__` keeps it clear of any series id. */
const INDEX_KEY = '__bin';

const DEFAULT_MARGIN = { top: 14, right: 22, bottom: 46, left: 58 };

/** Roughly the number of x labels that fit before they collide. */
const MAX_TICKS = 12;
const TABLE_LIMIT = 240;

export function BarChart({
  series,
  xLabel,
  yLabel,
  height = 280,
  colors,
  legend,
  layout = 'vertical',
  groupMode = 'grouped',
  maxValue = 'auto',
  valueFormat = formatNumber,
  ariaLabel,
  caption,
  margin,
}: BarChartProps) {
  const theme = useChartTheme();
  const bookColors = useBookColors();
  const animate = useChartAnimation();

  const colorById = useMemo(
    () => resolveSeriesColors(series, bookColors, colors),
    [series, bookColors, colors],
  );

  const { indices, valueAt } = useMemo(() => pivotSeries(series), [series]);

  const rows = useMemo<BarDatum[]>(
    () =>
      indices.map((index) => {
        const row: BarDatum = { [INDEX_KEY]: String(index) };
        for (const serie of series) {
          const value = valueAt(index, serie.id);
          if (value !== undefined) row[serie.id] = value;
        }
        return row;
      }),
    [indices, series, valueAt],
  );

  const colorOf = useCallback(
    (datum: ComputedDatum<BarDatum>) => colorById.get(String(datum.id)) ?? bookColors.truth,
    [colorById, bookColors.truth],
  );

  // Thin out labels rather than rotate them: a hallway with 40 cells still
  // needs a readable axis.
  const tickValues = useMemo(() => {
    if (indices.length <= MAX_TICKS) return undefined;
    const stride = Math.ceil(indices.length / MAX_TICKS);
    return indices.filter((_, index) => index % stride === 0).map(String);
  }, [indices]);

  const Tooltip = useMemo(
    () =>
      function BarTooltip({ id, value, indexValue, color }: BarTooltipProps<BarDatum>) {
        return (
          <TooltipCard
            title={`${xLabel} ${indexValue}`}
            rows={[{ color, label: String(id), value: valueFormat(value) }]}
          />
        );
      },
    [xLabel, valueFormat],
  );

  const table = useMemo(() => {
    const cells = indices.length * series.length;
    if (cells === 0 || cells > TABLE_LIMIT) return null;
    return (
      <ChartDataTable
        caption={`${yLabel} by ${xLabel}`}
        columns={[xLabel, ...series.map((serie) => serie.id)]}
        rows={indices.map((index) => [
          String(index),
          ...series.map((serie) => {
            const value = valueAt(index, serie.id);
            return value === undefined ? '—' : valueFormat(value);
          }),
        ])}
      />
    );
  }, [indices, series, valueAt, xLabel, yLabel, valueFormat]);

  const showLegend = legend ?? series.length > 1;
  const isVertical = layout === 'vertical';
  const description =
    ariaLabel ??
    `Bar chart of ${yLabel} across ${indices.length} ${xLabel} bins` +
      (series.length > 1 ? `, ${series.length} series: ${series.map((s) => s.id).join(', ')}.` : '.');

  const valueAxis = {
    tickSize: 0,
    tickPadding: 8,
    legend: yLabel,
    legendPosition: 'middle' as const,
    legendOffset: isVertical ? -46 : 36,
  };
  const indexAxis = {
    tickSize: 0,
    tickPadding: 8,
    tickValues,
    legend: xLabel,
    legendPosition: 'middle' as const,
    legendOffset: isVertical ? 36 : -46,
  };

  return (
    <ChartFrame
      height={height}
      caption={caption}
      table={table}
      legend={
        showLegend ? (
          <ChartLegend
            items={series.map((serie) => ({
              id: serie.id,
              color: colorById.get(serie.id) ?? bookColors.truth,
              shape: 'square',
            }))}
          />
        ) : null
      }
    >
      <ResponsiveBar<BarDatum>
        defaultWidth={720}
        defaultHeight={height}
        data={rows}
        keys={series.map((serie) => serie.id)}
        indexBy={INDEX_KEY}
        layout={layout}
        groupMode={groupMode}
        theme={theme}
        colors={colorOf}
        margin={{ ...DEFAULT_MARGIN, ...margin }}
        padding={0.3}
        // 2px of surface between neighbours does the separating, not a stroke.
        innerPadding={2}
        borderRadius={3}
        borderWidth={0}
        // Bars must grow from zero; a clipped baseline misstates every ratio.
        valueScale={{ type: 'linear', min: 0, max: maxValue }}
        indexScale={{ type: 'band', round: true }}
        enableGridX={!isVertical}
        enableGridY={isVertical}
        enableLabel={false}
        enableTotals={false}
        axisTop={null}
        axisRight={null}
        axisBottom={isVertical ? indexAxis : valueAxis}
        axisLeft={isVertical ? valueAxis : indexAxis}
        tooltip={Tooltip}
        animate={animate}
        motionConfig="stiff"
        isFocusable={false}
        role="img"
        ariaLabel={description}
      />
    </ChartFrame>
  );
}
