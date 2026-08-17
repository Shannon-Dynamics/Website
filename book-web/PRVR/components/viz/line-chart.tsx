'use client';

import { useCallback, useMemo } from 'react';
import type { ReactNode } from 'react';
import { ResponsiveLine } from '@nivo/line';
import type { PointColorContext, SliceTooltipProps } from '@nivo/line';
import type { CartesianMarkerProps, LineCurveFactoryId } from '@nivo/core';

import {
  formatNumber,
  useBookColors,
  useChartAnimation,
  useChartTheme,
  useChartTokens,
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

export interface LineChartSeries {
  id: string;
  /** Takes the book's reserved color for that meaning. */
  role?: BookRole;
  data: readonly { x: number; y: number }[];
}

/** A reference line — a landmark, a true state, a decision threshold. */
export interface ChartMarker {
  axis: 'x' | 'y';
  value: number;
  label?: string;
  role?: BookRole;
}

export interface LineChartProps {
  series: readonly LineChartSeries[];
  xLabel: string;
  yLabel: string;
  height?: number;
  /** Colors for series that declare no `role`, in series order. */
  colors?: readonly string[];
  /** Defaults to on for two or more series, off for one (the title names it). */
  legend?: boolean;
  curve?: LineCurveFactoryId;
  enableArea?: boolean;
  markers?: readonly ChartMarker[];
  yMin?: number | 'auto';
  yMax?: number | 'auto';
  /** Overrides the generated one-sentence description. */
  ariaLabel?: string;
  caption?: ReactNode;
  margin?: Partial<{ top: number; right: number; bottom: number; left: number }>;
}

const DEFAULT_MARGIN = { top: 14, right: 22, bottom: 46, left: 58 };

/** Above this many points per series, dots crowd the line and are dropped. */
const POINT_LIMIT = 24;
/** Above this many plotted values the hidden table stops being useful. */
const TABLE_LIMIT = 240;

export function LineChart({
  series,
  xLabel,
  yLabel,
  height = 280,
  colors,
  legend,
  curve = 'monotoneX',
  enableArea = false,
  markers,
  yMin = 'auto',
  yMax = 'auto',
  ariaLabel,
  caption,
  margin,
}: LineChartProps) {
  const theme = useChartTheme();
  const tokens = useChartTokens();
  const bookColors = useBookColors();
  const animate = useChartAnimation();

  const colorById = useMemo(
    () => resolveSeriesColors(series, bookColors, colors),
    [series, bookColors, colors],
  );
  const colorOf = useCallback(
    (serie: LineChartSeries) => colorById.get(serie.id) ?? bookColors.truth,
    [colorById, bookColors.truth],
  );
  const pointColor = useCallback((context: PointColorContext<LineChartSeries>) => context.series.color, []);

  const pointCount = series.reduce((total, serie) => total + serie.data.length, 0);
  const longestSeries = series.reduce((longest, serie) => Math.max(longest, serie.data.length), 0);
  const showPoints = longestSeries > 0 && longestSeries <= POINT_LIMIT;
  const showLegend = legend ?? series.length > 1;

  const nivoMarkers = useMemo<CartesianMarkerProps[] | undefined>(() => {
    if (!markers?.length) return undefined;
    return markers.map((marker) => ({
      axis: marker.axis,
      value: marker.value,
      legend: marker.label,
      legendPosition: marker.axis === 'x' ? 'top' : 'top-right',
      legendOrientation: 'horizontal',
      lineStyle: {
        stroke: marker.role ? bookColors[marker.role] : tokens.mutedForeground,
        strokeWidth: 1.5,
        strokeOpacity: 0.7,
      },
      textStyle: {
        fill: marker.role ? bookColors[marker.role] : tokens.mutedForeground,
        fontFamily: tokens.fontUi,
        fontSize: 10,
      },
    }));
  }, [markers, bookColors, tokens]);

  const SliceTooltip = useMemo(
    () =>
      function LineSliceTooltip({ slice }: SliceTooltipProps<LineChartSeries>) {
        const first = slice.points[0];
        return (
          <TooltipCard
            title={first ? `${xLabel} ${formatNumber(first.data.x)}` : undefined}
            rows={slice.points.map((point) => ({
              color: point.seriesColor,
              label: String(point.seriesId),
              value: formatNumber(point.data.y),
            }))}
          />
        );
      },
    [xLabel],
  );

  const table = useMemo(() => {
    if (pointCount === 0 || pointCount > TABLE_LIMIT) return null;
    const { indices, valueAt } = pivotSeries(series, { sort: true });
    return (
      <ChartDataTable
        caption={`${yLabel} by ${xLabel}`}
        columns={[xLabel, ...series.map((serie) => serie.id)]}
        rows={indices.map((index) => [
          formatNumber(index),
          ...series.map((serie) => {
            const value = valueAt(index, serie.id);
            return value === undefined ? '—' : formatNumber(value);
          }),
        ])}
      />
    );
  }, [series, pointCount, xLabel, yLabel]);

  const description =
    ariaLabel ??
    `Line chart of ${yLabel} against ${xLabel}` +
      (series.length > 1 ? `, ${series.length} series: ${series.map((s) => s.id).join(', ')}.` : '.');

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
              color: colorOf(serie),
              shape: 'line',
            }))}
          />
        ) : null
      }
    >
      <ResponsiveLine<LineChartSeries>
        // Explicit defaults are what make this render real SVG during export;
        // without them the responsive wrapper measures 0 and renders nothing.
        defaultWidth={720}
        defaultHeight={height}
        data={series}
        theme={theme}
        colors={colorOf}
        margin={{ ...DEFAULT_MARGIN, ...margin }}
        xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
        yScale={{ type: 'linear', min: yMin, max: yMax, nice: true, stacked: false }}
        curve={curve}
        lineWidth={2}
        enableArea={enableArea}
        areaOpacity={0.1}
        enablePoints={showPoints}
        pointSize={8}
        pointColor={pointColor}
        pointBorderWidth={2}
        pointBorderColor={tokens.surface}
        enableGridX={false}
        enableGridY
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 0,
          tickPadding: 8,
          legend: xLabel,
          legendPosition: 'middle',
          legendOffset: 36,
        }}
        axisLeft={{
          tickSize: 0,
          tickPadding: 8,
          legend: yLabel,
          legendPosition: 'middle',
          legendOffset: -46,
        }}
        markers={nivoMarkers}
        enableSlices="x"
        enableCrosshair
        crosshairType="x"
        sliceTooltip={SliceTooltip}
        animate={animate}
        motionConfig="stiff"
        isFocusable={false}
        role="img"
        ariaLabel={description}
      />
    </ChartFrame>
  );
}
