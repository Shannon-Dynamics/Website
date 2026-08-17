'use client';

import { useCallback, useMemo } from 'react';
import type { ReactNode } from 'react';
import { ResponsiveScatterPlot } from '@nivo/scatterplot';
import type { ScatterPlotTooltipProps } from '@nivo/scatterplot';

import {
  formatNumber,
  useBookColors,
  useChartAnimation,
  useChartTheme,
  withAlpha,
  type BookRole,
} from '@/lib/chart-theme';
import { ChartDataTable, ChartFrame, ChartLegend, TooltipCard, resolveSeriesColors } from './chart-frame';

interface Point {
  x: number;
  y: number;
}

export interface ScatterSeries {
  id: string;
  /** Takes the book's reserved color for that meaning. */
  role?: BookRole;
  data: readonly Point[];
}

export interface ScatterChartProps {
  series: readonly ScatterSeries[];
  xLabel: string;
  yLabel: string;
  height?: number;
  /** Colors for series that declare no `role`, in series order. */
  colors?: readonly string[];
  legend?: boolean;
  /** Marker diameter in pixels. Particle clouds want small, samples want large. */
  nodeSize?: number;
  /** Marker alpha. Defaults to thinning out as the cloud gets denser. */
  opacity?: number;
  /** Points per series kept for drawing; above this the series is thinned. */
  maxPoints?: number;
  xDomain?: [number, number];
  yDomain?: [number, number];
  ariaLabel?: string;
  caption?: ReactNode;
  margin?: Partial<{ top: number; right: number; bottom: number; left: number }>;
}

const DEFAULT_MARGIN = { top: 14, right: 22, bottom: 46, left: 58 };

/** Past this many marks a voronoi mesh costs more than the hover is worth. */
const INTERACTION_LIMIT = 900;
const TABLE_LIMIT = 120;

/**
 * Evenly strided subsample. Particle order carries no meaning, so a stride is
 * as faithful as a random draw and it keeps the figure identical on every
 * render — including the one that happens at build time.
 */
function thin<T>(points: readonly T[], limit: number): T[] {
  if (points.length <= limit) return [...points];
  const stride = points.length / limit;
  const kept: T[] = [];
  for (let index = 0; index < limit; index += 1) kept.push(points[Math.floor(index * stride)]);
  return kept;
}

export function ScatterChart({
  series,
  xLabel,
  yLabel,
  height = 320,
  colors,
  legend,
  nodeSize = 6,
  opacity,
  maxPoints = 1500,
  xDomain,
  yDomain,
  ariaLabel,
  caption,
  margin,
}: ScatterChartProps) {
  const theme = useChartTheme();
  const bookColors = useBookColors();
  const animate = useChartAnimation();

  const colorById = useMemo(
    () => resolveSeriesColors(series, bookColors, colors),
    [series, bookColors, colors],
  );

  const data = useMemo(
    () => series.map((serie) => ({ id: serie.id, data: thin(serie.data, maxPoints) })),
    [series, maxPoints],
  );

  const drawnPoints = data.reduce((total, serie) => total + serie.data.length, 0);
  // A dense cloud reads as a density only if the marks can overlap.
  const markAlpha = opacity ?? (drawnPoints > 400 ? 0.45 : drawnPoints > 120 ? 0.7 : 0.95);

  const colorOf = useCallback(
    ({ serieId }: { serieId: string | number }) =>
      withAlpha(colorById.get(String(serieId)) ?? bookColors.truth, markAlpha),
    [colorById, bookColors.truth, markAlpha],
  );

  const Tooltip = useMemo(
    () =>
      function ScatterTooltip({ node }: ScatterPlotTooltipProps<Point>) {
        return (
          <TooltipCard
            title={String(node.serieId)}
            rows={[
              { color: node.color, label: xLabel, value: formatNumber(node.data.x) },
              { label: yLabel, value: formatNumber(node.data.y) },
            ]}
          />
        );
      },
    [xLabel, yLabel],
  );

  const totalPoints = series.reduce((total, serie) => total + serie.data.length, 0);
  const showLegend = legend ?? series.length > 1;
  const interactive = drawnPoints <= INTERACTION_LIMIT;

  const description =
    ariaLabel ??
    `Scatter plot of ${yLabel} against ${xLabel}, ${totalPoints} points` +
      (series.length > 1 ? ` across ${series.length} series: ${series.map((s) => s.id).join(', ')}.` : '.');

  return (
    <ChartFrame
      height={height}
      caption={
        caption ??
        (drawnPoints < totalPoints
          ? `Showing ${drawnPoints.toLocaleString()} of ${totalPoints.toLocaleString()} samples.`
          : undefined)
      }
      table={
        totalPoints > 0 && totalPoints <= TABLE_LIMIT ? (
          <ChartDataTable
            caption={description}
            columns={['Series', xLabel, yLabel]}
            rows={series.flatMap((serie) =>
              serie.data.map((point) => [serie.id, formatNumber(point.x), formatNumber(point.y)]),
            )}
          />
        ) : null
      }
      legend={
        showLegend ? (
          <ChartLegend
            items={series.map((serie) => ({
              id: serie.id,
              color: colorById.get(serie.id) ?? bookColors.truth,
              shape: 'dot',
            }))}
          />
        ) : null
      }
    >
      <ResponsiveScatterPlot<Point>
        defaultWidth={720}
        defaultHeight={height}
        data={data}
        theme={theme}
        colors={colorOf}
        margin={{ ...DEFAULT_MARGIN, ...margin }}
        xScale={
          xDomain
            ? { type: 'linear', min: xDomain[0], max: xDomain[1] }
            : { type: 'linear', min: 'auto', max: 'auto' }
        }
        yScale={
          yDomain
            ? { type: 'linear', min: yDomain[0], max: yDomain[1] }
            : { type: 'linear', min: 'auto', max: 'auto' }
        }
        nodeSize={nodeSize}
        // A cloud is read in two dimensions, so both grids earn their ink.
        enableGridX
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
        isInteractive={interactive}
        useMesh={interactive}
        tooltip={Tooltip}
        animate={animate}
        motionConfig="stiff"
        role="img"
        ariaLabel={description}
      />
    </ChartFrame>
  );
}
