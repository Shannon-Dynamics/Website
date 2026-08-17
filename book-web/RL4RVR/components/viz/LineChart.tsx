'use client';

import { ResponsiveLine } from '@nivo/line';
import { useTheme } from '@/components/layout/ThemeProvider';
import { nivoTheme, seriesColor } from '@/lib/theme';
import { ChartFrame, Legend, type TableView } from './ChartFrame';

/** The book's series shape — numeric x/y throughout. */
export interface LineSerie {
  id: string;
  data: Array<{ x: number; y: number }>;
}

export interface LineChartProps {
  data: LineSerie[];
  title?: string;
  subtitle?: string;
  caption?: string;
  xLegend?: string;
  yLegend?: string;
  height?: number;
  /** Fix the y-domain; defaults to auto. */
  yMin?: number | 'auto';
  yMax?: number | 'auto';
  controls?: React.ReactNode;
  table?: TableView;
  /** Show point markers (≥8px) — off for dense curves. */
  showPoints?: boolean;
  /** Optional per-series dash pattern, keyed by series id. */
  dashed?: string[];
  legendOverride?: Array<{ label: string; color: string; dashed?: boolean }>;
}

/**
 * The book's standard time-series chart: learning curves, convergence traces,
 * gradient-variance histories.
 *
 * Follows the dataviz mark spec — 2px lines, recessive grid, crosshair +
 * tooltip on by default, one y-axis only (never a dual-axis chart: two
 * measures of different scale become two charts).
 */
export function LineChart({
  data,
  title,
  subtitle,
  caption,
  xLegend,
  yLegend,
  height = 300,
  yMin = 'auto',
  yMax = 'auto',
  controls,
  table,
  showPoints = false,
  dashed = [],
  legendOverride,
}: LineChartProps) {
  const { mode } = useTheme();
  // A series with no points makes the line generator return null, which the
  // browser then rejects as an invalid `d` attribute. Drop empties here so no
  // caller has to remember to.
  const seriesWithData = data.filter((s) => s.data && s.data.length > 0);
  const colors = seriesWithData.map((_, i) => seriesColor(i, mode));

  const legendItems =
    legendOverride ??
    seriesWithData.map((s, i) => ({
      label: String(s.id),
      color: colors[i],
      dashed: dashed.includes(String(s.id)),
    }));

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      height={height}
      controls={controls}
      table={table}
      legend={<Legend items={legendItems} />}
    >
      <ResponsiveLine
        data={seriesWithData}
        theme={nivoTheme(mode)}
        colors={colors}
        margin={{ top: 14, right: 22, bottom: xLegend ? 46 : 32, left: yLegend ? 58 : 46 }}
        xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
        yScale={{ type: 'linear', min: yMin, max: yMax, stacked: false }}
        curve="monotoneX"
        lineWidth={2}
        enablePoints={showPoints}
        pointSize={8}
        pointBorderWidth={2}
        pointBorderColor={{ from: 'seriesColor' }}
        pointColor="var(--surface-1)"
        enableGridX={false}
        enableGridY
        axisBottom={{
          tickSize: 0,
          tickPadding: 8,
          legend: xLegend,
          legendOffset: 36,
          legendPosition: 'middle',
        }}
        axisLeft={{
          tickSize: 0,
          tickPadding: 8,
          legend: yLegend,
          legendOffset: -46,
          legendPosition: 'middle',
        }}
        enableSlices="x"
        crosshairType="x"
        sliceTooltip={({ slice }) => (
          <div className="rounded-lg border border-hairline bg-surface-raised px-2.5 py-2 text-[11.5px] shadow-lg">
            <div className="tabular mb-1 font-semibold text-ink">
              {xLegend ?? 'x'} = {Number(slice.points[0]?.data.x).toFixed(0)}
            </div>
            {slice.points.map((p) => (
              <div key={p.id} className="flex items-center gap-1.5 text-ink-secondary">
                <span
                  aria-hidden
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: p.seriesColor }}
                />
                <span>{String(p.seriesId)}</span>
                <span className="tabular ml-auto pl-3 font-medium text-ink">
                  {Number(p.data.y).toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
        useMesh
        animate={false}
      />
    </ChartFrame>
  );
}
