'use client';

import { ResponsiveBar } from '@nivo/bar';
import { useTheme } from '@/components/layout/ThemeProvider';
import { nivoTheme, seriesColor } from '@/lib/theme';
import { ChartFrame, type TableView } from './ChartFrame';

/**
 * Magnitude comparison across a nominal dimension.
 *
 * Per the color formula: nominal categories all take the SAME slot-1 hue (bar
 * length already encodes the value — spending the identity channel to
 * re-encode it is the classic mistake). Pass `colorByIndex` only when the bars
 * genuinely are separate series.
 */
export function BarChart({
  data,
  indexBy = 'id',
  keys = ['value'],
  title,
  subtitle,
  caption,
  xLegend,
  yLegend,
  height = 280,
  colorByIndex = false,
  controls,
  table,
  layout = 'vertical',
}: {
  data: Array<Record<string, string | number>>;
  indexBy?: string;
  keys?: string[];
  title?: string;
  subtitle?: string;
  caption?: string;
  xLegend?: string;
  yLegend?: string;
  height?: number;
  colorByIndex?: boolean;
  controls?: React.ReactNode;
  table?: TableView;
  layout?: 'vertical' | 'horizontal';
}) {
  const { mode } = useTheme();

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      height={height}
      controls={controls}
      table={table}
    >
      <ResponsiveBar
        data={data}
        keys={keys}
        indexBy={indexBy}
        theme={nivoTheme(mode)}
        layout={layout}
        margin={{
          top: 14,
          right: 22,
          bottom: xLegend ? 48 : 34,
          left: layout === 'horizontal' ? 110 : yLegend ? 58 : 46,
        }}
        padding={0.28}
        // 4px rounded data-ends anchored to the baseline
        borderRadius={4}
        colors={
          colorByIndex
            ? ({ index }) => seriesColor(index, mode)
            : ({ id }) => seriesColor(keys.indexOf(String(id)), mode)
        }
        // 2px surface gap between adjacent fills
        borderWidth={2}
        borderColor={{ from: 'color', modifiers: [['opacity', 0]] }}
        enableGridX={false}
        enableGridY={layout === 'vertical'}
        enableLabel={false}
        axisBottom={{
          tickSize: 0,
          tickPadding: 8,
          legend: xLegend,
          legendOffset: 38,
          legendPosition: 'middle',
        }}
        axisLeft={{
          tickSize: 0,
          tickPadding: 8,
          legend: yLegend,
          legendOffset: -46,
          legendPosition: 'middle',
        }}
        tooltip={({ id, value, indexValue, color }) => (
          <div className="rounded-lg border border-hairline bg-surface-raised px-2.5 py-1.5 text-[11.5px] shadow-lg">
            <div className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: color }}
              />
              <span className="text-ink-secondary">{indexValue}</span>
              <span className="tabular ml-2 font-semibold text-ink">
                {typeof value === 'number' ? value.toFixed(3) : value}
              </span>
            </div>
            {keys.length > 1 ? <div className="mt-0.5 text-ink-muted">{String(id)}</div> : null}
          </div>
        )}
        animate={false}
      />
    </ChartFrame>
  );
}
