/** The book's charting layer: Nivo wrappers, dashboard shell, and shared chrome. */

export { LineChart } from './line-chart';
export type { ChartMarker, LineChartProps, LineChartSeries } from './line-chart';

export { BarChart } from './bar-chart';
export type { BarChartProps, BarChartSeries } from './bar-chart';

export { HeatMap } from './heatmap';
export type { HeatMapProps } from './heatmap';

export { ScatterChart } from './scatter-chart';
export type { ScatterChartProps, ScatterSeries } from './scatter-chart';

export { NetworkGraph } from './network-graph';
export type { GraphLink, GraphNode, NetworkGraphProps } from './network-graph';

export { StatTile } from './stat-tile';
export type { StatTileProps } from './stat-tile';

export { Dashboard, DashboardPanel } from './dashboard';
export type { DashboardPanelProps, DashboardProps } from './dashboard';

export {
  ChartDataTable,
  ChartFrame,
  ChartLegend,
  ScaleLegend,
  TooltipCard,
  cx,
  pivotSeries,
  resolveSeriesColors,
} from './chart-frame';
export type {
  ChartDataTableProps,
  ChartFrameProps,
  LegendItem,
  PivotableSeries,
  ScaleLegendProps,
  TooltipRow,
} from './chart-frame';

export {
  BOOK_ROLE_ORDER,
  ROLE_VAR,
  adjustLightness,
  formatNumber,
  mixColors,
  readableInk,
  relativeLuminance,
  useAfterMount,
  useBookColors,
  useChartAnimation,
  useChartTheme,
  useChartTokens,
  usePrefersReducedMotion,
  withAlpha,
} from '@/lib/chart-theme';
export type { BookColors, BookRole, ChartTokens } from '@/lib/chart-theme';
