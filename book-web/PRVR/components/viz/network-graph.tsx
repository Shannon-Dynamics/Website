'use client';

import { useCallback, useMemo } from 'react';
import type { ReactNode } from 'react';
import { ResponsiveNetwork } from '@nivo/network';
import type { CustomLayerProps, NodeTooltipProps } from '@nivo/network';

import {
  useAfterMount,
  useBookColors,
  useChartAnimation,
  useChartTheme,
  useChartTokens,
  withAlpha,
  type BookRole,
} from '@/lib/chart-theme';
import { ChartDataTable, ChartFrame, ChartLegend, TooltipCard } from './chart-frame';

export interface GraphNode {
  id: string;
  /** Takes the book's reserved color for that meaning. */
  role?: BookRole;
  label?: string;
  /** Marker diameter in pixels. */
  size?: number;
  /**
   * Optional layout hint in any consistent units — a pose graph already knows
   * where its nodes belong. Used for the pre-hydration render.
   */
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  /** Rest length for the force layout; shorter reads as a tighter constraint. */
  distance?: number;
  thickness?: number;
}

export interface NetworkGraphProps {
  nodes: readonly GraphNode[];
  links: readonly GraphLink[];
  height?: number;
  /** Node labels, on by default for graphs small enough to carry them. */
  labels?: boolean;
  legend?: boolean;
  ariaLabel?: string;
  caption?: ReactNode;
}

const LABEL_LIMIT = 18;
const EDGE_TABLE_LIMIT = 120;

const ROLE_LABEL: Record<BookRole, string> = {
  prior: 'Prior',
  prediction: 'Prediction',
  measurement: 'Measurement',
  posterior: 'Posterior',
  truth: 'Ground truth',
};

export function NetworkGraph({
  nodes,
  links,
  height = 360,
  labels,
  legend,
  ariaLabel,
  caption,
}: NetworkGraphProps) {
  const theme = useChartTheme();
  const tokens = useChartTokens();
  const bookColors = useBookColors();
  const animate = useChartAnimation();

  /*
   * Nivo runs the force layout in an effect, so on the server — and therefore
   * in the exported HTML — it has nothing to draw. The deterministic layout
   * below fills that first paint, and the simulation takes over on mount.
   * Rendering it for the first client pass too keeps hydration exact.
   */
  const simulated = useAfterMount();

  const colorOf = useCallback(
    (node: GraphNode) => (node.role ? bookColors[node.role] : bookColors.truth),
    [bookColors],
  );

  const data = useMemo(() => ({ nodes: [...nodes], links: [...links] }), [nodes, links]);
  const showLabels = labels ?? nodes.length <= LABEL_LIMIT;

  const roleItems = useMemo(() => {
    const seen = new Map<BookRole, string>();
    for (const node of nodes) {
      if (node.role && !seen.has(node.role)) seen.set(node.role, bookColors[node.role]);
    }
    return [...seen].map(([role, color]) => ({ id: role, label: ROLE_LABEL[role], color, shape: 'dot' as const }));
  }, [nodes, bookColors]);

  const LabelLayer = useMemo(
    () =>
      function NetworkLabels({ nodes: computed }: CustomLayerProps<GraphNode, GraphLink>) {
        if (!showLabels) return null;
        return (
          <g aria-hidden>
            {computed.map((node) => (
              <text
                key={node.id}
                x={node.x}
                y={node.y - node.size / 2 - 6}
                textAnchor="middle"
                style={{
                  fill: tokens.mutedForeground,
                  fontFamily: tokens.fontMono,
                  fontSize: 10,
                  pointerEvents: 'none',
                }}
              >
                {node.data.label ?? node.id}
              </text>
            ))}
          </g>
        );
      },
    [showLabels, tokens],
  );

  const Tooltip = useMemo(
    () =>
      function NetworkTooltip({ node }: NodeTooltipProps<GraphNode>) {
        const degree = links.filter((link) => link.source === node.id || link.target === node.id).length;
        return (
          <TooltipCard
            title={node.data.label ?? node.id}
            rows={[
              ...(node.data.role
                ? [{ color: node.color, label: 'role', value: ROLE_LABEL[node.data.role] }]
                : []),
              { label: 'edges', value: String(degree) },
            ]}
          />
        );
      },
    [links],
  );

  const description =
    ariaLabel ?? `Graph with ${nodes.length} nodes and ${links.length} edges.`;

  return (
    <ChartFrame
      height={height}
      caption={caption}
      legend={(legend ?? roleItems.length > 1) ? <ChartLegend items={roleItems} /> : null}
      table={
        links.length > 0 && links.length <= EDGE_TABLE_LIMIT ? (
          <ChartDataTable
            caption={description}
            columns={['Edge', 'From', 'To']}
            rows={links.map((link, index) => [String(index + 1), link.source, link.target])}
          />
        ) : null
      }
    >
      {simulated ? (
        <ResponsiveNetwork<GraphNode, GraphLink>
          defaultWidth={720}
          defaultHeight={height}
          data={data}
          theme={theme}
          margin={{ top: 16, right: 16, bottom: 16, left: 16 }}
          linkDistance={(link) => link.distance ?? 62}
          centeringStrength={0.5}
          repulsivity={90}
          iterations={120}
          nodeSize={(node) => node.size ?? 13}
          activeNodeSize={(node) => (node.size ?? 13) + 6}
          inactiveNodeSize={(node) => node.size ?? 13}
          nodeColor={colorOf}
          // A 2px surface ring keeps overlapping nodes legible.
          nodeBorderWidth={2}
          nodeBorderColor={tokens.surface}
          linkThickness={(link) => link.data.thickness ?? 1.5}
          linkColor={withAlpha(tokens.mutedForeground, 0.45)}
          layers={['links', 'nodes', LabelLayer]}
          nodeTooltip={Tooltip}
          animate={animate}
          motionConfig="gentle"
          role="img"
          ariaLabel={description}
        />
      ) : (
        <StaticGraph
          nodes={nodes}
          links={links}
          colorOf={colorOf}
          linkColor={withAlpha(tokens.mutedForeground, 0.45)}
          ringColor={tokens.surface}
          labelColor={tokens.mutedForeground}
          labelFont={tokens.fontMono}
          showLabels={showLabels}
          description={description}
        />
      )}
    </ChartFrame>
  );
}

interface StaticGraphProps {
  nodes: readonly GraphNode[];
  links: readonly GraphLink[];
  colorOf: (node: GraphNode) => string;
  linkColor: string;
  ringColor: string;
  labelColor: string;
  labelFont: string;
  showLabels: boolean;
  description: string;
}

const VIEW = 100;
const PAD = 14;

/**
 * The build-time picture: caller-supplied coordinates when the graph has them,
 * otherwise a ring in input order. No randomness, so the exported HTML and the
 * hydrating client agree byte for byte.
 */
function StaticGraph({
  nodes,
  links,
  colorOf,
  linkColor,
  ringColor,
  labelColor,
  labelFont,
  showLabels,
  description,
}: StaticGraphProps) {
  const positions = useMemo(() => {
    const placed = new Map<string, { x: number; y: number }>();
    const positioned = nodes.filter((node) => node.x !== undefined && node.y !== undefined);

    if (positioned.length === nodes.length && nodes.length > 0) {
      const xs = nodes.map((node) => node.x as number);
      const ys = nodes.map((node) => node.y as number);
      const spanX = Math.max(...xs) - Math.min(...xs) || 1;
      const spanY = Math.max(...ys) - Math.min(...ys) || 1;
      const scale = (VIEW - 2 * PAD) / Math.max(spanX, spanY);
      const minX = Math.min(...xs);
      const minY = Math.min(...ys);
      for (const node of nodes) {
        placed.set(node.id, {
          x: PAD + ((node.x as number) - minX) * scale,
          y: PAD + ((node.y as number) - minY) * scale,
        });
      }
      return placed;
    }

    const radius = VIEW / 2 - PAD;
    nodes.forEach((node, index) => {
      const angle = (index / Math.max(1, nodes.length)) * Math.PI * 2 - Math.PI / 2;
      placed.set(node.id, {
        x: VIEW / 2 + Math.cos(angle) * radius,
        y: VIEW / 2 + Math.sin(angle) * radius,
      });
    });
    return placed;
  }, [nodes]);

  return (
    <svg
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      preserveAspectRatio="xMidYMid meet"
      width="100%"
      height="100%"
      role="img"
      aria-label={description}
    >
      <g>
        {links.map((link, index) => {
          const from = positions.get(link.source);
          const to = positions.get(link.target);
          if (!from || !to) return null;
          return (
            <line
              key={`${link.source}-${link.target}-${index}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={linkColor}
              strokeWidth={(link.thickness ?? 1.5) * 0.22}
            />
          );
        })}
      </g>
      <g>
        {nodes.map((node) => {
          const at = positions.get(node.id);
          if (!at) return null;
          const radius = Math.max(1.4, (node.size ?? 13) * 0.09);
          return (
            <g key={node.id}>
              <circle
                cx={at.x}
                cy={at.y}
                r={radius}
                fill={colorOf(node)}
                stroke={ringColor}
                strokeWidth={0.3}
              />
              {showLabels ? (
                <text
                  x={at.x}
                  y={at.y - radius - 1.4}
                  textAnchor="middle"
                  style={{ fill: labelColor, fontFamily: labelFont, fontSize: 3 }}
                >
                  {node.label ?? node.id}
                </text>
              ) : null}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
