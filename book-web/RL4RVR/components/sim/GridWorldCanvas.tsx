'use client';

import { useMemo, useRef, useState } from 'react';
import { ACTION_ARROWS, GridWorld } from '@/lib/rl/gridworld';
import { useTheme } from '@/components/layout/ThemeProvider';
import { sequentialColor } from '@/lib/theme';

/**
 * Renders Rusty's warehouse: value function as a sequential heatmap, greedy
 * policy as directional arrows, trajectory as an overlay path.
 *
 * This is the shared canvas behind the Ch 4 MDP editor, the Ch 5 GPI
 * dashboard, and the Ch 6–7 learning dashboards — one visual language for the
 * whole gridworld thread.
 */
export function GridWorldCanvas({
  env,
  V,
  policy,
  path,
  agentState,
  cellSize = 36,
  showValues = false,
  showPolicy = true,
  onCellClick,
  highlight,
  dragPaint = false,
}: {
  env: GridWorld;
  V?: Float64Array;
  policy?: Int8Array;
  path?: number[];
  agentState?: number;
  cellSize?: number;
  showValues?: boolean;
  showPolicy?: boolean;
  onCellClick?: (state: number) => void;
  /** States to ring, e.g. the cells touched by the current sweep. */
  highlight?: number[];
  /**
   * Let a held pointer paint across cells. Editing a warehouse one click at a
   * time is tedious; dragging a wall into place is how an editor should feel.
   */
  dragPaint?: boolean;
}) {
  const { mode } = useTheme();
  const [hover, setHover] = useState<number | null>(null);
  const painting = useRef(false);
  // Cells already painted during this drag, so crossing one twice does not
  // toggle it back off.
  const paintedThisDrag = useRef<Set<number>>(new Set());

  const paint = (s: number) => {
    if (!onCellClick) return;
    if (paintedThisDrag.current.has(s)) return;
    paintedThisDrag.current.add(s);
    onCellClick(s);
  };

  const endPaint = () => {
    painting.current = false;
    paintedThisDrag.current.clear();
  };

  const { vMin, vMax } = useMemo(() => {
    if (!V) return { vMin: 0, vMax: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, V[s]);
      hi = Math.max(hi, V[s]);
    }
    if (!Number.isFinite(lo)) return { vMin: 0, vMax: 1 };
    return { vMin: lo, vMax: hi === lo ? lo + 1 : hi };
  }, [V, env]);

  const w = env.cols * cellSize;
  const h = env.rows * cellSize;
  const highlightSet = useMemo(() => new Set(highlight ?? []), [highlight]);

  const pathD = useMemo(() => {
    if (!path || path.length < 2) return '';
    return path
      .map((s, i) => {
        const [r, c] = env.rowCol(s);
        const x = c * cellSize + cellSize / 2;
        const y = r * cellSize + cellSize / 2;
        return `${i === 0 ? 'M' : 'L'}${x},${y}`;
      })
      .join(' ');
  }, [path, env, cellSize]);

  return (
    <div className="scroll-x thin-scroll">
      <svg
        width={w}
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        className="max-w-full touch-none"
        role="img"
        aria-label="Rusty's warehouse grid: value function heatmap with greedy policy arrows"
        onPointerUp={endPaint}
        onPointerLeave={() => {
          endPaint();
          setHover(null);
        }}
      >
        {/* Cells */}
        {Array.from({ length: env.rows }, (_, r) =>
          Array.from({ length: env.cols }, (_, c) => {
            const s = r * env.cols + c;
            const shelf = env.isShelf(r, c);
            const terminal = env.isTerminal(s);
            const x = c * cellSize;
            const y = r * cellSize;

            let fill = 'var(--surface-sunken)';
            if (shelf) {
              fill = mode === 'light' ? '#d8d7d0' : '#2c2c2a';
            } else if (terminal) {
              fill = 'var(--status-good)';
            } else if (V) {
              fill = sequentialColor((V[s] - vMin) / (vMax - vMin), mode);
            } else {
              fill = mode === 'light' ? '#ffffff' : '#232322';
            }

            return (
              <g key={s}>
                <rect
                  x={x}
                  y={y}
                  width={cellSize}
                  height={cellSize}
                  fill={fill}
                  stroke="var(--surface-1)"
                  strokeWidth={1}
                  onPointerEnter={() => {
                    if (!shelf) setHover(s);
                    // Shelves are paintable too — that is how you erase a wall.
                    if (dragPaint && painting.current) paint(s);
                  }}
                  onPointerDown={(e) => {
                    if (!onCellClick) return;
                    e.preventDefault();
                    if (dragPaint) {
                      painting.current = true;
                      paintedThisDrag.current.clear();
                      paint(s);
                    } else if (!shelf) {
                      onCellClick(s);
                    }
                  }}
                  style={{ cursor: onCellClick ? 'pointer' : 'default' }}
                />
                {highlightSet.has(s) && (
                  <rect
                    x={x + 1}
                    y={y + 1}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    fill="none"
                    stroke="var(--series-4)"
                    strokeWidth={2}
                    rx={3}
                  />
                )}
              </g>
            );
          }),
        )}

        {/* Policy arrows */}
        {showPolicy &&
          policy &&
          env.states.map((s) => {
            if (env.isTerminal(s) || policy[s] < 0) return null;
            const [r, c] = env.rowCol(s);
            const x = c * cellSize + cellSize / 2;
            const y = r * cellSize + cellSize / 2;
            // Arrow ink flips with the cell's darkness so it stays legible.
            const t = V ? (V[s] - vMin) / (vMax - vMin) : 0;
            const dark = mode === 'light' ? t > 0.55 : t < 0.45;
            return (
              <text
                key={`p${s}`}
                x={x}
                y={y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={cellSize * 0.46}
                fill={dark ? '#ffffff' : 'var(--text-primary)'}
                opacity={0.92}
                pointerEvents="none"
              >
                {ACTION_ARROWS[policy[s]]}
              </text>
            );
          })}

        {/* Value labels */}
        {showValues &&
          V &&
          env.states.map((s) => {
            if (env.isTerminal(s)) return null;
            const [r, c] = env.rowCol(s);
            const t = (V[s] - vMin) / (vMax - vMin);
            const dark = mode === 'light' ? t > 0.55 : t < 0.45;
            return (
              <text
                key={`v${s}`}
                x={c * cellSize + cellSize / 2}
                y={r * cellSize + cellSize - 4}
                textAnchor="middle"
                fontSize={9}
                fill={dark ? '#ffffff' : 'var(--text-secondary)'}
                opacity={0.85}
                pointerEvents="none"
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {V[s].toFixed(1)}
              </text>
            );
          })}

        {/* Trajectory overlay */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke="var(--series-2)"
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.9}
          />
        )}

        {/* Dock label */}
        {(() => {
          const [r, c] = env.rowCol(env.goalState);
          return (
            <text
              x={c * cellSize + cellSize / 2}
              y={r * cellSize + cellSize / 2}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={cellSize * 0.4}
              fill="#ffffff"
              fontWeight={700}
              pointerEvents="none"
            >
              D
            </text>
          );
        })()}

        {/* Rusty */}
        {agentState !== undefined &&
          (() => {
            const [r, c] = env.rowCol(agentState);
            return (
              <g pointerEvents="none">
                <circle
                  cx={c * cellSize + cellSize / 2}
                  cy={r * cellSize + cellSize / 2}
                  r={cellSize * 0.28}
                  fill="var(--series-2)"
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                />
              </g>
            );
          })()}

        {/* Hover read-out */}
        {hover !== null && V && (
          <g pointerEvents="none">
            {(() => {
              const [r, c] = env.rowCol(hover);
              const boxW = 92;
              const boxH = 34;
              const x = Math.min(Math.max(c * cellSize - boxW / 2 + cellSize / 2, 2), w - boxW - 2);
              const y = r * cellSize > boxH + 4 ? r * cellSize - boxH - 4 : r * cellSize + cellSize + 4;
              return (
                <>
                  <rect
                    x={x}
                    y={y}
                    width={boxW}
                    height={boxH}
                    rx={6}
                    fill="var(--surface-2)"
                    stroke="var(--baseline)"
                  />
                  <text x={x + 8} y={y + 14} fontSize={10} fill="var(--text-muted)">
                    state {hover}
                  </text>
                  <text
                    x={x + 8}
                    y={y + 27}
                    fontSize={11}
                    fontWeight={600}
                    fill="var(--text-primary)"
                    style={{ fontVariantNumeric: 'tabular-nums' }}
                  >
                    V = {V[hover].toFixed(2)}
                  </text>
                </>
              );
            })()}
          </g>
        )}
      </svg>
    </div>
  );
}

/** Legend strip for the sequential value ramp. */
export function ValueLegend({ min, max }: { min: number; max: number }) {
  const { mode } = useTheme();
  const stops = Array.from({ length: 24 }, (_, i) => sequentialColor(i / 23, mode));
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-muted">
      <span className="tabular">{min.toFixed(1)}</span>
      <span
        className="h-2 w-28 rounded-full"
        style={{ background: `linear-gradient(90deg, ${stops.join(',')})` }}
        aria-hidden
      />
      <span className="tabular">{max.toFixed(1)}</span>
      <span className="ml-1">V(s)</span>
    </div>
  );
}
