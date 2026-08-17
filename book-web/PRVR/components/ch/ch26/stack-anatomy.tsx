'use client';

import Link from 'next/link';
import { useCallback, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import type { Palette, Viewport } from '@/lib/sim/draw';
import { AutonomyStack, BASE_DT, TASKS, type TaskName } from '@/lib/capstone/stack';
import { Bar, drawScene } from './shared';

/**
 * w26.2 — Stack Anatomy.
 *
 * The architecture diagram, except the arrows carry measured traffic and the
 * boxes have switches. A second, identical `AutonomyStack` runs behind this
 * figure at 30 steps per second; the numbers on the edges are that stack's own
 * publication counters, and switching a block off really does stop the task.
 *
 * Architecture diagrams are usually decoration. This one is an instrument.
 */

interface Block {
  name: TaskName;
  x: number;
  y: number;
}

const LAYOUT: Block[] = [
  { name: 'lidar', x: 12, y: 13 },
  { name: 'slam', x: 37, y: 13 },
  { name: 'map', x: 62, y: 13 },
  { name: 'esdf', x: 87, y: 13 },
  { name: 'frontier', x: 87, y: 40 },
  { name: 'plan', x: 62, y: 40 },
  { name: 'control', x: 37, y: 40 },
  { name: 'supervisor', x: 12, y: 40 },
];

/** [from, to, message] — the publisher's counter is what the edge reports. */
const EDGES: [TaskName, TaskName, string][] = [
  ['lidar', 'slam', 'Scan'],
  ['slam', 'map', 'PoseBelief'],
  ['map', 'esdf', 'MapPatch'],
  ['esdf', 'frontier', 'DistanceField'],
  ['frontier', 'plan', 'Frontiers'],
  ['plan', 'control', 'Path'],
  ['control', 'supervisor', 'Cmd'],
];

const pos = (n: TaskName) => LAYOUT.find((b) => b.name === n) as Block;
const spec = (n: TaskName) => TASKS.find((t) => t.name === n)!;

const BW = 19;
const BH = 11;

interface State {
  stack: AutonomyStack;
  trailEst: { x: number; y: number }[];
  trailTruth: { x: number; y: number }[];
}

export function StackAnatomy() {
  const [selected, setSelected] = useState<TaskName>('slam');
  const [disabled, setDisabled] = useState<Partial<Record<TaskName, boolean>>>({});

  const init = useCallback(
    (seed: number): State => ({ stack: new AutonomyStack({ seed }), trailEst: [], trailTruth: [] }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      s.stack.cfg.disabled = disabled;
      s.stack.step();
      if (s.stack.tick % 6 === 0) {
        s.trailEst.push({ x: s.stack.belief.mean.x, y: s.stack.belief.mean.y });
        s.trailTruth.push({ x: s.stack.truth.x, y: s.stack.truth.y });
        if (s.trailEst.length > 400) {
          s.trailEst.shift();
          s.trailTruth.shift();
        }
      }
      return { ...s };
    },
    [disabled],
  );

  const sim = useSimulation<State>({ init, step, fps: 30, initialSeed: 7, loop: false });
  const st = sim.state.stack;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      drawScene(ctx, v, p, sim.state.stack, {
        showTruth: true,
        showRollouts: false,
        showFrontiers: true,
        showScan: false,
        showPath: true,
        trailEst: sim.state.trailEst,
        trailTruth: sim.state.trailTruth,
      });
    },
    [sim.state],
  );

  const detail = spec(selected);
  const stat = st.stats[selected];

  return (
    <WidgetFrame
      id="w26.2"
      title="Stack Anatomy"
      teaches="Every arrow in this diagram carries measured traffic, and every box has a switch — architecture is causal, not decorative."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          A second copy of the same stack runs behind this diagram. Each edge label is the
          publisher&rsquo;s live message rate, and each bar is that task&rsquo;s staleness against
          three of its own periods. <strong>Click a block</strong> to read its contract;{' '}
          <strong>switch one off</strong> and watch the consequence propagate in the inset. Turning
          off <em>SLAM front end</em> reduces the pose belief to dead reckoning and the map shears
          within twenty seconds. Turning off <em>Frontier explorer</em> leaves a perfectly healthy
          robot standing still beside unexplored space — the most instructive failure here, because
          nothing is broken and nothing happens.
        </>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)] lg:divide-x lg:divide-fd-border">
        <div className="p-3">
          <svg
            viewBox="0 0 100 56"
            className="w-full"
            role="img"
            aria-label="Block diagram of the autonomy stack: LiDAR feeds SLAM, which feeds mapping, which feeds the distance field, which feeds the frontier explorer, the planner and the controller, with a supervisor watching everything."
          >
            {/* --- edges --- */}
            {EDGES.map(([from, to, msg]) => {
              const a = pos(from);
              const b = pos(to);
              const off = disabled[from] || disabled[to];
              const hz = st.stats[from].hz;
              const [x1, y1, x2, y2] = anchor(a, b);
              const mx = (x1 + x2) / 2;
              const my = (y1 + y2) / 2;
              return (
                <g key={`${from}-${to}`} opacity={off ? 0.22 : 1}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="var(--pr-measurement)"
                    strokeWidth={0.35 + Math.min(0.9, hz / 24)}
                    markerEnd="url(#ch26-arrow)"
                  />
                  <text
                    x={mx}
                    y={my - 1}
                    textAnchor="middle"
                    style={{ fontSize: 2.1 }}
                    fill="var(--pr-canvas-ink)"
                    opacity={0.75}
                  >
                    {msg}
                  </text>
                  <text
                    x={mx}
                    y={my + 1.8}
                    textAnchor="middle"
                    style={{ fontSize: 2, fontWeight: 600 }}
                    fill="var(--pr-measurement)"
                  >
                    {hz.toFixed(1)} Hz
                  </text>
                </g>
              );
            })}

            {/* the supervisor watches everything, so its edges are dashed */}
            {(['lidar', 'slam', 'map', 'esdf'] as TaskName[]).map((n) => {
              const a = pos(n);
              const s = pos('supervisor');
              return (
                <path
                  key={`sup-${n}`}
                  d={`M ${a.x} ${a.y + BH / 2} Q ${(a.x + s.x) / 2} 30 ${s.x} ${s.y - BH / 2}`}
                  fill="none"
                  stroke="var(--pr-posterior)"
                  strokeWidth={0.22}
                  strokeDasharray="0.9 0.9"
                  opacity={0.5}
                />
              );
            })}

            <defs>
              <marker id="ch26-arrow" markerWidth="4" markerHeight="4" refX="3.4" refY="2" orient="auto">
                <path d="M 0 0.6 L 3.4 2 L 0 3.4 z" fill="var(--pr-measurement)" />
              </marker>
            </defs>

            {/* --- blocks --- */}
            {LAYOUT.map((b) => {
              const t = spec(b.name);
              const off = !!disabled[b.name];
              const active = selected === b.name;
              const s = st.stats[b.name];
              const period = t.period * BASE_DT;
              const heat = Math.min(1, s.staleness / (3 * period));
              return (
                <g
                  key={b.name}
                  transform={`translate(${b.x} ${b.y})`}
                  onClick={() => setSelected(b.name)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelected(b.name);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-pressed={active}
                  aria-label={`${t.label}, ${off ? 'disabled' : `${(1 / period).toFixed(0)} hertz`}`}
                  className="cursor-pointer focus-visible:outline-none"
                >
                  <rect
                    x={-BW / 2}
                    y={-BH / 2}
                    width={BW}
                    height={BH}
                    rx={1.2}
                    fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                    stroke={off ? 'var(--pr-prediction)' : active ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                    strokeWidth={0.5}
                    strokeDasharray={off ? '1 1' : undefined}
                    opacity={off ? 0.45 : 1}
                  />
                  {/* staleness heat: a bar along the bottom edge of the block */}
                  <rect
                    x={-BW / 2 + 1}
                    y={BH / 2 - 1.6}
                    width={(BW - 2) * (off ? 1 : heat)}
                    height={0.7}
                    fill={off || heat > 0.8 ? 'var(--pr-prediction)' : 'var(--pr-measurement)'}
                    opacity={0.85}
                  />
                  <text
                    y={-1.6}
                    textAnchor="middle"
                    style={{ fontSize: 2.5, fontWeight: 600 }}
                    fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-canvas-ink)'}
                  >
                    {t.label}
                  </text>
                  <text
                    y={1.6}
                    textAnchor="middle"
                    style={{ fontSize: 2.1 }}
                    fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
                  >
                    {off ? 'OFF' : `${(1 / period).toFixed(0)} Hz · ch ${t.chapter}`}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* ---- inspector ---- */}
        <div className="border-t border-fd-border lg:border-t-0">
          <div className="p-3">
            <div className="flex items-baseline justify-between gap-2">
              <h4 className="font-display text-base font-semibold">{detail.label}</h4>
              <Link
                href={`/chapters/${detail.chapterSlug}`}
                className="font-mono text-[0.7rem] text-fd-primary hover:underline"
              >
                Ch. {detail.chapter}
              </Link>
            </div>

            <dl className="mt-2.5 space-y-2">
              <Field k="publishes" v={detail.publishes} />
              <Field k="consumes" v={detail.consumes} />
              <Field k="period" v={`${(detail.period * BASE_DT * 1000).toFixed(0)} ms (${(1 / (detail.period * BASE_DT)).toFixed(0)} Hz)`} />
              <Field k="measured" v={`${stat.hz.toFixed(2)} Hz · ${stat.ran} messages · staleness ${stat.staleness.toFixed(2)} s`} />
              <div>
                <dt className="eyebrow">switched off</dt>
                <dd className="font-prose text-[0.84rem] leading-snug" style={{ color: 'var(--pr-prediction)' }}>
                  {detail.degradation}
                </dd>
              </div>
            </dl>

            <label className="mt-3 flex cursor-pointer items-center gap-2 font-ui text-[0.75rem] font-medium">
              <input
                type="checkbox"
                checked={!!disabled[selected]}
                onChange={(e) => setDisabled((d) => ({ ...d, [selected]: e.target.checked }))}
                className="size-3.5 accent-fd-primary"
              />
              <span>Switch {detail.label} off</span>
            </label>

            <div className="mt-3 space-y-1">
              <div className="flex justify-between font-mono text-[0.66rem]">
                <span className="opacity-65">coverage</span>
                <span className="tabular-nums">{(st.coverage() * 100).toFixed(1)}%</span>
              </div>
              <Bar frac={st.coverage()} color="var(--pr-measurement)" />
              <div className="flex justify-between font-mono text-[0.66rem]">
                <span className="opacity-65">|estimate − truth|</span>
                <span className="tabular-nums">{st.error().toFixed(2)} m</span>
              </div>
              <Bar frac={Math.min(1, st.error() / 2)} color="var(--pr-posterior)" warn={st.error() > 0.6} />
            </div>
          </div>

          <div className="border-t border-fd-border">
            <SimCanvas
              world={APARTMENT.bounds}
              draw={draw}
              deps={[sim.tick, sim.state]}
              aspect={12 / 9}
              padding={0.2}
              ariaLabel="Inset view of the same mission, showing how the map and trajectory degrade when a task is switched off."
            />
          </div>
        </div>
      </div>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="eyebrow">{k}</dt>
      <dd className="font-mono text-[0.76rem] leading-snug">{v}</dd>
    </div>
  );
}

/** Shorten an edge so it starts and ends on the block borders, not their centres. */
function anchor(a: Block, b: Block): [number, number, number, number] {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (Math.abs(dx) > Math.abs(dy)) {
    const s = Math.sign(dx);
    return [a.x + (s * BW) / 2, a.y, b.x - (s * BW) / 2 - s * 0.6, b.y];
  }
  const s = Math.sign(dy);
  return [a.x, a.y + (s * BH) / 2, b.x, b.y - (s * BH) / 2 - s * 0.6];
}
