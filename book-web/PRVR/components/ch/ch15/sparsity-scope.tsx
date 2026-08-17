'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { BarChart, Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import type { VarKey } from '@/lib/optim/factor-graph';
import {
  ORDERING_LABEL,
  ORDERING_NOTE,
  buildPattern,
  orderVariables,
  symbolicElimination,
  type OrderingName,
} from '@/lib/optim/ordering';
import { sparsityScene, type SparsityPreset } from '@/lib/optim/scenes';

/**
 * w15.2 — the Sparsity Scope.
 *
 * Two views of one event. On the left the factor graph; on the right the
 * information matrix. Eliminate a variable and its neighbors clique-connect —
 * in the graph as new edges, in the matrix as new nonzeros. Fill-in is not a
 * numerical artifact; it is the induced correlation the elimination *creates*,
 * and the ordering decides how much of it you pay for.
 */

const ORDERINGS: OrderingName[] = ['chronological', 'landmarks-first', 'poses-first', 'min-degree'];

const PRESET_LABEL: Record<SparsityPreset, string> = {
  loop: 'Loop',
  chain: 'Open chain',
  hub: 'One landmark, seen by all',
};

interface Params {
  preset: SparsityPreset;
  ordering: OrderingName;
}

interface Analysis {
  order: VarKey[];
  dims: number[];
  offsets: number[];
  total: number;
  /** Original structural edges, as "i,j" with i < j. */
  original: Set<string>;
  events: ReturnType<typeof symbolicElimination>['events'];
  nnzL: number;
  nnzLowerOmega: number;
  flops: number;
  fillCount: number;
  layout: Map<string, { x: number; y: number; label: string; kind: 'pose' | 'landmark' }>;
}

function analyze(preset: SparsityPreset, ordering: OrderingName): Analysis {
  const scene = sparsityScene(preset);
  const order = orderVariables(scene.graph, ordering);
  const pattern = buildPattern(scene.graph, order);
  const result = symbolicElimination(pattern);

  const original = new Set<string>();
  pattern.adjacency.forEach((nbrs, i) => {
    for (const j of nbrs) if (i < j) original.add(`${i},${j}`);
  });

  const offsets: number[] = [];
  let total = 0;
  for (const d of pattern.dims) {
    offsets.push(total);
    total += d;
  }

  const layout = new Map(
    scene.layout.map((n) => [n.key, { x: n.x, y: n.y, label: n.label, kind: n.kind }]),
  );

  return {
    order,
    dims: pattern.dims,
    offsets,
    total,
    original,
    events: result.events,
    nnzL: result.nnzL,
    nnzLowerOmega: result.nnzLowerOmega,
    flops: result.flops,
    fillCount: result.fill.length,
    layout,
  };
}

/** Edges present after `steps` eliminations, plus the ones that just appeared. */
function replay(a: Analysis, steps: number) {
  const edges = new Set(a.original);
  const fill = new Set<string>();
  const fresh = new Set<string>();
  const gone = new Set<number>();
  for (let s = 0; s < Math.min(steps, a.events.length); s++) {
    const ev = a.events[s];
    for (const [p, q] of ev.fill) {
      const k = p < q ? `${p},${q}` : `${q},${p}`;
      edges.add(k);
      fill.add(k);
      if (s === steps - 1) fresh.add(k);
    }
    gone.add(ev.slot);
  }
  return { edges, fill, fresh, gone };
}

export function SparsityScope() {
  const [params, setParams] = useState<Params>({ preset: 'loop', ordering: 'chronological' });
  const analysis = useMemo(() => analyze(params.preset, params.ordering), [params]);
  const n = analysis.order.length;
  const analysisRef = useRef(analysis);
  analysisRef.current = analysis;
  /** Where the scrubber should land after the next ordering change (presets). */
  const pendingStep = useRef<number | null>(null);

  const init = useCallback(() => ({ step: 0 }), []);
  const step = useCallback(
    (s: { step: number }) => ({ step: (s.step + 1) % (analysisRef.current.order.length + 3) }),
    [],
  );
  const sim = useSimulation<{ step: number }>({ init, step, fps: 1.1 });
  const { setState } = sim;

  // A new ordering restarts the scrub, so the reader always sees it from Ω —
  // unless a preset asked to jump somewhere specific.
  useEffect(() => {
    const target = pendingStep.current ?? 0;
    pendingStep.current = null;
    setState(() => ({ step: target }));
  }, [params, setState]);

  const shown = Math.min(sim.state.step, n);
  const { edges, fill, fresh, gone } = useMemo(() => replay(analysis, shown), [analysis, shown]);

  const leaderboard = useMemo(
    () =>
      ORDERINGS.map((o) => ({
        ordering: o,
        nnzL: analyze(params.preset, o).nnzL,
      })),
    [params.preset],
  );
  const best = Math.min(...leaderboard.map((l) => l.nnzL));

  const cellSize = 100 / analysis.total;
  const nextKey = shown < n ? analysis.order[shown] : null;
  const nLandmarks = analysis.order.filter((k) => k.kind === 'landmark').length;
  const reduced = params.ordering === 'landmarks-first' && shown === nLandmarks && shown > 0;

  return (
    <WidgetFrame
      id="w15.2"
      title="Sparsity Scope"
      teaches="Elimination order is not an implementation detail: marginalizing a variable does not simplify the problem, it relocates the problem as fill-in."
      colorKey={['prediction', 'posterior', 'truth']}
      wide
      caption={
        <>
          Left, the factor graph; right, the lower triangle of Ω in the same variable order. They are
          the same object. Press play and watch one variable at a time get eliminated: its neighbors
          are joined into a clique (new edges, drawn orange, appear <em>simultaneously</em> in the
          graph and in the matrix) and its row and column become a column of the Cholesky factor L.
          Landmark–landmark blocks start empty because no factor ever mentions two landmarks
          together — that emptiness is the entire reason smoothing scales. Now switch the ordering.
          The answer never changes; <code>nnz(L)</code> does, by a lot. Try{' '}
          <strong>{PRESET_LABEL.hub}</strong> with <strong>landmarks first</strong>: eliminating one
          landmark that every pose saw welds the whole trajectory into a single dense clique — the
          Schur trick at its worst. Then eliminate it last and the fill-in vanishes.
        </>
      }
    >
      <div className="grid gap-0 lg:grid-cols-2 lg:divide-x lg:divide-fd-border">
        {/* ------------------------------ the graph ------------------------ */}
        <div className="p-3">
          <p className="eyebrow mb-1">Factor graph</p>
          <svg
            viewBox="-3.4 -3.4 6.8 6.8"
            className="w-full"
            role="img"
            aria-label="A factor graph of eight poses in a loop with landmarks; eliminating a variable adds new edges between its neighbors."
          >
            {[...edges].map((k) => {
              const [i, j] = k.split(',').map(Number);
              const a = analysis.layout.get(keyOf(analysis.order[i]));
              const b = analysis.layout.get(keyOf(analysis.order[j]));
              if (!a || !b) return null;
              const isFill = fill.has(k);
              const dead = gone.has(i) || gone.has(j);
              return (
                <line
                  key={k}
                  x1={a.x}
                  y1={-a.y}
                  x2={b.x}
                  y2={-b.y}
                  stroke={isFill ? 'var(--pr-prediction)' : 'var(--pr-posterior)'}
                  strokeWidth={fresh.has(k) ? 0.075 : 0.035}
                  strokeDasharray={isFill ? '0.12 0.08' : undefined}
                  opacity={dead ? 0.18 : isFill ? 0.95 : 0.55}
                />
              );
            })}
            {analysis.order.map((key, i) => {
              const node = analysis.layout.get(keyOf(key));
              if (!node) return null;
              const dead = gone.has(i);
              const isNext = nextKey !== null && keyOf(nextKey) === keyOf(key);
              return (
                <g key={keyOf(key)} opacity={dead ? 0.25 : 1}>
                  <circle
                    cx={node.x}
                    cy={-node.y}
                    r={node.kind === 'pose' ? 0.19 : 0.15}
                    fill={dead ? 'var(--pr-canvas-bg)' : node.kind === 'pose' ? 'var(--pr-posterior)' : 'var(--pr-measurement)'}
                    stroke={isNext ? 'var(--pr-prediction)' : 'var(--pr-canvas-bg)'}
                    strokeWidth={isNext ? 0.09 : 0.03}
                  />
                  <text
                    x={node.x}
                    y={-node.y + 0.42}
                    textAnchor="middle"
                    style={{ fontSize: 0.26, fontFamily: 'ui-monospace, monospace' }}
                    fill="var(--pr-canvas-ink)"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* ------------------------------ the matrix ----------------------- */}
        <div className="border-t border-fd-border p-3 lg:border-t-0">
          <p className="eyebrow mb-1">Ω, in elimination order (lower triangle)</p>
          <svg
            viewBox="0 0 100 100"
            className="w-full"
            role="img"
            aria-label="A spy plot of the information matrix. Filled cells are nonzero blocks; orange cells are fill-in created by elimination."
          >
            {analysis.order.map((_, i) =>
              analysis.order.map((__, j) => {
                if (j > i) return null;
                const k = j < i ? `${j},${i}` : null;
                const present = i === j || (k !== null && edges.has(k));
                if (!present) return null;
                const isFill = k !== null && fill.has(k);
                const isFresh = k !== null && fresh.has(k);
                const dead = gone.has(i) || gone.has(j);
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={analysis.offsets[j] * cellSize}
                    y={analysis.offsets[i] * cellSize}
                    width={analysis.dims[j] * cellSize}
                    height={analysis.dims[i] * cellSize}
                    fill={isFill ? 'var(--pr-prediction)' : dead ? 'var(--pr-truth)' : 'var(--pr-posterior)'}
                    opacity={isFresh ? 1 : dead ? 0.3 : isFill ? 0.85 : 0.7}
                  />
                );
              }),
            )}
            {/* the frontier: everything above-left is already factored into L */}
            {shown > 0 && shown < n ? (
              <rect
                x={0}
                y={0}
                width={analysis.offsets[shown] * cellSize}
                height={analysis.offsets[shown] * cellSize}
                fill="none"
                stroke="var(--color-fd-primary)"
                strokeWidth={0.6}
                strokeDasharray="1.5 1"
              />
            ) : null}
          </svg>
          <p className="mt-1 font-mono text-[0.65rem] text-fd-muted-foreground">
            {shown === 0
              ? 'Ω as assembled — one block per factor, nothing else'
              : reduced
                ? 'EIF_reduce (Thrun, Table 11.3): the map is eliminated, and what is left is a pose graph'
                : shown >= n
                  ? 'fully eliminated: the whole triangle is now L'
                  : `eliminated ${shown}/${n} — teal frame = the finished columns of L`}
          </p>
        </div>
      </div>

      <div className="px-3 pt-3">
        <Dashboard columns={4}>
          <StatTile label="nnz(Ω) lower" value={analysis.nnzLowerOmega} role="posterior" precision={0} />
          <StatTile
            label="nnz(L)"
            value={analysis.nnzL}
            role="prediction"
            precision={0}
            trend={analysis.nnzL - analysis.nnzLowerOmega}
            trendLabel="fill-in cost"
          />
          <StatTile label="fill edges" value={analysis.fillCount} precision={0} />
          <StatTile label="factorization flops" value={analysis.flops} precision={0} />
        </Dashboard>
      </div>

      <div className="px-3 pt-3">
        <BarChart
          series={[
            {
              id: 'nnz(L)',
              role: 'prediction',
              data: leaderboard.map((l) => ({ x: ORDERING_LABEL[l.ordering], y: l.nnzL })),
            },
          ]}
          xLabel="ordering"
          yLabel="nonzeros in L"
          height={190}
          ariaLabel={`Nonzeros in the Cholesky factor for four orderings; the best is ${best}.`}
        />
      </div>

      <ControlPanel columns={1}>
        <div className="flex flex-col gap-2">
          <span className="eyebrow">Elimination ordering</span>
          <ButtonRow>
            {ORDERINGS.map((o) => (
              <ActionButton
                key={o}
                emphasis={params.ordering === o}
                onClick={() => setParams((p) => ({ ...p, ordering: o }))}
              >
                {ORDERING_LABEL[o]}
              </ActionButton>
            ))}
          </ButtonRow>
          <p className="font-ui text-[0.78rem] leading-snug text-fd-muted-foreground">
            {ORDERING_NOTE[params.ordering]}
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <span className="eyebrow">Graph</span>
          <ButtonRow>
            {(['loop', 'chain', 'hub'] as SparsityPreset[]).map((pr) => (
              <ActionButton
                key={pr}
                emphasis={params.preset === pr}
                onClick={() => setParams((p) => ({ ...p, preset: pr }))}
              >
                {PRESET_LABEL[pr]}
              </ActionButton>
            ))}
            <ActionButton
              onClick={() => {
                // EIF_reduce (Thrun, Table 11.3): eliminate the whole map, and
                // what is left is the pose graph of Chapter 16.
                const target = sparsityScene('loop').nLandmarks;
                pendingStep.current = target;
                setParams({ preset: 'loop', ordering: 'landmarks-first' });
                sim.pause();
                setState(() => ({ step: target }));
              }}
            >
              Reduce like it&apos;s 1999
            </ActionButton>
          </ButtonRow>
        </div>
        <Slider
          label="Variables eliminated"
          role="prediction"
          value={shown}
          min={0}
          max={n}
          step={1}
          format={(v) => `${v} / ${n}`}
          onChange={(v) => {
            sim.pause();
            setState(() => ({ step: v }));
          }}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={() => setState(() => ({ step: 0 }))}
        tick={shown}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

const keyOf = (k: VarKey): string => `${k.kind[0]}${k.id}`;
