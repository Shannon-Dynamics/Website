'use client';

import Link from 'next/link';
import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { HistogramFilter1D, gaussianKernel } from '@/lib/filters/bayes';
import { HALLWAY_1D, hallwayMeasurementLikelihood, isDoorAt } from '@/lib/sim/world';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w8.3 — the Grid Resolution Ladder.
 *
 * Four histogram filters, K ∈ {8, 32, 128, 512}, driven by one seeded Hallway
 * log. Same motion, same measurements, same code — only the decomposition
 * changes. The coarse rung cannot separate two doors that the fine rung
 * resolves cleanly, and the cost meter underneath says what the fine rung is
 * charging for that. All four are the real `HistogramFilter1D` from
 * `lib/filters/bayes.ts`.
 */

const { length: L } = HALLWAY_1D;
const RUNGS = [8, 32, 128, 512];
const STEP = 0.42;

interface Params {
  /** The headline knob: how many dimensions this grid would have to cover. */
  dim: number;
  motionNoise: number;
}

interface State {
  filters: HistogramFilter1D[];
  rng: Rng;
  truth: number;
  moved: boolean;
  sawDoor: boolean | null;
}

const wrap = (x: number) => ((x % L) + L) % L;

export function GridResolutionLadder() {
  const [params, setParams] = useState<Params>({ dim: 1, motionNoise: 0.12 });
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const init = useCallback((seed: number): State => {
    const filters = RUNGS.map((k) => {
      const f = new HistogramFilter1D({ length: L, cells: k, wrap: true });
      f.setUniform();
      return f;
    });
    return { filters, rng: new Rng(seed), truth: 0.9, moved: false, sawDoor: null };
  }, []);

  const step = useCallback((s: State, tick: number): State => {
    const p = paramsRef.current;
    const { filters, rng } = s;
    const move = tick % 2 === 0;

    if (move) {
      const truth = wrap(s.truth + STEP + rng.normal(0, p.motionNoise));
      const kernel = gaussianKernel(Math.max(p.motionNoise, 1e-3));
      // The identical control reaches every rung; only the cell size differs.
      for (const f of filters) f.predict(STEP, kernel);
      return { ...s, truth, moved: true, sawDoor: null };
    }

    const sawDoor = rng.next() < (isDoorAt(s.truth) ? 0.9 : 0.1);
    for (const f of filters) f.correct((x) => hallwayMeasurementLikelihood(x, sawDoor, 0.9));
    return { ...s, moved: false, sawDoor };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 2.5, initialSeed: 21 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { filters, truth, moved } = sim.state;

      const laneH = 1 / RUNGS.length;
      filters.forEach((f, r) => {
        const top = 1 - r * laneH - 0.02;
        const bottom = 1 - (r + 1) * laneH + 0.05;
        const baseY = sy(v, bottom);
        const topY = sy(v, top);
        const h = baseY - topY;
        const cells = f.belief();
        const peak = Math.max(...cells, 1e-9);
        const cw = sl(v, f.cellWidth);

        // Cell boundaries: at K = 8 these are the whole story.
        if (f.n <= 64) {
          ctx.strokeStyle = p.grid;
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 0; i <= f.n; i++) {
            ctx.moveTo(sx(v, i * f.cellWidth), topY);
            ctx.lineTo(sx(v, i * f.cellWidth), baseY);
          }
          ctx.stroke();
        }

        ctx.fillStyle = moved ? p.prediction : p.posterior;
        for (let i = 0; i < f.n; i++) {
          const bh = (cells[i] / peak) * h;
          if (bh <= 0) continue;
          ctx.fillRect(sx(v, i * f.cellWidth), baseY - bh, Math.max(cw - 0.6, 0.8), bh);
        }

        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx(v, 0), baseY);
        ctx.lineTo(sx(v, L), baseY);
        ctx.stroke();

        // The truth, drawn through every rung so the alias is visible at a glance.
        ctx.strokeStyle = p.truth;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, truth), topY);
        ctx.lineTo(sx(v, truth), baseY);
        ctx.stroke();
        ctx.setLineDash([]);

        label(ctx, `K = ${f.n}`, sx(v, 0.06), topY + 7, p.ink, { size: 10, weight: 600 });
        label(
          ctx,
          `Δx = ${f.cellWidth.toFixed(3)} m   H = ${f.entropy().toFixed(2)} bits`,
          sx(v, L - 0.06),
          topY + 7,
          p.truth,
          { size: 9, align: 'right' },
        );
      });
    },
    [sim.state],
  );

  const rows = useMemo(() => {
    const d = params.dim;
    return sim.state.filters.map((f) => {
      const cells = Math.pow(f.n, d);
      const err = Math.min(Math.abs(f.mode() - sim.state.truth), L - Math.abs(f.mode() - sim.state.truth));
      return {
        k: f.n,
        cells,
        // Naive prediction touches every ordered pair of cells; the banded
        // implementation only reaches W neighbours, and W grows as rᵈ too.
        naive: cells * cells,
        banded: cells * Math.pow(8, d),
        err,
      };
    });
  }, [sim.state, params.dim]);

  return (
    <WidgetFrame
      id="w8.3"
      title="Grid Resolution Ladder"
      teaches="Finer is not free: the same belief costs r cells in 1-D and r³ in 3-D, and the naive prediction step squares that."
      colorKey={['prediction', 'posterior', 'truth']}
      caption={
        <>
          One seeded Hallway log, four decompositions. Watch the top rung: with{' '}
          <code>K = 8</code> the cells are 1.25 m wide, two of the three doors fall inside a single
          cell, and the filter can never tell them apart — its posterior is <em>right</em> about a
          region that is useless to a planner. The bottom rung separates them on the first sighting.
          Then move the dimension slider. The 1-D story is comfortable; at <em>d</em> = 3 — an{' '}
          <em>(x, y, θ)</em> pose grid, which is what{' '}
          <Link href="/chapters/ch12-localization-global">Chapter 12</Link> actually needs — the fine rung
          is asking for 134 million cells and a naive prediction step of 10<sup>16</sup> operations.
          That number is the entire reason particles exist.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.1, maxX: L + 0.1, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={1.9}
        padding={0}
        ariaLabel="Four stacked histogram beliefs over the same corridor at 8, 32, 128 and 512 cells, with the true robot position marked as a dashed line through all four."
      />

      <div className="overflow-x-auto border-t border-fd-border">
        <table className="w-full border-collapse font-mono text-[0.72rem] tabular-nums">
          <thead>
            <tr className="border-b border-fd-border text-fd-muted-foreground">
              <th className="px-3 py-1.5 text-left font-normal">K</th>
              <th className="px-3 py-1.5 text-right font-normal">cells at d = {params.dim}</th>
              <th className="px-3 py-1.5 text-right font-normal">naive predict</th>
              <th className="px-3 py-1.5 text-right font-normal">banded predict</th>
              <th className="px-3 py-1.5 text-right font-normal">|MAP − truth|</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k} className="border-b border-fd-border/60 last:border-b-0">
                <td className="px-3 py-1.5">{r.k}</td>
                <td className="px-3 py-1.5 text-right">{fmt(r.cells)}</td>
                <td
                  className="px-3 py-1.5 text-right"
                  style={r.naive > 1e9 ? { color: 'var(--pr-prediction)' } : undefined}
                >
                  {fmt(r.naive)}
                </td>
                <td className="px-3 py-1.5 text-right">{fmt(r.banded)}</td>
                <td className="px-3 py-1.5 text-right">{r.err.toFixed(2)} m</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="State dimension d"
          value={params.dim}
          min={1}
          max={4}
          step={1}
          onChange={(v) => setParams((q) => ({ ...q, dim: Math.round(v) }))}
          format={(v) => String(Math.round(v))}
          help="The grid is r cells per axis. Cells = r^d — the curse, in one exponent."
        />
        <Slider
          label="Motion noise σ"
          role="prediction"
          value={params.motionNoise}
          min={0.02}
          max={0.4}
          step={0.01}
          unit="m"
          onChange={(v) => setParams((q) => ({ ...q, motionNoise: v }))}
          help="Below one cell width the coarse rungs stop blurring at all — discretization has swallowed the noise."
        />
      </ControlPanel>

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

/** Compact magnitudes: readers need the order, not the digits. */
function fmt(x: number): string {
  if (x < 1000) return String(Math.round(x));
  if (x < 1e6) return `${(x / 1e3).toFixed(1)}k`;
  if (x < 1e9) return `${(x / 1e6).toFixed(1)}M`;
  if (x < 1e12) return `${(x / 1e9).toFixed(1)}G`;
  return x.toExponential(1);
}
