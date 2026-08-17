'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Toggle, Transport } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import {
  blockNormMatrix,
  buildWindow,
  nnzBlocks,
  slideWindow,
  type WindowState,
} from '@/lib/vision/window';

/**
 * w18.3 — Marginalization Fill-In.
 *
 * A sliding-window smoother cannot keep every state, so it marginalizes the
 * oldest one out with a Schur complement. The information is *not* deleted: it
 * is relocated onto everything the departing state touched, and the prior
 * densifies. This widget runs both policies side by side on the same window —
 * exact marginalization, and SEIF's answer of cutting the new links to keep the
 * sparsity — and reports the price of the second one in the only currency that
 * matters: the covariance the estimator reports for a keyframe nobody measured
 * again.
 *
 * Everything is the real `lib/vision/window.ts`: a genuine block Schur
 * complement over 6-DOF keyframes and 3-DOF landmarks.
 */

/** Simulation ticks between window slides — one slide per second at fps 5. */
const TICKS_PER_SLIDE = 5;
const SLIDES = 9;
/** A stationary window: eight keyframes, twelve landmarks, each seen four times. */
const WINDOW = { keyframes: 8, landmarks: 12, coVisibility: 4 };

interface State {
  exact: WindowState;
  sparse: WindowState;
}

const ASPECT = 1.62;
const GRID = { x0: 0.13, y0: 0.06, x1: 0.99, y1: 0.92 };

export function MarginalizationFillIn() {
  const [showSparse, setShowSparse] = useState(false);
  const [seed, setSeed] = useState(1807);

  const init = useCallback(
    (): State => ({
      exact: buildWindow({ ...WINDOW, seed }),
      sparse: buildWindow({ ...WINDOW, seed }),
    }),
    [seed],
  );

  /** One slide of the window, applied to both policies at once. */
  const slide = useCallback(
    (s: State): State => ({
      exact: slideWindow(s.exact, { sparsify: false, coVisibility: WINDOW.coVisibility }),
      // threshold = 1 cuts every link the elimination just created: SEIF's
      // sparsification in its most aggressive form.
      sparse: slideWindow(s.sparse, {
        sparsify: true,
        threshold: 1,
        coVisibility: WINDOW.coVisibility,
      }),
    }),
    [],
  );

  const step = useCallback(
    (s: State, tick: number): State => ((tick + 1) % TICKS_PER_SLIDE === 0 ? slide(s) : s),
    [slide],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 5,
    maxTicks: TICKS_PER_SLIDE * SLIDES,
    loop: true,
    initialSeed: 1807,
  });

  // A new seed means a new window; `init` already closes over it, so the reset
  // is all that is needed to pick it up.
  const reset = sim.reset;
  useEffect(() => {
    reset();
  }, [seed, reset]);

  const shown = showSparse ? sim.state.sparse : sim.state.exact;

  const stats = useMemo(() => {
    const e = sim.state.exact;
    const s = sim.state.sparse;
    const last = s.history[s.history.length - 1];
    const over = last.sigmaExact > 0 ? 1 - last.sigmaActual / last.sigmaExact : 0;
    return {
      slides: e.slides,
      nnzExact: nnzBlocks(e),
      nnzSparse: nnzBlocks(s),
      fill: e.fresh.size,
      dropped: s.dropped.size,
      over: over * 100,
    };
  }, [sim.state]);

  const series = useMemo(
    () => [
      {
        id: 'exact marginalization',
        role: 'posterior' as const,
        data: sim.state.exact.history.map((h) => ({ x: h.slide, y: h.nnz })),
      },
      {
        id: 'links dropped (SEIF)',
        role: 'prior' as const,
        data: sim.state.sparse.history.map((h) => ({ x: h.slide, y: h.nnz })),
      },
    ],
    [sim.state],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const W = (wx: number, wy: number): [number, number] => [sx(v, wx), sy(v, wy)];
      const norms = blockNormMatrix(shown);
      const n = shown.blocks.length;
      const cell = Math.min((GRID.x1 - GRID.x0) / n, (GRID.y1 - GRID.y0) / n);
      let peak = 1e-12;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) if (i !== j) peak = Math.max(peak, norms[i][j]);
      }

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const x = GRID.x0 + j * cell;
          const y = GRID.y1 - i * cell;
          const key = `${Math.min(i, j)},${Math.max(i, j)}`;
          const mag = Math.min(1, Math.sqrt(norms[i][j] / peak));
          const isFresh = shown.fresh.has(key);
          const isDropped = shown.dropped.has(key);

          if (isDropped) {
            // The link the elimination created and the sparsifier deleted.
            ctx.strokeStyle = p.prediction;
            ctx.globalAlpha = 0.55;
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.strokeRect(sx(v, x), sy(v, y), sl(v, cell) - 1, sl(v, cell) - 1);
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
            continue;
          }
          if (norms[i][j] <= 1e-9) continue;

          ctx.fillStyle = isFresh ? p.prediction : i === j ? p.posterior : p.prior;
          ctx.globalAlpha = isFresh ? 0.95 : 0.12 + 0.88 * mag;
          ctx.fillRect(sx(v, x), sy(v, y), sl(v, cell) - 1, sl(v, cell) - 1);
          ctx.globalAlpha = 1;
        }
      }

      // Block labels: keyframes and landmarks, so the reader can see *which*
      // survivors got clique-connected.
      for (let i = 0; i < n; i++) {
        const b = shown.blocks[i];
        const colour = b.kind === 'keyframe' ? p.posterior : p.truth;
        label(ctx, b.label, sx(v, GRID.x0 - 0.015), sy(v, GRID.y1 - (i + 0.5) * cell), colour, {
          size: 8.5,
          align: 'right',
        });
        label(ctx, b.label, sx(v, GRID.x0 + (i + 0.5) * cell), sy(v, GRID.y1 + 0.03), colour, {
          size: 8.5,
          align: 'center',
        });
      }
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(sx(v, GRID.x0), sy(v, GRID.y1), sl(v, n * cell), sl(v, n * cell));

      /* ---- legend --------------------------------------------------------- */
      const lx = 1.06;
      const items: [string, string, number][] = [
        ['existing coupling', p.prior, 0.75],
        ['diagonal block', p.posterior, 0.75],
        ['fill-in from this slide', p.prediction, 0.95],
        ['link dropped (SEIF)', p.prediction, 0.0],
      ];
      items.forEach(([text, colour, alpha], i) => {
        const y = 0.86 - i * 0.075;
        if (alpha === 0) {
          ctx.strokeStyle = colour;
          ctx.setLineDash([2, 2]);
          ctx.lineWidth = 1;
          ctx.strokeRect(sx(v, lx), sy(v, y), sl(v, 0.035), sl(v, 0.035));
          ctx.setLineDash([]);
        } else {
          ctx.fillStyle = colour;
          ctx.globalAlpha = alpha;
          ctx.fillRect(sx(v, lx), sy(v, y), sl(v, 0.035), sl(v, 0.035));
          ctx.globalAlpha = 1;
        }
        label(ctx, text, sx(v, lx + 0.05), sy(v, y - 0.018), p.truth, { size: 9 });
      });

      label(
        ctx,
        `slide ${shown.slides} · ${shown.blocks.filter((b) => b.kind === 'keyframe').length} keyframes, ${
          shown.blocks.filter((b) => b.kind === 'landmark').length
        } landmarks`,
        sx(v, lx),
        sy(v, 0.44),
        p.truth,
        { size: 9.5 },
      );
      label(
        ctx,
        showSparse ? 'sparsified window' : 'exact window',
        sx(v, lx),
        sy(v, 0.37),
        showSparse ? p.prediction : p.posterior,
        { size: 10.5, weight: 600 },
      );
      label(ctx, `${nnzBlocks(shown)} nonzero blocks`, sx(v, lx), sy(v, 0.31), p.truth, {
        size: 9.5,
      });
    },
    [shown, showSparse],
  );

  return (
    <WidgetFrame
      id="w18.3"
      title="Marginalization Fill-In"
      teaches="Marginalization does not delete an old state — it redistributes that state's information onto everything it touched, and the prior densifies."
      colorKey={['prior', 'prediction', 'posterior', 'truth']}
      wide
      caption={
        <>
          Every second the window slides: the oldest keyframe and the landmarks it hosted are
          Schur-complemented out, and the couplings they carried reappear as{' '}
          <strong>orange fill-in</strong> among the survivors — precisely between the keyframes that
          co-observed the same landmark. Count the blocks: exact marginalization only ever adds. Turn
          on <strong>drop the new links</strong> and the sparsity comes back, at the price the banner
          reports: the oldest surviving keyframe&rsquo;s reported σ shrinks, though no measurement
          justified it. That is the 2000-vintage SEIF dilemma, and modern systems answer it by
          keeping the dense prior and freezing its linearization point instead.
        </>
      }
    >
      <div className="p-3">
        <Dashboard columns={4}>
          <StatTile
            label="nonzero blocks, exact"
            value={stats.nnzExact}
            role="posterior"
            precision={0}
            sparkline={sim.state.exact.history.map((h) => h.nnz)}
          />
          <StatTile
            label="fill-in created last slide"
            value={stats.fill}
            role="prediction"
            precision={0}
          />
          <StatTile
            label="links dropped, sparsified"
            value={stats.dropped}
            role="prior"
            precision={0}
          />
          <StatTile
            label="σ under-reported"
            value={stats.over}
            unit="%"
            role="prediction"
            precision={2}
            sparkline={sim.state.sparse.history.map((h) =>
              h.sigmaExact > 0 ? (1 - h.sigmaActual / h.sigmaExact) * 100 : 0,
            )}
          />
        </Dashboard>
      </div>

      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: ASPECT, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, shown, showSparse]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="A spy plot of the sliding window's information matrix. Keyframe and landmark blocks are shaded by coupling strength; blocks created by the most recent marginalization are highlighted, and blocks deleted by sparsification are shown as dashed outlines."
      />

      <div className="px-3 pb-1 pt-3">
        <LineChart
          series={series}
          xLabel="window slide"
          yLabel="nonzero blocks in Ω"
          height={200}
          curve="stepAfter"
          caption="Exact marginalization only ever densifies; dropping the new links holds the sparsity flat — and quietly buys it with confidence the window never earned."
        />
      </div>

      {showSparse ? (
        <p
          className="mx-3 mb-3 rounded-sm border-l-2 px-3 py-2 font-ui text-[0.78rem]"
          style={{ borderLeftColor: 'var(--pr-prediction)', color: 'var(--pr-prediction)' }}
        >
          Consistency warning: with the fill-in deleted, the anchor keyframe&rsquo;s reported σ is{' '}
          {Math.abs(stats.over).toFixed(2)}%{' '}
          <em>{stats.over >= 0 ? 'smaller' : 'larger'}</em> than the exact posterior — and nothing
          measured it. Sparsity was bought with a covariance that no longer answers to the evidence.
        </p>
      ) : null}

      <ControlPanel columns={2}>
        <Toggle
          label="SEIF-style: drop the links the elimination created"
          checked={showSparse}
          onChange={setShowSparse}
          role="prediction"
        />
        <p className="self-center font-ui text-[0.72rem] text-fd-muted-foreground">
          Both windows run at once — the toggle only chooses which one the spy plot shows.
        </p>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        // Step means "one slide", not "one animation tick" — the slide is the
        // only event in this widget worth stepping through.
        onStep={() => {
          sim.pause();
          sim.setState(slide);
        }}
        onReset={sim.reset}
        onReseed={() => setSeed(Math.floor(Math.random() * 100000))}
        seed={seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
