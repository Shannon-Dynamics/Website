'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { VolumeToy1D, type Camera1D } from '@/lib/mapping/volume-render';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w19.4 — Photometric Descent.
 *
 * The whole of radiance-field mapping at one spatial dimension: a density field
 * is a generative model of measurements, the residual between rendered and
 * observed depth is a negative log likelihood, and the map is whatever
 * minimizes it. No network, no rasterizer, no CUDA — just the mechanism, and
 * the two failure modes that dominate the literature.
 */

/** Eight stations, both directions, on both sides of both walls. */
const TRAIN: Camera1D[] = [
  { x: 0.3, dir: 1 },
  { x: 1.5, dir: 1 },
  { x: 4.5, dir: -1 },
  { x: 4.6, dir: 1 },
  { x: 5.5, dir: 1 },
  { x: 6.2, dir: 1 },
  { x: 8.5, dir: -1 },
  { x: 9.7, dir: -1 },
];

/** Two the optimizer never sees. The honesty check. */
const HELD_OUT: Camera1D[] = [
  { x: 2.0, dir: 1 },
  { x: 8.0, dir: -1 },
];

const LR = 20;
const DOMAIN = { minX: -0.35, maxX: 10.35, minY: -0.02, maxY: 1.06 };
/** World-space layout: density above the rule, one ray per row below it. */
const RULE_Y = 0.38;
const ROW_TOP = 0.34;
const ROW_GAP = 0.031;

interface State {
  toy: VolumeToy1D;
  history: { iter: number; train: number; held: number }[];
}

export function PhotometricDescent() {
  const [noise, setNoise] = useState(0.05);

  const init = useCallback(
    (seed: number): State => {
      const toy = new VolumeToy1D({ seed, noise, train: TRAIN, heldOut: HELD_OUT });
      return {
        toy,
        history: [{ iter: 0, train: toy.residualRms(), held: toy.heldOutError() }],
      };
    },
    [noise],
  );

  const step = useCallback((s: State): State => {
    // One plain gradient step. Momentum would converge faster and hide exactly
    // the behaviour worth seeing: the regions where the renderer supplies no
    // gradient at all, because nothing gets through to them.
    s.toy.stepDescent(LR);
    const history = [
      ...s.history,
      { iter: s.toy.iteration, train: s.toy.residualRms(), held: s.toy.heldOutError() },
    ].slice(-260);
    return { toy: s.toy, history };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 20, initialSeed: 3, maxTicks: 240, loop: false });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { toy } = sim.state;
      const sigma = toy.density();
      const peak = Math.max(toy.wallDensity, 1e-6);
      const top = 0.98;

      // ---- ground truth: two walls, drawn where they actually are ----------
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (const w of toy.walls) {
        ctx.moveTo(sx(v, w), sy(v, 0.02));
        ctx.lineTo(sx(v, w), sy(v, top));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, RULE_Y));
      ctx.lineTo(sx(v, toy.length), sy(v, RULE_Y));
      ctx.stroke();
      ctx.restore();

      // ---- the estimate: σ(x), the thing gradient descent is moving --------
      const barW = Math.max(sl(v, toy.delta) - 0.5, 1);
      ctx.save();
      ctx.fillStyle = p.prediction;
      for (let i = 0; i < toy.bins; i++) {
        const frac = Math.min(sigma[i] / peak, 1);
        // sqrt keeps the low-density fog visible next to an opaque wall.
        const h = Math.sqrt(frac) * (top - RULE_Y);
        if (h <= 0) continue;
        ctx.globalAlpha = 0.3 + 0.6 * frac;
        ctx.fillRect(sx(v, i * toy.delta), sy(v, RULE_Y + h), barW, sl(v, h));
      }
      ctx.restore();

      // ---- the rays: measured depth (green) vs rendered depth (purple) -----
      const row = (k: number) => ROW_TOP - k * ROW_GAP;
      const ray = (
        cam: Camera1D,
        y: number,
        zObs: number,
        zHat: number | null,
        color: string,
        dashed: boolean,
      ) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.globalAlpha = dashed ? 0.7 : 0.9;
        if (dashed) ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, cam.x), sy(v, y));
        ctx.lineTo(sx(v, cam.x + cam.dir * zObs), sy(v, y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(sx(v, cam.x + cam.dir * zObs), sy(v, y), 2.2, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();

        if (zHat !== null) {
          ctx.save();
          ctx.strokeStyle = p.posterior;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(sx(v, cam.x + cam.dir * zHat), sy(v, y + 0.012));
          ctx.lineTo(sx(v, cam.x + cam.dir * zHat), sy(v, y - 0.012));
          ctx.stroke();
          ctx.restore();
        }

        // The camera itself: a small triangle pointing the way it looks.
        ctx.save();
        ctx.fillStyle = p.truth;
        const cx = sx(v, cam.x);
        const cy = sy(v, y);
        ctx.beginPath();
        ctx.moveTo(cx + cam.dir * 5.5, cy);
        ctx.lineTo(cx - cam.dir * 3.5, cy - 3.5);
        ctx.lineTo(cx - cam.dir * 3.5, cy + 3.5);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      };

      for (let k = 0; k < toy.train.length; k++) {
        ray(toy.train[k], row(k), toy.observations[k], toy.renderDepth(toy.train[k], sigma), p.measurement, false);
      }
      for (let k = 0; k < toy.heldOut.length; k++) {
        ray(
          toy.heldOut[k],
          row(toy.train.length + k),
          toy.heldOutTruth[k],
          toy.renderDepth(toy.heldOut[k], sigma),
          p.truth,
          true,
        );
      }

      label(ctx, 'σ(x) estimate', sx(v, 0.15), sy(v, top - 0.03), p.prediction, { size: 11, weight: 600 });
      label(ctx, 'true walls', sx(v, 3.15), sy(v, top - 0.03), p.truth, { size: 10 });
      label(
        ctx,
        `step ${toy.iteration} · green = fitted rays, gray dashed = held out, purple tick = rendered depth`,
        sx(v, toy.length),
        sy(v, top - 0.03),
        p.posterior,
        { size: 10, align: 'right' },
      );
    },
    [sim.state],
  );

  const series = useMemo(
    () => [
      {
        id: 'fitted rays',
        role: 'measurement' as const,
        data: sim.state.history.map((h) => ({ x: h.iter, y: h.train })),
      },
      {
        id: 'held-out rays',
        role: 'truth' as const,
        data: sim.state.history.map((h) => ({ x: h.iter, y: h.held })),
      },
    ],
    [sim.state.history],
  );

  const last = sim.state.history[sim.state.history.length - 1];
  const ambiguous = last.held > 2.5 * Math.max(last.train, 1e-3);

  return (
    <WidgetFrame
      id="w19.4"
      title="Photometric Descent"
      teaches="Radiance-field mapping is not magic: it is MAP estimation under a rendering measurement model, optimized by gradient descent."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Eight 1-D cameras report depths (green rays). A density field <em>σ(x)</em> (orange) renders
          its own predicted depths through the volume-rendering integral (purple ticks), and gradient
          descent pushes <em>σ</em> until the two agree — after a couple of hundred steps it has
          invented two walls at the right places, from nothing but depths. Two things to watch. The
          field sharpens only where a ray still has transmittance left: behind an opaque bin the
          gradient is exactly zero, which is why these optimizers refine geometry they roughly have
          and cannot invent geometry they do not. And the two gray dashed rays were never fitted —
          when their curve sits well above the green one, the field has explained its training rays
          with geometry that is not there. Re-roll to see how much the initialization decides.
        </>
      }
    >
      <SimCanvas
        world={DOMAIN}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={2.35}
        padding={0.05}
        ariaLabel="A one-dimensional density field being optimized so that its volume-rendered depths match ten measured depths; the estimate sharpens into two walls as the residual falls."
      />

      <div className="border-t border-fd-border px-3 py-3">
        <LineChart
          series={series}
          xLabel="gradient step"
          yLabel="RMS depth error (m)"
          height={190}
          yMin={0}
          caption={
            ambiguous
              ? 'The held-out rays are several times worse than the fitted ones: this run has found a density that explains its own data and nothing else.'
              : 'Both curves falling together is the honest outcome — the field explains rays it never saw. The floor is the depth noise itself.'
          }
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Depth noise σ_z"
          role="measurement"
          value={noise}
          min={0.01}
          max={0.4}
          step={0.01}
          unit="m"
          onChange={setNoise}
          help="Std-dev of the depth measurements. The residual can never fall below it — past that point the field is fitting noise, and the held-out curve says so."
        />
        <div className="font-ui text-[0.72rem] text-fd-muted-foreground">
          <p className="eyebrow mb-1">state</p>
          <p className="font-mono tabular-nums">
            fitted {last.train.toFixed(3)} m · held-out {last.held.toFixed(3)} m
          </p>
          <p className="mt-1">
            {ambiguous
              ? 'geometry–appearance ambiguity: the map is overfitted'
              : 'the map generalizes to unseen rays'}
          </p>
        </div>
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
