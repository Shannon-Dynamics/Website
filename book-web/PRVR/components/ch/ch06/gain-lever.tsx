'use client';

import { useCallback, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { gaussianProduct } from '@/lib/prob/gaussian';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w6.3 — the Gain Lever.
 *
 * The scalar Kalman update, drawn as the physical fact it is. Put a weight of
 * mass ω̄ = 1/σ̄² at the prediction and a weight of mass ω_z = 1/σ_z² at the
 * measurement; the posterior mean is the centre of mass of that beam. The
 * fulcrum slides towards whichever blob is *tighter*, and it does so by an
 * amount nobody chose — the algebra of Ch. 6 fixes it exactly.
 *
 * The misconception this kills: "the Kalman filter averages the prediction and
 * the measurement". It does not. It precision-weights them.
 */

const ASPECT = 2.3;
const X_MIN = -5;
const X_MAX = 5;

interface Params {
  predSigma: number;
  measSigma: number;
}

interface State {
  rng: Rng;
  /** Where the prediction thinks the cart is. */
  predMean: number;
  /** The reading that just came in. */
  z: number;
}

export function GainLever() {
  const [params, setParams] = useState<Params>({ predSigma: 1.1, measSigma: 0.5 });

  const init = useCallback(
    (seed: number): State => ({ rng: new Rng(seed), predMean: -1.4, z: 1.6 }),
    [],
  );

  const step = useCallback((s: State, tick: number): State => {
    // A slow, seeded wander so the lever is always in motion and the reader can
    // see the fulcrum tracking, not just a static picture.
    const phase = tick * 0.06;
    return {
      rng: s.rng,
      predMean: -1.4 + 0.8 * Math.sin(phase),
      z: 1.6 + 1.1 * Math.sin(phase * 0.63 + 1.1) + 0.15 * s.rng.normal(),
    };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 18, initialSeed: 23 });

  const { predMean, z } = sim.state;
  const vPred = params.predSigma * params.predSigma;
  const vMeas = params.measSigma * params.measSigma;
  const post = gaussianProduct(predMean, vPred, z, vMeas);
  const K = vPred / (vPred + vMeas);
  const omegaPred = 1 / vPred;
  const omegaMeas = 1 / vMeas;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const px = (u: number) => sx(v, u * ASPECT);
      const py = (u: number) => sy(v, u);
      /** World position → normalized canvas x. */
      const wx = (m: number) => px(0.04 + 0.92 * ((m - X_MIN) / (X_MAX - X_MIN)));

      const curveTop = 0.94;
      const curveBase = 0.42;
      const peak = Math.max(1 / params.predSigma, 1 / params.measSigma, 1 / Math.sqrt(post.variance));

      const bell = (mean: number, sd: number, color: string, dashed: boolean, fill: boolean) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (let i = 0; i <= 200; i++) {
          const m = X_MIN + (i / 200) * (X_MAX - X_MIN);
          const d = (m - mean) / sd;
          const h = Math.exp(-0.5 * d * d) / sd / peak;
          const y = py(curveBase + h * (curveTop - curveBase));
          if (i === 0) ctx.moveTo(wx(m), y);
          else ctx.lineTo(wx(m), y);
        }
        ctx.stroke();
        if (fill) {
          ctx.lineTo(wx(X_MAX), py(curveBase));
          ctx.lineTo(wx(X_MIN), py(curveBase));
          ctx.closePath();
          ctx.globalAlpha = 0.12;
          ctx.fillStyle = color;
          ctx.fill();
        }
        ctx.restore();
      };

      bell(predMean, params.predSigma, p.prediction, false, false);
      bell(z, params.measSigma, p.measurement, true, false);
      bell(post.mean, Math.sqrt(post.variance), p.posterior, false, true);

      // Baseline of the density panel.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(wx(X_MIN), py(curveBase));
      ctx.lineTo(wx(X_MAX), py(curveBase));
      ctx.stroke();

      /* ---- the lever --------------------------------------------------- */
      const beamY = 0.24;
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(wx(predMean), py(beamY));
      ctx.lineTo(wx(z), py(beamY));
      ctx.stroke();

      // Weights: area proportional to precision, so a tighter blob is heavier.
      const rMax = 26;
      const wMax = Math.max(omegaPred, omegaMeas);
      const weight = (m: number, omega: number, color: string, text: string) => {
        const r = rMax * Math.sqrt(omega / wMax);
        ctx.save();
        ctx.globalAlpha = 0.2;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(wx(m), py(beamY) - r * 0.55, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
        label(ctx, text, wx(m), py(beamY) - r * 0.55, color, { size: 10, align: 'center', weight: 600 });
      };
      weight(predMean, omegaPred, p.prediction, `ω̄=${omegaPred.toFixed(2)}`);
      weight(z, omegaMeas, p.measurement, `ω_z=${omegaMeas.toFixed(2)}`);

      // The fulcrum: the centre of mass, which is the posterior mean.
      const fx = wx(post.mean);
      const fy = py(beamY);
      ctx.fillStyle = p.posterior;
      ctx.beginPath();
      ctx.moveTo(fx, fy);
      ctx.lineTo(fx - 9, fy + 16);
      ctx.lineTo(fx + 9, fy + 16);
      ctx.closePath();
      ctx.fill();
      label(ctx, `μ = ${post.mean.toFixed(2)}`, fx, fy + 27, p.posterior, {
        size: 10,
        align: 'center',
        weight: 600,
      });

      // The gain, read off the beam as a fraction of the way from μ̄ to z.
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(fx, py(curveBase));
      ctx.lineTo(fx, fy);
      ctx.stroke();
      ctx.setLineDash([]);

      label(ctx, `K = ${K.toFixed(3)} of the way to z`, px(0.04), py(0.06), p.ink, { size: 11 });
      label(ctx, 'prediction', wx(predMean), py(curveTop + 0.03), p.prediction, {
        size: 10,
        align: 'center',
      });
      label(ctx, 'measurement', wx(z), py(curveTop + 0.03), p.measurement, {
        size: 10,
        align: 'center',
      });
    },
    [params.predSigma, params.measSigma, predMean, z, post.mean, post.variance, K, omegaPred, omegaMeas],
  );

  return (
    <WidgetFrame
      id="w6.3"
      title="The Gain Lever"
      teaches="The Kalman filter does not average the prediction and the measurement — it balances them, and the balance point is derived, not tuned."
      colorKey={['prediction', 'measurement', 'posterior']}
      caption={
        <>
          The beam runs from the prediction to the measurement. Hang a weight of mass
          ω̄&nbsp;=&nbsp;1/σ̄² at one end and ω<sub>z</sub>&nbsp;=&nbsp;1/σ<sub>z</sub>² at the other:
          the purple fulcrum sits at their centre of mass, and that point is exactly
          μ&nbsp;=&nbsp;(1−K)μ̄&nbsp;+&nbsp;Kz. Narrow the measurement and its weight grows; the
          fulcrum slides towards it and K rises towards 1. Two things are worth checking by eye.
          The posterior curve is always <em>taller</em> than both parents — precisions add, so the
          answer is more certain than either input. And the fulcrum never leaves the segment
          between the two means, which is why a Kalman filter can be wrong but can never be wild.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: ASPECT, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, params.predSigma, params.measSigma]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="Two bell curves, an orange prediction and a green measurement, with a purple posterior between them; below, a beam carrying a weight at each mean and a fulcrum at the posterior mean."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Readout label="ω̄ = 1/σ̄²" value={omegaPred.toFixed(3)} role="prediction" />
        <Readout label="ω_z = 1/σ_z²" value={omegaMeas.toFixed(3)} role="measurement" />
        <Readout label="ω̄ + ω_z" value={(omegaPred + omegaMeas).toFixed(3)} role="posterior" />
        <Readout label="1/σ²  (check)" value={(1 / post.variance).toFixed(3)} role="posterior" />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="measurement σ_z"
          role="measurement"
          value={params.measSigma}
          min={0.15}
          max={2.5}
          step={0.05}
          unit="m"
          onChange={(m) => setParams((p) => ({ ...p, measSigma: m }))}
          help="Squeeze the green blob and watch the fulcrum come to meet it."
        />
        <Slider
          label="prediction σ̄"
          role="prediction"
          value={params.predSigma}
          min={0.15}
          max={2.5}
          step={0.05}
          unit="m"
          onChange={(m) => setParams((p) => ({ ...p, predSigma: m }))}
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
      />
    </WidgetFrame>
  );
}

function Readout({
  label: l,
  value,
  role,
}: {
  label: string;
  value: string;
  role: 'prediction' | 'measurement' | 'posterior';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow" style={{ color: `var(--pr-${role})` }}>
        {l}
      </div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
