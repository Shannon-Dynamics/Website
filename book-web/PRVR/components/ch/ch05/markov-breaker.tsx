'use client';

import { useCallback, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w5.2 — the Markov Breaker.
 *
 * A filter that assumes independent measurement noise, fed measurements whose
 * noise is a slow random walk instead. The belief keeps sharpening — it is
 * counting the same error over and over as if it were fresh evidence — while
 * the truth strolls out of the confidence band. Overconfidence is what a
 * violated Markov assumption looks like from the outside.
 */

const HORIZON = 160;

interface Sample {
  truth: number;
  mean: number;
  sigma: number;
  inside: boolean;
}

interface State {
  rng: Rng;
  truth: number;
  bias: number;
  mean: number;
  variance: number;
  samples: Sample[];
  covered: number;
}

interface Params {
  correlated: boolean;
  sensorSigma: number;
  driftRate: number;
}

export function MarkovBreaker() {
  const [params, setParams] = useState<Params>({
    correlated: true,
    sensorSigma: 0.35,
    driftRate: 0.06,
  });

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      truth: 0,
      bias: 0,
      mean: 0,
      // Start genuinely uncertain, so the collapse is visible.
      variance: 4,
      samples: [],
      covered: 0,
    }),
    [],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      const { rng } = s;

      // A stationary target keeps the story about the measurements, not motion.
      const truth = s.truth;

      // The only difference between the two modes: where the error comes from.
      // White noise is redrawn every step. A drifting bias persists, so
      // consecutive readings share it — and the filter has no way to know.
      const bias = params.correlated
        ? s.bias + rng.normal(0, params.driftRate)
        : 0;
      const z = truth + bias + rng.normal(0, params.sensorSigma * (params.correlated ? 0.35 : 1));

      // A textbook 1-D Kalman correction. The filter believes every reading is
      // fresh, independent evidence with variance sensorSigma².
      const R = params.sensorSigma * params.sensorSigma;
      const K = s.variance / (s.variance + R);
      const mean = s.mean + K * (z - s.mean);
      const variance = (1 - K) * s.variance;

      const sigma = Math.sqrt(variance);
      const inside = Math.abs(truth - mean) <= 2 * sigma;
      const samples = [...s.samples, { truth, mean, sigma, inside }].slice(-HORIZON);
      const covered = samples.filter((p) => p.inside).length / samples.length;

      return { rng, truth, bias, mean, variance, samples, covered };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 12, maxTicks: HORIZON, loop: true, initialSeed: 3 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { samples } = sim.state;
      if (samples.length < 2) return;

      const xAt = (i: number) => sx(v, (i / (HORIZON - 1)) * 10);

      // Zero line — the truth, which never moves in this experiment.
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, 0));
      ctx.lineTo(sx(v, 10), sy(v, 0));
      ctx.stroke();
      ctx.setLineDash([]);
      label(ctx, 'true position', sx(v, 0.15), sy(v, 0) - 10, p.truth, { size: 10 });

      // The ±2σ band the filter claims.
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.16;
      ctx.beginPath();
      samples.forEach((s, i) => {
        const y = sy(v, s.mean + 2 * s.sigma);
        if (i === 0) ctx.moveTo(xAt(i), y);
        else ctx.lineTo(xAt(i), y);
      });
      for (let i = samples.length - 1; i >= 0; i--) {
        ctx.lineTo(xAt(i), sy(v, samples[i].mean - 2 * samples[i].sigma));
      }
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      // The belief mean.
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 2;
      ctx.beginPath();
      samples.forEach((s, i) => {
        const y = sy(v, s.mean);
        if (i === 0) ctx.moveTo(xAt(i), y);
        else ctx.lineTo(xAt(i), y);
      });
      ctx.stroke();

      // Mark the moments the filter is provably wrong: truth outside its own band.
      ctx.fillStyle = p.prediction;
      samples.forEach((s, i) => {
        if (s.inside) return;
        ctx.beginPath();
        ctx.arc(xAt(i), sy(v, 0), 2.2, 0, Math.PI * 2);
        ctx.fill();
      });
    },
    [sim.state],
  );

  const covered = sim.state.samples.length > 8 ? sim.state.covered : null;
  const sigma = Math.sqrt(sim.state.variance);

  return (
    <WidgetFrame
      id="w5.2"
      title="The Markov Breaker"
      teaches="When errors are correlated, a Bayes filter does not become wrong — it becomes overconfident, which is worse."
      colorKey={['posterior', 'truth']}
      caption={
        <>
          A stationary target, measured repeatedly. The purple band is the ±2σ interval the filter
          claims; the dashed gray line is the truth. With <strong>correlated errors off</strong>,
          the noise is redrawn every step, the assumption holds, and the truth sits inside the band
          about 95% of the time — as it should. Turn correlation <strong>on</strong> and the sensor
          error becomes a slow drift shared between consecutive readings. The filter cannot tell
          the difference, so it keeps treating each reading as fresh evidence, and its variance
          keeps shrinking. Coverage collapses. Note that the total error magnitude is the same in
          both modes — only its <em>independence</em> changed.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: 10, minY: -3, maxY: 3 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={2.6}
        padding={0.1}
        ariaLabel="A time series showing a filter's confidence band shrinking around its estimate while the true value drifts outside the band."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <div className="px-2 py-1.5">
          <div className="eyebrow">claimed σ</div>
          <div className="font-mono text-sm tabular-nums">{sigma.toFixed(3)}</div>
        </div>
        <div className="px-2 py-1.5">
          <div className="eyebrow">|error|</div>
          <div className="font-mono text-sm tabular-nums">
            {Math.abs(sim.state.mean - sim.state.truth).toFixed(3)}
          </div>
        </div>
        <div className="px-2 py-1.5">
          <div className="eyebrow">2σ coverage</div>
          <div
            className="font-mono text-sm tabular-nums"
            style={{
              color:
                covered !== null && covered < 0.8 ? 'var(--pr-prediction)' : undefined,
            }}
          >
            {covered === null ? '—' : `${(covered * 100).toFixed(0)}%`}
            <span className="ml-1 text-[0.65rem] opacity-60">of 95%</span>
          </div>
        </div>
      </div>

      <ControlPanel columns={3}>
        <Toggle
          label="Correlated errors"
          checked={params.correlated}
          onChange={(v) => setParams((p) => ({ ...p, correlated: v }))}
        />
        <Slider
          label="Assumed sensor σ"
          role="measurement"
          value={params.sensorSigma}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(v) => setParams((p) => ({ ...p, sensorSigma: v }))}
          help="What the filter believes the noise is. Inflating this is the second Markov repair."
        />
        <Slider
          label="Drift rate"
          value={params.driftRate}
          min={0.01}
          max={0.2}
          step={0.01}
          onChange={(v) => setParams((p) => ({ ...p, driftRate: v }))}
          help="How fast the shared bias wanders. Zero would restore independence."
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
