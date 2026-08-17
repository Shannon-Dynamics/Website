'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Kf, rtsSmoother, type KfRecord } from '@/lib/filters/kf';
import { rmse } from '@/lib/filters/consistency';
import { sampleMvn } from '@/lib/prob/gaussian';
import { Rng } from '@/lib/prob/rng';
import { matVec, type Mat, type Vec } from '@/lib/prob/linalg';
import { clear, label, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w6.4 — Smoother Rewind.
 *
 * The forward filter is the best *causal* estimate: at time t it has seen
 * z_{1:t} and nothing more. The RTS smoother goes back through the stored run
 * and gives every interior state the benefit of the future as well. The band
 * visibly contracts as the backward pass sweeps right to left — and the two
 * ends barely move, because the first state has no past to borrow from and the
 * last has no future.
 *
 * This is the chapter's longest-range foreshadowing: Chapter 15 computes the
 * same posterior in one sparse solve instead of two passes.
 */

const T = 150;
const DT = 0.1;
const ASPECT = 2.4;
const TRUE_ACCEL_SIGMA = 0.5;

const F: Mat = [
  [1, DT],
  [0, 1],
];
const H: Mat = [[1, 0]];

function motionNoise(sigma: number): Mat {
  const q = sigma * sigma;
  return [
    [(q * DT ** 3) / 3, (q * DT ** 2) / 2],
    [(q * DT ** 2) / 2, q * DT],
  ];
}

interface Run {
  truth: number[];
  z: number[];
  filtered: { mean: number; sd: number }[];
  smoothed: { mean: number; sd: number }[];
  filteredRmse: number;
  smoothedRmse: number;
  midRatio: number;
}

/**
 * One complete forward pass, stored, then smoothed. Both passes are the real
 * library code: `Kf.updateWith` and `rtsSmoother` from `lib/filters/kf.ts`.
 */
function buildRun(seed: number, measSigma: number): Run {
  const rng = new Rng(seed);
  const R = motionNoise(TRUE_ACCEL_SIGMA);
  const Q: Mat = [[measSigma * measSigma]];
  const kf = new Kf([0, 0], [
    [2, 0],
    [0, 2],
  ]);

  const truth: number[] = [];
  const z: number[] = [];
  const records: KfRecord[] = [];
  let x: Vec = [0, 0.9];

  for (let k = 0; k < T; k++) {
    x = matVec(F, x);
    const w = sampleMvn([0, 0], R, rng);
    x = [x[0] + w[0], x[1] + w[1]];
    const zk = x[0] + rng.normal(0, measSigma);

    kf.predictWith(F, R);
    const prior = kf.belief();
    kf.updateWith([zk], H, Q);
    const post = kf.belief();

    records.push({ xPrior: prior.x, PPrior: prior.P, xPost: post.x, PPost: post.P, F });
    truth.push(x[0]);
    z.push(zk);
  }

  const smooth = rtsSmoother(records);
  const filtered = records.map((r) => ({ mean: r.xPost[0], sd: Math.sqrt(r.PPost[0][0]) }));
  const smoothed = smooth.map((g) => ({ mean: g.x[0], sd: Math.sqrt(g.P[0][0]) }));
  const mid = Math.floor(T / 2);

  return {
    truth,
    z,
    filtered,
    smoothed,
    filteredRmse: rmse(filtered.map((f, i) => f.mean - truth[i])),
    smoothedRmse: rmse(smoothed.map((s, i) => s.mean - truth[i])),
    midRatio: smoothed[mid].sd / filtered[mid].sd,
  };
}

export function SmootherRewind() {
  const [measSigma, setMeasSigma] = useState(0.6);

  const init = useCallback((): { horizon: number } => ({ horizon: T }), []);
  const step = useCallback(
    (s: { horizon: number }): { horizon: number } => ({ horizon: Math.max(s.horizon - 1, 0) }),
    [],
  );
  const sim = useSimulation<{ horizon: number }>({
    init,
    step,
    fps: 26,
    maxTicks: T + 26, // a beat at the end, so the finished band can be read
    loop: true,
    initialSeed: 17,
  });

  const run = useMemo(() => buildRun(sim.seed, measSigma), [sim.seed, measSigma]);
  const horizon = Math.min(sim.state.horizon, T);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const px = (u: number) => sx(v, u * ASPECT);
      const py = (u: number) => sy(v, u);

      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < T; i++) {
        lo = Math.min(lo, run.truth[i], run.z[i], run.filtered[i].mean - 2 * run.filtered[i].sd);
        hi = Math.max(hi, run.truth[i], run.z[i], run.filtered[i].mean + 2 * run.filtered[i].sd);
      }
      const pad = 0.08 * (hi - lo);
      lo -= pad;
      hi += pad;

      const tx = (i: number) => px(0.03 + 0.94 * (i / (T - 1)));
      const ty = (m: number) => py(0.09 + 0.84 * ((m - lo) / (hi - lo)));

      const band = (
        series: { mean: number; sd: number }[],
        from: number,
        to: number,
        alpha: number,
      ) => {
        if (to - from < 2) return;
        ctx.save();
        ctx.fillStyle = p.posterior;
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        for (let i = from; i < to; i++) ctx.lineTo(tx(i), ty(series[i].mean + 2 * series[i].sd));
        for (let i = to - 1; i >= from; i--) ctx.lineTo(tx(i), ty(series[i].mean - 2 * series[i].sd));
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      };

      const line = (
        series: { mean: number; sd: number }[],
        from: number,
        to: number,
        width: number,
        dashed: boolean,
        alpha: number,
      ) => {
        if (to - from < 2) return;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (let i = from; i < to; i++) {
          const y = ty(series[i].mean);
          if (i === from) ctx.moveTo(tx(i), y);
          else ctx.lineTo(tx(i), y);
        }
        ctx.stroke();
        ctx.restore();
      };

      // Measurements first, so everything else sits on top of them.
      ctx.fillStyle = p.measurement;
      ctx.globalAlpha = 0.6;
      for (let i = 0; i < T; i++) {
        ctx.beginPath();
        ctx.arc(tx(i), ty(run.z[i]), 1.4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // The causal estimate, always shown, as the thing to beat.
      band(run.filtered, 0, T, 0.12);
      line(run.filtered, 0, T, 1.4, true, 0.7);

      // Whatever the backward pass has already reached.
      band(run.smoothed, horizon, T, 0.3);
      line(run.smoothed, horizon, T, 2.2, false, 1);

      // Truth.
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let i = 0; i < T; i++) {
        const y = ty(run.truth[i]);
        if (i === 0) ctx.moveTo(tx(i), y);
        else ctx.lineTo(tx(i), y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // The sweep front.
      if (horizon > 0 && horizon < T) {
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(tx(horizon), py(0.06));
        ctx.lineTo(tx(horizon), py(0.96));
        ctx.stroke();
        label(ctx, '◀ backward pass', tx(horizon) - 6, py(0.985), p.posterior, {
          size: 10,
          align: 'right',
          weight: 600,
        });
      }

      label(ctx, 'filtered — causal, z₁:ₜ', px(0.035), py(0.985), p.posterior, { size: 10 });
      label(
        ctx,
        horizon === 0 ? 'smoothed — all of z₁:T' : 'smoothed so far',
        px(0.035),
        py(0.94),
        p.posterior,
        { size: 10, weight: 700 },
      );
    },
    [run, horizon],
  );

  return (
    <WidgetFrame
      id="w6.4"
      title="Smoother Rewind"
      teaches="Filtering is the best causal use of the data, not the best use of it. Interior states do strictly better with hindsight."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          A completed run of the cart, filtered forward (thin dashed line, pale band) and then
          smoothed backward. The vertical rule is the RTS backward pass; everything to its right has
          already been re-estimated with the future admitted, and the band there is visibly tighter.
          Two details are worth pausing on. The <em>last</em> state barely improves — it has no
          future to borrow from, so smoothed equals filtered at t&nbsp;=&nbsp;T. And the improvement
          is largest exactly where the filter was least sure. Drag the horizon by hand to compare
          any single instant, and raise the sensor noise to see the gap widen: the worse each
          individual measurement is, the more there is to gain from reading them all together.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: ASPECT, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[horizon, run]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="A time series of the cart's position: gray dashed truth, green measurement dots, a pale filtered confidence band across the whole run, and a tighter smoothed band covering the part of the run the backward pass has already reached."
      />

      <div className="grid grid-cols-1 gap-2 border-t border-fd-border p-3 sm:grid-cols-3">
        <StatTile label="filtered RMSE" value={run.filteredRmse} unit="m" precision={3} role="posterior" />
        <StatTile
          label="smoothed RMSE"
          value={run.smoothedRmse}
          unit="m"
          precision={3}
          role="posterior"
          trend={run.smoothedRmse - run.filteredRmse}
          trendLabel="vs. filtered"
        />
        <StatTile
          label="σ ratio at mid-trajectory (smoothed ÷ filtered)"
          value={run.midRatio}
          precision={3}
          role="truth"
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="measurement noise σ_z"
          role="measurement"
          value={measSigma}
          min={0.1}
          max={2}
          step={0.05}
          unit="m"
          onChange={setMeasSigma}
          help="Worse sensors mean more to gain from hindsight — watch the RMSE gap widen."
        />
        <Slider
          label="knowledge horizon (drag to scrub)"
          role="posterior"
          value={horizon}
          min={0}
          max={T}
          step={1}
          format={(x) => `t = ${x.toFixed(0)}`}
          onChange={(x) => {
            sim.pause();
            sim.setState(() => ({ horizon: Math.round(x) }));
          }}
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
