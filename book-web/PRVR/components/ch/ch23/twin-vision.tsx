'use client';

import Link from 'next/link';
import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { LineChart, useBookColors } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { normalPdf } from '@/lib/prob/gaussian';
import { lowVarianceResample, normalizeWeights, type Particle } from '@/lib/filters/pf';
import {
  Mppi,
  RUSTY_LIMITS,
  control2,
  effectiveSampleSize,
  rollout,
  type CostModel,
  type MppiResult,
} from '@/lib/control/mppi';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w23.3 — Twin Vision.
 *
 * The same estimator, twice. On the left, Chapter 8's particle filter asks
 * *where am I?*: hypotheses are positions, the weight of a hypothesis is the
 * measurement likelihood, and the answer is the weighted mean. On the right,
 * MPPI asks *what should I do?*: hypotheses are control sequences, the weight
 * of a hypothesis is exp(−S/λ), and the answer is the weighted mean.
 *
 * Both panels are literally the same picture — a scalar axis, a bundle of
 * weighted samples, a green weight profile, a purple mean — because both are
 * self-normalized importance sampling. The particle filter runs the real
 * `lowVarianceResample` from `lib/filters/pf.ts`; MPPI runs the real `Mppi`
 * from `lib/control/mppi.ts`, restricted to one dimension.
 */

const M = 60; // particles, and rollouts: the same number on purpose
const H = 12;
const DT = 0.25;
const LANDMARK = 7.5;
const SENSOR_SIGMA = 0.55;
const GOAL = 5.6;

type Phase = 'sample' | 'weight' | 'estimate' | 'renew';
const PHASES: Phase[] = ['sample', 'weight', 'estimate', 'renew'];

const PHASE_ROWS: { phase: Phase; pf: string; mppi: string; why: string }[] = [
  {
    phase: 'sample',
    pf: 'draw xᵢ ~ p(x | u, x⁻)',
    mppi: 'draw Uᵢ = U + εᵢ,  εᵢ ~ 𝒩(0, Σᵤ)',
    why: 'A proposal you can sample from, centred on what you already believe.',
  },
  {
    phase: 'weight',
    pf: 'wᵢ ∝ p(z | xᵢ)',
    mppi: 'wᵢ ∝ exp(−S(Uᵢ)/λ)',
    why: 'The likelihood ratio to the target. A cost is a negative log-likelihood in disguise.',
  },
  {
    phase: 'estimate',
    pf: 'x̂ = Σ wᵢ xᵢ',
    mppi: 'U ← U + Σ wᵢ εᵢ',
    why: 'The self-normalized estimate of an expectation under the target.',
  },
  {
    phase: 'renew',
    pf: 'resample, then predict forward',
    mppi: 'execute u₀, shift the plan, re-centre',
    why: 'Move the proposal to where the mass went, so next cycle starts warm.',
  },
];

interface Cycle {
  /** Particle positions before and after resampling. */
  particles: Particle[];
  resampled: Particle[];
  weightsPf: number[];
  pfPrior: number;
  pfMean: number;
  pfEss: number;
  z: number;
  mppi: MppiResult;
  /** exp(−(S(u₀) − ρ)/λ) evaluated on a grid of first controls. */
  profile: { u: number; w: number }[];
}

interface State {
  rng: Rng;
  planner: Mppi;
  particles: Particle[];
  truth: number;
  phase: Phase;
  cycle: Cycle | null;
  cycles: number;
}

/** One-dimensional cost: get to GOAL, do not thrash the motor. */
const oneDCost = (): CostModel => ({
  stage: (x, u) => (x.x - GOAL) * (x.x - GOAL) + 0.6 * u.v * u.v,
  terminal: (x) => 4 * (x.x - GOAL) * (x.x - GOAL),
});

export function TwinVision() {
  const [lambda, setLambda] = useState(4);
  const bookColors = useBookColors();

  const init = useCallback((seed: number): State => {
    const rng = new Rng(seed);
    const particles: Particle[] = Array.from({ length: M }, () => ({
      state: { x: 5 + rng.normal(0, 1.1), y: 0, theta: 0 },
      weight: 1 / M,
    }));
    const planner = new Mppi({
      horizon: H,
      samples: M,
      dt: DT,
      lambda: 4,
      sigmaV: 0.16,
      // Not zero: 1/σ_ω² appears in the cross-term algebra, and ∞ · 0 is NaN.
      sigmaOmega: 1e-4,
      limits: RUSTY_LIMITS,
      gamma: 0,
      smoothWindow: 0,
    });
    planner.reset(control2(0.25, 0));
    return { rng, planner, particles, truth: 6.3, phase: 'sample', cycle: null, cycles: 0 };
  }, []);

  const step = useCallback(
    (s: State): State => {
      const idx = PHASES.indexOf(s.phase);
      // Phases 1–3 only reveal more of the cycle that phase 0 computed.
      if (s.cycle && idx < PHASES.length - 1) return { ...s, phase: PHASES[idx + 1] };

      const { rng } = s;
      s.planner.cfg.lambda = lambda;

      /* ---- the particle filter: predict, weight, mean, resample --------- */
      const moved: Particle[] = (s.cycle ? s.cycle.resampled : s.particles).map((p) => ({
        state: { x: p.state.x + 0.22 + rng.normal(0, 0.18), y: 0, theta: 0 },
        weight: 1 / M,
      }));
      const pfPrior = moved.reduce((a, p) => a + p.state.x, 0) / M;
      const truth = s.truth + 0.22;
      const z = Math.abs(truth - LANDMARK) + rng.normal(0, SENSOR_SIGMA);
      for (const p of moved) p.weight = normalPdf(z, Math.abs(p.state.x - LANDMARK), SENSOR_SIGMA);
      normalizeWeights(moved);
      const weightsPf = moved.map((p) => p.weight);
      const pfMean = moved.reduce((a, p) => a + p.weight * p.state.x, 0);
      const resampled = lowVarianceResample(moved, rng);

      /* ---- MPPI: sample, cost, weighted update, shift ------------------- */
      const cost = oneDCost();
      const result = s.planner.plan({ x: 4.1, y: 0, theta: 0 }, cost, rng);

      // The weight profile: how exp(−S/λ) varies with the first command alone.
      const profile: { u: number; w: number }[] = [];
      const grid: number[] = [];
      for (let i = 0; i <= 48; i++) {
        const u0 = RUSTY_LIMITS.vMin + (i * (RUSTY_LIMITS.vMax - RUSTY_LIMITS.vMin)) / 48;
        const plan = result.previous.map((u, k) => (k === 0 ? control2(u0, u.omega) : u));
        grid.push(rollout({ x: 4.1, y: 0, theta: 0 }, plan, cost, DT, RUSTY_LIMITS).cost);
        profile.push({ u: u0, w: 0 });
      }
      const gridMin = Math.min(...grid);
      for (let i = 0; i < profile.length; i++) profile[i].w = Math.exp(-(grid[i] - gridMin) / lambda);

      return {
        ...s,
        truth,
        phase: 'sample',
        cycles: s.cycles + 1,
        particles: moved,
        cycle: {
          particles: moved,
          resampled,
          weightsPf,
          pfPrior,
          pfMean,
          pfEss: effectiveSampleSize(weightsPf),
          z,
          mppi: result,
          profile,
        },
      };
    },
    [lambda],
  );

  const sim = useSimulation<State>({ init, step, fps: 1.1, initialSeed: 8 });
  const c = sim.state.cycle;
  const phase = sim.state.phase;
  const shown = PHASES.indexOf(phase);

  /* ------------------------------------------------------------- panels --- */
  const drawBundle = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      v: Viewport,
      p: Palette,
      opts: {
        samples: { at: number; w: number }[];
        profile: { at: number; w: number }[] | null;
        prior: number;
        mean: number;
        renewed: number[] | null;
        title: string;
        axis: string;
        stickColor: string;
      },
    ) => {
      clear(ctx, v, p);
      const base = sy(v, 0);
      const top = sy(v, 1);
      const wMax = Math.max(...opts.samples.map((s) => s.w), 1e-12);

      ctx.save();
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, v.minX), base);
      ctx.lineTo(sx(v, v.maxX), base);
      ctx.stroke();
      ctx.restore();

      // Phase 3: the renewed proposal, drawn as dots on the baseline.
      if (opts.renewed) {
        ctx.save();
        ctx.fillStyle = p.prior;
        ctx.globalAlpha = 0.75;
        for (const x of opts.renewed) {
          ctx.beginPath();
          ctx.arc(sx(v, x), base + 7, 2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // The weight profile — the green curve both methods are sampling against.
      if (opts.profile) {
        ctx.save();
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 1.8;
        ctx.setLineDash([4, 3]);
        const pMax = Math.max(...opts.profile.map((q) => q.w), 1e-12);
        ctx.beginPath();
        opts.profile.forEach((q, i) => {
          const y = base - (q.w / pMax) * (base - top) * 0.92;
          if (i === 0) ctx.moveTo(sx(v, q.at), y);
          else ctx.lineTo(sx(v, q.at), y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }

      // The hypotheses themselves: one stick each, height ∝ weight.
      ctx.save();
      ctx.lineWidth = 1.6;
      for (const s of opts.samples) {
        const h = opts.profile ? (s.w / wMax) * (base - top) * 0.92 : 10;
        ctx.strokeStyle = opts.profile ? opts.stickColor : p.prior;
        ctx.globalAlpha = opts.profile ? 0.35 + 0.6 * (s.w / wMax) : 0.5;
        ctx.beginPath();
        ctx.moveTo(sx(v, s.at), base);
        ctx.lineTo(sx(v, s.at), base - h);
        ctx.stroke();
      }
      ctx.restore();

      // Prior mean (blue) and the estimate (purple).
      const rule = (at: number, color: string, text: string) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx(v, at), base + 4);
        ctx.lineTo(sx(v, at), top);
        ctx.stroke();
        label(ctx, text, sx(v, at) + 4, top + 8, color, { size: 9.5, weight: 600 });
        ctx.restore();
      };
      rule(opts.prior, p.prior, 'prior');
      if (opts.mean !== opts.prior && shown >= 2) rule(opts.mean, p.posterior, 'estimate');

      label(ctx, opts.title, 8, 12, p.ink, { size: 10.5, weight: 600 });
      label(ctx, opts.axis, v.width - 8, base + 18, p.ink, { size: 9, align: 'right' });
    },
    [shown],
  );

  const drawPf = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      if (!c) {
        clear(ctx, v, p);
        return;
      }
      const grid: { at: number; w: number }[] = [];
      for (let i = 0; i <= 80; i++) {
        const x = v.minX + (i * (v.maxX - v.minX)) / 80;
        grid.push({ at: x, w: normalPdf(c.z, Math.abs(x - LANDMARK), SENSOR_SIGMA) });
      }
      drawBundle(ctx, v, p, {
        samples: c.particles.map((q) => ({ at: q.state.x, w: q.weight })),
        profile: shown >= 1 ? grid : null,
        prior: c.pfPrior,
        mean: c.pfMean,
        renewed: shown >= 3 ? c.resampled.map((q) => q.state.x) : null,
        title: 'Particle filter — where am I?',
        axis: 'position x  (m)',
        stickColor: p.measurement,
      });
      if (shown >= 1) {
        ctx.save();
        ctx.fillStyle = p.measurement;
        ctx.beginPath();
        ctx.arc(sx(v, LANDMARK), sy(v, 0), 3.5, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, `z = ${c.z.toFixed(2)} m`, sx(v, LANDMARK) + 6, sy(v, 0) - 12, p.measurement, { size: 9.5 });
        ctx.restore();
      }
    },
    [c, shown, drawBundle],
  );

  const drawMppi = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      if (!c) {
        clear(ctx, v, p);
        return;
      }
      drawBundle(ctx, v, p, {
        samples: c.mppi.samples.map((s) => ({
          at: c.mppi.previous[0].v + s.eps[0].v,
          w: s.weight,
        })),
        profile: shown >= 1 ? c.profile.map((q) => ({ at: q.u, w: q.w })) : null,
        prior: c.mppi.previous[0].v,
        mean: c.mppi.applied.v,
        renewed: shown >= 3 ? [c.mppi.updated[1]?.v ?? c.mppi.applied.v] : null,
        title: 'MPPI — what should I do?',
        axis: 'first command u₀  (m/s)',
        stickColor: p.prediction,
      });
      if (shown >= 3) {
        ctx.save();
        label(ctx, 'plan shifted: u₁ becomes the new u₀', 8, sy(v, 0) + 22, p.prior, { size: 9.5 });
        ctx.restore();
      }
      // A reminder that each stick is the head of a whole sequence.
      if (shown >= 1) {
        ctx.save();
        ctx.strokeStyle = p.prediction;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1;
        const x0 = sx(v, v.maxX - 0.16);
        const y0 = sy(v, 0.86);
        for (let i = 0; i < 8; i++) {
          ctx.beginPath();
          ctx.moveTo(x0 - sl(v, 0.28), y0);
          ctx.quadraticCurveTo(x0 - sl(v, 0.14), y0 - 10 + 2.5 * i, x0, y0 - 18 + 4.6 * i);
          ctx.stroke();
        }
        ctx.restore();
        label(ctx, 'each stick = one H-step sequence', sx(v, v.maxX - 0.2), sy(v, 0.62), p.prediction, {
          size: 9,
          align: 'right',
        });
      }
    },
    [c, shown, drawBundle],
  );

  /* -------------------------------------------------------------- charts --- */
  const weightSeries = useMemo(() => {
    if (!c) return [];
    const sortDesc = (xs: number[]) => [...xs].sort((a, b) => b - a);
    const pf = sortDesc(c.weightsPf);
    const mp = sortDesc(c.mppi.samples.map((s) => s.weight));
    return [
      { id: 'particle weights  p(z | xᵢ)', data: pf.map((y, i) => ({ x: i + 1, y })), color: bookColors.measurement },
      { id: 'rollout weights  exp(−S/λ)', data: mp.map((y, i) => ({ x: i + 1, y })), color: bookColors.prediction },
    ];
  }, [c, bookColors]);

  return (
    <WidgetFrame
      id="w23.3"
      title="Twin Vision"
      teaches="MPPI is not a new algorithm. It is the particle filter's importance-sampling step, aimed at the future."
      colorKey={['prior', 'prediction', 'measurement', 'posterior']}
      caption={
        <>
          Two panels, one estimator. Left: sixty particles over the robot&apos;s position, weighted by
          the green measurement likelihood, collapsing to a purple posterior mean. Right: sixty
          sampled control sequences, weighted by the green profile exp(−S/λ), collapsing to a purple
          command. The cycle steps through the four sub-operations in lockstep — sample, weight,
          estimate, renew — and the table names the correspondence at each one.
          <br />
          <strong>What to notice.</strong> The sorted-weight curves below have the same shape,
          because they are the same object: self-normalized importance weights. Drag λ down and the
          orange curve becomes a spike while the effective sample size falls toward 1 — the identical
          degeneracy that makes a particle filter collapse in{' '}
          <Link href="/chapters/ch08-nonparametric-filters">Chapter 8</Link>, arriving here as jitter in
          the executed command rather than as a lost robot.
        </>
      }
    >
      <div className="grid grid-cols-1 gap-0 md:grid-cols-2 md:divide-x md:divide-fd-border">
        <SimCanvas
          world={{ minX: 2.2, maxX: 9.8, minY: -0.1, maxY: 1.12 }}
          draw={drawPf}
          deps={[sim.tick, sim.state, shown]}
          aspect={1.6}
          padding={0}
          ariaLabel="Particle filter panel: sixty vertical sticks along a position axis, their heights set by a green measurement likelihood curve, with a purple line at the weighted mean."
        />
        <SimCanvas
          world={{ minX: RUSTY_LIMITS.vMin - 0.05, maxX: RUSTY_LIMITS.vMax + 0.05, minY: -0.1, maxY: 1.12 }}
          draw={drawMppi}
          deps={[sim.tick, sim.state, shown]}
          aspect={1.6}
          padding={0}
          ariaLabel="MPPI panel: sixty vertical sticks along a command axis, their heights set by a green exponentiated-cost curve, with a purple line at the weighted mean command."
        />
      </div>

      <div className="border-t border-fd-border">
        <table className="w-full border-collapse text-[0.78rem]">
          <thead>
            <tr className="border-b border-fd-border">
              <th className="px-3 py-1.5 text-start font-ui text-[0.68rem] font-medium tracking-wide text-fd-muted-foreground uppercase">
                sub-step
              </th>
              <th className="px-3 py-1.5 text-start font-ui text-[0.68rem] font-medium tracking-wide text-fd-muted-foreground uppercase">
                particle filter
              </th>
              <th className="px-3 py-1.5 text-start font-ui text-[0.68rem] font-medium tracking-wide text-fd-muted-foreground uppercase">
                MPPI
              </th>
            </tr>
          </thead>
          <tbody>
            {PHASE_ROWS.map((row) => {
              const active = row.phase === phase;
              return (
                <tr
                  key={row.phase}
                  className="border-b border-fd-border/60 last:border-b-0"
                  style={active ? { background: 'color-mix(in oklab, var(--pr-posterior) 10%, transparent)' } : undefined}
                >
                  <td className="px-3 py-1.5 font-ui text-[0.75rem]" style={active ? { color: 'var(--pr-posterior)', fontWeight: 600 } : undefined}>
                    {row.phase}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-[0.72rem]">{row.pf}</td>
                  <td className="px-3 py-1.5 font-mono text-[0.72rem]">{row.mppi}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="px-3 py-2 font-ui text-[0.75rem] text-fd-muted-foreground">
          {PHASE_ROWS.find((r) => r.phase === phase)?.why}
        </p>
      </div>

      <div className="border-t border-fd-border px-3 py-3">
        <LineChart
          series={weightSeries.map(({ id, data }) => ({ id, data }))}
          colors={weightSeries.map((s) => s.color)}
          xLabel="rank"
          yLabel="normalized weight"
          height={190}
          curve="monotoneX"
          caption={
            c
              ? `ESS: ${c.pfEss.toFixed(1)} of ${M} particles, ${c.mppi.ess.toFixed(1)} of ${M} rollouts.`
              : undefined
          }
          ariaLabel="Sorted normalized weights for the particle filter and for MPPI, both decaying from a small number of dominant samples."
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Temperature λ"
          role="posterior"
          value={lambda}
          min={0.2}
          max={40}
          step={0.2}
          onChange={setLambda}
          help="Only the right-hand panel has one. A particle filter's likelihood has no temperature — which is exactly the freedom a control problem gives you and an estimation problem does not."
        />
        <div className="grid grid-cols-2 gap-2 text-center">
          <div>
            <div className="eyebrow">cycles</div>
            <div className="font-mono text-sm tabular-nums">{sim.state.cycles}</div>
          </div>
          <div>
            <div className="eyebrow">sub-step</div>
            <div className="font-mono text-sm">{phase}</div>
          </div>
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
      />
    </WidgetFrame>
  );
}
