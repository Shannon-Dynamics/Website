'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { unscentedTransform } from '@/lib/filters/ukf';
import { numericJacobian } from '@/lib/filters/ekf';
import { normalPdf } from '@/lib/prob/gaussian';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w7.1 — the Linearization Lens.
 *
 * One nonlinear function, one Gaussian input, three answers for the output:
 * the truth (gray, by quadrature and a Monte Carlo cloud), the EKF's tangent
 * line (orange), and the unscented transform's three scouts (green points,
 * purple result). The gap between orange and gray is the *whole* subject of
 * this chapter, and it is governed by exactly two numbers: how curved the
 * function is where you linearized, and how wide the input is.
 *
 * Everything runs the real library: `unscentedTransform` from lib/filters/ukf
 * and `numericJacobian` from lib/filters/ekf — the same routines the chapter's
 * Rust listings mirror.
 */

/* -------------------------------------------------------------------------- */
/* The functions on offer                                                      */
/* -------------------------------------------------------------------------- */

interface Curve {
  key: string;
  name: string;
  /** The nonlinear map y = f(x). */
  f: (x: number) => number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  /** Where the sweep starts and how far it travels. */
  sweepCenter: number;
  sweepAmp: number;
  /** Shown under the plot so the reader knows what they are looking at. */
  formula: string;
  /** Where in the domain the curvature is worst, in words. */
  note: string;
}

const CURVES: Curve[] = [
  {
    key: 'quad',
    name: 'x² / 20',
    f: (x) => (x * x) / 20,
    xMin: -6,
    xMax: 6,
    yMin: -1.5,
    yMax: 4.2,
    sweepCenter: 0,
    sweepAmp: 4.2,
    formula: 'f(x) = x² / 20',
    note: 'Constant curvature f″ = 1/10: the bias is the same everywhere, and grows exactly as σ².',
  },
  {
    key: 'range',
    name: 'range to beacon',
    f: (x) => Math.hypot(x, 2),
    xMin: -5.5,
    xMax: 5.5,
    yMin: 0.9,
    yMax: 6.14,
    sweepCenter: 0,
    sweepAmp: 4.2,
    formula: 'f(x) = √(x² + 2²)   — range from a beacon 2 m off the path',
    note: 'Curvature is concentrated at closest approach and vanishes far away — drive past the beacon and watch the bias switch on and off.',
  },
  {
    key: 'sin',
    name: 'r sin θ',
    f: (x) => Math.sin(x),
    xMin: -0.4,
    xMax: 3.6,
    yMin: -0.55,
    yMax: 1.35,
    sweepCenter: 1.5708,
    sweepAmp: 1.35,
    formula: 'f(θ) = sin θ   — the y-component of a unit-range polar measurement',
    note: 'At θ = π/2 the slope is zero, so the EKF reports almost no output spread at all. It is wrong about the mean and about the variance at the same time.',
  },
];

/* -------------------------------------------------------------------------- */
/* Numerics                                                                    */
/* -------------------------------------------------------------------------- */

const QUAD_N = 400; // Simpson panels over ±6σ — deterministic "truth"

/** ∫ f(x) N(x; μ, σ²) dx and its variance, by composite Simpson. */
function trueMoments(f: (x: number) => number, mu: number, sigma: number) {
  const a = mu - 6 * sigma;
  const h = (12 * sigma) / QUAD_N;
  let m0 = 0;
  let m1 = 0;
  let m2 = 0;
  for (let i = 0; i <= QUAD_N; i++) {
    const x = a + i * h;
    const w = (i === 0 || i === QUAD_N ? 1 : i % 2 === 1 ? 4 : 2) * (h / 3);
    const p = w * normalPdf(x, mu, sigma);
    const y = f(x);
    m0 += p;
    m1 += p * y;
    m2 += p * y * y;
  }
  const mean = m1 / m0;
  return { mean, variance: Math.max(m2 / m0 - mean * mean, 0) };
}

/** Central second difference — derivation 2's curvature term, measured. */
function secondDerivative(f: (x: number) => number, x: number, h = 1e-3): number {
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

interface Analysis {
  truth: { mean: number; sd: number };
  ekf: { mean: number; sd: number; slope: number };
  ut: { mean: number; sd: number; points: { x: number; y: number }[] };
  /** ½ f″(μ) σ² — the predicted bias from the second-order Taylor term. */
  predictedBias: number;
}

/** α = 1, β = 0, κ = 3 − n with n = 1: the three-scout scaled transform. */
const UT_PARAMS = { alpha: 1, beta: 0, kappa: 2 };

function analyze(f: (x: number) => number, mu: number, sigma: number): Analysis {
  const truth = trueMoments(f, mu, sigma);

  // EKF: propagate the mean through f, the covariance through the tangent.
  const slope = numericJacobian((v) => [f(v[0])], [mu])[0][0];
  const ekfMean = f(mu);
  const ekfSd = Math.abs(slope) * sigma;

  // UT: the real library routine, on a 1-D "state".
  const res = unscentedTransform([mu], [[sigma * sigma]], (v) => [f(v[0])], UT_PARAMS);

  return {
    truth: { mean: truth.mean, sd: Math.sqrt(truth.variance) },
    ekf: { mean: ekfMean, sd: ekfSd, slope },
    ut: {
      mean: res.mean[0],
      sd: Math.sqrt(Math.max(res.cov[0][0], 0)),
      points: res.points.map((p, i) => ({ x: p[0], y: res.transformed[i][0] })),
    },
    predictedBias: 0.5 * secondDerivative(f, mu) * sigma * sigma,
  };
}

/* -------------------------------------------------------------------------- */
/* Widget                                                                      */
/* -------------------------------------------------------------------------- */

interface Params {
  curveKey: string;
  sigma: number;
  showCloud: boolean;
  showScouts: boolean;
}

interface State {
  rng: Rng;
  /** Sweep phase; the operating point is a slow oscillation along the curve. */
  phase: number;
  cloud: number[];
}

const CLOUD_N = 700;

export function LinearizationLens() {
  const [params, setParams] = useState<Params>({
    curveKey: 'range',
    sigma: 0.9,
    showCloud: true,
    showScouts: true,
  });
  /** Non-null once the reader takes the operating point over from the sweep. */
  const [manualMu, setManualMu] = useState<number | null>(null);

  const curve = CURVES.find((c) => c.key === params.curveKey) ?? CURVES[0];

  const init = useCallback(
    (seed: number): State => ({ rng: new Rng(seed), phase: 0, cloud: [] }),
    [],
  );

  const step = useCallback((s: State): State => {
    const cloud = Array.from({ length: CLOUD_N }, () => s.rng.normal());
    return { rng: s.rng, phase: s.phase + 0.045, cloud };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 18, initialSeed: 7 });

  const mu =
    manualMu ?? curve.sweepCenter + curve.sweepAmp * Math.sin(sim.state.phase);

  const analysis = useMemo(
    () => analyze(curve.f, mu, params.sigma),
    [curve, mu, params.sigma],
  );

  /** The bias landscape: where along this curve does linearizing hurt? */
  const biasCurve = useMemo(() => {
    const n = 55;
    const ekf: { x: number; y: number }[] = [];
    const ut: { x: number; y: number }[] = [];
    for (let i = 0; i <= n; i++) {
      const x = curve.xMin + ((curve.xMax - curve.xMin) * i) / n;
      const a = analyze(curve.f, x, params.sigma);
      ekf.push({ x, y: a.ekf.mean - a.truth.mean });
      ut.push({ x, y: a.ut.mean - a.truth.mean });
    }
    return [
      { id: 'EKF', role: 'prediction' as const, data: ekf },
      { id: 'UKF', role: 'posterior' as const, data: ut },
    ];
  }, [curve, params.sigma]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { f } = curve;
      const sigma = params.sigma;
      const xSpan = v.maxX - v.minX;
      const ySpan = v.maxY - v.minY;
      const inputBand = 0.17 * ySpan; // bottom strip: the blue input density
      const outputBand = 0.17 * xSpan; // left strip: the three output densities
      const baseY = v.minY + 0.02 * ySpan;
      const baseX = v.minX + 0.02 * xSpan;

      // ---- axes ---------------------------------------------------------
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, baseX), sy(v, v.minY));
      ctx.lineTo(sx(v, baseX), sy(v, v.maxY));
      ctx.moveTo(sx(v, v.minX), sy(v, baseY));
      ctx.lineTo(sx(v, v.maxX), sy(v, baseY));
      ctx.stroke();

      // ---- the function --------------------------------------------------
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = v.minX + (xSpan * i) / 300;
        const y = f(x);
        if (i === 0) ctx.moveTo(sx(v, x), sy(v, y));
        else ctx.lineTo(sx(v, x), sy(v, y));
      }
      ctx.stroke();

      // ---- Monte Carlo truth cloud ---------------------------------------
      if (params.showCloud) {
        ctx.fillStyle = p.truth;
        ctx.globalAlpha = 0.3;
        for (const z of sim.state.cloud) {
          const x = mu + sigma * z;
          ctx.beginPath();
          ctx.arc(sx(v, x), sy(v, f(x)), 1.1, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // ---- the tangent line: the EKF's entire worldview --------------------
      const slope = analysis.ekf.slope;
      const y0 = f(mu);
      ctx.strokeStyle = p.prediction;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(v, v.minX), sy(v, y0 + slope * (v.minX - mu)));
      ctx.lineTo(sx(v, v.maxX), sy(v, y0 + slope * (v.maxX - mu)));
      ctx.stroke();
      ctx.setLineDash([]);

      // ---- input Gaussian, along the bottom -------------------------------
      const peakIn = normalPdf(mu, mu, sigma);
      ctx.fillStyle = p.prior;
      ctx.globalAlpha = 0.22;
      ctx.beginPath();
      ctx.moveTo(sx(v, v.minX), sy(v, baseY));
      for (let i = 0; i <= 200; i++) {
        const x = v.minX + (xSpan * i) / 200;
        const h = (normalPdf(x, mu, sigma) / peakIn) * inputBand;
        ctx.lineTo(sx(v, x), sy(v, baseY + h));
      }
      ctx.lineTo(sx(v, v.maxX), sy(v, baseY));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = p.prior;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const x = v.minX + (xSpan * i) / 200;
        const h = (normalPdf(x, mu, sigma) / peakIn) * inputBand;
        if (i === 0) ctx.moveTo(sx(v, x), sy(v, baseY + h));
        else ctx.lineTo(sx(v, x), sy(v, baseY + h));
      }
      ctx.stroke();

      // The operating point: a vertical drop from the curve to the input axis.
      ctx.strokeStyle = p.prior;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, mu), sy(v, baseY));
      ctx.lineTo(sx(v, mu), sy(v, y0));
      ctx.lineTo(sx(v, baseX), sy(v, y0));
      ctx.stroke();
      ctx.setLineDash([]);

      // ---- output densities, along the left -------------------------------
      const drawOutput = (
        mean: number,
        sd: number,
        color: string,
        width: number,
        dashed = false,
      ) => {
        const s = Math.max(sd, 1e-4);
        const peak = normalPdf(mean, mean, s);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (let i = 0; i <= 220; i++) {
          const y = v.minY + (ySpan * i) / 220;
          const w = (normalPdf(y, mean, s) / peak) * outputBand;
          if (i === 0) ctx.moveTo(sx(v, baseX + w), sy(v, y));
          else ctx.lineTo(sx(v, baseX + w), sy(v, y));
        }
        ctx.stroke();
        ctx.setLineDash([]);
      };

      // Truth as a histogram of the pushed-through cloud — the banana, seen edge on.
      if (params.showCloud && sim.state.cloud.length > 0) {
        const bins = 46;
        const counts = new Array<number>(bins).fill(0);
        for (const z of sim.state.cloud) {
          const y = f(mu + sigma * z);
          const b = Math.floor(((y - v.minY) / ySpan) * bins);
          if (b >= 0 && b < bins) counts[b] += 1;
        }
        const cMax = Math.max(...counts, 1);
        ctx.fillStyle = p.truth;
        ctx.globalAlpha = 0.3;
        for (let b = 0; b < bins; b++) {
          if (counts[b] === 0) continue;
          const w = (counts[b] / cMax) * outputBand;
          const yLo = v.minY + (ySpan * b) / bins;
          ctx.fillRect(
            sx(v, baseX),
            sy(v, yLo + ySpan / bins),
            sl(v, w),
            sl(v, ySpan / bins) + 0.5,
          );
        }
        ctx.globalAlpha = 1;
      }
      drawOutput(analysis.truth.mean, analysis.truth.sd, p.truth, 1.6, true);
      drawOutput(analysis.ekf.mean, analysis.ekf.sd, p.prediction, 1.8);
      drawOutput(analysis.ut.mean, analysis.ut.sd, p.posterior, 1.8);

      // ---- sigma points: the scouts ---------------------------------------
      if (params.showScouts) {
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        for (const pt of analysis.ut.points) {
          ctx.beginPath();
          ctx.moveTo(sx(v, pt.x), sy(v, baseY));
          ctx.lineTo(sx(v, pt.x), sy(v, pt.y));
          ctx.lineTo(sx(v, baseX), sy(v, pt.y));
          ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.fillStyle = p.measurement;
        for (const pt of analysis.ut.points) {
          ctx.beginPath();
          ctx.arc(sx(v, pt.x), sy(v, pt.y), 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // ---- annotation ------------------------------------------------------
      label(ctx, `μ = ${mu.toFixed(2)}`, sx(v, mu) + 6, sy(v, baseY) - 12, p.prior, {
        size: 10,
        weight: 600,
      });
      label(ctx, 'tangent = EKF', sx(v, v.maxX) - 8, sy(v, y0 + slope * (v.maxX - mu)) - 10, p.prediction, {
        size: 10,
        align: 'right',
      });
      label(ctx, curve.formula, sx(v, v.minX) + 8, sy(v, v.maxY) - 12, p.ink, { size: 10 });
    },
    [curve, params, analysis, mu, sim.state.cloud],
  );

  const biasEkf = analysis.ekf.mean - analysis.truth.mean;
  const biasUt = analysis.ut.mean - analysis.truth.mean;

  // Memoized as an *element*: the parent re-renders 18 times a second while the
  // sweep runs, and re-laying-out an SVG chart at that rate for data that has
  // not changed is the one way to make this widget feel slow.
  const biasChart = useMemo(
    () => (
      <div className="border-t border-fd-border px-3 pt-3">
        <p className="eyebrow mb-1">Mean error across the whole domain, at this σ</p>
        <LineChart
          series={biasCurve}
          xLabel="operating point μ"
          yLabel="mean error"
          height={190}
          markers={[{ axis: 'y', value: 0, label: 'truth', role: 'truth' }]}
          ariaLabel="Signed mean error of the EKF and the unscented transform as a function of the operating point. The EKF curve departs from zero wherever the function is curved; the unscented curve stays near zero."
        />
      </div>
    ),
    [biasCurve],
  );

  return (
    <WidgetFrame
      id="w7.1"
      title="The Linearization Lens"
      teaches="Linearization error is not a constant nuisance — it is curvature × spread, and you can watch both terms move it."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          A blue Gaussian sits on the horizontal axis; the three curves on the vertical axis are
          three answers to <em>what does it look like after passing through f</em>. Gray dashed is
          the truth (quadrature, with the Monte&nbsp;Carlo cloud showing where the mass actually
          lands). Orange is the EKF: it slides the mean along the true curve but pushes the spread
          through the dashed tangent line. Purple is the unscented transform, rebuilt from the three
          green scouts. <strong>Watch the operating point sweep</strong> — the tangent re-fits at
          every frame, which is what "the EKF re-linearizes each step" means, and the orange curve
          drifts off the gray one exactly where the function bends. Then push the spread slider: the
          bias grows like σ², not like σ. The chart underneath is the same experiment run at every
          operating point at once.
        </>
      }
    >
      <SimCanvas
        world={{ minX: curve.xMin, maxX: curve.xMax, minY: curve.yMin, maxY: curve.yMax }}
        draw={draw}
        deps={[sim.tick, analysis, params, mu]}
        aspect={2.1}
        padding={0}
        ariaLabel="A nonlinear curve with a Gaussian input distribution on the horizontal axis and three approximations of the output distribution on the vertical axis: the true one in gray, the EKF's linearized one in orange, and the unscented transform's in purple."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="EKF mean error" value={fmt(biasEkf)} role="prediction" />
        <Stat label="predicted: −½ f″(μ) σ²" value={fmt(-analysis.predictedBias)} />
        <Stat label="UKF mean error" value={fmt(biasUt)} role="posterior" />
        <Stat
          label="σ_out: EKF / UKF / true"
          value={`${analysis.ekf.sd.toFixed(2)} / ${analysis.ut.sd.toFixed(2)} / ${analysis.truth.sd.toFixed(2)}`}
        />
      </div>

      {biasChart}

      <ControlPanel columns={3}>
        <Slider
          label="Input spread σ"
          role="prior"
          value={params.sigma}
          min={0.05}
          max={curve.key === 'sin' ? 0.8 : 2.2}
          step={0.05}
          onChange={(v) => setParams((q) => ({ ...q, sigma: v }))}
          help="The headline control. Bias scales with σ², so doubling this quadruples the error."
        />
        <Slider
          label="Operating point μ"
          role="prediction"
          value={mu}
          min={curve.xMin + 0.3}
          max={curve.xMax - 0.3}
          step={0.02}
          onChange={(v) => {
            setManualMu(v);
            sim.pause();
          }}
          help="Take the sweep over by hand. Press play to hand it back."
        />
        <label className="flex flex-col gap-1">
          <span className="font-ui text-[0.72rem] font-medium">Function</span>
          <select
            value={params.curveKey}
            onChange={(e) => {
              setParams((q) => ({ ...q, curveKey: e.target.value }));
              setManualMu(null);
            }}
            className="rounded-sm border border-fd-border bg-fd-card px-1.5 py-1 font-mono text-[0.72rem]"
            aria-label="Nonlinear function"
          >
            {CURVES.map((c) => (
              <option key={c.key} value={c.key}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <Toggle
          label="Monte Carlo truth"
          role="truth"
          checked={params.showCloud}
          onChange={(v) => setParams((q) => ({ ...q, showCloud: v }))}
        />
        <Toggle
          label="Sigma points"
          role="measurement"
          checked={params.showScouts}
          onChange={(v) => setParams((q) => ({ ...q, showScouts: v }))}
        />
      </ControlPanel>

      <p className="border-t border-fd-border px-3 py-2 font-ui text-[0.75rem] text-fd-muted-foreground">
        {curve.note}
      </p>

      <Transport
        playing={sim.playing}
        onToggle={() => {
          setManualMu(null);
          sim.toggle();
        }}
        onStep={sim.stepOnce}
        onReset={() => {
          setManualMu(null);
          sim.reset();
        }}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function fmt(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(3);
}

function Stat({
  label: l,
  value,
  role,
}: {
  label: string;
  value: string;
  role?: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow" style={role ? { color: `var(--pr-${role})` } : undefined}>
        {l}
      </div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
