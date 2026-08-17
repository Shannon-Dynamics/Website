'use client';

import Link from 'next/link';
import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { BarChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Kf, type Gaussian } from '@/lib/filters/kf';
import { InfoFilter, flopLedger, toMoments } from '@/lib/filters/info';
import { sampleMvn } from '@/lib/prob/gaussian';
import { Rng } from '@/lib/prob/rng';
import { ellipse2, matVec, type Mat, type Vec } from '@/lib/prob/linalg';
import {
  clear,
  drawCovariance,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w6.2 — Moments vs. Information.
 *
 * One belief, two coordinate systems, run in lockstep. The left panel is the
 * familiar (μ, Σ) ellipse in the cart's (position, velocity) plane. The right
 * panel is the same belief as (ξ, Ω) — and when a measurement arrives it shows
 * the update for what it is in canonical form: an *addition*.
 *
 * The misconception this kills: "the information filter is a different filter".
 * It is the same posterior with the costs transposed, which is why the residual
 * between the two panels stays at machine epsilon while the flop ledger below
 * flips over as the state dimension grows.
 */

const DT = 0.2;
const ASPECT = 1.45;
/** Half-width of the (position, velocity) view window, in the y direction. */
const VIEW_HALF = 1.35;

const F: Mat = [
  [1, DT],
  [0, 1],
];
/** Sensor A: a beacon that measures position only. */
const H_POS: Mat = [[1, 0]];
/** Sensor B: a wheel-encoder tachometer that measures velocity only. */
const H_VEL: Mat = [[0, 1]];

const TRUE_ACCEL_SIGMA = 0.5;
const SIGMA_POS = 0.45;

function motionNoise(sigma: number): Mat {
  const q = sigma * sigma;
  return [
    [(q * DT ** 3) / 3, (q * DT ** 2) / 2],
    [(q * DT ** 2) / 2, q * DT],
  ];
}

type Phase = 'predict' | 'correct';

interface Params {
  secondSensor: boolean;
  velSigma: number;
  dim: number;
}

interface State {
  rng: Rng;
  truth: Vec;
  kf: Kf;
  inf: InfoFilter;
  phase: Phase;
  /** Belief entering this operation, drawn as the blue ghost. */
  before: Gaussian;
  after: Gaussian;
  /** Ω before this operation, and the matrices added to it if this was a correct. */
  omegaBefore: Mat;
  contributions: { name: string; Omega: Mat }[];
  /** Frozen for a whole predict/correct cycle so the ellipses stay comparable. */
  viewCenter: [number, number];
  /** Largest disagreement between the two panels' means — should stay ~1e-15. */
  residual: number;
}

export function MomentsVsInformation() {
  const [params, setParams] = useState<Params>({ secondSensor: false, velSigma: 0.5, dim: 12 });

  const init = useCallback((seed: number): State => {
    const P0: Mat = [
      [0.6, 0.15],
      [0.15, 0.35],
    ];
    const kf = new Kf([0, 0.8], P0);
    const inf = InfoFilter.fromMoments(kf.belief());
    return {
      rng: new Rng(seed),
      truth: [0, 0.8],
      kf,
      inf,
      phase: 'predict',
      before: kf.belief(),
      after: kf.belief(),
      omegaBefore: inf.canonical().Omega,
      contributions: [],
      viewCenter: [0, 0.8],
      residual: 0,
    };
  }, []);

  const step = useCallback(
    (s: State, tick: number): State => {
      const { rng, kf, inf } = s;
      const before = kf.belief();
      const omegaBefore = inf.canonical().Omega;
      const doPredict = tick % 2 === 0;

      let truth = s.truth;
      let contributions: { name: string; Omega: Mat }[] = [];
      let viewCenter = s.viewCenter;

      if (doPredict) {
        // The cart moves. Both representations must agree afterwards, and they
        // do — but only one of them got there without an inversion.
        truth = matVec(F, truth);
        const w = sampleMvn([0, 0], motionNoise(TRUE_ACCEL_SIGMA), rng);
        truth = [truth[0] + w[0], truth[1] + w[1]];

        kf.predictWith(F, motionNoise(TRUE_ACCEL_SIGMA));
        inf.predictWith(F, motionNoise(TRUE_ACCEL_SIGMA));
        viewCenter = [before.x[0], before.x[1]];
      } else {
        const zPos = truth[0] + rng.normal(0, SIGMA_POS);
        kf.updateWith([zPos], H_POS, [[SIGMA_POS ** 2]]);
        inf.correctWith([zPos], H_POS, [[SIGMA_POS ** 2]]);
        contributions = [
          { name: 'beacon  Hᵀ Q⁻¹ H', Omega: InfoFilter.contribution([zPos], H_POS, [[SIGMA_POS ** 2]]).Omega },
        ];

        if (params.secondSensor) {
          const zVel = truth[1] + rng.normal(0, params.velSigma);
          const Qv: Mat = [[params.velSigma ** 2]];
          kf.updateWith([zVel], H_VEL, Qv);
          inf.correctWith([zVel], H_VEL, Qv);
          contributions.push({
            name: 'tacho  Hᵀ Q⁻¹ H',
            Omega: InfoFilter.contribution([zVel], H_VEL, Qv).Omega,
          });
        }
      }

      const after = kf.belief();
      const alt = toMoments(inf.canonical());
      const residual = Math.max(
        Math.abs(after.x[0] - alt.x[0]),
        Math.abs(after.x[1] - alt.x[1]),
        Math.abs(after.P[0][0] - alt.P[0][0]),
      );

      return {
        rng,
        truth,
        kf,
        inf,
        phase: doPredict ? 'predict' : 'correct',
        before,
        after,
        omegaBefore,
        contributions,
        viewCenter,
        residual,
      };
    },
    [params.secondSensor, params.velSigma],
  );

  const sim = useSimulation<State>({ init, step, fps: 1.4, maxTicks: 60, loop: true, initialSeed: 5 });

  const world = useMemo(
    () => ({
      minX: sim.state.viewCenter[0] - VIEW_HALF * ASPECT,
      maxX: sim.state.viewCenter[0] + VIEW_HALF * ASPECT,
      minY: sim.state.viewCenter[1] - VIEW_HALF,
      maxY: sim.state.viewCenter[1] + VIEW_HALF,
    }),
    [sim.state.viewCenter],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { before, after, phase, truth } = sim.state;

      // Axes through the view centre: position runs right, velocity runs up.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, v.minX), sy(v, sim.state.viewCenter[1]));
      ctx.lineTo(sx(v, v.maxX), sy(v, sim.state.viewCenter[1]));
      ctx.moveTo(sx(v, sim.state.viewCenter[0]), sy(v, v.minY));
      ctx.lineTo(sx(v, sim.state.viewCenter[0]), sy(v, v.maxY));
      ctx.stroke();

      const afterColor = phase === 'predict' ? p.prediction : p.posterior;

      drawCovariance(
        ctx,
        v,
        [before.x[0], before.x[1]],
        ellipse2(before.P, 2),
        p.prior,
        { fill: false, alpha: 0.85, lineWidth: 1.5 },
      );
      drawCovariance(ctx, v, [after.x[0], after.x[1]], ellipse2(after.P, 2), afterColor, {
        fill: true,
        lineWidth: 2,
      });

      // The truth, which neither representation ever sees.
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(sx(v, truth[0]), sy(v, truth[1]), 3, 0, Math.PI * 2);
      ctx.fill();

      label(ctx, 'position →', sx(v, v.maxX) - 8, sy(v, sim.state.viewCenter[1]) - 9, p.ink, {
        size: 9,
        align: 'right',
      });
      label(ctx, '↑ velocity', sx(v, sim.state.viewCenter[0]) + 6, sy(v, v.maxY) + 10, p.ink, {
        size: 9,
      });
      label(
        ctx,
        phase === 'predict' ? 'PREDICT — Σ̄ = FΣFᵀ + R' : 'CORRECT — Σ = (I−KH)Σ̄(I−KH)ᵀ + KQKᵀ',
        sx(v, v.minX) + 8,
        sy(v, v.maxY) + 12,
        afterColor,
        { size: 10, weight: 600 },
      );
    },
    [sim.state],
  );

  const canonical = sim.state.inf.canonical();
  const ledger = useMemo(
    () => flopLedger(params.dim, 1, params.secondSensor ? 2 : 1),
    [params.dim, params.secondSensor],
  );

  const ledgerSeries = useMemo(
    () => [
      {
        id: 'predict',
        role: 'prediction' as const,
        data: [
          { x: 'moments (μ, Σ)', y: Math.log10(ledger.momentsPredict) },
          { x: 'information (ξ, Ω)', y: Math.log10(ledger.infoPredict) },
        ],
      },
      {
        id: 'correct',
        role: 'measurement' as const,
        data: [
          { x: 'moments (μ, Σ)', y: Math.log10(ledger.momentsCorrect) },
          { x: 'information (ξ, Ω)', y: Math.log10(ledger.infoCorrect) },
        ],
      },
    ],
    [ledger],
  );

  return (
    <WidgetFrame
      id="w6.2"
      title="Moments vs. Information"
      teaches="The information filter is not a different filter. It is the same posterior in the other coordinate system, with the cost of the two steps swapped."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Left: the belief as an ellipse in the cart&rsquo;s (position, velocity) plane — blue is
          what it was, orange or purple what the operation made of it. Right: the same belief as
          Ω&nbsp;=&nbsp;Σ⁻¹ and ξ&nbsp;=&nbsp;Σ⁻¹μ. Watch a <strong>correct</strong> step: in
          canonical form nothing is inverted and nothing is weighted — the sensor&rsquo;s
          HᵀQ⁻¹H is simply <em>added</em> in. Turn on the tachometer and a second matrix joins the
          sum; independent sensors commute and accumulate, which is the whole reason large fusion
          systems live in information form. The residual readout is the point of the widget: both
          panels are computing the same numbers. The ledger below says what that costs. Push the
          state dimension up and the two forms trade places — and if you are wondering why anyone
          would pay the information filter&rsquo;s prediction bill,
          <Link href="/chapters/ch15-factor-graphs"> Chapter 15</Link> is the answer: when Ω is sparse,
          you never form Σ at all.
        </>
      }
    >
      <div className="grid grid-cols-1 md:grid-cols-2 md:divide-x md:divide-fd-border">
        <SimCanvas
          world={world}
          draw={draw}
          deps={[sim.tick, sim.state]}
          aspect={ASPECT}
          padding={0}
          ariaLabel="A tilted confidence ellipse in the position-velocity plane; a faint ellipse shows the belief before the current operation and a filled one after it."
        />

        <div className="flex flex-col gap-3 border-t border-fd-border p-3 md:border-t-0">
          <div>
            <p className="eyebrow mb-1.5">information matrix Ω = Σ⁻¹</p>
            {sim.state.phase === 'correct' && sim.state.contributions.length > 0 ? (
              <div className="flex flex-wrap items-center gap-2">
                <MatrixTile m={sim.state.omegaBefore} role="prior" caption="Ω̄" />
                {sim.state.contributions.map((c) => (
                  <div key={c.name} className="flex items-center gap-2">
                    <span className="font-mono text-lg text-fd-muted-foreground">+</span>
                    <MatrixTile m={c.Omega} role="measurement" caption={c.name} />
                  </div>
                ))}
                <span className="font-mono text-lg text-fd-muted-foreground">=</span>
                <MatrixTile m={canonical.Omega} role="posterior" caption="Ω" />
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <MatrixTile m={sim.state.omegaBefore} role="prior" caption="Ω" />
                <span className="font-mono text-xs text-fd-muted-foreground">
                  → (F Ω⁻¹ Fᵀ + R)⁻¹ →
                </span>
                <MatrixTile m={canonical.Omega} role="prediction" caption="Ω̄" />
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-[0.72rem] tabular-nums">
            <span>
              <span className="eyebrow me-1.5">ξ</span>[{canonical.xi.map((x) => x.toFixed(3)).join(', ')}]
            </span>
            <span>
              <span className="eyebrow me-1.5">μ = Ω⁻¹ξ</span>[
              {sim.state.after.x.map((x) => x.toFixed(3)).join(', ')}]
            </span>
          </div>

          <p className="m-0 font-ui text-[0.72rem] leading-snug text-fd-muted-foreground">
            Largest disagreement between the moments filter and the information filter:{' '}
            <span className="font-mono text-fd-foreground">
              {sim.state.residual.toExponential(1)}
            </span>{' '}
            — machine epsilon. Same posterior, every step.
          </p>
        </div>
      </div>

      <div className="border-t border-fd-border px-3 pt-3">
        <BarChart
          series={ledgerSeries}
          xLabel="representation"
          yLabel="log₁₀ multiply–adds per step"
          height={230}
          valueFormat={(y) => Math.round(10 ** y).toLocaleString('en-US')}
          caption={`Dense flop ledger at n = ${params.dim}, m = 1, ${
            params.secondSensor ? 'two sensors' : 'one sensor'
          }. Bars are logarithmic; hover for the raw count.`}
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="state dimension n (for the ledger)"
          value={params.dim}
          min={2}
          max={60}
          step={1}
          format={(x) => x.toFixed(0)}
          onChange={(x) => setParams((p) => ({ ...p, dim: Math.round(x) }))}
          help="The ellipse stays 2-D; this is the dimension the cost ledger is priced at."
        />
        <Slider
          label="tachometer σ"
          role="measurement"
          value={params.velSigma}
          min={0.08}
          max={2}
          step={0.02}
          unit="m/s"
          onChange={(x) => setParams((p) => ({ ...p, velSigma: x }))}
          help="Only matters when the second sensor is on: a tighter tacho adds more information."
        />
        <Toggle
          label="second sensor (tachometer)"
          role="measurement"
          checked={params.secondSensor}
          onChange={(x) => setParams((p) => ({ ...p, secondSensor: x }))}
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

/* -------------------------------------------------------------------------- */

/** A small matrix, tinted by magnitude in one of the book's role colors. */
function MatrixTile({
  m,
  role,
  caption,
}: {
  m: Mat;
  role: 'prior' | 'prediction' | 'measurement' | 'posterior';
  caption: string;
}) {
  const peak = Math.max(...m.flat().map((x) => Math.abs(x)), 1e-9);
  return (
    <figure className="m-0">
      <div
        className="inline-grid overflow-hidden rounded-sm border border-fd-border"
        style={{ gridTemplateColumns: `repeat(${m[0].length}, minmax(0, 1fr))` }}
      >
        {m.map((row, i) =>
          row.map((x, j) => (
            <div key={`${i}-${j}`} className="relative min-w-[3.4rem] px-2 py-1">
              <span
                aria-hidden
                className="absolute inset-0"
                style={{ backgroundColor: `var(--pr-${role})`, opacity: 0.06 + 0.5 * (Math.abs(x) / peak) }}
              />
              <span className="relative font-mono text-[0.7rem] tabular-nums">{x.toFixed(2)}</span>
            </div>
          )),
        )}
      </div>
      <figcaption
        className="mt-0.5 font-mono text-[0.62rem]"
        style={{ color: `var(--pr-${role})` }}
      >
        {caption}
      </figcaption>
    </figure>
  );
}
