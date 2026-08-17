'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { EskfSe2, se2Nees } from '@/lib/filters/on-manifold-se2';
import { boxminus, boxplus, type Pose2, type Twist2 } from '@/lib/geom/se2';
import { ellipse2, matMul, transpose, type Mat } from '@/lib/prob/linalg';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w7.3 — the Error-State Ledger.
 *
 * Two panels showing the same instant. On the left, Rusty flying a figure-eight
 * across seven metres of plane, heading sweeping the whole circle: a
 * violently nonlinear trajectory. On the right, the *error* between the
 * truth and the filter's nominal state, plotted in the tangent space, where it
 * never leaves a box a hand's width across.
 *
 * That contrast is the entire argument for the error-state formulation. You do
 * not have to linearize the trajectory; you only have to linearize the error,
 * and the error's world is small, centered, and nearly flat.
 */

const DT = 0.1;
const PERIOD = 16; // seconds per figure-eight
const A = 3.5; // half-width of the lemniscate, metres
const FIX_EVERY = 8; // ticks between beacon fixes
const TRAIL = 60;

/* -------------------------------------------------------------------------- */
/* The reference trajectory: a Gerono lemniscate, differentiated analytically   */
/* so that the commanded (v, ω) are exactly consistent with the shape.          */
/* -------------------------------------------------------------------------- */

function command(t: number): { v: number; omega: number } {
  const w = (2 * Math.PI) / PERIOD;
  const s = w * t;
  const dx = A * Math.cos(s) * w;
  const dy = A * Math.cos(2 * s) * w;
  const ddx = -A * Math.sin(s) * w * w;
  const ddy = -2 * A * Math.sin(2 * s) * w * w;
  const sp2 = dx * dx + dy * dy;
  return { v: Math.sqrt(sp2), omega: (dx * ddy - dy * ddx) / Math.max(sp2, 1e-9) };
}

interface State {
  rng: Rng;
  t: number;
  truth: Pose2;
  eskf: EskfSe2;
  /** Covariance snapshot taken just before the last injection. */
  priorCov: Mat;
  /** The last injected error state ε̂, and how recently it happened. */
  lastEpsilon: Twist2;
  flash: number;
  fix: [number, number] | null;
  truthPath: { x: number; y: number }[];
  nominalPath: { x: number; y: number }[];
  errorTrail: Twist2[];
  hist: { pos: number; head: number; nees: number }[];
}

const diag3 = (a: number, b: number, c: number): Mat => [
  [a, 0, 0],
  [0, b, 0],
  [0, 0, c],
];

export function ErrorStateLedger() {
  const [noise, setNoise] = useState(0.7);

  const init = useCallback((seed: number): State => {
    const start: Pose2 = { x: 0, y: 0, theta: Math.PI / 4 };
    const eskf = new EskfSe2(start, diag3(0.02, 0.02, 0.01));
    return {
      rng: new Rng(seed),
      t: 0,
      truth: { ...start },
      eskf,
      priorCov: eskf.cov,
      lastEpsilon: [0, 0, 0],
      flash: 0,
      fix: null,
      truthPath: [],
      nominalPath: [],
      errorTrail: [],
      hist: [],
    };
  }, []);

  const step = useCallback(
    (s: State, tick: number): State => {
      const { rng, eskf } = s;
      const { v, omega } = command(s.t);
      const u: Twist2 = [v * DT, 0, omega * DT];

      // Process noise, as a body-frame twist covariance per step.
      const qv = 0.006 * noise * noise * DT;
      const qlat = 0.0015 * noise * noise * DT;
      const qw = 0.004 * noise * noise * DT;
      const R = diag3(qv, qlat, qw);

      // The world moves: the true pose gets the commanded twist plus a
      // body-frame perturbation, applied through ⊞ so it stays on SE(2).
      const truth = boxplus(s.truth, [
        u[0] + rng.normal(0, Math.sqrt(qv)),
        u[1] + rng.normal(0, Math.sqrt(qlat)),
        u[2] + rng.normal(0, Math.sqrt(qw)),
      ]);

      eskf.predict(u, R);
      const priorCov = eskf.cov;

      let fix: [number, number] | null = null;
      let lastEpsilon = s.lastEpsilon;
      let flash = Math.max(s.flash - 1, 0);
      if (tick % FIX_EVERY === FIX_EVERY - 1) {
        const sp = 0.12 * noise;
        fix = [truth.x + rng.normal(0, sp), truth.y + rng.normal(0, sp)];
        const out = eskf.correctPosition(fix, [
          [sp * sp, 0],
          [0, sp * sp],
        ]);
        lastEpsilon = [out.epsilon[0], out.epsilon[1], out.epsilon[2]];
        flash = 6;
      }

      const err = boxminus(truth, eskf.nominal);
      const nees = se2Nees(truth, eskf.belief());

      return {
        rng,
        t: s.t + DT,
        truth,
        eskf,
        priorCov,
        lastEpsilon,
        flash,
        fix: fix ?? s.fix,
        truthPath: [...s.truthPath, { x: truth.x, y: truth.y }].slice(-400),
        nominalPath: [...s.nominalPath, { x: eskf.nominal.x, y: eskf.nominal.y }].slice(-400),
        errorTrail: [...s.errorTrail, err].slice(-TRAIL),
        hist: [
          ...s.hist,
          { pos: Math.hypot(err[0], err[1]), head: (err[2] * 180) / Math.PI, nees },
        ].slice(-90),
      };
    },
    [noise],
  );

  const sim = useSimulation<State>({ init, step, fps: 20, initialSeed: 5 });

  /* ---- panel geometry, in canvas world units --------------------------- */
  const wx = (x: number) => 4.9 + x;
  const wy = (y: number) => 3.5 + y;
  const LED_CX = 12.9;
  const LED_CY = 3.9;
  const LED_HALF = 2.5; // canvas units
  const LED_RANGE = 0.55; // metres of tangent space per half-box

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;
      // Clamped so that an error which has escaped the ledger is still drawn —
      // pinned to the wall, which is the honest picture of what just happened.
      const clamp = (m: number) => Math.max(-LED_RANGE, Math.min(LED_RANGE, m));
      const tx = (m: number) => LED_CX + (clamp(m) / LED_RANGE) * LED_HALF;
      const ty = (m: number) => LED_CY + (clamp(m) / LED_RANGE) * LED_HALF;
      const escaped = (e: readonly number[]) =>
        Math.abs(e[0]) > LED_RANGE || Math.abs(e[1]) > LED_RANGE;

      const panel = (x0: number, y0: number, x1: number, y1: number, title: string) => {
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.strokeRect(sx(v, x0), sy(v, y1), sl(v, x1 - x0), sl(v, y1 - y0));
        label(ctx, title, sx(v, x0) + 6, sy(v, y1) + 11, p.ink, { size: 10, weight: 600 });
      };

      panel(0.3, 0.3, 9.6, 6.9, 'the state — SE(2), 7 m of plane, heading sweeping the whole circle');
      panel(10.1, 0.3, 15.7, 6.9, 'the error — ℝ³ tangent space, ±55 cm');

      /* ---- left: the trajectory ---------------------------------------- */
      const path = (pts: { x: number; y: number }[], color: string, dashed: boolean) => {
        if (pts.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        pts.forEach((q, i) => {
          const px = sx(v, wx(q.x));
          const py = sy(v, wy(q.y));
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      };
      path(s.truthPath, p.truth, true);
      path(s.nominalPath, p.posterior, false);

      // The world-frame position covariance: Σ_world = R Σ_body Rᵀ.
      const c = Math.cos(s.eskf.nominal.theta);
      const sn = Math.sin(s.eskf.nominal.theta);
      const Rm: Mat = [
        [c, -sn],
        [sn, c],
      ];
      const body: Mat = [
        [s.eskf.cov[0][0], s.eskf.cov[0][1]],
        [s.eskf.cov[1][0], s.eskf.cov[1][1]],
      ];
      const world = matMul(matMul(Rm, body), transpose(Rm));
      const e = ellipse2(world, 2);
      ctx.save();
      ctx.translate(sx(v, wx(s.eskf.nominal.x)), sy(v, wy(s.eskf.nominal.y)));
      ctx.rotate(-e.angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(sl(v, e.rx), 1), Math.max(sl(v, e.ry), 1), 0, 0, Math.PI * 2);
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      if (s.fix) {
        ctx.fillStyle = p.measurement;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(sx(v, wx(s.fix[0])), sy(v, wy(s.fix[1])), 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      const robot = (pose: Pose2, color: string, filled: boolean) => {
        const r = sl(v, 0.28);
        ctx.save();
        ctx.translate(sx(v, wx(pose.x)), sy(v, wy(pose.y)));
        ctx.rotate(-pose.theta);
        ctx.beginPath();
        ctx.moveTo(r, 0);
        ctx.lineTo(-r * 0.72, r * 0.66);
        ctx.lineTo(-r * 0.38, 0);
        ctx.lineTo(-r * 0.72, -r * 0.66);
        ctx.closePath();
        if (filled) {
          ctx.fillStyle = color;
          ctx.fill();
        } else {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.8;
          ctx.stroke();
        }
        ctx.restore();
      };
      robot(s.truth, p.truth, false);
      robot(s.eskf.nominal, p.posterior, true);

      /* ---- right: the ledger ------------------------------------------- */
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, LED_CX - LED_HALF), sy(v, LED_CY));
      ctx.lineTo(sx(v, LED_CX + LED_HALF), sy(v, LED_CY));
      ctx.moveTo(sx(v, LED_CX), sy(v, LED_CY - LED_HALF));
      ctx.lineTo(sx(v, LED_CX), sy(v, LED_CY + LED_HALF));
      ctx.stroke();
      for (const r of [0.2, 0.4]) {
        ctx.beginPath();
        ctx.arc(sx(v, LED_CX), sy(v, LED_CY), sl(v, (r / LED_RANGE) * LED_HALF), 0, Math.PI * 2);
        ctx.stroke();
        label(
          ctx,
          `${r} m`,
          sx(v, LED_CX) + sl(v, (r / LED_RANGE) * LED_HALF) + 3,
          sy(v, LED_CY) - 5,
          p.grid,
          { size: 8 },
        );
      }

      const tangentEllipse = (cov: Mat, color: string, width: number, dashed: boolean) => {
        const el = ellipse2(
          [
            [cov[0][0], cov[0][1]],
            [cov[1][0], cov[1][1]],
          ],
          2,
        );
        ctx.save();
        ctx.translate(sx(v, LED_CX), sy(v, LED_CY));
        ctx.rotate(-el.angle);
        ctx.beginPath();
        ctx.ellipse(
          0,
          0,
          Math.max(sl(v, (el.rx / LED_RANGE) * LED_HALF), 1),
          Math.max(sl(v, (el.ry / LED_RANGE) * LED_HALF), 1),
          0,
          0,
          Math.PI * 2,
        );
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      };
      tangentEllipse(s.priorCov, p.prior, 1.4, true);
      tangentEllipse(s.eskf.cov, p.posterior, 1.8, false);

      // The true error, and where it has been. This dot is the whole point: it
      // wanders inside a 40 cm circle while the robot flies around seven metres.
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      s.errorTrail.forEach((q, i) => {
        const px = sx(v, tx(q[0]));
        const py = sy(v, ty(q[1]));
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
      const cur = s.errorTrail[s.errorTrail.length - 1] ?? [0, 0, 0];
      const out = escaped(cur);
      ctx.fillStyle = p.truth;
      ctx.strokeStyle = p.prediction;
      ctx.beginPath();
      ctx.arc(sx(v, tx(cur[0])), sy(v, ty(cur[1])), 4, 0, Math.PI * 2);
      ctx.fill();
      if (out) {
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx(v, tx(cur[0])), sy(v, ty(cur[1])), 7, 0, Math.PI * 2);
        ctx.stroke();
      }
      label(ctx, out ? 'x ⊟ μ — off the ledger' : 'x ⊟ μ', sx(v, tx(cur[0])) + 9, sy(v, ty(cur[1])), out ? p.prediction : p.truth, { size: 9 });

      // The injection: the correction that was just emptied into the nominal.
      if (s.flash > 0) {
        const eps = s.lastEpsilon;
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 2 + s.flash * 0.3;
        ctx.globalAlpha = 0.35 + 0.65 * (s.flash / 6);
        ctx.beginPath();
        ctx.moveTo(sx(v, LED_CX), sy(v, LED_CY));
        ctx.lineTo(sx(v, tx(eps[0])), sy(v, ty(eps[1])));
        ctx.stroke();
        ctx.globalAlpha = 1;
        label(
          ctx,
          'inject ε̂ → μ, reset ε̂ → 0',
          sx(v, LED_CX - LED_HALF) + 4,
          sy(v, LED_CY - LED_HALF) - 10,
          p.measurement,
          { size: 9, weight: 600 },
        );
      }

      // Heading component: a separate bar, because it is measured in radians.
      const barY = 0.95;
      const barHalf = 2.2;
      const headRange = 0.35; // rad
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, LED_CX - barHalf), sy(v, barY));
      ctx.lineTo(sx(v, LED_CX + barHalf), sy(v, barY));
      ctx.stroke();
      const sigTheta = Math.sqrt(Math.max(s.eskf.cov[2][2], 0));
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(sx(v, LED_CX - (2 * sigTheta * barHalf) / headRange), sy(v, barY));
      ctx.lineTo(sx(v, LED_CX + (2 * sigTheta * barHalf) / headRange), sy(v, barY));
      ctx.stroke();
      const headMark = Math.max(-headRange, Math.min(headRange, cur[2]));
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(sx(v, LED_CX + (headMark * barHalf) / headRange), sy(v, barY), 3.5, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, 'ε_θ  (±2σ band)', sx(v, LED_CX - barHalf), sy(v, barY) + 13, p.ink, { size: 9 });
    },
    [sim.state],
  );

  const last = sim.state.hist[sim.state.hist.length - 1];
  const spark = useMemo(
    () => ({
      pos: sim.state.hist.map((h) => h.pos),
      head: sim.state.hist.map((h) => Math.abs(h.head)),
      nees: sim.state.hist.map((h) => h.nees),
    }),
    [sim.state.hist],
  );
  const meanNees = useMemo(() => {
    const h = sim.state.hist;
    return h.length ? h.reduce((a, x) => a + x.nees, 0) / h.length : 0;
  }, [sim.state.hist]);

  return (
    <WidgetFrame
      id="w7.3"
      title="The Error-State Ledger"
      teaches="You never had to linearize the trajectory. Linearize the error instead — its world is small, centered, and almost flat."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Left: the nominal state, an SE(2) pose flown around a figure-eight, purple, with the true
          pose in gray and beacon fixes in green. Right: the same instant seen in the tangent space
          at the nominal — the blue dashed ellipse is the predicted error covariance, purple is the
          posterior, and the gray dot is the <em>actual</em> error <code>x ⊟ μ</code>. Notice what
          the right panel never does: it never travels. Seven metres of motion and a heading that
          sweeps the whole circle produce an error that stays inside about thirty centimetres, which is
          why a first-order expansion of the <em>error</em> dynamics is honest even when a
          first-order expansion of the <em>state</em> dynamics would not be. Green flashes are
          injections: the ledger is emptied into the nominal pose with ⊞ and reset to zero. Push the
          noise slider up and watch both ellipses grow together — that is a consistent filter, and
          the NEES tile should stay near 3 the whole way. Keep pushing until the gray dot hits the
          wall of the ledger and turns orange: that is the moment the error has stopped being small,
          and the moment the error-state approximation stops being free.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: 16, minY: 0, maxY: 7 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={2.29}
        padding={0}
        ariaLabel="Left panel: a robot flying a figure-eight trajectory with its estimated path and a covariance ellipse. Right panel: the tangent-space error, which stays inside a small circle for the entire run, with periodic correction injections."
      />

      <div className="border-t border-fd-border p-3">
        <Dashboard columns={3}>
          <StatTile
            label="position error |ε_xy|"
            value={last?.pos ?? 0}
            unit="m"
            role="truth"
            precision={3}
            sparkline={spark.pos}
          />
          <StatTile
            label="heading error ε_θ"
            value={last?.head ?? 0}
            unit="deg"
            role="posterior"
            precision={2}
            sparkline={spark.head}
          />
          <StatTile
            label="NEES (target 3)"
            value={meanNees}
            role="prior"
            precision={2}
            trend={(last?.nees ?? 0) - meanNees}
            trendLabel="this step"
            sparkline={spark.nees}
          />
        </Dashboard>
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="Noise scale"
          role="prior"
          value={noise}
          min={0.25}
          max={3}
          step={0.05}
          onChange={setNoise}
          help="Scales the process and beacon noise together. The error grows; the ellipse grows with it. Past about 2, the error escapes the ledger."
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
