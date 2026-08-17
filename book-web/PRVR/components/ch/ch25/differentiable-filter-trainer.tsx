'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import {
  ActionButton,
  ButtonRow,
  ControlPanel,
  Slider,
  Toggle,
  Transport,
} from '@/components/sim/controls';
import { LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import {
  DiffKf1d,
  meanNees,
  meanNis,
  simulateTraj1d,
  type DkfLoss,
  type TrajLog1d,
} from '@/lib/learn/diff-kf';
import { clear, label, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w25.2 — the Differentiable Filter Trainer.
 *
 * Rusty's cart on its rail again, but this time nobody tunes R and Q. The
 * filter is handed a deliberately wrong θ = (log r, log q) and told to descend
 * a proper scoring rule through its own five equations. Every gradient the
 * trainer takes comes from `DiffKf1d.lossAndGrad`, which is the sensitivity
 * recursion derived in F5 — the derivation *is* the source code.
 *
 * The default loss is the **evidence**: it never looks at the ground truth, so
 * the gray dashed line on the canvas is decoration, not data. Switch to the
 * state NLL and the trainer starts cheating with information no real robot has.
 */

const STEPS = 320;
const ASPECT = 2.3;
const HISTORY = 260;

/** The system that actually generated the log. The trainer must find these. */
const TRUE_R = 0.06;
const TRUE_Q = 0.28;

/** Deliberately wrong both ways: "my model is terrible, my sensor is perfect". */
const START_LOG_R = Math.log(0.8);
const START_LOG_Q = Math.log(0.02);

interface Params {
  lr: number;
  loss: DkfLoss;
}

interface Sample {
  epoch: number;
  loss: number;
  r: number;
  q: number;
}

interface State {
  rng: Rng;
  log: TrajLog1d;
  kf: DiffKf1d;
  epoch: number;
  loss: number;
  gradR: number;
  gradQ: number;
  history: Sample[];
}

export function DifferentiableFilterTrainer() {
  const [params, setParams] = useState<Params>({ lr: 1, loss: 'evidence' });

  const init = useCallback((seed: number): State => {
    const rng = new Rng(seed);
    const log = simulateTraj1d(
      {
        steps: STEPS,
        a: 1,
        // A slow rightward drift, so the cart covers ground and the log is
        // informative about r as well as q.
        b: 0.05,
        r: TRUE_R,
        q: TRUE_Q,
        x0: 0,
        sigma0: 0.5,
        control: (t) => (t < STEPS / 2 ? 1 : -1),
      },
      rng,
    );
    const kf = new DiffKf1d(START_LOG_R, START_LOG_Q);
    const g = kf.lossAndGrad(log, 'evidence');
    return {
      rng,
      log,
      kf,
      epoch: 0,
      loss: g.loss,
      gradR: g.dLogR,
      gradQ: g.dLogQ,
      history: [{ epoch: 0, loss: g.loss, r: kf.r, q: kf.q }],
    };
  }, []);

  const step = useCallback(
    (s: State): State => {
      // One full-batch gradient step per tick. The learning rate is divided by
      // the trajectory length so that a reader who lengthens the log does not
      // silently change the step size.
      const g = s.kf.sgdStep(s.log, params.loss, params.lr / s.log.z.length);
      const epoch = s.epoch + 1;
      return {
        ...s,
        epoch,
        loss: g.loss,
        gradR: g.dLogR,
        gradQ: g.dLogQ,
        history: [...s.history, { epoch, loss: g.loss, r: s.kf.r, q: s.kf.q }].slice(-HISTORY),
      };
    },
    [params.lr, params.loss],
  );

  const sim = useSimulation<State>({ init, step, fps: 14, initialSeed: 7 });

  /* ---- the filter run at the current θ, for the canvas -------------------- */

  const run = useMemo(() => sim.state.kf.run(sim.state.log), [sim.state]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { log } = sim.state;
      if (run.length < 2) return;

      const px = (u: number) => sx(v, u * ASPECT);
      const py = (u: number) => sy(v, u);

      let lo = Infinity;
      let hi = -Infinity;
      for (let t = 0; t < run.length; t++) {
        const sd = Math.sqrt(run[t].sigma2);
        lo = Math.min(lo, log.x[t], log.z[t], run[t].mu - 2 * sd);
        hi = Math.max(hi, log.x[t], log.z[t], run[t].mu + 2 * sd);
      }
      const pad = 0.08 * (hi - lo) + 0.1;
      lo -= pad;
      hi += pad;

      const tx = (i: number) => px(0.02 + 0.96 * (i / (run.length - 1)));
      const ty = (m: number) => py(0.08 + 0.86 * ((m - lo) / (hi - lo)));

      // The ±2σ band the filter is currently claiming.
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.16;
      ctx.beginPath();
      run.forEach((s, i) => {
        const y = ty(s.mu + 2 * Math.sqrt(s.sigma2));
        if (i === 0) ctx.moveTo(tx(i), y);
        else ctx.lineTo(tx(i), y);
      });
      for (let i = run.length - 1; i >= 0; i--) {
        ctx.lineTo(tx(i), ty(run[i].mu - 2 * Math.sqrt(run[i].sigma2)));
      }
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      // The measurements the evidence loss is actually scored against.
      ctx.fillStyle = p.measurement;
      ctx.globalAlpha = 0.6;
      log.z.forEach((z, i) => {
        ctx.beginPath();
        ctx.arc(tx(i), ty(z), 1.3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;

      // Truth — drawn for the reader, never shown to the evidence loss.
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      log.x.forEach((x, i) => {
        const y = ty(x);
        if (i === 0) ctx.moveTo(tx(i), y);
        else ctx.lineTo(tx(i), y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // The posterior mean.
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 2;
      ctx.beginPath();
      run.forEach((s, i) => {
        const y = ty(s.mu);
        if (i === 0) ctx.moveTo(tx(i), y);
        else ctx.lineTo(tx(i), y);
      });
      ctx.stroke();

      // Every step where the truth escapes the filter's own 2σ band. A healthy
      // filter marks about 5% of them; an overconfident one marks most.
      let outside = 0;
      ctx.fillStyle = p.prediction;
      run.forEach((s, i) => {
        if (Math.abs(log.x[i] - s.mu) <= 2 * Math.sqrt(s.sigma2)) return;
        outside += 1;
        ctx.beginPath();
        ctx.arc(tx(i), ty(log.x[i]), 2.2, 0, Math.PI * 2);
        ctx.fill();
      });

      label(ctx, `epoch ${sim.state.epoch}`, px(0.035), py(0.96), p.ink, { size: 10, weight: 600 });
      label(
        ctx,
        `truth outside the claimed 2σ band: ${((100 * outside) / run.length).toFixed(0)}%  (honest ≈ 5%)`,
        px(0.035),
        py(0.90),
        outside / run.length > 0.12 ? p.prediction : p.truth,
        { size: 10 },
      );
      label(ctx, 'position (m) vs. time', px(1.965), py(0.96), p.ink, { size: 9, align: 'right' });
    },
    [sim.state, run],
  );

  /* ---- readouts ---------------------------------------------------------- */

  const stats = useMemo(() => {
    const { kf, log, history } = sim.state;
    return {
      loss: sim.state.loss / log.z.length,
      r: kf.r,
      q: kf.q,
      nis: meanNis(kf, log),
      nees: meanNees(kf, log),
      lossSpark: history.slice(-40).map((h) => h.loss / log.z.length),
    };
  }, [sim.state]);

  const lossSeries = useMemo(
    () => [
      {
        id: params.loss === 'evidence' ? 'evidence loss (no ground truth)' : 'state NLL (supervised)',
        role: 'posterior' as const,
        data: sim.state.history.map((h) => ({ x: h.epoch, y: h.loss / STEPS })),
      },
    ],
    [sim.state.history, params.loss],
  );

  const paramSeries = useMemo(
    () => [
      {
        id: 'r  (process noise)',
        role: 'prediction' as const,
        data: sim.state.history.map((h) => ({ x: h.epoch, y: h.r })),
      },
      {
        id: 'q  (measurement noise)',
        role: 'measurement' as const,
        data: sim.state.history.map((h) => ({ x: h.epoch, y: h.q })),
      },
    ],
    [sim.state.history],
  );

  return (
    <WidgetFrame
      id="w25.2"
      title="The Differentiable Filter Trainer"
      teaches="Tuning a filter and training a network are the same act: gradient descent on a proper scoring rule, straight through the five Kalman equations."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The filter starts wrong in both directions — it believes the cart is wildly unpredictable
          (r&nbsp;=&nbsp;0.8) and the range sensor is nearly perfect (q&nbsp;=&nbsp;0.02) — so its
          purple band is far too tight and the truth spends most of the run outside it. Press play
          and watch the band breathe out to the width it should have had. Nothing here is search or
          heuristics: each epoch is one exact gradient of the evidence loss with respect to
          (log&nbsp;r,&nbsp;log&nbsp;q), obtained by carrying ∂μ/∂θ and ∂σ²/∂θ alongside the
          filter&rsquo;s own recursion. The two dashed markers on the lower-right chart are the
          noises that actually generated the log; the trainer never sees them, and neither does the
          evidence loss, which is scored purely on one-step-ahead predictions of measurements the
          robot really took. Flip to the <strong>state NLL</strong> and the trainer starts using
          ground truth — a luxury available in simulation and in a motion-capture room, and nowhere
          else. Then hit <strong>perturb θ</strong> and watch it come back.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: ASPECT, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state, run]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="A time series of the cart's true position, its noisy measurements, and the filter's estimate with a two-sigma band that widens from far too tight to honest as training proceeds."
      />

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 lg:grid-cols-4">
        <StatTile
          label="loss per step"
          value={stats.loss}
          precision={3}
          role="posterior"
          sparkline={stats.lossSpark}
        />
        <StatTile
          label="r — learned / true"
          value={`${stats.r.toFixed(3)} / ${TRUE_R}`}
          role="prediction"
        />
        <StatTile
          label="q — learned / true"
          value={`${stats.q.toFixed(3)} / ${TRUE_Q}`}
          role="measurement"
        />
        <StatTile
          label="mean NEES (want ≈ 1)"
          value={stats.nees}
          precision={2}
          role={Math.abs(stats.nees - 1) > 0.4 ? 'prediction' : undefined}
        />
      </div>

      <div className="grid gap-0 border-t border-fd-border lg:grid-cols-2 lg:divide-x lg:divide-fd-border">
        <div className="p-3">
          <p className="eyebrow mb-1">Loss per step</p>
          <LineChart
            series={lossSeries}
            xLabel="epoch"
            yLabel="nats"
            height={220}
            ariaLabel="The training loss falling as gradient descent adjusts the two noise parameters."
          />
        </div>
        <div className="border-t border-fd-border p-3 lg:border-t-0">
          <p className="eyebrow mb-1">Where θ is going</p>
          <LineChart
            series={paramSeries}
            xLabel="epoch"
            yLabel="variance"
            height={220}
            markers={[
              { axis: 'y', value: TRUE_R, label: 'true r', role: 'truth' },
              { axis: 'y', value: TRUE_Q, label: 'true q', role: 'truth' },
            ]}
            ariaLabel="The learned process and measurement noise variances converging on the values that generated the log."
          />
        </div>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="learning rate"
          value={params.lr}
          min={0.05}
          max={3}
          step={0.05}
          onChange={(v) => setParams((p) => ({ ...p, lr: v }))}
          help="Per-sample step on log r and log q. Each step is clipped to a factor of e^0.5, so a large rate is fast rather than fatal."
        />
        <Toggle
          label="cheat: use ground truth (state NLL)"
          role="truth"
          checked={params.loss === 'stateNll'}
          onChange={(v) => setParams((p) => ({ ...p, loss: v ? 'stateNll' : 'evidence' }))}
        />
        <ButtonRow>
          <ActionButton
            onClick={() =>
              sim.setState((s) => {
                s.kf.logR += s.rng.normal(0, 1.4);
                s.kf.logQ += s.rng.normal(0, 1.4);
                const g = s.kf.lossAndGrad(s.log, params.loss);
                return {
                  ...s,
                  loss: g.loss,
                  gradR: g.dLogR,
                  gradQ: g.dLogQ,
                  history: [{ epoch: s.epoch, loss: g.loss, r: s.kf.r, q: s.kf.q }],
                };
              })
            }
          >
            perturb θ
          </ActionButton>
          <ActionButton onClick={sim.reset}>reset θ</ActionButton>
        </ButtonRow>
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
