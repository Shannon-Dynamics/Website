'use client';

import Link from 'next/link';
import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT } from '@/lib/sim/world';
import {
  Mppi,
  RUSTY_LIMITS,
  TrackAndClear,
  DEFAULT_TRACK_PARAMS,
  control2,
  executeStep,
  moveObstacles,
  rollout,
  type Control2,
  type CostModel,
  type MppiResult,
  type Obstacle,
} from '@/lib/control/mppi';
import { CORRIDOR_BAND, apartmentField, corridorRun } from '@/lib/control/scenes';
import type { Pose2 } from '@/lib/geom/se2';
import {
  clear,
  drawPath,
  drawRobot,
  drawWorld,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w23.1 — Rollout Storm.
 *
 * One MPPI control cycle, every frame, on the real controller from
 * `lib/control/mppi.ts`. The palette *is* the legend and it matches the twin
 * table in the Foundation section term for term: the plan the cycle inherits is
 * the prior (blue), the perturbed rollouts are the prediction (orange), the
 * cost field the rollouts are scored against is the measurement (green), and
 * the weighted blend that gets executed is the posterior (purple).
 */

const SCENE = corridorRun();
const DT = 0.1;
const ROBOT_R = DEFAULT_TRACK_PARAMS.robotRadius;

interface Params {
  lambda: number;
  samples: number;
  sigmaV: number;
  horizon: number;
  slowMotion: boolean;
}

/** Sub-steps of one cycle, animated when the reader asks for slow motion. */
type Phase = 'storm' | 'cost' | 'collapse' | 'execute';
const PHASES: Phase[] = ['storm', 'cost', 'collapse', 'execute'];

interface State {
  mppi: Mppi;
  rng: Rng;
  pose: Pose2;
  obstacles: Obstacle[];
  result: MppiResult | null;
  cost: TrackAndClear | null;
  phase: Phase;
  cycles: number;
  collisions: number;
  minClearance: number;
  clearance: number;
  crossTrack: number;
  arrived: boolean;
}

const makeMppi = (p: Params) =>
  new Mppi({
    horizon: p.horizon,
    samples: p.samples,
    dt: DT,
    lambda: p.lambda,
    sigmaV: p.sigmaV,
    sigmaOmega: p.sigmaV * 2.5,
    limits: RUSTY_LIMITS,
    // α = 1: the base distribution *is* the current plan, so the control-cost
    // cross term vanishes and λ is free to mean only "how sharp is the tilt".
    gamma: 0,
  });

export function RolloutStorm() {
  const [params, setParams] = useState<Params>({
    lambda: 8,
    samples: 200,
    sigmaV: 0.22,
    horizon: 25,
    slowMotion: false,
  });

  const init = useCallback((seed: number): State => {
    const mppi = makeMppi({ lambda: 8, samples: 200, sigmaV: 0.22, horizon: 25, slowMotion: false });
    mppi.reset(control2(0.3, 0));
    return {
      mppi,
      rng: new Rng(seed),
      pose: { ...SCENE.start },
      obstacles: SCENE.obstacles.map((o) => ({ ...o })),
      result: null,
      cost: null,
      phase: 'storm',
      cycles: 0,
      collisions: 0,
      minClearance: Infinity,
      clearance: Infinity,
      crossTrack: 0,
      arrived: false,
    };
  }, []);

  const step = useCallback(
    (s: State): State => {
      const { mppi } = s;
      // Live parameter edits: the controller object survives, its knobs move.
      mppi.cfg.lambda = params.lambda;
      mppi.cfg.samples = params.samples;
      mppi.cfg.sigmaV = params.sigmaV;
      mppi.cfg.sigmaOmega = params.sigmaV * 2.5;
      if (mppi.cfg.horizon !== params.horizon) {
        mppi.cfg.horizon = params.horizon;
        mppi.reset(control2(s.result?.applied.v ?? 0.3, 0));
      }

      // In slow motion the same cycle is held for three frames so the reader
      // can watch the storm, then the costing, then the collapse.
      if (params.slowMotion && s.result) {
        const idx = PHASES.indexOf(s.phase);
        if (idx < PHASES.length - 1) return { ...s, phase: PHASES[idx + 1] };
      }

      if (s.arrived) return s;

      const obstacles = moveObstacles(s.obstacles, DT, CORRIDOR_BAND);
      const cost = new TrackAndClear({
        ...DEFAULT_TRACK_PARAMS,
        path: SCENE.path,
        field: apartmentField(),
        obstacles,
      });

      const result = mppi.plan(s.pose, cost, s.rng);
      const exec = executeStep(s.pose, result.applied, DT, APARTMENT, obstacles, ROBOT_R);
      mppi.shift();

      const clearance = cost.clearance(exec.pose.x, exec.pose.y);
      const arrived = Math.hypot(exec.pose.x - SCENE.goal.x, exec.pose.y - SCENE.goal.y) < 0.32;

      return {
        ...s,
        obstacles,
        cost,
        result,
        pose: exec.pose,
        phase: 'storm',
        cycles: s.cycles + 1,
        collisions: s.collisions + (exec.collided ? 1 : 0),
        clearance,
        minClearance: Math.min(s.minClearance, clearance),
        crossTrack: cost.project(exec.pose.x, exec.pose.y).cross,
        arrived,
      };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 10, initialSeed: 23 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { pose, obstacles, result, cost } = sim.state;
      // Outside slow motion every frame shows the finished cycle, so the storm
      // is always drawn weighted — the picture the reader is meant to read.
      const phase: Phase = params.slowMotion ? sim.state.phase : 'collapse';

      // --- the obstacle cost field, sampled coarsely (measurement green) ----
      if (cost) {
        const cell = 0.1;
        const px = Math.ceil(sl(v, cell)) + 1;
        ctx.save();
        ctx.fillStyle = p.measurement;
        for (let x = 0.2; x < 11.9; x += cell) {
          for (let y = 3.4; y < 5.4; y += cell) {
            const c = cost.clearance(x, y);
            if (c > 0.45) continue;
            // exp(−d/σ) is what the rollouts pay; draw exactly that.
            ctx.globalAlpha = 0.42 * Math.min(1, Math.exp(-Math.max(c, 0) / DEFAULT_TRACK_PARAMS.sigmaObs));
            ctx.fillRect(sx(v, x), sy(v, y + cell), px, px);
          }
        }
        ctx.restore();
      }

      drawWorld(ctx, v, APARTMENT, p);

      // --- the reference path from Chapter 20 (ground truth, gray dashed) ---
      drawPath(ctx, v, SCENE.path, p.truth, { dashed: true, lineWidth: 1.5, alpha: 0.85 });

      // --- clutter ---------------------------------------------------------
      ctx.save();
      for (const o of obstacles) {
        ctx.fillStyle = p.wall;
        ctx.globalAlpha = o.vx || o.vy ? 0.55 : 0.8;
        ctx.beginPath();
        ctx.arc(sx(v, o.x), sy(v, o.y), sl(v, o.r), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      if (result) {
        const wMax = result.samples.reduce((m, s) => Math.max(m, s.weight), 1e-9);

        // --- the storm (prediction orange), opacity ∝ weight ---------------
        ctx.save();
        ctx.lineWidth = 1;
        for (const smp of result.samples) {
          const w = smp.weight / wMax;
          if (phase === 'storm') {
            ctx.strokeStyle = p.prediction;
            ctx.globalAlpha = 0.1;
          } else if (phase === 'cost') {
            // Costing pass: tint each rollout by how expensive it turned out.
            const rel = (smp.stateCost - result.sMin) / Math.max(result.sMean - result.sMin, 1e-6);
            ctx.strokeStyle = rel < 1 ? p.measurement : p.prediction;
            ctx.globalAlpha = 0.09 + 0.2 * Math.max(0, 1 - rel);
          } else {
            ctx.strokeStyle = p.prediction;
            ctx.globalAlpha = 0.05 + 0.6 * w;
          }
          ctx.beginPath();
          const pts = smp.states;
          ctx.moveTo(sx(v, pts[0].x), sy(v, pts[0].y));
          for (let k = 1; k < pts.length; k++) ctx.lineTo(sx(v, pts[k].x), sy(v, pts[k].y));
          ctx.stroke();
        }
        ctx.restore();

        // --- the plan this cycle inherited (prior blue) --------------------
        if (phase !== 'storm') {
          const prevStates = rolloutPreview(pose, result.previous);
          drawPath(ctx, v, prevStates, p.prior, { lineWidth: 2, alpha: 0.75, dashed: true });
        }

        // --- the executed plan (posterior purple) --------------------------
        if (phase === 'collapse' || phase === 'execute') {
          drawPath(ctx, v, result.updatedStates, p.posterior, { lineWidth: 2.6 });
          const head = result.updatedStates[result.updatedStates.length - 1];
          ctx.save();
          ctx.fillStyle = p.posterior;
          ctx.beginPath();
          ctx.arc(sx(v, head.x), sy(v, head.y), 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        }
      }

      drawRobot(ctx, v, pose, p.truth, 0.24);

      const caption = !params.slowMotion
        ? 'one full MPPI cycle per frame'
        : phase === 'storm'
          ? 'SAMPLE   K rollouts from the plan'
          : phase === 'cost'
            ? 'SCORE   S(V) for every rollout'
            : phase === 'collapse'
              ? 'UPDATE   u ← u + Σ wᵢ εᵢ'
              : 'EXECUTE  u₀, then shift';
      label(ctx, caption, 12, 16, p.ink, { size: 11, weight: 600 });
      if (sim.state.arrived) label(ctx, 'goal reached', 12, 32, p.posterior, { size: 11, weight: 600 });
    },
    [sim.state, params.slowMotion],
  );

  const stats = useMemo(() => {
    const r = sim.state.result;
    return {
      ess: r ? r.ess : 0,
      essFrac: r ? r.ess / Math.max(r.samples.length, 1) : 0,
      cost: r ? r.sMin : 0,
      speed: r ? r.applied.v : 0,
      clearance: Number.isFinite(sim.state.minClearance) ? sim.state.minClearance : 0,
      cross: sim.state.crossTrack,
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w23.1"
      title="Rollout Storm"
      teaches="MPPI is importance sampling over control sequences — and the command it executes is a weighted blend of the rollouts, never one of them."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty is tracking the gray dashed path that <Link href="/chapters/ch20-motion-planning">Chapter 20</Link>{' '}
          planned before anyone moved the chairs. Every frame is one complete control cycle: two
          hundred perturbed plans fan out in orange, each is scored against the green obstacle-cost
          field, and the purple curve is their weight-averaged blend — smoother than any single
          rollout, which is the visual signature of an <em>expectation</em> rather than a choice.
          <br />
          <strong>Drag the temperature λ.</strong> Near 0.5 the weights collapse onto a single lucky
          rollout (watch ESS fall to ~2 of 200) and the purple plan starts to twitch from frame to
          frame. Above 60 the weights go nearly uniform, the plan barely responds to the cost, and
          Rusty drifts toward the chairs — the run takes twice as long and passes half as wide.
          Turn on <em>slow motion</em> to hold each cycle for four frames and watch the storm, the
          scoring, and the collapse separately.
        </>
      }
    >
      <SimCanvas
        world={SCENE.view}
        draw={draw}
        deps={[sim.tick, sim.state, params.slowMotion]}
        aspect={3.1}
        padding={0.15}
        ariaLabel="A robot in an apartment corridor. Hundreds of orange candidate trajectories fan out ahead of it and collapse into a single purple executed plan that curves around two chairs and a moving obstacle."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="effective sample size" value={`${stats.ess.toFixed(1)} / ${params.samples}`} tint={stats.essFrac < 0.05 ? 'prediction' : undefined} />
        <Stat label="best rollout cost" value={stats.cost.toFixed(0)} />
        <Stat label="min clearance" value={`${stats.clearance.toFixed(2)} m`} tint={stats.clearance < 0.06 ? 'prediction' : undefined} />
        <Stat label="cross-track" value={`${stats.cross.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Temperature λ"
          role="posterior"
          value={params.lambda}
          min={0.5}
          max={120}
          step={0.5}
          onChange={(x) => setParams((q) => ({ ...q, lambda: x }))}
          help="How sharply the exponential tilt exp(−S/λ) favours cheap rollouts. This is the widget's one real knob."
        />
        <Slider
          label="Rollouts K"
          role="prediction"
          value={params.samples}
          min={16}
          max={400}
          step={8}
          format={(x) => x.toFixed(0)}
          onChange={(x) => setParams((q) => ({ ...q, samples: Math.round(x) }))}
        />
        <Slider
          label="Exploration σ_v"
          role="prediction"
          value={params.sigmaV}
          min={0.04}
          max={0.5}
          step={0.01}
          unit="m/s"
          onChange={(x) => setParams((q) => ({ ...q, sigmaV: x }))}
          help="σ_ω is held at 2.5 σ_v. Too small and no rollout ever finds the gap."
        />
        <Slider
          label="Horizon H"
          value={params.horizon}
          min={6}
          max={45}
          step={1}
          format={(x) => `${x.toFixed(0)} (${(x * DT).toFixed(1)} s)`}
          onChange={(x) => setParams((q) => ({ ...q, horizon: Math.round(x) }))}
        />
        <Toggle
          label="Slow motion (one cycle, four frames)"
          checked={params.slowMotion}
          onChange={(x) => setParams((q) => ({ ...q, slowMotion: x }))}
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

/** Integrate a plan for drawing only — the library's dynamics, a free cost. */
const FREE: CostModel = { stage: () => 0, terminal: () => 0 };
const rolloutPreview = (x0: Pose2, plan: Control2[]) =>
  rollout(x0, plan, FREE, DT, RUSTY_LIMITS).states;

function Stat({ label: l, value, tint }: { label: string; value: string; tint?: 'prediction' | 'posterior' }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums" style={tint ? { color: `var(--pr-${tint})` } : undefined}>
        {value}
      </div>
    </div>
  );
}
