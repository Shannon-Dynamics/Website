'use client';

import Link from 'next/link';
import { useCallback, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import {
  DEFAULT_TRACK_PARAMS,
  Mppi,
  RUSTY_LIMITS,
  SeekAndClear,
  TrackAndClear,
  control2,
  executeStep,
  moveObstacles,
  type Control2,
  type CostModel,
  type MppiResult,
  type Obstacle,
} from '@/lib/control/mppi';
import { DEFAULT_DWA_CONFIG, dwaPlan, type DwaResult } from '@/lib/control/dwa';
import { CORRIDOR_BAND, apartmentField, corridorRun, counterPocket } from '@/lib/control/scenes';
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
 * w23.2 — Dynamic Window Bench.
 *
 * Left: the velocity plane, where both controllers actually live. DWA lays a
 * grid over the box of commands it can reach this period and scores each one
 * along its single arc. MPPI draws a Gaussian cloud of *first* commands, each
 * one the head of a whole H-step sequence, and weights them by what the rest of
 * that sequence costs. Same axes, completely different hypothesis class.
 *
 * Right: the world those commands produce.
 */

type Controller = 'dwa' | 'mppi';
type SceneKey = 'corridor' | 'pocket';

const DT = 0.1;
const VIEW_V = { minX: -RUSTY_LIMITS.omegaMax * 1.12, maxX: RUSTY_LIMITS.omegaMax * 1.12, minY: RUSTY_LIMITS.vMin - 0.12, maxY: RUSTY_LIMITS.vMax + 0.12 };

interface Params {
  controller: Controller;
  scene: SceneKey;
  clearanceWeight: number;
  /** Fox et al.'s distance-to-contact term, or the modern minimum-margin one. */
  marginMode: boolean;
}

interface State {
  scene: SceneKey;
  controller: Controller;
  pose: Pose2;
  u: Control2;
  obstacles: Obstacle[];
  cost: CostModel & { clearance(x: number, y: number): number };
  dwa: DwaResult | null;
  mppi: MppiResult | null;
  planner: Mppi;
  rng: Rng;
  steps: number;
  minClearance: number;
  distance: number;
  arrived: boolean;
}

const sceneOf = (k: SceneKey) => (k === 'corridor' ? corridorRun() : counterPocket());

function makeCost(k: SceneKey, obstacles: Obstacle[]) {
  const field = apartmentField();
  const s = sceneOf(k);
  return k === 'corridor'
    ? new TrackAndClear({ ...DEFAULT_TRACK_PARAMS, path: s.path, field, obstacles })
    : new SeekAndClear(s.goal, field, obstacles);
}

function buildState(k: SceneKey, controller: Controller, seed: number): State {
  const s = sceneOf(k);
  const obstacles = s.obstacles.map((o) => ({ ...o }));
  const planner = new Mppi({
    horizon: 30,
    samples: 200,
    dt: DT,
    lambda: 8,
    sigmaV: 0.22,
    sigmaOmega: 0.6,
    limits: RUSTY_LIMITS,
    gamma: 0,
  });
  planner.reset(control2(0.2, 0));
  return {
    scene: k,
    controller,
    pose: { ...s.start },
    u: control2(0, 0),
    obstacles,
    cost: makeCost(k, obstacles),
    dwa: null,
    mppi: null,
    planner,
    rng: new Rng(seed),
    steps: 0,
    minClearance: Infinity,
    distance: Math.hypot(s.start.x - s.goal.x, s.start.y - s.goal.y),
    arrived: false,
  };
}

export function DynamicWindowBench() {
  const [params, setParams] = useState<Params>({
    controller: 'dwa',
    scene: 'corridor',
    clearanceWeight: 0.2,
    marginMode: false,
  });

  const init = useCallback((seed: number): State => buildState('corridor', 'dwa', seed), []);

  const step = useCallback(
    (s: State): State => {
      // Scene or controller switched: rebuild rather than patch. A controller
      // carrying a warm-started plan into a different room is a bug, not a demo.
      if (s.scene !== params.scene || s.controller !== params.controller) {
        return buildState(params.scene, params.controller, s.rng.next() * 1e6);
      }
      if (s.arrived) return s;

      const scene = sceneOf(s.scene);
      const obstacles = s.scene === 'corridor' ? moveObstacles(s.obstacles, DT, CORRIDOR_BAND) : s.obstacles;
      const cost = makeCost(s.scene, obstacles);

      let u: Control2;
      let dwa: DwaResult | null = null;
      let mppi: MppiResult | null = null;

      if (params.controller === 'dwa') {
        dwa = dwaPlan(s.pose, scene.goal, s.u, (x, y) => cost.clearance(x, y), {
          ...DEFAULT_DWA_CONFIG,
          dt: 0.25,
          limits: RUSTY_LIMITS,
          weights: { heading: 0.8, clearance: params.clearanceWeight, velocity: 0.15 },
          clearanceMode: params.marginMode ? 'margin' : 'contact',
        });
        u = dwa.best ? control2(dwa.best.v, dwa.best.omega) : control2(0, 0);
      } else {
        mppi = s.planner.plan(s.pose, cost, s.rng);
        u = mppi.applied;
        s.planner.shift();
      }

      const exec = executeStep(s.pose, u, DT, APARTMENT, obstacles, DEFAULT_TRACK_PARAMS.robotRadius);
      const distance = Math.hypot(exec.pose.x - scene.goal.x, exec.pose.y - scene.goal.y);
      return {
        ...s,
        obstacles,
        cost,
        u,
        dwa,
        mppi,
        pose: exec.pose,
        steps: s.steps + 1,
        minClearance: Math.min(s.minClearance, cost.clearance(exec.pose.x, exec.pose.y)),
        distance,
        arrived: distance < 0.32,
      };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 10, initialSeed: 23 });
  const scene = sceneOf(sim.state.scene);

  /* ----------------------------------------------------- the velocity plane */
  const drawVelocity = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;

      // Axes.
      ctx.save();
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, RUSTY_LIMITS.vMin));
      ctx.lineTo(sx(v, 0), sy(v, RUSTY_LIMITS.vMax));
      ctx.moveTo(sx(v, -RUSTY_LIMITS.omegaMax), sy(v, 0));
      ctx.lineTo(sx(v, RUSTY_LIMITS.omegaMax), sy(v, 0));
      ctx.stroke();
      // V_s: everything the actuators allow.
      ctx.strokeStyle = p.wall;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(
        sx(v, -RUSTY_LIMITS.omegaMax),
        sy(v, RUSTY_LIMITS.vMax),
        sl(v, 2 * RUSTY_LIMITS.omegaMax),
        sy(v, RUSTY_LIMITS.vMin) - sy(v, RUSTY_LIMITS.vMax),
      );
      ctx.setLineDash([]);
      ctx.restore();

      if (s.controller === 'dwa' && s.dwa) {
        const { candidates, window: win, nV, nOmega } = s.dwa;
        const cw = sl(v, (win.omegaHi - win.omegaLo) / Math.max(nOmega - 1, 1)) + 1;
        const ch = Math.abs(sy(v, 0) - sy(v, (win.vHi - win.vLo) / Math.max(nV - 1, 1))) + 1;
        let lo = Infinity;
        let hi = -Infinity;
        for (const c of candidates) {
          if (!c.admissible) continue;
          lo = Math.min(lo, c.score);
          hi = Math.max(hi, c.score);
        }
        ctx.save();
        for (const c of candidates) {
          const px = sx(v, c.omega) - cw / 2;
          const py = sy(v, c.v) - ch / 2;
          if (!c.admissible) {
            // Inadmissible: cannot stop before the first obstacle on that arc.
            ctx.fillStyle = p.truth;
            ctx.globalAlpha = 0.28;
          } else {
            ctx.fillStyle = p.measurement;
            ctx.globalAlpha = 0.12 + 0.75 * ((c.score - lo) / Math.max(hi - lo, 1e-9));
          }
          ctx.fillRect(px, py, cw, ch);
        }
        ctx.restore();

        // V_d itself — the prior on what is reachable this period.
        ctx.save();
        ctx.strokeStyle = p.prior;
        ctx.lineWidth = 1.6;
        ctx.strokeRect(
          sx(v, win.omegaLo),
          sy(v, win.vHi),
          sl(v, win.omegaHi - win.omegaLo),
          sy(v, win.vLo) - sy(v, win.vHi),
        );
        ctx.restore();

        if (s.dwa.best) star(ctx, sx(v, s.dwa.best.omega), sy(v, s.dwa.best.v), 6, p.posterior);
      } else if (s.mppi) {
        const r = s.mppi;
        const wMax = r.samples.reduce((m, x) => Math.max(m, x.weight), 1e-9);
        ctx.save();
        for (const smp of r.samples) {
          const u0v = r.previous[0].v + smp.eps[0].v;
          const u0w = r.previous[0].omega + smp.eps[0].omega;
          ctx.fillStyle = p.prediction;
          ctx.globalAlpha = 0.1 + 0.8 * (smp.weight / wMax);
          ctx.beginPath();
          ctx.arc(sx(v, u0w), sy(v, u0v), 1.6 + 3 * Math.sqrt(smp.weight / wMax), 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();

        ctx.save();
        ctx.fillStyle = p.prior;
        ctx.beginPath();
        ctx.arc(sx(v, r.previous[0].omega), sy(v, r.previous[0].v), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        star(ctx, sx(v, r.applied.omega), sy(v, r.applied.v), 6, p.posterior);
      }

      label(ctx, 'ω  (rad/s)', v.width - 8, sy(v, 0) - 10, p.ink, { size: 9, align: 'right' });
      label(ctx, 'v (m/s)', sx(v, 0) + 6, 12, p.ink, { size: 9 });
      label(
        ctx,
        s.controller === 'dwa' ? 'velocity space: 651 arcs, one step' : 'velocity space: 200 sequence heads',
        8,
        v.height - 8,
        p.ink,
        { size: 9.5, weight: 600 },
      );
    },
    [sim.state],
  );

  /* -------------------------------------------------------------- the world */
  const drawWorldPanel = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;
      const sc = sceneOf(s.scene);
      drawWorld(ctx, v, APARTMENT, p);
      if (sc.path.length > 1) drawPath(ctx, v, sc.path, p.truth, { dashed: true, lineWidth: 1.4, alpha: 0.8 });

      ctx.save();
      for (const o of s.obstacles) {
        ctx.fillStyle = p.wall;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(sx(v, o.x), sy(v, o.y), sl(v, o.r), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Goal.
      ctx.save();
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(sx(v, sc.goal.x), sy(v, sc.goal.y), sl(v, 0.16), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      if (s.controller === 'dwa' && s.dwa) {
        let lo = Infinity;
        let hi = -Infinity;
        for (const c of s.dwa.candidates) {
          if (!c.admissible) continue;
          lo = Math.min(lo, c.score);
          hi = Math.max(hi, c.score);
        }
        ctx.save();
        ctx.lineWidth = 1;
        for (const c of s.dwa.candidates) {
          if (!c.admissible) continue;
          ctx.strokeStyle = p.prediction;
          ctx.globalAlpha = 0.06 + 0.35 * ((c.score - lo) / Math.max(hi - lo, 1e-9));
          ctx.beginPath();
          ctx.moveTo(sx(v, c.arc[0].x), sy(v, c.arc[0].y));
          for (let k = 1; k < c.arc.length; k += 2) ctx.lineTo(sx(v, c.arc[k].x), sy(v, c.arc[k].y));
          ctx.stroke();
        }
        ctx.restore();
        if (s.dwa.best) drawPath(ctx, v, s.dwa.best.arc, p.posterior, { lineWidth: 2.4 });
      } else if (s.mppi) {
        const wMax = s.mppi.samples.reduce((m, x) => Math.max(m, x.weight), 1e-9);
        ctx.save();
        ctx.lineWidth = 1;
        ctx.strokeStyle = p.prediction;
        for (const smp of s.mppi.samples) {
          ctx.globalAlpha = 0.05 + 0.5 * (smp.weight / wMax);
          ctx.beginPath();
          ctx.moveTo(sx(v, smp.states[0].x), sy(v, smp.states[0].y));
          for (let k = 1; k < smp.states.length; k++) ctx.lineTo(sx(v, smp.states[k].x), sy(v, smp.states[k].y));
          ctx.stroke();
        }
        ctx.restore();
        drawPath(ctx, v, s.mppi.updatedStates, p.posterior, { lineWidth: 2.4 });
      }

      drawRobot(ctx, v, s.pose, p.truth, 0.24);
      if (s.arrived) label(ctx, 'goal reached', 10, 14, p.posterior, { size: 10, weight: 600 });
      else if (s.steps > 120 && s.distance > 0.9)
        label(ctx, 'stalled', 10, 14, p.prediction, { size: 10, weight: 600 });
    },
    [sim.state],
  );

  const admissible = sim.state.dwa
    ? sim.state.dwa.candidates.filter((c) => c.admissible).length
    : null;

  return (
    <WidgetFrame
      id="w23.2"
      title="Dynamic Window Bench"
      teaches="DWA searches one arc at a time inside the reachable velocity box; MPPI searches whole sequences. Neither one is a planner."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Left is the velocity plane: the dashed box is everything the motors allow, the blue box is
          the <em>dynamic window</em> — the commands reachable in one 0.25 s period at Rusty&apos;s
          acceleration limits. Gray cells are inadmissible (he could not brake before the first
          obstacle on that arc); green is the DWA objective; the purple star is the winner. Switch to
          MPPI and the same axes fill with the <em>first</em> commands of two hundred sampled
          sequences, sized by weight — note that they spill outside the blue box, because plain MPPI
          constrains velocity but not acceleration.
          <br />
          <strong>What to try.</strong> Run DWA down the corridor and drag its clearance weight from
          0.02 to 0.5. <em>Nothing happens.</em> Fox et al.&apos;s clearance term is the distance to
          the first <em>contact</em> along the arc, so squeezing past a chair with five centimetres
          to spare scores exactly as well as sailing past with fifty — the term is saturated and its
          weight is inert. Now switch on <em>minimum-margin clearance</em>, the modern repair, and the
          weight bites: above about 0.1 the robot stops dead 0.7 m short of the chair. There is no
          setting in between, because the maneuver that keeps margin <em>and</em> makes progress is an
          S-curve, and an S-curve is not one arc. MPPI, whose hypothesis is a 3-second sequence, just
          does it.
          <br />
          Then switch to the counter pocket, where the goal is two metres north through a wall.{' '}
          <em>Both</em> controllers stall — the escape costs 1.9 m of travel in the wrong direction,
          and no local controller will pay that. Finding the way out is{' '}
          <Link href="/chapters/ch20-motion-planning">Chapter 20</Link>&apos;s job; this chapter only
          executes it.
        </>
      }
    >
      <div className="grid grid-cols-1 gap-0 md:grid-cols-2 md:divide-x md:divide-fd-border">
        <SimCanvas
          world={VIEW_V}
          draw={drawVelocity}
          deps={[sim.tick, sim.state]}
          aspect={1.25}
          padding={0.06}
          ariaLabel="The velocity plane. For DWA, a grid of candidate velocity commands shaded by objective value with inadmissible ones grayed out. For MPPI, a cloud of sampled commands sized by importance weight."
        />
        <SimCanvas
          world={scene.view}
          draw={drawWorldPanel}
          deps={[sim.tick, sim.state]}
          aspect={1.25}
          padding={0.1}
          ariaLabel="The apartment, with the candidate trajectories each controller is considering and the one it chose."
        />
      </div>

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="controller" value={params.controller === 'dwa' ? 'DWA' : 'MPPI'} />
        <Stat
          label={params.controller === 'dwa' ? 'admissible' : 'effective sample size'}
          value={
            params.controller === 'dwa'
              ? `${admissible ?? 0} / ${sim.state.dwa?.candidates.length ?? 0}`
              : `${(sim.state.mppi?.ess ?? 0).toFixed(1)} / 200`
          }
        />
        <Stat
          label="min clearance"
          value={Number.isFinite(sim.state.minClearance) ? `${sim.state.minClearance.toFixed(2)} m` : '—'}
        />
        <Stat label="to goal" value={`${sim.state.distance.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={2}>
        <ButtonRow>
          <ActionButton emphasis={params.controller === 'dwa'} onClick={() => setParams((q) => ({ ...q, controller: 'dwa' }))}>
            DWA
          </ActionButton>
          <ActionButton emphasis={params.controller === 'mppi'} onClick={() => setParams((q) => ({ ...q, controller: 'mppi' }))}>
            MPPI
          </ActionButton>
        </ButtonRow>
        <ButtonRow>
          <ActionButton emphasis={params.scene === 'corridor'} onClick={() => setParams((q) => ({ ...q, scene: 'corridor' }))}>
            Corridor slalom
          </ActionButton>
          <ActionButton emphasis={params.scene === 'pocket'} onClick={() => setParams((q) => ({ ...q, scene: 'pocket' }))}>
            Counter pocket
          </ActionButton>
        </ButtonRow>
        <Slider
          label="DWA clearance weight w_c"
          role="measurement"
          value={params.clearanceWeight}
          min={0.02}
          max={0.5}
          step={0.02}
          onChange={(x) => setParams((q) => ({ ...q, clearanceWeight: x }))}
          help="Fox et al. call it β. Heading is fixed at 0.8 and velocity at 0.15."
        />
        <Toggle
          label="Minimum-margin clearance (instead of distance-to-contact)"
          role="measurement"
          checked={params.marginMode}
          onChange={(x) => setParams((q) => ({ ...q, marginMode: x }))}
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

function star(ctx: CanvasRenderingContext2D, px: number, py: number, r: number, color: string) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(px, py, r * 0.45, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
