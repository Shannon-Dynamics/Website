'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT } from '@/lib/sim/world';
import {
  apartmentGrid,
  cellAt,
  cellCenter,
  cellIndex,
  gridWorldMdp,
  MOVES8,
  type ApartmentGrid,
} from '@/lib/decision/gridworld';
import {
  greedyPolicy,
  sampleTransition,
  sweepInPlace,
  sweepJacobi,
  type Mdp,
} from '@/lib/decision/mdp';
import { clear, drawSegments, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w21.1 — Policy Painter.
 *
 * The chapter's thesis made draggable: a policy is not a stored path, it is an
 * answer for every state, and the answer bends as the world gets slipperier.
 * The reader paints payoff onto the Apartment; value iteration re-converges
 * live — a visible wave, because each frame runs a fixed number of sweeps of
 * the real `sweepInPlace` from `lib/decision/mdp.ts` — and the arrow field is
 * `greedyPolicy` on whatever the value function currently is.
 *
 * Nothing here is precomputed. Rusty walks the current policy with the current
 * slip, so a half-converged value function produces a visibly confused robot.
 */

const CELL = 0.3;
const INFLATION = 0.16;
const GOAL_PAYOFF = 100;
const HAZARD_PAYOFF = -60;
const STEP_COST = 1;
const SWEEPS_PER_TICK = 6;
const TRAIL = 90;

type Brush = 'goal' | 'hazard' | 'erase';

interface State {
  v: number[];
  policy: number[];
  residual: number;
  sweeps: number;
  rng: Rng;
  robot: number;
  trail: { x: number; y: number }[];
  arrivals: number;
  falls: number;
}

/** Defaults: charging dock in the bedroom, a spill in the corridor doorway. */
function defaultPainting(spec: ApartmentGrid) {
  const dock = cellAt(spec, 7.0, 7.5);
  const spill = cellAt(spec, 6.0, 4.4);
  spec.payoff[cellIndex(spec, dock[0], dock[1])] = GOAL_PAYOFF;
  spec.terminal[cellIndex(spec, dock[0], dock[1])] = true;
  spec.payoff[cellIndex(spec, spill[0], spill[1])] = HAZARD_PAYOFF;
  spec.terminal[cellIndex(spec, spill[0], spill[1])] = true;
}

function firstFreeCell(spec: ApartmentGrid, x: number, y: number): number {
  const [i, j] = cellAt(spec, x, y);
  const s = cellIndex(spec, i, j);
  if (!spec.blocked[s] && !spec.terminal[s]) return s;
  for (let k = 0; k < spec.blocked.length; k++) {
    if (!spec.blocked[k] && !spec.terminal[k]) return k;
  }
  return 0;
}

export function PolicyPainter() {
  const [params, setParams] = useState({ slip: 0.2, gamma: 0.98, synchronous: false });
  const [brush, setBrush] = useState<Brush>('hazard');
  const [paintVersion, setPaintVersion] = useState(0);

  // The grid spec is mutable state the reader edits with the mouse; the MDP is
  // recompiled from it whenever the painting or the physics changes.
  const specRef = useRef<ApartmentGrid | null>(null);
  if (specRef.current === null) {
    const spec = apartmentGrid(CELL, INFLATION, {
      slip: 0.2,
      gamma: 0.98,
      stepCost: STEP_COST,
      moves: 8,
      noCornerCutting: true,
    });
    defaultPainting(spec);
    specRef.current = spec;
  }
  const spec = specRef.current;

  const mdpRef = useRef<Mdp | null>(null);
  const compile = useCallback(() => {
    const s = specRef.current;
    if (!s) return;
    s.slip = params.slip;
    s.gamma = params.gamma;
    mdpRef.current = gridWorldMdp(s);
  }, [params.slip, params.gamma]);
  // Recompile on the render that follows any parameter or painting change.
  const signature = `${params.slip}|${params.gamma}|${paintVersion}`;
  const lastSignature = useRef('');
  if (lastSignature.current !== signature) {
    lastSignature.current = signature;
    compile();
  }

  const startCell = useMemo(() => firstFreeCell(spec, 1.2, 1.2), [spec]);

  const init = useCallback(
    (seed: number): State => {
      const n = spec.width * spec.height;
      return {
        v: new Array<number>(n).fill(0),
        policy: new Array<number>(n).fill(0),
        residual: Infinity,
        sweeps: 0,
        rng: new Rng(seed),
        robot: startCell,
        trail: [],
        arrivals: 0,
        falls: 0,
      };
    },
    [spec, startCell],
  );

  const step = useCallback(
    (s: State): State => {
      const mdp = mdpRef.current;
      if (!mdp) return s;

      // --- planning: a fixed budget of backups per frame, so convergence is
      // something the reader watches happen rather than a fait accompli.
      let v = s.v;
      let residual = 0;
      for (let k = 0; k < SWEEPS_PER_TICK; k++) {
        if (params.synchronous) {
          const out = sweepJacobi(mdp, v);
          v = out.v;
          residual = Math.max(residual, out.residual);
        } else {
          residual = Math.max(residual, sweepInPlace(mdp, v));
        }
      }
      const policy = greedyPolicy(mdp, v);

      // --- acting: one step of the slippery robot under the current policy.
      let robot = s.robot;
      let arrivals = s.arrivals;
      let falls = s.falls;
      if (mdp.absorbing[robot]) {
        if (spec.payoff[robot] > 0) arrivals++;
        else if (spec.payoff[robot] < 0) falls++;
        robot = startCell;
      } else {
        robot = sampleTransition(mdp.trans[robot][policy[robot]], s.rng.next());
      }
      const [ri, rj] = [robot % spec.width, Math.floor(robot / spec.width)];
      const trail = [...s.trail, cellCenter(spec, ri, rj)].slice(-TRAIL);

      return { ...s, v, policy, residual, sweeps: s.sweeps + SWEEPS_PER_TICK, robot, trail, arrivals, falls };
    },
    [params.synchronous, spec, startCell],
  );

  const sim = useSimulation<State>({ init, step, fps: 12, initialSeed: 21 });

  /* ---------------------------------------------------------------- paint */

  const paintingRef = useRef(false);
  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') {
        paintingRef.current = false;
        return;
      }
      if (phase === 'down') paintingRef.current = true;
      if (!paintingRef.current) return;
      const [i, j] = cellAt(spec, world[0], world[1]);
      const idx = cellIndex(spec, i, j);
      if (spec.blocked[idx]) return;
      const payoff = brush === 'goal' ? GOAL_PAYOFF : brush === 'hazard' ? HAZARD_PAYOFF : 0;
      if (spec.payoff[idx] === payoff) return;
      spec.payoff[idx] = payoff;
      spec.terminal[idx] = payoff !== 0;
      setPaintVersion((n) => n + 1);
    },
    [brush, spec],
  );

  const clearPainting = useCallback(() => {
    for (let k = 0; k < spec.payoff.length; k++) {
      spec.payoff[k] = 0;
      spec.terminal[k] = false;
    }
    setPaintVersion((n) => n + 1);
  }, [spec]);

  const resetPainting = useCallback(() => {
    for (let k = 0; k < spec.payoff.length; k++) {
      spec.payoff[k] = 0;
      spec.terminal[k] = false;
    }
    defaultPainting(spec);
    setPaintVersion((n) => n + 1);
  }, [spec]);

  /* ----------------------------------------------------------------- draw */

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, view: Viewport, p: Palette) => {
      clear(ctx, view, p);
      const { v, policy, robot, trail } = sim.state;
      const w = sl(view, CELL) + 0.6;

      // Value scale, computed over reachable cells only.
      let vMax = 1e-6;
      let vMin = -1e-6;
      for (let k = 0; k < v.length; k++) {
        if (spec.blocked[k] || spec.terminal[k]) continue;
        if (v[k] > vMax) vMax = v[k];
        if (v[k] < vMin) vMin = v[k];
      }

      for (let j = 0; j < spec.height; j++) {
        for (let i = 0; i < spec.width; i++) {
          const idx = cellIndex(spec, i, j);
          const c = cellCenter(spec, i, j);
          const px = sx(view, c.x - CELL / 2);
          const py = sy(view, c.y + CELL / 2);

          if (spec.blocked[idx]) {
            ctx.globalAlpha = 0.18;
            ctx.fillStyle = p.wall;
            ctx.fillRect(px, py, w, w);
            ctx.globalAlpha = 1;
            continue;
          }
          if (spec.terminal[idx]) {
            ctx.fillStyle = spec.payoff[idx] > 0 ? p.measurement : p.prediction;
            ctx.globalAlpha = 0.85;
            ctx.fillRect(px, py, w, w);
            ctx.globalAlpha = 1;
            continue;
          }

          // Value heat: purple where the future is worth having, orange where
          // it is not. Both ramps are the book's own hues, never a new one.
          const val = v[idx];
          if (val >= 0) {
            ctx.fillStyle = p.posterior;
            ctx.globalAlpha = 0.07 + 0.72 * (val / vMax);
          } else {
            ctx.fillStyle = p.prediction;
            ctx.globalAlpha = 0.07 + 0.72 * Math.min(1, val / (vMin - 1e-9));
          }
          ctx.fillRect(px, py, w, w);
          ctx.globalAlpha = 1;
        }
      }

      // Arrow field — the policy. Ink, not a data hue: it is the answer, not
      // a distribution.
      ctx.strokeStyle = p.ink;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const arm = CELL * 0.36;
      for (let j = 0; j < spec.height; j++) {
        for (let i = 0; i < spec.width; i++) {
          const idx = cellIndex(spec, i, j);
          if (spec.blocked[idx] || spec.terminal[idx]) continue;
          const d = MOVES8[policy[idx] % MOVES8.length];
          const norm = Math.hypot(d[0], d[1]) || 1;
          const c = cellCenter(spec, i, j);
          const hx = c.x + (d[0] / norm) * arm;
          const hy = c.y + (d[1] / norm) * arm;
          ctx.moveTo(sx(view, c.x - (d[0] / norm) * arm * 0.6), sy(view, c.y - (d[1] / norm) * arm * 0.6));
          ctx.lineTo(sx(view, hx), sy(view, hy));
          // Arrowhead: two short barbs, drawn in world space so they scale.
          const a = Math.atan2(d[1], d[0]);
          for (const off of [2.5, -2.5]) {
            ctx.moveTo(sx(view, hx), sy(view, hy));
            ctx.lineTo(
              sx(view, hx + arm * 0.45 * Math.cos(a + off)),
              sy(view, hy + arm * 0.45 * Math.sin(a + off)),
            );
          }
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // The true walls, over the discretization, so the reader sees what the
      // inflation threw away.
      drawSegments(ctx, view, APARTMENT.walls, p.wall, 1.6);

      // Rusty, and where he has actually been.
      if (trail.length > 1) {
        ctx.save();
        ctx.strokeStyle = p.truth;
        ctx.setLineDash([4, 3]);
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(sx(view, trail[0].x), sy(view, trail[0].y));
        for (const q of trail) ctx.lineTo(sx(view, q.x), sy(view, q.y));
        ctx.stroke();
        ctx.restore();
      }
      const rc = cellCenter(spec, robot % spec.width, Math.floor(robot / spec.width));
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(sx(view, rc.x), sy(view, rc.y), 4.5, 0, Math.PI * 2);
      ctx.fill();

      label(
        ctx,
        sim.state.residual < 1e-6 ? 'converged' : `‖ΔV‖∞ = ${sim.state.residual.toFixed(3)}`,
        8,
        14,
        sim.state.residual < 1e-6 ? p.posterior : p.prediction,
        { size: 11, weight: 600 },
      );
    },
    [sim.state, spec],
  );

  const stats = useMemo(() => {
    const { v } = sim.state;
    let best = -Infinity;
    for (let k = 0; k < v.length; k++) if (!spec.blocked[k] && v[k] > best) best = v[k];
    return {
      vStart: sim.state.v[startCell],
      best,
      arrivals: sim.state.arrivals,
      falls: sim.state.falls,
    };
  }, [sim.state, spec, startCell]);

  return (
    <WidgetFrame
      id="w21.1"
      title="Policy Painter"
      teaches="A policy is not a stored path. It is an answer for every cell — and the answer bends away from danger as the floor gets slipperier."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Purple is the value function, the expected discounted payoff of being in a cell and acting
          optimally forever after; orange cells are hazards and orange shading is negative value.
          The arrows are the greedy policy read off that value function. <strong>Drag on the map</strong>{' '}
          to paint — the current brush is shown below — and watch the value wave re-propagate from
          the change. <strong>Then move the slip slider.</strong> At <em>s</em> = 0 the arrows hug
          the hazard; by <em>s</em> = 0.3 they have opened a wide berth around it, because the
          expectation now includes the futures where the wheels betray you. Rusty walks the current
          policy: while the value function is still converging, watch him take steps he will
          regret.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state, paintVersion, brush]}
        aspect={12 / 9}
        padding={0.15}
        onPointer={onPointer}
        cursor="crosshair"
        ariaLabel="A floorplan of the apartment shaded by its value function, with an arrow in every free cell showing the optimal move. A robot walks the arrows from a room to the charging dock while avoiding a hazard cell in the corridor."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="V*(start)" value={stats.vStart.toFixed(2)} />
        <Stat label="best cell" value={stats.best.toFixed(2)} />
        <Stat label="dock arrivals" value={String(stats.arrivals)} />
        <Stat label="hazard hits" value={String(stats.falls)} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Slip s (per side)"
          role="prediction"
          value={params.slip}
          min={0}
          max={0.4}
          step={0.01}
          onChange={(v) => setParams((p) => ({ ...p, slip: v }))}
          help="Probability the robot veers into each of the two neighbouring directions."
        />
        <Slider
          label="Discount γ"
          role="posterior"
          value={params.gamma}
          min={0.7}
          max={0.995}
          step={0.005}
          onChange={(v) => setParams((p) => ({ ...p, gamma: v }))}
          help="How far into the future the robot can see. Below about 0.9 the dock becomes invisible from the far rooms."
        />
        <Toggle
          label="Synchronous sweeps (Jacobi)"
          role="prior"
          checked={params.synchronous}
          onChange={(v) => setParams((p) => ({ ...p, synchronous: v }))}
        />
      </ControlPanel>

      <div className="flex flex-wrap items-center gap-2 border-t border-fd-border px-3 py-2">
        <span className="eyebrow">brush</span>
        <ButtonRow>
          {(['goal', 'hazard', 'erase'] as Brush[]).map((b) => (
            <ActionButton key={b} onClick={() => setBrush(b)} emphasis={brush === b}>
              {b}
            </ActionButton>
          ))}
          <ActionButton onClick={resetPainting}>default scene</ActionButton>
          <ActionButton onClick={clearPainting}>clear payoff</ActionButton>
        </ButtonRow>
      </div>

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

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
