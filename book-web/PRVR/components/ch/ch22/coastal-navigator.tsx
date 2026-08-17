'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import {
  type CoastalRun,
  type Tally,
  cellCenter,
  colOf,
  emptyTally,
  record,
  rowOf,
  sigmaEntropy,
  solveCoastal,
  startRun,
  stepRun,
} from '@/lib/pomdp/amdp';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { HALL_PARAMS, SIGMA_0, makeHall } from './coastal-scene';

/**
 * w22.2 — the Coastal Navigator.
 *
 * Two pilots race across the same hall under the same seeded noise. The gray
 * one plans as if it always knew where it was — Chapter 21's value iteration on
 * the MAP estimate, which is what "plan a path, then track it" amounts to. The
 * purple one plans in the augmented belief space (pose, σ), so the cost of not
 * knowing where you are is inside the Bellman equation rather than bolted on
 * afterwards.
 *
 * The ribbon behind each robot is drawn with width proportional to σ, so the
 * belief's entropy is a visible physical quantity: the gray ribbon fattens all
 * the way across the room, the purple one pinches whenever it comes back within
 * scanning distance of the textured south wall.
 */

const WORLD = makeHall();
const MAX_STEPS = 90;

interface State {
  rng: Rng;
  coastal: CoastalRun;
  shortest: CoastalRun;
  tallyCoastal: Tally;
  tallyShortest: Tally;
}

export function CoastalNavigator() {
  const [range, setRange] = useState(2.5);

  /**
   * Two solves of the same augmented MDP: one honest, one told to pretend σ is
   * always at its floor. 7,321 states (610 free cells × 12 σ bins, plus an
   * absorbing terminal) and about 40 Gauss–Seidel sweeps each.
   */
  const { coastalPlan, shortestPlan } = useMemo(() => {
    const params = { ...HALL_PARAMS, range };
    return {
      coastalPlan: solveCoastal(WORLD, params),
      shortestPlan: solveCoastal(WORLD, params, { certaintyEquivalent: true }),
    };
  }, [range]);

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      coastal: startRun(WORLD, SIGMA_0),
      shortest: startRun(WORLD, SIGMA_0),
      tallyCoastal: emptyTally(),
      tallyShortest: emptyTally(),
    }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      // Both pilots are driven from the same generator, so a lucky seed is
      // lucky for both of them.
      let { coastal, shortest, tallyCoastal, tallyShortest } = s;
      coastal = stepRun(coastalPlan, coastal, s.rng, MAX_STEPS);
      shortest = stepRun(shortestPlan, shortest, s.rng, MAX_STEPS);
      if (coastal.outcome !== 'running' && shortest.outcome !== 'running') {
        tallyCoastal = record(tallyCoastal, coastal);
        tallyShortest = record(tallyShortest, shortest);
        return {
          rng: s.rng,
          coastal: startRun(WORLD, SIGMA_0),
          shortest: startRun(WORLD, SIGMA_0),
          tallyCoastal,
          tallyShortest,
        };
      }
      return { ...s, coastal, shortest };
    },
    [coastalPlan, shortestPlan],
  );

  const sim = useSimulation<State>({ init, step, fps: 14, initialSeed: 22 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const w = WORLD;
      const c = w.cell;

      // ---- the information field, as a wash --------------------------------
      // Where a scan is worth taking. Everything the coastal policy does is a
      // consequence of this picture, so it belongs underneath everything else.
      const nuMax = Math.max(...coastalPlan.nu, 1e-9);
      for (let i = 0; i < w.occ.length; i++) {
        if (w.occ[i]) continue;
        const a = coastalPlan.nu[i] / nuMax;
        if (a < 0.02) continue;
        ctx.globalAlpha = 0.3 * a;
        ctx.fillStyle = p.measurement;
        ctx.fillRect(sx(v, colOf(w, i) * c), sy(v, (rowOf(w, i) + 1) * c), sl(v, c) + 1, sl(v, c) + 1);
      }
      ctx.globalAlpha = 1;

      // ---- walls ----------------------------------------------------------
      for (let i = 0; i < w.occ.length; i++) {
        if (!w.occ[i]) continue;
        ctx.fillStyle = w.feature[i] ? p.accent : p.wall;
        ctx.fillRect(sx(v, colOf(w, i) * c), sy(v, (rowOf(w, i) + 1) * c), sl(v, c) + 1, sl(v, c) + 1);
      }

      const [gx, gy] = cellCenter(w, w.goal);
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx(v, gx - c), sy(v, gy + c), sl(v, 2 * c), sl(v, 2 * c));
      label(ctx, 'doorway', sx(v, gx) - 6, sy(v, gy + 2 * c), p.posterior, { size: 10, align: 'right' });
      label(ctx, 'textured wall — the coast', sx(v, 0.4), sy(v, 0.35), p.accent, { size: 10 });

      // ---- the two ribbons -------------------------------------------------
      const ribbon = (run: CoastalRun, color: string) => {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.22;
        ctx.beginPath();
        for (let k = 0; k < run.trail.length; k++) {
          const [x, y] = cellCenter(w, run.trail[k].cell);
          ctx.lineTo(sx(v, x), sy(v, y + run.trail[k].sigma));
        }
        for (let k = run.trail.length - 1; k >= 0; k--) {
          const [x, y] = cellCenter(w, run.trail[k].cell);
          ctx.lineTo(sx(v, x), sy(v, y - run.trail[k].sigma));
        }
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let k = 0; k < run.trail.length; k++) {
          const [x, y] = cellCenter(w, run.trail[k].cell);
          if (k === 0) ctx.moveTo(sx(v, x), sy(v, y));
          else ctx.lineTo(sx(v, x), sy(v, y));
        }
        ctx.stroke();

        const last = run.trail[run.trail.length - 1];
        const [rx, ry] = cellCenter(w, last.cell);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx(v, rx), sy(v, ry), 4.5, 0, Math.PI * 2);
        ctx.fill();
        if (run.outcome !== 'running') {
          label(ctx, run.outcome, sx(v, rx) + 8, sy(v, ry) - 9, color, { size: 10, weight: 600 });
        }
      };

      // Truth-gray for the certainty-equivalent pilot: it is the plan that
      // believes its own MAP estimate.
      ribbon(sim.state.shortest, p.truth);
      ribbon(sim.state.coastal, p.posterior);

      label(ctx, 'shortest path (plans on the MAP pose)', sx(v, 0.4), sy(v, 9.9), p.truth, { size: 10 });
      label(ctx, 'AMDP coastal policy (plans on the belief)', sx(v, 0.4), sy(v, 9.3), p.posterior, {
        size: 10,
        weight: 600,
      });
    },
    [coastalPlan, sim.state],
  );

  const arrive = (t: Tally) => (t.runs > 0 ? (100 * t.arrived) / t.runs : 0);
  const meanSteps = (t: Tally) => (t.runs > 0 ? t.totalSteps / t.runs : 0);
  const tc = sim.state.tallyCoastal;
  const ts = sim.state.tallyShortest;

  return (
    <WidgetFrame
      id="w22.2"
      title="The Coastal Navigator"
      teaches="The optimal route under uncertainty is not the shortest collision-free one. Wall-hugging is not a heuristic — it is what the Bellman equation returns when σ is part of the state."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          Green wash is the information field <code>ν(x)</code>: how much a scan taken here would
          tell you, given the LiDAR range. The gray pilot plans as if it always knew its pose and
          drives straight; its ribbon — width proportional to the position σ — fattens the whole way
          across, and it arrives at the far wall too uncertain either to thread a 1.2 m doorway or
          to keep off the plaster beside it.{' '}
          <strong>What to notice:</strong> the purple pilot dives toward the textured south wall
          mid-crossing, pinches its ribbon back down, and only then turns for the door. Over 400
          seeded runs at the default 2.5 m range it gets through 65% of the time against the straight
          pilot&rsquo;s 27%, and it pays about 1.6 extra steps for the privilege. Nobody coded the
          detour.{' '}
          <strong>Try this:</strong> push the LiDAR range past 5 m. The green wash floods the room,
          both pilots reach 96%, and the detour stops paying — coastal navigation is an artifact of{' '}
          <em>when</em> sensing is informative, not a law of robotics. That collapse is Figure 16.6
          of the Thrun draft, reproduced by the slider.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: WORLD.cols * WORLD.cell, minY: 0, maxY: WORLD.rows * WORLD.cell }}
        draw={draw}
        deps={[sim.tick, sim.state, coastalPlan]}
        aspect={WORLD.cols / WORLD.rows}
        padding={0.2}
        ariaLabel="A rectangular hall with a textured south wall and a doorway in the east wall. Two robot trajectories cross it: a straight gray one whose uncertainty ribbon widens continuously, and a purple one that detours toward the textured wall and stays narrow."
      />

      <div className="border-t border-fd-border p-3">
        <Dashboard columns={4}>
          <StatTile
            label="coastal — through the door"
            value={`${arrive(tc).toFixed(1)}%`}
            role="posterior"
            trend={arrive(tc) - arrive(ts)}
            trendLabel="vs shortest path"
          />
          <StatTile label="shortest — through the door" value={`${arrive(ts).toFixed(1)}%`} role="truth" />
          <StatTile
            label="mean steps (coastal / shortest)"
            value={`${meanSteps(tc).toFixed(1)} / ${meanSteps(ts).toFixed(1)}`}
          />
          <StatTile
            label="belief entropy at the door"
            value={
              tc.runs > 0
                ? `${sigmaEntropy(tc.totalFinalSigma / tc.runs).toFixed(2)} / ${sigmaEntropy(ts.totalFinalSigma / ts.runs).toFixed(2)} bits`
                : '—'
            }
            role="measurement"
          />
        </Dashboard>
        <p className="mt-2 font-mono text-[0.7rem] text-fd-muted-foreground tabular-nums">
          runs {tc.runs} · coastal: {tc.arrived} arrived, {tc.clipped} clipped a wall, {tc.missed}{' '}
          missed the gap · shortest: {ts.arrived} arrived, {ts.clipped} clipped, {ts.missed} missed
        </p>
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="LiDAR range"
          role="measurement"
          value={range}
          min={2}
          max={8}
          step={0.5}
          unit="m"
          onChange={setRange}
          help="Sets how far from the textured wall a scan still constrains the pose: ν(x) = ν_max · exp(−(d/R)²). Re-solves both 7,321-state MDPs."
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
