'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import {
  cliffRunMdp,
  criticalSlip,
  RISKY,
  SAFE,
  type CliffConfig,
} from '@/lib/decision/gridworld';
import { policyEvaluation, sampleTransition, valueIteration } from '@/lib/decision/mdp';
import { clear, label, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w21.3 — Cliff Run. Predict-then-verify.
 *
 * Two routes to the same goal: six cells along a ledge, or sixteen the long way
 * round. The reader commits a guess for the slip at which the optimal policy
 * flips before the widget will show the answer — the marker is computed from
 * the closed-form route values in `lib/decision/gridworld.ts`, and the policy
 * the robots actually walk comes from `valueIteration` on the same MDP, so the
 * two can be checked against each other on screen.
 *
 * Ten seeded runs animate continuously. Their mean realized return converges to
 * V*(start), slowly and noisily, which is the point: the expectation is a
 * promise about the average, and a single run is entitled to be a disaster.
 */

const BASE: CliffConfig = { riskyLen: 6, safeLen: 16, cliffPenalty: 6, slip: 0.05 };
const N_RUNS = 10;
const MAX_STEPS = 400;

interface Run {
  state: number;
  ret: number;
  steps: number;
  falls: number;
}

interface State {
  runs: Run[];
  rng: Rng;
  /** Realized returns of completed runs, newest last. */
  finished: number[];
  totalFalls: number;
}

export function CliffRun() {
  const [slip, setSlip] = useState(0.05);
  const [guess, setGuess] = useState(0.15);
  const [committed, setCommitted] = useState(false);

  const cfg = useMemo<CliffConfig>(() => ({ ...BASE, slip }), [slip]);
  const scene = useMemo(() => cliffRunMdp(cfg), [cfg]);
  const solved = useMemo(() => valueIteration(scene.mdp, { eps: 1e-10 }), [scene]);
  const sStar = useMemo(() => criticalSlip(BASE), []);

  /** The two route values across the whole slider range — real policy evaluation. */
  const curves = useMemo(() => {
    const xs: number[] = [];
    for (let s = 0.005; s <= 0.2500001; s += 0.005) xs.push(Number(s.toFixed(4)));
    const risky: { x: number; y: number }[] = [];
    const safe: { x: number; y: number }[] = [];
    for (const s of xs) {
      const { mdp, start } = cliffRunMdp({ ...BASE, slip: s });
      const all = (a: number) => new Array<number>(mdp.nStates).fill(a);
      risky.push({ x: s, y: policyEvaluation(mdp, all(RISKY), { tol: 1e-12 })[start] });
      safe.push({ x: s, y: policyEvaluation(mdp, all(SAFE), { tol: 1e-12 })[start] });
    }
    return { risky, safe };
  }, []);

  const init = useCallback(
    (seed: number): State => ({
      runs: Array.from({ length: N_RUNS }, () => ({ state: 0, ret: 0, steps: 0, falls: 0 })),
      rng: new Rng(seed),
      finished: [],
      totalFalls: 0,
    }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      const { mdp, start, goal } = scene;
      const policy = solved.policy;
      const runs = s.runs.map((r) => ({ ...r }));
      const finished = [...s.finished];
      let totalFalls = s.totalFalls;

      for (const r of runs) {
        if (r.state === goal || r.steps >= MAX_STEPS) {
          finished.push(r.ret);
          r.state = start;
          r.ret = 0;
          r.steps = 0;
          continue;
        }
        const a = policy[r.state];
        // γ = 1, so an SSP return is an undiscounted sum. The *realized* cost is
        // charged here, not the expected reward the MDP stores: one per step,
        // plus the cliff penalty on the runs that actually go over.
        r.ret -= 1;
        const onLedge = r.state === start ? policy[start] === RISKY : scene.route[r.state] === RISKY;
        const next = sampleTransition(mdp.trans[r.state][a], s.rng.next());
        if (onLedge && next === start) {
          r.ret -= BASE.cliffPenalty;
          r.falls += 1;
          totalFalls += 1;
        }
        r.state = next;
        r.steps += 1;
      }
      return { runs, rng: s.rng, finished: finished.slice(-200), totalFalls };
    },
    [scene, solved],
  );

  const sim = useSimulation<State>({ init, step, fps: 10, initialSeed: 21 });

  const span = Math.max(BASE.riskyLen, BASE.safeLen);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, view: Viewport, p: Palette) => {
      clear(ctx, view, p);
      const chosenRisky = solved.policy[scene.start] === RISKY;

      // Corridors. The ledge has nothing under it; the detour has walls.
      const corridor = (y: number, color: string, alpha: number, name: string) => {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.fillRect(sx(view, -0.15), sy(view, y + 0.34), sx(view, span + 0.15) - sx(view, -0.15), sy(view, 0) - sy(view, 0.68));
        ctx.restore();
        label(ctx, name, sx(view, -0.1), sy(view, y + 0.62), color, { size: 10, weight: 600 });
      };
      corridor(1, p.prediction, 0.14, `ledge · ${BASE.riskyLen} cells · a veer costs ${BASE.cliffPenalty} and a walk back`);
      corridor(-1, p.prior, 0.14, `detour · ${BASE.safeLen} cells · a veer costs one step`);

      // Cliff hatching above and below the ledge.
      ctx.save();
      ctx.strokeStyle = p.prediction;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = -0.1; x <= span + 0.1; x += 0.22) {
        for (const y of [1.42, 0.5]) {
          ctx.moveTo(sx(view, x), sy(view, y));
          ctx.lineTo(sx(view, x + 0.16), sy(view, y - 0.16));
        }
      }
      ctx.stroke();
      ctx.restore();

      // Connectors: the fork out of the start and back into the goal.
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1.5;
      for (const [y, color] of [
        [1, p.prediction],
        [-1, p.prior],
      ] as [number, string][]) {
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(sx(view, 0), sy(view, 0));
        ctx.lineTo(sx(view, 0.35), sy(view, y));
        ctx.moveTo(sx(view, span - 0.35), sy(view, y));
        ctx.lineTo(sx(view, span), sy(view, 0));
        ctx.stroke();
      }
      ctx.restore();

      // Start and goal.
      for (const [x, name, color] of [
        [0, 'start', p.truth],
        [span, 'goal', p.measurement],
      ] as [number, string, string][]) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(sx(view, x), sy(view, 0), 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        label(ctx, name, sx(view, x), sy(view, -0.42), color, { size: 10, align: 'center' });
      }

      // The policy's choice at the junction.
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx(view, 0), sy(view, 0));
      ctx.lineTo(sx(view, 0.55), sy(view, chosenRisky ? 0.85 : -0.85));
      ctx.stroke();
      label(
        ctx,
        chosenRisky ? 'π*(start) = ledge' : 'π*(start) = detour',
        sx(view, 0.75),
        sy(view, chosenRisky ? 1.05 : -1.05),
        p.posterior,
        { size: 11, weight: 700 },
      );

      // Ten robots, jittered along the corridor so they do not stack.
      sim.state.runs.forEach((r, k) => {
        const pos = scene.layout[r.state];
        const jitter = (k - (N_RUNS - 1) / 2) * 0.055;
        ctx.fillStyle = p.truth;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(sx(view, pos.x), sy(view, pos.y + jitter), 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      });
    },
    [scene, sim.state, solved, span],
  );

  const stats = useMemo(() => {
    const done = sim.state.finished;
    const mean = done.length ? done.reduce((a, b) => a + b, 0) / done.length : 0;
    const worst = done.length ? Math.min(...done) : 0;
    return {
      vStar: solved.v[scene.start],
      mean,
      worst,
      runs: done.length,
      falls: sim.state.totalFalls,
      spark: done.slice(-30),
      route: solved.policy[scene.start] === RISKY ? 'ledge' : 'detour',
    };
  }, [sim.state, solved, scene]);

  const chartSeries = useMemo(
    () => [
      { id: 'V of the ledge policy', role: 'prediction' as const, data: curves.risky },
      { id: 'V of the detour policy', role: 'prior' as const, data: curves.safe },
    ],
    [curves],
  );

  const markers = useMemo(() => {
    const m: { axis: 'x' | 'y'; value: number; label?: string; role?: 'posterior' | 'truth' }[] = [
      { axis: 'x', value: slip, label: 'you are here', role: 'truth' },
    ];
    if (committed) {
      m.push({ axis: 'x', value: sStar, label: `s* = ${sStar.toFixed(3)}`, role: 'posterior' });
      m.push({ axis: 'x', value: guess, label: 'your guess', role: 'truth' });
    }
    return m;
  }, [committed, guess, sStar, slip]);

  return (
    <WidgetFrame
      id="w21.3"
      title="Cliff Run"
      teaches="Optimal means best in expectation, not best case. The route flips at a critical noise level — and it is far lower than it feels."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The ledge is {BASE.riskyLen} cells; the detour is {BASE.safeLen}. On the ledge a veer puts
          Rusty over the edge: he pays {BASE.cliffPenalty} and is dragged back to the start. On the
          detour a veer just costs him a step against a wall.{' '}
          <strong>Before you touch anything</strong>, guess the slip <em>s*</em> at which the optimal
          policy abandons the ledge, set it with the guess slider, and commit — the answer stays
          hidden until you do. Then sweep the slip slider and watch the crossing. Notice the second
          lesson in the tiles: long after the policy has switched, individual runs on the losing
          route still sometimes beat the winning one. Expectation is a claim about the average, and
          nothing else.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.4, minY: -1.5, maxX: span + 0.4, maxY: 1.5 }}
        draw={draw}
        deps={[sim.tick, sim.state, solved]}
        aspect={3.1}
        padding={0.1}
        ariaLabel="Two corridors from a start to a goal: a short ledge flanked by a drop, and a long detour flanked by walls. Ten robots walk whichever route the optimal policy currently prefers."
      />

      <div className="px-3 py-3">
        <Dashboard columns={4}>
          <DashboardPanel title="route value vs slip" span={2}>
            <LineChart
              series={chartSeries}
              xLabel="slip s (per side)"
              yLabel="V^π(start)"
              height={210}
              curve="monotoneX"
              markers={markers}
            />
          </DashboardPanel>
          <StatTile label="π*(start)" value={stats.route} role="posterior" />
          <StatTile label="V*(start)" value={stats.vStar} precision={2} role="posterior" />
          <StatTile
            label="mean realized return"
            value={stats.mean}
            precision={2}
            role="truth"
            trend={stats.mean - stats.vStar}
            trendLabel={`over ${stats.runs} finished runs`}
            sparkline={stats.spark}
          />
          <StatTile label="worst single run" value={stats.worst} precision={1} role="prediction" />
          <StatTile label="falls" value={stats.falls} precision={0} role="prediction" />
        </Dashboard>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Slip s (per side)"
          role="prediction"
          value={slip}
          min={0.005}
          max={0.25}
          step={0.005}
          onChange={setSlip}
          help="The one parameter that matters here: the chance of veering to each side per step."
        />
        <Slider
          label="Your prediction for s*"
          role="posterior"
          value={guess}
          min={0.005}
          max={0.25}
          step={0.005}
          onChange={setGuess}
          help="Commit before revealing the analytic answer."
        />
      </ControlPanel>

      <div className="flex flex-wrap items-center gap-2 border-t border-fd-border px-3 py-2 font-ui text-xs">
        <ButtonRow>
          <ActionButton onClick={() => setCommitted(true)} emphasis={!committed}>
            {committed ? 'prediction committed' : 'commit my prediction'}
          </ActionButton>
          {committed ? <ActionButton onClick={() => setCommitted(false)}>hide again</ActionButton> : null}
        </ButtonRow>
        {committed ? (
          <span className="text-fd-muted-foreground">
            analytic s* = <strong className="font-mono">{sStar.toFixed(3)}</strong>; you said{' '}
            <span className="font-mono">{guess.toFixed(3)}</span> — off by{' '}
            <span className="font-mono">{Math.abs(guess - sStar).toFixed(3)}</span>.
          </span>
        ) : (
          <span className="text-fd-muted-foreground">
            The marker is hidden until you commit. Guessing badly here is the point.
          </span>
        )}
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
