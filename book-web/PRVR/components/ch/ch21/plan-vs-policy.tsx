'use client';

import { useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { Slider, ControlPanel } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { Rng } from '@/lib/prob/rng';
import { blankGrid, cellIndex, gridWorldMdp, MOVES4 } from '@/lib/decision/gridworld';
import { greedyPolicy, sampleTransition, valueIteration, type Mdp } from '@/lib/decision/mdp';

/**
 * f21.4 — plan versus policy, on the same MDP, with the same seeds.
 *
 * The left panel executes a *plan*: the action sequence that is optimal if
 * nothing ever goes wrong, replayed open-loop. The right panel executes a
 * *policy*: the same solver's answer for every cell, consulted afresh at each
 * step. Both run the identical transition model and the identical random
 * numbers, so every difference between the two pictures is caused by one thing
 * — whether the robot is allowed to look at where it actually is.
 */

const W = 13;
const H = 9;
const N_RUNS = 14;
const MAX_STEPS = 90;

interface Scene {
  mdp: Mdp;
  policy: number[];
  blocked: boolean[];
  goal: number;
  start: number;
  /** The noise-free optimal action sequence: the "plan". */
  plan: number[];
  planPath: number[];
}

function buildScene(slip: number): Scene {
  const spec = blankGrid(W, H, { slip, gamma: 0.97, moves: 4, stepCost: 1, noCornerCutting: true });
  // A pillar with a gap on each side: two homotopy classes, one shorter.
  for (let j = 2; j <= 6; j++) spec.blocked[cellIndex(spec, 6, j)] = true;
  const goal = cellIndex(spec, W - 2, H - 2);
  const start = cellIndex(spec, 1, 1);
  spec.payoff[goal] = 60;
  spec.terminal[goal] = true;

  const mdp = gridWorldMdp(spec);
  const { v } = valueIteration(mdp, { eps: 1e-9 });
  const policy = greedyPolicy(mdp, v);

  // The plan: follow the policy through the *most likely* successor only. That
  // is exactly what a deterministic planner returns, and exactly what the
  // Chapter 20 path is.
  const plan: number[] = [];
  const planPath: number[] = [start];
  let s = start;
  for (let t = 0; t < MAX_STEPS && !mdp.absorbing[s]; t++) {
    const a = policy[s];
    plan.push(a);
    const row = mdp.trans[s][a];
    const likely = row.reduce((b, x) => (x.p > b.p ? x : b), row[0]);
    s = likely.s;
    planPath.push(s);
  }
  return { mdp, policy, blocked: spec.blocked, goal, start, plan, planPath };
}

interface RunSet {
  paths: number[][];
  arrived: number;
  meanSteps: number;
}

function executeOpenLoop(scene: Scene, seed: number): RunSet {
  const { mdp, plan, start, goal } = scene;
  const paths: number[][] = [];
  let arrived = 0;
  let steps = 0;
  for (let k = 0; k < N_RUNS; k++) {
    const rng = new Rng(seed + k);
    let s = start;
    const path = [s];
    for (const a of plan) {
      if (mdp.absorbing[s]) break;
      s = sampleTransition(mdp.trans[s][a], rng.next());
      path.push(s);
    }
    if (s === goal) arrived++;
    steps += path.length - 1;
    paths.push(path);
  }
  return { paths, arrived, meanSteps: steps / N_RUNS };
}

function executeClosedLoop(scene: Scene, seed: number): RunSet {
  const { mdp, policy, start, goal } = scene;
  const paths: number[][] = [];
  let arrived = 0;
  let steps = 0;
  for (let k = 0; k < N_RUNS; k++) {
    const rng = new Rng(seed + k);
    let s = start;
    const path = [s];
    for (let t = 0; t < MAX_STEPS && !mdp.absorbing[s]; t++) {
      s = sampleTransition(mdp.trans[s][policy[s]], rng.next());
      path.push(s);
    }
    if (s === goal) arrived++;
    steps += path.length - 1;
    paths.push(path);
  }
  return { paths, arrived, meanSteps: steps / N_RUNS };
}

const cx = (s: number) => (s % W) + 0.5;
const cy = (s: number) => H - Math.floor(s / W) - 0.5;
const toPoints = (path: number[]) => path.map((s) => `${cx(s)},${cy(s)}`).join(' ');

export function PlanVsPolicy() {
  const [slip, setSlip] = useState(0.15);
  const scene = useMemo(() => buildScene(slip), [slip]);
  const open = useMemo(() => executeOpenLoop(scene, 2100), [scene]);
  const closed = useMemo(() => executeClosedLoop(scene, 2100), [scene]);

  return (
    <WidgetFrame
      id="f21.4"
      title="Plan versus policy, same world, same dice"
      teaches="A plan is a line through the state space and has nothing to say about the states you actually reach. A policy is defined everywhere, so drifting off is not a failure mode."
      colorKey={['prediction', 'measurement', 'truth']}
      wide
      caption={
        <>
          Both panels use one MDP, one solver, and one set of seeds. On the left the robot commits to
          the optimal action <em>sequence</em> and replays it open-loop — the orange line is what it
          expected, the gray dashed lines are what happened. On the right the same solver&apos;s
          answer is consulted at every step. <strong>Push the slip slider up.</strong> The plan
          degrades gracelessly, because an action sequence indexed by <em>time</em> is a claim about
          a trajectory nobody is on any more; the policy, indexed by <em>state</em>, simply keeps
          answering. At <em>s</em> = 0 the two pictures are identical — which is why deterministic
          planning ever worked.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-2 md:divide-x md:divide-fd-border">
        <Panel
          title="a plan, executed open-loop"
          scene={scene}
          runs={open}
          showPlan
          showArrows={false}
        />
        <Panel
          title="a policy, consulted every step"
          scene={scene}
          runs={closed}
          showPlan={false}
          showArrows
        />
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 sm:grid-cols-4">
        <StatTile
          label="plan reaches dock"
          value={`${Math.round((open.arrived / N_RUNS) * 100)}%`}
          role="prediction"
        />
        <StatTile label="plan mean steps" value={open.meanSteps} precision={1} role="prediction" />
        <StatTile
          label="policy reaches dock"
          value={`${Math.round((closed.arrived / N_RUNS) * 100)}%`}
          role="posterior"
        />
        <StatTile label="policy mean steps" value={closed.meanSteps} precision={1} role="posterior" />
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="Slip s (per side)"
          role="prediction"
          value={slip}
          min={0}
          max={0.35}
          step={0.01}
          onChange={setSlip}
          help="Both panels share this number, the same MDP, and the same random seeds."
        />
      </ControlPanel>
    </WidgetFrame>
  );
}

function Panel({
  title,
  scene,
  runs,
  showPlan,
  showArrows,
}: {
  title: string;
  scene: Scene;
  runs: RunSet;
  showPlan: boolean;
  showArrows: boolean;
}) {
  return (
    <div className="p-3">
      <p className="eyebrow mb-2">{title}</p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={`${title}: a grid world with a pillar, a start in one corner and a dock in the other, showing ${N_RUNS} executed trajectories.`}
      >
        {/* cells */}
        {Array.from({ length: W * H }, (_, s) => {
          const blocked = scene.blocked[s];
          const isGoal = s === scene.goal;
          return (
            <rect
              key={s}
              x={s % W}
              y={H - Math.floor(s / W) - 1}
              width={1}
              height={1}
              fill={blocked ? 'var(--pr-wall)' : isGoal ? 'var(--pr-measurement)' : 'var(--pr-canvas-bg)'}
              fillOpacity={blocked ? 0.35 : isGoal ? 0.85 : 1}
              stroke="var(--pr-grid)"
              strokeWidth={0.02}
            />
          );
        })}

        {showArrows
          ? Array.from({ length: W * H }, (_, s) => {
              if (scene.blocked[s] || scene.mdp.absorbing[s]) return null;
              const d = MOVES4[scene.policy[s] % MOVES4.length];
              return (
                <line
                  key={`a${s}`}
                  x1={cx(s) - d[0] * 0.22}
                  y1={cy(s) + d[1] * 0.22}
                  x2={cx(s) + d[0] * 0.3}
                  y2={cy(s) - d[1] * 0.3}
                  stroke="var(--pr-canvas-ink)"
                  strokeOpacity={0.45}
                  strokeWidth={0.06}
                  markerEnd="url(#pr-ch21-arrow)"
                />
              );
            })
          : null}

        {showPlan ? (
          <polyline
            points={toPoints(scene.planPath)}
            fill="none"
            stroke="var(--pr-prediction)"
            strokeWidth={0.16}
            strokeLinejoin="round"
          />
        ) : null}

        {runs.paths.map((path, k) => (
          <polyline
            key={k}
            points={toPoints(path)}
            fill="none"
            stroke="var(--pr-truth)"
            strokeOpacity={0.5}
            strokeWidth={0.07}
            strokeDasharray="0.18 0.14"
            strokeLinejoin="round"
          />
        ))}

        {runs.paths.map((path, k) => {
          const end = path[path.length - 1];
          if (end === scene.goal) return null;
          return (
            <g key={`x${k}`} stroke="var(--pr-prediction)" strokeWidth={0.09}>
              <line x1={cx(end) - 0.2} y1={cy(end) - 0.2} x2={cx(end) + 0.2} y2={cy(end) + 0.2} />
              <line x1={cx(end) - 0.2} y1={cy(end) + 0.2} x2={cx(end) + 0.2} y2={cy(end) - 0.2} />
            </g>
          );
        })}

        <circle cx={cx(scene.start)} cy={cy(scene.start)} r={0.22} fill="var(--pr-truth)" />

        <defs>
          {showArrows ? (
          <marker
            id="pr-ch21-arrow"
            viewBox="0 0 10 10"
            refX="7"
            refY="5"
            markerWidth="4"
            markerHeight="4"
            orient="auto-start-reverse"
          >
            <path d="M 0 1 L 8 5 L 0 9 z" fill="var(--pr-canvas-ink)" fillOpacity={0.45} />
          </marker>
          ) : null}
        </defs>
      </svg>
    </div>
  );
}
