'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { BarChart, Dashboard, DashboardPanel, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { Pomcp, layoutTree, type GenerativeModel, type LaidOutNode } from '@/lib/pomdp/pomcp';

/**
 * w22.4 — POMCP Tree Peek.
 *
 * The planner never touches a transition matrix. It is handed a black box that
 * answers "if the world were in state x and I did u, what would happen?" — the
 * Chapter 4 simulator — and a bag of particles standing in for the belief, and
 * it builds a search tree over *histories* by sampling through them.
 *
 * The picture is the tree: circles are belief nodes (area ∝ visit count,
 * annotated with the particle belief that arrived there), bars are action arms
 * labelled with Q(h, u). Scrub the simulation count and watch the tree stop
 * being a bush and start being a plan.
 */

const ACTIONS = ['listen', 'open-left', 'open-right'] as const;
const OBS = ['hear-left', 'hear-right'] as const;
const DONE = -1;
const N_PARTICLES = 500;
const ACCURACY = 0.85;
const LADDER = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

/**
 * One round of the tiger, as a generative model. Opening a door ends the
 * episode — that is the only difference from the infinite-horizon version the
 * α-vector solver takes, and it is what makes a *tree* the right object.
 */
function tigerModel(accuracy: number, gamma = 0.95): GenerativeModel<number> {
  return {
    nActions: 3,
    gamma,
    step(x, u, rng) {
      if (x === DONE) return { x, z: 0, r: 0 };
      if (u === 0) return { x, z: rng.next() < accuracy ? x : 1 - x, r: -1 };
      const opened = u === 1 ? 0 : 1;
      return { x: DONE, z: 0, r: opened === x ? -100 : 10 };
    },
    terminal: (x) => x === DONE,
    // The default policy is what a leaf is worth before the tree knows better,
    // and it dominates the early anytime curve: "listen twice, then guess"
    // beats uniform-random by 23 points of root value at 16 simulations, by 2
    // at 256, and by nothing at all past 1024.
    rollout: (_x, depth, rng) => (depth < 2 ? 0 : rng.next() < 0.5 ? 1 : 2),
  };
}

interface State {
  tree: Pomcp<number>;
  /** Root Q after each rung of the simulation ladder — the anytime curve. */
  history: { sims: number; q: number[] }[];
  rung: number;
}

export function PomcpTreePeek() {
  const [pLeft, setPLeft] = useState(0.5);
  const [ucb, setUcb] = useState(80);

  const init = useCallback(
    (seed: number): State => {
      const rng = new Rng(seed);
      // The root belief is a particle set, exactly as Chapter 8 leaves it.
      const particles = Array.from({ length: N_PARTICLES }, () => (rng.next() < pLeft ? 0 : 1));
      const tree = new Pomcp(tigerModel(ACCURACY), particles, rng, { c: ucb, maxDepth: 8 });
      return { tree, history: [], rung: 0 };
    },
    [pLeft, ucb],
  );

  const step = useCallback((s: State): State => {
    if (s.rung >= LADDER.length) return s;
    s.tree.search(LADDER[s.rung] - s.tree.simulations);
    return {
      tree: s.tree,
      rung: s.rung + 1,
      history: [...s.history, { sims: s.tree.simulations, q: s.tree.rootQ() }],
    };
  }, []);

  const sim = useSimulation<State>({
    init,
    step,
    fps: 1.6,
    maxTicks: LADDER.length,
    loop: true,
    initialSeed: 22,
  });

  const tree = sim.state.tree;
  const sims = tree.simulations;

  // Arms pulled fewer than 1% of the time are noise, not structure; dropping
  // them is what turns a 4,000-simulation tree into a legible figure.
  const layout = useMemo(
    () => layoutTree(tree, 2, Math.max(2, Math.round(sims / 60))),
    [tree, sims],
  );

  const visits = tree.rootVisits();
  const q = tree.rootQ();
  const best = tree.bestAction();

  const bars = useMemo(
    () => [
      {
        id: 'visits N(h,u)',
        role: 'prior' as const,
        data: ACTIONS.map((a, u) => ({ x: a, y: visits[u] })),
      },
    ],
    [visits],
  );

  return (
    <WidgetFrame
      id="w22.4"
      title="POMCP Tree Peek"
      teaches="Online planning is not exhaustive lookahead. It is sampled, bandit-guided lookahead over particle beliefs — and it needs no transition matrix at all."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Circles are belief nodes; their area is the visit count <code>N(h)</code> and the number
          inside is the particle estimate of <code>b(tiger-left)</code> at that history. Bars are
          action arms labelled with <code>Q(h,u)</code>. <strong>What to notice:</strong> at eight
          simulations the tree is a bush with no opinion; by four thousand, 99% of them run down the{' '}
          <em>listen</em> arm and the two belief nodes below it read <code>0.86</code> and{' '}
          <code>0.15</code> — no one wrote a Bayes filter here, the particles that survived each
          branch <em>are</em> the posterior.{' '}
          <strong>Try this:</strong> drag the root belief to 0.97. The tree stops searching and
          starts confirming: <em>open-right</em> takes 82% of the budget. Then set the belief back to
          0.85 and drop the exploration constant to 20. Both opening arms get exactly one pull, the
          right-hand one happens to draw the tiger, and neither is ever tried again — the planner
          listens forever on the strength of a single unlucky rollout.
        </>
      }
    >
      <div className="p-3">
        <svg
          viewBox="0 0 100 46"
          className="w-full"
          role="img"
          aria-label="A search tree: a root belief node above three action arms, each leading to belief nodes labelled by the observation that produced them. Node size grows with visit count."
        >
          <TreeFigure nodes={layout.nodes} width={layout.width} />
        </svg>
      </div>

      <div className="border-t border-fd-border p-3">
        <Dashboard columns={4}>
          <StatTile label="simulations" value={sims} role="prior" />
          <StatTile label="recommended action" value={ACTIONS[best]} role="posterior" />
          <StatTile
            label="Q(root, listen)"
            value={Number.isNaN(q[0]) ? '—' : q[0].toFixed(2)}
            role="measurement"
            sparkline={sim.state.history.map((h) => (Number.isNaN(h.q[0]) ? 0 : h.q[0]))}
          />
          <StatTile label="belief nodes in the tree" value={tree.beliefs.length} role="truth" />
          <DashboardPanel title="Where the budget went" span="full">
            <BarChart
              series={bars}
              xLabel="action"
              yLabel="simulations through this arm"
              height={170}
              caption="UCB1 is a bandit rule, not a schedule: the arm that looks best gets almost everything, and the others get just enough to keep the estimate honest."
            />
          </DashboardPanel>
        </Dashboard>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Root belief b(tiger-left)"
          role="prior"
          value={pLeft}
          min={0.5}
          max={0.99}
          step={0.01}
          onChange={setPLeft}
          help="The particle set the root is built from — 500 unweighted samples, as Chapter 8 leaves them."
        />
        <Slider
          label="UCB1 exploration c"
          role="truth"
          value={ucb}
          min={10}
          max={160}
          step={5}
          format={(x) => x.toFixed(0)}
          onChange={setUcb}
          help="Scale it to the spread of the returns (here 110), not to 1. Too small and a single unlucky rollout retires an arm forever."
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

/** The tree, laid out by `layoutTree` and drawn in SVG user units. */
function TreeFigure({ nodes, width }: { nodes: LaidOutNode[]; width: number }) {
  const xOf = (x: number) => 6 + ((x + 0.5) / Math.max(width, 1)) * 88;
  const yOf = (row: number) => 5 + row * 9.5;
  const maxN = Math.max(...nodes.map((n) => n.n), 1);
  const roleColor = ['var(--pr-measurement)', 'var(--pr-prediction)', 'var(--pr-posterior)'];

  return (
    <>
      {nodes.map((n, i) =>
        n.parent === null ? null : (
          <line
            key={`e${i}`}
            x1={xOf(nodes[n.parent].x)}
            y1={yOf(nodes[n.parent].row)}
            x2={xOf(n.x)}
            y2={yOf(n.row)}
            stroke={n.kind === 'action' ? roleColor[n.action ?? 0] : 'var(--pr-grid)'}
            strokeWidth={0.25 + 1.5 * Math.sqrt(n.n / maxN)}
          />
        ),
      )}
      {nodes.map((n, i) => {
        const cx = xOf(n.x);
        const cy = yOf(n.row);
        if (n.kind === 'action') {
          return (
            <g key={`n${i}`}>
              <rect
                x={cx - 5}
                y={cy - 1.9}
                width={10}
                height={3.8}
                rx={0.6}
                fill={roleColor[n.action ?? 0]}
                opacity={0.85}
              />
              <text
                x={cx}
                y={cy + 0.9}
                textAnchor="middle"
                style={{ fontSize: 2.2, fontWeight: 600 }}
                fill="var(--pr-canvas-bg)"
              >
                {ACTIONS[n.action ?? 0]} {Number.isNaN(n.q) ? '' : n.q.toFixed(1)}
              </text>
            </g>
          );
        }
        const r = 1.6 + 2.6 * Math.sqrt(n.n / maxN);
        return (
          <g key={`n${i}`}>
            <circle cx={cx} cy={cy} r={r} fill="var(--pr-prior)" opacity={0.25} />
            <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--pr-prior)" strokeWidth={0.3} />
            <text
              x={cx}
              y={cy + 0.8}
              textAnchor="middle"
              style={{ fontSize: 2.1, fontWeight: 600 }}
              fill="var(--pr-canvas-ink)"
            >
              {Number.isNaN(n.particleMean) ? '' : (1 - n.particleMean).toFixed(2)}
            </text>
            {n.obs !== null ? (
              <text
                x={cx}
                y={cy - r - 1}
                textAnchor="middle"
                style={{ fontSize: 1.9 }}
                fill="var(--pr-measurement)"
              >
                {OBS[n.obs]}
              </text>
            ) : null}
          </g>
        );
      })}
    </>
  );
}
