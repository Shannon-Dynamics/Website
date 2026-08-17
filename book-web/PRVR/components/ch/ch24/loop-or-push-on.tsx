'use client';

import { useCallback, useMemo, useState } from 'react';

import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { between, boxplus, compose, type Pose2, type Twist2 } from '@/lib/geom/se2';
import { OccupancyGrid } from '@/lib/mapping/occgrid';
import { PoseGraph, informationFromSigmas } from '@/lib/slam/posegraph';
import { dOptimality, graphLogDet } from '@/lib/explore/utility';
import { EXPLORE_INVERSE_MODEL } from '@/lib/explore/explorer';
import { RUSTY_LIDAR, raycastScan } from '@/lib/sim/rusty';
import { APARTMENT } from '@/lib/sim/world';
import {
  clear,
  drawOccupancyGrid,
  drawPath,
  drawRobot,
  drawSegments,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w24.3 — Loop or Push On?
 *
 * Twelve metres of battery, two ways to spend them. Push east into room E and
 * the map grows; drive back west to the start and one loop-closure factor lands
 * on the graph instead. Both futures are simulated in full: the same LiDAR, the
 * same odometry noise, the same Chapter 16 pose graph, the same `graphLogDet`
 * from `lib/explore/utility.ts`.
 *
 * The map you see is built by integrating the *true* scans at the *estimated*
 * poses, which is what a real system does — so the push-on map is visibly
 * sheared, and the loop-closed one is not.
 */

const CELL = 0.16;
const STEP = 0.4;
const N_NEW = 30;
const LIDAR = { ...RUSTY_LIDAR, nBeams: 36, maxRange: 5, sigmaR: 0.02, pDropout: 0.004 };
/** Odometry is the weakest constraint in the book, and this is why. */
const ODOM_SIGMA: [number, number, number] = [0.035, 0.02, 0.03];
/** A verified scan match is worth roughly an order of magnitude more. */
const LOOP_SIGMA: [number, number, number] = [0.05, 0.05, 0.02];

const BASE_ROUTE: [number, number][] = [
  [1.6, 2.2],
  [2.0, 4.4],
  [9.6, 4.4],
];

/** Push on: back to the north doorway at x ≈ 7.85, into room E, and around it. */
const PUSH_ROUTE: [number, number][] = [
  [9.6, 4.4],
  [7.85, 4.4],
  [7.85, 6.3],
  [9.2, 6.3],
  [9.2, 8.4],
  [6.6, 8.4],
  [6.6, 5.6],
];

/** Close the loop: back west down the corridor and into room A, where node 0 sits. */
const LOOP_ROUTE: [number, number][] = [
  [9.6, 4.4],
  [2.0, 4.4],
  [1.6, 2.2],
  [3.4, 2.2],
  [3.4, 0.8],
];

type Choice = 'push' | 'loop';

/**
 * Walk a polyline at a fixed arc length, heading along the segment.
 *
 * Both candidates are sampled to the *same* node count on purpose: with equal
 * numbers of odometry edges, the only thing that can separate their information
 * matrices is the loop-closure factor, which is exactly the comparison this
 * widget is for.
 */
function samplePath(route: [number, number][], step: number, count: number): Pose2[] {
  const cum: number[] = [0];
  for (let k = 1; k < route.length; k++) {
    cum.push(cum[k - 1] + Math.hypot(route[k][0] - route[k - 1][0], route[k][1] - route[k - 1][1]));
  }
  const total = cum[cum.length - 1];
  const out: Pose2[] = [];
  let seg = 0;
  for (let n = 0; n < count; n++) {
    const s = Math.min(n * step, total);
    while (seg + 2 < route.length && cum[seg + 1] <= s) seg++;
    const [x0, y0] = route[seg];
    const [x1, y1] = route[seg + 1];
    const len = Math.max(cum[seg + 1] - cum[seg], 1e-9);
    const t = Math.min(1, (s - cum[seg]) / len);
    out.push({
      x: x0 + t * (x1 - x0),
      y: y0 + t * (y1 - y0),
      theta: Math.atan2(y1 - y0, x1 - x0),
    });
  }
  return out;
}

function pathLength(poses: Pose2[]): number {
  let d = 0;
  for (let k = 1; k < poses.length; k++) d += Math.hypot(poses[k].x - poses[k - 1].x, poses[k].y - poses[k - 1].y);
  return d;
}

/** Integrate true scans at the supplied poses — pass estimates in to get shear. */
function buildGrid(sensePoses: Pose2[], truePoses: Pose2[], seed: number): OccupancyGrid {
  const grid = OccupancyGrid.forWorld(APARTMENT, CELL, 0.5);
  const rng = new Rng(seed);
  for (let k = 0; k < truePoses.length; k += 2) {
    const scan = raycastScan(APARTMENT, truePoses[k], LIDAR, rng);
    grid.integrateScan(sensePoses[k], scan.ranges, scan.angles, EXPLORE_INVERSE_MODEL);
  }
  return grid;
}

function ate(estimate: Pose2[], truth: Pose2[]): number {
  let s = 0;
  for (let k = 0; k < truth.length; k++) {
    s += (estimate[k].x - truth[k].x) ** 2 + (estimate[k].y - truth[k].y) ** 2;
  }
  return Math.sqrt(s / Math.max(truth.length, 1));
}

interface Branch {
  truth: Pose2[];
  estimate: Pose2[];
  grid: OccupancyGrid;
  logDet: number;
  dOpt: number;
  ate: number;
  mapGain: number;
  cost: number;
  loops: number;
  /** Index of the node the loop-closure factor attaches to, or −1. */
  loopNode: number;
}

interface Scenario {
  base: { truth: Pose2[]; estimate: Pose2[]; grid: OccupancyGrid; logDet: number; dOpt: number; entropy: number };
  push: Branch;
  loop: Branch;
}

/**
 * Build the whole staged dilemma for one seed: the explored prefix, then both
 * futures, each with its own graph, its own map, and its own errors.
 */
function buildScenario(seed: number): Scenario {
  const odomOmega = informationFromSigmas(...ODOM_SIGMA);
  const loopOmega = informationFromSigmas(...LOOP_SIGMA);

  const baseTruth = samplePath(BASE_ROUTE, STEP, 26);
  const rng = new Rng(seed);

  const graph = new PoseGraph();
  graph.addNode(baseTruth[0], true);
  for (let k = 1; k < baseTruth.length; k++) {
    const rel = between(baseTruth[k - 1], baseTruth[k]);
    const noise: Twist2 = [
      rng.normal(0, ODOM_SIGMA[0]),
      rng.normal(0, ODOM_SIGMA[1]),
      rng.normal(0, ODOM_SIGMA[2]),
    ];
    const z = boxplus(rel, noise);
    // Dead reckoning: the pose estimate is the composition of the measurements,
    // and it is wrong by the accumulated noise. That is the whole problem.
    graph.addNode(compose(graph.nodes[k - 1].pose, z));
    graph.addEdge(k - 1, k, z, odomOmega, 'odometry');
  }

  const baseEstimate = graph.poses();
  const baseGrid = buildGrid(baseEstimate, baseTruth, seed + 7);
  const baseTruthGrid = buildGrid(baseTruth, baseTruth, seed + 7);
  const base = {
    truth: baseTruth,
    estimate: baseEstimate,
    grid: baseGrid,
    logDet: graphLogDet(graph),
    dOpt: dOptimality(graph),
    entropy: baseTruthGrid.entropy(),
  };

  const branch = (route: [number, number][], closeLoop: boolean, salt: number): Branch => {
    const g = new PoseGraph();
    for (const node of graph.nodes) g.addNode(node.pose, node.fixed);
    for (const e of graph.edges) g.addEdge(e.i, e.j, e.z, e.omega, e.kind);

    const newTruth = samplePath(route, STEP, N_NEW);
    const truth = [...baseTruth, ...newTruth];
    const r = new Rng(seed + salt);
    let last = graph.nodes.length - 1;
    for (let k = 0; k < newTruth.length; k++) {
      const prevTruth = k === 0 ? baseTruth[baseTruth.length - 1] : newTruth[k - 1];
      const rel = between(prevTruth, newTruth[k]);
      const noise: Twist2 = [
        r.normal(0, ODOM_SIGMA[0]),
        r.normal(0, ODOM_SIGMA[1]),
        r.normal(0, ODOM_SIGMA[2]),
      ];
      const z = boxplus(rel, noise);
      const id = g.addNode(compose(g.nodes[last].pose, z));
      g.addEdge(last, id, z, odomOmega, 'odometry');
      last = id;
    }

    let loops = 0;
    let loopNode = -1;
    if (closeLoop) {
      // The node that actually comes back within a scan-match's reach of node 0.
      let best = -1;
      let bestD = Infinity;
      for (let k = baseTruth.length; k < truth.length; k++) {
        const d = Math.hypot(truth[k].x - truth[0].x, truth[k].y - truth[0].y);
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      }
      if (best >= 0 && bestD < 1.2) {
        const rel = between(truth[0], truth[best]);
        const noise: Twist2 = [
          r.normal(0, LOOP_SIGMA[0]),
          r.normal(0, LOOP_SIGMA[1]),
          r.normal(0, LOOP_SIGMA[2]),
        ];
        g.addEdge(0, best, boxplus(rel, noise), loopOmega, 'loop');
        loops = 1;
        loopNode = best;
      }
    }

    g.optimize(20, 1e-8, 0);
    const estimate = g.poses();
    const grid = buildGrid(estimate, truth, seed + 7);
    const truthGrid = buildGrid(truth, truth, seed + 7);

    return {
      truth,
      estimate,
      grid,
      logDet: graphLogDet(g),
      dOpt: dOptimality(g),
      ate: ate(estimate, truth),
      mapGain: base.entropy - truthGrid.entropy(),
      cost: pathLength(newTruth),
      loops,
      loopNode,
    };
  };

  return { base, push: branch(PUSH_ROUTE, false, 101), loop: branch(LOOP_ROUTE, true, 101) };
}

interface State {
  choice: Choice;
  reveal: number;
}

export function LoopOrPushOn() {
  const [wG, setWG] = useState(0.1);
  const [pinned, setPinned] = useState<Choice | null>(null);
  const [seed, setSeed] = useState(24);

  const scenario = useMemo(() => buildScenario(seed), [seed]);

  const init = useCallback((): State => ({ choice: 'push', reveal: 0 }), []);
  const step = useCallback((s: State): State => {
    const reveal = s.reveal + 0.045;
    if (reveal < 1.35) return { ...s, reveal };
    return { choice: s.choice === 'push' ? 'loop' : 'push', reveal: 0 };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 18, initialSeed: 1 });
  const choice: Choice = pinned ?? sim.state.choice;
  const shown = choice === 'push' ? scenario.push : scenario.loop;
  const reveal = pinned ? 1 : Math.min(1, sim.state.reveal);

  /**
   * Both candidates cost the same twelve metres, so C(a) cancels and the whole
   * decision is the exchange rate between bits of map and the graph term.
   *
   * The graph term is Δ log D-opt, *not* Δ log det Ω. Raw log det counts nodes:
   * both branches add thirty odometry factors, so both raise it by about the
   * same 650 nats and the one loop closure disappears into the noise. The 1/n
   * of D-optimality removes exactly that, and what is left is the theorem —
   * an odometry-only branch moves D-opt by zero.
   *
   * Both terms are scaled by the better of the two options, so w_G is a pure
   * preference: 0 is "coverage only", 1 is "certainty only".
   */
  const utilities = useMemo(() => {
    const iMax = Math.max(scenario.push.mapGain, scenario.loop.mapGain, 1e-9);
    const dPush = Math.log(scenario.push.dOpt) - Math.log(scenario.base.dOpt);
    const dLoop = Math.log(scenario.loop.dOpt) - Math.log(scenario.base.dOpt);
    const dMax = Math.max(dPush, dLoop, 1e-9);
    const score = (mapGain: number, delta: number) => {
      const map = (1 - wG) * (mapGain / iMax);
      const graph = wG * Math.max(0, delta / dMax);
      return { map, graph, total: map + graph };
    };
    return {
      push: score(scenario.push.mapGain, dPush),
      loop: score(scenario.loop.mapGain, dLoop),
    };
  }, [scenario, wG]);

  const winner: Choice = utilities.push.total >= utilities.loop.total ? 'push' : 'loop';

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawOccupancyGrid(ctx, v, shown.grid, p);

      ctx.save();
      ctx.globalAlpha = 0.3;
      ctx.setLineDash([4, 4]);
      drawSegments(ctx, v, APARTMENT.walls, p.truth, 1.2);
      ctx.restore();

      // Ground truth, then the estimate the graph settled on.
      drawPath(ctx, v, shown.truth, p.truth, { dashed: true, lineWidth: 1.4, alpha: 0.8 });

      const nBase = scenario.base.truth.length;
      const nShown = nBase + Math.round(reveal * (shown.estimate.length - nBase));
      drawPath(ctx, v, shown.estimate.slice(0, nBase), p.prior, { lineWidth: 2, alpha: 0.9 });
      drawPath(ctx, v, shown.estimate.slice(nBase - 1, nShown), p.posterior, {
        lineWidth: 2.2,
        alpha: 1,
      });

      if (shown.loops > 0 && nShown > shown.loopNode) {
        const a = shown.estimate[0];
        const b = shown.estimate[shown.loopNode];
        ctx.save();
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 2;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, a.x), sy(v, a.y));
        ctx.lineTo(sx(v, b.x), sy(v, b.y));
        ctx.stroke();
        ctx.restore();
        label(ctx, 'loop closure', sx(v, (a.x + b.x) / 2) + 6, sy(v, (a.y + b.y) / 2), p.measurement, {
          size: 10,
        });
      }

      drawRobot(ctx, v, shown.estimate[Math.max(nShown - 1, 0)], p.posterior, 0.26);
      drawRobot(ctx, v, shown.truth[Math.max(nShown - 1, 0)], p.truth, 0.26, { filled: false });

      label(
        ctx,
        choice === 'push' ? 'PREVIEW — push on (coverage)' : 'PREVIEW — close the loop (certainty)',
        10,
        14,
        p.posterior,
        { size: 11, weight: 600 },
      );
      label(
        ctx,
        winner === choice ? 'this is the argmax at the current w_G' : '',
        10,
        30,
        p.measurement,
        { size: 10 },
      );
    },
    [shown, scenario, reveal, choice, winner],
  );

  return (
    <WidgetFrame
      id="w24.3"
      title="Loop or Push On?"
      teaches="A good exploration policy does not maximize coverage. Without deliberate loop closing, the map you cover is a map you cannot trust."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The same twelve metres and the same thirty odometry factors, spent two ways.{' '}
          <em>Push on</em> drives into room E and buys about 740 bits of new map; <em>close the
          loop</em> drives back to where node 0 sits, still learns 620 bits through the doorways it
          passes, and buys one scan-match factor. Watch the two graph readouts disagree: Δ log det Ω
          is nearly identical for the two branches, because it mostly counts nodes — but D-opt,
          which divides that by the number of degrees of freedom, does not move <em>at all</em> for
          the odometry-only branch. That is not an artifact; it is the proposition below. The map
          drawn is built from the true scans at the <em>estimated</em> poses, which is why one
          apartment comes out sheared and the other comes out square. Slide{' '}
          <span className="font-mono">w_G</span> past about 0.14 and the argmax flips — then look at
          the trajectory error, which does not care what you weighted.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: 12, maxY: 9 }}
        draw={draw}
        deps={[sim.tick, choice, reveal, wG, seed, winner]}
        aspect={1.55}
        padding={0.3}
        ariaLabel="Two previewed futures for an exploring robot in an apartment: pushing on into an unmapped room, which leaves the estimated trajectory and map visibly sheared, or returning to the start to close a loop, which straightens both."
      />

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 sm:grid-cols-4">
        <StatTile label="map gain over 12 m" value={shown.mapGain} unit="bits" precision={0} role="measurement" />
        <StatTile
          label="Δ log det Ω"
          value={shown.logDet - scenario.base.logDet}
          unit="nats"
          precision={0}
          role="prior"
        />
        <StatTile
          label="D-opt(Ω)"
          value={shown.dOpt}
          precision={0}
          role="posterior"
          trend={shown.dOpt - scenario.base.dOpt}
          trendLabel="vs. now"
        />
        <StatTile label="trajectory error (RMS)" value={shown.ate} unit="m" precision={3} role="truth" />
      </div>

      <div className="grid gap-2 border-t border-fd-border px-3 py-2.5 sm:grid-cols-2">
        <UtilityCard
          title="Push on"
          map={utilities.push.map}
          graph={utilities.push.graph}
          total={utilities.push.total}
          winner={winner === 'push'}
          active={choice === 'push'}
        />
        <UtilityCard
          title="Close the loop"
          map={utilities.loop.map}
          graph={utilities.loop.graph}
          total={utilities.loop.total}
          winner={winner === 'loop'}
          active={choice === 'loop'}
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Graph weight w_G  (certainty vs. coverage)"
          role="posterior"
          value={wG}
          min={0}
          max={1}
          step={0.02}
          onChange={setWG}
          help="w_G = 0 is a pure coverage explorer — the 2005 objective. Raise it and closing the loop starts to outrank new territory."
        />
        <div className="flex flex-col gap-1.5">
          <span className="font-ui text-[0.72rem] font-medium">Preview</span>
          <ButtonRow>
            <ActionButton onClick={() => setPinned('push')} emphasis={pinned === 'push'}>
              Push on
            </ActionButton>
            <ActionButton onClick={() => setPinned('loop')} emphasis={pinned === 'loop'}>
              Close the loop
            </ActionButton>
            <ActionButton onClick={() => setPinned(null)} emphasis={pinned === null}>
              Auto-cycle
            </ActionButton>
          </ButtonRow>
          <span className="font-ui text-[0.68rem] text-fd-muted-foreground">
            {shown.loops > 0
              ? 'One loop-closure factor lands on the graph; Gauss–Newton redistributes 12 m of drift.'
              : 'Odometry only. Nothing in the graph can undo the accumulated heading error.'}
          </span>
        </div>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onReset={sim.reset}
        onReseed={() => setSeed((s) => (s * 1103515245 + 12345) % 100000)}
        seed={seed}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

function UtilityCard({
  title,
  map,
  graph,
  total,
  winner,
  active,
}: {
  title: string;
  map: number;
  graph: number;
  total: number;
  winner: boolean;
  active: boolean;
}) {
  return (
    <div
      className="rounded-sm border p-2.5"
      style={{
        borderColor: active ? 'var(--pr-posterior)' : 'var(--color-fd-border)',
      }}
    >
      <div className="flex items-baseline justify-between">
        <span className="font-ui text-xs font-semibold">{title}</span>
        {winner ? (
          <span className="font-mono text-[0.65rem]" style={{ color: 'var(--pr-posterior)' }}>
            argmax
          </span>
        ) : null}
      </div>
      <dl className="mt-1.5 grid grid-cols-3 gap-1 font-mono text-[0.68rem] tabular-nums text-fd-muted-foreground">
        <div>
          <dt className="eyebrow">(1−w_G)·Î</dt>
          <dd style={{ color: 'var(--pr-measurement)' }}>{map.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="eyebrow">w_G·Δ̂</dt>
          <dd style={{ color: 'var(--pr-posterior)' }}>{graph.toFixed(2)}</dd>
        </div>
        <div>
          <dt className="eyebrow">U(a)</dt>
          <dd className="text-fd-foreground">{total.toFixed(2)}</dd>
        </div>
      </dl>
    </div>
  );
}
