'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, drawSegments, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT, type Point2 } from '@/lib/sim/world';
import { CSpace2, cellCenter, freeCellNear, latticeFromCSpace } from '@/lib/plan/cspace';
import { GridSearch, pathPoints, polylineLength } from '@/lib/plan/search';
import { Prm, Rrt, type PlanResult } from '@/lib/plan/sampling';

/**
 * w20.1 — the Planner Arena.
 *
 * One query in the Apartment, four planners, the same wall clock. The
 * scoreboard is the argument: A* on the lattice returns the optimum and pays
 * for it in memory; RRT answers first and answers badly; PRM builds a structure
 * that is wasted on a single query but free on the next thousand; RRT* starts
 * where RRT does and walks its cost down toward the lattice optimum.
 *
 * The misconception it kills is "RRT finds good paths". RRT finds *feasible*
 * paths quickly. Optimality is a separate contract, and Karaman & Frazzoli's
 * theorem is the price list.
 */

const GAP = 0.6;
const W = 12;
const H = 9;
const START: Point2 = { x: 1.0, y: 1.0 };
const GOAL: Point2 = { x: 11.0, y: 8.0 };
const ROBOT_RADIUS = 0.25;
const LATTICE_CELL = 0.2;

const LANES = [
  { key: 'astar', name: 'A* on the lattice', ox: 0, oy: H + GAP },
  { key: 'prm', name: 'PRM (multi-query)', ox: W + GAP, oy: H + GAP },
  { key: 'rrt', name: 'RRT', ox: 0, oy: 0 },
  { key: 'rrtstar', name: 'RRT*', ox: W + GAP, oy: 0 },
] as const;

const ARENA = { minX: 0, minY: 0, maxX: 2 * W + GAP, maxY: 2 * H + GAP };

interface Score {
  cost: number;
  nodes: number;
  /** Samples (or expansions) spent before the first solution appeared. */
  firstAt: number | null;
}

interface State {
  astar: GridSearch;
  astarScore: Score;
  astarPath: Point2[];
  prm: Prm;
  prmRng: Rng;
  prmSol: PlanResult | null;
  prmScore: Score;
  rrt: Rrt;
  rrtRng: Rng;
  rrtScore: Score;
  star: Rrt;
  starRng: Rng;
  starScore: Score;
  samples: number;
  expansions: number;
  history: { n: number; prm: number; rrt: number; star: number }[];
}

const emptyScore = (): Score => ({ cost: Infinity, nodes: 0, firstAt: null });

/**
 * γ_RRT* must exceed 2(1 + 1/d)^{1/d} (μ(Q_free)/ζ_d)^{1/d}. For the Apartment
 * with a 0.25 m disc, μ(Q_free) ≈ 78.2 m², d = 2, ζ₂ = π, so the bound is
 * 12.22. Running below it — which most implementations quietly do — forfeits
 * the theorem, so this widget runs above it.
 */
const GAMMA_RRT_STAR = 12.3;

export function PlannerArena() {
  const [budget, setBudget] = useState(2500);
  const [showInflated, setShowInflated] = useState(false);
  const [keepRefining, setKeepRefining] = useState(false);

  const cs = useMemo(() => new CSpace2(APARTMENT, { radius: ROBOT_RADIUS, cellSize: 0.05 }), []);
  const lattice = useMemo(() => latticeFromCSpace(cs, LATTICE_CELL), [cs]);
  const startCell = useMemo(() => freeCellNear(lattice, START.x, START.y), [lattice]);
  const goalCell = useMemo(() => freeCellNear(lattice, GOAL.x, GOAL.y), [lattice]);

  /** Cells the disc robot cannot occupy — the C-obstacle, drawn on demand. */
  const inflated = useMemo(() => {
    const out: Point2[] = [];
    for (let y = 0.05; y < H; y += 0.1) {
      for (let x = 0.05; x < W; x += 0.1) {
        if (!cs.isFree(x, y)) out.push({ x, y });
      }
    }
    return out;
  }, [cs]);

  const init = useCallback(
    (seed: number): State => ({
      astar: new GridSearch(lattice, startCell, goalCell),
      astarScore: emptyScore(),
      astarPath: [],
      prm: new Prm(cs, { k: 8, maxEdgeLength: 2.5 }),
      prmRng: new Rng(seed),
      prmSol: null,
      prmScore: emptyScore(),
      rrt: new Rrt(cs, START, GOAL, { star: false, stepSize: 0.7, goalBias: 0.05 }),
      rrtRng: new Rng(seed + 1),
      rrtScore: emptyScore(),
      star: new Rrt(cs, START, GOAL, {
        star: true,
        stepSize: 0.7,
        goalBias: 0.05,
        gamma: GAMMA_RRT_STAR,
      }),
      starRng: new Rng(seed + 1),
      starScore: emptyScore(),
      samples: 0,
      expansions: 0,
      history: [],
    }),
    [cs, lattice, startCell, goalCell],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      const perTick = Math.max(6, Math.ceil(budget / 50));
      const expandPerTick = Math.max(10, Math.ceil(budget / 40));
      let changed = false;

      // ---- lane 1: A* on the occupancy lattice -----------------------------
      if (s.astar.status === 'running' && s.expansions < 40000) {
        const status = s.astar.step(expandPerTick);
        s.expansions = s.astar.expanded;
        changed = true;
        if (status === 'found') {
          const pts = pathPoints(lattice, s.astar.path());
          s.astarPath = pts;
          s.astarScore = {
            cost: polylineLength(pts),
            nodes: s.astar.expanded,
            firstAt: s.astar.expanded,
          };
        }
      }

      // ---- lanes 2–4: the samplers ----------------------------------------
      if (s.samples < budget) {
        for (let i = 0; i < perTick; i++) {
          s.prm.step(s.prmRng);
          s.rrt.step(s.rrtRng);
          s.star.step(s.starRng);
        }
        s.samples += perTick;
        changed = true;

        // PRM's query is a full graph search, so it is not run every frame.
        if (tick % 4 === 0 || s.samples >= budget) {
          const sol = s.prm.query(START, GOAL);
          if (sol) {
            s.prmSol = sol;
            s.prmScore = {
              cost: sol.cost,
              nodes: s.prm.nodes.length,
              firstAt: s.prmScore.firstAt ?? s.samples,
            };
          }
        }
        s.rrtScore = {
          cost: s.rrt.bestCost,
          nodes: s.rrt.nodes.length,
          firstAt: s.rrtScore.firstAt ?? (Number.isFinite(s.rrt.bestCost) ? s.samples : null),
        };
        s.starScore = {
          cost: s.star.bestCost,
          nodes: s.star.nodes.length,
          firstAt: s.starScore.firstAt ?? (Number.isFinite(s.star.bestCost) ? s.samples : null),
        };
      } else if (keepRefining) {
        // Only RRT* has anything left to do: it is the one planner here whose
        // answer improves with more samples rather than merely existing.
        for (let i = 0; i < perTick; i++) s.star.step(s.starRng);
        s.samples += perTick;
        s.starScore = { cost: s.star.bestCost, nodes: s.star.nodes.length, firstAt: s.starScore.firstAt };
        changed = true;
      }

      if (!changed) return s;
      const history = [
        ...s.history,
        {
          n: s.samples,
          prm: s.prmScore.cost,
          rrt: s.rrtScore.cost,
          star: s.starScore.cost,
        },
      ].slice(-400);
      return { ...s, history };
    },
    [budget, keepRefining, lattice],
  );

  const sim = useSimulation<State>({ init, step, fps: 18, initialSeed: 20 });

  const resetRef = useRef(sim.reset);
  resetRef.current = sim.reset;
  useEffect(() => {
    resetRef.current();
  }, [budget]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;

      for (const lane of LANES) {
        const { ox, oy } = lane;
        const X = (x: number) => sx(v, x + ox);
        const Y = (y: number) => sy(v, y + oy);

        // Panel frame + title.
        ctx.save();
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.strokeRect(X(0), Y(H), sl(v, W), sl(v, H));
        ctx.restore();

        if (showInflated) {
          ctx.save();
          ctx.fillStyle = p.truth;
          ctx.globalAlpha = 0.18;
          const w = Math.ceil(sl(v, 0.1)) + 1;
          for (const c of inflated) ctx.fillRect(X(c.x - 0.05), Y(c.y + 0.05), w, w);
          ctx.restore();
        }

        if (lane.key === 'astar') drawAstar(ctx, v, ox, oy, s, p, lattice);
        if (lane.key === 'prm') drawPrm(ctx, v, ox, oy, s, p);
        if (lane.key === 'rrt') drawTree(ctx, v, ox, oy, s.rrt, s.rrt.solution(), p);
        if (lane.key === 'rrtstar') drawTree(ctx, v, ox, oy, s.star, s.star.solution(), p);

        // The map goes on top: it is ground truth, and nothing may hide it.
        ctx.save();
        ctx.translate(sx(v, ox) - sx(v, 0), sy(v, oy) - sy(v, 0));
        drawSegments(ctx, v, APARTMENT.walls, p.wall, 2);
        ctx.restore();

        ctx.save();
        ctx.strokeStyle = p.truth;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(X(START.x), Y(START.y), 5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = p.measurement;
        ctx.beginPath();
        ctx.arc(X(GOAL.x), Y(GOAL.y), 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        const score = scoreFor(s, lane.key);
        label(ctx, lane.name, X(0.15), Y(H) - 9, p.ink, { size: 11, weight: 700 });
        label(
          ctx,
          Number.isFinite(score.cost) ? `${score.cost.toFixed(2)} m` : 'searching…',
          X(W - 0.15),
          Y(H) - 9,
          Number.isFinite(score.cost) ? p.posterior : p.truth,
          { size: 11, weight: 600, align: 'right' },
        );
      }
    },
    [inflated, lattice, showInflated, sim.state],
  );

  const s = sim.state;
  const optimal = s.astarScore.cost;
  const chartSeries = useMemo(() => {
    const pick = (key: 'prm' | 'rrt' | 'star') =>
      s.history.filter((h) => Number.isFinite(h[key])).map((h) => ({ x: h.n, y: h[key] }));
    const out = [
      { id: 'PRM', role: 'prior' as const, data: pick('prm') },
      { id: 'RRT', role: 'prediction' as const, data: pick('rrt') },
      { id: 'RRT*', role: 'posterior' as const, data: pick('star') },
    ];
    return out.filter((series) => series.data.length > 1);
  }, [s.history]);

  return (
    <WidgetFrame
      id="w20.1"
      title="The Planner Arena"
      teaches="RRT does not find good paths — it finds feasible ones fast. Optimality is a separate contract, and RRT* is its price."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The same query — gray ring to green disc, across the whole Apartment — given to four
          planners with the same sample budget. Blue is what each planner has <em>looked at</em>:
          A*’s expanded cells, PRM’s roadmap. Orange is a tree that has committed to its edges;
          purple is the path each planner would hand you right now. Watch three things. A* returns
          the lattice optimum and stores thousands of cells to do it. RRT answers first and its
          answer is visibly bad — jagged, and often through the wrong doorway. RRT* starts identical
          to RRT and then bends toward the optimum as its rewiring ball keeps shrinking; the chart
          is that cost coming down. Push the budget to its maximum and turn on{' '}
          <em>keep refining</em> to watch RRT*’s cost keep falling — slowly, which is the honest
          picture of asymptotic optimality. And note what it converges <em>past</em>: the ratio
          column drops below 1, because A* is optimal on the 0.2 m lattice and the lattice is not
          the plane. Its eight headings cost the octile detour that a sampler, free to place a node
          anywhere, never pays.
        </>
      }
    >
      <SimCanvas
        world={ARENA}
        draw={draw}
        deps={[sim.tick, sim.state, showInflated]}
        aspect={ARENA.maxX / ARENA.maxY}
        padding={0.25}
        ariaLabel="Four copies of the apartment floorplan, each showing a different planner solving the same start-to-goal query: A* expanding lattice cells, a probabilistic roadmap, an RRT tree, and an RRT* tree."
      />

      <div className="overflow-x-auto border-t border-fd-border">
        <table className="w-full min-w-[30rem] border-collapse text-left font-ui text-[0.78rem]">
          <thead>
            <tr className="border-b border-fd-border">
              <th className="px-3 py-1.5 font-medium">planner</th>
              <th className="px-3 py-1.5 text-right font-medium">path cost</th>
              <th className="px-3 py-1.5 text-right font-medium">vs. A* lattice</th>
              <th className="px-3 py-1.5 text-right font-medium">first answer at</th>
              <th className="px-3 py-1.5 text-right font-medium">nodes held</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {LANES.map((lane) => {
              const sc = scoreFor(s, lane.key);
              const ratio = Number.isFinite(sc.cost) && Number.isFinite(optimal) ? sc.cost / optimal : NaN;
              return (
                <tr key={lane.key} className="border-b border-fd-border last:border-b-0">
                  <td className="px-3 py-1.5 font-ui">{lane.name}</td>
                  <td className="px-3 py-1.5 text-right">
                    {Number.isFinite(sc.cost) ? `${sc.cost.toFixed(2)} m` : '—'}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    {Number.isFinite(ratio) ? `×${ratio.toFixed(3)}` : '—'}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    {sc.firstAt === null
                      ? '—'
                      : lane.key === 'astar'
                        ? `${sc.firstAt} exp.`
                        : `${sc.firstAt} samples`}
                  </td>
                  <td className="px-3 py-1.5 text-right">{sc.nodes}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {chartSeries.length > 0 ? (
        <div className="border-t border-fd-border px-3 py-3">
          <LineChart
            series={chartSeries}
            xLabel="samples drawn"
            yLabel="best known path cost (m)"
            height={230}
            curve="stepAfter"
            markers={
              Number.isFinite(optimal)
                ? [{ axis: 'y', value: optimal, role: 'truth', label: 'A* lattice optimum' }]
                : []
            }
            ariaLabel="Best known path cost against samples drawn, for PRM, RRT and RRT*, with the A* lattice optimum marked as a horizontal reference line."
          />
        </div>
      ) : null}

      <ControlPanel columns={3}>
        <Slider
          label="Sample budget n"
          value={budget}
          min={500}
          max={5000}
          step={100}
          onChange={setBudget}
          format={(v) => v.toFixed(0)}
          help="Shared by PRM, RRT and RRT*. A* gets a proportional expansion budget."
        />
        <Toggle
          label="Show the C-obstacle (inflated map)"
          role="truth"
          checked={showInflated}
          onChange={setShowInflated}
        />
        <Toggle
          label="Keep refining (RRT* only)"
          role="posterior"
          checked={keepRefining}
          onChange={setKeepRefining}
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

function scoreFor(s: State, key: (typeof LANES)[number]['key']): Score {
  if (key === 'astar') return s.astarScore;
  if (key === 'prm') return s.prmScore;
  if (key === 'rrt') return s.rrtScore;
  return s.starScore;
}

function drawAstar(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  ox: number,
  oy: number,
  s: State,
  p: Palette,
  lattice: ReturnType<typeof latticeFromCSpace>,
) {
  const w = Math.ceil(sl(v, lattice.cellSize)) + 1;
  ctx.save();
  for (let k = 0; k < lattice.nx * lattice.ny; k++) {
    const closed = s.astar.closed[k] === 1;
    const open = s.astar.open[k] === 1;
    if (!closed && !open) continue;
    const c = cellCenter(lattice, k);
    ctx.globalAlpha = closed ? 0.16 : 0.55;
    ctx.fillStyle = p.prior;
    ctx.fillRect(
      sx(v, c.x + ox - lattice.cellSize / 2),
      sy(v, c.y + oy + lattice.cellSize / 2),
      w,
      w,
    );
  }
  ctx.restore();

  if (s.astarPath.length > 1) {
    strokePath(ctx, v, ox, oy, s.astarPath, p.posterior, 2.6);
  }
}

function drawPrm(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  ox: number,
  oy: number,
  s: State,
  p: Palette,
) {
  const { prm } = s;
  ctx.save();
  ctx.strokeStyle = p.prior;
  ctx.globalAlpha = 0.2;
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let i = 0; i < prm.nodes.length; i++) {
    for (const j of prm.adj[i]) {
      if (j < i) continue;
      ctx.moveTo(sx(v, prm.nodes[i].x + ox), sy(v, prm.nodes[i].y + oy));
      ctx.lineTo(sx(v, prm.nodes[j].x + ox), sy(v, prm.nodes[j].y + oy));
    }
  }
  ctx.stroke();
  ctx.globalAlpha = 0.75;
  ctx.fillStyle = p.prior;
  for (const q of prm.nodes) {
    ctx.beginPath();
    ctx.arc(sx(v, q.x + ox), sy(v, q.y + oy), 1.3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  if (s.prmSol) strokePath(ctx, v, ox, oy, s.prmSol.path, p.posterior, 2.6);
}

function drawTree(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  ox: number,
  oy: number,
  tree: Rrt,
  sol: PlanResult | null,
  p: Palette,
) {
  ctx.save();
  ctx.strokeStyle = p.prediction;
  ctx.globalAlpha = 0.45;
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  for (let i = 1; i < tree.nodes.length; i++) {
    const n = tree.nodes[i];
    const parent = tree.nodes[n.parent];
    ctx.moveTo(sx(v, parent.x + ox), sy(v, parent.y + oy));
    ctx.lineTo(sx(v, n.x + ox), sy(v, n.y + oy));
  }
  ctx.stroke();
  ctx.restore();
  if (sol) strokePath(ctx, v, ox, oy, sol.path, p.posterior, 2.6);
}

function strokePath(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  ox: number,
  oy: number,
  path: Point2[],
  color: string,
  lineWidth: number,
) {
  if (path.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(sx(v, path[0].x + ox), sy(v, path[0].y + oy));
  for (let i = 1; i < path.length; i++) ctx.lineTo(sx(v, path[i].x + ox), sy(v, path[i].y + oy));
  ctx.stroke();
  ctx.restore();
}
