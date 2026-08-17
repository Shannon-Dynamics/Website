/**
 * Sampling-based motion planning: PRM, RRT, and RRT*.
 *
 * The move all three make is the one Chapter 8 made for beliefs — stop trying
 * to represent a continuous space exactly and *sample* it instead. What you
 * give up is completeness in the strict sense; what you get back is an
 * algorithm whose cost does not explode with the dimension of Q. The precise
 * replacement guarantee (probabilistic completeness, and for RRT* asymptotic
 * optimality) is derived in the chapter; this file is the machinery.
 *
 * Every planner here is *incremental*: one `step()` does one sample's worth of
 * work, so the Planner Arena can animate them side by side at a fixed frame
 * rate and the scoreboard can honestly report "nodes so far".
 */

import type { Rng } from '../prob/rng';
import type { Point2 } from '../sim/world';
import type { CSpace2 } from './cspace';

// ---------------------------------------------------------------------------
// Shared vocabulary
// ---------------------------------------------------------------------------

export const dist2 = (a: Point2, b: Point2) => Math.hypot(a.x - b.x, a.y - b.y);

/** Move from `a` toward `b`, but no further than `maxLen`. */
export function steerToward(a: Point2, b: Point2, maxLen: number): Point2 {
  const d = dist2(a, b);
  if (d <= maxLen) return { x: b.x, y: b.y };
  const t = maxLen / d;
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

export interface PlanResult {
  path: Point2[];
  cost: number;
  /** Nodes held in memory when the answer was produced. */
  nodes: number;
  /** Samples drawn — the budget the arena's slider controls. */
  iterations: number;
}

// ---------------------------------------------------------------------------
// Probabilistic roadmaps
// ---------------------------------------------------------------------------

export interface PrmOptions {
  /** Neighbours each new milestone tries to connect to. */
  k?: number;
  /** Connections beyond this radius are not attempted (metres). */
  maxEdgeLength?: number;
  /** Draw samples from this function instead of the uniform distribution. */
  sampler?: (rng: Rng) => Point2 | null;
}

/**
 * `build_prm(n, k, sample, connect)` — the multi-query roadmap.
 *
 * Learning phase: scatter milestones in Q_free and wire each to its k nearest
 * neighbours through the local planner (here: a straight line, checked by
 * sphere marching). Query phase: attach start and goal, then search the
 * roadmap. The roadmap is built *once* and reused, which is the whole reason
 * PRM exists — and the reason it loses to RRT on a single query.
 */
export class Prm {
  readonly cs: CSpace2;
  readonly nodes: Point2[] = [];
  readonly adj: number[][] = [];
  readonly edgeCost: number[][] = [];
  private k: number;
  private maxEdge: number;
  private sampler?: (rng: Rng) => Point2 | null;
  /** Straight-line checks attempted — the honest measure of PRM's real cost. */
  localPlans = 0;
  edges = 0;

  constructor(cs: CSpace2, opts: PrmOptions = {}) {
    this.cs = cs;
    this.k = opts.k ?? 8;
    this.maxEdge = opts.maxEdgeLength ?? 3.0;
    this.sampler = opts.sampler;
  }

  /** Draw one milestone and try to wire it in. Returns the new node index, or −1. */
  step(rng: Rng): number {
    const q = this.sampler ? this.sampler(rng) : this.cs.sampleFree(rng, 1);
    if (!q) return -1;
    return this.addMilestone(q);
  }

  addMilestone(q: Point2): number {
    const id = this.nodes.length;
    this.nodes.push(q);
    this.adj.push([]);
    this.edgeCost.push([]);
    this.connect(id);
    return id;
  }

  /** Wire node `id` to its k nearest existing neighbours. */
  private connect(id: number): void {
    const q = this.nodes[id];
    const cand: { j: number; d: number }[] = [];
    for (let j = 0; j < this.nodes.length; j++) {
      if (j === id) continue;
      const d = dist2(q, this.nodes[j]);
      if (d <= this.maxEdge) cand.push({ j, d });
    }
    cand.sort((a, b) => a.d - b.d);
    for (let n = 0; n < Math.min(this.k, cand.length); n++) {
      const { j, d } = cand[n];
      if (this.adj[id].includes(j)) continue;
      this.localPlans++;
      if (!this.cs.edgeFree(q, this.nodes[j])) continue;
      this.adj[id].push(j);
      this.edgeCost[id].push(d);
      this.adj[j].push(id);
      this.edgeCost[j].push(d);
      this.edges++;
    }
  }

  /**
   * Attach start and goal to the roadmap and run Dijkstra over it.
   *
   * The attachment is temporary: the roadmap itself is not modified, so the
   * same roadmap answers the next query too.
   */
  query(start: Point2, goal: Point2): PlanResult | null {
    const n = this.nodes.length;
    const S = n;
    const G = n + 1;
    const pts = [...this.nodes, start, goal];
    const adj: number[][] = [...this.adj.map((a) => [...a]), [], []];
    const cost: number[][] = [...this.edgeCost.map((c) => [...c]), [], []];

    const attach = (terminal: number) => {
      const q = pts[terminal];
      const cand = this.nodes
        .map((p, j) => ({ j, d: dist2(p, q) }))
        .filter((c) => c.d <= this.maxEdge)
        .sort((a, b) => a.d - b.d);
      let made = 0;
      for (const c of cand) {
        if (made >= this.k) break;
        if (!this.cs.edgeFree(q, this.nodes[c.j])) continue;
        adj[terminal].push(c.j);
        cost[terminal].push(c.d);
        adj[c.j].push(terminal);
        cost[c.j].push(c.d);
        made++;
      }
      // A start that sees no milestone is still connectable straight to goal.
      return made;
    };
    attach(S);
    attach(G);
    if (this.cs.edgeFree(start, goal)) {
      const d = dist2(start, goal);
      adj[S].push(G);
      cost[S].push(d);
      adj[G].push(S);
      cost[G].push(d);
    }

    const g = new Float64Array(pts.length).fill(Infinity);
    const parent = new Int32Array(pts.length).fill(-1);
    const done = new Uint8Array(pts.length);
    g[S] = 0;
    for (;;) {
      let best = -1;
      let bestG = Infinity;
      for (let i = 0; i < pts.length; i++) {
        if (!done[i] && g[i] < bestG) {
          bestG = g[i];
          best = i;
        }
      }
      if (best === -1) break;
      if (best === G) break;
      done[best] = 1;
      for (let e = 0; e < adj[best].length; e++) {
        const nb = adj[best][e];
        const cand = g[best] + cost[best][e];
        if (cand < g[nb]) {
          g[nb] = cand;
          parent[nb] = best;
        }
      }
    }
    if (!Number.isFinite(g[G])) return null;

    const path: Point2[] = [];
    for (let c = G; c !== -1; c = parent[c]) path.push(pts[c]);
    path.reverse();
    return { path, cost: g[G], nodes: n, iterations: n };
  }
}

/**
 * The **bridge test** of Hsu et al.: draw a point, draw a second point a
 * Gaussian step away, and keep the *midpoint* only when both endpoints are in
 * collision and the midpoint is free.
 *
 * Uniform sampling finds a narrow passage with probability proportional to the
 * passage's volume, which is exactly the quantity that vanishes as the passage
 * narrows. The bridge test conditions on a geometric signature of a passage
 * instead, so its hit rate falls far more slowly — visibly so in w20.2.
 */
export function bridgeSampler(cs: CSpace2, sigma = 0.6) {
  return (rng: Rng): Point2 | null => {
    for (let attempt = 0; attempt < 30; attempt++) {
      const b = cs.bounds;
      const p: Point2 = { x: rng.uniform(b.minX, b.maxX), y: rng.uniform(b.minY, b.maxY) };
      if (cs.isFree(p.x, p.y)) continue; // first endpoint must be *inside* an obstacle
      const q: Point2 = { x: p.x + rng.normal(0, sigma), y: p.y + rng.normal(0, sigma) };
      if (cs.isFree(q.x, q.y)) continue; // so must the second
      const m: Point2 = { x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 };
      if (cs.isFree(m.x, m.y)) return m; // the midpoint bridges them: a passage
    }
    return null;
  };
}

/**
 * Hsu et al.'s hybrid strategy: mix bridge samples with uniform ones. Pure
 * bridge sampling finds passages and nothing else — it would never populate the
 * open rooms the passage has to connect.
 */
export function hybridSampler(cs: CSpace2, bridgeFraction = 0.6, sigma = 0.7) {
  const bridge = bridgeSampler(cs, sigma);
  return (rng: Rng): Point2 | null => {
    if (rng.next() < bridgeFraction) {
      const b = bridge(rng);
      if (b) return b;
    }
    return cs.sampleFree(rng, 1);
  };
}

// ---------------------------------------------------------------------------
// Rapidly-exploring random trees
// ---------------------------------------------------------------------------

export interface RrtNode {
  x: number;
  y: number;
  parent: number;
  /** Cost-to-come from the root along tree edges. */
  cost: number;
}

export interface RrtOptions {
  /** Maximum edge length ("ε" in Choset's EXTEND). */
  stepSize?: number;
  /** Probability of sampling the goal instead of a uniform point. */
  goalBias?: number;
  /** How close to the goal counts as arrival. */
  goalTolerance?: number;
  /** Turn on choose-parent + rewire: this is what makes it RRT*. */
  star?: boolean;
  /** γ in r_n = γ (log n / n)^(1/d); must exceed the Karaman–Frazzoli bound. */
  gamma?: number;
  /** Hard cap on the connection radius, so early iterations stay cheap. */
  maxRadius?: number;
}

export type ExtendStatus = 'reached' | 'advanced' | 'trapped';

/**
 * RRT and RRT* in one tree, because they differ by exactly two operations.
 *
 * `star = false` gives Choset's Algorithm 13: sample, find the nearest node,
 * extend by ε, keep it if the edge is free. The tree is never revisited, and
 * that is precisely why RRT converges to a suboptimal path with probability 1.
 *
 * `star = true` adds Karaman & Frazzoli's two repairs inside a ball of radius
 * r_n = γ (log n / n)^(1/d):
 *   • **choose parent** — connect the new node to whichever near node minimises
 *     its cost-to-come, not to the nearest one;
 *   • **rewire** — re-parent any near node whose cost-to-come improves by going
 *     through the new node.
 * The radius shrinks, but slowly enough (log n / n) that near-optimal
 * connections keep being available. That is the whole proof idea.
 */
export class Rrt {
  readonly cs: CSpace2;
  readonly nodes: RrtNode[] = [];
  /** Child lists, kept in step with `parent` so a rewire can refresh a subtree. */
  readonly children: number[][] = [];
  readonly goal: Point2;
  private opts: Required<RrtOptions>;
  /** Index of the cheapest node that reached the goal region, or −1. */
  goalNode = -1;
  bestCost = Infinity;
  iterations = 0;
  /** (samples, best cost) whenever the solution improves — the arena plots this. */
  readonly costHistory: { n: number; cost: number }[] = [];
  /** Index of the most recently added node, for drawing the growing edge. */
  lastAdded = -1;
  lastRewired: number[] = [];

  constructor(cs: CSpace2, start: Point2, goal: Point2, opts: RrtOptions = {}) {
    this.cs = cs;
    this.goal = goal;
    this.opts = {
      stepSize: opts.stepSize ?? 0.6,
      goalBias: opts.goalBias ?? 0.05,
      goalTolerance: opts.goalTolerance ?? 0.35,
      star: opts.star ?? false,
      gamma: opts.gamma ?? 6.0,
      maxRadius: opts.maxRadius ?? 2.0,
    };
    this.nodes.push({ x: start.x, y: start.y, parent: -1, cost: 0 });
    this.children.push([]);
  }

  private addNode(q: Point2, parent: number, cost: number): number {
    const id = this.nodes.length;
    this.nodes.push({ x: q.x, y: q.y, parent, cost });
    this.children.push([]);
    if (parent >= 0) this.children[parent].push(id);
    return id;
  }

  /** r_n = min(γ (log n / n)^{1/d}, ε) with d = 2 — the shrinking ball. */
  connectionRadius(): number {
    const n = Math.max(this.nodes.length, 2);
    const r = this.opts.gamma * Math.sqrt(Math.log(n) / n);
    return Math.min(r, this.opts.maxRadius);
  }

  private nearest(q: Point2): number {
    let best = 0;
    let bestD = Infinity;
    for (let i = 0; i < this.nodes.length; i++) {
      const d = (this.nodes[i].x - q.x) ** 2 + (this.nodes[i].y - q.y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  private near(q: Point2, r: number): number[] {
    const out: number[] = [];
    const r2 = r * r;
    for (let i = 0; i < this.nodes.length; i++) {
      if ((this.nodes[i].x - q.x) ** 2 + (this.nodes[i].y - q.y) ** 2 <= r2) out.push(i);
    }
    return out;
  }

  /** `rrt_extend(T, q_rand)` — one sample's worth of growth. */
  step(rng: Rng): ExtendStatus {
    this.iterations++;
    this.lastAdded = -1;
    this.lastRewired = [];

    const b = this.cs.bounds;
    const qRand: Point2 =
      rng.next() < this.opts.goalBias
        ? { ...this.goal }
        : { x: rng.uniform(b.minX, b.maxX), y: rng.uniform(b.minY, b.maxY) };

    const nearestIdx = this.nearest(qRand);
    const qNear = this.nodes[nearestIdx];
    const qNew = steerToward(qNear, qRand, this.opts.stepSize);
    if (!this.cs.isFree(qNew.x, qNew.y)) return 'trapped';
    if (!this.cs.edgeFree(qNear, qNew)) return 'trapped';

    let parent = nearestIdx;
    let cost = qNear.cost + dist2(qNear, qNew);

    if (this.opts.star) {
      const r = this.connectionRadius();
      const nearIdx = this.near(qNew, r);
      // choose parent: the cheapest *collision-free* connection in the ball
      for (const i of nearIdx) {
        const c = this.nodes[i].cost + dist2(this.nodes[i], qNew);
        if (c < cost - 1e-12 && this.cs.edgeFree(this.nodes[i], qNew)) {
          parent = i;
          cost = c;
        }
      }
      const id = this.addNode(qNew, parent, cost);
      this.lastAdded = id;
      // rewire: does anyone in the ball reach home cheaper through the newcomer?
      for (const i of nearIdx) {
        if (i === parent || i === 0) continue;
        const c = cost + dist2(qNew, this.nodes[i]);
        if (c < this.nodes[i].cost - 1e-12 && this.cs.edgeFree(qNew, this.nodes[i])) {
          this.reparent(i, id, c);
          this.lastRewired.push(i);
        }
      }
    } else {
      this.lastAdded = this.addNode(qNew, parent, cost);
    }

    this.checkGoal(this.nodes.length - 1);
    if (this.opts.star && this.goalNode >= 0) this.refreshBest();
    return dist2(qNew, qRand) < 1e-9 ? 'reached' : 'advanced';
  }

  /**
   * Hang node `i` under `newParent` at cost `cost`, then push the saving down
   * its subtree. Skipping this propagation is the classic RRT* bug: the tree
   * looks rewired but the costs it minimises are stale, so it silently stops
   * being asymptotically optimal.
   */
  private reparent(i: number, newParent: number, cost: number): void {
    const old = this.nodes[i].parent;
    if (old >= 0) {
      const siblings = this.children[old];
      const at = siblings.indexOf(i);
      if (at >= 0) siblings.splice(at, 1);
    }
    this.nodes[i].parent = newParent;
    this.children[newParent].push(i);
    this.nodes[i].cost = cost;

    const stack = [i];
    while (stack.length > 0) {
      const p = stack.pop()!;
      for (const c of this.children[p]) {
        this.nodes[c].cost = this.nodes[p].cost + dist2(this.nodes[p], this.nodes[c]);
        stack.push(c);
      }
    }
  }

  private checkGoal(id: number): void {
    const n = this.nodes[id];
    if (dist2(n, this.goal) > this.opts.goalTolerance) return;
    if (!this.cs.edgeFree(n, this.goal)) return;
    const total = n.cost + dist2(n, this.goal);
    if (total < this.bestCost - 1e-9) {
      this.bestCost = total;
      this.goalNode = id;
      this.costHistory.push({ n: this.iterations, cost: total });
    }
  }

  /** After a rewire the incumbent goal branch may have got cheaper. */
  private refreshBest(): void {
    let best = Infinity;
    let bestIdx = -1;
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      if (dist2(n, this.goal) > this.opts.goalTolerance) continue;
      const total = n.cost + dist2(n, this.goal);
      if (total < best) {
        best = total;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0 && best < this.bestCost - 1e-9) {
      this.bestCost = best;
      this.goalNode = bestIdx;
      this.costHistory.push({ n: this.iterations, cost: best });
    }
  }

  solution(): PlanResult | null {
    if (this.goalNode < 0) return null;
    const path: Point2[] = [{ ...this.goal }];
    for (let c = this.goalNode; c !== -1; c = this.nodes[c].parent) {
      path.push({ x: this.nodes[c].x, y: this.nodes[c].y });
    }
    path.reverse();
    return {
      path,
      cost: this.bestCost,
      nodes: this.nodes.length,
      iterations: this.iterations,
    };
  }
}

/**
 * Random pairwise shortcutting: pick two points on the path, and if the
 * straight line between them is free, splice it in.
 *
 * A one-line post-process that removes most of RRT's jaggedness — and the
 * reason "RRT paths look bad" is a weaker complaint than it first appears.
 * It cannot, however, change the path's homotopy class: a shortcut never moves
 * the path to the other side of an obstacle, which is exactly what RRT* can do
 * and shortcutting cannot.
 */
export function shortcut(cs: CSpace2, path: Point2[], rng: Rng, iterations = 100): Point2[] {
  let out = path.map((p) => ({ ...p }));
  for (let k = 0; k < iterations && out.length > 2; k++) {
    const i = Math.floor(rng.uniform(0, out.length - 2));
    const j = Math.floor(rng.uniform(i + 2, out.length));
    if (j <= i + 1 || j >= out.length) continue;
    if (cs.edgeFree(out[i], out[j])) out = [...out.slice(0, i + 1), ...out.slice(j)];
  }
  return out;
}

/** Total length of a polyline path. */
export function pathCost(path: Point2[]): number {
  let c = 0;
  for (let i = 1; i < path.length; i++) c += dist2(path[i - 1], path[i]);
  return c;
}
