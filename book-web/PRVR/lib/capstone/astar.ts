/**
 * The global planner: A* and a full Dijkstra cost field on the occupancy grid,
 * with the chance-constraint margin of Derivation F2 as a hard traversability
 * test and the ESDF as a soft preference.
 *
 * This is the Chapter 20 planner specialised to the one thing the capstone
 * needs it to do: route through a map that is mostly wrong, on the assumption
 * that the pose is right. That assumption has a name — certainty equivalence —
 * and the price of it is that a plan is only ever a hypothesis. The stack pays
 * that price by replanning at 1 Hz and by invalidating a path the instant a
 * novelty measurement lands on it.
 *
 * Rust counterpart: `crates/capstone/src/tasks/plan.rs`.
 */

import type { OccupancyGrid } from '../mapping/occgrid';
import { esdfIndexAt, type Esdf } from './esdf';

export const FREE = 0;
export const OCCUPIED = 1;
export const UNKNOWN = 2;

export interface PlanConfig {
  /** Hard clearance requirement, metres: r_robot + k_σ σ_pose (Derivation F2). */
  margin: number;
  /** Clearance the planner would *like*; below it, cost rises quadratically. */
  prefer: number;
  /** Extra multiplier for stepping into a cell the map has never seen. */
  unknownPenalty: number;
  /** Weight on the clearance preference term. */
  clearanceWeight: number;
}

export const DEFAULT_PLAN: PlanConfig = {
  margin: 0.22,
  prefer: 0.45,
  unknownPenalty: 1.6,
  clearanceWeight: 3.0,
};

/**
 * Three-way cell classification.
 *
 * The book's mapping chapter draws the same three states in grayscale; the
 * planner needs them as an enum because "unknown" is not halfway between free
 * and occupied — it is a different kind of thing, traversable but expensive,
 * and it is where all the information gain lives.
 */
export function classifyGrid(grid: OccupancyGrid, freeBelow = 0.38, occAbove = 0.62): Uint8Array {
  const probs = grid.getProbabilityArray();
  const out = new Uint8Array(probs.length);
  for (let k = 0; k < probs.length; k++) {
    out[k] = probs[k] >= occAbove ? OCCUPIED : probs[k] <= freeBelow ? FREE : UNKNOWN;
  }
  return out;
}

/** Cost of *entering* cell (i, j), or Infinity if the margin forbids it. */
export function cellCost(cls: Uint8Array, esdf: Esdf, i: number, j: number, cfg: PlanConfig): number {
  const k = j * esdf.width + i;
  if (cls[k] === OCCUPIED) return Infinity;
  const d = esdfIndexAt(esdf, i, j);
  if (d < cfg.margin) return Infinity;
  const shortfall = Math.max(0, cfg.prefer - d);
  return (
    (cls[k] === UNKNOWN ? cfg.unknownPenalty : 1) +
    cfg.clearanceWeight * shortfall * shortfall
  );
}

/* -------------------------------------------------------------------------- */
/* A binary heap, because the alternative is a linear scan per pop             */
/* -------------------------------------------------------------------------- */

class MinHeap {
  private keys: number[] = [];
  private vals: number[] = [];

  get size(): number {
    return this.keys.length;
  }

  push(key: number, val: number): void {
    this.keys.push(key);
    this.vals.push(val);
    let i = this.keys.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.keys[p] <= this.keys[i]) break;
      this.swap(i, p);
      i = p;
    }
  }

  pop(): number {
    const top = this.vals[0];
    const lastKey = this.keys.pop() as number;
    const lastVal = this.vals.pop() as number;
    if (this.keys.length > 0) {
      this.keys[0] = lastKey;
      this.vals[0] = lastVal;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.keys.length && this.keys[l] < this.keys[m]) m = l;
        if (r < this.keys.length && this.keys[r] < this.keys[m]) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }

  private swap(a: number, b: number): void {
    const k = this.keys[a];
    this.keys[a] = this.keys[b];
    this.keys[b] = k;
    const v = this.vals[a];
    this.vals[a] = this.vals[b];
    this.vals[b] = v;
  }
}

const NEIGHBORS: [number, number, number][] = [
  [1, 0, 1],
  [-1, 0, 1],
  [0, 1, 1],
  [0, -1, 1],
  [1, 1, Math.SQRT2],
  [1, -1, Math.SQRT2],
  [-1, 1, Math.SQRT2],
  [-1, -1, Math.SQRT2],
];

export interface PlanResult {
  /** World-space waypoints, start first. Empty when no route exists. */
  path: [number, number][];
  cost: number;
  expanded: number;
  found: boolean;
}

/**
 * `plan_tick` — A* on the 8-connected grid with the octile heuristic.
 *
 * The heuristic is admissible for 8-connectivity with unit step costs, and
 * every cell cost here is ≥ 1, so it stays admissible when the clearance
 * penalty is switched on. That matters: an inadmissible heuristic on a costmap
 * produces paths that hug walls in a way readers reliably mistake for a bug.
 */
export function astarGrid(
  grid: OccupancyGrid,
  cls: Uint8Array,
  esdf: Esdf,
  start: [number, number],
  goal: [number, number],
  cfg: PlanConfig = DEFAULT_PLAN,
): PlanResult {
  const { width: nx, height: ny } = grid;
  const [si, sj] = grid.worldToCell(start[0], start[1]);
  const [gi, gj] = grid.worldToCell(goal[0], goal[1]);
  const sIdx = sj * nx + si;
  const gIdx = gj * nx + gi;
  if (si < 0 || sj < 0 || si >= nx || sj >= ny || gi < 0 || gj < 0 || gi >= nx || gj >= ny) {
    return { path: [], cost: Infinity, expanded: 0, found: false };
  }

  const g = new Float64Array(nx * ny).fill(Infinity);
  const from = new Int32Array(nx * ny).fill(-1);
  const closed = new Uint8Array(nx * ny);
  const open = new MinHeap();

  const h = (i: number, j: number) => {
    const dx = Math.abs(i - gi);
    const dy = Math.abs(j - gj);
    return (Math.max(dx, dy) + (Math.SQRT2 - 1) * Math.min(dx, dy)) * grid.cellSize;
  };

  g[sIdx] = 0;
  open.push(h(si, sj), sIdx);
  let expanded = 0;

  while (open.size > 0) {
    const cur = open.pop();
    if (closed[cur]) continue;
    closed[cur] = 1;
    expanded++;
    if (cur === gIdx) break;

    const ci = cur % nx;
    const cj = (cur - ci) / nx;
    for (const [di, dj, len] of NEIGHBORS) {
      const ni = ci + di;
      const nj = cj + dj;
      if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
      const nIdx = nj * nx + ni;
      if (closed[nIdx]) continue;
      const c = cellCost(cls, esdf, ni, nj, cfg);
      if (!Number.isFinite(c)) continue;
      const tentative = g[cur] + c * len * grid.cellSize;
      if (tentative < g[nIdx]) {
        g[nIdx] = tentative;
        from[nIdx] = cur;
        open.push(tentative + h(ni, nj), nIdx);
      }
    }
  }

  if (!Number.isFinite(g[gIdx])) return { path: [], cost: Infinity, expanded, found: false };

  const cells: number[] = [];
  for (let k = gIdx; k !== -1; k = from[k]) {
    cells.push(k);
    if (k === sIdx) break;
  }
  cells.reverse();
  const path = cells.map((k) => {
    const i = k % nx;
    const j = (k - i) / nx;
    return grid.cellCenter(i, j);
  });
  return { path, cost: g[gIdx], expanded, found: true };
}

/**
 * One Dijkstra sweep from the robot, giving the true path cost to *every*
 * reachable cell.
 *
 * The frontier scorer needs a distance to each candidate, and Euclidean
 * distance is a lie in a floorplan: the room across the wall is 1 m away and
 * 14 m of driving. One sweep costs about the same as one A* and prices every
 * frontier honestly, so the greedy choice in Chapter 24's utility is at least
 * greedy about the right number.
 */
export function dijkstraCostField(
  grid: OccupancyGrid,
  cls: Uint8Array,
  esdf: Esdf,
  start: [number, number],
  cfg: PlanConfig = DEFAULT_PLAN,
): Float64Array {
  const { width: nx, height: ny } = grid;
  const [si, sj] = grid.worldToCell(start[0], start[1]);
  const g = new Float64Array(nx * ny).fill(Infinity);
  if (si < 0 || sj < 0 || si >= nx || sj >= ny) return g;

  const closed = new Uint8Array(nx * ny);
  const open = new MinHeap();
  g[sj * nx + si] = 0;
  open.push(0, sj * nx + si);

  while (open.size > 0) {
    const cur = open.pop();
    if (closed[cur]) continue;
    closed[cur] = 1;
    const ci = cur % nx;
    const cj = (cur - ci) / nx;
    for (const [di, dj, len] of NEIGHBORS) {
      const ni = ci + di;
      const nj = cj + dj;
      if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
      const nIdx = nj * nx + ni;
      if (closed[nIdx]) continue;
      const c = cellCost(cls, esdf, ni, nj, cfg);
      if (!Number.isFinite(c)) continue;
      const tentative = g[cur] + c * len * grid.cellSize;
      if (tentative < g[nIdx]) {
        g[nIdx] = tentative;
        open.push(tentative, nIdx);
      }
    }
  }
  return g;
}

/**
 * Drop waypoints that lie on the straight line between their neighbours.
 *
 * Purely cosmetic for the controller — MPPI tracks the polyline either way —
 * but it turns a 90-cell staircase into six segments, which is what the reader
 * sees and what a Dubins smoother would consume in the Rust version.
 */
export function simplifyPath(path: readonly [number, number][], tol = 0.05): [number, number][] {
  if (path.length <= 2) return path.map((p) => [p[0], p[1]] as [number, number]);
  const out: [number, number][] = [[path[0][0], path[0][1]]];
  for (let i = 1; i < path.length - 1; i++) {
    const a = out[out.length - 1];
    const b = path[i];
    const c = path[i + 1];
    const cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const len = Math.hypot(c[0] - a[0], c[1] - a[1]);
    if (len < 1e-9 || Math.abs(cross) / len > tol) out.push([b[0], b[1]]);
  }
  out.push([path[path.length - 1][0], path[path.length - 1][1]]);
  return out;
}
