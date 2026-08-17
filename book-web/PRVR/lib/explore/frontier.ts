/**
 * Frontiers: the structural backbone of exploration — Yamauchi (1997).
 *
 * A **frontier cell** is a known-free cell with at least one unknown 4-neighbour.
 * It is the only place in the map where a sensor can convert ignorance into
 * evidence, and the reason the definition is worth taking seriously is the
 * completeness argument in the chapter: every path from the robot into unknown
 * space must cross a frontier, so "drive to any reachable frontier until none
 * remain" maps the whole reachable closure.
 *
 * This module also carries the navigation field the utility function needs.
 * Cost is not Euclidean distance — a frontier three metres away through a wall
 * is not three metres away — so path cost comes from a Dijkstra expansion over
 * the *known-free* cells, which doubles as the reachability test.
 *
 * Rust counterpart: `crates/ch24_explore/src/frontier.rs`.
 */

import { logOddsToProb, type OccupancyGrid } from '../mapping/occgrid';

export interface GridIdx {
  i: number;
  j: number;
}

export type CellClass = 'free' | 'occupied' | 'unknown';

export interface ClassThresholds {
  /** p below this is known-free. Two sweeps of Chapter 13's model get you here. */
  freeBelow: number;
  /** p above this is known-occupied. */
  occAbove: number;
}

export const DEFAULT_THRESHOLDS: ClassThresholds = { freeBelow: 0.3, occAbove: 0.7 };

export function classifyCell(
  grid: OccupancyGrid,
  i: number,
  j: number,
  th: ClassThresholds = DEFAULT_THRESHOLDS,
): CellClass {
  if (!grid.inBounds(i, j)) return 'occupied'; // outside the map is not explorable
  const p = logOddsToProb(grid.logOdds[grid.index(i, j)]);
  if (p < th.freeBelow) return 'free';
  if (p > th.occAbove) return 'occupied';
  return 'unknown';
}

const N4: [number, number][] = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

const N8: [number, number][] = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

/** Free, and touching the unknown — Yamauchi's definition, cell for cell. */
export function isFrontierCell(
  grid: OccupancyGrid,
  i: number,
  j: number,
  th: ClassThresholds = DEFAULT_THRESHOLDS,
): boolean {
  if (classifyCell(grid, i, j, th) !== 'free') return false;
  for (const [di, dj] of N4) {
    if (!grid.inBounds(i + di, j + dj)) continue;
    if (classifyCell(grid, i + di, j + dj, th) === 'unknown') return true;
  }
  return false;
}

export interface Frontier {
  cells: GridIdx[];
  /** Centre of mass in world coordinates. */
  centroid: [number, number];
  /** The member cell nearest the centroid — a real, occupiable target. */
  representative: GridIdx;
  size: number;
}

export interface FrontierOptions {
  thresholds?: ClassThresholds;
  /**
   * Regions smaller than this are dropped. A single stray frontier cell is
   * usually a speckle in the log-odds map, and chasing it costs a whole
   * navigation cycle to gain a fraction of a bit.
   */
  minSize?: number;
}

/**
 * `detect_frontiers` — label the frontier cells, then group them into regions
 * by 8-connected BFS. One pass over the grid to mark, one to flood: O(#cells).
 *
 * Grouping matters more than it looks. Scoring individual cells produces
 * hundreds of near-identical candidates and a robot that dithers between two of
 * them; scoring regions produces a handful of genuinely different places to go.
 */
export function detectFrontiers(grid: OccupancyGrid, opts: FrontierOptions = {}): Frontier[] {
  const th = opts.thresholds ?? DEFAULT_THRESHOLDS;
  const minSize = opts.minSize ?? 4;
  const n = grid.width * grid.height;

  const mark = new Uint8Array(n);
  for (let j = 0; j < grid.height; j++) {
    for (let i = 0; i < grid.width; i++) {
      if (isFrontierCell(grid, i, j, th)) mark[grid.index(i, j)] = 1;
    }
  }

  const seen = new Uint8Array(n);
  const out: Frontier[] = [];
  const queue = new Int32Array(n);

  for (let j0 = 0; j0 < grid.height; j0++) {
    for (let i0 = 0; i0 < grid.width; i0++) {
      const start = grid.index(i0, j0);
      if (!mark[start] || seen[start]) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = start;
      seen[start] = 1;
      const cells: GridIdx[] = [];

      while (head < tail) {
        const idx = queue[head++];
        const i = idx % grid.width;
        const j = (idx - i) / grid.width;
        cells.push({ i, j });
        for (const [di, dj] of N8) {
          const ni = i + di;
          const nj = j + dj;
          if (!grid.inBounds(ni, nj)) continue;
          const nidx = grid.index(ni, nj);
          if (!mark[nidx] || seen[nidx]) continue;
          seen[nidx] = 1;
          queue[tail++] = nidx;
        }
      }

      if (cells.length < minSize) continue;

      let cx = 0;
      let cy = 0;
      for (const c of cells) {
        const [wx, wy] = grid.cellCenter(c.i, c.j);
        cx += wx;
        cy += wy;
      }
      cx /= cells.length;
      cy /= cells.length;

      // The centroid of a C-shaped region can land on a wall, so the target is
      // the member cell closest to it — always a real frontier cell.
      let rep = cells[0];
      let best = Infinity;
      for (const c of cells) {
        const [wx, wy] = grid.cellCenter(c.i, c.j);
        const d = (wx - cx) ** 2 + (wy - cy) ** 2;
        if (d < best) {
          best = d;
          rep = c;
        }
      }

      out.push({ cells, centroid: [cx, cy], representative: rep, size: cells.length });
    }
  }

  return out;
}

/* -------------------------------------------------------------------------- */
/* Navigation over known-free space                                            */
/* -------------------------------------------------------------------------- */

/** A tiny binary heap keyed by cost — enough for a 100 × 80 grid. */
class MinHeap {
  private cost: number[] = [];
  private item: number[] = [];

  get size(): number {
    return this.item.length;
  }

  push(c: number, v: number): void {
    this.cost.push(c);
    this.item.push(v);
    let k = this.item.length - 1;
    while (k > 0) {
      const parent = (k - 1) >> 1;
      if (this.cost[parent] <= this.cost[k]) break;
      this.swap(parent, k);
      k = parent;
    }
  }

  pop(): number {
    const top = this.item[0];
    const lastC = this.cost.pop() as number;
    const lastI = this.item.pop() as number;
    if (this.item.length > 0) {
      this.cost[0] = lastC;
      this.item[0] = lastI;
      let k = 0;
      for (;;) {
        const l = 2 * k + 1;
        const r = l + 1;
        let m = k;
        if (l < this.item.length && this.cost[l] < this.cost[m]) m = l;
        if (r < this.item.length && this.cost[r] < this.cost[m]) m = r;
        if (m === k) break;
        this.swap(m, k);
        k = m;
      }
    }
    return top;
  }

  private swap(a: number, b: number): void {
    [this.cost[a], this.cost[b]] = [this.cost[b], this.cost[a]];
    [this.item[a], this.item[b]] = [this.item[b], this.item[a]];
  }
}

export interface NavFieldOptions {
  thresholds?: ClassThresholds;
  /**
   * Cells within this many cells of an occupied cell are refused. This is the
   * planner's body radius, and it is also the chapter's favourite failure mode:
   * inflate by one cell too many and every doorway in the apartment closes.
   */
  inflate?: number;
}

/**
 * Dijkstra over known-free cells, 8-connected, in metres.
 *
 * `cost[idx]` is the length of the shortest known-free path from the start;
 * `Infinity` means "not reachable through what we have mapped so far", which is
 * exactly the reachability predicate the completeness argument needs.
 */
export class NavField {
  readonly cost: Float64Array;
  readonly parent: Int32Array;
  readonly blocked: Uint8Array;
  readonly start: number;

  constructor(
    readonly grid: OccupancyGrid,
    startCell: GridIdx,
    opts: NavFieldOptions = {},
  ) {
    const th = opts.thresholds ?? DEFAULT_THRESHOLDS;
    const inflate = opts.inflate ?? 1;
    const n = grid.width * grid.height;

    this.blocked = new Uint8Array(n);
    for (let j = 0; j < grid.height; j++) {
      for (let i = 0; i < grid.width; i++) {
        if (classifyCell(grid, i, j, th) === 'occupied') {
          for (let dj = -inflate; dj <= inflate; dj++) {
            for (let di = -inflate; di <= inflate; di++) {
              if (grid.inBounds(i + di, j + dj)) this.blocked[grid.index(i + di, j + dj)] = 1;
            }
          }
        }
      }
    }

    this.cost = new Float64Array(n).fill(Infinity);
    this.parent = new Int32Array(n).fill(-1);
    this.start = grid.index(startCell.i, startCell.j);

    // The robot is standing where it is standing: never let inflation strand it.
    const heap = new MinHeap();
    this.cost[this.start] = 0;
    heap.push(0, this.start);

    const diag = Math.SQRT2 * grid.cellSize;
    const straight = grid.cellSize;

    while (heap.size > 0) {
      const idx = heap.pop();
      const i = idx % grid.width;
      const j = (idx - i) / grid.width;
      const base = this.cost[idx];
      for (const [di, dj] of N8) {
        const ni = i + di;
        const nj = j + dj;
        if (!grid.inBounds(ni, nj)) continue;
        const nidx = grid.index(ni, nj);
        if (this.blocked[nidx]) continue;
        // Travel only through space we have seen and believe is empty.
        if (classifyCell(grid, ni, nj, th) !== 'free') continue;
        const step = di !== 0 && dj !== 0 ? diag : straight;
        const next = base + step;
        if (next < this.cost[nidx] - 1e-12) {
          this.cost[nidx] = next;
          this.parent[nidx] = idx;
          heap.push(next, nidx);
        }
      }
    }
  }

  costAt(cell: GridIdx): number {
    if (!this.grid.inBounds(cell.i, cell.j)) return Infinity;
    return this.cost[this.grid.index(cell.i, cell.j)];
  }

  /**
   * The cheapest reachable cell within `radius` cells of `goal`.
   *
   * Frontier cells sit against the unknown, and inflation often makes the exact
   * cell unreachable while its neighbour two cells back is fine. Aiming at the
   * neighbour is what keeps a real explorer from declaring a doorway impossible.
   */
  nearestReachable(goal: GridIdx, radius = 4): { cell: GridIdx; cost: number } | null {
    let best: { cell: GridIdx; cost: number } | null = null;
    for (let dj = -radius; dj <= radius; dj++) {
      for (let di = -radius; di <= radius; di++) {
        const cell = { i: goal.i + di, j: goal.j + dj };
        const c = this.costAt(cell);
        if (!Number.isFinite(c)) continue;
        // Prefer cheap paths, breaking ties toward the frontier itself.
        const penalty = 0.25 * this.grid.cellSize * Math.hypot(di, dj);
        if (best === null || c + penalty < best.cost) best = { cell, cost: c + penalty };
      }
    }
    if (best === null) return null;
    return { cell: best.cell, cost: this.costAt(best.cell) };
  }

  /** Walk the parent pointers back from `goal`, returning world waypoints. */
  pathTo(goal: GridIdx): { x: number; y: number }[] {
    const out: { x: number; y: number }[] = [];
    let idx = this.grid.index(goal.i, goal.j);
    if (!this.grid.inBounds(goal.i, goal.j) || !Number.isFinite(this.cost[idx])) return out;
    let guard = 0;
    while (idx !== -1 && guard++ < this.cost.length) {
      const i = idx % this.grid.width;
      const j = (idx - i) / this.grid.width;
      const [x, y] = this.grid.cellCenter(i, j);
      out.push({ x, y });
      if (idx === this.start) break;
      idx = this.parent[idx];
    }
    return out.reverse();
  }
}

/** Fraction of the grid that is no longer at the prior — the "coverage" readout. */
export function coverage(grid: OccupancyGrid, band = 0.05): number {
  let n = 0;
  for (let k = 0; k < grid.logOdds.length; k++) {
    if (Math.abs(logOddsToProb(grid.logOdds[k]) - 0.5) >= band) n++;
  }
  return n / grid.logOdds.length;
}
