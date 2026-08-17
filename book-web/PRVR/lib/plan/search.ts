/**
 * Graph search on the planning lattice: Dijkstra, A*, and the wave-front.
 *
 * All three are the same loop. Dijkstra is A* with h ≡ 0; the wave-front
 * planner is Dijkstra run backwards from the goal without an early exit, so
 * that every cell ends up labelled with its cost-to-go. Writing them once and
 * parameterising the heuristic is not code golf — it is the chapter's point:
 * *the heuristic is the only difference*, and it buys nothing but expansions.
 *
 * The searcher is a state machine rather than a function so the Planner Arena
 * can advance it a few expansions per animation frame and draw the frontier.
 * `aStarGrid` wraps it for callers that only want the answer.
 */

import { NEIGHBORS, cellCenter, type Lattice } from './cspace';

/**
 * A binary min-heap keyed on f. Ties are broken by insertion order, which keeps
 * a run reproducible: two cells with identical f must still be expanded in a
 * defined order or the drawn frontier flickers between frames.
 */
class MinHeap {
  private key: number[] = [];
  private val: number[] = [];

  get size(): number {
    return this.key.length;
  }

  push(k: number, v: number): void {
    this.key.push(k);
    this.val.push(v);
    let i = this.key.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.key[parent] <= this.key[i]) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  pop(): number | undefined {
    if (this.key.length === 0) return undefined;
    const top = this.val[0];
    const lastK = this.key.pop()!;
    const lastV = this.val.pop()!;
    if (this.key.length > 0) {
      this.key[0] = lastK;
      this.val[0] = lastV;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.key.length && this.key[l] < this.key[m]) m = l;
        if (r < this.key.length && this.key[r] < this.key[m]) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }

  private swap(a: number, b: number): void {
    [this.key[a], this.key[b]] = [this.key[b], this.key[a]];
    [this.val[a], this.val[b]] = [this.val[b], this.val[a]];
  }
}

export type SearchStatus = 'running' | 'found' | 'failed';

export interface SearchOptions {
  /**
   * Heuristic weight ε. ε = 0 is Dijkstra, ε = 1 is A*, ε > 1 is weighted A*:
   * faster, and no longer optimal — the returned path is within a factor ε of
   * optimal, which is the bound the chapter derives.
   */
  epsilon?: number;
  /** Diagonal moves cost √2 by default; set false for a 4-connected lattice. */
  diagonals?: boolean;
}

/** Octile distance: the exact cost-to-go on an obstacle-free 8-connected grid. */
export function octile(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cellSize: number,
): number {
  const dx = Math.abs(ax - bx);
  const dy = Math.abs(ay - by);
  return (Math.max(dx, dy) + (Math.SQRT2 - 1) * Math.min(dx, dy)) * cellSize;
}

export interface SearchSnapshot {
  status: SearchStatus;
  /** Cost-to-come, ∞ where unreached. Indexed by cell. */
  g: Float64Array;
  /** 1 where the cell has been expanded (Thrun's "closed set"). */
  closed: Uint8Array;
  /** 1 where the cell is on the frontier (in the open list). */
  open: Uint8Array;
  expanded: number;
  /** Cells from start to goal, once found. */
  path: number[];
  cost: number;
}

/**
 * Best-first search over the lattice, one expansion at a time.
 *
 * The invariant that makes A* correct lives in `step`: a cell is expanded only
 * once (`closed`), and it is expanded in order of f = g + ε·h. With ε ≤ 1 and h
 * admissible, the first expansion of the goal is optimal.
 */
export class GridSearch {
  readonly grid: Lattice;
  readonly start: number;
  readonly goal: number;
  readonly g: Float64Array;
  readonly parent: Int32Array;
  readonly closed: Uint8Array;
  readonly open: Uint8Array;
  private heap = new MinHeap();
  private eps: number;
  private diagonals: boolean;
  private goalPt: { x: number; y: number };
  status: SearchStatus = 'running';
  expanded = 0;

  constructor(grid: Lattice, start: number, goal: number, opts: SearchOptions = {}) {
    this.grid = grid;
    this.start = start;
    this.goal = goal;
    this.eps = opts.epsilon ?? 1;
    this.diagonals = opts.diagonals ?? true;
    const n = grid.nx * grid.ny;
    this.g = new Float64Array(n).fill(Infinity);
    this.parent = new Int32Array(n).fill(-1);
    this.closed = new Uint8Array(n);
    this.open = new Uint8Array(n);
    this.goalPt = cellCenter(grid, goal);
    this.g[start] = 0;
    this.open[start] = 1;
    this.heap.push(this.f(start, 0), start);
  }

  /** Octile distance to the goal, in metres — admissible on an 8-connected grid. */
  private h(cell: number): number {
    const p = cellCenter(this.grid, cell);
    return octile(p.x, p.y, this.goalPt.x, this.goalPt.y, 1);
  }

  /** f = g + ε·h, the priority the open list is sorted on. */
  private f(cell: number, g: number): number {
    return g + this.eps * this.h(cell);
  }

  /** Advance the search by `n` expansions. */
  step(n = 1): SearchStatus {
    const { grid } = this;
    for (let k = 0; k < n && this.status === 'running'; k++) {
      const cur = this.heap.pop();
      if (cur === undefined) {
        this.status = 'failed';
        return this.status;
      }
      if (this.closed[cur]) {
        k--; // a stale heap entry costs no expansion
        continue;
      }
      this.closed[cur] = 1;
      this.open[cur] = 0;
      this.expanded++;

      if (cur === this.goal) {
        this.status = 'found';
        return this.status;
      }

      const i = cur % grid.nx;
      const j = Math.floor(cur / grid.nx);
      for (const [di, dj, w] of NEIGHBORS) {
        if (!this.diagonals && di !== 0 && dj !== 0) continue;
        const ni = i + di;
        const nj = j + dj;
        if (ni < 0 || nj < 0 || ni >= grid.nx || nj >= grid.ny) continue;
        const nb = nj * grid.nx + ni;
        if (grid.free[nb] === 0 || this.closed[nb]) continue;
        // No corner cutting: a diagonal move needs both orthogonal cells free,
        // or the disc clips the corner of an obstacle the lattice says is fine.
        if (di !== 0 && dj !== 0) {
          if (grid.free[j * grid.nx + ni] === 0 || grid.free[nj * grid.nx + i] === 0) continue;
        }
        const tentative = this.g[cur] + w * grid.cellSize;
        if (tentative < this.g[nb] - 1e-12) {
          this.g[nb] = tentative;
          this.parent[nb] = cur;
          this.open[nb] = 1;
          this.heap.push(this.f(nb, tentative), nb);
        }
      }
    }
    return this.status;
  }

  /** Run to completion (or to `limit` expansions). */
  run(limit = 1e7): SearchStatus {
    while (this.status === 'running' && this.expanded < limit) this.step(256);
    return this.status;
  }

  path(): number[] {
    if (this.status !== 'found') return [];
    const out: number[] = [];
    for (let c = this.goal; c !== -1; c = this.parent[c]) out.push(c);
    return out.reverse();
  }

  snapshot(): SearchSnapshot {
    const path = this.path();
    return {
      status: this.status,
      g: this.g,
      closed: this.closed,
      open: this.open,
      expanded: this.expanded,
      path,
      cost: this.status === 'found' ? this.g[this.goal] : Infinity,
    };
  }
}

/** `a_star(G, s, g, h)` — the whole search, for callers that want the answer. */
export function aStarGrid(
  grid: Lattice,
  start: number,
  goal: number,
  opts: SearchOptions = {},
): SearchSnapshot {
  const s = new GridSearch(grid, start, goal, opts);
  s.run();
  return s.snapshot();
}

/** `dijkstra(G, s)` — A* with the heuristic switched off. */
export function dijkstraGrid(grid: Lattice, start: number, goal: number): SearchSnapshot {
  return aStarGrid(grid, start, goal, { epsilon: 0 });
}

/**
 * `wavefront(grid, goal)` — cost-to-go for *every* free cell, by running
 * Dijkstra backwards from the goal with no early exit.
 *
 * This is Choset's wave-front planner, and it is also the reader's first value
 * function: descend it greedily from anywhere and you reach the goal, because
 * by construction every cell has a neighbour strictly closer to the goal.
 * Chapter 21 replaces "distance" with "expected reward" and calls the same
 * object V(x).
 */
export function wavefront(grid: Lattice, goal: number): Float64Array {
  const n = grid.nx * grid.ny;
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new MinHeap();
  dist[goal] = 0;
  heap.push(0, goal);

  for (;;) {
    const cur = heap.pop();
    if (cur === undefined) break;
    const dcur = dist[cur];
    const i = cur % grid.nx;
    const j = Math.floor(cur / grid.nx);
    for (const [di, dj, w] of NEIGHBORS) {
      const ni = i + di;
      const nj = j + dj;
      if (ni < 0 || nj < 0 || ni >= grid.nx || nj >= grid.ny) continue;
      const nb = nj * grid.nx + ni;
      if (grid.free[nb] === 0) continue;
      if (di !== 0 && dj !== 0) {
        if (grid.free[j * grid.nx + ni] === 0 || grid.free[nj * grid.nx + i] === 0) continue;
      }
      const cand = dcur + w * grid.cellSize;
      if (cand < dist[nb] - 1e-12) {
        dist[nb] = cand;
        heap.push(cand, nb);
      }
    }
  }
  return dist;
}

/**
 * `brushfire(grid)` — distance from every free cell to the nearest obstacle,
 * by the same Dijkstra with *all* obstacle cells as sources.
 *
 * Choset presents brushfire as a wave washing out from the obstacles; the
 * chapter's reveal is that it computes (an 8-connected approximation of) the
 * Euclidean distance transform of Chapter 19 — see
 * `lib/mapping/edt.ts` for the exact O(N) version, and
 * `__checks_ch20__.ts` for the invariant that pins the two together.
 */
export function brushfire(grid: Lattice): Float64Array {
  const n = grid.nx * grid.ny;
  const dist = new Float64Array(n).fill(Infinity);
  const heap = new MinHeap();
  for (let k = 0; k < n; k++) {
    if (grid.free[k] === 0) {
      dist[k] = 0;
      heap.push(0, k);
    }
  }
  for (;;) {
    const cur = heap.pop();
    if (cur === undefined) break;
    const dcur = dist[cur];
    const i = cur % grid.nx;
    const j = Math.floor(cur / grid.nx);
    for (const [di, dj, w] of NEIGHBORS) {
      const ni = i + di;
      const nj = j + dj;
      if (ni < 0 || nj < 0 || ni >= grid.nx || nj >= grid.ny) continue;
      const nb = nj * grid.nx + ni;
      const cand = dcur + w * grid.cellSize;
      if (cand < dist[nb] - 1e-12) {
        dist[nb] = cand;
        heap.push(cand, nb);
      }
    }
  }
  return dist;
}

/** Turn a cell path into a world-space polyline. */
export function pathPoints(grid: Lattice, path: number[]): { x: number; y: number }[] {
  return path.map((c) => cellCenter(grid, c));
}

/** Euclidean length of a polyline — the cost the arena scores every planner on. */
export function polylineLength(pts: { x: number; y: number }[]): number {
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    total += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  }
  return total;
}
