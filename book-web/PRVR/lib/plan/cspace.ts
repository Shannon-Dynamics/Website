/**
 * Configuration space for a disc robot in a polyline world.
 *
 * The whole of Chapter 20 rests on one reduction: a disc of radius `r` moving
 * among obstacles is a *point* moving among those obstacles inflated by `r`
 * (the Minkowski sum WO ⊕ B_r). So `CSpace2` never asks "does the footprint
 * overlap a wall?" — it asks "is the distance from this point to the nearest
 * wall greater than r?", which is one lookup into the exact Euclidean distance
 * transform Chapter 19 already built.
 *
 * That choice buys three things at once:
 *   1. `isFree` is O(1) instead of O(#walls);
 *   2. `clearance` is available everywhere, which the potential fields and the
 *      clearance-weighted costs of §"Putting it together" need anyway;
 *   3. `edgeFree` can *sphere-march* — take a step as long as the current
 *      clearance allows — so a long edge through open space costs a handful of
 *      lookups rather than hundreds of samples, and the check is still
 *      conservative because the distance field is 1-Lipschitz.
 */

import { distanceAt, exactDistanceField, type ExactDistanceField } from '../mapping/edt';
import type { Bounds, Point2, World } from '../sim/world';

export interface CSpaceOptions {
  /** Robot radius, in metres. The inflation applied to every obstacle. */
  radius?: number;
  /** Resolution of the distance field backing the free-space test. */
  cellSize?: number;
  /** Restrict planning to a sub-rectangle of the world. */
  bounds?: Bounds;
}

export class CSpace2 {
  readonly world: World;
  readonly radius: number;
  readonly bounds: Bounds;
  readonly field: ExactDistanceField;

  constructor(world: World, opts: CSpaceOptions = {}) {
    const { radius = 0.22, cellSize = 0.05 } = opts;
    this.world = world;
    this.radius = radius;
    this.bounds = opts.bounds ?? world.bounds;
    this.field = exactDistanceField(world, cellSize, this.bounds);
  }

  /** Distance from (x, y) to the nearest obstacle, in metres. */
  clearance(x: number, y: number): number {
    return distanceAt(this.field, x, y);
  }

  /** Inside the map, and outside every C-obstacle. */
  isFree(x: number, y: number): boolean {
    const b = this.bounds;
    if (x < b.minX || x > b.maxX || y < b.minY || y > b.maxY) return false;
    return this.clearance(x, y) > this.radius;
  }

  /**
   * Is the straight segment a→b collision-free?
   *
   * Sphere marching: standing at p with clearance c, every point within c − r
   * of p is free, so we may advance by exactly that much and re-test. The
   * `minStep` floor keeps the loop terminating when the path grazes an
   * obstacle, and `maxIter` caps the worst case (a path that hugs a wall for
   * its whole length).
   */
  edgeFree(a: Point2, b: Point2, minStep = 0.02): boolean {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (!this.isFree(a.x, a.y) || !this.isFree(b.x, b.y)) return false;
    if (len < 1e-9) return true;

    const ux = dx / len;
    const uy = dy / len;
    let s = 0;
    let iter = 0;
    while (s < len && iter++ < 4000) {
      const x = a.x + ux * s;
      const y = a.y + uy * s;
      const c = this.clearance(x, y) - this.radius;
      if (c <= 0) return false;
      s += Math.max(c, minStep);
    }
    return true;
  }

  /** Rejection-sample a free configuration. Returns null if the space looks full. */
  sampleFree(rng: { uniform: (lo: number, hi: number) => number }, tries = 100): Point2 | null {
    const b = this.bounds;
    for (let k = 0; k < tries; k++) {
      const x = rng.uniform(b.minX, b.maxX);
      const y = rng.uniform(b.minY, b.maxY);
      if (this.isFree(x, y)) return { x, y };
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// The planning lattice
// ---------------------------------------------------------------------------

/**
 * An 8-connected grid over C-space: the substrate every search algorithm in
 * this chapter runs on, and the discrete stand-in for `Q_free`.
 *
 * A cell is free iff its centre is free in the *inflated* map, so the lattice
 * inherits the disc reduction rather than repeating it. This is also why grid
 * planners are only *resolution* complete: a passage narrower than one cell can
 * fall between the centres and vanish.
 */
export interface Lattice {
  nx: number;
  ny: number;
  cellSize: number;
  bounds: Bounds;
  /** 1 = free, 0 = C-obstacle. Row-major, index j·nx + i. */
  free: Uint8Array;
}

export function latticeFromCSpace(cs: CSpace2, cellSize = 0.2): Lattice {
  const b = cs.bounds;
  const nx = Math.max(1, Math.floor((b.maxX - b.minX) / cellSize));
  const ny = Math.max(1, Math.floor((b.maxY - b.minY) / cellSize));
  const free = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = b.minX + (i + 0.5) * cellSize;
      const y = b.minY + (j + 0.5) * cellSize;
      free[j * nx + i] = cs.isFree(x, y) ? 1 : 0;
    }
  }
  return { nx, ny, cellSize, bounds: b, free };
}

export const cellIndex = (g: Lattice, i: number, j: number) => j * g.nx + i;

export function cellCenter(g: Lattice, index: number): Point2 {
  const i = index % g.nx;
  const j = Math.floor(index / g.nx);
  return {
    x: g.bounds.minX + (i + 0.5) * g.cellSize,
    y: g.bounds.minY + (j + 0.5) * g.cellSize,
  };
}

/** Nearest cell to a world point, clamped into the lattice. */
export function cellAt(g: Lattice, x: number, y: number): number {
  const i = Math.min(g.nx - 1, Math.max(0, Math.floor((x - g.bounds.minX) / g.cellSize)));
  const j = Math.min(g.ny - 1, Math.max(0, Math.floor((y - g.bounds.minY) / g.cellSize)));
  return j * g.nx + i;
}

/** Nearest *free* cell to a world point — a start pose sometimes lands in a wall. */
export function freeCellNear(g: Lattice, x: number, y: number): number {
  const start = cellAt(g, x, y);
  if (g.free[start] === 1) return start;
  const si = start % g.nx;
  const sj = Math.floor(start / g.nx);
  for (let r = 1; r < Math.max(g.nx, g.ny); r++) {
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue;
        const i = si + di;
        const j = sj + dj;
        if (i < 0 || j < 0 || i >= g.nx || j >= g.ny) continue;
        const k = j * g.nx + i;
        if (g.free[k] === 1) return k;
      }
    }
  }
  return start;
}

/** The eight lattice moves, with their step costs (1 or √2 cell widths). */
export const NEIGHBORS: readonly (readonly [number, number, number])[] = [
  [1, 0, 1],
  [-1, 0, 1],
  [0, 1, 1],
  [0, -1, 1],
  [1, 1, Math.SQRT2],
  [1, -1, Math.SQRT2],
  [-1, 1, Math.SQRT2],
  [-1, -1, Math.SQRT2],
];
