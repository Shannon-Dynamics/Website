/**
 * The exact Euclidean distance transform, in O(N).
 *
 * Felzenszwalb & Huttenlocher (2012), *Distance Transforms of Sampled
 * Functions*. The likelihood field of Chapter 10 needs, for every cell of the
 * map, the distance to the nearest occupied cell. Chamfer sweeps approximate
 * that with propagating integer costs and are off by a few percent; this
 * transform is *exact* on the sampled grid, and it is not slower — one pass per
 * axis, linear in the number of cells, no square roots until the very end.
 *
 * The trick is that the squared distance transform is separable:
 *
 *   D(x, y) = min_{x'} [ (x − x')² + min_{y'} ( (y − y')² + f(x', y') ) ]
 *
 * so a 2-D transform is a 1-D transform down the columns followed by a 1-D
 * transform along the rows. And the 1-D transform is the lower envelope of the
 * parabolas z ↦ (z − q)² + f(q), one per sample, which can be built in a single
 * left-to-right pass because the parabolas all have the same curvature.
 */

import type { Bounds, World } from '../sim/world';

/** Stands in for +∞: a real number, so the envelope arithmetic stays finite. */
export const EDT_INF = 1e20;

/**
 * `dt(f)` of Felzenszwalb & Huttenlocher, Figure 1: the lower envelope of the
 * parabolas rooted at each sample.
 *
 * `v[k]` holds the sample index of the k-th parabola on the envelope and
 * `z[k]` the coordinate where it takes over from its predecessor. Every sample
 * is pushed once and popped at most once, which is the whole O(n) argument.
 */
export function edt1d(f: Float64Array, out: Float64Array = new Float64Array(f.length)): Float64Array {
  const n = f.length;
  if (n === 0) return out;
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);

  let k = 0;
  v[0] = 0;
  z[0] = -EDT_INF;
  z[1] = EDT_INF;

  for (let q = 1; q < n; q++) {
    // Intersection of the parabola at q with the one currently on top.
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (k > 0 && s <= z[k]) {
      // The top parabola is entirely under the new one: pop it.
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = EDT_INF;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const d = q - v[k];
    out[q] = d * d + f[v[k]];
  }
  return out;
}

/**
 * Exact squared distance, in **cell units**, from every cell to the nearest
 * seed. `seed[j*nx + i]` is 0 at an occupied cell and {@link EDT_INF}
 * elsewhere; anything in between works too, and behaves as a soft seed.
 */
export function edt2dSquared(seed: Float64Array, nx: number, ny: number): Float64Array {
  const out = Float64Array.from(seed);

  // Columns.
  const col = new Float64Array(ny);
  const colOut = new Float64Array(ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) col[j] = out[j * nx + i];
    edt1d(col, colOut);
    for (let j = 0; j < ny; j++) out[j * nx + i] = colOut[j];
  }

  // Rows.
  const row = new Float64Array(nx);
  const rowOut = new Float64Array(nx);
  for (let j = 0; j < ny; j++) {
    const base = j * nx;
    for (let i = 0; i < nx; i++) row[i] = out[base + i];
    edt1d(row, rowOut);
    for (let i = 0; i < nx; i++) out[base + i] = rowOut[i];
  }

  return out;
}

export interface ExactDistanceField {
  nx: number;
  ny: number;
  cellSize: number;
  bounds: Bounds;
  /** Row-major (j·nx + i) distance to the nearest occupied cell, in metres. */
  data: Float64Array;
}

/**
 * Rasterize a polyline world into the seed grid the transform consumes: 0 at
 * every cell a wall passes through, {@link EDT_INF} everywhere else.
 *
 * Walls are walked at a third of a cell so no crossing is missed, and indices
 * are clamped rather than dropped — the Apartment's exterior shell lies exactly
 * on the bounding box, and a wall that falls off the grid would leave the whole
 * border looking like open space.
 *
 * This is deliberately the *literal* reading of Table 6.3's line 7: "the
 * distance to the nearest ⟨x', y'⟩ occupied in m". The map is a set of occupied
 * cells; sub-cell geometry is gone before the transform starts, and what that
 * costs is measured in the chapter.
 */
export function rasterizeWalls(
  world: World,
  cellSize: number,
  bounds: Bounds,
  nx: number,
  ny: number,
): Float64Array {
  const seed = new Float64Array(nx * ny).fill(EDT_INF);
  const mark = (px: number, py: number) => {
    const i = Math.min(nx - 1, Math.max(0, Math.floor((px - bounds.minX) / cellSize)));
    const j = Math.min(ny - 1, Math.max(0, Math.floor((py - bounds.minY) / cellSize)));
    seed[j * nx + i] = 0;
  };
  for (const s of world.walls) {
    const len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
    const steps = Math.max(1, Math.ceil((len / cellSize) * 3));
    for (let n = 0; n <= steps; n++) {
      const t = n / steps;
      mark(s.x1 + t * (s.x2 - s.x1), s.y1 + t * (s.y2 - s.y1));
    }
  }
  return seed;
}

/**
 * The likelihood field's $d(x, y)$, built exactly: rasterize the map, then take
 * one distance transform. Every cell in one pass; no per-beam nearest-neighbour
 * search ever again.
 */
export function exactDistanceField(
  world: World,
  cellSize = 0.05,
  bounds: Bounds = world.bounds,
): ExactDistanceField {
  const nx = Math.max(1, Math.ceil((bounds.maxX - bounds.minX) / cellSize));
  const ny = Math.max(1, Math.ceil((bounds.maxY - bounds.minY) / cellSize));
  const seed = rasterizeWalls(world, cellSize, bounds, nx, ny);

  const sq = edt2dSquared(seed, nx, ny);
  const data = new Float64Array(sq.length);
  // One square root per cell, at the end — the whole point of carrying squares.
  for (let k = 0; k < sq.length; k++) data[k] = Math.sqrt(sq[k]) * cellSize;

  return { nx, ny, cellSize, bounds, data };
}

/**
 * Brute-force lattice transform: the definition, O(N²), for tests only.
 * `edt2dSquared` must agree with this exactly, not approximately.
 */
export function bruteForceEdtSquared(seed: Float64Array, nx: number, ny: number): Float64Array {
  const out = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      let best = EDT_INF;
      for (let q = 0; q < ny; q++) {
        for (let p = 0; p < nx; p++) {
          const f = seed[q * nx + p];
          if (f >= EDT_INF) continue;
          const d = (i - p) * (i - p) + (j - q) * (j - q) + f;
          if (d < best) best = d;
        }
      }
      out[j * nx + i] = best;
    }
  }
  return out;
}

/** Nearest-cell lookup into an {@link ExactDistanceField}, clamped at the border. */
export function distanceAt(field: ExactDistanceField, x: number, y: number): number {
  const i = Math.min(
    field.nx - 1,
    Math.max(0, Math.round((x - field.bounds.minX) / field.cellSize - 0.5)),
  );
  const j = Math.min(
    field.ny - 1,
    Math.max(0, Math.round((y - field.bounds.minY) / field.cellSize - 0.5)),
  );
  return field.data[j * field.nx + i];
}
