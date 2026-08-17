/**
 * Euclidean signed distance fields, built from a TSDF in O(n).
 *
 * A TSDF answers "how far to the surface?" only inside the truncation band and
 * only along the sensor's line of sight. A planner needs the *true* Euclidean
 * distance everywhere, plus its gradient, and it needs it in O(1) per query.
 * The bridge is the exact distance transform of Felzenszwalb & Huttenlocher
 * (2012): a lower envelope of parabolas swept once per row and once per column.
 *
 * The trick that makes the result sub-cell accurate is to seed the transform
 * with the TSDF's own value rather than with a binary mask: a cell holding
 * D = 0.017 m is 0.017 m from the surface, not 0.
 */

import type { Tsdf2 } from './tsdf';

/** Stand-in for +∞ that survives the subtraction in the envelope arithmetic. */
const BIG = 1e12;

/**
 * `dt_1d` — the lower envelope of the parabolas y = (q − v)² + f(v).
 *
 * Each sample `f(q)` defines an upward parabola rooted at `q`; the distance
 * transform is their pointwise minimum. Scanning left to right, the stack `v`
 * holds the parabolas currently on the envelope and `z` the abscissae where
 * successive ones cross. Each index is pushed and popped at most once, so the
 * pass is Θ(n) — not O(n log n), and not the O(n²) the definition suggests.
 */
export function distanceTransform1D(f: Float64Array, n: number, out: Float64Array): void {
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  let k = 0;
  v[0] = 0;
  z[0] = -BIG;
  z[1] = BIG;

  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = BIG;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dq = q - v[k];
    out[q] = dq * dq + f[v[k]];
  }
}

/**
 * The separable 2-D transform: rows, then columns. Separability is why the
 * total cost is Θ(width·height) — a genuinely linear algorithm for a quantity
 * whose definition is a minimum over every pair of cells.
 *
 * `f` is in units of cells², and so is the result.
 */
export function squaredDistanceTransform(
  f: Float64Array,
  width: number,
  height: number,
): Float64Array {
  const out = new Float64Array(f);
  const col = new Float64Array(Math.max(width, height));
  const res = new Float64Array(Math.max(width, height));

  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) col[i] = out[j * width + i];
    distanceTransform1D(col, width, res);
    for (let i = 0; i < width; i++) out[j * width + i] = res[i];
  }
  for (let i = 0; i < width; i++) {
    for (let j = 0; j < height; j++) col[j] = out[j * width + i];
    distanceTransform1D(col, height, res);
    for (let j = 0; j < height; j++) out[j * width + i] = res[j];
  }
  return out;
}

export interface Esdf2Options {
  width: number;
  height: number;
  cellSize: number;
  origin: { x: number; y: number };
}

export class Esdf2 {
  readonly width: number;
  readonly height: number;
  readonly cellSize: number;
  readonly origin: { x: number; y: number };
  /** Signed distance to the nearest extracted surface, metres. */
  readonly d: Float32Array;

  constructor(opts: Esdf2Options) {
    this.width = opts.width;
    this.height = opts.height;
    this.cellSize = opts.cellSize;
    this.origin = { ...opts.origin };
    this.d = new Float32Array(this.width * this.height);
  }

  index(i: number, j: number): number {
    return j * this.width + i;
  }

  at(i: number, j: number): number {
    const ci = i < 0 ? 0 : i >= this.width ? this.width - 1 : i;
    const cj = j < 0 ? 0 : j >= this.height ? this.height - 1 : j;
    return this.d[this.index(ci, cj)];
  }

  /** `esdf_query` — O(1) bilinear lookup. This is the Chapter 20/23 contract. */
  distance(x: number, y: number): number {
    const fx = (x - this.origin.x) / this.cellSize - 0.5;
    const fy = (y - this.origin.y) / this.cellSize - 0.5;
    const i = Math.floor(fx);
    const j = Math.floor(fy);
    const tx = fx - i;
    const ty = fy - j;
    return (
      this.at(i, j) * (1 - tx) * (1 - ty) +
      this.at(i + 1, j) * tx * (1 - ty) +
      this.at(i, j + 1) * (1 - tx) * ty +
      this.at(i + 1, j + 1) * tx * ty
    );
  }

  /**
   * `esdf_gradient` — central differences on the interpolated field.
   *
   * Away from the medial axis ‖∇d‖ = 1 (the eikonal property), so the gradient
   * is a pure direction: "this way is away from the nearest obstacle". That
   * vector is what MPPI's obstacle cost differentiates in Chapter 23.
   */
  gradient(x: number, y: number): [number, number] {
    const h = this.cellSize;
    return [
      (this.distance(x + h, y) - this.distance(x - h, y)) / (2 * h),
      (this.distance(x, y + h) - this.distance(x, y - h)) / (2 * h),
    ];
  }

  memoryBytes(): number {
    return this.d.byteLength;
  }
}

export interface EsdfFromTsdfOptions {
  /** Only cells with at least this much weight may seed the transform. */
  minWeight?: number;
  /**
   * Seed band, as a multiple of the cell size. Cells whose |D| exceeds this are
   * not trusted to localize the surface sub-cell, so they only inherit distance.
   */
  bandCells?: number;
}

/**
 * `esdf_from_tsdf` — two linear passes and one square root.
 *
 * Seeds are the cells the TSDF places near its own zero crossing, carrying
 * f = (D/r)² so that the transform starts from the sub-cell offset rather than
 * from the cell centre. Everything else starts at +∞ and inherits the distance
 * to the nearest seed.
 *
 * Sign is taken from the TSDF where it has an opinion: a cell fused to a
 * negative distance sits behind the surface. Outside the observed band the
 * field is unsigned — which is the correct statement, since an unobserved cell
 * has no inside or outside.
 */
export function esdfFromTsdf(tsdf: Tsdf2, opts: EsdfFromTsdfOptions = {}): Esdf2 {
  const minWeight = opts.minWeight ?? 0.5;
  const bandCells = opts.bandCells ?? 1.5;
  const { width, height, cellSize } = tsdf;
  const band = bandCells * cellSize;

  const f = new Float64Array(width * height).fill(BIG);
  for (let k = 0; k < f.length; k++) {
    if (tsdf.w[k] < minWeight) continue;
    const dv = tsdf.d[k];
    if (Math.abs(dv) > band) continue;
    const sub = dv / cellSize;
    f[k] = sub * sub;
  }

  const sq = squaredDistanceTransform(f, width, height);
  const esdf = new Esdf2({ width, height, cellSize, origin: tsdf.origin });
  // Nothing in the grid can be further from a seed than the diagonal; capping
  // there keeps a seedless field (no scan yet) finite and drawable.
  const diag = Math.hypot(width, height) * cellSize;
  for (let k = 0; k < sq.length; k++) {
    const dist = Math.min(Math.sqrt(sq[k]) * cellSize, diag);
    const inside = tsdf.w[k] >= minWeight && tsdf.d[k] < 0;
    esdf.d[k] = inside ? -dist : dist;
  }
  return esdf;
}

/** The largest finite distance in the field — handy for scaling a heatmap. */
export function maxDistance(esdf: Esdf2): number {
  let m = 0;
  for (let k = 0; k < esdf.d.length; k++) {
    const v = Math.abs(esdf.d[k]);
    if (Number.isFinite(v) && v > m) m = v;
  }
  return m;
}
