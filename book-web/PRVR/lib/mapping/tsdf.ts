/**
 * Truncated signed distance fields in 2-D — Curless & Levoy (1996), read as a
 * per-voxel recursive least-squares filter.
 *
 * Two scalars per cell: the fused signed distance `D` and the accumulated
 * weight `W`. Every scan supplies a *projective* distance observation `d` with
 * weight `w`, and the fusion rule
 *
 *     D ← (W·D + w·d) / (W + w),      W ← min(W + w, W_max)
 *
 * is the scalar information filter for a static state: `W` is accumulated
 * precision (Chapter 6's Ω), `w/(W+w)` is the Kalman gain, and the clamp on `W`
 * turns the estimator into a fading-memory filter. Nothing here is graphics
 * folklore; it is Chapter 5's recursion with the integral done in closed form.
 *
 * The truncation `τ` is the honest part: a projective distance is only a good
 * estimate of the true signed distance *near* the surface, so the model is only
 * fitted where it holds.
 */

import type { Pose2 } from '../geom/se2';
import type { Segment } from '../sim/world';

export interface Tsdf2Options {
  /** Grid size in cells. */
  width: number;
  height: number;
  cellSize: number;
  /** World coordinates of the grid's lower-left corner. */
  origin: { x: number; y: number };
  /** Truncation distance τ, in metres. */
  truncation: number;
  /** Weight clamp W_max. Large = stable and stubborn; small = responsive and noisy. */
  wMax: number;
}

export interface IntegrateOptions {
  /** Beams reporting at least this range saw nothing; they carve but never write a surface. */
  maxRange: number;
  /**
   * Voxblox's inverse-square weighting: a distant return is a noisier estimate
   * of the surface, so it should count for less. Off = every beam weighs 1.
   */
  inverseSquareWeight?: boolean;
  /**
   * Write `+τ` into the free space in front of the truncation band. Without it
   * the TSDF is undefined more than τ from a surface — which is exactly why an
   * ESDF is a different object, not a rescaled TSDF.
   */
  carveFreeSpace?: boolean;
  /** Weight given to a carving update, relative to a surface update. */
  carveWeight?: number;
}

export const DEFAULT_INTEGRATE: IntegrateOptions = {
  maxRange: 8,
  inverseSquareWeight: false,
  carveFreeSpace: true,
  carveWeight: 0.25,
};

/** One fusion event, kept so the voxel inspector can replay the filter. */
export interface FusionEvent {
  /** Prior estimate D_{t−1} and its weight W_{t−1}. */
  before: number;
  weightBefore: number;
  /** The scan's projective observation d_t and its weight w_t. */
  observation: number;
  observationWeight: number;
  /** Posterior D_t, W_t. */
  after: number;
  weightAfter: number;
  /** K_t = w_t / (W_{t−1} + w_t) — the Kalman gain of Chapter 6, verbatim. */
  gain: number;
}

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

export class Tsdf2 {
  readonly width: number;
  readonly height: number;
  readonly cellSize: number;
  readonly origin: { x: number; y: number };
  readonly truncation: number;
  /** Mutable on purpose: W_max is the one slider the TSDF widget foregrounds. */
  wMax: number;

  /** D_t(v) — fused signed distance, metres. Unobserved cells hold +τ. */
  readonly d: Float32Array;
  /** W_t(v) — accumulated weight. Zero means "never seen". */
  readonly w: Float32Array;

  /** Set to a cell index to record every fusion touching it. */
  probe: number | null = null;
  probeLog: FusionEvent[] = [];

  constructor(opts: Tsdf2Options) {
    this.width = opts.width;
    this.height = opts.height;
    this.cellSize = opts.cellSize;
    this.origin = { ...opts.origin };
    this.truncation = opts.truncation;
    this.wMax = opts.wMax;
    this.d = new Float32Array(this.width * this.height).fill(opts.truncation);
    this.w = new Float32Array(this.width * this.height);
  }

  static forBounds(
    bounds: { minX: number; minY: number; maxX: number; maxY: number },
    cellSize: number,
    truncation: number,
    wMax: number,
  ): Tsdf2 {
    return new Tsdf2({
      width: Math.max(1, Math.ceil((bounds.maxX - bounds.minX) / cellSize)),
      height: Math.max(1, Math.ceil((bounds.maxY - bounds.minY) / cellSize)),
      cellSize,
      origin: { x: bounds.minX, y: bounds.minY },
      truncation,
      wMax,
    });
  }

  index(i: number, j: number): number {
    return j * this.width + i;
  }

  inBounds(i: number, j: number): boolean {
    return i >= 0 && j >= 0 && i < this.width && j < this.height;
  }

  worldToCell(x: number, y: number): [number, number] {
    return [
      Math.floor((x - this.origin.x) / this.cellSize),
      Math.floor((y - this.origin.y) / this.cellSize),
    ];
  }

  cellCenter(i: number, j: number): [number, number] {
    return [
      this.origin.x + (i + 0.5) * this.cellSize,
      this.origin.y + (j + 0.5) * this.cellSize,
    ];
  }

  /** (D, W) of one cell. Out of bounds reads as unobserved. */
  voxel(i: number, j: number): [number, number] {
    if (!this.inBounds(i, j)) return [this.truncation, 0];
    const k = this.index(i, j);
    return [this.d[k], this.w[k]];
  }

  /** Bilinearly interpolated D at a world point — the surface is a level set of *this*. */
  value(x: number, y: number): number {
    const fx = (x - this.origin.x) / this.cellSize - 0.5;
    const fy = (y - this.origin.y) / this.cellSize - 0.5;
    const i = Math.floor(fx);
    const j = Math.floor(fy);
    const tx = fx - i;
    const ty = fy - j;
    const at = (a: number, b: number) => this.voxel(a, b)[0];
    const v00 = at(i, j);
    const v10 = at(i + 1, j);
    const v01 = at(i, j + 1);
    const v11 = at(i + 1, j + 1);
    return (
      v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty
    );
  }

  /**
   * The whole chapter in six lines: precision-weighted averaging with a clamp.
   *
   * Returns the fusion event so a widget can show the prior, the incoming
   * sample, and the posterior as the three-term update it actually is.
   */
  fuse(k: number, dObs: number, wObs: number): FusionEvent {
    const before = this.d[k];
    const weightBefore = this.w[k];
    const denom = weightBefore + wObs;
    const gain = denom > 0 ? wObs / denom : 1;
    // D_t = D_{t−1} + K_t (d_t − D_{t−1}): identical in form to the 1-D Kalman
    // correction, and identical in content — a static state observed in noise.
    const after = before + gain * (dObs - before);
    this.d[k] = after;
    this.w[k] = Math.min(denom, this.wMax);
    const event: FusionEvent = {
      before,
      weightBefore,
      observation: dObs,
      observationWeight: wObs,
      after,
      weightAfter: this.w[k],
      gain,
    };
    if (this.probe === k) {
      this.probeLog.push(event);
      if (this.probeLog.length > 64) this.probeLog.shift();
    }
    return event;
  }

  /**
   * `tsdf_integrate` — fold one range scan into the field.
   *
   * Each beam is walked from the sensor to `z + τ` at half-cell steps. The
   * projective observation at arc length `s` is `d = z − s`, truncated to
   * ±τ. Weight is constant in front of the surface and tapers linearly to zero
   * behind it (Voxblox's drop-off), because a beam says progressively less
   * about what lies past the thing it hit.
   *
   * Cost is O(beams · τ/r) once carving is off — only the truncation band is
   * touched, which is what makes TSDF fusion cheaper than it looks.
   */
  integrateScan(
    pose: Pose2,
    ranges: number[],
    angles: number[],
    opts: IntegrateOptions = DEFAULT_INTEGRATE,
  ): void {
    const { maxRange } = opts;
    const carve = opts.carveFreeSpace ?? true;
    const carveWeight = opts.carveWeight ?? 0.25;
    const tau = this.truncation;
    const step = this.cellSize * 0.5;
    // One visited set per *scan*, not per beam: cells near the sensor sit in
    // many beams' paths, and counting them once per beam would fabricate
    // evidence the scan does not contain.
    const visited = new Set<number>();

    for (let b = 0; b < ranges.length; b++) {
      const z = ranges[b];
      const a = pose.theta + angles[b];
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      const hit = z < maxRange;
      const wBeam = opts.inverseSquareWeight ? 1 / Math.max(z * z, 0.25) : 1;

      const sEnd = hit ? Math.min(z + tau, maxRange + tau) : maxRange;
      const sStart = carve ? step * 0.5 : Math.max(step * 0.5, z - tau);

      for (let s = sStart; s <= sEnd; s += step) {
        const [i, j] = this.worldToCell(pose.x + s * ca, pose.y + s * sa);
        if (!this.inBounds(i, j)) continue;
        const k = this.index(i, j);
        if (visited.has(k)) continue;
        visited.add(k);

        const sdf = hit ? z - s : tau; // a max-range beam only ever reports free
        if (sdf < -tau) continue; // past the truncation band: no information

        let wObs: number;
        if (sdf >= tau) {
          if (!carve) continue;
          wObs = wBeam * carveWeight; // far free space: real, but weak, evidence
        } else if (sdf >= 0) {
          wObs = wBeam;
        } else {
          wObs = wBeam * (1 + sdf / tau); // linear taper behind the surface
        }
        if (wObs <= 0) continue;

        this.fuse(k, clamp(sdf, -tau, tau), wObs);
      }
    }
  }

  /** Cells with any evidence at all — the denominator of "how much did we see?". */
  observedCells(): number {
    let n = 0;
    for (let k = 0; k < this.w.length; k++) if (this.w[k] > 0) n++;
    return n;
  }

  /** Two f32 planes. The honest cost of storing a distance instead of a bit. */
  memoryBytes(): number {
    return this.d.byteLength + this.w.byteLength;
  }

  /**
   * `marching_squares` — the zero level set, as line segments.
   *
   * Each cell of the *dual* grid has the four neighbouring sample points at its
   * corners; the sign pattern of those four samples selects a case, and the
   * crossing point on each edge comes from linear interpolation, which is exact
   * for the piecewise-linear field the samples define.
   */
  surface(minWeight = 0.5): Segment[] {
    const out: Segment[] = [];
    for (let j = 0; j + 1 < this.height; j++) {
      for (let i = 0; i + 1 < this.width; i++) {
        const c: number[] = [
          this.d[this.index(i, j)],
          this.d[this.index(i + 1, j)],
          this.d[this.index(i + 1, j + 1)],
          this.d[this.index(i, j + 1)],
        ];
        const wts = [
          this.w[this.index(i, j)],
          this.w[this.index(i + 1, j)],
          this.w[this.index(i + 1, j + 1)],
          this.w[this.index(i, j + 1)],
        ];
        // An unobserved corner has no opinion; extracting a surface from it
        // would be inventing geometry.
        if (wts[0] < minWeight || wts[1] < minWeight || wts[2] < minWeight || wts[3] < minWeight) {
          continue;
        }

        let code = 0;
        for (let n = 0; n < 4; n++) if (c[n] < 0) code |= 1 << n;
        const edges = MARCHING_SQUARES[code];
        if (edges.length === 0) continue;

        const [cx, cy] = this.cellCenter(i, j);
        const h = this.cellSize;
        const corners: [number, number][] = [
          [cx, cy],
          [cx + h, cy],
          [cx + h, cy + h],
          [cx, cy + h],
        ];
        const point = (e: number): [number, number] => {
          const a = e;
          const b = (e + 1) % 4;
          const da = c[a];
          const db = c[b];
          const t = Math.abs(da - db) < 1e-9 ? 0.5 : da / (da - db);
          return [
            corners[a][0] + t * (corners[b][0] - corners[a][0]),
            corners[a][1] + t * (corners[b][1] - corners[a][1]),
          ];
        };

        for (let e = 0; e + 1 < edges.length; e += 2) {
          const [x1, y1] = point(edges[e]);
          const [x2, y2] = point(edges[e + 1]);
          out.push({ x1, y1, x2, y2 });
        }
      }
    }
    return out;
  }
}

/**
 * The 16-case table. Corner order is counter-clockwise from the lower left, and
 * edge `e` joins corner `e` to corner `(e+1) mod 4`. Cases 5 and 10 are the
 * saddle ambiguities: two sign changes are consistent with two different
 * topologies, and this table picks the "separated" one. In 3-D the same
 * ambiguity is what puts holes in a naive marching-cubes mesh.
 */
const MARCHING_SQUARES: number[][] = [
  [], // 0000
  [3, 0], // 0001
  [0, 1], // 0010
  [3, 1], // 0011
  [1, 2], // 0100
  [3, 0, 1, 2], // 0101 — ambiguous
  [0, 2], // 0110
  [3, 2], // 0111
  [2, 3], // 1000
  [2, 0], // 1001
  [0, 1, 2, 3], // 1010 — ambiguous
  [2, 1], // 1011
  [1, 3], // 1100
  [1, 0], // 1101
  [0, 3], // 1110
  [], // 1111
];
