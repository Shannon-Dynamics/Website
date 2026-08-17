/**
 * `Grid_localization` — Thrun et al., **Table 8.1**.
 *
 * The discrete Bayes filter of Chapter 5, run over a regular decomposition of
 * *pose* space rather than a corridor: (x, y, θ) cells, typically 15 cm × 15 cm
 * × 5° in the literature. Nothing about the mathematics is new. What is new is
 * the bookkeeping, and the bookkeeping is the lesson — a metric pose grid is
 * three-dimensional, so halving the cell size multiplies both memory and the
 * per-update cost by eight.
 *
 * Two departures from the pseudocode, both deliberate and both discussed in the
 * chapter:
 *
 *  1. **Bounded motion kernel.** Table 8.1 line 3 is a sum over *all* cells for
 *     *each* cell — O(|G|²), which for a 28 800-cell grid is 8.3 × 10⁸ kernel
 *     evaluations per step. The motion kernel has compact support in practice,
 *     so we scatter each cell's mass into the neighbourhood of its predicted
 *     pose instead: O(|G| · 27).
 *  2. **Coarsening correction.** Evaluating a point model at the cell centre
 *     (Thrun's `mean(x_k)`) treats a 40 cm cell as if it were a point. Both
 *     models therefore run with their noise inflated by the cell's own spread,
 *     added in quadrature. Without it a coarse grid is *overconfident*: it
 *     rejects the true cell because the centre of that cell does not explain
 *     the scan, even though some point inside it would.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import { applyOdom, type OdomDelta } from '../models/motion';
import { isFree, type Bounds, type World } from '../sim/world';

export interface GridSpec {
  /** Metric cell size in x and y, metres. */
  cellSize: number;
  /** Number of heading bins over the full circle. */
  nTheta: number;
  /** Cells whose centre is closer than this to a wall are not free. */
  clearance?: number;
  /** Restrict the grid to a sub-rectangle of the world. */
  bounds?: Bounds;
}

export interface GridMotionNoise {
  /** Std-dev of the translation error, metres. */
  sigmaTrans: number;
  /** Std-dev of the heading error, radians. */
  sigmaRot: number;
}

const EPS = 1e-12;

export class GridLocalizer {
  readonly cellSize: number;
  readonly nTheta: number;
  readonly dTheta: number;
  readonly bounds: Bounds;
  readonly nx: number;
  readonly ny: number;
  /** 1 where the cell centre is in free space, 0 where it is inside a wall. */
  readonly free: Uint8Array;
  readonly freeCells: number;
  /** p_{k,t}, indexed ((k · ny) + j) · nx + i. */
  belief: Float64Array;

  constructor(world: World, spec: GridSpec) {
    this.cellSize = spec.cellSize;
    this.nTheta = Math.max(1, Math.round(spec.nTheta));
    this.dTheta = (2 * Math.PI) / this.nTheta;
    this.bounds = spec.bounds ?? world.bounds;

    const { minX, minY, maxX, maxY } = this.bounds;
    this.nx = Math.max(1, Math.ceil((maxX - minX) / this.cellSize));
    this.ny = Math.max(1, Math.ceil((maxY - minY) / this.cellSize));

    this.free = new Uint8Array(this.nx * this.ny);
    let n = 0;
    const clearance = spec.clearance ?? 0.12;
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) {
        const [cx, cy] = this.cellCenterXY(i, j);
        if (isFree(world, cx, cy, clearance)) {
          this.free[j * this.nx + i] = 1;
          n++;
        }
      }
    }
    this.freeCells = n;
    this.belief = new Float64Array(this.nx * this.ny * this.nTheta);
    this.setUniform();
  }

  /** Total cells including headings — the number that grows as 1/resolution³. */
  get cellCount(): number {
    return this.nx * this.ny * this.nTheta;
  }

  /** Bytes of belief storage, at f64 per cell. */
  get bytes(): number {
    return this.belief.byteLength;
  }

  index(i: number, j: number, k: number): number {
    return (k * this.ny + j) * this.nx + i;
  }

  cellCenterXY(i: number, j: number): [number, number] {
    return [
      this.bounds.minX + (i + 0.5) * this.cellSize,
      this.bounds.minY + (j + 0.5) * this.cellSize,
    ];
  }

  cellCenter(i: number, j: number, k: number): Pose2 {
    const [x, y] = this.cellCenterXY(i, j);
    return { x, y, theta: normalizeAngle(-Math.PI + (k + 0.5) * this.dTheta) };
  }

  /** Global localization: uniform over every free cell and every heading. */
  setUniform(): void {
    const p = this.freeCells > 0 ? 1 / (this.freeCells * this.nTheta) : 0;
    for (let k = 0; k < this.nTheta; k++) {
      for (let j = 0; j < this.ny; j++) {
        for (let i = 0; i < this.nx; i++) {
          this.belief[this.index(i, j, k)] = this.free[j * this.nx + i] ? p : 0;
        }
      }
    }
  }

  /** Position tracking: all the mass on the cell containing `pose`. */
  setDelta(pose: Pose2): void {
    this.belief.fill(0);
    const i = Math.floor((pose.x - this.bounds.minX) / this.cellSize);
    const j = Math.floor((pose.y - this.bounds.minY) / this.cellSize);
    const k = Math.floor((normalizeAngle(pose.theta) + Math.PI) / this.dTheta);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return;
    this.belief[this.index(i, j, Math.min(k, this.nTheta - 1))] = 1;
  }

  /**
   * Line 3 of Table 8.1: p̄ = Σ_j p(x_k | u, x_j) p_j, as a scatter with a
   * bounded kernel.
   *
   * The kernel widths are the motion noise **plus the cell's own extent**, in
   * quadrature. That second term is what stops a coarse grid from freezing:
   * with 40 cm cells and 10 cm of motion per step, a naive centre-to-centre
   * transition never leaves the cell it started in, and the belief simply stops
   * moving while the robot drives away.
   */
  predict(u: OdomDelta, noise: GridMotionNoise): void {
    const { nx, ny, nTheta, cellSize, dTheta } = this;
    const next = new Float64Array(this.belief.length);

    const half = cellSize / 2;
    const sx = Math.sqrt(noise.sigmaTrans * noise.sigmaTrans + half * half);
    const st = Math.sqrt(noise.sigmaRot * noise.sigmaRot + (dTheta / 2) * (dTheta / 2));

    const rXY = Math.max(1, Math.min(2, Math.ceil((2 * sx) / cellSize)));
    const rT = Math.max(1, Math.min(2, Math.ceil((2 * st) / dTheta)));

    for (let k = 0; k < nTheta; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const mass = this.belief[this.index(i, j, k)];
          if (mass < EPS) continue; // selective updating, Thrun §8.2.3

          const pred = applyOdom(this.cellCenter(i, j, k), u);
          const fx = (pred.x - this.bounds.minX) / cellSize - 0.5;
          const fy = (pred.y - this.bounds.minY) / cellSize - 0.5;
          const ft = (normalizeAngle(pred.theta) + Math.PI) / dTheta - 0.5;

          const ci = Math.round(fx);
          const cj = Math.round(fy);
          const ck = Math.round(ft);

          let norm = 0;
          const wt: number[] = [];
          const idx: number[] = [];
          for (let dk = -rT; dk <= rT; dk++) {
            const kk = ((ck + dk) % nTheta + nTheta) % nTheta;
            const dth = (ck + dk - ft) * dTheta;
            for (let dj = -rXY; dj <= rXY; dj++) {
              const jj = cj + dj;
              if (jj < 0 || jj >= ny) continue;
              for (let di = -rXY; di <= rXY; di++) {
                const ii = ci + di;
                if (ii < 0 || ii >= nx) continue;
                if (!this.free[jj * nx + ii]) continue;
                const dx = (ci + di - fx) * cellSize;
                const dy = (cj + dj - fy) * cellSize;
                const w = Math.exp(
                  -0.5 * ((dx * dx + dy * dy) / (sx * sx) + (dth * dth) / (st * st)),
                );
                if (w < 1e-6) continue;
                wt.push(w);
                idx.push(this.index(ii, jj, kk));
                norm += w;
              }
            }
          }
          if (norm <= 0) continue; // predicted into a wall: that mass is gone
          const scale = mass / norm;
          for (let n = 0; n < idx.length; n++) next[idx[n]] += wt[n] * scale;
        }
      }
    }

    this.belief = next;
    this.normalize();
  }

  /**
   * Line 4 of Table 8.1, in log space: p ← η · exp(log L(x_k)) · p̄.
   *
   * The caller supplies the *log* likelihood so a 20-beam scan does not
   * underflow, and the maximum is subtracted before exponentiating, which is
   * exactly the η the line already allows for. Returns the log evidence
   * log p(z | z_{1:t−1}) — the same diagnostic the particle filter's average
   * weight provides.
   */
  correctLog(logLikelihood: (pose: Pose2) => number): number {
    const { nx, ny, nTheta } = this;
    const logs = new Float64Array(this.belief.length);
    let max = -Infinity;

    for (let k = 0; k < nTheta; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const idx = this.index(i, j, k);
          if (this.belief[idx] < EPS) continue;
          const l = logLikelihood(this.cellCenter(i, j, k));
          logs[idx] = l;
          if (l > max) max = l;
        }
      }
    }
    if (!Number.isFinite(max)) return -Infinity;

    let total = 0;
    for (let idx = 0; idx < this.belief.length; idx++) {
      if (this.belief[idx] < EPS) {
        this.belief[idx] = 0;
        continue;
      }
      this.belief[idx] *= Math.exp(logs[idx] - max);
      total += this.belief[idx];
    }
    if (total > 0) {
      for (let idx = 0; idx < this.belief.length; idx++) this.belief[idx] /= total;
    }
    return Math.log(Math.max(total, Number.MIN_VALUE)) + max;
  }

  private normalize(): void {
    let total = 0;
    for (let idx = 0; idx < this.belief.length; idx++) total += this.belief[idx];
    if (total > 0) {
      for (let idx = 0; idx < this.belief.length; idx++) this.belief[idx] /= total;
    } else {
      this.setUniform();
    }
  }

  /** Belief marginalized over heading — the picture the widget draws. */
  marginalXY(): Float64Array {
    const out = new Float64Array(this.nx * this.ny);
    for (let k = 0; k < this.nTheta; k++) {
      for (let j = 0; j < this.ny; j++) {
        for (let i = 0; i < this.nx; i++) {
          out[j * this.nx + i] += this.belief[this.index(i, j, k)];
        }
      }
    }
    return out;
  }

  /** The MAP cell, returned as a pose. Grid error is quantized, not stochastic. */
  argmax(): { pose: Pose2; p: number } {
    let best = -1;
    let bestP = -1;
    for (let idx = 0; idx < this.belief.length; idx++) {
      if (this.belief[idx] > bestP) {
        bestP = this.belief[idx];
        best = idx;
      }
    }
    if (best < 0) return { pose: { x: 0, y: 0, theta: 0 }, p: 0 };
    const i = best % this.nx;
    const j = Math.floor(best / this.nx) % this.ny;
    const k = Math.floor(best / (this.nx * this.ny));
    return { pose: this.cellCenter(i, j, k), p: bestP };
  }

  /** Mass-weighted (x, y) centroid of the marginal — reported, never trusted blindly. */
  meanXY(): { x: number; y: number } {
    const m = this.marginalXY();
    let wx = 0;
    let wy = 0;
    let total = 0;
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) {
        const w = m[j * this.nx + i];
        if (w <= 0) continue;
        const [cx, cy] = this.cellCenterXY(i, j);
        wx += w * cx;
        wy += w * cy;
        total += w;
      }
    }
    if (total <= 0) return { x: 0, y: 0 };
    return { x: wx / total, y: wy / total };
  }

  /** Entropy of the full pose belief, in bits. log₂(free · nθ) when uniform. */
  entropyBits(): number {
    let h = 0;
    for (let idx = 0; idx < this.belief.length; idx++) {
      const p = this.belief[idx];
      if (p > 0) h -= p * Math.log2(p);
    }
    return h;
  }
}

/**
 * The sensor-side coarsening correction: σ_eff = √(σ² + (d/2)²), where d is the
 * cell diagonal.
 *
 * Thrun's phrasing is that the main Gaussian cone "may be enlarged by half the
 * diameter of the grid cell". Adding in quadrature is the version that follows
 * from the smoothing-kernel view — evaluating the model at the cell centre and
 * convolving it with the cell's own uniform extent — and it has the right limit:
 * as the grid gets fine, the correction disappears.
 */
export function coarsenedSigma(sigma: number, cellSize: number): number {
  const halfDiagonal = (cellSize * Math.SQRT2) / 2;
  return Math.sqrt(sigma * sigma + halfDiagonal * halfDiagonal);
}
