/**
 * NDT — the Normal Distributions Transform (Biber & Straßer, 2003).
 *
 * ICP's cost is piecewise: change the pose a little, a correspondence flips,
 * and the objective takes a step. NDT removes correspondences entirely.
 * Partition the target scan into cells, fit one Gaussian per cell, and score a
 * candidate transform by how likely the *source* points are under that mixture:
 *
 *     s(T) = Σ_k Σ_{i ∈ N(k)} exp( −½ d_ikᵀ Σ_i⁻¹ d_ik ),   d_ik = T p_k − μ_i
 *
 * Up to mixture weights this is the log-likelihood of the scan under a
 * Gaussian-mixture map — the same per-beam-independence assumption Chapter 10
 * makes, with the same caveats — and it is smooth, so Newton's method has
 * something to hold on to. It is also, read the other way, Chapter 10's
 * likelihood field with an anisotropic kernel fitted per cell instead of one
 * isotropic σ everywhere.
 *
 * Rust counterpart: `crates/ch16_slam2d/src/ndt.rs`.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import { symEig } from '../models/motion-se2';
import { solve, type Mat } from '../prob/linalg';
import { cellKey, type Pt } from './cloud';

export interface NdtCell {
  i: number;
  j: number;
  mean: Pt;
  cov: Mat;
  /** Σ⁻¹, precomputed: the score evaluates it once per point per cell. */
  info: Mat;
  count: number;
}

export class NdtMap {
  readonly cellSize: number;
  readonly cells = new Map<number, NdtCell>();

  constructor(cellSize: number) {
    this.cellSize = cellSize;
  }

  list(): NdtCell[] {
    return [...this.cells.values()];
  }

  /**
   * The mixture density at one point, summed over the 3×3 cell neighbourhood.
   *
   * Summing neighbours rather than snapping to the containing cell is the cheap
   * cure for NDT's one genuine ugliness: a hard cell boundary puts a
   * discontinuity in the middle of an otherwise smooth objective. (Biber &
   * Straßer's own fix is four overlapping grids; this is the same idea with
   * less bookkeeping.)
   */
  scoreAt(p: Pt): number {
    const ci = Math.floor(p[0] / this.cellSize);
    const cj = Math.floor(p[1] / this.cellSize);
    let s = 0;
    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        const cell = this.cells.get(cellKey(ci + di, cj + dj));
        if (!cell) continue;
        const dx = p[0] - cell.mean[0];
        const dy = p[1] - cell.mean[1];
        const m =
          dx * (cell.info[0][0] * dx + cell.info[0][1] * dy) +
          dy * (cell.info[1][0] * dx + cell.info[1][1] * dy);
        s += Math.exp(-0.5 * m);
      }
    }
    return s;
  }

  /** s(T) for a whole cloud. Larger is better; the aligner maximizes it. */
  score(points: readonly Pt[], pose: Pose2): number {
    const c = Math.cos(pose.theta);
    const s = Math.sin(pose.theta);
    let total = 0;
    for (const [x, y] of points) {
      total += this.scoreAt([pose.x + c * x - s * y, pose.y + s * x + c * y]);
    }
    return total;
  }
}

/**
 * `build_ndt` — one Gaussian per occupied cell.
 *
 * The regularization is not optional. A cell straddling a flat wall gets a
 * covariance whose smaller eigenvalue is the range noise squared — often
 * numerically zero — and Σ⁻¹ then reports infinite confidence across the wall.
 * Clamping λ_min to a fraction of λ_max is Biber & Straßer's own fix and it is
 * the difference between a smooth basin and a wall of numerical spikes.
 */
export function buildNdt(points: readonly Pt[], cellSize = 0.5, minPoints = 4): NdtMap {
  const map = new NdtMap(cellSize);
  const buckets = new Map<number, Pt[]>();
  for (const p of points) {
    const i = Math.floor(p[0] / cellSize);
    const j = Math.floor(p[1] / cellSize);
    const key = cellKey(i, j);
    const b = buckets.get(key);
    if (b) b.push(p);
    else buckets.set(key, [p]);
  }

  for (const [key, pts] of buckets) {
    if (pts.length < minPoints) continue;
    let mx = 0;
    let my = 0;
    for (const p of pts) {
      mx += p[0];
      my += p[1];
    }
    mx /= pts.length;
    my /= pts.length;
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (const p of pts) {
      const dx = p[0] - mx;
      const dy = p[1] - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const nrm = pts.length - 1;
    const cov = regularize([
      [sxx / nrm, sxy / nrm],
      [sxy / nrm, syy / nrm],
    ]);
    const det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    const info: Mat = [
      [cov[1][1] / det, -cov[0][1] / det],
      [-cov[1][0] / det, cov[0][0] / det],
    ];
    const first = pts[0];
    map.cells.set(key, {
      i: Math.floor(first[0] / cellSize),
      j: Math.floor(first[1] / cellSize),
      mean: [mx, my],
      cov,
      info,
      count: pts.length,
    });
  }
  return map;
}

/** Clamp the covariance's smaller eigenvalue to `ratio` of the larger. */
export function regularize(cov: Mat, ratio = 0.01, floor = 1e-4): Mat {
  const { values, vectors } = symEig(cov);
  const lMax = Math.max(values[0], floor);
  const lMin = Math.max(values[1], ratio * lMax);
  const e1 = vectors[0];
  const e2 = vectors[1];
  return [
    [lMax * e1[0] * e1[0] + lMin * e2[0] * e2[0], lMax * e1[0] * e1[1] + lMin * e2[0] * e2[1]],
    [lMax * e1[1] * e1[0] + lMin * e2[1] * e2[0], lMax * e1[1] * e1[1] + lMin * e2[1] * e2[1]],
  ];
}

export interface NdtAlignResult {
  pose: Pose2;
  score: number;
  iterations: number;
  trace: Pose2[];
}

/**
 * `ndt_align` — Newton's method on −log s(T), with a Gauss–Newton Hessian.
 *
 * The derivatives are analytic and cheap. With d = T p − μ, C = Σ⁻¹, and
 * ξ = (t_x, t_y, θ):
 *
 *   ∂d/∂t = I,   ∂d/∂θ = J R p = (−r_y, r_x),   J = [[0,−1],[1,0]]
 *   g_i = Σ_k s_k · (dᵀ C ∂d/∂ξ_i)
 *   H_ij ≈ Σ_k s_k · (∂d/∂ξ_i)ᵀ C (∂d/∂ξ_j)
 *
 * The exact Hessian carries two more terms and is indefinite away from the
 * optimum — on the corridor scene this chapter uses to make its point, it is
 * indefinite *at* the optimum too. Dropping them leaves a positive
 * semi-definite matrix, which is the same Gauss–Newton bargain Chapter 15
 * strikes: give up quadratic convergence, keep a descent direction always.
 */
export function ndtAlign(
  source: readonly Pt[],
  map: NdtMap,
  init: Pose2,
  maxIters = 25,
  tolerance = 1e-6,
): NdtAlignResult {
  let pose = { ...init };
  const trace: Pose2[] = [{ ...pose }];
  let score = map.score(source, pose);
  let iterations = 0;
  let lambda = 1e-4;

  for (let it = 0; it < maxIters; it++) {
    const c = Math.cos(pose.theta);
    const s = Math.sin(pose.theta);
    const g = [0, 0, 0];
    const h: Mat = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];

    for (const p of source) {
      const rx = c * p[0] - s * p[1];
      const ry = s * p[0] + c * p[1];
      const wx = pose.x + rx;
      const wy = pose.y + ry;
      const ci = Math.floor(wx / map.cellSize);
      const cj = Math.floor(wy / map.cellSize);
      // Jacobian columns ∂d/∂ξ, as rows of a 3×2 table.
      const jac: Pt[] = [
        [1, 0],
        [0, 1],
        [-ry, rx],
      ];
      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          const cell = map.cells.get(cellKey(ci + di, cj + dj));
          if (!cell) continue;
          const dx = wx - cell.mean[0];
          const dy = wy - cell.mean[1];
          const cdx = cell.info[0][0] * dx + cell.info[0][1] * dy;
          const cdy = cell.info[1][0] * dx + cell.info[1][1] * dy;
          const sk = Math.exp(-0.5 * (dx * cdx + dy * cdy));
          if (sk < 1e-9) continue;
          for (let i = 0; i < 3; i++) {
            g[i] += sk * (cdx * jac[i][0] + cdy * jac[i][1]);
            for (let j = 0; j < 3; j++) {
              const cj0 = cell.info[0][0] * jac[j][0] + cell.info[0][1] * jac[j][1];
              const cj1 = cell.info[1][0] * jac[j][0] + cell.info[1][1] * jac[j][1];
              h[i][j] += sk * (jac[i][0] * cj0 + jac[i][1] * cj1);
            }
          }
        }
      }
    }

    if (Math.hypot(g[0], g[1]) + Math.abs(g[2]) < tolerance) break;

    let accepted = false;
    for (let back = 0; back < 10; back++) {
      const damped = h.map((row, i) => row.map((v, j) => (i === j ? v * (1 + lambda) + 1e-9 : v)));
      const delta = solve(
        damped,
        g.map((v) => -v),
      );
      const cand: Pose2 = {
        x: pose.x + delta[0],
        y: pose.y + delta[1],
        theta: normalizeAngle(pose.theta + delta[2]),
      };
      const cs = map.score(source, cand);
      if (cs > score) {
        pose = cand;
        score = cs;
        accepted = true;
        lambda = Math.max(lambda * 0.5, 1e-6);
        break;
      }
      lambda *= 6;
    }
    iterations = it + 1;
    trace.push({ ...pose });
    if (!accepted) break;
  }

  return { pose, score, iterations, trace };
}
