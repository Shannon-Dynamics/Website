/**
 * ICP — the iterative closest point algorithm, in both classic flavours.
 *
 * Registration is maximum likelihood in disguise. Model each matched target
 * point as a noisy observation of a transformed source point,
 * q_{c(k)} = T p_k + ε, ε ~ N(0, σ²I), and the MLE for T is
 *
 *     T* = argmin_T Σ_k ‖ T p_k − q_{c(k)} ‖²
 *
 * — least squares, exactly the objective Chapter 15's optimizer minimizes. ICP
 * alternates two steps that each minimize that joint cost over one argument:
 * pick correspondences with T fixed (the E-step analogue), then solve for T
 * with correspondences fixed (the M-step analogue). Both decrease the cost, the
 * cost is bounded below, so the iteration converges — to a *local* minimum, and
 * this file is also where you can watch it find the wrong one.
 *
 * Rust counterpart: `crates/ch16_slam2d/src/icp.rs`.
 */

import { compose, normalizeAngle, se2Log, type Pose2 } from '../geom/se2';
import { solve, type Mat } from '../prob/linalg';
import { transformCloud, VoxelMap, type Pt } from './cloud';

export type IcpVariant = 'point-to-point' | 'point-to-plane';

export interface IcpConfig {
  maxIters: number;
  /** Correspondence rejection radius τ, in metres. */
  tau: number;
  variant: IcpVariant;
  /** Stop when the incremental motion falls below this (metres + radians). */
  tolerance: number;
  /** Give up if fewer than this many correspondences survive τ. */
  minPairs: number;
}

export const DEFAULT_ICP: IcpConfig = {
  maxIters: 25,
  tau: 0.6,
  variant: 'point-to-point',
  tolerance: 1e-4,
  minPairs: 8,
};

/** One correspondence: a source point (already transformed) and its target. */
export interface Correspondence {
  src: Pt;
  dst: Pt;
  normal: Pt | null;
  d: number;
}

export interface IcpIteration {
  pose: Pose2;
  rmse: number;
  pairs: Correspondence[];
}

export interface IcpResult {
  pose: Pose2;
  rmse: number;
  inliers: number;
  converged: boolean;
  /** Pose after every iteration, iteration 0 being the initial guess. */
  trace: IcpIteration[];
}

/**
 * `svd_align` — the closed-form rigid alignment of Arun et al. (1987),
 * specialized to the plane where the SVD collapses to a single `atan2`.
 *
 * With correspondences fixed, translating both clouds to their centroids
 * removes **t** from the problem, and what is left is
 * maximize tr(RᵀW) with W = Σ (q_k − q̄)(p_k − p̄)ᵀ. In 2-D,
 *
 *     tr(RᵀW) = cos θ (W₁₁ + W₂₂) + sin θ (W₂₁ − W₁₂)
 *
 * which is maximized at θ = atan2(W₂₁ − W₁₂, W₁₁ + W₂₂) — and since `atan2`
 * only ever returns a rotation, the det-correction that stops the 3-D SVD from
 * returning a reflection is built in here for free.
 */
export function svdAlign(src: readonly Pt[], dst: readonly Pt[], weights?: readonly number[]): Pose2 {
  const n = Math.min(src.length, dst.length);
  if (n === 0) return { x: 0, y: 0, theta: 0 };
  let wsum = 0;
  let px = 0;
  let py = 0;
  let qx = 0;
  let qy = 0;
  for (let k = 0; k < n; k++) {
    const w = weights?.[k] ?? 1;
    wsum += w;
    px += w * src[k][0];
    py += w * src[k][1];
    qx += w * dst[k][0];
    qy += w * dst[k][1];
  }
  if (wsum <= 0) return { x: 0, y: 0, theta: 0 };
  px /= wsum;
  py /= wsum;
  qx /= wsum;
  qy /= wsum;

  // sDot = W₁₁ + W₂₂ (Σ p′·q′), sCross = W₂₁ − W₁₂ (Σ p′ × q′).
  let sDot = 0;
  let sCross = 0;
  for (let k = 0; k < n; k++) {
    const w = weights?.[k] ?? 1;
    const ax = src[k][0] - px;
    const ay = src[k][1] - py;
    const bx = dst[k][0] - qx;
    const by = dst[k][1] - qy;
    sDot += w * (ax * bx + ay * by);
    sCross += w * (ax * by - ay * bx);
  }
  const theta = Math.atan2(sCross, sDot);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return { x: qx - (c * px - s * py), y: qy - (s * px + c * py), theta: normalizeAngle(theta) };
}

/**
 * One Gauss–Newton step of the point-to-plane cost, in closed form.
 *
 * Minimizing Σ (n_kᵀ(R p_k + t − q_k))² has no closed-form solution, but with
 * R ≈ I + θJ, J = [[0,−1],[1,0]], every residual becomes linear in (θ, tx, ty):
 *
 *     a_kᵀ = ( p̃_k × n_k ,  n_x ,  n_y ),   b_k = −n_kᵀ(p_k − q_k)
 *
 * where p̃_k is the source point measured from the cloud's centroid — rotating
 * about the centroid rather than the world origin is what keeps the 3×3 normal
 * matrix well conditioned when the scan sits ten metres from the origin.
 *
 * The returned pose is the *increment* to apply in the target frame.
 */
export function pointToPlaneStep(pairs: readonly Correspondence[]): Pose2 {
  let cx = 0;
  let cy = 0;
  let used = 0;
  for (const pr of pairs) {
    if (!pr.normal) continue;
    cx += pr.src[0];
    cy += pr.src[1];
    used += 1;
  }
  if (used < 3) return { x: 0, y: 0, theta: 0 };
  cx /= used;
  cy /= used;

  const h: Mat = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const g = [0, 0, 0];
  for (const pr of pairs) {
    const n = pr.normal;
    if (!n) continue;
    const rx = pr.src[0] - cx;
    const ry = pr.src[1] - cy;
    const a = [rx * n[1] - ry * n[0], n[0], n[1]];
    const b = -(n[0] * (pr.src[0] - pr.dst[0]) + n[1] * (pr.src[1] - pr.dst[1]));
    for (let i = 0; i < 3; i++) {
      g[i] += a[i] * b;
      for (let j = 0; j < 3; j++) h[i][j] += a[i] * a[j];
    }
  }
  // Damping, and not the token kind. Along a featureless corridor this matrix
  // is *genuinely* singular in the along-wall direction: both H and g vanish
  // there, and an absolute ridge of 1e-9 would divide the numerical dust in g
  // by the numerical dust in H and slide the scan a metre down the hallway.
  // Damping relative to the trace makes the unconstrained direction resolve to
  // zero motion instead — "the scan says nothing about this axis, so keep the
  // prediction", which is the honest answer.
  const ridge = 1e-4 * ((h[0][0] + h[1][1] + h[2][2]) / 3) + 1e-12;
  for (let i = 0; i < 3; i++) h[i][i] += ridge;

  const x = solve(h, g);
  const theta = x[0];
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  // The increment rotates about the centroid, so its translation part has to
  // carry the (c − Rc) offset back out to world coordinates.
  return {
    x: cx - (c * cx - s * cy) + x[1],
    y: cy - (s * cx + c * cy) + x[2],
    theta: normalizeAngle(theta),
  };
}

/** Nearest-neighbour correspondences within τ, with the target's normals. */
export function findCorrespondences(
  source: readonly Pt[],
  target: VoxelMap,
  pose: Pose2,
  tau: number,
): Correspondence[] {
  const moved = transformCloud(pose, source);
  const pairs: Correspondence[] = [];
  for (const p of moved) {
    const idx = target.nearestIndex(p, tau);
    if (idx < 0) continue;
    const q = target.pts[idx];
    pairs.push({ src: p, dst: q, normal: target.normals[idx], d: Math.hypot(q[0] - p[0], q[1] - p[1]) });
  }
  return pairs;
}

const rmseOf = (pairs: readonly Correspondence[], variant: IcpVariant): number => {
  if (pairs.length === 0) return Infinity;
  let s = 0;
  for (const pr of pairs) {
    if (variant === 'point-to-plane' && pr.normal) {
      const e = pr.normal[0] * (pr.src[0] - pr.dst[0]) + pr.normal[1] * (pr.src[1] - pr.dst[1]);
      s += e * e;
    } else {
      s += pr.d * pr.d;
    }
  }
  return Math.sqrt(s / pairs.length);
};

/**
 * `icp_point_to_point` / `icp_point_to_plane` — the loop itself.
 *
 * Each iteration re-associates and re-solves; the full pose trace is kept
 * because the interesting part of ICP is never the answer, it is the path it
 * took to get there.
 */
export function icp(
  source: readonly Pt[],
  target: VoxelMap,
  init: Pose2,
  cfg: Partial<IcpConfig> = {},
): IcpResult {
  const c: IcpConfig = { ...DEFAULT_ICP, ...cfg };
  let pose = { ...init };
  const trace: IcpIteration[] = [];
  let converged = false;
  let pairs = findCorrespondences(source, target, pose, c.tau);
  trace.push({ pose: { ...pose }, rmse: rmseOf(pairs, c.variant), pairs });

  for (let it = 0; it < c.maxIters; it++) {
    if (pairs.length < c.minPairs) break;

    const delta =
      c.variant === 'point-to-plane'
        ? pointToPlaneStep(pairs)
        : svdAlign(
            pairs.map((p) => p.src),
            pairs.map((p) => p.dst),
          );

    // The increment lives in the target frame, so it composes on the left.
    pose = compose(delta, pose);
    pairs = findCorrespondences(source, target, pose, c.tau);
    trace.push({ pose: { ...pose }, rmse: rmseOf(pairs, c.variant), pairs });

    const step = se2Log(delta);
    if (Math.hypot(step[0], step[1]) + Math.abs(step[2]) < c.tolerance) {
      converged = true;
      break;
    }
  }

  const last = trace[trace.length - 1];
  return { pose, rmse: last.rmse, inliers: last.pairs.length, converged, trace };
}

/**
 * The information matrix ICP *would* report for its own answer: Ω = Jᵀ J / σ².
 *
 * This is the Gauss–Newton approximation of the Hessian at the solution — the
 * curvature of the cost, which is exactly what an information matrix is. Take
 * it at face value and the pose graph will believe a corridor match far more
 * than it should, because the derivation assumes every beam is an independent
 * observation and consecutive LiDAR beams are anything but. `inflate` is the
 * same honest fudge Chapter 10 applies to the beam model: divide the effective
 * number of independent measurements by a constant, and say so out loud.
 */
export function icpInformation(
  pairs: readonly Correspondence[],
  sigma: number,
  inflate = 8,
  /**
   * Heading of the frame the information should be expressed in. The pairs
   * arrive in world coordinates; a pose-graph edge wants Ω in the tangent at
   * the *relative* pose, which for translation-first ordering is a plain
   * rotation of the (vx, vy) block. The lever-arm part of the adjoint is
   * already absorbed by measuring the rotational Jacobian from the centroid.
   */
  bodyTheta = 0,
): Mat {
  let cx = 0;
  let cy = 0;
  let used = 0;
  for (const pr of pairs) {
    cx += pr.src[0];
    cy += pr.src[1];
    used += 1;
  }
  const omega: Mat = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  if (used === 0) return omega;
  cx /= used;
  cy /= used;

  const w = 1 / (sigma * sigma * inflate);
  const cb = Math.cos(bodyTheta);
  const sb = Math.sin(bodyTheta);
  for (const pr of pairs) {
    // Without a normal, the point-to-point residual constrains both axes; the
    // two unit directions below are then the identity's rows.
    const dirs: Pt[] = pr.normal
      ? [pr.normal]
      : [
          [1, 0],
          [0, 1],
        ];
    const rx = pr.src[0] - cx;
    const ry = pr.src[1] - cy;
    for (const n of dirs) {
      const lever = rx * n[1] - ry * n[0];
      // Rᵀn: express the constrained direction in the requested frame.
      const a = [cb * n[0] + sb * n[1], -sb * n[0] + cb * n[1], lever];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) omega[i][j] += w * a[i] * a[j];
    }
  }
  return omega;
}

/**
 * KISS-ICP's adaptive correspondence threshold.
 *
 * Rather than tuning τ per dataset, estimate how wrong the constant-velocity
 * prediction has been so far and set τ = 3σ_t from that. σ_t² is the running
 * mean of the squared model deviation δ = ‖Δt‖ + 2 r_max sin(Δθ/2), the worst
 * displacement the missed rotation could have caused at the far end of a scan.
 */
export class AdaptiveThreshold {
  private sumSq = 0;
  private count = 0;

  constructor(
    readonly initial = 1.0,
    readonly minMotion = 0.05,
    readonly maxRange = 6,
  ) {}

  get tau(): number {
    if (this.count < 3) return this.initial;
    return 3 * Math.sqrt(this.sumSq / this.count);
  }

  /** Feed the deviation between the predicted pose and the registered one. */
  update(deviation: Pose2): void {
    const dt = Math.hypot(deviation.x, deviation.y);
    const dr = 2 * this.maxRange * Math.sin(Math.abs(normalizeAngle(deviation.theta)) / 2);
    const d = dt + dr;
    if (d < this.minMotion) return;
    this.sumSq += d * d;
    this.count += 1;
  }
}
