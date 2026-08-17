/**
 * Pose-graph SLAM — a factor graph whose only variables are poses.
 *
 * Every constraint is a relative-pose measurement Z_ij with information Ω_ij,
 * and its residual lives in the tangent space of SE(2):
 *
 *     e_ij = log( Z_ij⁻¹ T_i⁻¹ T_j )^∨  ∈ ℝ³
 *
 * The whole back-end is then Chapter 15's Gauss–Newton, unchanged:
 * linearize, build H = Σ AᵀΩA, solve H δ = −b, retract with ⊞, repeat. The
 * only chapter-specific work is the pair of Jacobians, which the adjoint hands
 * over almost for free.
 *
 * Rust counterpart: `crates/ch16_slam2d/src/graph.rs`, where the same normal
 * equations are assembled sparsely and factored with `faer`'s sparse Cholesky
 * rather than the dense solve used here.
 */

import {
  adjoint,
  between,
  boxplus,
  compose,
  inverse,
  se2Log,
  type Pose2,
  type Twist2,
} from '../geom/se2';
import { matMul, matVec, solve, transpose, zerosMat, type Mat, type Vec } from '../prob/linalg';

export type EdgeKind = 'odometry' | 'loop';

export interface PoseNode {
  id: number;
  pose: Pose2;
  /** A fixed node pins the gauge: without one the graph slides and spins freely. */
  fixed: boolean;
}

export interface PoseEdge {
  i: number;
  j: number;
  /** The measured relative pose T_i⁻¹ T_j. */
  z: Pose2;
  omega: Mat;
  kind: EdgeKind;
}

export interface OptimizeReport {
  /** χ² before each iteration, then after the last one. */
  chi2: number[];
  iterations: number;
  /** Largest per-node correction, in metres — how far history moved. */
  maxShift: number;
}

/** The measurement a perfect odometry would report between two poses. */
export const relativePose = (a: Pose2, b: Pose2): Pose2 => between(a, b);

/** A diagonal information matrix from per-axis standard deviations. */
export function informationFromSigmas(sx: number, sy: number, stheta: number): Mat {
  return [
    [1 / (sx * sx), 0, 0],
    [0, 1 / (sy * sy), 0],
    [0, 0, 1 / (stheta * stheta)],
  ];
}

export class PoseGraph {
  readonly nodes: PoseNode[] = [];
  readonly edges: PoseEdge[] = [];

  addNode(pose: Pose2, fixed = false): number {
    const id = this.nodes.length;
    this.nodes.push({ id, pose: { ...pose }, fixed: fixed || id === 0 });
    return id;
  }

  addEdge(i: number, j: number, z: Pose2, omega: Mat, kind: EdgeKind = 'odometry'): void {
    this.edges.push({ i, j, z: { ...z }, omega, kind });
  }

  /** e_ij = log(Z⁻¹ T_i⁻¹ T_j)^∨, computed as (T_j) ⊟ (T_i ∘ Z). */
  residual(e: PoseEdge): Twist2 {
    return se2Log(between(compose(this.nodes[e.i].pose, e.z), this.nodes[e.j].pose));
  }

  /** Mahalanobis norm of one edge — the quantity the χ² loop-closure gate reads. */
  edgeChi2(e: PoseEdge): number {
    const r = this.residual(e);
    const w = matVec(e.omega, r as unknown as Vec);
    return r[0] * w[0] + r[1] * w[1] + r[2] * w[2];
  }

  chi2(): number {
    let s = 0;
    for (const e of this.edges) s += this.edgeChi2(e);
    return s;
  }

  poses(): Pose2[] {
    return this.nodes.map((n) => ({ ...n.pose }));
  }

  /**
   * `pose_graph_optimize` — Gauss–Newton on the manifold.
   *
   * Jacobians. Perturb on the right, T ← T ⊞ δ. Sliding exp(δ_i) past Z⁻¹
   * turns it into exp(−Ad_{Z⁻¹} δ_i) acting on the left of the error, so
   *
   *     ∂e/∂δ_i = −Ad_{Z⁻¹},      ∂e/∂δ_j = I
   *
   * both to first order in the residual: the exact expressions carry a right-
   * Jacobian factor J_r⁻¹(e) which is I + O(‖e‖) and which every practical
   * SE(2) implementation (g2o's included) is content to drop, because Gauss–
   * Newton only needs a descent direction — the *residual* is exact, and that
   * is what fixes the answer.
   *
   * `huber` down-weights an edge whose Mahalanobis norm exceeds the threshold,
   * so one bad loop closure bends the trajectory instead of snapping it.
   */
  optimize(maxIters = 12, tolerance = 1e-6, huber = 0): OptimizeReport {
    const n = this.nodes.length;
    const dim = 3 * n;
    const chi2: number[] = [this.chi2()];
    const before = this.poses();
    let iterations = 0;

    for (let iter = 0; iter < maxIters; iter++) {
      const h: Mat = zerosMat(dim, dim);
      const b: Vec = new Array(dim).fill(0);

      for (const e of this.edges) {
        const r = this.residual(e) as unknown as Vec;
        const ai = matMul(
          [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
          ],
          adjoint(inverse(e.z)),
        );
        const aj: Mat = [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ];

        let omega = e.omega;
        if (huber > 0) {
          const w = matVec(omega, r);
          const m = Math.sqrt(Math.max(r[0] * w[0] + r[1] * w[1] + r[2] * w[2], 0));
          if (m > huber) {
            const scale = huber / m;
            omega = omega.map((row) => row.map((v) => v * scale));
          }
        }

        const blocks: [number, Mat][] = [
          [e.i, ai],
          [e.j, aj],
        ];
        for (const [na, a] of blocks) {
          const at = transpose(a);
          const atO = matMul(at, omega);
          const gb = matVec(atO, r);
          for (let k = 0; k < 3; k++) b[3 * na + k] += gb[k];
          for (const [nb, bMat] of blocks) {
            const hb = matMul(atO, bMat);
            for (let k = 0; k < 3; k++) {
              for (let l = 0; l < 3; l++) h[3 * na + k][3 * nb + l] += hb[k][l];
            }
          }
        }
      }

      // Gauge fixing: replace the fixed node's block with the identity so its
      // increment comes back exactly zero, rather than trusting a huge prior.
      for (const node of this.nodes) {
        if (!node.fixed) continue;
        for (let k = 0; k < 3; k++) {
          const row = 3 * node.id + k;
          for (let c = 0; c < dim; c++) {
            h[row][c] = 0;
            h[c][row] = 0;
          }
          h[row][row] = 1;
          b[row] = 0;
        }
      }
      // Levenberg-flavoured damping: the graph is well conditioned right up to
      // the moment a loop closure lands, and then it briefly is not.
      for (let k = 0; k < dim; k++) h[k][k] += 1e-9;

      const delta = solve(
        h,
        b.map((v) => -v),
      );

      let maxStep = 0;
      for (const node of this.nodes) {
        if (node.fixed) continue;
        const d: Twist2 = [delta[3 * node.id], delta[3 * node.id + 1], delta[3 * node.id + 2]];
        maxStep = Math.max(maxStep, Math.hypot(d[0], d[1]) + Math.abs(d[2]));
        node.pose = boxplus(node.pose, d);
      }

      iterations = iter + 1;
      chi2.push(this.chi2());
      if (maxStep < tolerance) break;
    }

    let maxShift = 0;
    for (let k = 0; k < n; k++) {
      maxShift = Math.max(maxShift, Math.hypot(this.nodes[k].pose.x - before[k].x, this.nodes[k].pose.y - before[k].y));
    }
    return { chi2, iterations, maxShift };
  }
}

/**
 * `detect_loop_candidates` — which past poses might I be standing on top of?
 *
 * The cheap channel: everything inside `radius` of the current estimate that is
 * at least `minGap` nodes old. Old enough matters — the previous ten nodes are
 * always nearby and closing a loop against them tells you nothing you did not
 * already have from odometry. Production systems replace the radius with a
 * place descriptor (Scan Context, DBoW) precisely because the radius depends on
 * a pose estimate that, by the time you need a loop closure, is wrong.
 */
export function detectLoopCandidates(
  graph: PoseGraph,
  current: number,
  radius = 2.0,
  minGap = 25,
): number[] {
  const here = graph.nodes[current].pose;
  const out: number[] = [];
  for (const node of graph.nodes) {
    if (current - node.id < minGap) continue;
    if (Math.hypot(node.pose.x - here.x, node.pose.y - here.y) <= radius) out.push(node.id);
  }
  return out;
}

/**
 * The χ² acceptance gate.
 *
 * A verified loop is one whose residual is small *relative to how uncertain the
 * two poses were*. With three degrees of freedom, the 0.95 quantile of χ²₃ is
 * 7.815: a candidate whose Mahalanobis norm exceeds it would be a 1-in-20
 * coincidence under the hypothesis that the match is real, and false loop
 * closures are the one failure mode a pose graph cannot survive.
 */
export const CHI2_3DOF_95 = 7.815;

export function chi2Gate(residual: Twist2, omega: Mat, threshold = CHI2_3DOF_95): boolean {
  const w = matVec(omega, residual as unknown as Vec);
  return residual[0] * w[0] + residual[1] * w[1] + residual[2] * w[2] <= threshold;
}
