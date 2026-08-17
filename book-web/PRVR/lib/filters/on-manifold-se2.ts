/**
 * Gaussian filters whose *state* lives on SE(2) — Chapter 7.
 *
 * Chapter 6's `Kf` and the `Ekf` above both keep a mean in ℝⁿ. For a pose that
 * is a lie: adding a Kalman correction to a heading is only meaningful modulo
 * 2π, and averaging two headings as reals can produce a direction neither of
 * them pointed in. The repair is not `mod 2π`. It is to keep
 *
 *   the mean on the manifold, μ ∈ SE(2),
 *   the covariance in its tangent space, Σ ∈ ℝ³ˣ³,
 *
 * and to replace every `+` and `−` acting on the state with ⊞ and ⊟.
 *
 * Two filters follow that recipe:
 *
 *   `EskfSe2` — error-state EKF. Keeps a nominal pose plus a tangent-space
 *               error that is always small, filters the error, injects it with
 *               ⊞ and resets. The production pattern behind every modern
 *               VIO/LIO stack.
 *   `UkfmSe2` — the unscented filter on a manifold (Brossard et al., UKF-M):
 *               sigma points are retracted out of the tangent space with ⊞ and
 *               recombined with a ⊟-mean iteration. No Jacobians at all.
 *
 * Conventions (fixed book-wide, Chapter 3):
 *   right/local perturbation  x = μ ⊞ ε = μ · exp(ε)
 *   tangent ordering translation-first, body frame: ε = (εₓ, ε_y, ε_θ)
 */

import {
  adjoint,
  angleDiff,
  boxminus,
  boxplus,
  inverse,
  se2Exp,
  type Pose2,
  type Twist2,
} from '../geom/se2';
import {
  cholesky,
  eye,
  inv,
  matAdd,
  matMul,
  matScale,
  matSub,
  matVec,
  outer,
  sub,
  symmetrize,
  transpose,
  zerosMat,
  type Mat,
  type Vec,
} from '../prob/linalg';

/** A Gaussian whose mean lives on SE(2) and whose covariance lives in its tangent space. */
export interface Se2Gaussian {
  mean: Pose2;
  /** 3×3, expressed in the body-frame tangent basis at `mean`. */
  cov: Mat;
}

/* -------------------------------------------------------------------------- */
/* Jacobians                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Motion Jacobian of a body-twist update, in tangent coordinates:
 *
 *   G_t = ∂ ( ((μ ⊞ ε) ⊞ u) ⊟ (μ ⊞ u) ) / ∂ε |₀ = Ad_{exp(u)⁻¹}
 *
 * Two lines of algebra: exp(ε)·exp(u) = exp(u)·exp(Ad_{exp(u)⁻¹} ε) to first
 * order in ε. Note what it does *not* contain — the state μ. A body-frame
 * odometry step propagates its error the same way wherever the robot is, which
 * is the seed of the invariant-filtering idea.
 */
export function se2MotionJacobian(u: Twist2): Mat {
  return adjoint(inverse(se2Exp(u)));
}

/**
 * Reset Jacobian applied after injecting a correction ε̂.
 *
 * After μ⁺ = μ ⊞ ε̂ the remaining error is measured from the *new* nominal, and
 * the two tangent spaces differ by exactly the adjoint of the injection:
 * ε⁺ = Ad_{exp(ε̂)⁻¹}(ε − ε̂). Most implementations skip it because ε̂ is a
 * centimetre-scale twist and the adjoint is then within a rounding error of the
 * identity; it is kept here so a reader can switch it off and see for
 * themselves how little it matters — until Chapter 14, where ε̂ is a
 * loop-closure-sized jump and it matters a great deal.
 */
export function se2ResetJacobian(epsilon: Vec): Mat {
  return adjoint(inverse(se2Exp([epsilon[0], epsilon[1], epsilon[2]])));
}

/**
 * World-frame position measurement Jacobian, in tangent coordinates.
 *
 * Moving the pose one metre "forward" in its own frame moves the world origin
 * of the robot one metre along its heading, so the block is just R(θ). The
 * heading column is zero: rotating in place does not translate you.
 */
export function se2PositionJacobian(pose: Pose2): Mat {
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  return [
    [c, -s, 0],
    [s, c, 0],
  ];
}

export interface EskfCorrection {
  /** The tangent-space correction ε̂ = K·y that was injected. */
  epsilon: Vec;
  /** Measurement residual y = z ⊟ h(μ̄). */
  innovation: Vec;
  /** Innovation covariance S. */
  S: Mat;
  /** Normalized innovation squared, yᵀS⁻¹y — the NIS consistency statistic. */
  nis: number;
}

/* -------------------------------------------------------------------------- */
/* Error-state EKF                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Error-state EKF on SE(2).
 *
 * `predict` advances the nominal pose through the true nonlinear motion and the
 * covariance through the tangent Jacobian; `correct*` computes an error state,
 * injects it with ⊞, and resets it to zero. The error is never carried between
 * updates — that is the entire point, and it is why `epsilon` is returned
 * rather than stored.
 */
export class EskfSe2 {
  nominal: Pose2;
  cov: Mat;
  /** Apply the exact reset Jacobian (see {@link se2ResetJacobian}). */
  exactReset: boolean;

  constructor(nominal: Pose2, cov: Mat, opts: { exactReset?: boolean } = {}) {
    this.nominal = { ...nominal };
    this.cov = cov.map((r) => r.slice());
    this.exactReset = opts.exactReset ?? true;
  }

  /**
   * μ̄ = μ ⊞ u,  Σ̄ = G Σ Gᵀ + R.
   *
   * `R` is the process noise as a body-frame twist covariance. Strictly the
   * noise enters through the right Jacobian J_r(u), which differs from the
   * identity by O(|u|) — negligible at control rates, and stated here rather
   * than hidden.
   */
  predict(u: Twist2, R: Mat): void {
    const G = se2MotionJacobian(u);
    this.nominal = boxplus(this.nominal, u);
    this.cov = symmetrize(matAdd(matMul(matMul(G, this.cov), transpose(G)), R));
  }

  /** Any measurement with a residual in ℝᵐ and a Jacobian in tangent coordinates. */
  correct(residual: Vec, H: Mat, Q: Mat): EskfCorrection {
    const Ht = transpose(H);
    const S = symmetrize(matAdd(matMul(matMul(H, this.cov), Ht), Q));
    const Sinv = inv(S);
    const K = matMul(matMul(this.cov, Ht), Sinv);
    const epsilon = matVec(K, residual);

    // Inject: the correction is a twist, so it enters through ⊞, never through +.
    this.nominal = boxplus(this.nominal, [epsilon[0], epsilon[1], epsilon[2]]);

    // Joseph form, as in Chapter 6 — positive definite even with a sloppy gain.
    const IKH = matSub(eye(3), matMul(K, H));
    let P = matAdd(
      matMul(matMul(IKH, this.cov), transpose(IKH)),
      matMul(matMul(K, Q), transpose(K)),
    );

    // Reset: the error is now measured in the tangent space of the new nominal.
    if (this.exactReset) {
      const J = se2ResetJacobian(epsilon);
      P = matMul(matMul(J, P), transpose(J));
    }
    this.cov = symmetrize(P);

    const Sy = matVec(Sinv, residual);
    let nis = 0;
    for (let i = 0; i < residual.length; i++) nis += residual[i] * Sy[i];

    return { epsilon, innovation: residual, S, nis };
  }

  /** A position fix (beacon, GPS, motion capture): z = t(x) + δ. */
  correctPosition(z: [number, number], Q: Mat): EskfCorrection {
    const residual: Vec = [z[0] - this.nominal.x, z[1] - this.nominal.y];
    return this.correct(residual, se2PositionJacobian(this.nominal), Q);
  }

  /**
   * A compass: z = θ + δ, wrapped to (−π, π].
   *
   * The residual goes through {@link angleDiff}, not subtraction. That single
   * substitution is the difference between a filter that survives the ±π seam
   * and one that spins the robot round at every crossing.
   */
  correctHeading(z: number, variance: number): EskfCorrection {
    return this.correct([angleDiff(z, this.nominal.theta)], [[0, 0, 1]], [[variance]]);
  }

  /** The true error state x ⊟ μ — available only in simulation, and only for plots. */
  errorAgainst(truth: Pose2): Twist2 {
    return boxminus(truth, this.nominal);
  }

  belief(): Se2Gaussian {
    return { mean: { ...this.nominal }, cov: this.cov.map((r) => r.slice()) };
  }
}

/* -------------------------------------------------------------------------- */
/* UKF on the manifold (UKF-M)                                                 */
/* -------------------------------------------------------------------------- */

export interface UkfmParams {
  alpha: number;
  beta: number;
  kappa: number;
}

export interface Se2SigmaPoints {
  points: Pose2[];
  wm: number[];
  wc: number[];
}

/**
 * 2d+1 sigma points drawn in the tangent space and retracted onto the group.
 *
 * The only change from the vector version in `ukf.ts` is the last character of
 * each line: `μ + column` becomes `μ ⊞ column`. Every point is a genuine pose;
 * none of them is a 3-vector pretending to be one.
 */
export function se2SigmaPoints(
  g: Se2Gaussian,
  params: Partial<UkfmParams> = {},
): Se2SigmaPoints {
  const n = 3;
  const alpha = params.alpha ?? 1;
  const beta = params.beta ?? 2;
  const kappa = params.kappa ?? 3 - n;
  const lambda = alpha * alpha * (n + kappa) - n;
  const c = Math.sqrt(n + lambda);
  const L = cholesky(g.cov.map((row, i) => row.map((v, j) => (i === j ? v + 1e-12 : v))));

  const points: Pose2[] = [{ ...g.mean }];
  for (let i = 0; i < n; i++) {
    const col: Twist2 = [c * L[0][i], c * L[1][i], c * L[2][i]];
    points.push(boxplus(g.mean, col));
    points.push(boxplus(g.mean, [-col[0], -col[1], -col[2]]));
  }

  const wm = new Array<number>(2 * n + 1).fill(1 / (2 * (n + lambda)));
  const wc = wm.slice();
  wm[0] = lambda / (n + lambda);
  wc[0] = lambda / (n + lambda) + (1 - alpha * alpha + beta);
  return { points, wm, wc };
}

/**
 * The weighted mean of a set of poses, computed by fixed-point iteration.
 *
 * There is no closed form: the mean of points on a curved space is defined as
 * the pose from which the weighted tangent displacements sum to zero, and you
 * find it by repeatedly stepping in the direction of that sum. Three iterations
 * is plenty at filter scale; the loop is what a vector-space UKF gets for free
 * with a single weighted sum.
 */
export function se2WeightedMean(points: Pose2[], w: number[], iters = 6): Pose2 {
  let m: Pose2 = { ...points[0] };
  for (let k = 0; k < iters; k++) {
    const d: Twist2 = [0, 0, 0];
    for (let i = 0; i < points.length; i++) {
      const e = boxminus(points[i], m);
      d[0] += w[i] * e[0];
      d[1] += w[i] * e[1];
      d[2] += w[i] * e[2];
    }
    m = boxplus(m, d);
    if (Math.hypot(d[0], d[1], d[2]) < 1e-12) break;
  }
  return m;
}

/**
 * Unscented Kalman filter on SE(2) — Brossard, Barrau & Bonnabel's UKF-M,
 * specialised to the plane.
 *
 * No Jacobians anywhere: the motion and measurement models are black boxes,
 * probed at 2d+1 poses. Everything the vector UKF does with `+`, `−` and a
 * weighted sum, this does with ⊞, ⊟ and {@link se2WeightedMean}.
 */
export class UkfmSe2 {
  mean: Pose2;
  cov: Mat;
  params: UkfmParams;

  constructor(mean: Pose2, cov: Mat, params: Partial<UkfmParams> = {}) {
    this.mean = { ...mean };
    this.cov = cov.map((r) => r.slice());
    this.params = {
      alpha: params.alpha ?? 1,
      beta: params.beta ?? 2,
      kappa: params.kappa ?? 0,
    };
  }

  /** Push the sigma poses through the true motion, refit on the manifold, add R. */
  predict(g: (x: Pose2) => Pose2, R: Mat): void {
    const { points, wm, wc } = se2SigmaPoints(this, this.params);
    const moved = points.map(g);
    const mean = se2WeightedMean(moved, wm);

    let cov = zerosMat(3, 3);
    for (let i = 0; i < moved.length; i++) {
      const e = boxminus(moved[i], mean) as unknown as Vec;
      cov = matAdd(cov, matScale(outer(e, e), wc[i]));
    }
    this.mean = mean;
    this.cov = symmetrize(matAdd(cov, R));
  }

  /**
   * Measurement update. `residual` defaults to plain subtraction — override it
   * for a bearing or a compass, whose measurement space is a circle too.
   */
  correct(
    z: Vec,
    h: (x: Pose2) => Vec,
    Q: Mat,
    residual: (a: Vec, b: Vec) => Vec = sub,
  ): { innovation: Vec; S: Mat; K: Mat } {
    const { points, wm, wc } = se2SigmaPoints(this, this.params);
    const Z = points.map(h);
    const m = Z[0].length;

    const zHat = new Array<number>(m).fill(0);
    for (let i = 0; i < points.length; i++) {
      const d = residual(Z[i], Z[0]);
      for (let r = 0; r < m; r++) zHat[r] += wm[i] * d[r];
    }
    for (let r = 0; r < m; r++) zHat[r] += Z[0][r];

    let S = zerosMat(m, m);
    let Pxz = zerosMat(3, m);
    for (let i = 0; i < points.length; i++) {
      const dz = residual(Z[i], zHat);
      const dx = boxminus(points[i], this.mean) as unknown as Vec;
      S = matAdd(S, matScale(outer(dz, dz), wc[i]));
      Pxz = matAdd(Pxz, matScale(outer(dx, dz), wc[i]));
    }
    S = symmetrize(matAdd(S, Q));

    const K = matMul(Pxz, inv(S));
    const y = residual(z, zHat);
    const delta = matVec(K, y);
    this.mean = boxplus(this.mean, [delta[0], delta[1], delta[2]]);
    this.cov = symmetrize(matSub(this.cov, matMul(matMul(K, S), transpose(K))));
    return { innovation: y, S, K };
  }

  belief(): Se2Gaussian {
    return { mean: { ...this.mean }, cov: this.cov.map((r) => r.slice()) };
  }
}

/* -------------------------------------------------------------------------- */
/* Diagnostics                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Normalized estimation error squared for an on-manifold Gaussian:
 * εᵀ Σ⁻¹ ε with ε = x ⊟ μ.
 *
 * A consistent 3-DOF filter averages 3. Chapter 6 introduced NEES for vector
 * states; the only change here is that the error comes from ⊟ rather than −,
 * which is exactly the change this chapter is about.
 */
export function se2Nees(truth: Pose2, belief: Se2Gaussian): number {
  const e = boxminus(truth, belief.mean);
  const Se = matVec(inv(belief.cov), e as unknown as Vec);
  return e[0] * Se[0] + e[1] * Se[1] + e[2] * Se[2];
}
