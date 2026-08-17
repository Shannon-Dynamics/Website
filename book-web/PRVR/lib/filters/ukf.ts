/**
 * The unscented Kalman filter — Thrun et al., **Table 3.4**.
 *
 * The EKF linearises the *function*; the UKF linearises nothing and instead
 * samples the *distribution* at 2n+1 deterministically chosen points, pushes
 * them through the true nonlinearity, and refits a Gaussian. It is exact to
 * third order for Gaussian inputs (the EKF only to first), needs no Jacobians,
 * and costs about the same.
 */

import {
  cholesky,
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
import type { Gaussian } from './kf';

export interface UkfParams {
  /** Spread of the sigma points around the mean. */
  alpha: number;
  /** Prior knowledge of the distribution; 2 is optimal for a Gaussian. */
  beta: number;
  /** Secondary scaling; defaults to 3 − n, the classic Julier choice. */
  kappa: number;
}

export interface SigmaPoints {
  points: Vec[];
  /** Weights for reconstructing the mean. */
  wm: number[];
  /** Weights for reconstructing the covariance (wc[0] differs by (1−α²+β)). */
  wc: number[];
  lambda: number;
}

/**
 * Generate the 2n+1 sigma points of Julier & Uhlmann's scaled transform:
 *
 *   λ = α²(n + κ) − n
 *   X₀ = μ,  X_i = μ ± [√((n+λ) Σ)]_i
 *
 * The square root is the lower Cholesky factor, and "±column i" is what makes
 * the point cloud trace the covariance ellipse — which is exactly the picture
 * the sigma-point widget draws.
 */
export function sigmaPoints(
  mean: Vec,
  cov: Mat,
  params: Partial<UkfParams> = {},
): SigmaPoints {
  const n = mean.length;
  const alpha = params.alpha ?? 1;
  const beta = params.beta ?? 2;
  const kappa = params.kappa ?? 3 - n;
  const lambda = alpha * alpha * (n + kappa) - n;
  const c = Math.sqrt(n + lambda);
  const L = cholesky(jitter(cov));

  const points: Vec[] = [mean.slice()];
  for (let i = 0; i < n; i++) {
    const plus = mean.slice();
    const minus = mean.slice();
    for (let r = 0; r < n; r++) {
      plus[r] += c * L[r][i];
      minus[r] -= c * L[r][i];
    }
    points.push(plus);
    points.push(minus);
  }

  const wm = new Array<number>(2 * n + 1).fill(1 / (2 * (n + lambda)));
  const wc = wm.slice();
  wm[0] = lambda / (n + lambda);
  wc[0] = lambda / (n + lambda) + (1 - alpha * alpha + beta);
  return { points, wm, wc, lambda };
}

/** Nudge the diagonal so a numerically deflated covariance still factorises. */
function jitter(cov: Mat): Mat {
  return cov.map((row, i) => row.map((v, j) => (i === j ? v + 1e-12 : v)));
}

export interface UtResult {
  mean: Vec;
  cov: Mat;
  /** The input sigma points — keep them to draw the "before" cloud. */
  points: Vec[];
  /** Their images under `fn` — the "after" cloud. */
  transformed: Vec[];
  wm: number[];
  wc: number[];
}

/**
 * The unscented transform: push a Gaussian through an arbitrary function.
 *
 * Exported on its own because it is the whole idea, and because a widget wants
 * to draw the sigma points on both sides of the map to show why the refitted
 * ellipse beats the EKF's linearised one on a strongly curved function.
 *
 * `residual` lets an angular output component be averaged correctly.
 */
export function unscentedTransform(
  mean: Vec,
  cov: Mat,
  fn: (x: Vec) => Vec,
  opts: Partial<UkfParams> & { residual?: (a: Vec, b: Vec) => Vec } = {},
): UtResult {
  const { points, wm, wc } = sigmaPoints(mean, cov, opts);
  const transformed = points.map(fn);
  const residual = opts.residual ?? sub;

  const m = transformed[0].length;
  const outMean = new Array<number>(m).fill(0);
  for (let i = 0; i < points.length; i++) {
    for (let r = 0; r < m; r++) outMean[r] += wm[i] * transformed[i][r];
  }

  let outCov = zerosMat(m, m);
  for (let i = 0; i < points.length; i++) {
    const d = residual(transformed[i], outMean);
    outCov = matAdd(outCov, matScale(outer(d, d), wc[i]));
  }

  return { mean: outMean, cov: symmetrize(outCov), points, transformed, wm, wc };
}

export interface UkfUpdateOptions {
  /** Residual in measurement space (wrap bearings here). */
  residualZ?: (a: Vec, b: Vec) => Vec;
  /** Residual in state space (wrap headings here). */
  residualX?: (a: Vec, b: Vec) => Vec;
  /** State retraction for the mean correction; defaults to addition. */
  boxplus?: (x: Vec, dx: Vec) => Vec;
}

export class Ukf {
  x: Vec;
  P: Mat;
  params: UkfParams;

  constructor(x: Vec, P: Mat, params: Partial<UkfParams> = {}) {
    this.x = x.slice();
    this.P = P.map((r) => r.slice());
    this.params = {
      alpha: params.alpha ?? 1,
      beta: params.beta ?? 2,
      kappa: params.kappa ?? 3 - x.length,
    };
  }

  /** Push the sigma points through g, refit, add the process noise. */
  predict(f: (x: Vec) => Vec, Q: Mat, residualX?: (a: Vec, b: Vec) => Vec): UtResult {
    const ut = unscentedTransform(this.x, this.P, f, {
      ...this.params,
      residual: residualX,
    });
    this.x = ut.mean;
    this.P = symmetrize(matAdd(ut.cov, Q));
    return ut;
  }

  /**
   * Measurement update.
   *
   * Sigma points are regenerated from the *predicted* belief, propagated
   * through h, and used for both the innovation covariance S and the
   * cross-covariance P_xz. The gain is then K = P_xz S⁻¹ — the same shape as
   * the linear filter's P Hᵀ S⁻¹, with the cross-covariance standing in for
   * P Hᵀ. This is the cleanest way to see what the Kalman gain *is*.
   */
  update(
    z: Vec,
    h: (x: Vec) => Vec,
    R: Mat,
    opts: UkfUpdateOptions = {},
  ): { innovation: Vec; S: Mat; K: Mat; sigmaZ: Vec[] } {
    const residualZ = opts.residualZ ?? sub;
    const residualX = opts.residualX ?? sub;
    const boxplus = opts.boxplus ?? ((a: Vec, b: Vec) => a.map((v, i) => v + b[i]));

    const { points, wm, wc } = sigmaPoints(this.x, this.P, this.params);
    const Z = points.map(h);

    const m = Z[0].length;
    const zHat = new Array<number>(m).fill(0);
    for (let i = 0; i < points.length; i++) {
      for (let r = 0; r < m; r++) zHat[r] += wm[i] * Z[i][r];
    }

    let S = zerosMat(m, m);
    let Pxz = zerosMat(this.x.length, m);
    for (let i = 0; i < points.length; i++) {
      const dz = residualZ(Z[i], zHat);
      const dx = residualX(points[i], this.x);
      S = matAdd(S, matScale(outer(dz, dz), wc[i]));
      Pxz = matAdd(Pxz, matScale(outer(dx, dz), wc[i]));
    }
    S = symmetrize(matAdd(S, R));

    const K = matMul(Pxz, inv(S));
    const y = residualZ(z, zHat);
    this.x = boxplus(this.x, matVec(K, y));
    this.P = symmetrize(matSub(this.P, matMul(matMul(K, S), transpose(K))));

    return { innovation: y, S, K, sigmaZ: Z };
  }

  belief(): Gaussian {
    return { x: this.x.slice(), P: this.P.map((r) => r.slice()) };
  }
}
