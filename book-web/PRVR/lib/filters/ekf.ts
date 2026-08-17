/**
 * The extended Kalman filter — Thrun et al., **Table 3.3**.
 *
 * The EKF is the KF with two substitutions: propagate the mean through the
 * *nonlinear* g and h, and propagate the covariance through their Jacobians.
 * Everything that can go wrong with an EKF comes from that swap — the Jacobian
 * is only honest near the linearisation point, so a wide covariance or a sharp
 * nonlinearity makes the reported uncertainty a fiction.
 *
 * Two hooks matter for robotics:
 *
 *   `boxplus`  — the state may live on a manifold (a `Pose2`, packed into a
 *                Vec), where "x + K y" is wrong for the heading component.
 *   `residual` — the innovation z − h(x) must wrap for bearing measurements.
 */

import {
  add,
  eye,
  inv,
  matAdd,
  matMul,
  matSub,
  matVec,
  sub,
  symmetrize,
  transpose,
  type Mat,
  type Vec,
} from '../prob/linalg';
import { logMvnPdf } from '../prob/gaussian';
import type { Gaussian, KfUpdateInfo } from './kf';

export interface EkfOptions {
  /** State retraction; defaults to vector addition. */
  boxplus?: (x: Vec, dx: Vec) => Vec;
}

export interface EkfUpdateOptions {
  /** Measurement residual; defaults to plain subtraction. Wrap angles here. */
  residual?: (z: Vec, zPred: Vec) => Vec;
}

export class Ekf {
  x: Vec;
  P: Mat;
  private boxplus: (x: Vec, dx: Vec) => Vec;

  constructor(x: Vec, P: Mat, opts: EkfOptions = {}) {
    this.x = x.slice();
    this.P = P.map((r) => r.slice());
    this.boxplus = opts.boxplus ?? add;
  }

  /**
   * x̄ = g(x, u),  P̄ = G P Gᵀ + Q
   *
   * `FJac` may be a constant matrix or a function of the current state, which
   * is the common case: the motion Jacobian of a differential drive depends on
   * the heading it is linearised about.
   */
  predict(f: (x: Vec) => Vec, FJac: Mat | ((x: Vec) => Mat), Q: Mat): void {
    const G = typeof FJac === 'function' ? FJac(this.x) : FJac;
    this.x = f(this.x);
    this.P = symmetrize(matAdd(matMul(matMul(G, this.P), transpose(G)), Q));
  }

  /**
   * Measurement update, Joseph form (see `Kf.updateWith` for why).
   *
   * The mean correction goes through ⊞ so that an on-manifold state stays on
   * the manifold: for a pose state the Kalman gain produces a *tangent* vector,
   * not a new pose.
   */
  update(
    z: Vec,
    h: (x: Vec) => Vec,
    HJac: Mat | ((x: Vec) => Mat),
    R: Mat,
    opts: EkfUpdateOptions = {},
  ): KfUpdateInfo {
    const H = typeof HJac === 'function' ? HJac(this.x) : HJac;
    const Ht = transpose(H);
    const residual = opts.residual ?? sub;

    const y = residual(z, h(this.x));
    const S = symmetrize(matAdd(matMul(matMul(H, this.P), Ht), R));
    const K = matMul(matMul(this.P, Ht), inv(S));

    this.x = this.boxplus(this.x, matVec(K, y));

    const IKH = matSub(eye(this.x.length), matMul(K, H));
    this.P = symmetrize(
      matAdd(
        matMul(matMul(IKH, this.P), transpose(IKH)),
        matMul(matMul(K, R), transpose(K)),
      ),
    );

    return {
      innovation: y,
      S,
      K,
      logLikelihood: logMvnPdf(y, new Array<number>(y.length).fill(0), S),
    };
  }

  belief(): Gaussian {
    return { x: this.x.slice(), P: this.P.map((r) => r.slice()) };
  }
}

/**
 * Central-difference Jacobian — a safety net for readers experimenting with
 * their own models, and the thing to compare an analytic Jacobian against when
 * a filter diverges for no visible reason.
 */
export function numericJacobian(
  f: (x: Vec) => Vec,
  x: Vec,
  eps = 1e-6,
): Mat {
  const f0 = f(x);
  const m = f0.length;
  const n = x.length;
  const J: Mat = Array.from({ length: m }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    const xp = x.slice();
    const xm = x.slice();
    xp[j] += eps;
    xm[j] -= eps;
    const fp = f(xp);
    const fm = f(xm);
    for (let i = 0; i < m; i++) J[i][j] = (fp[i] - fm[i]) / (2 * eps);
  }
  return J;
}
