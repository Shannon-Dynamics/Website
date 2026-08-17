/**
 * The Kalman filter — Thrun et al., **Table 3.1** — and the RTS smoother.
 *
 * Exact for linear-Gaussian systems, and the reference implementation the EKF,
 * UKF, and information filter are all approximating.
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
import type { BayesFilter } from './bayes';

export interface Gaussian {
  x: Vec;
  P: Mat;
}

/** What `update` reports back, for gating, NEES plots, and innovation charts. */
export interface KfUpdateInfo {
  innovation: Vec;
  S: Mat;
  K: Mat;
  logLikelihood: number;
}

export class Kf implements BayesFilter<Gaussian, { F: Mat; Q: Mat; B?: Mat; u?: Vec }, Vec> {
  x: Vec;
  P: Mat;
  /** Set by `update`; the H and R of the pending measurement model. */
  private lastH: Mat | null = null;
  private lastR: Mat | null = null;

  constructor(x: Vec, P: Mat) {
    this.x = x.slice();
    this.P = P.map((r) => r.slice());
  }

  /**
   * x̄ = F x + B u,  P̄ = F P Fᵀ + Q
   *
   * The covariance term is the one to read twice: F P Fᵀ is the old uncertainty
   * *dragged through* the dynamics (it can shrink, if the system is stable),
   * and +Q is the new uncertainty the step injects. Prediction always ends up
   * less certain than it started only because Q is positive definite.
   */
  predictWith(F: Mat, Q: Mat, B?: Mat | null, u?: Vec | null): void {
    let xNew = matVec(F, this.x);
    if (B && u) xNew = add(xNew, matVec(B, u));
    this.x = xNew;
    this.P = symmetrize(matAdd(matMul(matMul(F, this.P), transpose(F)), Q));
  }

  /** `BayesFilter` shape: `predict({F, Q, B, u})`. */
  predict(cmd: { F: Mat; Q: Mat; B?: Mat; u?: Vec }): void {
    this.predictWith(cmd.F, cmd.Q, cmd.B, cmd.u);
  }

  /**
   * Measurement update with the **Joseph form** covariance:
   *
   *   P = (I − K H) P (I − K H)ᵀ + K R Kᵀ
   *
   * Algebraically identical to (I − K H)P *when K is the optimal gain*, but it
   * stays symmetric and positive-semidefinite under round-off and under a
   * suboptimal K. Since these filters run in float64 inside a browser loop for
   * thousands of steps, that robustness is worth the extra matrix product.
   */
  updateWith(z: Vec, H: Mat, R: Mat): KfUpdateInfo {
    const Ht = transpose(H);
    const y = sub(z, matVec(H, this.x));
    const S = symmetrize(matAdd(matMul(matMul(H, this.P), Ht), R));
    const K = matMul(matMul(this.P, Ht), inv(S));

    this.x = add(this.x, matVec(K, y));

    const I = eye(this.x.length);
    const IKH = matSub(I, matMul(K, H));
    this.P = symmetrize(
      matAdd(
        matMul(matMul(IKH, this.P), transpose(IKH)),
        matMul(matMul(K, R), transpose(K)),
      ),
    );

    this.lastH = H;
    this.lastR = R;
    return {
      innovation: y,
      S,
      K,
      logLikelihood: logMvnPdf(y, new Array<number>(y.length).fill(0), S),
    };
  }

  /** `BayesFilter` shape — requires a prior `setMeasurementModel`. */
  correct(z: Vec): void {
    if (!this.lastH || !this.lastR) {
      throw new Error('Kf.correct: call setMeasurementModel(H, R) or updateWith(z, H, R)');
    }
    this.updateWith(z, this.lastH, this.lastR);
  }

  setMeasurementModel(H: Mat, R: Mat): void {
    this.lastH = H;
    this.lastR = R;
  }

  belief(): Gaussian {
    return { x: this.x.slice(), P: this.P.map((r) => r.slice()) };
  }

  clone(): Kf {
    return new Kf(this.x, this.P);
  }
}

/** One time step of a filter run, as the smoother needs to see it. */
export interface KfRecord {
  /** Predicted (prior) belief at this step, before the measurement. */
  xPrior: Vec;
  PPrior: Mat;
  /** Corrected (posterior) belief at this step. */
  xPost: Vec;
  PPost: Mat;
  /** Transition matrix that produced this step's prior from the previous posterior. */
  F: Mat;
}

/**
 * Rauch–Tung–Striebel fixed-interval smoother.
 *
 * Runs backwards over a completed filter trajectory, letting later evidence
 * inform earlier estimates:
 *
 *   C_k   = P_k^post F_{k+1}ᵀ (P_{k+1}^prior)⁻¹
 *   x_k^s = x_k^post + C_k (x_{k+1}^s − x_{k+1}^prior)
 *   P_k^s = P_k^post + C_k (P_{k+1}^s − P_{k+1}^prior) C_kᵀ
 *
 * The middle line says it all: correct each estimate by how wrong its own
 * prediction of the future turned out to be. Smoothed covariances are never
 * larger than filtered ones — the widget plots both bands to show the gap.
 */
export function rtsSmoother(states: KfRecord[]): Gaussian[] {
  const n = states.length;
  if (n === 0) return [];
  const out: Gaussian[] = states.map((s) => ({
    x: s.xPost.slice(),
    P: s.PPost.map((r) => r.slice()),
  }));

  for (let k = n - 2; k >= 0; k--) {
    const next = states[k + 1];
    const C = matMul(
      matMul(states[k].PPost, transpose(next.F)),
      inv(next.PPrior),
    );
    out[k].x = add(out[k].x, matVec(C, sub(out[k + 1].x, next.xPrior)));
    out[k].P = symmetrize(
      matAdd(
        states[k].PPost,
        matMul(matMul(C, matSub(out[k + 1].P, next.PPrior)), transpose(C)),
      ),
    );
  }
  return out;
}
