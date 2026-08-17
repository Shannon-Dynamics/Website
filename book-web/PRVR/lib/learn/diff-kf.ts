/**
 * The Kalman filter as a differentiable program — Chapter 25, D25.5.
 *
 * The scalar filter of Chapter 6 with its two noise parameters exposed as
 * `θ = (log r, log q)` and *differentiated*. Nothing about the filter changes:
 * predict, innovate, gain, update, exactly as in `lib/filters/kf.ts`. What is
 * added is a second recursion carried alongside the first, propagating
 * `∂μ/∂θ` and `∂σ²/∂θ` through the same five equations.
 *
 * Two things are worth saying about the implementation choice. First, the
 * parameters are stored as *logs* so that gradient descent can never produce a
 * negative variance — the constraint is enforced by the parameterization rather
 * than by clipping. Second, with only two parameters, forward-mode sensitivity
 * propagation is cheaper than reverse-mode BPTT and gives bit-identical
 * gradients; the widget runs forward mode, and the book's Rust cross-checks it
 * against both finite differences and `candle`'s autodiff.
 */

import type { Rng } from '../prob/rng';

/** A logged run of a scalar linear-Gaussian system. */
export interface TrajLog1d {
  /** x_t = a·x_{t-1} + b·u_t + w_t */
  a: number;
  b: number;
  /** Controls, one per step. */
  u: number[];
  /** Measurements z_t = x_t + v_t. */
  z: number[];
  /** Ground truth, when there is any. The evidence loss never looks at it. */
  x: number[];
  /** Initial belief — known, and *not* trained. */
  mu0: number;
  sigma0: number;
  /** The noises that actually generated the log, for scoring the trainer. */
  trueR: number;
  trueQ: number;
}

export type DkfLoss = 'evidence' | 'stateNll';

export interface DkfGrad {
  loss: number;
  dLogR: number;
  dLogQ: number;
}

/** One step of the filter, kept so the widget can draw what the trainer sees. */
export interface DkfStep {
  muBar: number;
  sigma2Bar: number;
  /** Innovation ν_t = z_t − μ̄_t. */
  nu: number;
  /** Innovation covariance S_t = σ̄²_t + q. */
  S: number;
  K: number;
  mu: number;
  sigma2: number;
}

export class DiffKf1d {
  logR: number;
  logQ: number;

  constructor(logR: number, logQ: number) {
    this.logR = logR;
    this.logQ = logQ;
  }

  get r(): number {
    return Math.exp(this.logR);
  }

  get q(): number {
    return Math.exp(this.logQ);
  }

  /** Run the filter over a log, returning every intermediate quantity. */
  run(log: TrajLog1d): DkfStep[] {
    const r = this.r;
    const q = this.q;
    let mu = log.mu0;
    let sigma2 = log.sigma0 * log.sigma0;
    const out: DkfStep[] = [];

    for (let t = 0; t < log.z.length; t++) {
      const muBar = log.a * mu + log.b * (log.u[t] ?? 0);
      const sigma2Bar = log.a * log.a * sigma2 + r;
      const nu = log.z[t] - muBar;
      const S = sigma2Bar + q;
      const K = sigma2Bar / S;
      mu = muBar + K * nu;
      sigma2 = (1 - K) * sigma2Bar;
      out.push({ muBar, sigma2Bar, nu, S, K, mu, sigma2 });
    }
    return out;
  }

  /**
   * Loss and its exact gradient in one forward pass.
   *
   * **Evidence loss** (F4) — the prediction-error decomposition:
   *
   *   L = ½ Σ_t [ log 2π S_t + ν_t² / S_t ]
   *
   * It needs no ground truth, because every term is a one-step-ahead
   * predictive density of a quantity the robot actually observed.
   *
   * **State NLL** — the supervised alternative, available only in simulation
   * or with a motion-capture rig:
   *
   *   L = ½ Σ_t [ log 2π σ²_t + (x_t − μ_t)² / σ²_t ]
   *
   * The sensitivity recursion (F5), for each parameter separately:
   *
   *   dσ̄² = a² dσ² + [r if ∂/∂log r]
   *   dμ̄  = a dμ,            dν = −dμ̄
   *   dS  = dσ̄² + [q if ∂/∂log q]
   *   dK  = (dσ̄² − K dS) / S
   *   dμ  = dμ̄ + dK ν + K dν
   *   dσ² = (1 − K) dσ̄² − dK σ̄²
   */
  lossAndGrad(log: TrajLog1d, loss: DkfLoss = 'evidence'): DkfGrad {
    const r = this.r;
    const q = this.q;
    const { a, b } = log;

    let mu = log.mu0;
    let sigma2 = log.sigma0 * log.sigma0;
    // Sensitivities of (μ, σ²) with respect to log r and log q. The initial
    // belief is given, not learned, so both start at zero.
    let dMu = [0, 0];
    let dSig = [0, 0];

    let total = 0;
    const grad = [0, 0];

    for (let t = 0; t < log.z.length; t++) {
      const u = log.u[t] ?? 0;

      // ---- predict, and its derivative -------------------------------
      const muBar = a * mu + b * u;
      const sigma2Bar = a * a * sigma2 + r;
      const dMuBar = [a * dMu[0], a * dMu[1]];
      // ∂r/∂log r = r; ∂r/∂log q = 0.
      const dSigBar = [a * a * dSig[0] + r, a * a * dSig[1]];

      // ---- innovate ---------------------------------------------------
      const nu = log.z[t] - muBar;
      const S = sigma2Bar + q;
      const dNu = [-dMuBar[0], -dMuBar[1]];
      // ∂q/∂log q = q; ∂q/∂log r = 0.
      const dS = [dSigBar[0], dSigBar[1] + q];

      // ---- gain and update -------------------------------------------
      const K = sigma2Bar / S;
      const dK = [(dSigBar[0] - K * dS[0]) / S, (dSigBar[1] - K * dS[1]) / S];

      const muNew = muBar + K * nu;
      const sigma2New = (1 - K) * sigma2Bar;
      const dMuNew = [
        dMuBar[0] + dK[0] * nu + K * dNu[0],
        dMuBar[1] + dK[1] * nu + K * dNu[1],
      ];
      const dSigNew = [
        (1 - K) * dSigBar[0] - dK[0] * sigma2Bar,
        (1 - K) * dSigBar[1] - dK[1] * sigma2Bar,
      ];

      // ---- accumulate the loss ---------------------------------------
      if (loss === 'evidence') {
        total += 0.5 * (Math.log(2 * Math.PI * S) + (nu * nu) / S);
        for (let p = 0; p < 2; p++) {
          grad[p] +=
            0.5 * ((1 - (nu * nu) / S) * (dS[p] / S) + (2 * nu * dNu[p]) / S);
        }
      } else {
        const e = (log.x[t] ?? 0) - muNew;
        total += 0.5 * (Math.log(2 * Math.PI * sigma2New) + (e * e) / sigma2New);
        for (let p = 0; p < 2; p++) {
          grad[p] +=
            0.5 *
            ((1 - (e * e) / sigma2New) * (dSigNew[p] / sigma2New) -
              (2 * e * dMuNew[p]) / sigma2New);
        }
      }

      mu = muNew;
      sigma2 = sigma2New;
      dMu = dMuNew;
      dSig = dSigNew;
    }

    return { loss: total, dLogR: grad[0], dLogQ: grad[1] };
  }

  /** One gradient-descent step on θ = (log r, log q). Returns the loss *before* the step. */
  sgdStep(log: TrajLog1d, loss: DkfLoss, lr: number): DkfGrad {
    const g = this.lossAndGrad(log, loss);
    // A trust region on the parameter, not the gradient: a single step may not
    // move a variance by more than e^0.5 ≈ 65%, which keeps the trainer stable
    // at learning rates a reader is likely to try.
    const clip = (d: number) => Math.max(-0.5, Math.min(0.5, d));
    this.logR -= clip(lr * g.dLogR);
    this.logQ -= clip(lr * g.dLogQ);
    return g;
  }

  clone(): DiffKf1d {
    return new DiffKf1d(this.logR, this.logQ);
  }
}

/**
 * Central-difference gradient — the referee the analytic gradient is checked
 * against, both in `__checks_ch25__` and in the book's Rust test.
 */
export function finiteDiffGrad(
  kf: DiffKf1d,
  log: TrajLog1d,
  loss: DkfLoss = 'evidence',
  h = 1e-5,
): { dLogR: number; dLogQ: number } {
  const at = (dr: number, dq: number) =>
    new DiffKf1d(kf.logR + dr, kf.logQ + dq).lossAndGrad(log, loss).loss;
  return {
    dLogR: (at(h, 0) - at(-h, 0)) / (2 * h),
    dLogQ: (at(0, h) - at(0, -h)) / (2 * h),
  };
}

/** Normalized estimation error squared, averaged over the run — 1 when honest. */
export function meanNees(kf: DiffKf1d, log: TrajLog1d): number {
  const steps = kf.run(log);
  let sum = 0;
  for (let t = 0; t < steps.length; t++) {
    const e = (log.x[t] ?? 0) - steps[t].mu;
    sum += (e * e) / Math.max(steps[t].sigma2, 1e-12);
  }
  return steps.length > 0 ? sum / steps.length : Number.NaN;
}

/** Normalized innovation squared, averaged — the ground-truth-free twin of NEES. */
export function meanNis(kf: DiffKf1d, log: TrajLog1d): number {
  const steps = kf.run(log);
  let sum = 0;
  for (const s of steps) sum += (s.nu * s.nu) / Math.max(s.S, 1e-12);
  return steps.length > 0 ? sum / steps.length : Number.NaN;
}

export interface Traj1dOptions {
  steps: number;
  a: number;
  b: number;
  r: number;
  q: number;
  /** Control sequence; a constant drift by default. */
  control?: (t: number) => number;
  x0?: number;
  mu0?: number;
  sigma0?: number;
}

/**
 * Generate a logged trajectory of Rusty's cart on its rail — the training set
 * the trainer widget descends on. Seeded, so a reader who types the seed back
 * gets the same log and the same learned parameters.
 */
export function simulateTraj1d(opts: Traj1dOptions, rng: Rng): TrajLog1d {
  const { steps, a, b, r, q } = opts;
  const control = opts.control ?? (() => 1);
  let x = opts.x0 ?? 0;
  const u: number[] = [];
  const z: number[] = [];
  const xs: number[] = [];

  for (let t = 0; t < steps; t++) {
    const ut = control(t);
    x = a * x + b * ut + rng.normal(0, Math.sqrt(r));
    u.push(ut);
    xs.push(x);
    z.push(x + rng.normal(0, Math.sqrt(q)));
  }

  return {
    a,
    b,
    u,
    z,
    x: xs,
    mu0: opts.mu0 ?? opts.x0 ?? 0,
    sigma0: opts.sigma0 ?? 0.5,
    trueR: r,
    trueQ: q,
  };
}
