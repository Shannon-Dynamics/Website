/**
 * Filter consistency metrics — NEES, NIS, and their χ² envelopes.
 *
 * A filter is *consistent* when its claimed covariance matches the errors it
 * actually makes. RMSE cannot see this: an overconfident filter can have
 * excellent RMSE right up to the moment it diverges. NEES and NIS can, and they
 * are the standard instruments for it (Bar-Shalom, Li & Kirubarajan, 2001, §5.4).
 *
 *   NEES  (x_t − μ_t)ᵀ Σ_t⁻¹ (x_t − μ_t)   needs ground truth → simulation only
 *   NIS   (z_t − H μ̄_t)ᵀ S_t⁻¹ (z_t − H μ̄_t)   needs only data → works on a real robot
 *
 * Under a correct linear-Gaussian model both are χ² distributed — NEES with n
 * degrees of freedom, NIS with m — so "is my filter lying?" becomes a
 * hypothesis test with a number attached.
 */

import { cholesky, dot, sub, type Mat, type Vec } from '../prob/linalg';
import type { Gaussian } from './kf';

/** Squared Mahalanobis distance via the Cholesky factor: yᵀy with L y = d. */
function squaredMahalanobis(d: Vec, cov: Mat): number {
  const l = cholesky(cov);
  const n = d.length;
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = d[i];
    for (let j = 0; j < i; j++) s -= l[i][j] * y[j];
    y[i] = s / l[i][i];
  }
  return Math.max(dot(y, y), 0);
}

/**
 * Normalized estimation error squared. Expectation is `n` for a consistent
 * filter: a run averaging well above it is overconfident, well below it
 * conservative.
 */
export function nees(truth: Vec, belief: Gaussian): number {
  return squaredMahalanobis(sub(truth, belief.x), belief.P);
}

/**
 * Normalized innovation squared. Expectation is `m`. Unlike NEES this needs no
 * ground truth, which is why it is the one a deployed robot can actually run.
 */
export function nis(innovation: Vec, S: Mat): number {
  return squaredMahalanobis(innovation, S);
}

/* -------------------------------------------------------------------------- */
/* χ² tails                                                                    */
/* -------------------------------------------------------------------------- */

/** Lanczos log Γ(x), g = 7, n = 9 — accurate to ~1e-13 for x > 0. */
function logGamma(x: number): number {
  const g = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    // Reflection, so the series is only ever evaluated on its good half.
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  const z = x - 1;
  let a = g[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += g[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Regularized lower incomplete gamma P(a, x) — series below the crossover,
 * Lentz's continued fraction for Q(a, x) above it. Numerical Recipes §6.2.
 */
function gammaP(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let i = 1; i < 400; i++) {
      term *= x / (a + i);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
  }

  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 400; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-15) break;
  }
  const q = Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  return 1 - q;
}

/** CDF of the χ² distribution with `dof` degrees of freedom. */
export function chi2Cdf(x: number, dof: number): number {
  return gammaP(dof / 2, x / 2);
}

/** Quantile of the χ² distribution, by bisection on the CDF. */
export function chi2Quantile(p: number, dof: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return Number.POSITIVE_INFINITY;
  let lo = 0;
  let hi = dof + 10 * Math.sqrt(2 * dof) + 20;
  while (chi2Cdf(hi, dof) < p) hi *= 2;
  // 80 halvings takes the bracket below double precision; more is wasted work
  // in a widget that recomputes this on every animation frame.
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (chi2Cdf(mid, dof) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * The two-sided acceptance region a consistency test plots as a band.
 *
 * Averaging `samples` independent statistics of dimension `dim` gives a χ² with
 * `samples · dim` degrees of freedom, divided by `samples` — which is why the
 * envelope tightens around `dim` as a run gets longer, and why a single-step
 * NIS bouncing outside the band means nothing on its own.
 */
export function chi2Envelope(dim: number, samples = 1, alpha = 0.05) {
  const dof = Math.max(dim * samples, 1);
  return {
    dof,
    lo: chi2Quantile(alpha / 2, dof) / samples,
    hi: chi2Quantile(1 - alpha / 2, dof) / samples,
    expected: dim,
  };
}

/** Root-mean-square of a sequence of scalar errors. */
export function rmse(errors: readonly number[]): number {
  if (errors.length === 0) return 0;
  let s = 0;
  for (const e of errors) s += e * e;
  return Math.sqrt(s / errors.length);
}
