/**
 * Gaussian densities, samplers, and entropy.
 *
 * These are the building blocks every filter in the book leans on. The
 * multivariate routines all go through the Cholesky factor rather than an
 * explicit inverse: it is faster, it is what the Rust side does with
 * `nalgebra`'s `Cholesky`, and it keeps `logMvnPdf` numerically honest for the
 * near-singular covariances an over-confident filter produces.
 */

import { cholesky, dot, sub, type Mat, type Vec } from './linalg';
import type { Rng } from './rng';

export const LOG_2PI = Math.log(2 * Math.PI);

/** Smallest variance we will divide by — keeps degenerate models finite. */
const VAR_FLOOR = 1e-12;

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

/** N(x; mean, std²). */
export function normalPdf(x: number, mean = 0, std = 1): number {
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
}

/** log N(x; mean, std²) — use this inside particle filters, not `log(normalPdf)`. */
export function logNormalPdf(x: number, mean = 0, std = 1): number {
  const z = (x - mean) / std;
  return -0.5 * z * z - Math.log(std) - 0.5 * LOG_2PI;
}

/**
 * Thrun's `prob(a, b²)`: the density of a zero-centred normal with **variance**
 * `b²` evaluated at `a`. Every motion and sensor model in Chapters 5–6 is
 * written in terms of it, so it is spelled the same way here.
 *
 * Thrun et al., *Probabilistic Robotics*, Table 5.2.
 */
export function prob(a: number, variance: number): number {
  const v = Math.max(variance, VAR_FLOOR);
  return Math.exp(-0.5 * (a * a) / v) / Math.sqrt(2 * Math.PI * v);
}

/** log of {@link prob}. */
export function logProb(a: number, variance: number): number {
  const v = Math.max(variance, VAR_FLOOR);
  return -0.5 * (a * a) / v - 0.5 * Math.log(2 * Math.PI * v);
}

/** Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, plenty for a 3-digit plot. */
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
    t;
  return sign * (1 - poly * Math.exp(-ax * ax));
}

/** Φ((x − mean)/std). */
export function normalCdf(x: number, mean = 0, std = 1): number {
  return 0.5 * (1 + erf((x - mean) / (std * Math.SQRT2)));
}

/**
 * Fusing two 1-D Gaussians — the worked example that motivates the Kalman
 * filter. The product of two normals is an unnormalised normal whose precision
 * is the sum of the precisions:
 *
 *   1/σ² = 1/v₁ + 1/v₂,   μ = σ² (m₁/v₁ + m₂/v₂)
 *
 * which is the algebra below in the numerically friendlier "weighted average"
 * form. Note the posterior variance is smaller than *either* input: two
 * independent opinions always beat one.
 */
export function gaussianProduct(
  m1: number,
  v1: number,
  m2: number,
  v2: number,
): { mean: number; variance: number } {
  const s = Math.max(v1 + v2, VAR_FLOOR);
  return { mean: (v2 * m1 + v1 * m2) / s, variance: (v1 * v2) / s };
}

// ---------------------------------------------------------------------------
// Multivariate
// ---------------------------------------------------------------------------

/** Solve L y = b for lower-triangular L (forward substitution). */
function forwardSub(l: Mat, b: Vec): Vec {
  const n = b.length;
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) s -= l[i][j] * y[j];
    y[i] = s / l[i][i];
  }
  return y;
}

/** log N(x; mean, cov) via the Cholesky factor: −½(yᵀy + log|Σ| + k log 2π). */
export function logMvnPdf(x: Vec, mean: Vec, cov: Mat): number {
  const k = x.length;
  const l = cholesky(cov);
  const y = forwardSub(l, sub(x, mean));
  let logDet = 0;
  for (let i = 0; i < k; i++) logDet += 2 * Math.log(l[i][i]);
  return -0.5 * (dot(y, y) + logDet + k * LOG_2PI);
}

/** N(x; mean, cov). */
export function mvnPdf(x: Vec, mean: Vec, cov: Mat): number {
  return Math.exp(logMvnPdf(x, mean, cov));
}

/**
 * Mahalanobis *distance* √((x−μ)ᵀ Σ⁻¹ (x−μ)).
 *
 * Squaring it gives the χ² statistic used for gating in Chapter 7 — a 2-D
 * association gate at 95% is `mahalanobis(...)² < 5.99`.
 */
export function mahalanobis(x: Vec, mean: Vec, cov: Mat): number {
  const y = forwardSub(cholesky(cov), sub(x, mean));
  return Math.sqrt(Math.max(dot(y, y), 0));
}

/** Draw x ~ N(mean, cov) as mean + L z with L Lᵀ = cov and z ~ N(0, I). */
export function sampleMvn(mean: Vec, cov: Mat, rng: Rng): Vec {
  const n = mean.length;
  const l = cholesky(cov);
  const z = Array.from({ length: n }, () => rng.normal());
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = mean[i];
    for (let j = 0; j <= i; j++) s += l[i][j] * z[j];
    out[i] = s;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

export const natsToBits = (nats: number): number => nats / Math.LN2;

/** Differential entropy of N(μ, σ²) in **nats**: ½ log(2πe σ²). */
export function gaussianEntropy(std: number): number {
  return 0.5 * Math.log(2 * Math.PI * Math.E * std * std);
}

/** Differential entropy of N(μ, Σ) in **nats**: ½ log((2πe)^k |Σ|). */
export function mvnEntropy(cov: Mat): number {
  const k = cov.length;
  const l = cholesky(cov);
  let logDet = 0;
  for (let i = 0; i < k; i++) logDet += 2 * Math.log(l[i][i]);
  return 0.5 * (k * Math.log(2 * Math.PI * Math.E) + logDet);
}

/**
 * Shannon entropy of a discrete distribution in **bits**.
 *
 * This is the number the histogram-filter widget plots as the reader watches a
 * belief collapse: log₂ n bits when uniform over n cells, 0 when certain.
 * Input is normalised defensively so an un-normalised belief still reads out
 * sensibly.
 */
export function discreteEntropy(p: number[]): number {
  let total = 0;
  for (const v of p) if (v > 0) total += v;
  if (total <= 0) return 0;
  let h = 0;
  for (const v of p) {
    if (v <= 0) continue;
    const q = v / total;
    h -= q * Math.log2(q);
  }
  return h;
}

/** Binary entropy H(p) in bits — one occupancy-grid cell's worth of ignorance. */
export function binaryEntropy(p: number): number {
  if (p <= 0 || p >= 1) return 0;
  return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

/** log Σ exp(xs) without overflowing — the safe way to normalise log-weights. */
export function logSumExp(xs: number[]): number {
  let max = -Infinity;
  for (const x of xs) if (x > max) max = x;
  if (!Number.isFinite(max)) return max;
  let s = 0;
  for (const x of xs) s += Math.exp(x - max);
  return max + Math.log(s);
}
