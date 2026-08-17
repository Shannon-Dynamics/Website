/**
 * Partitioned Gaussians: marginals, conditionals, and the ellipse geometry
 * that draws them.
 *
 * Split a jointly Gaussian vector into blocks (x_a, x_b). Two different
 * questions get two different answers, and confusing them is the single most
 * common error in a first estimation course:
 *
 *   · *marginalize* x_b out — "squash the cloud flat" — and x_a keeps its own
 *     mean and its own covariance block. Nothing else survives.
 *   · *condition* on x_b = β — "slice the cloud" — and x_a's mean moves toward
 *     the slice while its covariance shrinks by the Schur complement, by an
 *     amount that does not depend on β at all.
 *
 * The Schur complement Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba introduced here is the same
 * matrix that reappears as the Kalman gain (Chapter 6) and as the fill-in of
 * marginalizing a landmark out of a factor graph (Chapter 15).
 *
 * Introduced in Chapter 2 (Derivation 6).
 */

import { inv, matMul, matSub, matVec, symmetrize, transpose, type Mat, type Vec } from './linalg';
import type { Moments } from './canonical';

/** Pick rows `idx` out of a vector. */
const pickVec = (v: Vec, idx: number[]): Vec => idx.map((i) => v[i]);

/** Pick the (rows, cols) sub-block of a matrix. */
const pickMat = (m: Mat, rows: number[], cols: number[]): Mat =>
  rows.map((i) => cols.map((j) => m[i][j]));

/** Indices of `n` dimensions that are not in `idx`, in ascending order. */
const complement = (n: number, idx: number[]): number[] =>
  Array.from({ length: n }, (_, i) => i).filter((i) => !idx.includes(i));

/**
 * Build a 2×2 covariance from two standard deviations and a correlation.
 *
 * Σ = [[σₐ², ρ σₐ σ_b], [ρ σₐ σ_b, σ_b²]] — the parameterization the widgets
 * expose, because ρ is the one number a reader can reason about directly.
 */
export function covFromCorrelation(sa: number, sb: number, rho: number): Mat {
  const c = rho * sa * sb;
  return [
    [sa * sa, c],
    [c, sb * sb],
  ];
}

/** ρ = Σ_ab / (σₐ σ_b), read back out of a 2×2 covariance. */
export function correlationOf(cov: Mat): number {
  const denom = Math.sqrt(Math.max(cov[0][0] * cov[1][1], 1e-18));
  return cov[0][1] / denom;
}

/**
 * The marginal over the kept dimensions: **copy the blocks out and stop**.
 *
 * There is no arithmetic here, and that is the point — marginalizing is free
 * in moments form and expensive in canonical form, which is exactly the
 * reverse of multiplying (see `canonical.ts`).
 */
export function marginalize(mean: Vec, cov: Mat, keep: number[]): Moments {
  return { mean: pickVec(mean, keep), cov: pickMat(cov, keep, keep) };
}

/**
 * The conditional p(x_a | x_b = values), by the Schur complement:
 *
 *   μ_{a|b} = μ_a + Σ_ab Σ_bb⁻¹ (values − μ_b)
 *   Σ_{a|b} = Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba
 *
 * Note what each line does and does not depend on: the mean tracks the value
 * observed, the covariance does not. Learning *that* x_b was measured shrinks
 * the uncertainty; learning *what* it measured only moves the estimate.
 */
export function conditionOn(mean: Vec, cov: Mat, givenIdx: number[], values: Vec): Moments {
  const keep = complement(mean.length, givenIdx);
  const muA = pickVec(mean, keep);
  const muB = pickVec(mean, givenIdx);
  const sAA = pickMat(cov, keep, keep);
  const sAB = pickMat(cov, keep, givenIdx);
  const sBBinv = inv(pickMat(cov, givenIdx, givenIdx));

  const gain = matMul(sAB, sBBinv); // Σ_ab Σ_bb⁻¹ — the Kalman gain in disguise
  const innovation = values.map((v, i) => v - muB[i]);
  const shift = matVec(gain, innovation);

  return {
    mean: muA.map((m, i) => m + shift[i]),
    cov: symmetrize(matSub(sAA, matMul(gain, transpose(sAB)))),
  };
}

/**
 * Scalar specialization of {@link conditionOn} for a 2-D joint: the density of
 * x_a once x_b has been pinned to `beta`.
 *
 * σ²_{a|b} = σₐ²(1 − ρ²) — the conditional is narrower than the marginal by a
 * factor √(1 − ρ²), and by nothing else.
 */
export function conditional2(
  mean: Vec,
  cov: Mat,
  beta: number,
): { mean: number; variance: number } {
  const k = cov[0][1] / Math.max(cov[1][1], 1e-12);
  return {
    mean: mean[0] + k * (beta - mean[1]),
    variance: Math.max(cov[0][0] - k * cov[0][1], 1e-12),
  };
}

// ---------------------------------------------------------------------------
// Ellipse coverage
// ---------------------------------------------------------------------------

/**
 * The χ² quantile with **two** degrees of freedom, in closed form.
 *
 * The χ²₂ CDF is 1 − exp(−c/2), so the inverse needs no special functions:
 * `chi2Quantile2(0.95) = 5.9915…`, the number every 2-D confidence ellipse in
 * this book is drawn at. The ellipse's semi-axes are √(c λᵢ), so the 95%
 * ellipse is the 2.4477σ ellipse — not the 2σ one, which only covers 86.5%.
 */
export function chi2Quantile2(p: number): number {
  const q = Math.min(Math.max(p, 0), 1 - 1e-15);
  return -2 * Math.log(1 - q);
}

/** Probability mass inside the `nSigma` ellipse of a 2-D Gaussian. */
export function ellipseCoverage2(nSigma: number): number {
  return 1 - Math.exp(-0.5 * nSigma * nSigma);
}
