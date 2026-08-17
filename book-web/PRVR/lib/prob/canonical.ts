/**
 * The canonical (information) parameterization of a Gaussian.
 *
 *   p(x) ∝ exp(−½ xᵀ Ω x + xᵀ ξ),    Ω = Σ⁻¹,   ξ = Σ⁻¹ μ
 *
 * Moments form (μ, Σ) answers "where is it and how wide"; canonical form
 * (ξ, Ω) answers "how much do I know". They carry identical information and
 * the conversion is one matrix solve each way — but the *costs* of the two
 * Bayes-filter operations swap between them:
 *
 *   · multiplying two Gaussians  → add ξ and Ω        (O(n²) here, O(n³) there)
 *   · marginalizing one out      → drop a block of Σ  (O(1) there, O(n³) here)
 *
 * That asymmetry is the whole reason Chapter 6 ships two filters instead of
 * one, and the reason Chapter 15's factor graphs live in information form.
 *
 * Introduced in Chapter 2 (Derivation 3); reused by Chapters 6, 14 and 15.
 */

import { add, inv, matAdd, matVec, solve, symmetrize, type Mat, type Vec } from './linalg';

/** A Gaussian written as (ξ, Ω). */
export interface Canonical {
  /** Information vector ξ = Σ⁻¹ μ. */
  xi: Vec;
  /** Information (precision) matrix Ω = Σ⁻¹. */
  omega: Mat;
}

/** A Gaussian written as (μ, Σ). */
export interface Moments {
  mean: Vec;
  cov: Mat;
}

/**
 * (μ, Σ) → (ξ, Ω). Costs one inverse; do it once and stay in canonical form
 * for as long as you are only multiplying.
 */
export function toCanonical(mean: Vec, cov: Mat): Canonical {
  const omega = symmetrize(inv(cov));
  return { xi: matVec(omega, mean), omega };
}

/** (ξ, Ω) → (μ, Σ). One solve for the mean, one inverse for the covariance. */
export function toMoments(c: Canonical): Moments {
  return { mean: solve(c.omega, c.xi), cov: symmetrize(inv(c.omega)) };
}

/**
 * The Bayes product in canonical form: exponents add, so the parameters add.
 *
 * This is Derivation 3 of Chapter 2, and it is the entire content of the
 * information filter's measurement update.
 */
export function canonicalProduct(a: Canonical, b: Canonical): Canonical {
  return { xi: add(a.xi, b.xi), omega: matAdd(a.omega, b.omega) };
}

// ---------------------------------------------------------------------------
// Scalar specializations
//
// The 1-D case is worth spelling out because it is the one a reader can check
// by hand, and it is what the Blob Multiplier widget prints.
// ---------------------------------------------------------------------------

export interface Canonical1 {
  xi: number;
  omega: number;
}

/** Precision ω = 1/σ², information ξ = μ/σ². */
export function toCanonical1(mean: number, variance: number): Canonical1 {
  const omega = 1 / Math.max(variance, 1e-12);
  return { xi: mean * omega, omega };
}

export function toMoments1(c: Canonical1): { mean: number; variance: number } {
  const omega = Math.max(c.omega, 1e-12);
  return { mean: c.xi / omega, variance: 1 / omega };
}

/** Two scalar Gaussians multiplied: ξ and ω add. Nothing else happens. */
export function canonicalProduct1(a: Canonical1, b: Canonical1): Canonical1 {
  return { xi: a.xi + b.xi, omega: a.omega + b.omega };
}

/**
 * Log of the constant that normalization throws away when two Gaussians are
 * multiplied — i.e. log ∫ N(x; μ₁, v₁) N(x; μ₂, v₂) dx, which is itself a
 * Gaussian evaluated at the difference of the means.
 *
 * The Bayes filter calls this the *evidence*; Chapter 11 uses it for gating
 * and Chapter 12 to notice that a robot has been kidnapped. It is the reason
 * "η absorbs the leftover constant" is bookkeeping and not hand-waving.
 */
export function logEvidence1(m1: number, v1: number, m2: number, v2: number): number {
  const s = v1 + v2;
  const d = m1 - m2;
  return -0.5 * ((d * d) / s + Math.log(2 * Math.PI * s));
}
