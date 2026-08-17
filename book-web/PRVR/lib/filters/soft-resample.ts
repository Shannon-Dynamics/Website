/**
 * Soft resampling — Karkus, Hsu and Lee (2018), Chapter 25 derivation F6.
 *
 * Multinomial and low-variance resampling are piecewise constant in the
 * weights: nudge a weight and, almost surely, the same particles survive with
 * the same new weight `1/M`. The derivative is zero almost everywhere, so a
 * gradient travelling backwards through a particle filter dies at the first
 * resample node and never reaches the measurement model that produced the
 * weights.
 *
 * The repair is to resample from a *mixture* of the weights and the uniform
 * distribution, and to pay for the difference with an importance weight:
 *
 *   q(i)  = λ w_i + (1 − λ)/M
 *   w'_i  = w_i / q(i)
 *
 * The estimator stays unbiased for every λ ∈ (0, 1], and `w'` now depends
 * smoothly on `w`, so gradient flows through the survivors. λ = 1 recovers the
 * classical (blind) resampler; λ → 0 recovers plain importance sampling, which
 * transmits gradient perfectly but never concentrates its particles.
 *
 * This file is additive: it never touches `pf.ts`, and a `ParticleFilter` opts
 * in by calling `softResample` instead of `resample`.
 */

import type { Rng } from '../prob/rng';
import { normalizeWeights, type Particle } from './pf';

export interface SoftResampleResult {
  particles: Particle[];
  /**
   * Mean of `∂ log w'_i / ∂ log w_i` over the surviving particles: the fraction
   * of the gradient signal that crosses this resample node. 1 at λ = 0, 0 at
   * λ = 1, and the number the Resampling Gradient Microscope animates.
   */
  transmission: number;
  /** Variance of the post-resampling weights — what soft resampling costs. */
  weightVariance: number;
}

/**
 * `soft_resample(X_t, λ, rng)`.
 *
 * The comb of `low_variance_sampler` (Thrun Table 4.4) is reused verbatim, but
 * stepped through the *proposal* `q` rather than through `w`, and each survivor
 * carries `w'` instead of `1/M`.
 */
export function softResample(
  particles: readonly Particle[],
  lambda: number,
  rng: Rng,
): SoftResampleResult {
  const M = particles.length;
  if (M === 0) return { particles: [], transmission: 0, weightVariance: 0 };

  // Work from normalized weights so that q is a distribution.
  let total = 0;
  for (const p of particles) total += p.weight;
  const w = total > 0 ? particles.map((p) => p.weight / total) : particles.map(() => 1 / M);

  const lam = Math.min(1, Math.max(0, lambda));
  const q = w.map((wi) => lam * wi + (1 - lam) / M);

  // One random offset, M evenly spaced teeth — through q, not through w.
  const step = 1 / M;
  const start = rng.uniform(0, step);
  const out: Particle[] = [];
  let acc = q[0];
  let i = 0;
  let transmission = 0;

  for (let m = 0; m < M; m++) {
    const u = start + m * step;
    while (u > acc && i < M - 1) {
      i += 1;
      acc += q[i];
    }
    // The importance weight that keeps the estimator unbiased.
    out.push({ state: { ...particles[i].state }, weight: w[i] / q[i] });
    // ∂ log w'_i / ∂ log w_i = ((1 − λ)/M) / q_i.
    transmission += (1 - lam) / M / q[i];
  }

  transmission /= M;

  // Weight variance is measured *before* renormalization, in units of 1/M, so
  // that a healthy classical resample reads 0 and a degenerate one reads large.
  let mean = 0;
  for (const p of out) mean += p.weight;
  mean /= M;
  let variance = 0;
  for (const p of out) variance += (p.weight - mean) * (p.weight - mean);
  variance = M > 1 ? variance / (M - 1) : 0;

  normalizeWeights(out);
  return { particles: out, transmission, weightVariance: variance };
}

/**
 * Expected transmission coefficient, averaged under the proposal that actually
 * draws the indices.
 *
 * The sum telescopes: `Σ_i q_i · ((1−λ)/M)/q_i = Σ_i (1−λ)/M = 1 − λ`, whatever
 * the weights are. So λ is not merely *related* to how much gradient survives a
 * resample node — in expectation it **is** the amount that dies. The loop is
 * kept rather than replaced by `1 - lambda` because seeing the cancellation is
 * the point; the self-checks assert the two agree.
 */
export function gradientTransmission(weights: readonly number[], lambda: number): number {
  const M = weights.length;
  if (M === 0) return 0;
  let total = 0;
  for (const x of weights) total += x;
  const w = total > 0 ? weights.map((x) => x / total) : weights.map(() => 1 / M);
  const lam = Math.min(1, Math.max(0, lambda));
  let acc = 0;
  for (let i = 0; i < M; i++) {
    const q = lam * w[i] + (1 - lam) / M;
    // Weighted by q, because that is how often particle i is actually drawn.
    acc += q * (((1 - lam) / M) / q);
  }
  return acc;
}
