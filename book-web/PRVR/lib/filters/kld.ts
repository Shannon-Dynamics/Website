/**
 * KLD-adaptive sample size — Fox, *Adapting the Sample Size in Particle Filters
 * Through KLD-Sampling*, IJRR 22(12), 2003.
 *
 * The particle count a filter needs is not a property of how *accurate* you
 * want to be; it is a property of how *spread out* the belief currently is. Fox
 * makes that precise: bin the state space, count how many bins the samples
 * actually occupy, and ask how many samples are needed so that — with
 * probability 1 − δ — the KL divergence between the sample-based maximum
 * likelihood estimate and the true posterior stays under ε.
 *
 * The answer, via the Wilson–Hilferty approximation to the χ²ₖ₋₁ quantile, is
 *
 *   M ≥ (k − 1) / (2ε) · [ 1 − 2/(9(k−1)) + √(2/(9(k−1))) · z₁₋δ ]³
 *
 * which is O(1) to evaluate and depends on the belief only through k, the
 * number of *occupied* bins. Thousands of particles during global uncertainty;
 * dozens once the cloud has condensed.
 *
 * This file is the only piece of Chapter 8's machinery that is not already in
 * `lib/filters/pf.ts`: the resamplers, the ESS, and the particle set itself all
 * live there, and this module deliberately does not duplicate them.
 */

/** z₁₋δ for the confidences the book uses; δ = 0.01 is Fox's default. */
export const Z_ONE_MINUS_DELTA = Object.freeze({
  0.1: 1.2815515655446004,
  0.05: 1.6448536269514722,
  0.01: 2.3263478740408408,
});

/**
 * Inverse standard normal CDF (Acklam's rational approximation, |ε| < 1.15e-9).
 *
 * Present so a reader can dial δ to something other than the three tabulated
 * values without leaving the page. Rust uses `statrs`'s `Normal::inverse_cdf`
 * for the same job.
 */
export function standardNormalQuantile(p: number): number {
  if (!(p > 0) || !(p < 1)) return p <= 0 ? -Infinity : Infinity;

  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
    138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
    66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
    3.754408661907416];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export interface KldOptions {
  /** Tolerated KL divergence between the sample MLE and the true posterior. */
  epsilon?: number;
  /** Confidence: the bound holds with probability 1 − δ. */
  delta?: number;
  /** Never go below this many particles, however tight the belief gets. */
  minParticles?: number;
  /** Never go above this many, however lost the robot is. */
  maxParticles?: number;
}

export const DEFAULT_KLD: Required<KldOptions> = {
  epsilon: 0.05,
  delta: 0.01,
  minParticles: 25,
  maxParticles: 5000,
};

/**
 * `KLD_sample_size(k, ε, δ)` — Fox (2003), eq. 7.
 *
 * `kBins` is the number of bins the particles currently *occupy*, not the
 * number of bins in the grid: an empty bin costs nothing, which is exactly why
 * the number shrinks as the belief condenses.
 *
 * With fewer than two occupied bins the χ² statistic has no degrees of freedom
 * and the bound is vacuous, so we fall back to `minParticles`.
 */
export function kldSampleSize(kBins: number, opts: KldOptions = {}): number {
  const { epsilon, delta, minParticles, maxParticles } = { ...DEFAULT_KLD, ...opts };
  if (kBins <= 1) return minParticles;

  const z = standardNormalQuantile(1 - delta);
  const k1 = kBins - 1;
  const a = 2 / (9 * k1);
  const inner = 1 - a + Math.sqrt(a) * z;
  const m = (k1 / (2 * epsilon)) * inner * inner * inner;

  return Math.min(maxParticles, Math.max(minParticles, Math.ceil(m)));
}

/**
 * How many distinct bins of width `binSize` the values occupy.
 *
 * The bookkeeping that makes KLD sampling cheap: a hash set of bin indices,
 * updated as each particle is drawn, so `k` is known before the sample set is
 * finished and the loop can stop the moment it has enough.
 */
export function countOccupiedBins1D(values: readonly number[], binSize: number): number {
  const seen = new Set<number>();
  for (const v of values) seen.add(Math.floor(v / binSize));
  return seen.size;
}

/**
 * `KLD_Sampling` — the adaptive draw loop, Fox (2003), Table 2 lines 4–15.
 *
 * Draw particles one at a time with probability proportional to weight. Every
 * time a draw lands in a bin nothing has landed in yet, `k` grows and the
 * required sample size is recomputed. Stop as soon as the number drawn meets
 * the bound. The loop therefore *discovers* how many particles it needs while
 * it is filling the set — no second pass, no pre-scan of the belief.
 *
 * Sampling and binning are injected so this stays independent of the state
 * type: `pick` returns an index with probability proportional to that
 * particle's weight (`Rng.choice`), and `binOf` maps an index to whichever bin
 * key the state space uses.
 */
export function kldResampleIndices(
  weights: readonly number[],
  binOf: (index: number) => number | string,
  pick: () => number,
  opts: KldOptions = {},
): number[] {
  const { minParticles, maxParticles } = { ...DEFAULT_KLD, ...opts };
  if (weights.length === 0) return [];

  const bins = new Set<number | string>();
  const out: number[] = [];
  // Before any bin is occupied the bound is undefined, so the floor governs.
  let required = minParticles;

  while (out.length < required && out.length < maxParticles) {
    const idx = pick();
    out.push(idx);
    const key = binOf(idx);
    if (!bins.has(key)) {
      bins.add(key);
      required = kldSampleSize(bins.size, opts);
    }
  }
  return out;
}

/** The same, for planar poses binned on (x, y, θ) — the shape Chapter 12 uses. */
export function countOccupiedBins3D(
  poses: readonly { x: number; y: number; theta: number }[],
  binXY: number,
  binTheta: number,
): number {
  const seen = new Set<string>();
  for (const p of poses) {
    seen.add(
      `${Math.floor(p.x / binXY)},${Math.floor(p.y / binXY)},${Math.floor(p.theta / binTheta)}`,
    );
  }
  return seen.size;
}
