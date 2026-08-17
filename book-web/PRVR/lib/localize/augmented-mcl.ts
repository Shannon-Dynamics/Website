/**
 * Augmented MCL — Thrun et al., **Table 8.3**.
 *
 * Plain MCL cannot recover from kidnapping, and no resampling scheme can fix
 * that: resampling only ever *reuses* poses that are already in the set, so a
 * region with zero particles stays at zero forever. Recovery has to come from
 * outside the recursion, by injecting fresh poses — and the whole engineering
 * question is *when*.
 *
 * Augmented MCL answers it without a per-map threshold. Track the average
 * measurement likelihood at two timescales,
 *
 *   w_slow ← w_slow + α_slow (w_avg − w_slow)     (the long-run baseline)
 *   w_fast ← w_fast + α_fast (w_avg − w_fast)     (what just happened)
 *
 * and inject a random pose with probability max(0, 1 − w_fast / w_slow). The
 * statistic is a *ratio*, so it is invariant to the absolute scale of the
 * likelihood — which is why the same two α's work in a corridor and in a
 * warehouse. It also self-extinguishes: once the filter re-converges, w_fast
 * climbs back above w_slow and injection switches itself off.
 *
 * This file holds only what `lib/filters/pf.ts` does not: the two-timescale
 * detector and the injection-aware resampler. Particles, weights, ESS and the
 * low-variance sampler all live there and are reused unchanged.
 */

import type { Pose2 } from '../geom/se2';
import { lowVarianceResample, type Particle } from '../filters/pf';
import type { Rng } from '../prob/rng';

export interface SurpriseParams {
  /** Short-horizon gain. Thrun requires 0 ≤ α_slow ≪ α_fast. */
  alphaFast: number;
  /** Long-horizon gain: the baseline the filter compares itself against. */
  alphaSlow: number;
}

/** Decades apart, as the algorithm demands. Ours are the chapter's worked example. */
export const DEFAULT_SURPRISE: SurpriseParams = { alphaFast: 0.5, alphaSlow: 0.05 };

/**
 * Injection probability max(0, 1 − w_fast / w_slow) — Table 8.3, line 13.
 *
 * Zero whenever the present looks at least as good as the past, which is the
 * common case; it only becomes positive when the short-term average *dives*.
 */
export function injectionProbability(wFast: number, wSlow: number): number {
  if (!(wSlow > 0)) return 0;
  return Math.max(0, 1 - wFast / wSlow);
}

/**
 * The two exponential filters of Table 8.3, lines 10–11.
 *
 * The first update seeds both averages with the same value instead of starting
 * from zero: a filter that begins at w_fast = w_slow = 0 would report an
 * injection probability of 0 on step 1 and then a spurious spike on step 2,
 * which is an artefact of the initialisation rather than a property of the data.
 */
export class SurpriseDetector {
  wFast = 0;
  wSlow = 0;
  private seeded = false;

  constructor(public params: SurpriseParams = DEFAULT_SURPRISE) {}

  /** Fold in this step's average likelihood; returns the injection probability. */
  update(wAvg: number): number {
    const { alphaFast, alphaSlow } = this.params;
    if (!this.seeded) {
      this.wFast = wAvg;
      this.wSlow = wAvg;
      this.seeded = true;
      return 0;
    }
    this.wFast += alphaFast * (wAvg - this.wFast);
    this.wSlow += alphaSlow * (wAvg - this.wSlow);
    return this.injectionProbability;
  }

  get injectionProbability(): number {
    return injectionProbability(this.wFast, this.wSlow);
  }

  reset(): void {
    this.wFast = 0;
    this.wSlow = 0;
    this.seeded = false;
  }
}

/** w_avg — the mean *unnormalized* weight (Table 8.3, line 8). */
export function averageWeight(particles: readonly Particle[]): number {
  if (particles.length === 0) return 0;
  let s = 0;
  for (const p of particles) s += p.weight;
  return s / particles.length;
}

/**
 * Per-beam geometric-mean likelihood, exp(log q / K).
 *
 * The raw product over K beams is unusable as a *level*: a 200-beam sweep with
 * per-beam likelihoods of order 1e-2 gives 1e-400, which is not a double, and
 * even when it is representable its magnitude swings with the beam count, the
 * map and σ_hit — so α_fast and α_slow would have to be re-tuned per
 * deployment, which defeats the purpose. Taking the
 * K-th root is a monotone rescaling that puts w_avg on a per-beam scale of
 * order 0.1–1 and makes the two gains portable. Production AMCL does the same
 * thing by another route: its likelihood-field score is a *sum* over beams
 * rather than a product.
 */
export function perBeamLikelihood(logLikelihood: number, nBeams: number): number {
  if (nBeams <= 0) return 0;
  return Math.exp(logLikelihood / nBeams);
}

export interface AugmentedResampleResult {
  particles: Particle[];
  /** How many of the M outputs are freshly injected poses. */
  injected: number;
}

/**
 * Resampling with random injection — Table 8.3, lines 12–17.
 *
 * Table 8.3 draws each of the M outputs independently: a coin flip decides
 * inject-or-resample, and the resample branch draws one index with probability
 * proportional to its weight. We keep the coin flip exactly, but hand the
 * surviving draws to the low-variance sampler of Chapter 8 in one batch instead
 * of drawing them independently. Both target the same distribution; the comb
 * simply has lower variance, and it is what every deployed implementation does.
 *
 * `drawRandomPose` is the injector. Uniform-over-free-space is the textbook
 * choice; drawing from the measurement model instead (Chapter 10's
 * `sample_pose`, or the likelihood field) recovers far faster, because the
 * injected particles land where the current scan says the robot might be.
 */
export function augmentedResample(
  particles: readonly Particle[],
  rng: Rng,
  pInject: number,
  drawRandomPose: (rng: Rng) => Pose2,
): AugmentedResampleResult {
  const M = particles.length;
  if (M === 0) return { particles: [], injected: 0 };

  const p = Math.max(0, Math.min(1, pInject));
  let injected = 0;
  for (let m = 0; m < M; m++) if (rng.next() < p) injected++;

  const keep = M - injected;
  const out: Particle[] = [];
  if (keep > 0) {
    // The comb needs a set of the right size, so resample M and take `keep` of
    // them by stride — equivalent to running the comb with `keep` teeth.
    const combed = lowVarianceResample(
      particles.map((q) => ({ state: { ...q.state }, weight: q.weight })),
      rng,
    );
    const stride = M / keep;
    for (let i = 0; i < keep; i++) {
      const src = combed[Math.min(M - 1, Math.floor(i * stride))];
      out.push({ state: { ...src.state }, weight: 1 / M });
    }
  }
  for (let i = 0; i < injected; i++) {
    out.push({ state: drawRandomPose(rng), weight: 1 / M });
  }
  return { particles: out, injected };
}

/**
 * Weighted mean of the *dominant cluster*, not of the whole set.
 *
 * The mean of a bimodal cloud sits in the wall between the two rooms — a pose
 * the filter assigns essentially zero probability. Reporting it as "the
 * estimate" is the single most common way to make a correct filter look broken.
 * We instead grow a cluster around the heaviest particle: everything within
 * `radius` metres of it, iterated once so the cluster can walk toward its own
 * centre of mass, and return that cluster's mass alongside the pose so the
 * caller can say "60% of my belief is here" rather than pretending it is all.
 */
export function dominantCluster(
  particles: readonly Particle[],
  radius = 0.6,
): { pose: Pose2; mass: number; members: number } {
  if (particles.length === 0) {
    return { pose: { x: 0, y: 0, theta: 0 }, mass: 0, members: 0 };
  }
  let seed = particles[0];
  for (const q of particles) if (q.weight > seed.weight) seed = q;

  let cx = seed.state.x;
  let cy = seed.state.y;
  let mass = 0;
  let members = 0;
  let sinSum = 0;
  let cosSum = 0;

  for (let pass = 0; pass < 2; pass++) {
    let wx = 0;
    let wy = 0;
    mass = 0;
    members = 0;
    sinSum = 0;
    cosSum = 0;
    for (const q of particles) {
      if (Math.hypot(q.state.x - cx, q.state.y - cy) > radius) continue;
      wx += q.weight * q.state.x;
      wy += q.weight * q.state.y;
      sinSum += q.weight * Math.sin(q.state.theta);
      cosSum += q.weight * Math.cos(q.state.theta);
      mass += q.weight;
      members++;
    }
    if (mass <= 0) break;
    cx = wx / mass;
    cy = wy / mass;
  }

  const total = particles.reduce((s, q) => s + q.weight, 0) || 1;
  return {
    pose: { x: cx, y: cy, theta: Math.atan2(sinSum, cosSum) },
    mass: mass / total,
    members,
  };
}

/**
 * How many distinct hypotheses the cloud is holding, by single-link clustering
 * on a `binSize` grid. It is the honest readout of "the filter is still
 * ambiguous", and it is what the symmetric-wing scenario is measured with.
 */
export function countClusters(particles: readonly Particle[], binSize = 0.5): number {
  const occupied = new Set<string>();
  for (const p of particles) {
    occupied.add(`${Math.floor(p.state.x / binSize)},${Math.floor(p.state.y / binSize)}`);
  }
  const seen = new Set<string>();
  let clusters = 0;
  for (const key of occupied) {
    if (seen.has(key)) continue;
    clusters++;
    const stack = [key];
    seen.add(key);
    while (stack.length > 0) {
      const [i, j] = stack.pop()!.split(',').map(Number);
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          const nk = `${i + di},${j + dj}`;
          if (occupied.has(nk) && !seen.has(nk)) {
            seen.add(nk);
            stack.push(nk);
          }
        }
      }
    }
  }
  return clusters;
}
