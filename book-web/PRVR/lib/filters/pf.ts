/**
 * The particle filter (Monte Carlo localization) — Thrun et al., **Table 4.3**
 * and **Table 8.2**.
 *
 * The belief is a bag of weighted poses. That representation is the reason MCL
 * can hold "I am in room A *or* room C" as a single object, which no Gaussian
 * filter can do — and the reason it needs resampling, which is where all of its
 * failure modes live.
 */

import { meanPose, type Pose2 } from '../geom/se2';
import { discreteEntropy } from '../prob/gaussian';
import type { Mat } from '../prob/linalg';
import type { Rng } from '../prob/rng';
import { isFree, type World } from '../sim/world';

export interface Particle {
  state: Pose2;
  weight: number;
}

export type ResampleMethod = 'multinomial' | 'lowVariance';

const clone = (p: Particle): Particle => ({ state: { ...p.state }, weight: p.weight });

/** Scale weights to sum to 1; a fully-dead population resets to uniform. */
export function normalizeWeights(particles: Particle[]): void {
  let total = 0;
  for (const p of particles) total += p.weight;
  if (total > 0 && Number.isFinite(total)) {
    for (const p of particles) p.weight /= total;
  } else {
    const w = 1 / particles.length;
    for (const p of particles) p.weight = w;
  }
}

/**
 * Textbook resampling: M independent draws from the categorical distribution
 * defined by the weights. Simple, correct, and noticeably noisier than the
 * low-variance version — a particle with weight 1/M has a 37% chance of
 * vanishing outright every single step.
 */
export function multinomialResample(particles: Particle[], rng: Rng): Particle[] {
  const M = particles.length;
  const cdf = new Array<number>(M);
  let acc = 0;
  for (let i = 0; i < M; i++) {
    acc += particles[i].weight;
    cdf[i] = acc;
  }
  const out: Particle[] = [];
  for (let m = 0; m < M; m++) {
    const u = rng.next() * acc;
    // Binary search into the CDF.
    let lo = 0;
    let hi = M - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < u) lo = mid + 1;
      else hi = mid;
    }
    out.push({ state: { ...particles[lo].state }, weight: 1 / M });
  }
  return out;
}

/**
 * `low_variance_sampler` — Thrun et al., **Table 4.4**.
 *
 * One random number, then a comb of M evenly spaced teeth stepped through the
 * cumulative weights. Two properties fall out that make it the default
 * everywhere: any particle with weight ≥ 1/M is guaranteed to survive, and the
 * whole pass is O(M) with a single RNG draw instead of M.
 */
export function lowVarianceResample(particles: Particle[], rng: Rng): Particle[] {
  const M = particles.length;
  if (M === 0) return [];

  let total = 0;
  for (const p of particles) total += p.weight;
  if (!(total > 0)) return particles.map(clone);

  const step = total / M;
  const r = rng.uniform(0, step);
  const out: Particle[] = [];
  let c = particles[0].weight;
  let i = 0;
  for (let m = 0; m < M; m++) {
    const u = r + m * step;
    while (u > c && i < M - 1) {
      i++;
      c += particles[i].weight;
    }
    out.push({ state: { ...particles[i].state }, weight: 1 / M });
  }
  return out;
}

export class ParticleFilter {
  particles: Particle[];

  constructor(particles: Particle[]) {
    this.particles = particles;
  }

  /** Global localization: scatter M particles uniformly over the free space. */
  static uniformInWorld(n: number, world: World, rng: Rng, clearance = 0.15): ParticleFilter {
    const { minX, minY, maxX, maxY } = world.bounds;
    const particles: Particle[] = [];
    let guard = 0;
    while (particles.length < n && guard < n * 200) {
      guard++;
      const x = rng.uniform(minX, maxX);
      const y = rng.uniform(minY, maxY);
      if (!isFree(world, x, y, clearance)) continue;
      particles.push({
        state: { x, y, theta: rng.uniform(-Math.PI, Math.PI) },
        weight: 1 / n,
      });
    }
    return new ParticleFilter(particles);
  }

  /** Position tracking: a Gaussian blob around a known starting pose. */
  static gaussian(
    n: number,
    mean: Pose2,
    std: { x: number; y: number; theta: number },
    rng: Rng,
  ): ParticleFilter {
    return new ParticleFilter(
      Array.from({ length: n }, () => ({
        state: {
          x: rng.normal(mean.x, std.x),
          y: rng.normal(mean.y, std.y),
          theta: rng.normal(mean.theta, std.theta),
        },
        weight: 1 / n,
      })),
    );
  }

  get size(): number {
    return this.particles.length;
  }

  /** Push every particle through a *sampling* motion model — noise per particle. */
  predict(sampleMotion: (state: Pose2) => Pose2): void {
    for (const p of this.particles) p.state = sampleMotion(p.state);
  }

  /**
   * Importance weighting: w ← w · p(z | x). Because the proposal is the motion
   * model, the importance weight *is* the measurement likelihood — the single
   * cancellation that makes MCL as simple as it is.
   */
  correct(likelihood: (state: Pose2) => number): void {
    for (const p of this.particles) p.weight *= Math.max(likelihood(p.state), 0);
    normalizeWeights(this.particles);
  }

  /** Same, for a likelihood returned in log space (the numerically safe path). */
  correctLog(logLikelihood: (state: Pose2) => number): void {
    const ls = this.particles.map((p) => logLikelihood(p.state));
    let max = -Infinity;
    for (const l of ls) if (l > max) max = l;
    if (!Number.isFinite(max)) {
      normalizeWeights(this.particles);
      return;
    }
    this.particles.forEach((p, i) => {
      p.weight *= Math.exp(ls[i] - max);
    });
    normalizeWeights(this.particles);
  }

  resample(rng: Rng, method: ResampleMethod = 'lowVariance'): void {
    this.particles =
      method === 'lowVariance'
        ? lowVarianceResample(this.particles, rng)
        : multinomialResample(this.particles, rng);
  }

  /**
   * Effective sample size, 1 / Σ wᵢ².
   *
   * M when the weights are uniform, 1 when a single particle carries
   * everything. Resampling only when N_eff drops below ~M/2 is the standard way
   * to avoid the *other* failure mode: resampling a healthy population throws
   * away diversity for nothing.
   */
  effectiveSampleSize(): number {
    let s = 0;
    let total = 0;
    for (const p of this.particles) {
      s += p.weight * p.weight;
      total += p.weight;
    }
    if (s <= 0 || total <= 0) return 0;
    return (total * total) / s;
  }

  /** Weight entropy in bits — log₂ M when healthy, → 0 as the filter degenerates. */
  weightEntropy(): number {
    return discreteEntropy(this.particles.map((p) => p.weight));
  }

  /** Weighted mean pose, with the heading averaged on the circle. */
  mean(): Pose2 {
    return meanPose(
      this.particles.map((p) => p.state),
      this.particles.map((p) => p.weight),
    );
  }

  /** Weighted 2×2 position covariance — what the widget draws as an ellipse. */
  positionCovariance(): Mat {
    const m = this.mean();
    let wsum = 0;
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (const p of this.particles) {
      const dx = p.state.x - m.x;
      const dy = p.state.y - m.y;
      sxx += p.weight * dx * dx;
      sxy += p.weight * dx * dy;
      syy += p.weight * dy * dy;
      wsum += p.weight;
    }
    if (wsum <= 0) return [[0, 0], [0, 0]];
    return [
      [sxx / wsum, sxy / wsum],
      [sxy / wsum, syy / wsum],
    ];
  }

  /** The single most likely particle — the MAP estimate of the bag. */
  best(): Particle {
    let b = this.particles[0];
    for (const p of this.particles) if (p.weight > b.weight) b = p;
    return b;
  }

  belief(): Particle[] {
    return this.particles.map(clone);
  }
}
