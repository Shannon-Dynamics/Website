/**
 * Seeded, reproducible pseudo-random numbers.
 *
 * Every simulation in this book is deterministic given a seed: the reader can
 * re-roll and get a *different* run, or type the same seed and get the *same*
 * run. This mirrors the Rust side, where the book uses `rand::rngs::SmallRng`
 * with an explicit seed rather than `thread_rng()`.
 */

/** A seeded uniform generator. `next()` returns a float in [0, 1). */
export class Rng {
  private s: number;

  constructor(seed = 42) {
    // Avoid the zero state, which is a fixed point of the mixer.
    this.s = (seed >>> 0) || 0x9e3779b9;
  }

  /** mulberry32 — small, fast, good enough for visualization. */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [lo, hi). */
  uniform(lo = 0, hi = 1): number {
    return lo + (hi - lo) * this.next();
  }

  /** Standard normal via Box–Muller. */
  normal(mean = 0, std = 1): number {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * Thrun's cheap normal approximation (Table 5.4): sum of 12 uniforms.
   * Kept because several chapters discuss it explicitly.
   */
  normalApprox(std: number): number {
    let sum = 0;
    for (let i = 0; i < 12; i++) sum += this.uniform(-1, 1);
    return (std * sum) / 2;
  }

  /** Triangular noise, the second sampler in Thrun's motion models. */
  triangular(std: number): number {
    return std * Math.sqrt(6) * 0.5 * (this.uniform(-1, 1) + this.uniform(-1, 1));
  }

  /** Exponential with rate lambda — used by the beam model's p_short. */
  exponential(lambda: number): number {
    return -Math.log(1 - this.next()) / lambda;
  }

  /** Pick an index from unnormalized weights. */
  choice(weights: number[]): number {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = this.next() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }

  /** Fork a child generator — useful to keep sub-simulations independent. */
  fork(): Rng {
    return new Rng(Math.floor(this.next() * 2 ** 32));
  }
}

/** Convenience: a fresh generator per call site. */
export const rng = (seed = 42) => new Rng(seed);
