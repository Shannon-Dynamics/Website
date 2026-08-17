/**
 * Seeded, reproducible randomness.
 *
 * Every experiment in the book pins a seed so a reader's run matches the
 * figure in the text. mulberry32 is small, fast and has good equidistribution
 * for simulation work of this scale.
 */

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller. */
export function gaussian(rng: Rng, mean = 0, std = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Sample an index from a probability vector (assumed to sum to 1). */
export function sampleCategorical(rng: Rng, probs: number[]): number {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  return probs.length - 1;
}

/** Index of the maximum, ties broken uniformly at random. */
export function argmaxRandomTie(values: number[], rng: Rng): number {
  let best = -Infinity;
  let ties: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] > best + 1e-12) {
      best = values[i];
      ties = [i];
    } else if (Math.abs(values[i] - best) <= 1e-12) {
      ties.push(i);
    }
  }
  return ties[Math.floor(rng() * ties.length)];
}

export function argmax(values: number[]): number {
  let best = -Infinity;
  let bestIdx = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > best) {
      best = values[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

/** Sample from a Beta distribution (Cheng's BB/BC algorithm, simplified). */
export function sampleBeta(rng: Rng, alpha: number, beta: number): number {
  const x = sampleGamma(rng, alpha);
  const y = sampleGamma(rng, beta);
  return x / (x + y);
}

/** Marsaglia–Tsang gamma sampler (shape ≥ 1 handled by boost for shape < 1). */
function sampleGamma(rng: Rng, shape: number): number {
  if (shape < 1) {
    return sampleGamma(rng, shape + 1) * Math.pow(rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const z = gaussian(rng);
    const v = Math.pow(1 + c * z, 3);
    if (v <= 0) continue;
    const u = rng();
    if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v)) {
      return d * v;
    }
  }
}
