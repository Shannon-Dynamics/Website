/**
 * Robust kernels and their IRLS weights — Chapter 15, derivation D6.
 *
 * Least squares gives every residual unbounded leverage: the gradient of
 * ½e² grows linearly in e forever, so one bad loop closure can drag an entire
 * map with it. A robust kernel replaces ½e² by ρ(e), a function that stops
 * growing (or starts shrinking) once the residual is implausibly large.
 *
 * The only thing an optimizer needs from ρ is its **weight**
 *
 *     w(e) = ρ′(e) / e
 *
 * because stationarity of Σₖ ρ(eₖ) is exactly the stationarity of a *weighted*
 * least-squares problem Σₖ ½ wₖ eₖ². That is iteratively reweighted least
 * squares (IRLS): recompute w each iteration, multiply it into the normal
 * equations, and every line of the Gauss–Newton machinery is untouched.
 *
 * All kernels here are written so that ρ(e) ≈ ½e² for small e — a robust kernel
 * must not change what the estimator does when nothing is wrong.
 */

export type Kernel =
  | { type: 'l2' }
  | { type: 'huber'; k: number }
  | { type: 'cauchy'; c: number }
  | { type: 'geman'; c: number };

export const L2: Kernel = { type: 'l2' };

export const KERNEL_NAMES: Record<Kernel['type'], string> = {
  l2: 'L2 (no kernel)',
  huber: 'Huber',
  cauchy: 'Cauchy',
  geman: 'Geman–McClure',
};

/** The scale parameter of a kernel, in units of whitened residual (σ). */
export function kernelScale(kernel: Kernel): number {
  switch (kernel.type) {
    case 'huber':
      return kernel.k;
    case 'cauchy':
    case 'geman':
      return kernel.c;
    default:
      return Number.POSITIVE_INFINITY;
  }
}

/** Replace a kernel's scale, keeping its family. */
export function withScale(kernel: Kernel, s: number): Kernel {
  switch (kernel.type) {
    case 'huber':
      return { type: 'huber', k: s };
    case 'cauchy':
      return { type: 'cauchy', c: s };
    case 'geman':
      return { type: 'geman', c: s };
    default:
      return kernel;
  }
}

/**
 * ρ(e) — the contribution of one factor to the objective, with ρ(e) → ½e² as
 * e → 0 for every family.
 */
export function kernelRho(kernel: Kernel, e: number): number {
  const a = Math.abs(e);
  switch (kernel.type) {
    case 'l2':
      return 0.5 * a * a;
    case 'huber':
      return a <= kernel.k ? 0.5 * a * a : kernel.k * (a - 0.5 * kernel.k);
    case 'cauchy': {
      const c2 = kernel.c * kernel.c;
      return 0.5 * c2 * Math.log1p((a * a) / c2);
    }
    case 'geman': {
      const c2 = kernel.c * kernel.c;
      return (0.5 * c2 * a * a) / (c2 + a * a);
    }
  }
}

/**
 * The IRLS weight w = ρ′(e)/e.
 *
 * L2 returns 1 identically, which is the sense in which "no kernel" is a
 * kernel. Huber saturates the *influence* ρ′ at k, so w decays like k/|e|.
 * Cauchy and Geman–McClure are redescending: ρ′ → 0, so a far-enough outlier is
 * not merely capped but switched off — and that is exactly what makes them
 * non-convex.
 */
export function kernelWeight(kernel: Kernel, e: number): number {
  const a = Math.abs(e);
  if (a < 1e-12) return 1;
  switch (kernel.type) {
    case 'l2':
      return 1;
    case 'huber':
      return a <= kernel.k ? 1 : kernel.k / a;
    case 'cauchy': {
      const c2 = kernel.c * kernel.c;
      return 1 / (1 + (a * a) / c2);
    }
    case 'geman': {
      const c2 = kernel.c * kernel.c;
      const d = c2 + a * a;
      return (c2 * c2) / (d * d);
    }
  }
}

/**
 * The influence function ψ(e) = ρ′(e) = w(e)·e — how hard one residual pulls on
 * the solution. Plotting this is the fastest way to see why L2 is fragile: its
 * influence is the identity, unbounded in both directions.
 */
export function kernelInfluence(kernel: Kernel, e: number): number {
  return kernelWeight(kernel, e) * e;
}
