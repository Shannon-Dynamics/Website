/**
 * The **information filter** — Thrun et al., *Probabilistic Robotics*, Table 3.4.
 *
 * The same Gaussian belief as {@link Kf}, written in the other coordinate
 * system: instead of moments $(\mu, \Sigma)$ it carries the canonical (natural,
 * information) parameters
 *
 *   Ω = Σ⁻¹        the information matrix — curvature of the log-belief
 *   ξ = Σ⁻¹ μ      the information vector
 *
 * Same posterior, transposed costs. Correction becomes literal addition —
 * fusing two sensors is `Ω += H₁ᵀR₁⁻¹H₁ + H₂ᵀR₂⁻¹H₂` — while prediction now
 * pays for two inversions. Chapter 15 cashes this in: when Ω is *sparse*, the
 * additive side is the only side worth being on.
 */

import {
  add,
  inv,
  matAdd,
  matMul,
  matVec,
  symmetrize,
  transpose,
  type Mat,
  type Vec,
} from '../prob/linalg';
import type { Gaussian } from './kf';

/** A Gaussian in canonical form: Ω = Σ⁻¹, ξ = Σ⁻¹μ. */
export interface Canonical {
  xi: Vec;
  Omega: Mat;
}

/** (μ, Σ) → (ξ, Ω). One inversion; the bijection Thrun calls functionally equivalent. */
export function toCanonical(g: Gaussian): Canonical {
  const Omega = symmetrize(inv(g.P));
  return { xi: matVec(Omega, g.x), Omega };
}

/** (ξ, Ω) → (μ, Σ). The other direction, and the same one inversion. */
export function toMoments(c: Canonical): Gaussian {
  const P = symmetrize(inv(c.Omega));
  return { x: matVec(P, c.xi), P };
}

/**
 * The linear information filter.
 *
 * Line numbering below follows Thrun Table 3.4 so the code and the printed
 * algorithm can be read side by side.
 */
export class InfoFilter {
  xi: Vec;
  Omega: Mat;

  constructor(xi: Vec, Omega: Mat) {
    this.xi = xi.slice();
    this.Omega = Omega.map((r) => r.slice());
  }

  /** Start from a moments-form belief — the usual way a filter is initialized. */
  static fromMoments(g: Gaussian): InfoFilter {
    const c = toCanonical(g);
    return new InfoFilter(c.xi, c.Omega);
  }

  /**
   * Table 3.4, lines 2–3 — the *expensive* step here:
   *
   *   Ω̄ = (F Ω⁻¹ Fᵀ + Q)⁻¹
   *   ξ̄ = Ω̄ (F Ω⁻¹ ξ + B u)
   *
   * Two n×n inversions, because propagating uncertainty forward is natural in
   * covariance and unnatural in information. Compare `Kf.predictWith`, which
   * does the same job with one matrix product.
   */
  predictWith(F: Mat, Q: Mat, B?: Mat | null, u?: Vec | null): void {
    const Sigma = inv(this.Omega);
    const mu = matVec(Sigma, this.xi);

    const Pbar = symmetrize(matAdd(matMul(matMul(F, Sigma), transpose(F)), Q));
    let muBar = matVec(F, mu);
    if (B && u) muBar = add(muBar, matVec(B, u));

    this.Omega = symmetrize(inv(Pbar));
    this.xi = matVec(this.Omega, muBar);
  }

  /**
   * Table 3.4, lines 4–5 — the *cheap* step here, and the whole point:
   *
   *   Ω ← Ω + HᵀR⁻¹H
   *   ξ ← ξ + HᵀR⁻¹z
   *
   * No gain, no innovation covariance, no n×n inversion. Independent sensors
   * commute and simply accumulate, which is why multi-sensor fusion in
   * information form is addition and nothing else.
   */
  correctWith(z: Vec, H: Mat, R: Mat): void {
    const Ht = transpose(H);
    const HtRinv = matMul(Ht, inv(R));
    this.Omega = symmetrize(matAdd(this.Omega, matMul(HtRinv, H)));
    this.xi = add(this.xi, matVec(HtRinv, z));
  }

  /** The measurement's own contribution, for widgets that want to draw it. */
  static contribution(z: Vec, H: Mat, R: Mat): Canonical {
    const HtRinv = matMul(transpose(H), inv(R));
    return { xi: matVec(HtRinv, z), Omega: symmetrize(matMul(HtRinv, H)) };
  }

  /** The same belief, read back in moments form. */
  belief(): Gaussian {
    return toMoments({ xi: this.xi, Omega: this.Omega });
  }

  canonical(): Canonical {
    return { xi: this.xi.slice(), Omega: this.Omega.map((r) => r.slice()) };
  }

  clone(): InfoFilter {
    return new InfoFilter(this.xi, this.Omega);
  }
}

/**
 * An order-of-magnitude flop ledger for the duality table — multiply–add
 * counts for dense, unstructured matrices, not an instruction count.
 *
 * `n` is the state dimension, `m` the dimension of one measurement, `sensors`
 * how many independent measurements are folded in per step. Inversions are
 * charged at d³ (Gauss–Jordan on a small dense block, which is what both the
 * Rust and the TypeScript here actually do).
 */
export function flopLedger(n: number, m: number, sensors = 1) {
  const momentsPredict = 2 * n ** 3 + 2 * n ** 2;
  const momentsCorrect =
    sensors * (2 * n ** 3 + 4 * m * n ** 2 + 2 * m ** 2 * n + m ** 3);

  const infoPredict = 4 * n ** 3 + 2 * n ** 2;
  const infoCorrect = sensors * (m ** 3 + m * n ** 2 + m ** 2 * n + n ** 2);

  return { momentsPredict, momentsCorrect, infoPredict, infoCorrect };
}
