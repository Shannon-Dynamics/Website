/**
 * The Bayes filter itself, plus its most literal implementation.
 *
 * Every other filter in this directory is the same two lines —
 *
 *   predict:  b̄(x') = ∫ p(x' | u, x) b(x) dx
 *   correct:  b(x') = η p(z | x') b̄(x')
 *
 * — with a different way of writing down b. The histogram filter writes it as a
 * list of numbers, which is why you can *see* the integral happening.
 */

import { discreteEntropy } from '../prob/gaussian';

/** The contract every filter in this book satisfies. */
export interface BayesFilter<B, U, Z> {
  predict(u: U): void;
  correct(z: Z): void;
  belief(): B;
}

export interface HistogramOptions {
  /** Corridor length in metres. */
  length: number;
  /** Number of equal-width cells. */
  cells: number;
  /** Treat the corridor as a loop, so mass leaving one end re-enters the other. */
  wrap?: boolean;
}

/**
 * Discrete Bayes filter over a 1-D corridor — Thrun et al., **Table 2.1**.
 *
 * The belief is a plain array of cell probabilities. Prediction is a discrete
 * convolution with the motion kernel; correction is a pointwise multiply and a
 * renormalisation. No Gaussians, no linearisation, no approximations beyond the
 * discretisation itself — which makes this the reference every later filter is
 * measured against.
 */
export class HistogramFilter1D implements BayesFilter<number[], number, (x: number) => number> {
  readonly length: number;
  readonly n: number;
  readonly cellWidth: number;
  readonly wrap: boolean;
  cells: number[];
  /** Normaliser η from the last `correct` — the evidence p(z), useful for plots. */
  lastEvidence = 1;

  constructor(opts: HistogramOptions) {
    this.length = opts.length;
    this.n = opts.cells;
    this.cellWidth = opts.length / opts.cells;
    this.wrap = opts.wrap ?? false;
    this.cells = new Array<number>(this.n).fill(1 / this.n);
  }

  /** Centre of cell `i` in metres. */
  cellCenter(i: number): number {
    return (i + 0.5) * this.cellWidth;
  }

  /** All cell centres — the x-axis of every histogram plot. */
  centers(): number[] {
    return Array.from({ length: this.n }, (_, i) => this.cellCenter(i));
  }

  /** Replace the belief with a normalised copy of `p`. */
  setBelief(p: number[]): void {
    this.cells = p.slice(0, this.n);
    this.normalize();
  }

  /** Reset to the uniform prior — maximal ignorance, log₂ n bits of entropy. */
  setUniform(): void {
    this.cells = new Array<number>(this.n).fill(1 / this.n);
  }

  /**
   * Prediction step: b̄(i) = Σⱼ b(j) · kernel(cᵢ − cⱼ − u) · Δx
   *
   * `u` is the commanded displacement in metres and `kernel(e)` is the density
   * of the displacement *error* e. Multiplying by the cell width Δx converts
   * the density into the probability mass of landing in cell i — forget it and
   * the belief is scaled by 1/Δx (invisible after normalisation, but wrong if
   * you ever read the numbers).
   *
   * O(n²) on purpose: the double loop is the integral, written out.
   *
   * Omitting the kernel gives noise-free odometry: a discrete delta one cell
   * wide, which shifts the belief without blurring it. (It also keeps the
   * signature assignable to `BayesFilter.predict`.)
   */
  predict(u: number, kernel?: (delta: number) => number): void {
    const k =
      kernel ?? ((d: number) => (Math.abs(d) < this.cellWidth / 2 ? 1 / this.cellWidth : 0));
    const out = new Array<number>(this.n).fill(0);
    for (let i = 0; i < this.n; i++) {
      let s = 0;
      for (let j = 0; j < this.n; j++) {
        if (this.cells[j] === 0) continue;
        let d = this.cellCenter(i) - this.cellCenter(j) - u;
        if (this.wrap) {
          // Shortest displacement around the loop.
          d -= this.length * Math.round(d / this.length);
        }
        s += this.cells[j] * k(d) * this.cellWidth;
      }
      out[i] = s;
    }
    this.cells = out;
    this.normalize();
  }

  /** Correction step: multiply by the measurement likelihood, then renormalise. */
  correct(likelihood: (cellCenter: number) => number): void {
    let total = 0;
    for (let i = 0; i < this.n; i++) {
      this.cells[i] *= likelihood(this.cellCenter(i));
      total += this.cells[i];
    }
    this.lastEvidence = total;
    this.normalize();
  }

  private normalize(): void {
    let total = 0;
    for (const c of this.cells) total += c;
    if (total > 0) {
      for (let i = 0; i < this.n; i++) this.cells[i] /= total;
    } else {
      // Total measurement failure: fall back to ignorance rather than NaN.
      this.setUniform();
    }
  }

  belief(): number[] {
    return this.cells.slice();
  }

  /** Expected position, Σ p(i) · cᵢ. Meaningless when the belief is multimodal. */
  mean(): number {
    let s = 0;
    for (let i = 0; i < this.n; i++) s += this.cells[i] * this.cellCenter(i);
    return s;
  }

  /** Position of the most probable cell — the MAP estimate. */
  mode(): number {
    let best = 0;
    for (let i = 1; i < this.n; i++) if (this.cells[i] > this.cells[best]) best = i;
    return this.cellCenter(best);
  }

  /** Belief entropy in **bits**: log₂ n when uniform, → 0 as it collapses. */
  entropy(): number {
    return discreteEntropy(this.cells);
  }
}

/** Gaussian motion kernel for {@link HistogramFilter1D.predict}. */
export function gaussianKernel(sigma: number): (delta: number) => number {
  const v = Math.max(sigma * sigma, 1e-12);
  const norm = 1 / Math.sqrt(2 * Math.PI * v);
  return (d: number) => norm * Math.exp(-0.5 * (d * d) / v);
}
