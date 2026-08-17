/**
 * Differentiable volume rendering, in one dimension.
 *
 * A radiance field is a *generative model of measurements*: given densities
 * σ(x) along a ray, the rendering integral predicts what the sensor will
 * report. With Gaussian photometric noise, −log p(z | σ) is a squared residual,
 * so fitting a NeRF is MAP estimation under a rendering measurement model —
 * the same η·p(z | m, x) structure as occupancy mapping, optimized by gradients
 * instead of closed-form updates.
 *
 * Stripping the model to one spatial dimension and one channel (expected depth)
 * leaves the mechanism completely visible: alpha compositing, transmittance,
 * the analytic gradient that autodiff would otherwise hand you, and the
 * ambiguity that makes the whole thing fragile.
 */

import { Rng } from '../prob/rng';

export interface Camera1D {
  /** Position along the line, metres. */
  x: number;
  /** +1 looks toward increasing x. */
  dir: 1 | -1;
}

export interface VolumeToyOptions {
  /** Domain length, metres. */
  length?: number;
  bins?: number;
  /** True surface positions. */
  walls?: number[];
  /** Cameras whose depths the optimizer fits. */
  train?: Camera1D[];
  /** Cameras it never sees — the honesty check. */
  heldOut?: Camera1D[];
  /** Std-dev of the depth measurement noise, metres. */
  noise?: number;
  /** Density of a true wall bin; τ·δ ≈ 5 makes it essentially opaque. */
  wallDensity?: number;
  seed?: number;
}

export interface RayGeometry {
  /** Bin indices in the order the ray meets them. */
  bins: number[];
  /** Distance from the camera to each bin centre. */
  t: number[];
  /** Distance to the far edge of the domain: where an un-blocked ray ends. */
  tFar: number;
}

const softplus = (v: number) => (v > 20 ? v : Math.log1p(Math.exp(v)));
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/**
 * The 1-D toy: densities on a regular grid, cameras at fixed stations, and one
 * gradient step at a time.
 */
export class VolumeToy1D {
  readonly length: number;
  readonly bins: number;
  readonly delta: number;
  readonly walls: number[];
  readonly train: Camera1D[];
  readonly heldOut: Camera1D[];
  readonly noise: number;
  readonly wallDensity: number;

  /** Unconstrained parameters; density is softplus(θ) so σ ≥ 0 by construction. */
  readonly theta: Float64Array;
  /** The depths the optimizer is fitting, drawn once with the given seed. */
  readonly observations: number[];
  readonly heldOutTruth: number[];
  /** Learning-rate schedule state — plain gradient descent, no momentum. */
  iteration = 0;

  constructor(opts: VolumeToyOptions = {}) {
    this.length = opts.length ?? 10;
    this.bins = opts.bins ?? 80;
    this.delta = this.length / this.bins;
    this.walls = opts.walls ?? [3, 7];
    this.train = opts.train ?? [
      { x: 0.4, dir: 1 },
      { x: 5.0, dir: -1 },
      { x: 5.0, dir: 1 },
      { x: 9.6, dir: -1 },
    ];
    this.heldOut = opts.heldOut ?? [
      { x: 2.0, dir: 1 },
      { x: 8.0, dir: -1 },
    ];
    this.noise = opts.noise ?? 0.05;
    this.wallDensity = opts.wallDensity ?? 5 / this.delta;

    const rng = new Rng(opts.seed ?? 7);
    // A small random field, not zero: an exactly flat start is a saddle the
    // optimizer would leave only by symmetry breaking, and readers deserve to
    // see the honest sensitivity to initialization.
    this.theta = new Float64Array(this.bins);
    for (let i = 0; i < this.bins; i++) this.theta[i] = rng.normal(-1.5, 0.8);

    this.observations = this.train.map((c) => this.trueDepth(c) + rng.normal(0, this.noise));
    this.heldOutTruth = this.heldOut.map((c) => this.trueDepth(c));
  }

  /** σ(x) on the bin grid. */
  density(): Float64Array {
    const out = new Float64Array(this.bins);
    for (let i = 0; i < this.bins; i++) out[i] = softplus(this.theta[i]);
    return out;
  }

  /** The density that generated the data, for the gray dashed curve. */
  trueDensity(): Float64Array {
    const out = new Float64Array(this.bins);
    for (const w of this.walls) {
      const i = Math.min(this.bins - 1, Math.max(0, Math.floor(w / this.delta)));
      out[i] = this.wallDensity;
    }
    return out;
  }

  /** Exact geometric depth: distance to the first wall the camera faces. */
  trueDepth(cam: Camera1D): number {
    let best = cam.dir > 0 ? this.length - cam.x : cam.x;
    for (const w of this.walls) {
      const s = (w - cam.x) * cam.dir;
      if (s > 0 && s < best) best = s;
    }
    return best;
  }

  /** Which bins a ray traverses, and how far away each one is. */
  rayGeometry(cam: Camera1D): RayGeometry {
    const bins: number[] = [];
    const t: number[] = [];
    if (cam.dir > 0) {
      for (let i = 0; i < this.bins; i++) {
        const c = (i + 0.5) * this.delta;
        if (c > cam.x) {
          bins.push(i);
          t.push(c - cam.x);
        }
      }
      return { bins, t, tFar: this.length - cam.x };
    }
    for (let i = this.bins - 1; i >= 0; i--) {
      const c = (i + 0.5) * this.delta;
      if (c < cam.x) {
        bins.push(i);
        t.push(cam.x - c);
      }
    }
    return { bins, t, tFar: cam.x };
  }

  /**
   * The rendering equation, discretized — Max (1995), as used by NeRF.
   *
   *   α_i = 1 − exp(−σ_i δ),   T_i = Π_{j<i} (1 − α_j),
   *   ẑ  = Σ_i T_i α_i t_i + T_n t_far
   *
   * The last term is the background: a ray that gets through everything
   * reports the far plane, which is what a max-range LiDAR return means too.
   */
  renderDepth(cam: Camera1D, sigma: Float64Array = this.density()): number {
    const { bins, t, tFar } = this.rayGeometry(cam);
    let T = 1;
    let z = 0;
    for (let n = 0; n < bins.length; n++) {
      const alpha = 1 - Math.exp(-sigma[bins[n]] * this.delta);
      z += T * alpha * t[n];
      T *= 1 - alpha;
    }
    return z + T * tFar;
  }

  /** Σ (ẑ − z)² / (2σ_z²) over the training rays: the negative log likelihood. */
  loss(sigma: Float64Array = this.density()): number {
    let l = 0;
    for (let k = 0; k < this.train.length; k++) {
      const r = this.renderDepth(this.train[k], sigma) - this.observations[k];
      l += (r * r) / (2 * this.noise * this.noise);
    }
    return l;
  }

  /** The same residual in metres, which is the number worth plotting. */
  residualRms(sigma: Float64Array = this.density()): number {
    let s = 0;
    for (let k = 0; k < this.train.length; k++) {
      const r = this.renderDepth(this.train[k], sigma) - this.observations[k];
      s += r * r;
    }
    return Math.sqrt(s / Math.max(1, this.train.length));
  }

  /** RMS depth error on cameras the optimizer never fitted. */
  heldOutError(sigma: Float64Array = this.density()): number {
    let s = 0;
    for (let k = 0; k < this.heldOut.length; k++) {
      const r = this.renderDepth(this.heldOut[k], sigma) - this.heldOutTruth[k];
      s += r * r;
    }
    return Math.sqrt(s / Math.max(1, this.heldOut.length));
  }

  /**
   * ∂ẑ/∂σ_k, in closed form.
   *
   * Differentiating the composite gives
   *
   *   ∂ẑ/∂σ_k = δ [ T_{k+1} t_k − (Σ_{i>k} T_i α_i t_i + T_n t_far) ]
   *
   * i.e. "adding density here pulls the depth toward t_k and away from
   * everything behind it". Two facts fall straight out: the bracket vanishes
   * once T ≈ 0, so an occluded region receives *no gradient at all*; and the
   * derivative is largest exactly at the current surface, which is why these
   * optimizers refine geometry they already roughly have and struggle to invent
   * geometry they do not.
   */
  depthGradient(cam: Camera1D, sigma: Float64Array, grad: Float64Array): number {
    const { bins, t, tFar } = this.rayGeometry(cam);
    const n = bins.length;
    const T = new Float64Array(n + 1);
    const alpha = new Float64Array(n);
    T[0] = 1;
    for (let i = 0; i < n; i++) {
      alpha[i] = 1 - Math.exp(-sigma[bins[i]] * this.delta);
      T[i + 1] = T[i] * (1 - alpha[i]);
    }
    let z = 0;
    for (let i = 0; i < n; i++) z += T[i] * alpha[i] * t[i];
    z += T[n] * tFar;

    // Backward accumulation of the tail Σ_{i≥k} T_i α_i t_i + T_n t_far.
    let tail = T[n] * tFar;
    for (let k = n - 1; k >= 0; k--) {
      grad[bins[k]] += this.delta * (T[k + 1] * t[k] - tail);
      tail += T[k] * alpha[k] * t[k];
    }
    return z;
  }

  /**
   * One gradient-descent step on ½Σ(ẑ − z)². Returns the RMS residual before
   * it, in metres.
   *
   * The objective is the negative log likelihood up to the constant 1/σ_z²,
   * which is folded into the learning rate: scaling the step with the assumed
   * noise would make the slider change the optimizer as well as the data, and
   * then the reader could not tell which effect they were looking at.
   */
  stepDescent(lr = 8): number {
    const sigma = this.density();
    const grad = new Float64Array(this.bins);
    const scratch = new Float64Array(this.bins);
    let total = 0;

    for (let k = 0; k < this.train.length; k++) {
      scratch.fill(0);
      const z = this.depthGradient(this.train[k], sigma, scratch);
      const r = z - this.observations[k];
      total += r * r;
      for (let i = 0; i < this.bins; i++) grad[i] += r * scratch[i];
    }

    // Chain through the softplus so the update lives in θ, where σ ≥ 0 is free.
    for (let i = 0; i < this.bins; i++) {
      this.theta[i] -= lr * grad[i] * sigmoid(this.theta[i]);
    }
    this.iteration++;
    return Math.sqrt(total / Math.max(1, this.train.length));
  }
}
