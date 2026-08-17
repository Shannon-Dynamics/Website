/**
 * A minimal neural network: dense layers, tanh/ReLU, and SGD with momentum.
 *
 * This exists so the browser simulations can train *actual* function
 * approximators rather than display fitted curves. It is deliberately tiny —
 * enough for behaviour cloning on a two-dimensional control problem or a
 * one-step dynamics model, not for anything resembling deep RL. The book's
 * real deep-learning code lives in the Rust crates.
 */

import type { Rng } from './random';
import { gaussian } from './random';

export type Activation = 'tanh' | 'relu' | 'linear';

interface Layer {
  w: Float64Array; // [out * in]
  b: Float64Array; // [out]
  vw: Float64Array; // momentum buffers
  vb: Float64Array;
  nIn: number;
  nOut: number;
  act: Activation;
}

function applyAct(x: number, act: Activation): number {
  if (act === 'tanh') return Math.tanh(x);
  if (act === 'relu') return x > 0 ? x : 0;
  return x;
}

/** Derivative expressed in terms of the *output* value, which we already have. */
function actGrad(y: number, act: Activation): number {
  if (act === 'tanh') return 1 - y * y;
  if (act === 'relu') return y > 0 ? 1 : 0;
  return 1;
}

export class Mlp {
  readonly layers: Layer[] = [];

  constructor(sizes: number[], rng: Rng, hidden: Activation = 'tanh', output: Activation = 'linear') {
    for (let i = 0; i < sizes.length - 1; i++) {
      const nIn = sizes[i];
      const nOut = sizes[i + 1];
      const w = new Float64Array(nIn * nOut);
      // He/Xavier-ish scaling keeps early activations in a sane range.
      const scale = Math.sqrt(2 / (nIn + nOut));
      for (let k = 0; k < w.length; k++) w[k] = gaussian(rng, 0, scale);
      this.layers.push({
        w,
        b: new Float64Array(nOut),
        vw: new Float64Array(nIn * nOut),
        vb: new Float64Array(nOut),
        nIn,
        nOut,
        act: i === sizes.length - 2 ? output : hidden,
      });
    }
  }

  /** Forward pass, retaining activations so `backward` can use them. */
  forward(x: number[], cache?: number[][]): number[] {
    let a = x;
    cache?.push(a);
    for (const L of this.layers) {
      const out = new Array<number>(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o];
        const row = o * L.nIn;
        for (let i = 0; i < L.nIn; i++) s += L.w[row + i] * a[i];
        out[o] = applyAct(s, L.act);
      }
      a = out;
      cache?.push(a);
    }
    return a;
  }

  /**
   * One SGD step on the mean squared error of a minibatch.
   * Returns the batch loss so callers can stream a learning curve.
   */
  trainBatch(xs: number[][], ys: number[][], lr: number, momentum = 0.9): number {
    const gradsW = this.layers.map((L) => new Float64Array(L.w.length));
    const gradsB = this.layers.map((L) => new Float64Array(L.b.length));
    let loss = 0;

    for (let n = 0; n < xs.length; n++) {
      const cache: number[][] = [];
      const pred = this.forward(xs[n], cache);

      // dL/d(output) for squared error.
      let delta = pred.map((p, i) => {
        const e = p - ys[n][i];
        loss += e * e;
        return 2 * e;
      });

      for (let li = this.layers.length - 1; li >= 0; li--) {
        const L = this.layers[li];
        const aIn = cache[li];
        const aOut = cache[li + 1];

        // Push through the activation.
        const d = delta.map((g, o) => g * actGrad(aOut[o], L.act));

        for (let o = 0; o < L.nOut; o++) {
          gradsB[li][o] += d[o];
          const row = o * L.nIn;
          for (let i = 0; i < L.nIn; i++) gradsW[li][row + i] += d[o] * aIn[i];
        }

        if (li > 0) {
          const next = new Array<number>(L.nIn).fill(0);
          for (let o = 0; o < L.nOut; o++) {
            const row = o * L.nIn;
            for (let i = 0; i < L.nIn; i++) next[i] += L.w[row + i] * d[o];
          }
          delta = next;
        }
      }
    }

    const scale = 1 / xs.length;
    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li];
      for (let k = 0; k < L.w.length; k++) {
        L.vw[k] = momentum * L.vw[k] - lr * scale * gradsW[li][k];
        L.w[k] += L.vw[k];
      }
      for (let k = 0; k < L.b.length; k++) {
        L.vb[k] = momentum * L.vb[k] - lr * scale * gradsB[li][k];
        L.b[k] += L.vb[k];
      }
    }
    return loss / (xs.length * (ys[0]?.length ?? 1));
  }
}

/** Standardizes inputs; unnormalized joint angles and velocities train badly. */
export class Standardizer {
  mean: number[] = [];
  std: number[] = [];

  fit(xs: number[][]): void {
    const d = xs[0].length;
    this.mean = new Array(d).fill(0);
    this.std = new Array(d).fill(1);
    for (const x of xs) for (let i = 0; i < d; i++) this.mean[i] += x[i] / xs.length;
    for (let i = 0; i < d; i++) {
      let v = 0;
      for (const x of xs) v += (x[i] - this.mean[i]) ** 2;
      this.std[i] = Math.sqrt(v / xs.length) || 1;
    }
  }

  apply(x: number[]): number[] {
    return x.map((v, i) => (v - this.mean[i]) / this.std[i]);
  }
}
