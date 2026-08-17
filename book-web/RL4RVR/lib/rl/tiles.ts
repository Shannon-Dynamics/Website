/**
 * Tile coding and semi-gradient SARSA — Chapter 8's workhorse.
 *
 * Several widgets need a policy that genuinely learns on a continuous state
 * space but trains in well under a second. Tile coding does exactly that: the
 * features are sparse and binary, so an update touches only `nTilings` weights
 * regardless of how large the table is.
 */

import type { Rng } from './random';
import { argmaxRandomTie } from './random';

export interface TileCoderConfig {
  /** Per-dimension [low, high] bounds. */
  bounds: Array<[number, number]>;
  /** Number of offset tilings; resolution finer than any single tiling's width. */
  nTilings: number;
  /** Tiles per dimension within one tiling. */
  tilesPerDim: number;
}

export class TileCoder {
  readonly nTilings: number;
  readonly tilesPerDim: number;
  private bounds: Array<[number, number]>;
  private offsets: number[][];
  readonly nFeatures: number;

  constructor(cfg: TileCoderConfig) {
    this.bounds = cfg.bounds;
    this.nTilings = cfg.nTilings;
    this.tilesPerDim = cfg.tilesPerDim;
    const d = cfg.bounds.length;

    // Asymmetric offsets (the odd-number rule) spread tilings evenly.
    this.offsets = Array.from({ length: cfg.nTilings }, (_, t) =>
      Array.from({ length: d }, (_, i) => (t * (2 * i + 1)) / cfg.nTilings),
    );
    this.nFeatures = cfg.nTilings * Math.pow(cfg.tilesPerDim, d);
  }

  /** Indices of the active features — exactly `nTilings` of them. */
  active(state: number[], out?: Int32Array): Int32Array {
    const idx = out ?? new Int32Array(this.nTilings);
    for (let t = 0; t < this.nTilings; t++) {
      let flat = 0;
      for (let i = 0; i < state.length; i++) {
        const [lo, hi] = this.bounds[i];
        const scaled = ((state[i] - lo) / (hi - lo)) * this.tilesPerDim;
        let c = Math.floor(scaled + this.offsets[t][i]);
        c = Math.max(0, Math.min(this.tilesPerDim - 1, c));
        flat = flat * this.tilesPerDim + c;
      }
      idx[t] = t * Math.pow(this.tilesPerDim, state.length) + flat;
    }
    return idx;
  }
}

/**
 * Semi-gradient SARSA with binary features. The weight vector is per action,
 * so `q(s,a)` is a sum over the active tiles of action `a`'s slice.
 */
export class TileSarsa {
  readonly w: Float64Array;
  private scratch: Int32Array;

  constructor(
    private coder: TileCoder,
    private nActions: number,
    private alpha = 0.5,
    private gamma = 0.99,
  ) {
    this.w = new Float64Array(coder.nFeatures * nActions);
    this.scratch = new Int32Array(coder.nTilings);
  }

  q(state: number[], action: number): number {
    const idx = this.coder.active(state, this.scratch);
    const base = action * this.coder.nFeatures;
    let s = 0;
    for (let k = 0; k < idx.length; k++) s += this.w[base + idx[k]];
    return s;
  }

  qAll(state: number[]): number[] {
    const idx = this.coder.active(state, this.scratch);
    const out = new Array<number>(this.nActions).fill(0);
    for (let a = 0; a < this.nActions; a++) {
      const base = a * this.coder.nFeatures;
      for (let k = 0; k < idx.length; k++) out[a] += this.w[base + idx[k]];
    }
    return out;
  }

  act(state: number[], epsilon: number, rng: Rng): number {
    if (rng() < epsilon) return Math.floor(rng() * this.nActions);
    return argmaxRandomTie(this.qAll(state), rng);
  }

  /** One SARSA update; returns the TD error. */
  update(
    state: number[],
    action: number,
    reward: number,
    next: number[] | null,
    nextAction: number,
  ): number {
    const idx = Int32Array.from(this.coder.active(state));
    const base = action * this.coder.nFeatures;

    let current = 0;
    for (let k = 0; k < idx.length; k++) current += this.w[base + idx[k]];

    const target = next === null ? reward : reward + this.gamma * this.q(next, nextAction);
    const delta = target - current;

    // With n active tiles the effective step is alpha/n, which keeps the
    // update size independent of how many tilings are configured.
    const step = (this.alpha / this.coder.nTilings) * delta;
    for (let k = 0; k < idx.length; k++) this.w[base + idx[k]] += step;
    return delta;
  }
}
