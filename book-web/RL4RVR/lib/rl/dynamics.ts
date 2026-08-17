/**
 * Learned dynamics models and CEM planning — the machinery of Chapter 12.
 *
 * The ensemble here is genuinely trained on collected transitions, so the fan
 * a reader sees in the imagination widget is real model disagreement rather
 * than a drawn shape. Members differ by initialization and by bootstrap
 * resampling of the data, which is exactly how PETS-style ensembles work.
 */

import { Mlp, Standardizer } from './nn';
import type { Rng } from './random';
import { gaussian, mulberry32 } from './random';

export interface Transition {
  state: number[];
  action: number[];
  next: number[];
}

/**
 * An ensemble of one-step models predicting the state DELTA rather than the
 * next state — a standard trick, since deltas are small and better centred.
 */
export class DynamicsEnsemble {
  private members: Mlp[] = [];
  private scaler = new Standardizer();
  /** Deltas are also standardized; otherwise the loss is dominated by
   *  whichever state coordinate happens to have the largest units. */
  private deltaScaler = new Standardizer();
  private trained = false;

  constructor(
    private stateDim: number,
    private actionDim: number,
    private nMembers = 5,
    private hidden = 24,
    seed = 7,
  ) {
    for (let m = 0; m < nMembers; m++) {
      const rng = mulberry32(seed + m * 977);
      this.members.push(new Mlp([stateDim + actionDim, hidden, hidden, stateDim], rng));
    }
  }

  get size(): number {
    return this.nMembers;
  }

  /** Train every member on a bootstrap resample of the data. */
  fit(data: Transition[], epochs = 60, lr = 0.02, batch = 32, seed = 11): number[] {
    const xs = data.map((t) => [...t.state, ...t.action]);
    const deltas = data.map((t) => t.next.map((v, k) => v - t.state[k]));
    this.scaler.fit(xs);
    this.deltaScaler.fit(deltas);

    const losses: number[] = [];
    for (let m = 0; m < this.members.length; m++) {
      const rng = mulberry32(seed + m * 31);
      // Bootstrap: sample with replacement so members see different data.
      const idx = Array.from({ length: data.length }, () =>
        Math.floor(rng() * data.length),
      );
      let last = 0;
      for (let e = 0; e < epochs; e++) {
        for (let b = 0; b < idx.length; b += batch) {
          const slice = idx.slice(b, b + batch);
          if (slice.length === 0) continue;
          const bx = slice.map((i) => this.scaler.apply([...data[i].state, ...data[i].action]));
          const by = slice.map((i) =>
            this.deltaScaler.apply(data[i].next.map((v, k) => v - data[i].state[k])),
          );
          last = this.members[m].trainBatch(bx, by, lr);
        }
      }
      losses.push(last);
    }
    this.trained = true;
    return losses;
  }

  /** One-step prediction from a single member. */
  step(member: number, state: number[], action: number[]): number[] {
    if (!this.trained) return state.slice();
    const norm = this.members[member].forward(this.scaler.apply([...state, ...action]));
    // Undo the target standardization to get a physical delta back.
    const delta = norm.map((v, i) => v * this.deltaScaler.std[i] + this.deltaScaler.mean[i]);
    return state.map((v, i) => v + delta[i]);
  }

  /**
   * Roll every member forward from the same start, returning one trajectory
   * each. The spread across trajectories is the epistemic uncertainty that
   * Chapter 12 says should decide the planning horizon.
   */
  rolloutAll(
    start: number[],
    actions: number[][],
  ): { trajectories: number[][][]; spread: number[] } {
    const trajectories: number[][][] = [];
    for (let m = 0; m < this.members.length; m++) {
      let s = start.slice();
      const traj = [s.slice()];
      for (const a of actions) {
        s = this.step(m, s, a);
        if (!s.every(Number.isFinite)) break;
        traj.push(s.slice());
      }
      trajectories.push(traj);
    }

    const horizon = Math.min(...trajectories.map((t) => t.length));
    const spread: number[] = [];
    for (let h = 0; h < horizon; h++) {
      // Standard deviation of the first state coordinate across members.
      const vals = trajectories.map((t) => t[h][0]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      spread.push(
        Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length),
      );
    }
    return { trajectories, spread };
  }
}

export interface CemConfig {
  horizon: number;
  population: number;
  elites: number;
  iterations: number;
  actionLow: number;
  actionHigh: number;
}

/**
 * Cross-entropy method over action sequences: sample, keep the elites, refit
 * the sampling distribution, repeat. Chapter 12 derives this as importance-
 * sampled optimization; here it plans against whatever step function it is
 * handed, learned or exact.
 */
export function cemPlan(
  start: number[],
  step: (s: number[], a: number[]) => number[],
  reward: (s: number[], a: number[]) => number,
  cfg: CemConfig,
  rng: Rng,
): { action: number[]; best: number[][]; bestReturn: number } {
  const H = cfg.horizon;
  let mean = new Array(H).fill(0);
  let std = new Array(H).fill((cfg.actionHigh - cfg.actionLow) / 3);
  let best: number[][] = [];
  let bestReturn = -Infinity;

  for (let iter = 0; iter < cfg.iterations; iter++) {
    const candidates: Array<{ ret: number; seq: number[] }> = [];

    for (let p = 0; p < cfg.population; p++) {
      const seq = mean.map((m, i) =>
        Math.max(cfg.actionLow, Math.min(cfg.actionHigh, m + std[i] * gaussian(rng))),
      );
      let s = start.slice();
      let ret = 0;
      const traj: number[][] = [s.slice()];
      for (let t = 0; t < H; t++) {
        const a = [seq[t]];
        ret += reward(s, a);
        s = step(s, a);
        if (!s.every(Number.isFinite)) {
          ret -= 1e6;
          break;
        }
        traj.push(s.slice());
      }
      candidates.push({ ret, seq });
      if (ret > bestReturn) {
        bestReturn = ret;
        best = traj;
      }
    }

    candidates.sort((a, b) => b.ret - a.ret);
    const elite = candidates.slice(0, cfg.elites);
    mean = mean.map((_, i) => elite.reduce((acc, e) => acc + e.seq[i], 0) / elite.length);
    std = std.map((_, i) => {
      const m = mean[i];
      const v = elite.reduce((acc, e) => acc + (e.seq[i] - m) ** 2, 0) / elite.length;
      // A floor keeps the search from collapsing before it has converged.
      return Math.max(Math.sqrt(v), 0.05);
    });
  }

  return { action: [mean[0]], best, bestReturn };
}
