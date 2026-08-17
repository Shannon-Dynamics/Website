/**
 * Model-free learning (Chapters 6–7): Monte Carlo, TD(0), SARSA, Q-learning,
 * Double Q-learning, and eligibility traces.
 *
 * Every learner exposes `stepEpisode()` so a dashboard can drive training one
 * episode at a time and stream the quantities the math named — the TD error
 * δ_t above all.
 */

import type { Action } from './gridworld';
import { GridWorld } from './gridworld';
import type { Rng } from './random';
import { argmaxRandomTie } from './random';

export interface EpisodeStats {
  episode: number;
  totalReward: number;
  steps: number;
  /** Mean |δ| over the episode — the learning signal's magnitude. */
  meanAbsDelta: number;
  epsilon: number;
  /** Visited states, for the trajectory overlay. */
  path: number[];
}

export type LearnerKind = 'sarsa' | 'qlearning' | 'expected-sarsa' | 'double-q' | 'sarsa-lambda';

export interface LearnerOptions {
  alpha?: number;
  epsilon?: number;
  epsilonDecay?: number;
  epsilonMin?: number;
  lambda?: number;
  maxSteps?: number;
}

/**
 * Tabular control learner. Q is flat: index = state * nActions + action.
 */
export class TabularLearner {
  readonly Q: Float64Array;
  /** Second table for Double Q-learning; unused otherwise. */
  readonly Q2: Float64Array;
  readonly visits: Float64Array;
  private traces: Float64Array;
  private episode = 0;
  epsilon: number;

  constructor(
    private env: GridWorld,
    readonly kind: LearnerKind,
    private opts: LearnerOptions = {},
  ) {
    const n = env.nStates * env.nActions;
    this.Q = new Float64Array(n);
    this.Q2 = new Float64Array(n);
    this.visits = new Float64Array(env.nStates);
    this.traces = new Float64Array(n);
    this.epsilon = opts.epsilon ?? 0.1;
  }

  private idx(s: number, a: number): number {
    return s * this.env.nActions + a;
  }

  /** Q-values for a state, as an array over actions. */
  qRow(s: number, table: Float64Array = this.Q): number[] {
    const out: number[] = [];
    for (let a = 0; a < this.env.nActions; a++) out.push(table[this.idx(s, a)]);
    return out;
  }

  /** Combined estimate — Double Q averages its two tables. */
  qCombined(s: number): number[] {
    if (this.kind !== 'double-q') return this.qRow(s);
    return this.qRow(s).map((v, a) => (v + this.Q2[this.idx(s, a)]) / 2);
  }

  /** State-value view V(s) = max_a Q(s,a), for the heatmap. */
  valueFunction(): Float64Array {
    const V = new Float64Array(this.env.nStates);
    for (const s of this.env.states) {
      V[s] = this.env.isTerminal(s) ? 0 : Math.max(...this.qCombined(s));
    }
    return V;
  }

  greedyPolicy(): Int8Array {
    const policy = new Int8Array(this.env.nStates).fill(-1);
    for (const s of this.env.states) {
      if (this.env.isTerminal(s)) continue;
      const row = this.qCombined(s);
      let best = 0;
      for (let a = 1; a < row.length; a++) if (row[a] > row[best]) best = a;
      policy[s] = best;
    }
    return policy;
  }

  private epsilonGreedy(s: number, rng: Rng): Action {
    if (rng() < this.epsilon) return Math.floor(rng() * this.env.nActions) as Action;
    return argmaxRandomTie(this.qCombined(s), rng) as Action;
  }

  /** Probabilities of the ε-greedy policy — needed by Expected SARSA. */
  private policyProbs(s: number): number[] {
    const n = this.env.nActions;
    const row = this.qCombined(s);
    let best = 0;
    for (let a = 1; a < n; a++) if (row[a] > row[best]) best = a;
    return Array.from({ length: n }, (_, a) =>
      (a === best ? 1 - this.epsilon : 0) + this.epsilon / n,
    );
  }

  /** Run one full episode, applying updates online. */
  stepEpisode(rng: Rng): EpisodeStats {
    const { alpha = 0.1, maxSteps = 400, lambda = 0.9 } = this.opts;
    const { gamma } = this.env.config;
    const env = this.env;

    let s = env.startState;
    let a = this.epsilonGreedy(s, rng);
    let totalReward = 0;
    let absDeltaSum = 0;
    let steps = 0;
    const path = [s];
    if (this.kind === 'sarsa-lambda') this.traces.fill(0);

    for (; steps < maxSteps; steps++) {
      const t = env.step(s, a, rng);
      totalReward += t.reward;
      this.visits[s] += 1;
      const sNext = t.next;
      const aNext = t.done ? (0 as Action) : this.epsilonGreedy(sNext, rng);

      let target: number;
      if (t.done) {
        target = t.reward;
      } else if (this.kind === 'sarsa' || this.kind === 'sarsa-lambda') {
        target = t.reward + gamma * this.Q[this.idx(sNext, aNext)];
      } else if (this.kind === 'expected-sarsa') {
        const probs = this.policyProbs(sNext);
        const row = this.qRow(sNext);
        target = t.reward + gamma * probs.reduce((acc, p, i) => acc + p * row[i], 0);
      } else if (this.kind === 'double-q') {
        // Decouple selection from evaluation — the fix for maximization bias.
        const useFirst = rng() < 0.5;
        const selectTable = useFirst ? this.Q : this.Q2;
        const evalTable = useFirst ? this.Q2 : this.Q;
        const row = this.qRow(sNext, selectTable);
        let best = 0;
        for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
        const delta =
          t.reward + gamma * evalTable[this.idx(sNext, best)] - selectTable[this.idx(s, a)];
        selectTable[this.idx(s, a)] += alpha * delta;
        absDeltaSum += Math.abs(delta);
        s = sNext;
        a = aNext;
        path.push(s);
        if (t.done) break;
        continue;
      } else {
        target = t.reward + gamma * Math.max(...this.qRow(sNext));
      }

      const delta = target - this.Q[this.idx(s, a)];
      absDeltaSum += Math.abs(delta);

      if (this.kind === 'sarsa-lambda') {
        // Accumulating traces: one update per step touches every recently
        // visited (s,a) pair — the cheap implementation of the λ-return.
        this.traces[this.idx(s, a)] += 1;
        for (const st of env.states) {
          for (let ac = 0; ac < env.nActions; ac++) {
            const i = this.idx(st, ac);
            if (this.traces[i] > 1e-4) {
              this.Q[i] += alpha * delta * this.traces[i];
              this.traces[i] *= gamma * lambda;
            }
          }
        }
      } else {
        this.Q[this.idx(s, a)] += alpha * delta;
      }

      s = sNext;
      a = aNext;
      path.push(s);
      if (t.done) break;
    }

    this.episode += 1;
    const decay = this.opts.epsilonDecay ?? 1;
    this.epsilon = Math.max(this.opts.epsilonMin ?? 0.01, this.epsilon * decay);

    return {
      episode: this.episode,
      totalReward,
      steps: steps + 1,
      meanAbsDelta: absDeltaSum / Math.max(1, steps + 1),
      epsilon: this.epsilon,
      path,
    };
  }
}

/**
 * First-visit Monte Carlo prediction — the contrast case for TD.
 * Returns the per-episode value snapshot so the "credit propagation" widget
 * can show MC updating only after the episode ends.
 */
export function monteCarloPrediction(
  env: GridWorld,
  policy: Int8Array,
  rng: Rng,
  episodes: number,
  maxSteps = 400,
): { V: Float64Array; returns: number[] } {
  const V = new Float64Array(env.nStates);
  const counts = new Float64Array(env.nStates);
  const { gamma } = env.config;
  const episodeReturns: number[] = [];

  for (let e = 0; e < episodes; e++) {
    const trajectory: Array<{ s: number; r: number }> = [];
    let s = env.startState;
    for (let i = 0; i < maxSteps; i++) {
      const a = policy[s];
      if (a < 0) break;
      const t = env.step(s, a as Action, rng);
      trajectory.push({ s, r: t.reward });
      s = t.next;
      if (t.done) break;
    }

    let G = 0;
    const seen = new Set<number>();
    let total = 0;
    for (let i = trajectory.length - 1; i >= 0; i--) {
      G = trajectory[i].r + gamma * G;
      total += trajectory[i].r;
      if (!seen.has(trajectory[i].s)) {
        seen.add(trajectory[i].s);
        counts[trajectory[i].s] += 1;
        V[trajectory[i].s] += (G - V[trajectory[i].s]) / counts[trajectory[i].s];
      }
    }
    episodeReturns.push(total);
  }

  return { V, returns: episodeReturns };
}

/**
 * TD(0) prediction — same interface as MC so the widget can race them.
 */
export function tdPrediction(
  env: GridWorld,
  policy: Int8Array,
  rng: Rng,
  episodes: number,
  alpha = 0.1,
  maxSteps = 400,
): { V: Float64Array; returns: number[] } {
  const V = new Float64Array(env.nStates);
  const { gamma } = env.config;
  const episodeReturns: number[] = [];

  for (let e = 0; e < episodes; e++) {
    let s = env.startState;
    let total = 0;
    for (let i = 0; i < maxSteps; i++) {
      const a = policy[s];
      if (a < 0) break;
      const t = env.step(s, a as Action, rng);
      total += t.reward;
      const target = t.done ? t.reward : t.reward + gamma * V[t.next];
      V[s] += alpha * (target - V[s]);
      s = t.next;
      if (t.done) break;
    }
    episodeReturns.push(total);
  }

  return { V, returns: episodeReturns };
}

/** Exponentially weighted moving average, for smoothing noisy learning curves. */
export function smooth(values: number[], window = 20): number[] {
  if (values.length === 0) return [];
  const alpha = 2 / (window + 1);
  const out = [values[0]];
  for (let i = 1; i < values.length; i++) {
    out.push(alpha * values[i] + (1 - alpha) * out[i - 1]);
  }
  return out;
}
