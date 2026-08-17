/**
 * Multi-armed bandits (Chapter 3) — the atom of RL.
 *
 * The testbed follows Sutton & Barto's 10-armed setup: true values q*(a) drawn
 * from N(0,1), rewards drawn from N(q*(a), 1). Robot framing in the chapter:
 * Reacher choosing among grasp primitives, where every pull wears hardware.
 */

import type { Rng } from './random';
import { argmax, argmaxRandomTie, gaussian, sampleBeta, sampleCategorical } from './random';

export interface BanditProblem {
  /** True action values q*(a). */
  qStar: number[];
  optimalAction: number;
}

export function makeBandit(rng: Rng, k = 10, spread = 1): BanditProblem {
  const qStar = Array.from({ length: k }, () => gaussian(rng, 0, spread));
  return { qStar, optimalAction: argmax(qStar) };
}

export function pull(problem: BanditProblem, action: number, rng: Rng): number {
  return gaussian(rng, problem.qStar[action], 1);
}

export interface BanditPolicy {
  readonly name: string;
  selectAction(step: number, rng: Rng): number;
  update(action: number, reward: number): void;
  /** Current estimates, for the live estimate-vs-truth widget. */
  readonly estimates: number[];
}

/** ε-greedy with sample-average (or constant-α) updates. */
export class EpsilonGreedy implements BanditPolicy {
  readonly name: string;
  readonly estimates: number[];
  private counts: number[];

  constructor(
    private k: number,
    private epsilon: number,
    private alpha: number | null = null,
    initial = 0,
  ) {
    this.name = epsilon === 0 ? `greedy (Q₁=${initial})` : `ε-greedy (ε=${epsilon})`;
    this.estimates = new Array(k).fill(initial);
    this.counts = new Array(k).fill(0);
  }

  selectAction(_step: number, rng: Rng): number {
    if (rng() < this.epsilon) return Math.floor(rng() * this.k);
    return argmaxRandomTie(this.estimates, rng);
  }

  update(action: number, reward: number): void {
    this.counts[action] += 1;
    const step = this.alpha ?? 1 / this.counts[action];
    this.estimates[action] += step * (reward - this.estimates[action]);
  }
}

/** Upper-Confidence-Bound action selection: Q(a) + c·√(ln t / N(a)). */
export class Ucb1 implements BanditPolicy {
  readonly name: string;
  readonly estimates: number[];
  private counts: number[];

  constructor(
    private k: number,
    private c = 2,
  ) {
    this.name = `UCB (c=${c})`;
    this.estimates = new Array(k).fill(0);
    this.counts = new Array(k).fill(0);
  }

  /** The exploration bonus, exposed so the widget can draw the interval. */
  bonus(step: number, a: number): number {
    if (this.counts[a] === 0) return Infinity;
    return this.c * Math.sqrt(Math.log(Math.max(step, 2)) / this.counts[a]);
  }

  selectAction(step: number, rng: Rng): number {
    const scores = this.estimates.map((q, a) => {
      const b = this.bonus(step, a);
      return b === Infinity ? Infinity : q + b;
    });
    return argmaxRandomTie(
      scores.map((s) => (s === Infinity ? 1e9 : s)),
      rng,
    );
  }

  update(action: number, reward: number): void {
    this.counts[action] += 1;
    this.estimates[action] += (reward - this.estimates[action]) / this.counts[action];
  }
}

/**
 * Gradient bandit: softmax over learned preferences H(a) with a reward
 * baseline. This is the score-function trick on a one-state MDP — Chapter 10
 * re-derives it with states and calls it the policy gradient theorem.
 */
export class GradientBandit implements BanditPolicy {
  readonly name: string;
  /** Preferences H(a); `estimates` exposes the induced probabilities. */
  private H: number[];
  private baseline = 0;
  private n = 0;

  constructor(
    private k: number,
    private alpha = 0.1,
    private useBaseline = true,
  ) {
    this.name = `gradient (α=${alpha}${useBaseline ? '' : ', no baseline'})`;
    this.H = new Array(k).fill(0);
  }

  get probabilities(): number[] {
    const max = Math.max(...this.H);
    const exp = this.H.map((h) => Math.exp(h - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((e) => e / sum);
  }

  get estimates(): number[] {
    return this.probabilities;
  }

  selectAction(_step: number, rng: Rng): number {
    return sampleCategorical(rng, this.probabilities);
  }

  update(action: number, reward: number): void {
    this.n += 1;
    if (this.useBaseline) this.baseline += (reward - this.baseline) / this.n;
    const probs = this.probabilities;
    const advantage = reward - (this.useBaseline ? this.baseline : 0);
    for (let a = 0; a < this.k; a++) {
      const indicator = a === action ? 1 : 0;
      this.H[a] += this.alpha * advantage * (indicator - probs[a]);
    }
  }
}

/** Thompson sampling with Normal-Normal conjugate updates (known σ² = 1). */
export class ThompsonSampling implements BanditPolicy {
  readonly name = 'Thompson sampling';
  readonly estimates: number[];
  private counts: number[];

  constructor(private k: number) {
    this.estimates = new Array(k).fill(0);
    this.counts = new Array(k).fill(0);
  }

  selectAction(_step: number, rng: Rng): number {
    const samples = this.estimates.map((mu, a) => {
      const n = this.counts[a];
      const posteriorStd = Math.sqrt(1 / (n + 1));
      return gaussian(rng, mu, posteriorStd);
    });
    return argmaxRandomTie(samples, rng);
  }

  update(action: number, reward: number): void {
    this.counts[action] += 1;
    this.estimates[action] += (reward - this.estimates[action]) / this.counts[action];
  }
}

export interface BanditRunResult {
  name: string;
  /** Mean reward per step, averaged over runs. */
  avgReward: number[];
  /** Fraction of runs choosing the optimal arm at each step. */
  optimalPct: number[];
  /** Cumulative regret, averaged over runs. */
  regret: number[];
}

/**
 * Average a policy over many independent bandit problems — the standard
 * testbed protocol. `makePolicy` is called fresh per run.
 */
export function runBanditExperiment(
  makePolicy: (k: number) => BanditPolicy,
  opts: {
    runs?: number;
    steps?: number;
    k?: number;
    seed?: number;
    rngFactory: (seed: number) => Rng;
  },
): BanditRunResult {
  const { runs = 200, steps = 1000, k = 10, seed = 1, rngFactory } = opts;
  const avgReward = new Array(steps).fill(0);
  const optimalPct = new Array(steps).fill(0);
  const regret = new Array(steps).fill(0);
  let name = '';

  for (let run = 0; run < runs; run++) {
    const rng = rngFactory(seed + run * 7919);
    const problem = makeBandit(rng, k);
    const policy = makePolicy(k);
    name = policy.name;
    const best = problem.qStar[problem.optimalAction];
    let cumulativeRegret = 0;

    for (let t = 0; t < steps; t++) {
      const a = policy.selectAction(t + 1, rng);
      const r = pull(problem, a, rng);
      policy.update(a, r);
      avgReward[t] += r;
      if (a === problem.optimalAction) optimalPct[t] += 1;
      cumulativeRegret += best - problem.qStar[a];
      regret[t] += cumulativeRegret;
    }
  }

  return {
    name,
    avgReward: avgReward.map((v) => v / runs),
    optimalPct: optimalPct.map((v) => (100 * v) / runs),
    regret: regret.map((v) => v / runs),
  };
}

export { sampleBeta };
