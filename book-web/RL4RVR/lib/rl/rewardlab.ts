/**
 * The reward-design laboratory behind Chapter 14.
 *
 * A small, fully-observable gridworld whose reward function the reader
 * composes from named terms. The optimal policy for *that* reward is then
 * solved exactly by value iteration, so whatever behaviour appears is not an
 * approximation artifact or a training failure — it is genuinely what the
 * stated objective asks for. That is the whole argument of the chapter:
 * reward hacking is the optimizer working correctly on what you wrote.
 */

export const LAB_MAP = [
  '..........',
  '..........',
  '...HHH....',
  '...H..H...',
  '...H..H..G',
  '...HHHH...',
  'S.........',
];

export const LAB_ROWS = LAB_MAP.length;
export const LAB_COLS = LAB_MAP[0].length;

export interface RewardTerms {
  /** Bonus on arriving at the goal. The task itself. */
  goal: number;
  /** Cost per elapsed step. */
  step: number;
  /** Penalty for entering a hazard cell. */
  hazard: number;
  /** Naive shaping: a standing bonus for being near the goal. */
  proximity: number;
  /** Potential-based shaping: γΦ(s′) − Φ(s) with Φ = −distance. Provably safe. */
  potential: number;
  /** Asymmetric progress bonus: paid for getting closer, never charged for retreating. */
  progress: number;
}

export const NEUTRAL_TERMS: RewardTerms = {
  goal: 25,
  step: -1,
  hazard: -10,
  proximity: 0,
  potential: 0,
  progress: 0,
};

const DELTAS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];

export class RewardLab {
  readonly rows = LAB_ROWS;
  readonly cols = LAB_COLS;
  readonly states: number[] = [];
  readonly start: number;
  readonly goal: number;
  /** Manhattan distance to the goal, ignoring walls — what a designer would write. */
  private dist: number[] = [];

  constructor(
    readonly terms: RewardTerms,
    readonly gamma = 0.95,
  ) {
    let start = 0;
    let goal = 0;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const s = r * this.cols + c;
        this.states.push(s);
        if (LAB_MAP[r][c] === 'S') start = s;
        if (LAB_MAP[r][c] === 'G') goal = s;
      }
    }
    this.start = start;
    this.goal = goal;

    const [gr, gc] = [Math.floor(goal / this.cols), goal % this.cols];
    this.dist = this.states.map((s) => {
      const r = Math.floor(s / this.cols);
      const c = s % this.cols;
      return Math.abs(r - gr) + Math.abs(c - gc);
    });
  }

  isHazard(s: number): boolean {
    return LAB_MAP[Math.floor(s / this.cols)][s % this.cols] === 'H';
  }

  isTerminal(s: number): boolean {
    return s === this.goal;
  }

  /** Deterministic movement; walls of the grid block. */
  next(s: number, a: number): number {
    const r = Math.floor(s / this.cols);
    const c = s % this.cols;
    const [dr, dc] = DELTAS[a];
    const nr = r + dr;
    const nc = c + dc;
    if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) return s;
    return nr * this.cols + nc;
  }

  /**
   * R(s, a, s′), assembled from the reader's terms. Every term is applied
   * exactly as its label promises — including the ones that are a bad idea.
   */
  reward(s: number, sNext: number): number {
    const t = this.terms;
    let r = t.step;

    if (sNext === this.goal) r += t.goal;
    if (this.isHazard(sNext)) r += t.hazard;

    // Standing bonus for proximity: pays every step you are near the goal,
    // which is a reason to stay near it rather than to enter it.
    if (t.proximity !== 0) {
      r += t.proximity * (1 / (1 + this.dist[sNext]));
    }

    // Potential-based shaping. Ng et al. prove this cannot change the optimal
    // policy, whatever weight it carries.
    if (t.potential !== 0) {
      const phi = (x: number) => -t.potential * this.dist[x];
      r += this.gamma * phi(sNext) - phi(s);
    }

    // Asymmetric progress: paid for approaching, never charged for retreating.
    // This is the classic exploit — a cycle of step-away, step-back earns
    // forever, and no term opposes it strongly enough.
    if (t.progress !== 0) {
      const closer = this.dist[s] - this.dist[sNext];
      r += t.progress * Math.max(0, closer);
    }

    return r;
  }

  /** Value iteration to convergence — exact, so behaviour is not a training fluke. */
  solve(maxSweeps = 800, theta = 1e-9): { V: Float64Array; policy: Int8Array; sweeps: number } {
    const V = new Float64Array(this.rows * this.cols);
    let sweeps = 0;
    for (let k = 0; k < maxSweeps; k++) {
      sweeps = k + 1;
      let delta = 0;
      for (const s of this.states) {
        if (this.isTerminal(s)) continue;
        const old = V[s];
        let best = -Infinity;
        for (let a = 0; a < 4; a++) {
          const sn = this.next(s, a);
          const r = this.reward(s, sn);
          best = Math.max(best, r + (sn === this.goal ? 0 : this.gamma * V[sn]));
        }
        V[s] = best;
        delta = Math.max(delta, Math.abs(old - V[s]));
      }
      if (delta < theta) break;
    }

    const policy = new Int8Array(this.rows * this.cols).fill(-1);
    for (const s of this.states) {
      if (this.isTerminal(s)) continue;
      let best = -Infinity;
      let bestA = 0;
      for (let a = 0; a < 4; a++) {
        const sn = this.next(s, a);
        const q = this.reward(s, sn) + (sn === this.goal ? 0 : this.gamma * V[sn]);
        if (q > best + 1e-12) {
          best = q;
          bestA = a;
        }
      }
      policy[s] = bestA;
    }
    return { V, policy, sweeps };
  }

  /** Follow the greedy policy and report what the agent actually does. */
  rollout(policy: Int8Array, maxSteps = 120) {
    const path: number[] = [this.start];
    const seen = new Map<number, number>();
    let s = this.start;
    let taskReward = 0;
    let designedReward = 0;
    let cycleAt = -1;

    for (let i = 0; i < maxSteps; i++) {
      if (this.isTerminal(s)) break;
      if (seen.has(s) && cycleAt < 0) cycleAt = i;
      seen.set(s, i);

      const a = policy[s];
      const sn = this.next(s, a);
      designedReward += this.reward(s, sn);
      // What a stakeholder actually cares about, regardless of what was written.
      taskReward += (sn === this.goal ? 25 : 0) - 1 + (this.isHazard(sn) ? -10 : 0);
      s = sn;
      path.push(s);
    }

    const reached = s === this.goal;
    return {
      path,
      reached,
      steps: path.length - 1,
      designedReward,
      taskReward,
      looping: !reached && cycleAt >= 0,
      hazardHits: path.filter((p) => this.isHazard(p)).length,
    };
  }
}

/** A short diagnosis of what the composed reward produced. */
export function diagnose(
  terms: RewardTerms,
  result: ReturnType<RewardLab['rollout']>,
): { verdict: string; detail: string; status: 'good' | 'warning' | 'critical' } {
  if (result.looping || !result.reached) {
    if (terms.proximity > 0) {
      return {
        verdict: 'Loitering beside the goal',
        detail:
          'The proximity bonus pays every step spent near the goal, and arriving ends the episode — which ends the payments. Hovering strictly outscores finishing, so the optimal policy never finishes. You rewarded a state of affairs you wanted the robot to pass through, not one you wanted it to sustain.',
        status: 'critical',
      };
    }
  }
  if (result.looping) {
    if (terms.progress > 0) {
      return {
        verdict: 'Farming the progress bonus',
        detail:
          'The agent found a cycle that collects the approach bonus repeatedly without ever arriving. You rewarded a rate of change and never charged for undoing it, so the loop pays forever.',
        status: 'critical',
      };
    }
    return {
      verdict: 'Stuck in a loop',
      detail:
        'The optimal policy cycles rather than finishing. Some term pays more for a repeatable state change than the goal pays for arriving.',
      status: 'critical',
    };
  }
  if (!result.reached && terms.proximity > 0) {
    return {
      verdict: 'Hovering near the goal',
      detail:
        'The proximity bonus pays every step spent close to the goal. Entering it ends the episode and stops the payments, so the agent loiters instead — it is maximizing exactly what you asked for.',
      status: 'critical',
    };
  }
  if (!result.reached) {
    return {
      verdict: 'Never arrives',
      detail:
        'The goal bonus does not outweigh the cost of the journey under this reward, so the optimal policy is to avoid finishing.',
      status: 'critical',
    };
  }
  if (result.hazardHits > 0) {
    return {
      verdict: 'Reaches the goal, through the hazard',
      detail:
        'The route crosses cells you penalized. The penalty is real but too small relative to the steps saved — a priced risk, not an ignored one.',
      status: 'warning',
    };
  }
  return {
    verdict: 'Does the task',
    detail:
      'The agent goes to the goal, avoids the hazard, and does not dawdle. This is what you wanted and what you wrote.',
    status: 'good',
  };
}
