/**
 * Rusty's warehouse — the book's canonical finite MDP.
 *
 * Constants are fixed in Chapter 4 and reused verbatim by Chapters 5–7:
 *   12 × 9 grid minus shelf cells (~76 free states)
 *   A = {N, E, S, W}
 *   p_slip = 0.2 (lateral slip: 0.8 intended, 0.1 each perpendicular)
 *   R: +25 delivery at the dock, −1 per step, −10 shelf bump
 *   γ = 0.95
 *
 * The environment exposes BOTH faces of the Ch 4 gym:
 *   `transitions(s, a)` — the white-box `Mdp` view that Ch 5 plans against
 *   `step(s, a, rng)`   — the black-box `Env` view that Ch 6 learns against
 */

import type { Rng } from './random';
import { sampleCategorical } from './random';

export const ACTIONS = ['N', 'E', 'S', 'W'] as const;
export type Action = 0 | 1 | 2 | 3;
export const ACTION_LABELS = ACTIONS;
export const ACTION_ARROWS = ['↑', '→', '↓', '←'] as const;

/** (dRow, dCol) per action, with row 0 at the top. */
const DELTAS: ReadonlyArray<readonly [number, number]> = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];

export interface Transition {
  next: number;
  reward: number;
  prob: number;
  done: boolean;
}

export interface GridConfig {
  pSlip: number;
  gamma: number;
  stepReward: number;
  bumpReward: number;
  goalReward: number;
}

export const DEFAULT_CONFIG: GridConfig = {
  pSlip: 0.2,
  gamma: 0.95,
  stepReward: -1,
  bumpReward: -10,
  goalReward: 25,
};

/**
 * The default warehouse: aisles between four shelf blocks, dock at the right
 * of the lower cross-aisle. '#' = shelf (blocked), 'D' = dock (terminal),
 * 'S' = Rusty's start, '.' = free floor.
 */
export const WAREHOUSE_MAP = [
  'S...........',
  '.####..####.',
  '.####..####.',
  '............',
  '.####..####.',
  '.####..####.',
  '............',
  '..........D.',
  '............',
];

export class GridWorld {
  readonly rows: number;
  readonly cols: number;
  readonly map: string[];
  readonly config: GridConfig;
  /** Index of every traversable cell; the agent's state space. */
  readonly states: number[];
  readonly startState: number;
  readonly goalState: number;
  /**
   * Optional per-cell slip probability, overriding the global one. Chapter 4
   * describes "slippery patches near the loading bay"; this is what lets a
   * reader paint them and watch the optimal policy route around them.
   */
  readonly slipAt?: (state: number) => number | undefined;

  constructor(
    map: string[] = WAREHOUSE_MAP,
    config: Partial<GridConfig> = {},
    slipAt?: (state: number) => number | undefined,
  ) {
    this.map = map;
    this.slipAt = slipAt;
    this.rows = map.length;
    this.cols = map[0].length;
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.states = [];
    let start = 0;
    let goal = 0;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const ch = map[r][c];
        if (ch === '#') continue;
        const s = r * this.cols + c;
        this.states.push(s);
        if (ch === 'S') start = s;
        if (ch === 'D') goal = s;
      }
    }
    this.startState = start;
    this.goalState = goal;
  }

  get nStates(): number {
    return this.rows * this.cols;
  }

  get nActions(): number {
    return 4;
  }

  rowCol(s: number): [number, number] {
    return [Math.floor(s / this.cols), s % this.cols];
  }

  isShelf(r: number, c: number): boolean {
    return r < 0 || r >= this.rows || c < 0 || c >= this.cols || this.map[r][c] === '#';
  }

  isTerminal(s: number): boolean {
    return s === this.goalState;
  }

  isTraversable(s: number): boolean {
    const [r, c] = this.rowCol(s);
    return !this.isShelf(r, c);
  }

  /**
   * The white-box view: the full distribution p(s', r | s, a).
   * Lateral slip splits the remaining probability equally between the two
   * directions perpendicular to the intended one.
   */
  transitions(s: number, a: Action): Transition[] {
    if (this.isTerminal(s)) return [{ next: s, reward: 0, prob: 1, done: true }];

    const { stepReward, bumpReward, goalReward } = this.config;
    // A painted cell may be slipperier than the rest of the floor.
    const pSlip = this.slipAt?.(s) ?? this.config.pSlip;
    const outcomes: Array<[Action, number]> = [
      [a, 1 - pSlip],
      [((a + 1) % 4) as Action, pSlip / 2],
      [((a + 3) % 4) as Action, pSlip / 2],
    ];

    // Merge duplicate (next, reward) pairs so probabilities stay normalised.
    const merged = new Map<string, Transition>();
    for (const [dir, prob] of outcomes) {
      const [r, c] = this.rowCol(s);
      const [dr, dc] = DELTAS[dir];
      const nr = r + dr;
      const nc = c + dc;
      const blocked = this.isShelf(nr, nc);
      const next = blocked ? s : nr * this.cols + nc;
      const done = !blocked && next === this.goalState;
      const reward = blocked ? bumpReward : done ? goalReward : stepReward;
      const key = `${next}:${reward}`;
      const existing = merged.get(key);
      if (existing) existing.prob += prob;
      else merged.set(key, { next, reward, prob, done });
    }
    return [...merged.values()];
  }

  /** The black-box view: sample one transition. */
  step(s: number, a: Action, rng: Rng): Transition {
    const ts = this.transitions(s, a);
    return ts[sampleCategorical(rng, ts.map((t) => t.prob))];
  }

  /** Greedy policy w.r.t. a value function, via one-step lookahead. */
  greedyPolicy(V: Float64Array): Int8Array {
    const policy = new Int8Array(this.nStates).fill(-1);
    for (const s of this.states) {
      if (this.isTerminal(s)) continue;
      let best = -Infinity;
      let bestA = 0;
      for (let a = 0 as Action; a < 4; a = (a + 1) as Action) {
        const q = this.actionValue(s, a, V);
        if (q > best) {
          best = q;
          bestA = a;
        }
      }
      policy[s] = bestA;
    }
    return policy;
  }

  /** q(s,a) = Σ p(s',r|s,a)[r + γ V(s')] — the one-step lookahead. */
  actionValue(s: number, a: Action, V: Float64Array): number {
    const { gamma } = this.config;
    let q = 0;
    for (const t of this.transitions(s, a)) {
      q += t.prob * (t.reward + (t.done ? 0 : gamma * V[t.next]));
    }
    return q;
  }

  /** Roll out a policy for the trajectory widgets. */
  rollout(policy: Int8Array, rng: Rng, maxSteps = 200): { path: number[]; totalReward: number } {
    let s = this.startState;
    const path = [s];
    let totalReward = 0;
    for (let i = 0; i < maxSteps; i++) {
      const a = policy[s];
      if (a < 0) break;
      const t = this.step(s, a as Action, rng);
      totalReward += t.reward;
      s = t.next;
      path.push(s);
      if (t.done) break;
    }
    return { path, totalReward };
  }
}
