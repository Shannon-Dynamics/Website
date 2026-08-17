/**
 * Finite Markov decision processes and the dynamic-programming algorithms that
 * solve them — Chapter 21.
 *
 * This is the TypeScript twin of `crates/ch21_mdp/src/{mdp,vi,pi}.rs`. The
 * structure is deliberately identical: a sparse transition row per (state,
 * action), an expected immediate reward per (state, action), and a discount.
 * Everything else — value iteration, Gauss–Seidel sweeps, policy iteration,
 * prioritized sweeping, rollouts — is a function of that one object.
 *
 * Two conventions worth knowing before reading further:
 *
 *  1. `reward[s][a]` is the *expected* immediate payoff r(x, u), with the
 *     next-state payoff already folded in. Thrun's draft writes the backup as
 *     max_a ∫ [c(s') + V(s')] p(s'|a,s) ds'; taking the expectation of c(s')
 *     out front turns that into the textbook r(x,u) + γ Σ p V form without
 *     changing a single number.
 *  2. An absorbing state has V = 0 by definition: every action self-loops with
 *     zero reward. Goal and terminal-hazard cells are absorbing, so the payoff
 *     for reaching them is collected by the *neighbour* that steps in.
 */

export interface Transition {
  /** Successor state index. */
  s: number;
  /** p(x' | x, u) for that successor. Rows sum to 1. */
  p: number;
}

/** A sparse row of the transition model: p(· | x, u) as (state, prob) pairs. */
export type SparseDist = Transition[];

export interface Mdp {
  nStates: number;
  nActions: number;
  /** trans[state][action] — a sparse distribution over successors. */
  trans: SparseDist[][];
  /** reward[state][action] = r(x, u), the expected immediate payoff. */
  reward: number[][];
  /** Discount γ ∈ (0, 1]; γ = 1 is legal only for a stochastic shortest path. */
  gamma: number;
  /** absorbing[s]: the episode ends here. V(s) ≡ 0. */
  absorbing: boolean[];
  actionLabels?: string[];
}

/* -------------------------------------------------------------------------- */
/* The Bellman backup                                                          */
/* -------------------------------------------------------------------------- */

/** Q(x, u) = r(x, u) + γ Σ_{x'} p(x' | x, u) V(x'). */
export function qValue(mdp: Mdp, v: readonly number[], s: number, a: number): number {
  if (mdp.absorbing[s]) return 0;
  let acc = 0;
  const row = mdp.trans[s][a];
  for (let k = 0; k < row.length; k++) acc += row[k].p * v[row[k].s];
  return mdp.reward[s][a] + mdp.gamma * acc;
}

/** Every Q(x, ·) at once — the bars the Bellman Stepper draws. */
export function qValues(mdp: Mdp, v: readonly number[], s: number): number[] {
  const out = new Array<number>(mdp.nActions);
  for (let a = 0; a < mdp.nActions; a++) out[a] = qValue(mdp, v, s, a);
  return out;
}

/** The max gate: the best action at `s` under `v`, and its value. */
export function backup(mdp: Mdp, v: readonly number[], s: number): { value: number; action: number } {
  if (mdp.absorbing[s]) return { value: 0, action: 0 };
  let best = -Infinity;
  let arg = 0;
  for (let a = 0; a < mdp.nActions; a++) {
    const q = qValue(mdp, v, s, a);
    if (q > best) {
      best = q;
      arg = a;
    }
  }
  return { value: best, action: arg };
}

/** |(TV)(x) − V(x)|: how far this state is from satisfying Bellman's equation. */
export function bellmanResidual(mdp: Mdp, v: readonly number[], s: number): number {
  if (mdp.absorbing[s]) return 0;
  return Math.abs(backup(mdp, v, s).value - v[s]);
}

/**
 * One **synchronous** (Jacobi) sweep: V_{k+1} = T V_k, every state backed up
 * from the same old vector. Returns the new vector and ‖V_{k+1} − V_k‖∞.
 */
export function sweepJacobi(mdp: Mdp, v: readonly number[]): { v: number[]; residual: number } {
  const next = new Array<number>(mdp.nStates);
  let residual = 0;
  for (let s = 0; s < mdp.nStates; s++) {
    next[s] = mdp.absorbing[s] ? 0 : backup(mdp, v, s).value;
    residual = Math.max(residual, Math.abs(next[s] - v[s]));
  }
  return { v: next, residual };
}

/**
 * One **asynchronous** (Gauss–Seidel) sweep, in place: each backup already sees
 * the states updated earlier in this sweep, so information travels arbitrarily
 * far in a single pass if the order happens to run backwards from the goal.
 * Mutates `v` and returns ‖ΔV‖∞ — the number that drives the widgets' waves.
 */
export function sweepInPlace(mdp: Mdp, v: number[], order?: readonly number[]): number {
  let residual = 0;
  const n = order ? order.length : mdp.nStates;
  for (let k = 0; k < n; k++) {
    const s = order ? order[k] : k;
    if (mdp.absorbing[s]) {
      v[s] = 0;
      continue;
    }
    const next = backup(mdp, v, s).value;
    residual = Math.max(residual, Math.abs(next - v[s]));
    v[s] = next;
  }
  return residual;
}

/* -------------------------------------------------------------------------- */
/* Value iteration                                                             */
/* -------------------------------------------------------------------------- */

export interface ViOptions {
  /** Target accuracy ‖V_k − V*‖∞ ≤ eps. */
  eps?: number;
  maxSweeps?: number;
  /** Gauss–Seidel (default) or the synchronous textbook sweep. */
  inPlace?: boolean;
  /** Warm start. Copied, never aliased. */
  v0?: readonly number[];
}

export interface ViResult {
  v: number[];
  policy: number[];
  sweeps: number;
  /** ‖V_k − V_{k−1}‖∞ after each sweep. */
  residuals: number[];
  converged: boolean;
}

/**
 * The stopping rule of §"Value iteration converges": once one sweep moves the
 * value function by less than ε(1−γ)/γ, the *fixed point* is within ε.
 *
 * For an undiscounted stochastic shortest path (γ = 1) no such bound exists —
 * the contraction modulus is 1 in the sup norm — so the caller falls back to
 * the raw residual and the text says so out loud.
 */
export function stoppingThreshold(gamma: number, eps: number): number {
  return gamma >= 1 ? eps : (eps * (1 - gamma)) / gamma;
}

/** Thrun et al., Table 15.1 (`MDP_value_iteration`), with a stopping rule. */
export function valueIteration(mdp: Mdp, opts: ViOptions = {}): ViResult {
  const { eps = 1e-8, maxSweeps = 5000, inPlace = true } = opts;
  let v = opts.v0 ? Array.from(opts.v0) : new Array<number>(mdp.nStates).fill(0);
  const threshold = stoppingThreshold(mdp.gamma, eps);
  const residuals: number[] = [];
  let converged = false;

  for (let k = 0; k < maxSweeps; k++) {
    let residual: number;
    if (inPlace) {
      residual = sweepInPlace(mdp, v);
    } else {
      const step = sweepJacobi(mdp, v);
      v = step.v;
      residual = step.residual;
    }
    residuals.push(residual);
    if (residual < threshold) {
      converged = true;
      break;
    }
  }

  return { v, policy: greedyPolicy(mdp, v), sweeps: residuals.length, residuals, converged };
}

/** π(x) = argmax_u [ r(x,u) + γ Σ p(x'|x,u) V(x') ] — optimal once V = V*. */
export function greedyPolicy(mdp: Mdp, v: readonly number[]): number[] {
  const pi = new Array<number>(mdp.nStates).fill(0);
  for (let s = 0; s < mdp.nStates; s++) pi[s] = backup(mdp, v, s).action;
  return pi;
}

/** max_x |(TV)(x) − V(x)| — zero exactly at the fixed point. */
export function maxResidual(mdp: Mdp, v: readonly number[]): number {
  let worst = 0;
  for (let s = 0; s < mdp.nStates; s++) {
    if (mdp.absorbing[s]) continue;
    worst = Math.max(worst, Math.abs(backup(mdp, v, s).value - v[s]));
  }
  return worst;
}

/* -------------------------------------------------------------------------- */
/* Policy iteration                                                            */
/* -------------------------------------------------------------------------- */

/**
 * V^π: the value of *following* π, not of acting optimally.
 *
 * Exactly this is the linear system (I − γ P_π) V^π = r_π. The Rust side hands
 * it to `faer`'s sparse solver; here we iterate the linear backup, which is the
 * same fixed point reached the slow, allocation-free way.
 */
export function policyEvaluation(
  mdp: Mdp,
  policy: readonly number[],
  opts: { tol?: number; maxIter?: number; v0?: readonly number[] } = {},
): number[] {
  const { tol = 1e-12, maxIter = 20000 } = opts;
  const v = opts.v0 ? Array.from(opts.v0) : new Array<number>(mdp.nStates).fill(0);
  for (let k = 0; k < maxIter; k++) {
    let residual = 0;
    for (let s = 0; s < mdp.nStates; s++) {
      if (mdp.absorbing[s]) {
        v[s] = 0;
        continue;
      }
      const next = qValue(mdp, v, s, policy[s]);
      residual = Math.max(residual, Math.abs(next - v[s]));
      v[s] = next;
    }
    if (residual < tol) break;
  }
  return v;
}

export interface PiResult extends ViResult {
  /** Outer iterations: evaluate, then improve. */
  iterations: number;
  /** How many states changed action at each improvement step. */
  changed: number[];
}

/** Howard's policy iteration: exact evaluation, greedy improvement, repeat. */
export function policyIteration(
  mdp: Mdp,
  opts: { maxIter?: number; tol?: number; policy0?: readonly number[] } = {},
): PiResult {
  const { maxIter = 200, tol = 1e-12 } = opts;
  let policy = opts.policy0 ? Array.from(opts.policy0) : new Array<number>(mdp.nStates).fill(0);
  let v = new Array<number>(mdp.nStates).fill(0);
  const changed: number[] = [];

  for (let it = 0; it < maxIter; it++) {
    v = policyEvaluation(mdp, policy, { tol, v0: v });
    const next = greedyPolicy(mdp, v);
    let nChanged = 0;
    for (let s = 0; s < mdp.nStates; s++) {
      // Only count a switch when it strictly helps: ties would otherwise make
      // the loop oscillate forever between two equally good actions.
      if (next[s] !== policy[s] && qValue(mdp, v, s, next[s]) > qValue(mdp, v, s, policy[s]) + 1e-12) {
        nChanged++;
      } else {
        next[s] = policy[s];
      }
    }
    changed.push(nChanged);
    policy = next;
    if (nChanged === 0) {
      return {
        v,
        policy,
        sweeps: it + 1,
        residuals: [],
        converged: true,
        iterations: it + 1,
        changed,
      };
    }
  }
  return {
    v,
    policy,
    sweeps: maxIter,
    residuals: [],
    converged: false,
    iterations: maxIter,
    changed,
  };
}

/* -------------------------------------------------------------------------- */
/* Prioritized sweeping                                                        */
/* -------------------------------------------------------------------------- */

/** preds[s'] = every state that can reach s' under some action. */
export function predecessors(mdp: Mdp): number[][] {
  const preds: Set<number>[] = Array.from({ length: mdp.nStates }, () => new Set<number>());
  for (let s = 0; s < mdp.nStates; s++) {
    if (mdp.absorbing[s]) continue;
    for (let a = 0; a < mdp.nActions; a++) {
      for (const t of mdp.trans[s][a]) if (t.s !== s) preds[t.s].add(s);
    }
  }
  return preds.map((set) => Array.from(set));
}

/** A max-heap keyed on |Bellman residual|; ties broken by insertion order. */
class MaxHeap {
  private keys: number[] = [];
  private items: number[] = [];

  get size(): number {
    return this.items.length;
  }

  push(item: number, key: number): void {
    this.items.push(item);
    this.keys.push(key);
    let i = this.items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.keys[parent] >= this.keys[i]) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  pop(): { item: number; key: number } | undefined {
    if (this.items.length === 0) return undefined;
    const item = this.items[0];
    const key = this.keys[0];
    const lastItem = this.items.pop() as number;
    const lastKey = this.keys.pop() as number;
    if (this.items.length > 0) {
      this.items[0] = lastItem;
      this.keys[0] = lastKey;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let big = i;
        if (l < this.items.length && this.keys[l] > this.keys[big]) big = l;
        if (r < this.items.length && this.keys[r] > this.keys[big]) big = r;
        if (big === i) break;
        this.swap(i, big);
        i = big;
      }
    }
    return { item, key };
  }

  private swap(i: number, j: number): void {
    [this.items[i], this.items[j]] = [this.items[j], this.items[i]];
    [this.keys[i], this.keys[j]] = [this.keys[j], this.keys[i]];
  }
}

export interface SweepingResult extends ViResult {
  /** Individual state backups performed — the fair unit against a full sweep. */
  backups: number;
}

/** maxInflow[s'] = max over (predecessor, action) of p(s' | s, a). */
function maxInflow(mdp: Mdp): { preds: number[][]; weight: Map<number, number>[] } {
  const preds = predecessors(mdp);
  const weight: Map<number, number>[] = Array.from({ length: mdp.nStates }, () => new Map());
  for (let s = 0; s < mdp.nStates; s++) {
    if (mdp.absorbing[s]) continue;
    for (let a = 0; a < mdp.nActions; a++) {
      for (const t of mdp.trans[s][a]) {
        if (t.s === s) continue;
        const m = weight[t.s];
        const cur = m.get(s) ?? 0;
        if (t.p > cur) m.set(s, t.p);
      }
    }
  }
  return { preds, weight };
}

/**
 * Prioritized sweeping (Moore & Atkeson 1993): back up the state with the
 * largest pending change, then enqueue only its predecessors, keyed by how much
 * of that change can reach them — γ · p(s' | s, u) · |ΔV(s')|.
 *
 * The order in which states are updated is irrelevant to *whether* value
 * iteration converges — Thrun's draft says so explicitly — but it decides how
 * much work convergence costs. A full sweep spends most of its budget on states
 * whose value did not move; this spends none.
 *
 * Its real payoff is repair. Warm-start it after a local change to the reward
 * (the reader painting one hazard cell) and it touches only the states that
 * change, while a sweep-based solver re-visits the entire map.
 */
export function prioritizedSweeping(
  mdp: Mdp,
  opts: { eps?: number; maxBackups?: number; v0?: readonly number[] } = {},
): SweepingResult {
  const { eps = 1e-8 } = opts;
  const maxBackups = opts.maxBackups ?? 200 * mdp.nStates;
  const threshold = stoppingThreshold(mdp.gamma, eps);
  const v = opts.v0 ? Array.from(opts.v0) : new Array<number>(mdp.nStates).fill(0);
  const { preds, weight } = maxInflow(mdp);
  const heap = new MaxHeap();
  const queued = new Float64Array(mdp.nStates);

  for (let s = 0; s < mdp.nStates; s++) {
    // A warm start may carry a stale value for a state that has just *become*
    // absorbing — the reader painting a goal onto an ordinary cell.
    if (mdp.absorbing[s]) {
      v[s] = 0;
      continue;
    }
  }
  for (let s = 0; s < mdp.nStates; s++) {
    if (mdp.absorbing[s]) continue;
    const r = Math.abs(backup(mdp, v, s).value - v[s]);
    if (r > threshold) {
      heap.push(s, r);
      queued[s] = r;
    }
  }

  let backups = 0;
  while (heap.size > 0 && backups < maxBackups) {
    const top = heap.pop();
    if (!top) break;
    const s = top.item;
    // Stale entry: this state was re-queued with a bigger key after this push.
    if (top.key < queued[s] - 1e-15) continue;
    queued[s] = 0;
    if (top.key < threshold) break; // the heap is sorted: nothing left matters

    const next = backup(mdp, v, s).value;
    const delta = Math.abs(next - v[s]);
    v[s] = next;
    backups++;
    if (delta <= threshold) continue;

    for (const p of preds[s]) {
      if (mdp.absorbing[p]) continue;
      // The most this backup can move p, without paying for a trial backup.
      const key = mdp.gamma * (weight[s].get(p) ?? 0) * delta;
      if (key > threshold && key > queued[p]) {
        queued[p] = key;
        heap.push(p, key);
      }
    }
  }

  return {
    v,
    policy: greedyPolicy(mdp, v),
    sweeps: Math.ceil(backups / Math.max(mdp.nStates, 1)),
    residuals: [],
    converged: backups < maxBackups,
    backups,
  };
}

/* -------------------------------------------------------------------------- */
/* Rollouts — Monte Carlo meeting dynamic programming                          */
/* -------------------------------------------------------------------------- */

export interface Rollout {
  /** Visited states, starting at s0. */
  path: number[];
  /** Actions taken, one per transition. */
  actions: number[];
  /** Σ γ^t r(x_t, u_t) — the realized return of this single run. */
  discountedReturn: number;
  steps: number;
  /** Did the run end in an absorbing state (rather than hitting the cap)? */
  absorbed: boolean;
}

/** One seeded run of a policy. `sample` must return a uniform in [0, 1). */
export function simulatePolicy(
  mdp: Mdp,
  policy: readonly number[],
  s0: number,
  sample: () => number,
  maxSteps = 500,
): Rollout {
  const path = [s0];
  const actions: number[] = [];
  let s = s0;
  let ret = 0;
  let discount = 1;
  let absorbed = mdp.absorbing[s0];

  for (let t = 0; t < maxSteps && !absorbed; t++) {
    const a = policy[s];
    actions.push(a);
    ret += discount * mdp.reward[s][a];
    discount *= mdp.gamma;
    s = sampleTransition(mdp.trans[s][a], sample());
    path.push(s);
    absorbed = mdp.absorbing[s];
  }

  return { path, actions, discountedReturn: ret, steps: actions.length, absorbed };
}

/** Inverse-CDF sample from a sparse row. */
export function sampleTransition(row: SparseDist, u: number): number {
  let acc = 0;
  for (let k = 0; k < row.length; k++) {
    acc += row[k].p;
    if (u < acc) return row[k].s;
  }
  return row[row.length - 1].s;
}

/* -------------------------------------------------------------------------- */
/* Construction helpers                                                        */
/* -------------------------------------------------------------------------- */

export interface MdpBuilderOptions {
  nStates: number;
  nActions: number;
  gamma: number;
  actionLabels?: string[];
}

/** An MDP with every action a zero-reward self-loop; fill in the rows you need. */
export function emptyMdp(opts: MdpBuilderOptions): Mdp {
  const { nStates, nActions, gamma, actionLabels } = opts;
  const trans: SparseDist[][] = [];
  const reward: number[][] = [];
  for (let s = 0; s < nStates; s++) {
    trans.push(Array.from({ length: nActions }, () => [{ s, p: 1 }]));
    reward.push(new Array<number>(nActions).fill(0));
  }
  return {
    nStates,
    nActions,
    trans,
    reward,
    gamma,
    absorbing: new Array<boolean>(nStates).fill(false),
    actionLabels,
  };
}

/** Merge duplicate successors and renormalize — sparse rows must sum to 1. */
export function condense(pairs: readonly Transition[]): SparseDist {
  const acc = new Map<number, number>();
  for (const { s, p } of pairs) {
    if (p <= 0) continue;
    acc.set(s, (acc.get(s) ?? 0) + p);
  }
  let total = 0;
  for (const p of acc.values()) total += p;
  const out: SparseDist = [];
  for (const [s, p] of acc) out.push({ s, p: p / total });
  out.sort((a, b) => a.s - b.s);
  return out;
}

/** Largest deviation between two value vectors — the sup norm the proofs use. */
export function supNorm(a: readonly number[], b: readonly number[]): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}
