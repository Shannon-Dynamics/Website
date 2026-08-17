/**
 * Dynamic programming (Chapter 5) — planning with a known model.
 *
 * Each algorithm is exposed as a *generator* yielding one snapshot per sweep,
 * so the GPI dashboard can animate the value function rippling outward from
 * the dock exactly as the math describes, rather than showing only the answer.
 */

import type { Action } from './gridworld';
import { GridWorld } from './gridworld';

export interface DpSnapshot {
  /** Sweep index, 1-based. */
  sweep: number;
  V: Float64Array;
  policy: Int8Array;
  /** max_s |V_{k+1}(s) − V_k(s)| — the convergence measure. */
  delta: number;
  /** Bellman-error bound on ‖v_π − v_*‖_∞ implied by this Δ. */
  suboptimalityBound: number;
  phase: 'evaluation' | 'improvement';
  /** Set when a policy-improvement step changed at least one action. */
  policyChanged?: boolean;
}

function bound(delta: number, gamma: number): number {
  return (2 * delta * gamma) / (1 - gamma);
}

/**
 * Iterative policy evaluation: repeatedly apply the Bellman expectation
 * operator T_π until the sup-norm change drops below `theta`.
 */
export function* policyEvaluation(
  env: GridWorld,
  policy: Int8Array,
  theta = 1e-6,
  maxSweeps = 500,
  V0?: Float64Array,
): Generator<DpSnapshot> {
  const V = V0 ? Float64Array.from(V0) : new Float64Array(env.nStates);
  const { gamma } = env.config;

  for (let sweep = 1; sweep <= maxSweeps; sweep++) {
    let delta = 0;
    // In-place ("Gauss–Seidel") sweep: uses fresh values as soon as they exist,
    // which converges faster than the two-array Jacobi form.
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      const a = policy[s];
      if (a < 0) continue;
      const v = V[s];
      V[s] = env.actionValue(s, a as Action, V);
      delta = Math.max(delta, Math.abs(v - V[s]));
    }
    yield {
      sweep,
      V: Float64Array.from(V),
      policy: Int8Array.from(policy),
      delta,
      suboptimalityBound: bound(delta, gamma),
      phase: 'evaluation',
    };
    if (delta < theta) return;
  }
}

/**
 * Value iteration: the Bellman OPTIMALITY operator T_*, one sweep at a time.
 * Equivalent to policy evaluation truncated to a single sweep per improvement.
 */
export function* valueIteration(
  env: GridWorld,
  theta = 1e-6,
  maxSweeps = 500,
): Generator<DpSnapshot> {
  const V = new Float64Array(env.nStates);
  const { gamma } = env.config;

  for (let sweep = 1; sweep <= maxSweeps; sweep++) {
    let delta = 0;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      const v = V[s];
      let best = -Infinity;
      for (let a = 0 as Action; a < 4; a = (a + 1) as Action) {
        best = Math.max(best, env.actionValue(s, a, V));
      }
      V[s] = best;
      delta = Math.max(delta, Math.abs(v - V[s]));
    }
    yield {
      sweep,
      V: Float64Array.from(V),
      policy: env.greedyPolicy(V),
      delta,
      suboptimalityBound: bound(delta, gamma),
      phase: 'evaluation',
    };
    if (delta < theta) return;
  }
}

/**
 * Policy iteration: alternate full evaluation with greedy improvement.
 * Terminates when an improvement sweep changes nothing — the policy is then
 * optimal by the policy improvement theorem.
 */
export function* policyIteration(
  env: GridWorld,
  theta = 1e-6,
  maxIterations = 60,
): Generator<DpSnapshot> {
  let policy: Int8Array = new Int8Array(env.nStates).fill(-1);
  for (const s of env.states) if (!env.isTerminal(s)) policy[s] = 1; // start: "go east"

  let V: Float64Array = new Float64Array(env.nStates);
  let sweep = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    // --- Evaluation to convergence -----------------------------------------
    for (const snap of policyEvaluation(env, policy, theta, 500, V)) {
      sweep += 1;
      V = snap.V;
      yield { ...snap, sweep };
    }

    // --- Greedy improvement -------------------------------------------------
    const improved = env.greedyPolicy(V);
    let changed = false;
    for (const s of env.states) {
      if (!env.isTerminal(s) && improved[s] !== policy[s]) changed = true;
    }
    policy = improved;
    sweep += 1;
    yield {
      sweep,
      V: Float64Array.from(V),
      policy: Int8Array.from(policy),
      delta: 0,
      suboptimalityBound: 0,
      phase: 'improvement',
      policyChanged: changed,
    };
    if (!changed) return;
  }
}

/** Run a DP generator to completion and return the final snapshot. */
export function runToConvergence(gen: Generator<DpSnapshot>): DpSnapshot {
  let last: DpSnapshot | undefined;
  for (const snap of gen) last = snap;
  if (!last) throw new Error('generator produced no snapshots');
  return last;
}

/** Collect every snapshot (for scrubbing back and forth in a widget). */
export function collect(gen: Generator<DpSnapshot>, cap = 400): DpSnapshot[] {
  const out: DpSnapshot[] = [];
  for (const snap of gen) {
    out.push(snap);
    if (out.length >= cap) break;
  }
  return out;
}
