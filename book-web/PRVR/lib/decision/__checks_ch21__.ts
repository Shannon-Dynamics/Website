/**
 * Numerical self-checks for Chapter 21's MDP module.
 *
 * Same contract as `lib/__checks__.ts`: these are *invariants* — identities the
 * mathematics guarantees — plus the chapter's one hand-computed worked example,
 * which the Rust unit test `hallway_ssp_worked_example` pins to the same digits.
 *
 * Wire them into `runSelfChecks()` (or run them directly) with:
 *
 *     import { runCh21Checks } from './decision/__checks_ch21__';
 *     out.push(...runCh21Checks());
 */

import { Rng } from '../prob/rng';
import {
  criticalSlip,
  cliffRunMdp,
  gridWorldMdp,
  blankGrid,
  hallwaySsp,
  riskyRouteValue,
  safeRouteValue,
  slipFromVelocityModel,
  RISKY,
  SAFE,
  cellIndex,
} from './gridworld';
import {
  greedyPolicy,
  maxResidual,
  policyEvaluation,
  policyIteration,
  prioritizedSweeping,
  simulatePolicy,
  supNorm,
  sweepJacobi,
  valueIteration,
} from './mdp';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

export function runCh21Checks(): CheckResult[] {
  const out: CheckResult[] = [];
  const check = (name: string, fn: () => { pass: boolean; detail?: string }) => {
    try {
      const r = fn();
      out.push({ name, pass: r.pass, detail: r.detail });
    } catch (e) {
      out.push({ name, pass: false, detail: `threw: ${(e as Error).message}` });
    }
  };

  // 1 ---------------------------------------------------------------------
  check('mdp: hallway SSP fixed point is (−3.75, −2.5, −1.25, 0)', () => {
    const { mdp } = hallwaySsp(0.8, 2);
    const { v, policy } = valueIteration(mdp, { eps: 1e-12, maxSweeps: 20000 });
    const want = [-3.75, -2.5, -1.25, 0];
    const err = supNorm(v, want);
    // Every cell costs 1/0.8 = 1.25 expected steps, and buying determinism at
    // 2 per cell is a bad deal, so `roll` must win in all three cells.
    const rolls = policy.slice(0, 3).every((a) => a === 0);
    return { pass: err < 1e-8 && rolls, detail: `‖V − V*‖∞ = ${fmt(err)}, policy = ${policy.join('')}` };
  });

  // 2 ---------------------------------------------------------------------
  check('mdp: the first three Jacobi sweeps match the hand computation', () => {
    const { mdp } = hallwaySsp(0.8, 2);
    let v = [0, 0, 0, 0];
    const want = [
      [-1, -1, -1, 0],
      [-2, -2, -1.2, 0],
      [-3, -2.36, -1.24, 0],
    ];
    let worst = 0;
    for (let k = 0; k < 3; k++) {
      v = sweepJacobi(mdp, v).v;
      worst = Math.max(worst, supNorm(v, want[k]));
    }
    return { pass: worst < 1e-12, detail: `worst deviation ${fmt(worst)}` };
  });

  // 3 ---------------------------------------------------------------------
  check('mdp: the fixed point of T is the fixed point (zero Bellman residual)', () => {
    const spec = blankGrid(9, 7, { slip: 0.15, gamma: 0.95, moves: 4 });
    spec.payoff[cellIndex(spec, 8, 6)] = 10;
    spec.terminal[cellIndex(spec, 8, 6)] = true;
    for (let j = 1; j < 5; j++) spec.blocked[cellIndex(spec, 4, j)] = true;
    const mdp = gridWorldMdp(spec);
    const { v } = valueIteration(mdp, { eps: 1e-12 });
    return { pass: maxResidual(mdp, v) < 1e-9, detail: `‖TV − V‖∞ = ${fmt(maxResidual(mdp, v))}` };
  });

  // 4 ---------------------------------------------------------------------
  check('mdp: value iteration and policy iteration agree', () => {
    const spec = blankGrid(8, 8, { slip: 0.2, gamma: 0.9, moves: 8 });
    spec.payoff[cellIndex(spec, 7, 7)] = 20;
    spec.terminal[cellIndex(spec, 7, 7)] = true;
    spec.payoff[cellIndex(spec, 3, 3)] = -20;
    spec.terminal[cellIndex(spec, 3, 3)] = true;
    const mdp = gridWorldMdp(spec);
    const vi = valueIteration(mdp, { eps: 1e-12 });
    const pi = policyIteration(mdp);
    const err = supNorm(vi.v, pi.v);
    return {
      pass: err < 1e-6,
      detail: `‖V_VI − V_PI‖∞ = ${fmt(err)} after ${pi.iterations} policy improvements`,
    };
  });

  // 5 ---------------------------------------------------------------------
  check('mdp: greedy policy evaluated equals the optimal value function', () => {
    const spec = blankGrid(7, 7, { slip: 0.12, gamma: 0.93, moves: 4 });
    spec.payoff[cellIndex(spec, 6, 3)] = 15;
    spec.terminal[cellIndex(spec, 6, 3)] = true;
    const mdp = gridWorldMdp(spec);
    const { v } = valueIteration(mdp, { eps: 1e-12 });
    const evaluated = policyEvaluation(mdp, greedyPolicy(mdp, v), { tol: 1e-13 });
    const err = supNorm(v, evaluated);
    return { pass: err < 1e-6, detail: `‖V^{π*} − V*‖∞ = ${fmt(err)}` };
  });

  // 6 ---------------------------------------------------------------------
  check('mdp: prioritized sweeping repairs a local reward edit in far fewer backups', () => {
    const spec = blankGrid(20, 20, { slip: 0.1, gamma: 0.95, moves: 4 });
    spec.payoff[cellIndex(spec, 19, 19)] = 50;
    spec.terminal[cellIndex(spec, 19, 19)] = true;
    const cold = valueIteration(gridWorldMdp(spec), { eps: 1e-6 });

    // Drop a hazard into the far corner — nowhere near the optimal routes.
    const edited = { ...spec, payoff: spec.payoff.slice(), terminal: spec.terminal.slice() };
    edited.payoff[cellIndex(edited, 0, 19)] = -40;
    edited.terminal[cellIndex(edited, 0, 19)] = true;
    const mdp = gridWorldMdp(edited);
    const gs = valueIteration(mdp, { eps: 1e-6, v0: cold.v });
    const ps = prioritizedSweeping(mdp, { eps: 1e-6, v0: cold.v });
    const err = supNorm(gs.v, ps.v);
    const sweepBackups = gs.sweeps * mdp.nStates;
    return {
      pass: err < 1e-3 && ps.backups * 4 < sweepBackups,
      detail: `‖ΔV‖∞ = ${fmt(err)}; ${ps.backups} backups vs ${sweepBackups} for ${gs.sweeps} sweeps`,
    };
  });

  // 7 ---------------------------------------------------------------------
  check('mdp: the contraction bound γ‖U − V‖∞ is never violated', () => {
    const spec = blankGrid(10, 10, { slip: 0.25, gamma: 0.85, moves: 8 });
    spec.payoff[cellIndex(spec, 9, 0)] = 12;
    spec.terminal[cellIndex(spec, 9, 0)] = true;
    const mdp = gridWorldMdp(spec);
    const rng = new Rng(2101);
    let worst = 0;
    for (let trial = 0; trial < 40; trial++) {
      const u = Array.from({ length: mdp.nStates }, () => rng.uniform(-20, 5));
      const w = Array.from({ length: mdp.nStates }, () => rng.uniform(-20, 5));
      const tu = sweepJacobi(mdp, u).v;
      const tw = sweepJacobi(mdp, w).v;
      worst = Math.max(worst, supNorm(tu, tw) - mdp.gamma * supNorm(u, w));
    }
    return { pass: worst < 1e-9, detail: `max(‖TU − TV‖∞ − γ‖U − V‖∞) = ${fmt(worst)}` };
  });

  // 8 ---------------------------------------------------------------------
  check('mdp: cliff-run value iteration matches the closed-form route values', () => {
    let worst = 0;
    for (const slip of [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]) {
      const cfg = { riskyLen: 6, safeLen: 16, cliffPenalty: 6, slip };
      const { mdp, start } = cliffRunMdp(cfg);
      const risky = policyEvaluation(mdp, new Array(mdp.nStates).fill(RISKY), { tol: 1e-14 });
      const safe = policyEvaluation(mdp, new Array(mdp.nStates).fill(SAFE), { tol: 1e-14 });
      worst = Math.max(
        worst,
        Math.abs(risky[start] - riskyRouteValue(cfg)),
        Math.abs(safe[start] - safeRouteValue(cfg)),
      );
    }
    return { pass: worst < 1e-6, detail: `worst |V^π − closed form| = ${fmt(worst)}` };
  });

  // 9 ---------------------------------------------------------------------
  check('mdp: the optimal cliff route flips exactly at the critical slip', () => {
    const base = { riskyLen: 6, safeLen: 16, cliffPenalty: 6, slip: 0 };
    const sStar = criticalSlip(base);
    const below = cliffRunMdp({ ...base, slip: sStar - 0.005 });
    const above = cliffRunMdp({ ...base, slip: sStar + 0.005 });
    const piBelow = valueIteration(below.mdp, { eps: 1e-12 }).policy[below.start];
    const piAbove = valueIteration(above.mdp, { eps: 1e-12 }).policy[above.start];
    return {
      pass: sStar > 0.01 && sStar < 0.3 && piBelow === RISKY && piAbove === SAFE,
      detail: `s* = ${fmt(sStar)}; π(start) = ${piBelow === RISKY ? 'risky' : 'safe'} → ${piAbove === RISKY ? 'risky' : 'safe'}`,
    };
  });

  // 10 --------------------------------------------------------------------
  check('mdp: seeded rollouts of π* average to V*(x₀)', () => {
    const { mdp } = hallwaySsp(0.8, 2);
    const { v, policy } = valueIteration(mdp, { eps: 1e-12 });
    const rng = new Rng(21);
    const runs = 20000;
    let total = 0;
    for (let k = 0; k < runs; k++) {
      total += simulatePolicy(mdp, policy, 0, () => rng.next(), 500).discountedReturn;
    }
    const mean = total / runs;
    // σ of a single run is about 2.4 steps here, so 20 000 samples put the
    // standard error near 0.017; 0.1 is a comfortable five-sigma gate.
    return { pass: Math.abs(mean - v[0]) < 0.1, detail: `MC ${fmt(mean)} vs V* ${fmt(v[0])}` };
  });

  // 11 --------------------------------------------------------------------
  check('mdp: the derived slip matches its closed form, at every speed', () => {
    // The chord of a circular arc bisects the turn, so the bearing of a one-cell
    // transit is exactly θ₀ + ½ω̂Δt. With Δt = ℓ/v the speed cancels out and
    //     s = Φ( −(π/8) / sqrt(σ_θ² + α₃ℓ²/4) ).
    // α₁ is set to zero here only because a Gaussian v̂ is occasionally negative,
    // which flips the bearing by π — a real artifact of the Chapter 9 model, and
    // one the chapter discusses rather than hides.
    const half = Math.PI / 8;
    const alpha3 = 0.05;
    const cellSize = 0.3;
    let worst = 0;
    for (const v of [0.2, 0.4, 0.9]) {
      for (const sigmaTheta of [0.175, 0.26, 0.35]) {
        const s = slipFromVelocityModel({
          v,
          alpha1: 0,
          alpha2: 0,
          alpha3,
          alpha4: 0,
          cellSize,
          sigmaTheta,
        });
        const closed = normalCdf(
          -half / Math.sqrt(sigmaTheta * sigmaTheta + (alpha3 * cellSize * cellSize) / 4),
        );
        worst = Math.max(worst, Math.abs(s - closed));
      }
    }
    // 40 000 samples put the Monte Carlo standard error near 0.001.
    return { pass: worst < 0.005, detail: `max |s_MC − closed form| = ${fmt(worst)}` };
  });

  return out;
}

/** Φ(z), via Abramowitz & Stegun 7.1.26 — accurate to ~1.5·10⁻⁷. */
function normalCdf(z: number): number {
  const x = z / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + (x >= 0 ? y : -y));
}
