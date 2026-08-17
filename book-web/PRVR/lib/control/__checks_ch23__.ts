/**
 * Numerical self-checks for Chapter 23's control module.
 *
 * Same contract as `lib/__checks__.ts`: invariants the mathematics guarantees,
 * not frozen outputs — except for the chapter's worked example, which is pinned
 * digit for digit because the reader is invited to reproduce it with a pencil.
 *
 *     import { runCh23Checks } from './control/__checks_ch23__';
 *     out.push(...runCh23Checks());
 */

import { Rng } from '../prob/rng';
import { diffDriveStep } from '../sim/world';
import { dwaPlan, dynamicWindow, DEFAULT_DWA_CONFIG } from './dwa';
import {
  Mppi,
  RUSTY_LIMITS,
  SeekAndClear,
  control2,
  effectiveSampleSize,
  informationTheoreticWeights,
  rollout,
  savitzkyGolay,
  type CostModel,
} from './mppi';
import { apartmentField } from './scenes';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const close = (a: number, b: number, tol = 1e-9) => Math.abs(a - b) <= tol;
const fmt = (x: number) => (Math.abs(x) < 1e-4 && x !== 0 ? x.toExponential(2) : x.toFixed(6));

export function runCh23Checks(): CheckResult[] {
  const out: CheckResult[] = [];

  /* ---------------------------------------------------------------- weights */

  {
    // The chapter's worked example: S = (4, 2, 6), λ = 2, ε = (+0.4, 0, −0.4).
    const w = informationTheoreticWeights([4, 2, 6], 2);
    const eps = [0.4, 0, -0.4];
    const du = w.weights.reduce((s, wi, i) => s + wi * eps[i], 0);
    const pass =
      close(w.weights[0], 0.2447284, 1e-6) &&
      close(w.weights[1], 0.6652406, 1e-6) &&
      close(w.weights[2], 0.090031, 1e-6) &&
      close(w.ess, 1.9587, 1e-4) &&
      close(du, 0.0618787, 1e-6);
    out.push({
      name: 'ch23: worked example — weights, ESS, plan update',
      pass,
      detail: `w = (${w.weights.map((x) => fmt(x)).join(', ')}), ESS = ${fmt(w.ess)}, Δu = ${fmt(du)}`,
    });
  }

  {
    // Same example with the control-cost cross term, γ = λ = 2, σ_v = 0.4:
    // S̃ = S + γ·u·Σ⁻¹·ε = S + 6.25 ε. The update changes sign.
    const eps = [0.4, 0, -0.4];
    const tilted = [4, 2, 6].map((s, i) => s + 2 * 0.5 * (1 / 0.16) * eps[i]);
    const w = informationTheoreticWeights(tilted, 2);
    const du = w.weights.reduce((s, wi, i) => s + wi * eps[i], 0);
    out.push({
      name: 'ch23: cross term flips the sign of the update',
      pass: close(tilted[0], 6.5, 1e-12) && close(du, -0.0930347, 1e-6) && du < 0,
      detail: `S̃ = (${tilted.join(', ')}), Δu = ${fmt(du)}`,
    });
  }

  {
    // Self-normalized weights are invariant to a constant shift of the cost.
    const a = informationTheoreticWeights([3, 7, 11, 2], 1.5);
    const b = informationTheoreticWeights([3, 7, 11, 2].map((s) => s + 1000), 1.5);
    const maxDiff = Math.max(...a.weights.map((w, i) => Math.abs(w - b.weights[i])));
    out.push({
      name: 'ch23: weights invariant to a constant cost shift',
      pass: maxDiff < 1e-12,
      detail: `max |Δw| = ${fmt(maxDiff)}`,
    });
  }

  {
    const costs = [5, 1, 9, 3, 7];
    const cold = informationTheoreticWeights(costs, 1e-4);
    const hot = informationTheoreticWeights(costs, 1e6);
    const coldOk = close(cold.weights[1], 1, 1e-9) && close(cold.ess, 1, 1e-6);
    const hotOk = hot.weights.every((w) => close(w, 0.2, 1e-4)) && close(hot.ess, 5, 1e-3);
    out.push({
      name: 'ch23: λ → 0 is best-of-K, λ → ∞ is the unweighted mean',
      pass: coldOk && hotOk,
      detail: `ESS(λ→0) = ${fmt(cold.ess)}, ESS(λ→∞) = ${fmt(hot.ess)} of 5`,
    });
  }

  {
    const equal = informationTheoreticWeights([4, 4, 4, 4, 4, 4, 4, 4], 3);
    out.push({
      name: 'ch23: equal costs give ESS = K',
      pass: close(equal.ess, 8, 1e-9) && close(effectiveSampleSize([0.5, 0.5]), 2, 1e-12),
      detail: `ESS = ${fmt(equal.ess)} of 8`,
    });
  }

  /* --------------------------------------------------------------- smoothing */

  {
    // A Savitzky–Golay filter of order 2 reproduces any quadratic exactly.
    const quad = Array.from({ length: 21 }, (_, i) => 0.3 + 0.4 * i - 0.02 * i * i);
    const s5 = savitzkyGolay(quad, 5);
    let worst = 0;
    for (let i = 2; i < quad.length - 2; i++) worst = Math.max(worst, Math.abs(s5[i] - quad[i]));
    out.push({
      name: 'ch23: Savitzky–Golay reproduces a quadratic exactly',
      pass: worst < 1e-12,
      detail: `max interior error = ${fmt(worst)}`,
    });
  }

  /* ---------------------------------------------------------------- rollout */

  {
    const x0 = { x: 1, y: 2, theta: 0.3 };
    const plan = [control2(0.4, 0.2), control2(0.5, -0.1), control2(0.6, 0)];
    const zero: CostModel = { stage: () => 0, terminal: () => 0 };
    const r = rollout(x0, plan, zero, 0.2, RUSTY_LIMITS);
    let x = x0;
    for (const u of plan) x = diffDriveStep(x, u.v, u.omega, 0.2);
    const err = Math.hypot(r.states[3].x - x.x, r.states[3].y - x.y) + Math.abs(r.states[3].theta - x.theta);
    out.push({
      name: 'ch23: rollout is the Chapter 9 velocity model, composed',
      pass: err < 1e-12 && r.states.length === 4,
      detail: `‖x_H − diffDriveStep³(x₀)‖ = ${fmt(err)}`,
    });
  }

  {
    // Clamping happens before integration, so an out-of-envelope plan is
    // simulated as the robot would actually execute it.
    const zero: CostModel = { stage: () => 0, terminal: () => 0 };
    const wild = rollout({ x: 0, y: 0, theta: 0 }, [control2(99, 0)], zero, 1, RUSTY_LIMITS);
    out.push({
      name: 'ch23: input constraints live inside the dynamics',
      pass: close(wild.states[1].x, RUSTY_LIMITS.vMax, 1e-12),
      detail: `x after 1 s at u = 99 m/s is ${fmt(wild.states[1].x)} m (v_max = ${RUSTY_LIMITS.vMax})`,
    });
  }

  /* ------------------------------------------------------------------- MPPI */

  {
    // With smoothing off, the update is exactly the weighted mean of the
    // perturbations: u_new − u_prev = Σ_i w_i ε_i, the estimator itself.
    const field = apartmentField();
    const cost = new SeekAndClear({ x: 8, y: 4.4 }, field, []);
    const mppi = new Mppi({
      horizon: 6,
      samples: 64,
      dt: 0.1,
      lambda: 5,
      sigmaV: 0.2,
      sigmaOmega: 0.5,
      limits: RUSTY_LIMITS,
      gamma: 0,
      smoothWindow: 0,
    });
    mppi.reset(control2(0.3, 0));
    const res = mppi.plan({ x: 5.5, y: 4.4, theta: 0 }, cost, new Rng(4));
    let worst = 0;
    for (let k = 0; k < 6; k++) {
      let dv = 0;
      let dw = 0;
      for (const s of res.samples) {
        dv += s.weight * s.eps[k].v;
        dw += s.weight * s.eps[k].omega;
      }
      worst = Math.max(
        worst,
        Math.abs(res.updated[k].v - res.previous[k].v - dv),
        Math.abs(res.updated[k].omega - res.previous[k].omega - dw),
      );
    }
    const sums = res.samples.reduce((s, x) => s + x.weight, 0);
    out.push({
      name: 'ch23: the plan update is the self-normalized IS estimate',
      pass: worst < 1e-12 && close(sums, 1, 1e-12),
      detail: `max |Δu − Σ w ε| = ${fmt(worst)}, Σw = ${fmt(sums)}`,
    });
  }

  {
    // Determinism: the same seed must reproduce the same storm exactly.
    const field = apartmentField();
    const cost = new SeekAndClear({ x: 8, y: 4.4 }, field, []);
    const run = () => {
      const m = new Mppi({
        horizon: 8,
        samples: 32,
        dt: 0.1,
        lambda: 4,
        sigmaV: 0.2,
        sigmaOmega: 0.5,
        limits: RUSTY_LIMITS,
        gamma: 0,
      });
      m.reset(control2(0.3, 0));
      return m.plan({ x: 5.5, y: 4.4, theta: 0 }, cost, new Rng(23)).applied;
    };
    const a = run();
    const b = run();
    out.push({
      name: 'ch23: seeded MPPI is reproducible',
      pass: close(a.v, b.v, 0) && close(a.omega, b.omega, 0),
      detail: `u₀ = (${fmt(a.v)}, ${fmt(a.omega)}) both times`,
    });
  }

  /* -------------------------------------------------------------------- DWA */

  {
    const cfg = { ...DEFAULT_DWA_CONFIG, limits: RUSTY_LIMITS };
    const win = dynamicWindow(control2(0.5, 0), cfg);
    const reach = RUSTY_LIMITS.aMax * cfg.dt;
    out.push({
      name: 'ch23: the dynamic window is one acceleration period wide',
      pass: close(win.vHi, Math.min(RUSTY_LIMITS.vMax, 0.5 + reach), 1e-12) && close(win.vLo, 0.5 - reach, 1e-12),
      detail: `v ∈ [${fmt(win.vLo)}, ${fmt(win.vHi)}] from v = 0.5 with a_max·Δt = ${fmt(reach)}`,
    });
  }

  {
    // Every candidate DWA returns must be inside its own window and admissible,
    // and driving at a wall must leave nothing fast admissible.
    const field = apartmentField();
    const cost = new SeekAndClear({ x: 11, y: 4.4 }, field, []);
    const cfg = { ...DEFAULT_DWA_CONFIG, limits: RUSTY_LIMITS };
    const res = dwaPlan({ x: 8, y: 4.4, theta: 0 }, { x: 11, y: 4.4 }, control2(0.5, 0), (x, y) => cost.clearance(x, y), cfg);
    const b = res.best;
    const inWindow = !!b && b.v >= res.window.vLo - 1e-9 && b.v <= res.window.vHi + 1e-9;

    // Nose against the corridor's south wall, pointing into it.
    const blocked = dwaPlan(
      { x: 8, y: 4.05, theta: -Math.PI / 2 },
      { x: 8, y: 1 },
      control2(0.4, 0),
      (x, y) => cost.clearance(x, y),
      cfg,
    );
    const fastBlocked = blocked.candidates.filter((c) => c.admissible && c.v > 0.35).length;
    out.push({
      name: 'ch23: DWA returns an admissible in-window command; a wall kills the fast ones',
      pass: inWindow && !!b && b.admissible && fastBlocked === 0,
      detail: `best = (${fmt(b?.v ?? NaN)}, ${fmt(b?.omega ?? NaN)}); facing a wall, ${fastBlocked} candidates above 0.35 m/s remain admissible`,
    });
  }

  {
    // Fox et al.'s clearance term is a distance to *contact*, so two candidates
    // that both clear the world score identically no matter how close one of
    // them passes. That saturation is why no weighting of it buys margin — the
    // claim w23.2 lets the reader test with a slider.
    const field = apartmentField();
    const cost = new SeekAndClear({ x: 11.4, y: 4.4 }, field, [{ x: 6.0, y: 4.03, r: 0.17 }]);
    const cfg = { ...DEFAULT_DWA_CONFIG, limits: RUSTY_LIMITS };
    const contact = dwaPlan({ x: 4.6, y: 4.4, theta: 0 }, { x: 11.4, y: 4.4 }, control2(0.5, 0), (x, y) => cost.clearance(x, y), cfg);
    const margin = dwaPlan({ x: 4.6, y: 4.4, theta: 0 }, { x: 11.4, y: 4.4 }, control2(0.5, 0), (x, y) => cost.clearance(x, y), {
      ...cfg,
      clearanceMode: 'margin',
    });
    const live = contact.candidates.filter((c) => c.admissible && c.margin > 0);
    const spread = (xs: number[]) => Math.max(...xs) - Math.min(...xs);
    const contactSpread = spread(live.map((c) => c.clearanceTerm));
    const marginSpread = spread(
      margin.candidates.filter((c) => c.admissible && c.margin > 0).map((c) => c.clearanceTerm),
    );
    out.push({
      name: 'ch23: distance-to-contact saturates where minimum-margin discriminates',
      pass: contactSpread < 1e-12 && marginSpread > 0.2,
      detail: `clearance-term spread over collision-free candidates: contact ${fmt(contactSpread)}, margin ${fmt(marginSpread)}`,
    });
  }

  return out;
}
