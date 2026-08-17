/**
 * Model Predictive Path Integral control — Williams et al., *Information
 * Theoretic Model Predictive Control* (T-RO 2018), Algorithms 1 and 2.
 *
 * The whole method is one line of importance sampling. Over control sequences
 * $U = (u_0 \dots u_{H-1})$ there is an *optimal* distribution
 *
 *     q*(V) = (1/η) exp(−S(V)/λ) p(V),
 *
 * the base distribution reweighted by the exponentiated trajectory cost. It is
 * not sampleable, so we sample the proposal we do have — Gaussian perturbations
 * of the current plan — and correct with the likelihood ratio of q* to p. The
 * resulting weights are self-normalized importance weights, and the plan update
 * is the self-normalized estimate of E_{q*}[V].
 *
 * That is the *same* estimator as the particle filter of `lib/filters/pf.ts`,
 * pointed at the future instead of the past: rollouts are particles, the cost is
 * a negative log-likelihood, and the horizon shift plays the role of resampling.
 * The code below is deliberately written so the correspondence is visible —
 * `informationTheoreticWeights` is `Algorithm 2`, and it is the exponential
 * tilt; `effectiveSampleSize` is the identical formula Chapter 8 uses.
 *
 * Dynamics come from `lib/sim/world.ts` (`diffDriveStep`, the noise-free
 * Chapter 9 velocity model), so the rollouts obey exactly the same kinematics
 * the simulator integrates.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import type { Rng } from '../prob/rng';
import { diffDriveStep, distanceToWalls, type Point2, type World } from '../sim/world';
import { distanceAt, type ExactDistanceField } from '../mapping/edt';

/* -------------------------------------------------------------------------- */
/* Controls and limits                                                         */
/* -------------------------------------------------------------------------- */

/** One command for a differential drive: body-frame linear and angular rate. */
export interface Control2 {
  v: number;
  omega: number;
}

export interface ControlLimits {
  vMin: number;
  vMax: number;
  omegaMax: number;
  /** Linear acceleration limit, m/s² — used by DWA and by the plan's rate cap. */
  aMax: number;
  /** Angular acceleration limit, rad/s². */
  alphaMax: number;
}

/** Rusty's actuator envelope. Reverse is allowed, and slower than forward. */
export const RUSTY_LIMITS: ControlLimits = {
  vMin: -0.3,
  vMax: 0.7,
  omegaMax: 1.6,
  aMax: 1.2,
  alphaMax: 3.2,
};

export const control2 = (v: number, omega: number): Control2 => ({ v, omega });

const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

/**
 * Input constraints, Williams et al. §III-D3: rather than reject samples or
 * solve a QP, push the clamp *into the dynamics*. The sampler stays Gaussian
 * and unconstrained; the simulator sees only feasible commands. This costs
 * nothing because the update law never differentiates the dynamics.
 */
export function clampControl(u: Control2, lim: ControlLimits): Control2 {
  return {
    v: clamp(u.v, lim.vMin, lim.vMax),
    omega: clamp(u.omega, -lim.omegaMax, lim.omegaMax),
  };
}

/* -------------------------------------------------------------------------- */
/* Costs                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * The cost functional, split Thrun-style into a running and a terminal part.
 * Note what is *not* required: no gradient, no convexity, not even continuity.
 * A cost may be an `if` statement over an occupancy grid.
 */
export interface CostModel {
  stage(x: Pose2, u: Control2, k: number): number;
  terminal(x: Pose2): number;
}

export interface RolloutResult {
  /** H + 1 poses, starting at x0. */
  states: Pose2[];
  /** S(V) — the state-dependent cost of the whole rollout. */
  cost: number;
}

/**
 * `rollout(x0, U)` — integrate the plan and accumulate its cost.
 *
 * Clamp *then* step, so the trajectory drawn on screen is one the robot could
 * actually execute; a plan scored on commands the actuators would refuse is a
 * plan scored on a lie.
 */
export function rollout(
  x0: Pose2,
  plan: readonly Control2[],
  cost: CostModel,
  dt: number,
  limits: ControlLimits,
): RolloutResult {
  const states: Pose2[] = new Array(plan.length + 1);
  states[0] = x0;
  let s = 0;
  let x = x0;
  for (let k = 0; k < plan.length; k++) {
    const u = clampControl(plan[k], limits);
    x = diffDriveStep(x, u.v, u.omega, dt);
    states[k + 1] = x;
    s += cost.stage(x, u, k);
  }
  s += cost.terminal(x);
  return { states, cost: s };
}

/* -------------------------------------------------------------------------- */
/* Weights — Williams et al., Algorithm 2                                      */
/* -------------------------------------------------------------------------- */

export interface WeightSet {
  weights: number[];
  /** ρ = min_k S_k, subtracted before exponentiating. */
  sMin: number;
  sMean: number;
  /** 1 / Σ w², the Chapter 8 gauge — how many rollouts actually voted. */
  ess: number;
}

/**
 * `ComputeWeights(S_1 … S_K, λ)` — Williams et al., **Algorithm 2**.
 *
 * Subtracting ρ = min_k S_k before the exponential is not a hack: a
 * self-normalized importance weight is invariant to any constant shift of the
 * cost, and this particular shift guarantees the largest exponent is exactly 1
 * so nothing underflows to a set of all-zero weights.
 */
export function informationTheoreticWeights(costs: readonly number[], lambda: number): WeightSet {
  const K = costs.length;
  if (K === 0) return { weights: [], sMin: 0, sMean: 0, ess: 0 };

  let sMin = Infinity;
  let sum = 0;
  for (const s of costs) {
    if (s < sMin) sMin = s;
    sum += s;
  }

  const lam = Math.max(lambda, 1e-9);
  const weights = new Array<number>(K);
  let eta = 0;
  for (let k = 0; k < K; k++) {
    const w = Math.exp(-(costs[k] - sMin) / lam);
    weights[k] = w;
    eta += w;
  }
  if (eta > 0 && Number.isFinite(eta)) {
    for (let k = 0; k < K; k++) weights[k] /= eta;
  } else {
    // Every exponent underflowed (λ far below the cost spread): fall back to
    // best-of-K, which is the λ → 0 limit the weights were approaching anyway.
    weights.fill(0);
    weights[costs.indexOf(sMin)] = 1;
  }

  return { weights, sMin, sMean: sum / K, ess: effectiveSampleSize(weights) };
}

/** ESS = 1 / Σ wᵢ², for weights that already sum to one. Chapter 8, unchanged. */
export function effectiveSampleSize(weights: readonly number[]): number {
  let s2 = 0;
  for (const w of weights) s2 += w * w;
  return s2 > 0 ? 1 / s2 : 0;
}

/* -------------------------------------------------------------------------- */
/* Savitzky–Golay smoothing                                                    */
/* -------------------------------------------------------------------------- */

/** Quadratic-fit convolution coefficients, normalized. Window 5 and 7 only. */
const SG_COEFFS: Record<number, number[]> = {
  5: [-3 / 35, 12 / 35, 17 / 35, 12 / 35, -3 / 35],
  7: [-2 / 21, 3 / 21, 6 / 21, 7 / 21, 6 / 21, 3 / 21, -2 / 21],
};

/**
 * Savitzky–Golay smoothing of a scalar sequence — Williams et al. §III-D4.
 *
 * The chatter in an MPPI plan is Monte-Carlo noise, not signal: two consecutive
 * cycles disagree because they drew different perturbations. A local quadratic
 * fit removes it while reproducing any genuine quadratic ramp *exactly*, which
 * a moving average would flatten.
 */
export function savitzkyGolay(xs: readonly number[], window: 5 | 7 = 5): number[] {
  const c = SG_COEFFS[window];
  const half = (window - 1) / 2;
  const n = xs.length;
  if (n < window) return [...xs];
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let j = -half; j <= half; j++) {
      // Clamp at the ends: the last entries of the plan are the least certain
      // anyway, and reflecting would invent curvature that is not there.
      acc += c[j + half] * xs[clamp(i + j, 0, n - 1)];
    }
    out[i] = acc;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* The controller                                                              */
/* -------------------------------------------------------------------------- */

export interface MppiConfig {
  /** H — steps in the horizon. */
  horizon: number;
  /** K — rollouts per control cycle. */
  samples: number;
  /** Δt — control period, seconds. */
  dt: number;
  /** λ — temperature. Small: winner-takes-all. Large: unweighted average. */
  lambda: number;
  /** Perturbation std-devs; Σ_u = diag(σ_v², σ_ω²). */
  sigmaV: number;
  sigmaOmega: number;
  limits: ControlLimits;
  /**
   * γ = λ(1 − α), the coefficient of the control-cost cross term. The default,
   * γ = λ, is Williams' α = 0: the base distribution is the *uncontrolled*
   * system, so the quadratic control cost is fully in play.
   */
  gamma?: number;
  /** Turn the cross term off to reproduce Exercise 1's biased update. */
  crossTerm?: boolean;
  /** Savitzky–Golay window applied to the updated plan; 0 disables. */
  smoothWindow?: 0 | 5 | 7;
}

export interface MppiSample {
  /** ε — the perturbation sequence drawn for this rollout. */
  eps: Control2[];
  states: Pose2[];
  /** S(V), state cost only. */
  stateCost: number;
  /** S̃ = S(V) + γ Σ_k u_kᵀ Σ_u⁻¹ ε_k, the cost the weight actually sees. */
  tiltedCost: number;
  weight: number;
}

export interface MppiResult {
  /** The command to execute this cycle: the first entry of the updated plan. */
  applied: Control2;
  /** The plan the cycle started from — the prior. */
  previous: Control2[];
  /** The plan after the weighted update — the posterior. */
  updated: Control2[];
  /** Noise-free rollout of the updated plan. */
  updatedStates: Pose2[];
  samples: MppiSample[];
  ess: number;
  sMin: number;
  sMean: number;
  /** Cost of the updated plan — usually below every sampled cost. */
  updatedCost: number;
}

/**
 * `MPPI(x_0, U, K, λ, Σ_u)` — one receding-horizon control cycle.
 *
 * `plan()` leaves the improved sequence in `nominal` but does **not** advance
 * time; call `shift()` after the command has been executed. Splitting them is
 * what lets the widget freeze a cycle mid-collapse.
 */
export class Mppi {
  readonly cfg: Required<MppiConfig>;
  nominal: Control2[];

  constructor(cfg: MppiConfig) {
    this.cfg = {
      gamma: cfg.lambda,
      crossTerm: true,
      smoothWindow: 5,
      ...cfg,
    };
    this.nominal = Array.from({ length: cfg.horizon }, () => control2(0, 0));
  }

  /** Restart the plan from a constant command (usually stop). */
  reset(u: Control2 = control2(0, 0)): void {
    this.nominal = Array.from({ length: this.cfg.horizon }, () => ({ ...u }));
  }

  /**
   * The receding step. Drop the executed command, slide the rest forward, and
   * repeat the last entry — the warm start that makes K = 200 enough where
   * cold-starting every cycle would need thousands.
   *
   * This is the twin of the particle filter's *prediction*: the same belief,
   * moved one step into the future and re-centred there.
   */
  shift(): void {
    const H = this.cfg.horizon;
    for (let k = 0; k < H - 1; k++) this.nominal[k] = this.nominal[k + 1];
    this.nominal[H - 1] = { ...this.nominal[H - 1] };
  }

  plan(x0: Pose2, cost: CostModel, rng: Rng): MppiResult {
    const { horizon: H, samples: K, dt, lambda, sigmaV, sigmaOmega, limits } = this.cfg;
    const gamma = this.cfg.crossTerm ? this.cfg.gamma : 0;
    const invV = 1 / (sigmaV * sigmaV);
    const invW = 1 / (sigmaOmega * sigmaOmega);

    const previous = this.nominal.map((u) => ({ ...u }));
    const samples: MppiSample[] = new Array(K);
    const tilted = new Array<number>(K);

    for (let i = 0; i < K; i++) {
      const eps: Control2[] = new Array(H);
      const perturbed: Control2[] = new Array(H);
      // The cross term Σ_k u_kᵀ Σ_u⁻¹ ε_k: the part of the likelihood ratio
      // q*/p that everyone forgets. It is what charges control effort.
      let cross = 0;
      for (let k = 0; k < H; k++) {
        const e = control2(rng.normal(0, sigmaV), rng.normal(0, sigmaOmega));
        eps[k] = e;
        perturbed[k] = control2(previous[k].v + e.v, previous[k].omega + e.omega);
        cross += previous[k].v * invV * e.v + previous[k].omega * invW * e.omega;
      }
      const { states, cost: stateCost } = rollout(x0, perturbed, cost, dt, limits);
      const tiltedCost = stateCost + gamma * cross;
      tilted[i] = tiltedCost;
      samples[i] = { eps, states, stateCost, tiltedCost, weight: 0 };
    }

    const { weights, ess, sMin, sMean } = informationTheoreticWeights(tilted, lambda);
    for (let i = 0; i < K; i++) samples[i].weight = weights[i];

    // u_k ← u_k + Σ_i w_i ε_k^(i): the self-normalized importance-sampling
    // estimate of E_{q*}[v_k]. The result is a *blend*, not a winner — it is
    // typically smoother than any single rollout, and need not be one of them.
    const dv = new Array<number>(H).fill(0);
    const dw = new Array<number>(H).fill(0);
    for (let i = 0; i < K; i++) {
      const w = weights[i];
      if (w < 1e-12) continue;
      const eps = samples[i].eps;
      for (let k = 0; k < H; k++) {
        dv[k] += w * eps[k].v;
        dw[k] += w * eps[k].omega;
      }
    }

    let vs = previous.map((u, k) => u.v + dv[k]);
    let ws = previous.map((u, k) => u.omega + dw[k]);
    if (this.cfg.smoothWindow) {
      vs = savitzkyGolay(vs, this.cfg.smoothWindow);
      ws = savitzkyGolay(ws, this.cfg.smoothWindow);
    }

    const updated = vs.map((v, k) => clampControl(control2(v, ws[k]), limits));
    this.nominal = updated.map((u) => ({ ...u }));

    const nom = rollout(x0, updated, cost, dt, limits);
    return {
      applied: { ...updated[0] },
      previous,
      updated,
      updatedStates: nom.states,
      updatedCost: nom.cost,
      samples,
      ess,
      sMin,
      sMean,
    };
  }
}

/* -------------------------------------------------------------------------- */
/* A concrete cost: track a path, keep clear of everything                     */
/* -------------------------------------------------------------------------- */

/** A movable piece of clutter. Chairs are circles; walls are the world's job. */
export interface Obstacle {
  x: number;
  y: number;
  r: number;
  /** Non-zero for the one obstacle that is being pushed around while Rusty drives. */
  vx?: number;
  vy?: number;
}

export interface TrackAndClearParams {
  /** The reference path, e.g. Chapter 20's RRT* output, as a polyline. */
  path: readonly Point2[];
  /** Static obstacle distance field — Chapter 19's ESDF, built once. */
  field: ExactDistanceField;
  obstacles: readonly Obstacle[];
  robotRadius: number;
  /** Cross-track weight. */
  wTrack: number;
  /** Progress reward per metre of arclength gained. */
  wProgress: number;
  /** Obstacle barrier height and length scale. */
  wObs: number;
  sigmaObs: number;
  /** Flat penalty for a rollout state inside an obstacle. */
  collisionCost: number;
  /** Preferred cruise speed and its weight. */
  vRef: number;
  wSpeed: number;
  wOmega: number;
  /** Terminal weight on remaining path length. */
  wTerminal: number;
}

export const DEFAULT_TRACK_PARAMS: Omit<TrackAndClearParams, 'path' | 'field' | 'obstacles'> = {
  robotRadius: 0.19,
  wTrack: 6,
  wProgress: 10,
  wObs: 40,
  sigmaObs: 0.16,
  collisionCost: 900,
  vRef: 0.5,
  wSpeed: 4,
  wOmega: 0.4,
  wTerminal: 8,
};

/**
 * Path tracking with an ESDF barrier — the cost the lab actually runs.
 *
 * Three deliberate choices, each one a claim the chapter defends:
 *
 *  1. The obstacle term is a *smooth* exponential barrier in the signed
 *     distance, not an indicator. Indicators make almost every rollout equally
 *     terrible, the weights collapse onto whichever sample got lucky, and the
 *     effective sample size falls through the floor.
 *  2. Progress is rewarded by arclength along the reference path rather than by
 *     Euclidean distance to the goal, so a detour around a chair is not scored
 *     as failure.
 *  3. A hard collision still costs a flat 900 on top. Sampling-based control
 *     does not need that term to be differentiable, which is the entire point.
 */
export class TrackAndClear implements CostModel {
  private cum: number[];
  constructor(readonly p: TrackAndClearParams) {
    // Arclength coordinate of each waypoint, precomputed once.
    this.cum = new Array(p.path.length).fill(0);
    for (let i = 1; i < p.path.length; i++) {
      this.cum[i] = this.cum[i - 1] + Math.hypot(p.path[i].x - p.path[i - 1].x, p.path[i].y - p.path[i - 1].y);
    }
  }

  get length(): number {
    return this.cum[this.cum.length - 1] ?? 0;
  }

  /** Signed clearance: distance to the nearest wall or chair, minus the body radius. */
  clearance(x: number, y: number): number {
    let d = distanceAt(this.p.field, x, y);
    for (const o of this.p.obstacles) {
      const dd = Math.hypot(x - o.x, y - o.y) - o.r;
      if (dd < d) d = dd;
    }
    return d - this.p.robotRadius;
  }

  /** Nearest point on the reference polyline: cross-track error and arclength. */
  project(x: number, y: number): { cross: number; s: number } {
    const path = this.p.path;
    let best = Infinity;
    let bestS = 0;
    for (let i = 1; i < path.length; i++) {
      const ax = path[i - 1].x;
      const ay = path[i - 1].y;
      const dx = path[i].x - ax;
      const dy = path[i].y - ay;
      const len2 = dx * dx + dy * dy;
      let t = len2 > 1e-12 ? ((x - ax) * dx + (y - ay) * dy) / len2 : 0;
      t = clamp(t, 0, 1);
      const px = ax + t * dx;
      const py = ay + t * dy;
      const d = Math.hypot(x - px, y - py);
      if (d < best) {
        best = d;
        bestS = this.cum[i - 1] + t * Math.sqrt(len2);
      }
    }
    return { cross: best, s: bestS };
  }

  private obstacleCost(x: number, y: number): number {
    const c = this.clearance(x, y);
    if (c <= 0) return this.p.collisionCost;
    return this.p.wObs * Math.exp(-c / this.p.sigmaObs);
  }

  stage(x: Pose2, u: Control2, _k: number): number {
    const { cross, s } = this.project(x.x, x.y);
    const speed = u.v - this.p.vRef;
    return (
      this.p.wTrack * cross * cross +
      this.p.wSpeed * speed * speed +
      this.p.wOmega * u.omega * u.omega +
      this.obstacleCost(x.x, x.y) -
      this.p.wProgress * s
    );
  }

  terminal(x: Pose2): number {
    const { s } = this.project(x.x, x.y);
    return this.p.wTerminal * (this.length - s);
  }
}

/** Goal-seeking cost with the same barrier — the doorway-trap experiment. */
export class SeekAndClear implements CostModel {
  constructor(
    readonly goal: Point2,
    readonly field: ExactDistanceField,
    readonly obstacles: readonly Obstacle[],
    readonly robotRadius = 0.19,
    readonly wGoal = 9,
    readonly wObs = 34,
    readonly sigmaObs = 0.22,
    readonly collisionCost = 900,
    readonly wOmega = 0.3,
  ) {}

  clearance(x: number, y: number): number {
    let d = distanceAt(this.field, x, y);
    for (const o of this.obstacles) {
      const dd = Math.hypot(x - o.x, y - o.y) - o.r;
      if (dd < d) d = dd;
    }
    return d - this.robotRadius;
  }

  private obstacleCost(x: number, y: number): number {
    const c = this.clearance(x, y);
    if (c <= 0) return this.collisionCost;
    return this.wObs * Math.exp(-c / this.sigmaObs);
  }

  stage(x: Pose2, u: Control2, _k: number): number {
    return (
      this.wGoal * Math.hypot(x.x - this.goal.x, x.y - this.goal.y) +
      this.wOmega * u.omega * u.omega +
      this.obstacleCost(x.x, x.y)
    );
  }

  terminal(x: Pose2): number {
    return 4 * this.wGoal * Math.hypot(x.x - this.goal.x, x.y - this.goal.y);
  }
}

/* -------------------------------------------------------------------------- */
/* Execution helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Apply a command to the true robot, refusing motions that drive through a
 * wall or a chair. The controller believes its plan; the world does not care.
 */
export function executeStep(
  pose: Pose2,
  u: Control2,
  dt: number,
  world: World,
  obstacles: readonly Obstacle[],
  radius: number,
): { pose: Pose2; collided: boolean } {
  const next = diffDriveStep(pose, u.v, u.omega, dt);
  let clear = distanceToWalls(world, next.x, next.y);
  for (const o of obstacles) {
    clear = Math.min(clear, Math.hypot(next.x - o.x, next.y - o.y) - o.r);
  }
  if (clear <= radius * 0.55) {
    return { pose: { ...pose, theta: normalizeAngle(pose.theta) }, collided: true };
  }
  return { pose: next, collided: false };
}

/**
 * Advance the moving clutter, bouncing it inside a band.
 *
 * The controller is never told about this. Every cycle it re-plans against a
 * snapshot in which the obstacle is *stationary*, which is both what most
 * deployed stacks do and the honest reason a receding horizon has to be short:
 * the model is only trustworthy for as long as the world stays still.
 */
export function moveObstacles(
  obstacles: Obstacle[],
  dt: number,
  band: { minX: number; maxX: number; minY: number; maxY: number },
): Obstacle[] {
  return obstacles.map((o) => {
    if (!o.vx && !o.vy) return o;
    let { x, y, vx = 0, vy = 0 } = o;
    x += vx * dt;
    y += vy * dt;
    if (x < band.minX + o.r) {
      x = band.minX + o.r;
      vx = -vx;
    } else if (x > band.maxX - o.r) {
      x = band.maxX - o.r;
      vx = -vx;
    }
    if (y < band.minY + o.r) {
      y = band.minY + o.r;
      vy = -vy;
    } else if (y > band.maxY - o.r) {
      y = band.maxY - o.r;
      vy = -vy;
    }
    return { ...o, x, y, vx, vy };
  });
}
