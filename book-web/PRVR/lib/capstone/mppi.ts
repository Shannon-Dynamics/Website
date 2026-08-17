/**
 * The local controller: model predictive path integral control (MPPI), the
 * sampling-based stochastic MPC of Chapter 23.
 *
 * Every control tick it rolls out K noisy copies of its own plan through the
 * unicycle model, scores each against the ESDF and the global path, and folds
 * them back into one command with an exponential (softmin) weighting. No
 * gradients, no convexity, no linearisation: the cost may be a wall of
 * infinities and the update still works, because it is an importance-weighted
 * average rather than a descent direction.
 *
 * The one assumption it makes, and the one this chapter cares about, is that
 * the state it rolls out from is *known*. It is not — it is the mean of a
 * belief. That substitution is certainty equivalence (Derivation F1, row 3),
 * and the margin passed in from Derivation F2 is the patch that keeps it safe
 * while the belief is uncertain.
 *
 * Rust counterpart: `crates/capstone/src/tasks/control.rs`.
 */

import { diffDriveStep } from '../sim/world';
import type { Pose2 } from '../geom/se2';
import type { Rng } from '../prob/rng';
import { esdfAt, type Esdf } from './esdf';

export interface Cmd {
  v: number;
  omega: number;
}

export interface MppiConfig {
  /** Rollouts per tick. */
  K: number;
  /** Horizon, in control steps. */
  H: number;
  /** Control period, seconds. */
  dt: number;
  /** Temperature λ: small = winner-take-all, large = averaging. */
  lambda: number;
  sigmaV: number;
  sigmaW: number;
  vMax: number;
  wMax: number;
  /** Desired cruise speed, m/s. */
  vRef: number;
  /** Weight on cross-track error to the global path. */
  wPath: number;
  /** Weight on making progress along it. */
  wProgress: number;
  /** Weight on the soft clearance barrier. */
  wClear: number;
  /** Clearance the controller prefers, above the hard margin. */
  prefer: number;
  /** Penalty charged once per step inside the hard margin. */
  collisionPenalty: number;
  wSpeed: number;
}

export const DEFAULT_MPPI: MppiConfig = {
  K: 56,
  H: 16,
  dt: 0.1,
  lambda: 0.55,
  sigmaV: 0.22,
  sigmaW: 0.85,
  vMax: 0.65,
  wMax: 1.9,
  vRef: 0.5,
  wPath: 6.0,
  wProgress: 3.2,
  wClear: 9.0,
  prefer: 0.42,
  collisionPenalty: 260,
  wSpeed: 1.1,
};

export interface Rollout {
  pts: [number, number][];
  cost: number;
  weight: number;
}

export interface MppiResult {
  cmd: Cmd;
  rollouts: Rollout[];
  /** The rolled-out nominal after the update — the orange spine of the fan. */
  nominal: [number, number][];
  /** Cost of the best sample; a spike means every future looks bad. */
  bestCost: number;
  /** Fraction of rollouts that entered the hard margin. */
  infeasible: number;
}

/** Cumulative arc length of a polyline, so "progress" is a real coordinate. */
function arcLengths(path: readonly [number, number][]): number[] {
  const s = [0];
  for (let i = 1; i < path.length; i++) {
    s.push(s[i - 1] + Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]));
  }
  return s;
}

/** Cross-track distance to the polyline and the arc length at the foot point. */
function projectOnPath(
  path: readonly [number, number][],
  s: readonly number[],
  x: number,
  y: number,
): { d: number; s: number } {
  let bestD = Infinity;
  let bestS = 0;
  for (let i = 0; i + 1 < path.length; i++) {
    const ax = path[i][0];
    const ay = path[i][1];
    const bx = path[i + 1][0];
    const by = path[i + 1][1];
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    let t = 0;
    if (len2 > 1e-12) t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2));
    const px = ax + t * dx;
    const py = ay + t * dy;
    const d = Math.hypot(x - px, y - py);
    if (d < bestD) {
      bestD = d;
      bestS = s[i] + t * Math.sqrt(len2);
    }
  }
  return { d: bestD, s: bestS };
}

/**
 * `control_tick` — Williams et al.'s information-theoretic MPC, kept warm
 * across ticks.
 *
 * The nominal sequence is shifted forward by one step each tick rather than
 * rebuilt, which is what makes K = 56 enough: the sampler only has to explore
 * around a plan that was already good one control period ago.
 */
export class Mppi {
  readonly cfg: MppiConfig;
  /** The warm-started nominal control sequence, length H. */
  private u: Cmd[];

  constructor(cfg: Partial<MppiConfig> = {}) {
    this.cfg = { ...DEFAULT_MPPI, ...cfg };
    this.u = Array.from({ length: this.cfg.H }, () => ({ v: 0, omega: 0 }));
  }

  /** Forget the plan — used when the supervisor changes mode. */
  reset(): void {
    this.u = Array.from({ length: this.cfg.H }, () => ({ v: 0, omega: 0 }));
  }

  step(
    pose: Pose2,
    path: readonly [number, number][],
    esdf: Esdf,
    margin: number,
    rng: Rng,
    speedScale = 1,
  ): MppiResult {
    const c = this.cfg;
    const hasPath = path.length >= 2;
    const s = hasPath ? arcLengths(path) : [0];
    const total = hasPath ? s[s.length - 1] : 0;

    const noise: Cmd[][] = [];
    const costs: number[] = [];
    const rollouts: Rollout[] = [];
    let infeasible = 0;

    for (let k = 0; k < c.K; k++) {
      const eps: Cmd[] = [];
      const pts: [number, number][] = [];
      let x = { ...pose };
      let cost = 0;
      let hit = false;

      for (let h = 0; h < c.H; h++) {
        const ev = rng.normal(0, c.sigmaV);
        const ew = rng.normal(0, c.sigmaW);
        eps.push({ v: ev, omega: ew });
        const v = clamp(this.u[h].v + ev, -0.25 * c.vMax, c.vMax * speedScale);
        const w = clamp(this.u[h].omega + ew, -c.wMax, c.wMax);
        x = diffDriveStep(x, v, w, c.dt);
        pts.push([x.x, x.y]);

        // --- running cost -------------------------------------------------
        const d = esdfAt(esdf, x.x, x.y);
        if (d < margin) {
          cost += c.collisionPenalty;
          hit = true;
        }
        const shortfall = Math.max(0, c.prefer + margin - d);
        cost += c.wClear * shortfall * shortfall;

        if (hasPath) {
          const proj = projectOnPath(path, s, x.x, x.y);
          cost += c.wPath * proj.d * proj.d;
          cost += c.wProgress * (total - proj.s);
        }
        cost += c.wSpeed * (v - c.vRef * speedScale) ** 2;
      }

      if (hit) infeasible++;
      noise.push(eps);
      costs.push(cost);
      rollouts.push({ pts, cost, weight: 0 });
    }

    // --- the path-integral weights ---------------------------------------
    let best = Infinity;
    for (const s0 of costs) if (s0 < best) best = s0;
    let z = 0;
    const w = costs.map((s0) => {
      const e = Math.exp(-(s0 - best) / c.lambda);
      z += e;
      return e;
    });
    if (z > 0) for (let k = 0; k < w.length; k++) w[k] /= z;
    for (let k = 0; k < rollouts.length; k++) rollouts[k].weight = w[k];

    // --- update the nominal ----------------------------------------------
    for (let h = 0; h < c.H; h++) {
      let dv = 0;
      let dw = 0;
      for (let k = 0; k < c.K; k++) {
        dv += w[k] * noise[k][h].v;
        dw += w[k] * noise[k][h].omega;
      }
      this.u[h] = {
        v: clamp(this.u[h].v + dv, -0.25 * c.vMax, c.vMax * speedScale),
        omega: clamp(this.u[h].omega + dw, -c.wMax, c.wMax),
      };
    }

    const cmd = { ...this.u[0] };
    const nominal: [number, number][] = [];
    let xn = { ...pose };
    for (let h = 0; h < c.H; h++) {
      xn = diffDriveStep(xn, this.u[h].v, this.u[h].omega, c.dt);
      nominal.push([xn.x, xn.y]);
    }

    // Receding horizon: shift left, repeat the tail.
    this.u = [...this.u.slice(1), { ...this.u[this.u.length - 1] }];

    return { cmd, rollouts, nominal, bestCost: best, infeasible: infeasible / c.K };
  }
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/**
 * The fallback when there is no usable pose: steer by the raw scan.
 *
 * During `Relocalize` the map-frame position is meaningless, so every
 * map-referenced cost in MPPI is meaningless too. What is still valid is the
 * sensor: the robot can creep along the most open direction it can see and
 * stop before it touches anything. It is a worse controller in every way except
 * the one that matters — it does not depend on knowing where it is.
 */
export function reactiveCreep(
  ranges: readonly number[],
  angles: readonly number[],
  vMax = 0.22,
  clearStop = 0.45,
): Cmd {
  let forward = Infinity;
  let bestAngle = 0;
  let bestRange = -Infinity;
  for (let i = 0; i < ranges.length; i++) {
    const a = angles[i];
    if (Math.abs(a) < 0.35 && ranges[i] < forward) forward = ranges[i];
    // Prefer open directions that are not behind us.
    if (Math.abs(a) < 1.4 && ranges[i] > bestRange) {
      bestRange = ranges[i];
      bestAngle = a;
    }
  }
  const v = forward < clearStop ? 0 : vMax * Math.min(1, (forward - clearStop) / 0.6);
  return { v, omega: clamp(1.6 * bestAngle, -1.4, 1.4) };
}
