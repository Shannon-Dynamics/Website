/**
 * The Dynamic Window Approach — Fox, Burgard & Thrun (1997), *IEEE Robotics &
 * Automation Magazine* 4(1), 23–33.
 *
 * The classical baseline for local control, and the one idea worth keeping from
 * it: **search velocities, not paths.** A diff-drive robot cannot execute an
 * arbitrary curve, so DWA restricts the search to the commands it can reach in
 * one control period given its acceleration limits — the *dynamic window* — and
 * scores each one along the constant-curvature arc it produces.
 *
 * The method has exactly one structural flaw, and this file is written to make
 * it visible rather than to hide it: every candidate is a **single arc**, so no
 * candidate can ever express "reverse two metres, then turn". That is not a
 * tuning problem. It is the hypothesis class.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import { diffDriveStep, type Point2 } from '../sim/world';
import { clampControl, control2, type Control2, type ControlLimits } from './mppi';

export interface DwaWeights {
  /** α — alignment of the predicted heading with the goal bearing. */
  heading: number;
  /** β — clearance along the arc. */
  clearance: number;
  /** γ — reward for going fast, the term that stops it from creeping. */
  velocity: number;
}

export const DEFAULT_DWA_WEIGHTS: DwaWeights = { heading: 0.8, clearance: 0.2, velocity: 0.15 };

export interface DwaConfig {
  /** Grid resolution of the velocity search. */
  nV: number;
  nOmega: number;
  /** Control period Δt — also the width of the dynamic window in time. */
  dt: number;
  /** How far ahead each arc is simulated when measuring clearance, seconds. */
  simTime: number;
  /**
   * Integration step used *inside* an arc. Deliberately finer than the control
   * period: `dist(v, ω)` is a distance-to-first-collision, and a coarse walk
   * overestimates it, which would hand the admissibility test a free pass.
   */
  arcDt: number;
  limits: ControlLimits;
  weights: DwaWeights;
  robotRadius: number;
  /** Clearance is capped here, so an open corridor does not dominate the score. */
  maxClearance: number;
  /**
   * What "clearance" means.
   *
   * `'contact'` is Fox et al.'s `dist(v, ω)`: the distance travelled along the
   * arc before the robot first *touches* something. It is the faithful 1997
   * term, and it has a consequence worth seeing — a gap the robot squeezes
   * through scores exactly as well as a gap with a metre to spare, so no
   * weighting of this term will ever buy margin.
   *
   * `'margin'` is the modern repair: the *smallest* signed clearance seen
   * anywhere along the arc. Now the term discriminates, and its weight does
   * something.
   */
  clearanceMode: 'contact' | 'margin';
  /** Cap for `'margin'` mode, in metres. Beyond this, more room is not better. */
  maxMargin: number;
}

export const DEFAULT_DWA_CONFIG: Omit<DwaConfig, 'limits'> = {
  nV: 21,
  nOmega: 31,
  dt: 0.25,
  simTime: 2.0,
  arcDt: 0.05,
  weights: DEFAULT_DWA_WEIGHTS,
  robotRadius: 0.19,
  maxClearance: 2.5,
  clearanceMode: 'contact',
  maxMargin: 0.5,
};

export interface DwaCandidate {
  v: number;
  omega: number;
  /** Grid indices, so a widget can draw the velocity plane as an image. */
  i: number;
  j: number;
  /** Stoppable before the first collision on this arc? */
  admissible: boolean;
  /** Smallest signed clearance seen anywhere along the arc, in metres. */
  margin: number;
  /** Raw terms, before normalization. */
  headingTerm: number;
  clearanceTerm: number;
  velocityTerm: number;
  /** Weighted, normalized objective. −∞ for inadmissible candidates. */
  score: number;
  /** The arc, for drawing. */
  arc: Pose2[];
}

export interface DwaWindow {
  vLo: number;
  vHi: number;
  omegaLo: number;
  omegaHi: number;
}

export interface DwaResult {
  candidates: DwaCandidate[];
  best: DwaCandidate | null;
  window: DwaWindow;
  nV: number;
  nOmega: number;
}

/**
 * The dynamic window V_d: velocities reachable within one control period from
 * the current command, intersected with the actuator envelope V_s.
 *
 * This is the whole reason the method is called *dynamic*. A robot travelling
 * at 0.7 m/s with a 1.2 m/s² limit simply cannot choose 0 m/s in the next
 * 0.25 s, so offering that command to the optimizer would be a lie.
 */
export function dynamicWindow(current: Control2, cfg: DwaConfig): DwaWindow {
  const { limits: L, dt } = cfg;
  return {
    vLo: Math.max(L.vMin, current.v - L.aMax * dt),
    vHi: Math.min(L.vMax, current.v + L.aMax * dt),
    omegaLo: Math.max(-L.omegaMax, current.omega - L.alphaMax * dt),
    omegaHi: Math.min(L.omegaMax, current.omega + L.alphaMax * dt),
  };
}

/**
 * `dwa_plan(x, x_goal, u_prev, clearance)` — Fox et al., §III.
 *
 * `clearance(x, y)` returns the signed distance from the robot's *centre* to
 * the nearest obstacle surface minus its body radius, i.e. the same function
 * MPPI's barrier consumes. Passing it in keeps the two controllers scoring the
 * identical world, which is what makes the duel in w23.2 fair.
 */
export function dwaPlan(
  x: Pose2,
  goal: Point2,
  current: Control2,
  clearance: (x: number, y: number) => number,
  cfg: DwaConfig,
): DwaResult {
  const win = dynamicWindow(current, cfg);
  const steps = Math.max(1, Math.round(cfg.simTime / cfg.arcDt));
  const candidates: DwaCandidate[] = [];

  for (let j = 0; j < cfg.nOmega; j++) {
    const omega =
      cfg.nOmega === 1
        ? (win.omegaLo + win.omegaHi) / 2
        : win.omegaLo + (j * (win.omegaHi - win.omegaLo)) / (cfg.nOmega - 1);
    for (let i = 0; i < cfg.nV; i++) {
      const v = cfg.nV === 1 ? (win.vLo + win.vHi) / 2 : win.vLo + (i * (win.vHi - win.vLo)) / (cfg.nV - 1);
      const u = clampControl(control2(v, omega), cfg.limits);

      // Walk the constant-curvature arc, stopping at the first collision.
      const arc: Pose2[] = [x];
      let pose = x;
      let travelled = 0;
      let dist = cfg.maxClearance;
      let margin = cfg.maxMargin;
      let hit = false;
      for (let k = 0; k < steps; k++) {
        const next = diffDriveStep(pose, u.v, u.omega, cfg.arcDt);
        travelled += Math.abs(u.v) * cfg.arcDt;
        const c = clearance(next.x, next.y);
        if (c < margin) margin = c;
        if (c <= 0) {
          dist = travelled;
          hit = true;
          arc.push(next);
          break;
        }
        pose = next;
        arc.push(next);
      }
      if (!hit) dist = cfg.maxClearance;

      // Admissibility, Fox et al. eq. (14): the robot must be able to stop
      // before it reaches that obstacle, at its braking limit.
      const stoppable = Math.abs(u.v) <= Math.sqrt(Math.max(2 * dist * cfg.limits.aMax, 0)) + 1e-9;
      const admissible = !hit || stoppable;

      // heading: 1 when the predicted pose faces the goal, 0 when it faces away.
      const end = arc[arc.length - 1];
      const bearing = Math.atan2(goal.y - end.y, goal.x - end.x);
      const headingTerm = 1 - Math.abs(normalizeAngle(bearing - end.theta)) / Math.PI;
      const clearanceTerm =
        cfg.clearanceMode === 'margin'
          ? Math.max(margin, 0) / cfg.maxMargin
          : Math.min(dist, cfg.maxClearance) / cfg.maxClearance;
      const velocityTerm = Math.max(0, u.v) / cfg.limits.vMax;

      candidates.push({
        v: u.v,
        omega: u.omega,
        i,
        j,
        admissible,
        margin,
        headingTerm,
        clearanceTerm,
        velocityTerm,
        score: -Infinity,
        arc,
      });
    }
  }

  // σ in Fox et al.: normalize each term across the window before the weighted
  // sum, so that no term's units decide the outcome.
  const live = candidates.filter((c) => c.admissible);
  const w = cfg.weights;
  const denom = w.heading + w.clearance + w.velocity;
  for (const c of live) {
    c.score = (w.heading * c.headingTerm + w.clearance * c.clearanceTerm + w.velocity * c.velocityTerm) / denom;
  }

  let best: DwaCandidate | null = null;
  for (const c of live) {
    if (!best || c.score > best.score) best = c;
  }
  return { candidates, best, window: win, nV: cfg.nV, nOmega: cfg.nOmega };
}
