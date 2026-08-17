/**
 * The costmap layer: a Euclidean signed-distance field over the *robot's own*
 * occupancy grid, plus the safety margin that turns a pose covariance into a
 * clearance requirement.
 *
 * Two things separate this from the distance field of Chapter 10. First, it is
 * built from the map Rusty has *made*, not from the walls the simulator knows
 * about — so unknown space really is unknown, and the failure mode "plan
 * straight through a wall you have not seen yet" is available to be watched.
 * Second, it is rebuilt on a clock (5 Hz in the default rate table) rather than
 * once, so the staleness in Derivation F3 is a real quantity with a real cost.
 *
 * The transform itself is Chapter 19's `edt2dSquared`, unchanged. This file
 * only supplies the seeding, the lookup, and the margin.
 *
 * Rust counterpart: `crates/capstone/src/tasks/control.rs` (`EsdfLayer`).
 */

import { edt2dSquared, EDT_INF } from '../mapping/edt';
import type { OccupancyGrid } from '../mapping/occgrid';

export interface Esdf {
  width: number;
  height: number;
  cellSize: number;
  origin: { x: number; y: number };
  /** Row-major (j·width + i) distance to the nearest occupied cell, in metres. */
  data: Float64Array;
}

export interface EsdfOptions {
  /** A cell counts as an obstacle once the map believes it this strongly. */
  occupied?: number;
  /**
   * Extra obstacle points that are *not* in the map — the endpoints of beams
   * the novelty test flagged as dynamic. They steer the controller without ever
   * being written into the map, which is the whole trick behind the walker
   * failure not poisoning the map.
   */
  transient?: readonly [number, number][];
}

/**
 * Distance to the nearest thing the map calls solid.
 *
 * Unknown cells are deliberately **not** seeded. That is the certainty-
 * equivalence assumption of Derivation F1 made concrete: the planner treats
 * "no evidence" as "free", plans through it, and pays for the assumption the
 * moment the LiDAR disagrees. Seeding unknown as occupied would be safer and
 * would also stop exploration dead, since every frontier is by definition next
 * to unknown space.
 */
export function esdfFromGrid(grid: OccupancyGrid, opts: EsdfOptions = {}): Esdf {
  const occupied = opts.occupied ?? 0.62;
  const { width, height, cellSize, origin } = grid;
  const seed = new Float64Array(width * height).fill(EDT_INF);

  const probs = grid.getProbabilityArray();
  for (let k = 0; k < probs.length; k++) {
    if (probs[k] >= occupied) seed[k] = 0;
  }

  for (const [x, y] of opts.transient ?? []) {
    const i = Math.floor((x - origin.x) / cellSize);
    const j = Math.floor((y - origin.y) / cellSize);
    if (i < 0 || j < 0 || i >= width || j >= height) continue;
    seed[j * width + i] = 0;
  }

  const sq = edt2dSquared(seed, width, height);
  const data = new Float64Array(sq.length);
  // One square root per cell, at the very end — see Chapter 19.
  for (let k = 0; k < sq.length; k++) data[k] = Math.sqrt(sq[k]) * cellSize;

  return { width, height, cellSize, origin, data };
}

/** Nearest-cell lookup, clamped at the border. */
export function esdfAt(e: Esdf, x: number, y: number): number {
  const i = Math.min(e.width - 1, Math.max(0, Math.floor((x - e.origin.x) / e.cellSize)));
  const j = Math.min(e.height - 1, Math.max(0, Math.floor((y - e.origin.y) / e.cellSize)));
  return e.data[j * e.width + i];
}

export const esdfIndexAt = (e: Esdf, i: number, j: number): number =>
  i < 0 || j < 0 || i >= e.width || j >= e.height ? 0 : e.data[j * e.width + i];

/**
 * Φ⁻¹(p) — the standard normal quantile, Acklam's rational approximation.
 *
 * Accurate to about 1.15·10⁻⁹ over the whole open interval, which is four
 * orders of magnitude more than a safety margin quoted in centimetres needs.
 * We need it because the chance constraint of Derivation F2 is stated at a
 * collision probability δ and consumed as a number of standard deviations.
 */
export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;

  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

  const pLow = 0.02425;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pLow) return -normalQuantile(1 - p);

  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

/**
 * Derivation F2, as one line of code.
 *
 *   d_esdf(x) ≥ r_robot + k_σ · σ_pose,     k_σ = Φ⁻¹(1 − δ)
 *
 * `sigmaPose` is the largest standard deviation of the *position* block of the
 * pose belief — the worst direction, because the nearest obstacle is free to
 * lie in it. When the filter is confident the margin collapses to the robot
 * radius; when it is lost the margin explodes and the planner refuses to move,
 * which is exactly the behaviour a lost robot should have.
 */
export function safetyMargin(rRobot: number, sigmaPose: number, delta: number): number {
  return rRobot + normalQuantile(1 - delta) * sigmaPose;
}

/**
 * Largest position standard deviation of a 3×3 SE(2) covariance in the
 * translation-first ordering — the square root of the larger eigenvalue of the
 * upper-left 2×2 block, in closed form.
 */
export function positionSigma(cov: number[][]): number {
  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][1];
  const mean = (a + c) / 2;
  const disc = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b));
  return Math.sqrt(Math.max(0, mean + disc));
}
