/**
 * Expected information gain over an occupancy grid — Chapter 24.
 *
 * The map is the same log-odds grid Chapter 13 built, and the objective is the
 * same entropy that grid already reports. What is new is asking the question
 * *forwards*: before taking a scan, how many bits do we expect it to remove?
 *
 * The answer factors twice. Over cells, because the grid already assumes the
 * cells are independent (Chapter 13's standing approximation, inherited here).
 * And along a beam, because a beam only reaches cell i if every cell before it
 * was empty — a running product we call the **reach probability**.
 *
 * Rust counterpart: `crates/ch24_explore/src/info_gain.rs`.
 */

import type { Pose2 } from '../geom/se2';
import { bresenham, logOddsToProb, type OccupancyGrid } from '../mapping/occgrid';
import { binaryEntropy } from '../prob/gaussian';
import { beamAngles } from '../sim/world';

/**
 * The sensor, as the *planner* models it — deliberately cruder than the beam
 * mixture of Chapter 10. All the planner needs from a range finder is: how many
 * beams, how far, and how much to trust one cell's verdict.
 */
export interface SensingParams {
  nBeams: number;
  /** Total angular field of view, radians. */
  fov: number;
  maxRange: number;
  /**
   * Per-cell reliability q of the inverse model's verdict: the scan calls a
   * cell occupied when it is occupied with probability q. It is the crossover
   * of a binary symmetric channel, and it is the only sensor number the gain
   * estimator sees.
   */
  zHit: number;
}

export const DEFAULT_SENSING: SensingParams = {
  nBeams: 24,
  fov: 2 * Math.PI,
  maxRange: 5,
  zHit: 0.9,
};

/**
 * `cell_mutual_information` — I(mᵢ ; zᵢ) in **bits** for one cell.
 *
 * The cell's occupancy mᵢ ∈ {0, 1} has prior p. The scan reports it through a
 * binary symmetric channel with crossover 1 − q. Then
 *
 *   I = H_b(p) − E_z[H_b(p | z)]
 *
 * with the two posteriors given by Bayes' rule. Two limits are worth carrying
 * around: at p = ½ this collapses to the channel capacity 1 − H_b(q), and as
 * p → 0 or 1 it goes to zero — a cell you already know says nothing back.
 */
export function cellMutualInformation(p: number, q: number): number {
  if (!(p > 0) || !(p < 1)) return 0;
  const pOcc = p * q + (1 - p) * (1 - q);
  const pFree = 1 - pOcc;
  const postOcc = pOcc > 0 ? (p * q) / pOcc : 0;
  const postFree = pFree > 0 ? (p * (1 - q)) / pFree : 0;
  const expected = pOcc * binaryEntropy(postOcc) + pFree * binaryEntropy(postFree);
  return Math.max(0, binaryEntropy(p) - expected);
}

/** One cell's row in the ray table — exactly the columns of the chapter's worked example. */
export interface RayCell {
  /** Position along the ray, 1-based, as the worked example numbers them. */
  k: number;
  p: number;
  /** R_k = Π_{m<k} (1 − p_m): the chance the beam gets this far. */
  reach: number;
  /** I(m_k ; z_k), the gain *if* the beam arrives. */
  mi: number;
  /** reach · mi — the contribution actually banked. */
  gain: number;
}

export interface RayGain {
  cells: RayCell[];
  /** Σ_k reach·mi, in bits. */
  total: number;
}

/**
 * `ray_info_gain` — expected entropy reduction along one beam, in bits.
 *
 *   I(ray) = Σ_k R_k · I(m_k ; z_k),   R_k = Π_{m<k} (1 − p_m)
 *
 * The recursion stops early once the reach probability falls below
 * `reachFloor`: a beam already stopped by three half-occupied cells is not
 * going to tell you much about the fourth.
 */
export function rayInfoGain(probs: number[], q: number, reachFloor = 1e-3): RayGain {
  const cells: RayCell[] = [];
  let reach = 1;
  let total = 0;
  for (let k = 0; k < probs.length; k++) {
    if (reach < reachFloor) break;
    const p = probs[k];
    const mi = cellMutualInformation(p, q);
    const gain = reach * mi;
    cells.push({ k: k + 1, p, reach, mi, gain });
    total += gain;
    reach *= 1 - p;
  }
  return { cells, total };
}

export interface GainOptions {
  /**
   * Count each cell at most once per scan. Neighbouring beams share the cells
   * nearest the robot, and without this the estimator happily rewards standing
   * still with its nose against a wall. Chapter 13's `integrateScan` keeps the
   * same visited set for the same reason.
   */
  dedupe?: boolean;
  reachFloor?: number;
}

/**
 * `expected_info_gain` — bits of map entropy one scan from `pose` is expected
 * to remove, summed over beams.
 *
 * Beams are treated as independent, which they are not: two adjacent beams
 * through the same doorway are counted separately even though one measurement
 * largely determines the other. The `dedupe` visited set removes the crudest
 * part of that double count; what remains is an over-estimate, and the chapter
 * says so rather than hiding it.
 */
export function expectedInfoGain(
  grid: OccupancyGrid,
  pose: Pose2,
  sensor: SensingParams = DEFAULT_SENSING,
  opts: GainOptions = {},
): number {
  const { dedupe = true, reachFloor = 1e-3 } = opts;
  const [ri, rj] = grid.worldToCell(pose.x, pose.y);
  const visited = dedupe ? new Set<number>() : null;
  let total = 0;

  for (const rel of beamAngles(sensor)) {
    const a = pose.theta + rel;
    const ex = pose.x + sensor.maxRange * Math.cos(a);
    const ey = pose.y + sensor.maxRange * Math.sin(a);
    const [ei, ej] = grid.worldToCell(ex, ey);

    let reach = 1;
    for (const [i, j] of bresenham(ri, rj, ei, ej)) {
      if (reach < reachFloor) break;
      if (!grid.inBounds(i, j)) break;
      const idx = grid.index(i, j);
      const p = logOddsToProb(grid.logOdds[idx]);
      if (!visited || !visited.has(idx)) {
        visited?.add(idx);
        total += reach * cellMutualInformation(p, sensor.zHit);
      }
      // The beam is stopped by the cell whether or not we have already scored
      // it: occlusion is a property of the map, not of the bookkeeping.
      reach *= 1 - p;
    }
  }
  return total;
}

/**
 * The surrogate everybody actually ships: count the *unknown* cells the scan
 * would touch, weighted by reach probability.
 *
 * It is `expectedInfoGain` with `I(m_k ; z_k)` replaced by the indicator
 * "is this cell still at the prior", so it is exactly proportional to the exact
 * gain on a map whose unknown cells all sit at p = ½ — which, before the first
 * scan, is all of them. The chapter measures how well the correlation survives
 * once the map is half built.
 */
export function unknownCellSurrogate(
  grid: OccupancyGrid,
  pose: Pose2,
  sensor: SensingParams = DEFAULT_SENSING,
  opts: GainOptions & { unknownBand?: number } = {},
): number {
  const { dedupe = true, reachFloor = 1e-3, unknownBand = 0.05 } = opts;
  const [ri, rj] = grid.worldToCell(pose.x, pose.y);
  const visited = dedupe ? new Set<number>() : null;
  let total = 0;

  for (const rel of beamAngles(sensor)) {
    const a = pose.theta + rel;
    const ex = pose.x + sensor.maxRange * Math.cos(a);
    const ey = pose.y + sensor.maxRange * Math.sin(a);
    const [ei, ej] = grid.worldToCell(ex, ey);

    let reach = 1;
    for (const [i, j] of bresenham(ri, rj, ei, ej)) {
      if (reach < reachFloor) break;
      if (!grid.inBounds(i, j)) break;
      const idx = grid.index(i, j);
      const p = logOddsToProb(grid.logOdds[idx]);
      if ((!visited || !visited.has(idx)) && Math.abs(p - 0.5) < unknownBand) {
        visited?.add(idx);
        total += reach;
      } else {
        visited?.add(idx);
      }
      reach *= 1 - p;
    }
  }
  return total;
}
