/**
 * The scene w22.2 plans in: one long hall, one textured wall.
 *
 * The geometry is the whole argument, so it is worth stating plainly. The south
 * wall is shelving, radiators and skirting board — a LiDAR scan that reaches it
 * pins Rusty's pose down. Every other surface has just been plastered: it stops
 * the robot but tells it nothing, which is exactly the situation Chapter 16's
 * scan matcher degenerates in. Rusty starts at the west end and must drive
 * through a doorway in the smooth east wall.
 *
 * The straight crossing is 31 steps and completely blind. Following the coast
 * costs a detour and stays localized. Nothing in the map says "prefer walls":
 * the AMDP value function has to earn that, and it only does so while the LiDAR
 * range is short enough that the middle of the hall really is featureless.
 *
 * This is scene data, not algorithm — every function that acts on it lives in
 * `lib/pomdp/amdp.ts`.
 */

import { idx, type CoastalParams, type CoastalWorld, DEFAULT_COASTAL } from '@/lib/pomdp/amdp';

export const DOOR_ROW = 10;

export function makeHall(): CoastalWorld {
  const cols = 34;
  const rows = 21;
  const cell = 0.5;
  const n = cols * rows;
  const occ = new Array<boolean>(n).fill(false);
  const feature = new Array<boolean>(n).fill(false);
  const set = (x: number, y: number, textured: boolean) => {
    occ[y * cols + x] = true;
    feature[y * cols + x] = textured;
  };

  for (let x = 0; x < cols; x++) {
    set(x, 0, true); // the coast: textured, and the only thing worth scanning
    set(x, rows - 1, false); // smooth north partition
  }
  for (let y = 0; y < rows; y++) {
    set(0, y, false);
    set(cols - 1, y, false);
  }
  // A two-cell doorway, so the final approach is a gap and not a knife edge.
  occ[DOOR_ROW * cols + (cols - 1)] = false;
  occ[(DOOR_ROW + 1) * cols + (cols - 1)] = false;

  const w: CoastalWorld = {
    cols,
    rows,
    cell,
    occ,
    feature,
    start: 0,
    goal: 0,
    doorHalfWidth: 0.6,
  };
  w.start = idx(w, 2, DOOR_ROW);
  w.goal = idx(w, cols - 1, DOOR_ROW);
  return w;
}

/**
 * σ bins fine enough that the value function does not develop a limit cycle at
 * a bin boundary — a real failure mode of the AMDP compression, and one the
 * chapter says out loud.
 */
export const SIGMA_BINS = [0.04, 0.055, 0.075, 0.1, 0.135, 0.18, 0.24, 0.32, 0.42, 0.56, 0.75, 1.0];

export const HALL_PARAMS: CoastalParams = {
  ...DEFAULT_COASTAL,
  motionSigma: 0.1,
  nuMax: 20,
  riskWeight: 3,
  goalReward: 100,
  lostPenalty: -60,
  gamma: 0.985,
  slip: 0.08,
  sigmaBins: SIGMA_BINS,
  range: 2.5,
};

/** Where a run starts believing it is, in metres of position standard deviation. */
export const SIGMA_0 = 0.15;
