/**
 * Where should Rusty go in order to learn something? Chapter 24's answer,
 * wired into the capstone.
 *
 * A **frontier** is a known-free cell with an unknown neighbour: the boundary
 * of what the robot has seen. Yamauchi's 1997 observation is that driving to
 * one is guaranteed to reveal new space, and that when none remain the map is
 * finished — which is half of the stopping criterion D26.3. The other half,
 * the entropy-rate test, lives in `stack.ts`, because a frontier count alone is
 * gameable: a single stubborn unreachable cell keeps a robot alive forever.
 *
 * Rust counterpart: `crates/capstone/src/tasks/explore.rs`.
 */

import type { OccupancyGrid } from '../mapping/occgrid';
import { FREE, UNKNOWN } from './astar';

export interface ScoredFrontier {
  id: number;
  /** Cell indices (j·width + i) that make up this frontier. */
  cells: number[];
  centroid: [number, number];
  /** The cheapest-to-reach cell of the cluster — the actual navigation goal. */
  goal: [number, number];
  /** Cluster size in cells; `size · cellArea` is the frontier's length-scale. */
  size: number;
  /** Unknown cells within one sensor radius of the goal: the information on offer. */
  gain: number;
  /** Dijkstra path cost from the robot, in metres. Infinite if unreachable. */
  cost: number;
  /** gain · exp(−λ · cost): Chapter 24's utility, one greedy step. */
  utility: number;
}

export interface FrontierConfig {
  /**
   * A_min of D26.3, in cells. Ignore clusters smaller than this — sensor
   * speckle and one-cell holes behind wall corners, not rooms. Set it too small
   * and the mission never terminates, because a floorplan rasterised onto a
   * grid always leaves a few unreachable specks of unknown.
   */
  minCells: number;
  /** Frontiers offering less than this many unknown cells are not worth a drive. */
  minGain: number;
  /** Sensor radius used to estimate information gain, metres. */
  sensorRadius: number;
  /** Distance discount λ, per metre. Larger = lazier robot. */
  lambda: number;
}

export const DEFAULT_FRONTIER: FrontierConfig = {
  minCells: 6,
  minGain: 14,
  sensorRadius: 2.4,
  lambda: 0.22,
};

/**
 * `frontier_tick` — one linear pass to label frontier cells, then one
 * flood-fill to group them.
 *
 * Both passes are O(cells); on the capstone's 80 × 60 map that is under five
 * thousand visits, which is why this task can afford to run at 1 Hz on the same
 * thread as everything else.
 */
export function detectFrontiers(
  grid: OccupancyGrid,
  cls: Uint8Array,
  costField: Float64Array,
  cfg: FrontierConfig = DEFAULT_FRONTIER,
): ScoredFrontier[] {
  const nx = grid.width;
  const ny = grid.height;

  // --- label ---------------------------------------------------------------
  const isFrontier = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (cls[k] !== FREE) continue;
      // 4-connectivity for the unknown test: a diagonal-only touch is usually
      // a rasterisation artefact of the beam, not a real opening.
      const up = j + 1 < ny && cls[k + nx] === UNKNOWN;
      const dn = j > 0 && cls[k - nx] === UNKNOWN;
      const lf = i > 0 && cls[k - 1] === UNKNOWN;
      const rt = i + 1 < nx && cls[k + 1] === UNKNOWN;
      if (up || dn || lf || rt) isFrontier[k] = 1;
    }
  }

  // --- cluster -------------------------------------------------------------
  const seen = new Uint8Array(nx * ny);
  const out: ScoredFrontier[] = [];
  const stack: number[] = [];
  let id = 0;

  for (let start = 0; start < nx * ny; start++) {
    if (!isFrontier[start] || seen[start]) continue;
    seen[start] = 1;
    stack.length = 0;
    stack.push(start);
    const cells: number[] = [];

    while (stack.length > 0) {
      const k = stack.pop() as number;
      cells.push(k);
      const i = k % nx;
      const j = (k - i) / nx;
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dj === 0) continue;
          const ni = i + di;
          const nj = j + dj;
          if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
          const nk = nj * nx + ni;
          if (!isFrontier[nk] || seen[nk]) continue;
          seen[nk] = 1;
          stack.push(nk);
        }
      }
    }

    if (cells.length < cfg.minCells) continue;

    let sx = 0;
    let sy = 0;
    let best = Infinity;
    let bestCell = cells[0];
    for (const k of cells) {
      const i = k % nx;
      const j = (k - i) / nx;
      const [cx, cy] = grid.cellCenter(i, j);
      sx += cx;
      sy += cy;
      if (costField[k] < best) {
        best = costField[k];
        bestCell = k;
      }
    }
    const bi = bestCell % nx;
    const bj = (bestCell - bi) / nx;
    const goal = grid.cellCenter(bi, bj);
    const centroid: [number, number] = [sx / cells.length, sy / cells.length];
    const gain = unknownWithin(grid, cls, goal, cfg.sensorRadius);
    if (gain < cfg.minGain) continue;
    const cost = best;
    const utility = Number.isFinite(cost) ? gain * Math.exp(-cfg.lambda * cost) : 0;

    out.push({ id: id++, cells, centroid, goal, size: cells.length, gain, cost, utility });
  }

  out.sort((a, b) => b.utility - a.utility);
  return out;
}

/**
 * Expected information gain, approximated by counting unknown cells inside the
 * sensor disc.
 *
 * The honest quantity is the expected entropy reduction, which needs a ray-cast
 * per candidate pose and a model of what each beam would resolve. This count is
 * its cheap upper bound: it ignores occlusion, so a frontier facing a wall is
 * over-valued. The capstone accepts that because the greedy choice is revisited
 * every second — and because over-valuing a blocked frontier costs one wasted
 * replan, while ray-casting every frontier costs the frame budget.
 */
function unknownWithin(
  grid: OccupancyGrid,
  cls: Uint8Array,
  at: [number, number],
  radius: number,
): number {
  const r = Math.ceil(radius / grid.cellSize);
  const [ci, cj] = grid.worldToCell(at[0], at[1]);
  let n = 0;
  for (let dj = -r; dj <= r; dj++) {
    for (let di = -r; di <= r; di++) {
      if (di * di + dj * dj > r * r) continue;
      const i = ci + di;
      const j = cj + dj;
      if (i < 0 || j < 0 || i >= grid.width || j >= grid.height) continue;
      if (cls[j * grid.width + i] === UNKNOWN) n++;
    }
  }
  return n;
}

/**
 * D26.3, first half: is there anything left worth driving to?
 *
 * Both conditions of the definition have to hold, and this is only one of them.
 * The stack ANDs it with the entropy-rate test before it will say `Done`.
 */
export function frontiersExhausted(frontiers: readonly ScoredFrontier[], minCells: number): boolean {
  return !frontiers.some((f) => f.size >= minCells && Number.isFinite(f.cost));
}
