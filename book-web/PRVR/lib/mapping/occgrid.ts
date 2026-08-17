/**
 * Occupancy grid mapping with known poses — Thrun et al., Ch. 9.
 *
 * The map is stored in **log odds** l = log p/(1−p), for two reasons the
 * chapter makes much of: Bayes' rule becomes addition, so integrating a scan is
 * a loop of `+=`; and probabilities never saturate at exactly 0 or 1, so a
 * cell that was wrongly marked occupied can still be argued out of it.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import { binaryEntropy } from '../prob/gaussian';
import type { Bounds, World } from '../sim/world';

export const logOddsToProb = (l: number): number => 1 - 1 / (1 + Math.exp(l));
export const probToLogOdds = (p: number): number => Math.log(p / (1 - p));

export interface InverseModelParams {
  /** Obstacle thickness, metres: the band around z marked occupied. */
  alpha: number;
  /** Beam width, radians: how far off-axis a cell may sit and still be seen. */
  beta: number;
  maxRange: number;
  /** Log odds written for a cell believed occupied / free / unknown. */
  lOcc: number;
  lFree: number;
  l0: number;
  /** Symmetric clamp on the accumulated log odds, keeping the map revisable. */
  clamp?: number;
}

export const DEFAULT_INVERSE_MODEL: InverseModelParams = {
  alpha: 0.2,
  beta: (5 * Math.PI) / 180,
  maxRange: 8,
  lOcc: probToLogOdds(0.75),
  lFree: probToLogOdds(0.35),
  l0: 0,
  clamp: 12,
};

/**
 * `inverse_range_sensor_model` — Thrun et al., **Table 9.2**.
 *
 * Answers "what does this scan say about *this* cell?" and returns a log-odds
 * value. Note it is an *inverse* model: p(m | z, x), not p(z | m, x). That
 * inversion is the approximation the whole chapter rests on, and it is why two
 * beams disagreeing about one cell simply average out instead of fighting.
 *
 * Three cases: beyond the reading (or outside the beam) → no information;
 * within α/2 of the reading → occupied; nearer than the reading → free.
 */
export function inverseRangeSensorModel(
  cellCenter: [number, number],
  pose: Pose2,
  z: number[],
  angles: number[],
  params: InverseModelParams,
): number {
  const { alpha, beta, maxRange, lOcc, lFree, l0 } = params;
  const dx = cellCenter[0] - pose.x;
  const dy = cellCenter[1] - pose.y;
  const r = Math.hypot(dx, dy);
  const phi = normalizeAngle(Math.atan2(dy, dx) - pose.theta);

  // k = argmin_j |φ − θ_j|: the beam this cell belongs to.
  let k = 0;
  let bestDiff = Infinity;
  for (let j = 0; j < angles.length; j++) {
    const d = Math.abs(normalizeAngle(phi - angles[j]));
    if (d < bestDiff) {
      bestDiff = d;
      k = j;
    }
  }

  const zk = z[k];
  if (r > Math.min(maxRange, zk + alpha / 2) || bestDiff > beta / 2) return l0;
  if (zk < maxRange && Math.abs(r - zk) < alpha / 2) return lOcc;
  if (r <= zk) return lFree;
  return l0;
}

/**
 * Bresenham's line algorithm — integer arithmetic only, all eight octants.
 *
 * Used to enumerate the cells a beam passes through. Exact rasterisation is
 * what keeps the free-space carving crisp; stepping along the ray at fixed
 * intervals instead leaves diagonal gaps that show up as speckle in the map.
 */
export function bresenham(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): [number, number][] {
  const out: [number, number][] = [];
  let x = Math.round(x0);
  let y = Math.round(y0);
  const ex = Math.round(x1);
  const ey = Math.round(y1);
  const dx = Math.abs(ex - x);
  const dy = -Math.abs(ey - y);
  const sx = x < ex ? 1 : -1;
  const sy = y < ey ? 1 : -1;
  let err = dx + dy;

  for (;;) {
    out.push([x, y]);
    if (x === ex && y === ey) break;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
  return out;
}

export interface OccupancyGridOptions {
  /** Grid size in cells. */
  width: number;
  height: number;
  cellSize: number;
  /** World coordinates of the grid's lower-left corner. */
  origin: { x: number; y: number };
  /** Prior occupancy probability; 0.5 (l₀ = 0) means "no idea". */
  prior?: number;
}

export class OccupancyGrid {
  readonly width: number;
  readonly height: number;
  readonly cellSize: number;
  readonly origin: { x: number; y: number };
  readonly logOdds: Float64Array;
  /** l₀, the prior in log odds — subtracted off every update, per Table 9.1. */
  readonly l0: number;

  constructor(opts: OccupancyGridOptions) {
    this.width = opts.width;
    this.height = opts.height;
    this.cellSize = opts.cellSize;
    this.origin = { ...opts.origin };
    this.l0 = probToLogOdds(opts.prior ?? 0.5);
    this.logOdds = new Float64Array(this.width * this.height).fill(this.l0);
  }

  /** A grid covering a world's bounds at the given resolution. */
  static forBounds(bounds: Bounds, cellSize: number, prior = 0.5): OccupancyGrid {
    return new OccupancyGrid({
      width: Math.max(1, Math.ceil((bounds.maxX - bounds.minX) / cellSize)),
      height: Math.max(1, Math.ceil((bounds.maxY - bounds.minY) / cellSize)),
      cellSize,
      origin: { x: bounds.minX, y: bounds.minY },
      prior,
    });
  }

  static forWorld(world: World, cellSize: number, prior = 0.5): OccupancyGrid {
    return OccupancyGrid.forBounds(world.bounds, cellSize, prior);
  }

  index(i: number, j: number): number {
    return j * this.width + i;
  }

  inBounds(i: number, j: number): boolean {
    return i >= 0 && j >= 0 && i < this.width && j < this.height;
  }

  worldToCell(x: number, y: number): [number, number] {
    return [
      Math.floor((x - this.origin.x) / this.cellSize),
      Math.floor((y - this.origin.y) / this.cellSize),
    ];
  }

  cellCenter(i: number, j: number): [number, number] {
    return [
      this.origin.x + (i + 0.5) * this.cellSize,
      this.origin.y + (j + 0.5) * this.cellSize,
    ];
  }

  probAt(i: number, j: number): number {
    if (!this.inBounds(i, j)) return logOddsToProb(this.l0);
    return logOddsToProb(this.logOdds[this.index(i, j)]);
  }

  probAtWorld(x: number, y: number): number {
    const [i, j] = this.worldToCell(x, y);
    return this.probAt(i, j);
  }

  /** Reset every cell to a uniform prior probability. */
  setPrior(p = 0.5): void {
    this.logOdds.fill(probToLogOdds(p));
  }

  /**
   * Integrate one scan — Thrun et al., **Table 9.1**:
   *
   *   l_{t,i} = l_{t−1,i} + inverse_sensor_model(mᵢ, xₜ, zₜ) − l₀
   *
   * Table 9.1 loops over *every* cell in the map; we instead enumerate only the
   * cells inside the perceptual field, by Bresenham-tracing each beam out to
   * z + α/2. Same result, minus the 99% of the map the scan says nothing about.
   *
   * The visited set matters: without it, cells near the robot sit in several
   * beams' rasters and would be counted once per beam, over-carving free space
   * into a hard 0 that later evidence cannot undo.
   */
  integrateScan(
    pose: Pose2,
    ranges: number[],
    angles: number[],
    params: InverseModelParams = DEFAULT_INVERSE_MODEL,
  ): void {
    const clamp = params.clamp ?? Infinity;
    const [ri, rj] = this.worldToCell(pose.x, pose.y);
    const visited = new Set<number>();

    for (let k = 0; k < ranges.length; k++) {
      const a = pose.theta + angles[k];
      const reach = Math.min(ranges[k] + params.alpha / 2, params.maxRange);
      const ex = pose.x + reach * Math.cos(a);
      const ey = pose.y + reach * Math.sin(a);
      const [ei, ej] = this.worldToCell(ex, ey);

      for (const [i, j] of bresenham(ri, rj, ei, ej)) {
        if (!this.inBounds(i, j)) continue;
        const idx = this.index(i, j);
        if (visited.has(idx)) continue;
        visited.add(idx);

        const l = inverseRangeSensorModel(this.cellCenter(i, j), pose, ranges, angles, params);
        const next = this.logOdds[idx] + l - params.l0;
        this.logOdds[idx] = Math.max(-clamp, Math.min(clamp, next));
      }
    }
  }

  /** Row-major (j·width + i) probabilities — feed straight into an ImageData loop. */
  getProbabilityArray(): Float64Array {
    const out = new Float64Array(this.logOdds.length);
    for (let i = 0; i < this.logOdds.length; i++) {
      out[i] = logOddsToProb(this.logOdds[i]);
    }
    return out;
  }

  /**
   * Total map entropy in **bits**: Σᵢ H(pᵢ), treating cells as independent.
   *
   * This is the curve the mapping widget plots as the robot explores — it
   * starts at exactly `width·height` bits (every cell at p = 0.5) and falls as
   * the map is resolved. It is also the objective an information-gain explorer
   * would minimise in the active-localization chapter.
   */
  entropy(): number {
    let h = 0;
    for (let i = 0; i < this.logOdds.length; i++) {
      h += binaryEntropy(logOddsToProb(this.logOdds[i]));
    }
    return h;
  }
}
