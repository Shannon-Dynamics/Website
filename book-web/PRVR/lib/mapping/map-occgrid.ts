/**
 * The three pieces of Chapter 9 that exist to show where the standard mapper in
 * `occgrid.ts` is *wrong*.
 *
 *  1. `occupancyGridMappingAllCells` — Thrun et al., **Table 9.1** taken
 *     literally: a loop over every cell in the perceptual field, not just the
 *     cells a Bresenham raster happens to touch. The optimised version in
 *     `occgrid.ts` is exact for a pencil-thin LiDAR beam and *silently wrong*
 *     for a sonar whose cone is wider than the beam spacing — and the wide cone
 *     is precisely the regime in which per-cell independence falls apart. You
 *     cannot demonstrate the failure without the honest sweep.
 *
 *  2. `MAP_occupancy_grid_mapping` — Thrun et al., **Table 9.3**. Drops the
 *     factorisation entirely and maximises the un-factored log posterior
 *     `log p(z_{1:t} | x_{1:t}, m) + log p(m)` by hill climbing over binary
 *     maps, scoring candidate maps with the *forward* beam model of Chapter 10.
 *
 *  3. `fuseGrids` — Thrun et al., **eq. (9.9)**. One grid per sensor modality,
 *     combined pessimistically with a max instead of by adding log odds.
 *
 * Everything here is deterministic: no RNG, no time, no hidden state.
 */

import type { Pose2 } from '../geom/se2';
import { beamLikelihood, type BeamParams } from '../models/sensor';
import {
  inverseRangeSensorModel,
  logOddsToProb,
  probToLogOdds,
  type InverseModelParams,
  type OccupancyGrid,
} from './occgrid';

// ---------------------------------------------------------------------------
// Table 9.1, unabridged
// ---------------------------------------------------------------------------

/**
 * `occupancy_grid_mapping` — Thrun et al., **Table 9.1**, written the way the
 * table writes it:
 *
 *   for all cells m_i: if m_i ∈ perceptual field of z_t,
 *                      l_{t,i} = l_{t−1,i} + inverse_sensor_model(m_i, x_t, z_t) − l_0
 *
 * The only concession to speed is the bounding box: cells further than
 * `maxRange + α` from the sensor cannot be in any beam's cone, so they are
 * skipped without evaluating the inverse model. Everything inside the box is
 * visited, which is what lets a cell sitting *off* the beam axis — inside the
 * cone but never on the ray — receive evidence. That is the whole difference
 * between a LiDAR and a sonar, and the source of the conflicts of §9.4.
 */
export function occupancyGridMappingAllCells(
  grid: OccupancyGrid,
  pose: Pose2,
  ranges: number[],
  angles: number[],
  params: InverseModelParams,
): void {
  const clamp = params.clamp ?? Infinity;
  const reach = params.maxRange + params.alpha;
  const [iLo, jLo] = grid.worldToCell(pose.x - reach, pose.y - reach);
  const [iHi, jHi] = grid.worldToCell(pose.x + reach, pose.y + reach);

  for (let j = Math.max(0, jLo); j <= Math.min(grid.height - 1, jHi); j++) {
    for (let i = Math.max(0, iLo); i <= Math.min(grid.width - 1, iHi); i++) {
      const l = inverseRangeSensorModel(grid.cellCenter(i, j), pose, ranges, angles, params);
      // Line 6 of the table: outside the perceptual field the cell is untouched.
      if (l === params.l0) continue;
      const idx = grid.index(i, j);
      const next = grid.logOdds[idx] + l - params.l0;
      grid.logOdds[idx] = Math.max(-clamp, Math.min(clamp, next));
    }
  }
}

// ---------------------------------------------------------------------------
// Binary maps and the forward (cone) model
// ---------------------------------------------------------------------------

/** Everything about a grid's layout, without its contents. */
export interface GridGeometry {
  width: number;
  height: number;
  cellSize: number;
  origin: { x: number; y: number };
}

/** The cells one ray crosses, and the range at which it enters each of them. */
export interface RayTrace {
  cells: Int32Array;
  dists: Float64Array;
}

/**
 * One measurement, pre-compiled against a fixed grid.
 *
 * `rays` holds the sub-rays that tile the beam's opening angle β. Sub-rays are
 * what make the forward model a *cone* model: a reading is explained as soon as
 * **some** ray in the cone is blocked at the right range, which is exactly the
 * asymmetry the factored filter cannot express (it demands the whole arc be
 * occupied).
 */
export interface ConeBeam {
  /** The reading the sensor actually produced. */
  z: number;
  pose: Pose2;
  /** Absolute bearing of the beam axis, radians. */
  bearing: number;
  rays: RayTrace[];
}

/** March a ray through the grid, recording each new cell and its entry range. */
export function traceRay(
  geom: GridGeometry,
  ox: number,
  oy: number,
  angle: number,
  maxRange: number,
): RayTrace {
  // Half-cell steps: fine enough that no cell of the 4-connected path is
  // skipped, coarse enough that the trace stays short.
  const step = geom.cellSize * 0.5;
  const n = Math.ceil(maxRange / step);
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const cells: number[] = [];
  const dists: number[] = [];
  let last = -1;

  for (let k = 1; k <= n; k++) {
    const d = k * step;
    const i = Math.floor((ox + d * dx - geom.origin.x) / geom.cellSize);
    const j = Math.floor((oy + d * dy - geom.origin.y) / geom.cellSize);
    if (i < 0 || j < 0 || i >= geom.width || j >= geom.height) break;
    const idx = j * geom.width + i;
    if (idx === last) continue;
    last = idx;
    cells.push(idx);
    dists.push(d);
  }
  return { cells: Int32Array.from(cells), dists: Float64Array.from(dists) };
}

export interface ConeBeamOptions {
  /** Opening angle of one beam, radians. */
  beta: number;
  /** Rays used to tile the cone. 1 gives a pencil beam (a LiDAR). */
  subRays: number;
  maxRange: number;
}

/**
 * Compile a set of scans into `ConeBeam`s over one grid.
 *
 * Traces are taken on the *empty* map deliberately. Occupancy only decides
 * where a ray stops, never where it goes, so a beam whose empty-map trace misses
 * cell i can never be affected by flipping cell i. That makes the incidence
 * structure below exact rather than a heuristic.
 */
export function buildConeBeams(
  geom: GridGeometry,
  poses: Pose2[],
  scans: { ranges: number[]; angles: number[] }[],
  opts: ConeBeamOptions,
): ConeBeam[] {
  const out: ConeBeam[] = [];
  for (let t = 0; t < poses.length; t++) {
    const pose = poses[t];
    const scan = scans[t];
    for (let k = 0; k < scan.ranges.length; k++) {
      const bearing = pose.theta + scan.angles[k];
      const rays: RayTrace[] = [];
      for (let s = 0; s < opts.subRays; s++) {
        const frac = opts.subRays === 1 ? 0 : s / (opts.subRays - 1) - 0.5;
        rays.push(traceRay(geom, pose.x, pose.y, bearing + frac * opts.beta, opts.maxRange));
      }
      out.push({ z: scan.ranges[k], pose, bearing, rays });
    }
  }
  return out;
}

/**
 * z*(m) for one cone: the range at which the *first* sub-ray is blocked.
 *
 * "It suffices to assume an obstacle somewhere in the cone of a measurement,
 * and not everywhere" (Thrun et al., §9.4.1). This one `min` is that sentence.
 */
export function coneRange(beam: ConeBeam, occ: Uint8Array, maxRange: number): number {
  let best = maxRange;
  for (const ray of beam.rays) {
    const { cells, dists } = ray;
    for (let k = 0; k < cells.length; k++) {
      if (occ[cells[k]] === 1) {
        if (dists[k] < best) best = dists[k];
        break;
      }
    }
  }
  return best;
}

/** log p(z_t^k | x_t, m) for one beam under the Chapter 10 forward mixture. */
export function beamLogLikelihood(beam: ConeBeam, occ: Uint8Array, params: BeamParams): number {
  const zStar = coneRange(beam, occ, params.maxRange);
  return Math.log(Math.max(beamLikelihood(beam.z, zStar, params), 1e-300));
}

/**
 * The objective of Table 9.3, up to the map-independent constant:
 *
 *   Σ_t log p(z_t | x_t, m) + Σ_i (l_0)^{m_i}
 */
export function mapLogPosterior(
  beams: ConeBeam[],
  occ: Uint8Array,
  params: BeamParams,
  l0: number,
): number {
  let total = 0;
  for (const b of beams) total += beamLogLikelihood(b, occ, params);
  for (let i = 0; i < occ.length; i++) if (occ[i] === 1) total += l0;
  return total;
}

/** For each cell, the beams whose cone passes through it. */
export function beamIncidence(beams: ConeBeam[], cellCount: number): number[][] {
  const out: number[][] = Array.from({ length: cellCount }, () => [] as number[]);
  for (let b = 0; b < beams.length; b++) {
    const seen = new Set<number>();
    for (const ray of beams[b].rays) {
      for (let k = 0; k < ray.cells.length; k++) {
        const c = ray.cells[k];
        if (seen.has(c)) continue;
        seen.add(c);
        out[c].push(b);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Table 9.3 — MAP occupancy grid mapping
// ---------------------------------------------------------------------------

export interface FlipCandidate {
  cell: number;
  /** Δ of the log posterior if this cell is flipped. Positive means climb. */
  gain: number;
  /** 0 → 1 (claiming an obstacle) or 1 → 0 (retracting one). */
  to: 0 | 1;
  /** The beams whose likelihood the flip changes — the bars a widget draws. */
  beams: number[];
}

/**
 * Change in the log posterior from flipping one cell.
 *
 * Only beams incident on the cell are re-evaluated: "only a small number of
 * measurements are affected by flipping a grid cell" (Thrun et al., §9.4.2).
 * `occ` is restored before returning, so this is a pure query.
 */
export function flipGain(
  beams: ConeBeam[],
  occ: Uint8Array,
  incidence: number[][],
  params: BeamParams,
  l0: number,
  cell: number,
): number {
  const list = incidence[cell];
  const was = occ[cell];
  // Flipping a cell no beam can see only pays the prior, so it never climbs
  // (l₀ < 0 for a prior below one half).
  const priorDelta = was === 1 ? -l0 : l0;
  if (list.length === 0) return priorDelta;

  let before = 0;
  for (const b of list) before += beamLogLikelihood(beams[b], occ, params);
  occ[cell] = was === 1 ? 0 : 1;
  let after = 0;
  for (const b of list) after += beamLogLikelihood(beams[b], occ, params);
  occ[cell] = was;

  return after - before + priorDelta;
}

/** The single most profitable flip, or `null` once the map is a local maximum. */
export function bestFlip(
  beams: ConeBeam[],
  occ: Uint8Array,
  incidence: number[][],
  params: BeamParams,
  l0: number,
  tolerance = 1e-9,
): FlipCandidate | null {
  let best: FlipCandidate | null = null;
  for (let c = 0; c < occ.length; c++) {
    const gain = flipGain(beams, occ, incidence, params, l0, c);
    if (gain > tolerance && (best === null || gain > best.gain)) {
      best = { cell: c, gain, to: occ[c] === 1 ? 0 : 1, beams: incidence[c] };
    }
  }
  return best;
}

/**
 * `MAP_occupancy_grid_mapping` — Thrun et al., **Table 9.3**.
 *
 * Table 9.3 sweeps the cells in index order and sets each to its argmax; we
 * take the steepest flip instead. Both are hill climbing on the same objective
 * and both stop at a local maximum; the greedy order is used because it makes
 * the *reason* for each flip legible — the first cells to move are the ones
 * that explain the most measurement mass.
 *
 * Starts from the all-free map, as the table does. Returns the flips in the
 * order they were taken, so a widget can replay them.
 */
export function mapOccupancyGridMapping(
  beams: ConeBeam[],
  cellCount: number,
  params: BeamParams,
  l0: number,
  maxFlips = 2000,
): { occ: Uint8Array; flips: FlipCandidate[] } {
  const occ = new Uint8Array(cellCount);
  const incidence = beamIncidence(beams, cellCount);
  const flips: FlipCandidate[] = [];
  for (let n = 0; n < maxFlips; n++) {
    const f = bestFlip(beams, occ, incidence, params, l0);
    if (!f) break;
    occ[f.cell] = f.to;
    flips.push(f);
  }
  return { occ, flips };
}

// ---------------------------------------------------------------------------
// Multi-sensor fusion — eq. (9.9)
// ---------------------------------------------------------------------------

export type FusionRule = 'sum' | 'max';

/** A read-only grid of probabilities, shaped for `drawOccupancyGrid`. */
export interface FusedGrid extends GridGeometry {
  getProbabilityArray: () => Float64Array;
}

/**
 * Combine one grid per sensor modality.
 *
 * `sum` adds log odds, which is what you get by running a single Bayes filter
 * per cell over both sensors' evidence. It is correct only when every sensor
 * answers the *same* question, and the result otherwise depends on how often
 * each sensor is polled.
 *
 * `max` is Thrun's eq. (9.9), `p(m_i) = max_k p(m_i^{[k]})`: the most
 * pessimistic component. Biased toward occupied, and the right answer whenever
 * the sensors are sensitive to different obstacles.
 */
export function fuseGrids(grids: OccupancyGrid[], rule: FusionRule): FusedGrid {
  if (grids.length === 0) throw new Error('fuseGrids: need at least one grid');
  const ref = grids[0];
  const n = ref.width * ref.height;

  const getProbabilityArray = (): Float64Array => {
    const out = new Float64Array(n);
    if (rule === 'max') {
      for (let i = 0; i < n; i++) {
        let p = 0;
        for (const g of grids) p = Math.max(p, logOddsToProb(g.logOdds[i]));
        out[i] = p;
      }
      return out;
    }
    for (let i = 0; i < n; i++) {
      let l = ref.l0;
      for (const g of grids) l += g.logOdds[i] - g.l0;
      out[i] = logOddsToProb(l);
    }
    return out;
  };

  return {
    width: ref.width,
    height: ref.height,
    cellSize: ref.cellSize,
    origin: ref.origin,
    getProbabilityArray,
  };
}

/** Binary entropy of a whole probability field, in bits — the Chapter 24 currency. */
export function fieldEntropy(probs: Float64Array): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p <= 0 || p >= 1) continue;
    h += -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
  }
  return h;
}

/** Convenience: the log-odds prior for a map prior probability. */
export const mapPriorLogOdds = (prior: number): number => probToLogOdds(prior);
