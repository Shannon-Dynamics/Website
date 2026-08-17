/**
 * Point clouds, voxel hashing, and surface normals — the data structures a
 * scan-matching front-end needs before it can match anything.
 *
 * A LiDAR scan arrives as ranges along known bearings. Registration wants
 * *points*, so the first thing every front-end does is project the scan into
 * Cartesian coordinates in the sensor frame. Everything downstream (ICP, NDT,
 * the local map) then works on `Pt` arrays and never looks at a range again.
 *
 * The Rust counterpart is `crates/ch16_slam2d/src/{scan,voxel_map}.rs`, where
 * `Pt` is `nalgebra::Point2<f64>` and `VoxelMap` is a `hashbrown::HashMap`
 * keyed by the same integer cell coordinates.
 */

import { transformPoint, type Pose2 } from '../geom/se2';
import { symEig } from '../models/motion-se2';

/** A planar point. A tuple, not an object: these arrays get long. */
export type Pt = [number, number];

export interface PointCloud {
  points: Pt[];
  /** Simulation time the sweep was taken at, in seconds. */
  stamp: number;
}

/**
 * Project a range scan into sensor-frame points, dropping max-range returns.
 *
 * A beam that reports `maxRange` hit *nothing*. Keeping it would plant a
 * phantom point on the horizon that ICP will happily match to a real wall, so
 * the dropout must be discarded rather than clamped — the single most common
 * bug in a first scan matcher.
 */
export function scanToCloud(
  ranges: readonly number[],
  angles: readonly number[],
  maxRange: number,
  stamp = 0,
): PointCloud {
  const points: Pt[] = [];
  for (let k = 0; k < ranges.length; k++) {
    const r = ranges[k];
    if (r >= maxRange * 0.995) continue;
    points.push([r * Math.cos(angles[k]), r * Math.sin(angles[k])]);
  }
  return { points, stamp };
}

/** Map every point through a pose: p ↦ R p + t. */
export function transformCloud(pose: Pose2, points: readonly Pt[]): Pt[] {
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  return points.map(([x, y]) => [pose.x + c * x - s * y, pose.y + s * x + c * y] as Pt);
}

/** Convenience wrapper matching the SE(2) helper, for a single point. */
export const applyPose = (pose: Pose2, p: Pt): Pt => transformPoint(pose, p) as Pt;

/**
 * One point per occupied voxel — the centroid of the points that landed there.
 *
 * Downsampling is not (only) about speed. A raw LiDAR sweep is far denser near
 * the sensor than far from it, so an un-downsampled cost is dominated by
 * whatever wall happens to be closest. Voxelizing makes the point density
 * uniform in *space*, which makes the least-squares problem an unbiased one.
 */
export function voxelDownsample(points: readonly Pt[], cell: number): Pt[] {
  const acc = new Map<number, [number, number, number]>();
  for (const [x, y] of points) {
    const key = cellKey(Math.floor(x / cell), Math.floor(y / cell));
    const a = acc.get(key);
    if (a) {
      a[0] += x;
      a[1] += y;
      a[2] += 1;
    } else {
      acc.set(key, [x, y, 1]);
    }
  }
  const out: Pt[] = [];
  for (const [sx, sy, n] of acc.values()) out.push([sx / n, sy / n]);
  return out;
}

/** Pack integer cell coordinates into one number key. ±32768 cells is plenty. */
export const cellKey = (i: number, j: number): number => (i + 32768) * 65536 + (j + 32768);

/**
 * A voxel-hash point map: KISS-ICP's local map, minus the 3-D.
 *
 * The only query a scan matcher makes is "what is the nearest map point to
 * this one, within τ?", and a hash grid answers it in expected O(1) by looking
 * at the ⌈τ/cell⌉-ring of cells around the query. A k-d tree answers the same
 * question in O(log M) but has to be rebuilt whenever the map grows, which for
 * a map that grows every frame is the wrong trade.
 *
 * `maxPerCell` bounds memory *and* acts as a crude outlier filter: a cell that
 * has already seen its quota of points ignores further evidence, so a passing
 * pedestrian cannot stuff the map with a wall that is not there.
 */
export class VoxelMap {
  readonly cell: number;
  readonly maxPerCell: number;
  readonly pts: Pt[] = [];
  readonly normals: (Pt | null)[] = [];
  private readonly cells = new Map<number, number[]>();

  constructor(cell = 0.2, maxPerCell = 4) {
    this.cell = cell;
    this.maxPerCell = maxPerCell;
  }

  get size(): number {
    return this.pts.length;
  }

  insert(points: readonly Pt[], normals?: readonly (Pt | null)[]): void {
    for (let k = 0; k < points.length; k++) {
      const p = points[k];
      const key = cellKey(Math.floor(p[0] / this.cell), Math.floor(p[1] / this.cell));
      let bucket = this.cells.get(key);
      if (!bucket) {
        bucket = [];
        this.cells.set(key, bucket);
      }
      if (bucket.length >= this.maxPerCell) continue;
      bucket.push(this.pts.length);
      this.pts.push(p);
      this.normals.push(normals?.[k] ?? null);
    }
  }

  /** Index of the nearest stored point within `rMax`, or −1. */
  nearestIndex(p: Pt, rMax: number): number {
    const ring = Math.max(1, Math.ceil(rMax / this.cell));
    const ci = Math.floor(p[0] / this.cell);
    const cj = Math.floor(p[1] / this.cell);
    let best = -1;
    let bestD2 = rMax * rMax;
    for (let di = -ring; di <= ring; di++) {
      for (let dj = -ring; dj <= ring; dj++) {
        const bucket = this.cells.get(cellKey(ci + di, cj + dj));
        if (!bucket) continue;
        for (const idx of bucket) {
          const q = this.pts[idx];
          const dx = q[0] - p[0];
          const dy = q[1] - p[1];
          const d2 = dx * dx + dy * dy;
          if (d2 < bestD2) {
            bestD2 = d2;
            best = idx;
          }
        }
      }
    }
    return best;
  }

  /** Every point within `r` — used by normal estimation and by the NDT build. */
  within(p: Pt, r: number): number[] {
    const ring = Math.max(1, Math.ceil(r / this.cell));
    const ci = Math.floor(p[0] / this.cell);
    const cj = Math.floor(p[1] / this.cell);
    const r2 = r * r;
    const out: number[] = [];
    for (let di = -ring; di <= ring; di++) {
      for (let dj = -ring; dj <= ring; dj++) {
        const bucket = this.cells.get(cellKey(ci + di, cj + dj));
        if (!bucket) continue;
        for (const idx of bucket) {
          const q = this.pts[idx];
          const dx = q[0] - p[0];
          const dy = q[1] - p[1];
          if (dx * dx + dy * dy <= r2) out.push(idx);
        }
      }
    }
    return out;
  }
}

/**
 * Per-point surface normals by local PCA.
 *
 * Fit a 2×2 scatter matrix to the neighbours inside `radius`; the eigenvector
 * of its **smaller** eigenvalue points across the surface, because that is the
 * direction the neighbours vary least in. Points whose neighbourhood is round
 * rather than elongated (a corner, an isolated speckle) get `null` — a normal
 * we do not believe is worse than no normal, since point-to-plane weights every
 * residual by it.
 */
export function estimateNormals(
  points: readonly Pt[],
  radius = 0.35,
  minNeighbors = 4,
  /** Reject a normal when λ_min/λ_max exceeds this — the surface is not flat. */
  planarity = 0.35,
): (Pt | null)[] {
  const grid = new VoxelMap(Math.max(radius, 0.05), 64);
  grid.insert(points);
  return points.map((p) => {
    const idx = grid.within(p, radius);
    if (idx.length < minNeighbors) return null;
    let mx = 0;
    let my = 0;
    for (const i of idx) {
      mx += points[i][0];
      my += points[i][1];
    }
    mx /= idx.length;
    my /= idx.length;
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (const i of idx) {
      const dx = points[i][0] - mx;
      const dy = points[i][1] - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const { values, vectors } = symEig([
      [sxx, sxy],
      [sxy, syy],
    ]);
    // symEig sorts descending and returns eigenvectors as rows, so `vectors[1]`
    // is the direction of *least* variance — across the surface.
    const lMax = Math.max(values[0], 1e-12);
    if (values[1] / lMax > planarity) return null;
    const [nx, ny] = vectors[1];
    const len = Math.hypot(nx, ny) || 1;
    return [nx / len, ny / len] as Pt;
  });
}
