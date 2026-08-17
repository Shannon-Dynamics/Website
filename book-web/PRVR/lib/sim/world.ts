/**
 * The two worlds this book simulates in.
 *
 * Everything is polyline geometry — a world is a list of wall segments plus a
 * bounding box — because exact ray/segment intersection is cheap, exact, and
 * resolution-independent. The occupancy-grid chapter *builds* a grid from these
 * walls; the walls themselves are never a grid.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import type { Rng } from '../prob/rng';

export interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface Landmark {
  x: number;
  y: number;
  id: number;
  /** Optional signature s, the third component of Thrun's feature vector. */
  signature?: number;
}

export interface World {
  name: string;
  bounds: Bounds;
  walls: Segment[];
  landmarks?: Landmark[];
}

export interface Point2 {
  x: number;
  y: number;
}

// ---------------------------------------------------------------------------
// Floorplan construction helpers
// ---------------------------------------------------------------------------

/** A horizontal wall at height `y`, broken into the given x-spans (doorways are the gaps). */
const hWall = (y: number, spans: [number, number][]): Segment[] =>
  spans.map(([a, b]) => ({ x1: a, y1: y, x2: b, y2: y }));

/** A vertical wall at `x`, broken into the given y-spans. */
const vWall = (x: number, spans: [number, number][]): Segment[] =>
  spans.map(([a, b]) => ({ x1: x, y1: a, x2: x, y2: b }));

/**
 * The apartment: 12 m × 9 m, five rooms off one long corridor.
 *
 *      y=9 ┌─────────────────┬──────────────────┐
 *          │   D  (study)    │   E  (bedroom)   │   north rooms
 *      y=5 ├───┤ ├───────────┴──┤ ├─────────────┤   corridor walls (doors = gaps)
 *          │            corridor  (11.6 m)      │
 *    y=3.8 ├─┤ ├──────┬───┤ ├───┬──────┤ ├──────┤
 *          │    A     │    B      │     C       │   south rooms
 *      y=0 └──────────┴───────────┴─────────────┘
 *          x=0        4           8            12
 *
 * Two deliberate design choices make this a *good* localization problem:
 *
 *  1. Rooms A and C, and their doorways, are exact mirror images about x = 6.
 *     A range-only filter started in the wrong one has no way to tell — the
 *     posterior stays bimodal until the robot reaches the corridor. This is the
 *     symmetry the global-localization widget exploits.
 *  2. The corridor is long, straight, and featureless along its length, so
 *     travelling down it grows the along-corridor variance while pinning the
 *     across-corridor variance. That anisotropy is the whole point of drawing
 *     covariance *ellipses* rather than circles.
 *
 * The north half is intentionally *not* symmetric (its doors sit at different
 * offsets, and the study has a nook), so a robot that looks north can break the
 * tie. Freestanding stubs — a kitchen counter in B, a closet in E — give the
 * likelihood field something to say inside the rooms.
 */
export const APARTMENT: World = {
  name: 'Apartment',
  bounds: { minX: 0, minY: 0, maxX: 12, maxY: 9 },
  walls: [
    // Exterior shell.
    ...hWall(0, [[0, 12]]),
    ...hWall(9, [[0, 12]]),
    ...vWall(0, [[0, 9]]),
    ...vWall(12, [[0, 9]]),

    // Corridor's south wall. Gaps: A's door [1.6, 2.5], B's [5.5, 6.5],
    // C's [9.5, 10.4] — the mirror image of A's about x = 6.
    ...hWall(3.8, [
      [0, 1.6],
      [2.5, 5.5],
      [6.5, 9.5],
      [10.4, 12],
    ]),
    // Corridor's north wall. Gaps at [2.8, 3.7] and [7.4, 8.3]: NOT mirrored,
    // which is what lets a north-facing observation resolve the A/C ambiguity.
    ...hWall(5, [
      [0, 2.8],
      [3.7, 7.4],
      [8.3, 12],
    ]),

    // Party walls between the three south rooms.
    ...vWall(4, [[0, 3.8]]),
    ...vWall(8, [[0, 3.8]]),
    // Party wall between the two north rooms.
    ...vWall(6, [[5, 9]]),

    // Kitchen counter in room B: a peninsula off the west wall.
    ...hWall(1.4, [[4, 6.2]]),
    // Reading nook in the study (room D).
    ...vWall(2, [[7.6, 9]]),
    // Closet in the bedroom (room E), open on its east side.
    ...vWall(9.6, [[7.4, 9]]),
    ...hWall(7.4, [[9.6, 10.9]]),
  ],
  landmarks: [
    { x: 0.35, y: 0.35, id: 0 },
    { x: 3.65, y: 0.35, id: 1 },
    { x: 11.65, y: 0.35, id: 2 },
    { x: 6.0, y: 8.65, id: 3 },
    { x: 0.35, y: 8.65, id: 4 },
    { x: 11.0, y: 4.4, id: 5 },
  ],
};

export interface Hallway1D {
  length: number;
  doors: number[];
  doorWidth: number;
}

/**
 * The 1-D pedagogical world: a corridor with three identical doors. The robot
 * can only sense "door" or "no door", so the belief after one sighting is
 * genuinely trimodal — the picture the whole Bayes-filter chapter is built on.
 */
export const HALLWAY_1D: Hallway1D = {
  length: 10,
  doors: [2.0, 4.5, 7.5],
  doorWidth: 0.6,
};

/** Is position `x` (metres along the hallway) inside a doorway? */
export function isDoorAt(x: number, hallway: Hallway1D = HALLWAY_1D): boolean {
  const half = hallway.doorWidth / 2;
  return hallway.doors.some((d) => Math.abs(x - d) <= half);
}

/**
 * p(z | x) for the binary door sensor: `pHit` is the chance of reporting
 * correctly, so a sensor with pHit = 0.9 still leaves 10% of the mass on the
 * wrong hypothesis — which is exactly why the belief never fully collapses
 * after a single reading.
 */
export function hallwayMeasurementLikelihood(
  x: number,
  sawDoor: boolean,
  pHit = 0.9,
  hallway: Hallway1D = HALLWAY_1D,
): number {
  const door = isDoorAt(x, hallway);
  return door === sawDoor ? pHit : 1 - pHit;
}

// ---------------------------------------------------------------------------
// Geometry primitives
// ---------------------------------------------------------------------------

const EPS = 1e-12;

/**
 * Distance from a point to a line **segment** (not the infinite line): project,
 * clamp the parameter to [0, 1], measure. Used by the likelihood field and by
 * collision clearance.
 */
export function distanceToSegment(px: number, py: number, s: Segment): number {
  const dx = s.x2 - s.x1;
  const dy = s.y2 - s.y1;
  const len2 = dx * dx + dy * dy;
  let t = 0;
  if (len2 > EPS) {
    t = ((px - s.x1) * dx + (py - s.y1) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
  }
  return Math.hypot(px - (s.x1 + t * dx), py - (s.y1 + t * dy));
}

/** Exact distance from a point to the nearest wall in the world. */
export function distanceToWalls(world: World, x: number, y: number): number {
  let best = Infinity;
  for (const s of world.walls) {
    const d = distanceToSegment(x, y, s);
    if (d < best) best = d;
  }
  return best;
}

/**
 * Ray/segment intersection.
 *
 * Ray:      p(t) = o + t·d,        t ≥ 0, d a unit vector
 * Segment:  q(u) = s₁ + u·(s₂−s₁), u ∈ [0, 1]
 *
 * Setting them equal gives a 2×2 solve whose determinant is the cross product
 * d × s. When that vanishes the ray is parallel to the wall and we report a
 * miss — grazing hits are not physically meaningful for a range finder.
 * Returns the distance to the closest hit, or `maxRange` if nothing is struck.
 */
export function rayCast(
  world: World,
  ox: number,
  oy: number,
  angle: number,
  maxRange: number,
): number {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  let best = maxRange;
  for (const s of world.walls) {
    const sx = s.x2 - s.x1;
    const sy = s.y2 - s.y1;
    const denom = dx * sy - dy * sx;
    if (Math.abs(denom) < EPS) continue;
    const ex = s.x1 - ox;
    const ey = s.y1 - oy;
    const t = (ex * sy - ey * sx) / denom;
    const u = (ex * dy - ey * dx) / denom;
    if (t >= 0 && t < best && u >= 0 && u <= 1) best = t;
  }
  return best;
}

/** Do two segments cross? Standard orientation test, with collinear overlap ignored. */
export function segmentsIntersect(a: Segment, b: Segment): boolean {
  const r1 = a.x2 - a.x1;
  const r2 = a.y2 - a.y1;
  const s1 = b.x2 - b.x1;
  const s2 = b.y2 - b.y1;
  const denom = r1 * s2 - r2 * s1;
  if (Math.abs(denom) < EPS) return false;
  const ex = b.x1 - a.x1;
  const ey = b.y1 - a.y1;
  const t = (ex * s2 - ey * s1) / denom;
  const u = (ex * r2 - ey * r1) / denom;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

/** Inside the bounds and at least `clearance` away from every wall. */
export function isFree(world: World, x: number, y: number, clearance = 0): boolean {
  const b = world.bounds;
  if (x < b.minX || x > b.maxX || y < b.minY || y > b.maxY) return false;
  return distanceToWalls(world, x, y) > clearance;
}

/** Would travelling in a straight line from `from` to `to` cross a wall (or leave the map)? */
export function collides(world: World, from: Point2, to: Point2): boolean {
  const b = world.bounds;
  if (to.x < b.minX || to.x > b.maxX || to.y < b.minY || to.y > b.maxY) return true;
  const path: Segment = { x1: from.x, y1: from.y, x2: to.x, y2: to.y };
  return world.walls.some((w) => segmentsIntersect(path, w));
}

// ---------------------------------------------------------------------------
// Range sensing
// ---------------------------------------------------------------------------

export interface ScanParams {
  nBeams: number;
  /** Total angular field of view in radians. 2π means a full 360° scan. */
  fov: number;
  maxRange: number;
  /** Std-dev of the Gaussian hit noise, in metres. */
  sigma: number;
  /** Mixture weights of the beam model; default is pure-hit (zHit = 1). */
  zHit?: number;
  zRand?: number;
  zMax?: number;
}

/**
 * Beam bearings **relative to the robot's heading**, evenly spread over the
 * field of view. A full-circle scan does not repeat the first beam at the end,
 * so it steps by fov/n rather than fov/(n−1).
 */
export function beamAngles(params: Pick<ScanParams, 'nBeams' | 'fov'>): number[] {
  const { nBeams, fov } = params;
  if (nBeams <= 1) return [0];
  const wraps = Math.abs(fov - 2 * Math.PI) < 1e-9;
  const step = wraps ? fov / nBeams : fov / (nBeams - 1);
  return Array.from({ length: nBeams }, (_, k) => -fov / 2 + k * step);
}

/**
 * Simulate one LiDAR sweep, corrupting each true range with the beam model of
 * Thrun et al., Chapter 6.3: mostly Gaussian noise about the true range, plus a
 * `zRand` chance of a uniformly random reading and a `zMax` chance of a
 * max-range dropout (a beam that hit glass, or nothing at all).
 */
export function simulateScan(
  world: World,
  pose: Pose2,
  params: ScanParams,
  rng: Rng,
): number[] {
  const { maxRange, sigma } = params;
  const zRand = params.zRand ?? 0;
  const zMax = params.zMax ?? 0;
  return beamAngles(params).map((rel) => {
    const zStar = rayCast(world, pose.x, pose.y, pose.theta + rel, maxRange);
    const u = rng.next();
    if (u < zMax) return maxRange;
    if (u < zMax + zRand) return rng.uniform(0, maxRange);
    return Math.max(0, Math.min(maxRange, zStar + rng.normal(0, sigma)));
  });
}

/** The noise-free scan — the `z*` the sensor models compare against. */
export function trueScan(world: World, pose: Pose2, params: ScanParams): number[] {
  return beamAngles(params).map((rel) =>
    rayCast(world, pose.x, pose.y, pose.theta + rel, params.maxRange),
  );
}

// ---------------------------------------------------------------------------
// Robot
// ---------------------------------------------------------------------------

export interface Robot {
  pose: Pose2;
}

/**
 * Noise-free differential-drive integration over Δt at constant (v, ω).
 *
 * The exact solution is an arc of radius r = v/ω:
 *
 *   x' = x − r sin θ + r sin(θ + ωΔt)
 *   y' = y + r cos θ − r cos(θ + ωΔt)
 *
 * (Thrun et al., eq. 5.9, with the noise terms dropped.) As ω → 0 the radius
 * blows up while the arc flattens; rather than trust the cancellation we switch
 * to the straight-line limit, which is what the Rust version does too.
 */
export function diffDriveStep(pose: Pose2, v: number, omega: number, dt: number): Pose2 {
  const { x, y, theta } = pose;
  if (Math.abs(omega) < 1e-9) {
    return {
      x: x + v * Math.cos(theta) * dt,
      y: y + v * Math.sin(theta) * dt,
      theta: normalizeAngle(theta + omega * dt),
    };
  }
  const r = v / omega;
  const nt = theta + omega * dt;
  return {
    x: x - r * Math.sin(theta) + r * Math.sin(nt),
    y: y + r * Math.cos(theta) - r * Math.cos(nt),
    theta: normalizeAngle(nt),
  };
}
