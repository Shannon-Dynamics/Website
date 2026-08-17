/**
 * Range-finder and landmark measurement models — Thrun et al., Ch. 6.
 *
 * Three models, in increasing order of pragmatism:
 *
 *   1. the *beam* model — physically motivated, ray-casts per beam per
 *      hypothesis, and has the notorious jagged likelihood surface;
 *   2. the *likelihood field* — precompute distance-to-nearest-obstacle once,
 *      then every beam is a table lookup and the surface is smooth;
 *   3. the *landmark* model — when the front end already gives you (r, φ, s).
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import { normalCdf, prob } from '../prob/gaussian';
import {
  distanceToSegment,
  rayCast,
  type Bounds,
  type Landmark,
  type World,
} from '../sim/world';

// ---------------------------------------------------------------------------
// Beam model — Table 6.1
// ---------------------------------------------------------------------------

export interface BeamParams {
  /** Mixture weights; the four should sum to 1. */
  zHit: number;
  zShort: number;
  zMax: number;
  zRand: number;
  /** Std-dev of the hit peak, metres. */
  sigmaHit: number;
  /** Rate of the unexpected-obstacle exponential, 1/metres. */
  lambdaShort: number;
  maxRange: number;
}

export const DEFAULT_BEAM_PARAMS: BeamParams = {
  zHit: 0.74,
  zShort: 0.1,
  zMax: 0.06,
  zRand: 0.1,
  sigmaHit: 0.12,
  lambdaShort: 1.0,
  maxRange: 8,
};

/** A reading is "max range" if it is within this of the sensor's limit. */
const MAX_EPS = 1e-9;

/**
 * The four-way mixture p(z | x, m) for a single beam — Thrun et al.,
 * **Table 6.1**, eq. 6.4–6.12. Exported on its own because the shape of this
 * curve *is* the lesson: a Gaussian peak at the true range, a decaying ramp in
 * front of it for people and chair legs, a spike at z_max for beams that hit
 * nothing, and a uniform floor for everything unmodelled.
 *
 * Both p_hit and p_short carry a normaliser η, because both are truncated
 * distributions: p_hit is only defined on [0, z_max] and p_short only on
 * [0, z*]. Dropping those (a common bug) makes the model prefer short true
 * ranges for no physical reason.
 */
export function beamLikelihood(zk: number, zStar: number, params: BeamParams): number {
  const { zHit, zShort, zMax, zRand, sigmaHit, lambdaShort, maxRange } = params;

  let pHit = 0;
  if (zk >= 0 && zk <= maxRange) {
    // η = 1 / ∫₀^{zmax} N(z; z*, σ²) dz
    const mass = normalCdf(maxRange, zStar, sigmaHit) - normalCdf(0, zStar, sigmaHit);
    if (mass > 1e-12) {
      pHit = prob(zk - zStar, sigmaHit * sigmaHit) / mass;
    }
  }

  let pShort = 0;
  if (zk >= 0 && zk <= zStar && zStar > 0) {
    const eta = 1 / (1 - Math.exp(-lambdaShort * zStar));
    pShort = eta * lambdaShort * Math.exp(-lambdaShort * zk);
  }

  // A point mass, not a density: it integrates to 1 all by itself.
  const pMax = zk >= maxRange - MAX_EPS ? 1 : 0;

  const pRand = zk >= 0 && zk < maxRange ? 1 / maxRange : 0;

  return zHit * pHit + zShort * pShort + zMax * pMax + zRand * pRand;
}

/**
 * `beam_range_finder_model` — Thrun et al., **Table 6.1**. The product over
 * beams, which assumes beam independence: false, wildly so, and the reason
 * real implementations sub-sample the scan rather than use all 1080 rays.
 *
 * `angles` are bearings relative to the robot heading, as produced by
 * `beamAngles`.
 */
export function beamRangeFinderModel(
  z: number[],
  pose: Pose2,
  world: World,
  params: BeamParams,
  angles: number[],
): number {
  let q = 1;
  for (let k = 0; k < z.length; k++) {
    const zStar = rayCast(world, pose.x, pose.y, pose.theta + angles[k], params.maxRange);
    q *= beamLikelihood(z[k], zStar, params);
  }
  return q;
}

/** Log of {@link beamRangeFinderModel} — what you actually want with 60+ beams. */
export function logBeamRangeFinderModel(
  z: number[],
  pose: Pose2,
  world: World,
  params: BeamParams,
  angles: number[],
): number {
  let q = 0;
  for (let k = 0; k < z.length; k++) {
    const zStar = rayCast(world, pose.x, pose.y, pose.theta + angles[k], params.maxRange);
    q += Math.log(Math.max(beamLikelihood(z[k], zStar, params), 1e-300));
  }
  return q;
}

// ---------------------------------------------------------------------------
// Likelihood field — Table 6.3
// ---------------------------------------------------------------------------

/**
 * Distance-to-nearest-obstacle grid.
 *
 * Built in two passes of a chamfer (brushfire) transform: seed every cell a
 * wall passes through with its exact distance to that wall, then sweep
 * forwards propagating `d(neighbour) + step` and backwards doing the same. Two
 * sweeps suffice because any shortest path through the 8-connected grid is
 * monotone in one of the two sweep orders.
 *
 * The result is O(cells) to build and O(1) to query — the whole reason the
 * likelihood field beats ray casting when you have 10 000 particles.
 */
export class LikelihoodField {
  readonly cellSize: number;
  readonly bounds: Bounds;
  readonly nx: number;
  readonly ny: number;
  readonly data: Float64Array;

  constructor(world: World, cellSize = 0.05, bounds?: Bounds) {
    this.cellSize = cellSize;
    this.bounds = bounds ?? world.bounds;
    const { minX, minY, maxX, maxY } = this.bounds;
    this.nx = Math.max(1, Math.ceil((maxX - minX) / cellSize));
    this.ny = Math.max(1, Math.ceil((maxY - minY) / cellSize));
    this.data = new Float64Array(this.nx * this.ny).fill(Infinity);
    this.seed(world);
    this.chamfer();
  }

  static fromWorld(world: World, cellSize = 0.05, bounds?: Bounds): LikelihoodField {
    return new LikelihoodField(world, cellSize, bounds);
  }

  private index(i: number, j: number): number {
    return j * this.nx + i;
  }

  /** World coordinates of a cell's centre. */
  cellCenter(i: number, j: number): [number, number] {
    return [
      this.bounds.minX + (i + 0.5) * this.cellSize,
      this.bounds.minY + (j + 0.5) * this.cellSize,
    ];
  }

  /** Walk each wall, giving every cell it crosses its exact distance to that wall. */
  private seed(world: World): void {
    const cs = this.cellSize;
    for (const s of world.walls) {
      const len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
      const steps = Math.max(1, Math.ceil((len / cs) * 3));
      for (let n = 0; n <= steps; n++) {
        const t = n / steps;
        const px = s.x1 + t * (s.x2 - s.x1);
        const py = s.y1 + t * (s.y2 - s.y1);
        const i = Math.floor((px - this.bounds.minX) / cs);
        const j = Math.floor((py - this.bounds.minY) / cs);
        // Seed the cell and its 8 neighbours: a wall clipping a corner still
        // makes the adjacent cell centres genuinely close to it.
        for (let dj = -1; dj <= 1; dj++) {
          for (let di = -1; di <= 1; di++) {
            const ci = i + di;
            const cj = j + dj;
            if (ci < 0 || cj < 0 || ci >= this.nx || cj >= this.ny) continue;
            const [cx, cy] = this.cellCenter(ci, cj);
            const d = distanceToSegment(cx, cy, s);
            const idx = this.index(ci, cj);
            if (d < this.data[idx]) this.data[idx] = d;
          }
        }
      }
    }
  }

  /** Forward then backward 8-neighbour sweep. */
  private chamfer(): void {
    const d1 = this.cellSize;
    const d2 = this.cellSize * Math.SQRT2;
    const { nx, ny, data } = this;

    const relax = (idx: number, from: number, cost: number) => {
      const v = data[from] + cost;
      if (v < data[idx]) data[idx] = v;
    };

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = this.index(i, j);
        if (j > 0) {
          relax(idx, this.index(i, j - 1), d1);
          if (i > 0) relax(idx, this.index(i - 1, j - 1), d2);
          if (i < nx - 1) relax(idx, this.index(i + 1, j - 1), d2);
        }
        if (i > 0) relax(idx, this.index(i - 1, j), d1);
      }
    }
    for (let j = ny - 1; j >= 0; j--) {
      for (let i = nx - 1; i >= 0; i--) {
        const idx = this.index(i, j);
        if (j < ny - 1) {
          relax(idx, this.index(i, j + 1), d1);
          if (i > 0) relax(idx, this.index(i - 1, j + 1), d2);
          if (i < nx - 1) relax(idx, this.index(i + 1, j + 1), d2);
        }
        if (i < nx - 1) relax(idx, this.index(i + 1, j), d1);
      }
    }
  }

  /**
   * Distance from an arbitrary world point to the nearest obstacle.
   *
   * Queries outside the grid are clamped to the border cell and charged the
   * extra Euclidean distance, so the field keeps growing outwards instead of
   * flat-lining at the map edge — a beam that lands off the map should look
   * *unlikely*, not merely uninformative.
   */
  distanceAt(x: number, y: number): number {
    const cs = this.cellSize;
    const { minX, minY } = this.bounds;
    const fi = (x - minX) / cs - 0.5;
    const fj = (y - minY) / cs - 0.5;
    const i = Math.max(0, Math.min(this.nx - 1, Math.round(fi)));
    const j = Math.max(0, Math.min(this.ny - 1, Math.round(fj)));
    const [cx, cy] = this.cellCenter(i, j);
    const outside = Math.hypot(x - cx, y - cy) - cs * 0.71;
    return this.data[this.index(i, j)] + Math.max(0, outside);
  }

  /** Row-major (j·nx + i) copy for rendering the field as an image. */
  getDistanceArray(): Float64Array {
    return this.data.slice();
  }
}

/**
 * `likelihood_field_range_finder_model` — Thrun et al., **Table 6.3**.
 *
 * Each beam is projected to its endpoint in world coordinates, and scored by
 * how far that endpoint is from *any* obstacle — no ray casting, no
 * correspondence. Max-range readings are skipped entirely: "I saw nothing" says
 * nothing about where the nearest wall is.
 *
 * `sensorOffset` is the sensor's pose in the robot frame (Thrun's (x_k, y_k)).
 */
export function likelihoodFieldRangeFinderModel(
  z: number[],
  pose: Pose2,
  field: LikelihoodField,
  params: BeamParams,
  angles: number[],
  sensorOffset: [number, number] = [0, 0],
): number {
  return Math.exp(
    logLikelihoodFieldRangeFinderModel(z, pose, field, params, angles, sensorOffset),
  );
}

/** Log-space twin of {@link likelihoodFieldRangeFinderModel}. */
export function logLikelihoodFieldRangeFinderModel(
  z: number[],
  pose: Pose2,
  field: LikelihoodField,
  params: BeamParams,
  angles: number[],
  sensorOffset: [number, number] = [0, 0],
): number {
  const { zHit, zRand, sigmaHit, maxRange } = params;
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const sx = pose.x + c * sensorOffset[0] - s * sensorOffset[1];
  const sy = pose.y + s * sensorOffset[0] + c * sensorOffset[1];

  let q = 0;
  for (let k = 0; k < z.length; k++) {
    if (z[k] >= maxRange - MAX_EPS) continue;
    const a = pose.theta + angles[k];
    const ex = sx + z[k] * Math.cos(a);
    const ey = sy + z[k] * Math.sin(a);
    const dist = field.distanceAt(ex, ey);
    const p = zHit * prob(dist, sigmaHit * sigmaHit) + zRand / maxRange;
    q += Math.log(Math.max(p, 1e-300));
  }
  return q;
}

// ---------------------------------------------------------------------------
// Landmark model — Table 6.4
// ---------------------------------------------------------------------------

/** A range–bearing–signature feature, Thrun's f = (r, φ, s). */
export interface RangeBearingFeature {
  r: number;
  phi: number;
  s?: number;
}

export interface LandmarkSigmas {
  r: number;
  phi: number;
  /** Signature noise; omit to ignore the signature term entirely. */
  s?: number;
}

/**
 * `landmark_model_known_correspondence` — Thrun et al., **Table 6.4**.
 *
 * With the correspondence given, the measurement is just a noisy polar
 * observation of a known point, and the density factorises into range, bearing,
 * and signature terms. The bearing residual is wrapped: a predicted 179° and an
 * observed −179° are 2° apart, not 358°.
 */
export function landmarkModelKnownCorrespondence(
  feature: RangeBearingFeature,
  landmark: Landmark,
  pose: Pose2,
  sigmas: LandmarkSigmas,
): number {
  const dx = landmark.x - pose.x;
  const dy = landmark.y - pose.y;
  const rHat = Math.hypot(dx, dy);
  const phiHat = normalizeAngle(Math.atan2(dy, dx) - pose.theta);

  let p =
    prob(feature.r - rHat, sigmas.r * sigmas.r) *
    prob(normalizeAngle(feature.phi - phiHat), sigmas.phi * sigmas.phi);

  if (sigmas.s !== undefined && feature.s !== undefined) {
    const sHat = landmark.signature ?? landmark.id;
    p *= prob(feature.s - sHat, sigmas.s * sigmas.s);
  }
  return p;
}

/** Noise-free (r, φ, s) for a landmark seen from `pose` — the h(x) of the EKF. */
export function landmarkObservation(landmark: Landmark, pose: Pose2): RangeBearingFeature {
  const dx = landmark.x - pose.x;
  const dy = landmark.y - pose.y;
  return {
    r: Math.hypot(dx, dy),
    phi: normalizeAngle(Math.atan2(dy, dx) - pose.theta),
    s: landmark.signature ?? landmark.id,
  };
}
