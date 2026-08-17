/**
 * Global relocalisation inside the robot's own map.
 *
 * When the kidnap detector fires, the pose belief is not merely wrong — it is
 * wrong in a way no Gaussian correction can repair, because the truth is
 * nowhere near the ellipse. Recovery therefore leaves the Gaussian world
 * entirely and runs Chapter 12's Monte Carlo localisation over the **frozen
 * map**: scatter particles across every cell the map already calls free, score
 * them against the live scan, resample, and hand the answer back once the cloud
 * has collapsed to a single mode.
 *
 * Everything here except the scorer is the library's existing particle
 * machinery — `ParticleFilter`, the low-variance sampler, `dominantCluster`,
 * `countClusters`. What is new is only that the likelihood field comes from the
 * ESDF of a map Rusty built, so the quality of the recovery is bounded by the
 * quality of the mapping that preceded it.
 *
 * Rust counterpart: `crates/capstone/src/tasks/supervisor.rs` (`relocalize_global`).
 */

import { ParticleFilter, type Particle } from '../filters/pf';
import { countClusters, dominantCluster } from '../localize/augmented-mcl';
import { sampleMotionModelOdometry, type OdomAlphas, type OdomDelta } from '../models/motion';
import type { Pose2 } from '../geom/se2';
import type { OccupancyGrid } from '../mapping/occgrid';
import type { Rng } from '../prob/rng';
import { FREE } from './astar';
import { esdfAt, type Esdf } from './esdf';

export interface MapLikelihoodParams {
  /** σ_hit of the likelihood field, metres. */
  sigmaHit: number;
  zHit: number;
  zRand: number;
  maxRange: number;
  /** Score every n-th beam. 360 beams do not carry 360 independent bits. */
  stride: number;
  /**
   * Tempering exponent on the summed log-likelihood.
   *
   * Chapter 10's standing complaint, made operational. Treating each beam as
   * independent evidence makes the product of thirty-six near-Gaussian factors,
   * whose dynamic range is `exp(130)`; one particle then takes every unit of
   * weight, the low-variance sampler faithfully clones it 1200 times, and the
   * filter reports total confidence in whichever hypothesis happened to be
   * least bad on the very first scan. Raising the likelihood to a power < 1 is
   * the standard, honest repair: it says "these beams are worth about
   * `temper · n` independent measurements", which is true.
   */
  temper: number;
}

export const DEFAULT_MAP_LIKELIHOOD: MapLikelihoodParams = {
  sigmaHit: 0.3,
  zHit: 0.86,
  zRand: 0.14,
  maxRange: 6,
  stride: 3,
  temper: 0.25,
};

/**
 * `likelihood_field_range_finder_model` (Thrun et al., Table 6.3) evaluated on
 * the **map's** distance field rather than a ground-truth one.
 *
 * The mixture floor `zRand / zMax` is what keeps a single unexplained beam from
 * zeroing a pose that is otherwise perfect — the same robustness argument
 * Chapter 10 makes, and the reason relocalisation still works while a person is
 * standing in front of the LiDAR.
 */
export function logMapLikelihood(
  ranges: readonly number[],
  angles: readonly number[],
  pose: Pose2,
  esdf: Esdf,
  params: MapLikelihoodParams = DEFAULT_MAP_LIKELIHOOD,
): number {
  const { sigmaHit, zHit, zRand, maxRange, stride } = params;
  const norm = 1 / (Math.sqrt(2 * Math.PI) * sigmaHit);
  const floor = zRand / maxRange;
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  let log = 0;
  for (let i = 0; i < ranges.length; i += stride) {
    const z = ranges[i];
    if (z >= maxRange) continue; // "I saw nothing" says nothing about the map
    const a = angles[i];
    const dx = z * Math.cos(a);
    const dy = z * Math.sin(a);
    const px = pose.x + c * dx - s * dy;
    const py = pose.y + s * dx + c * dy;
    const d = esdfAt(esdf, px, py);
    const p = zHit * norm * Math.exp(-(d * d) / (2 * sigmaHit * sigmaHit)) + floor;
    log += Math.log(Math.max(p, 1e-12));
  }
  return params.temper * log;
}

export interface RelocalizeConfig {
  particles: number;
  alphas: OdomAlphas;
  /** Declare victory once the cloud's position spread is under this, in metres. */
  convergeSigma: number;
  /** …and it holds a single mode… */
  maxClusters: number;
  /** …for this many consecutive scans. */
  patience: number;
  /**
   * Never declare victory before this many scans, however tight the cloud gets.
   *
   * A stationary robot in a rectangular room can drive the particle cloud into
   * one confident mode in half a second — in the *wrong* room, because from one
   * viewpoint the two rooms are the same measurement. What separates them is
   * motion, so the recovery is required to drive for a while before it is
   * allowed to believe itself. This is the cheapest possible taste of
   * belief-space planning: act to disambiguate, then commit.
   */
  minSteps: number;
  /** Resample once the effective sample size falls below this fraction of M. */
  essFrac: number;
  likelihood: MapLikelihoodParams;
}

export const DEFAULT_RELOCALIZE: RelocalizeConfig = {
  particles: 6000,
  alphas: [0.02, 0.02, 0.04, 0.02],
  convergeSigma: 0.28,
  maxClusters: 1,
  patience: 4,
  minSteps: 55,
  essFrac: 0.5,
  likelihood: DEFAULT_MAP_LIKELIHOOD,
};

export interface RelocalizeStatus {
  converged: boolean;
  pose: Pose2;
  /** Position spread of the cloud, metres — the number the panel plots. */
  spread: number;
  clusters: number;
  ess: number;
  particles: readonly Particle[];
}

/**
 * The recovery behaviour behind `Mode::Relocalize`.
 *
 * Note what it does *not* do: it never touches the map. Mapping is suspended
 * for the whole of the recovery, because integrating scans at a pose you have
 * just admitted is wrong is the fastest way to destroy a map that was fine.
 */
export class GlobalRelocalizer {
  readonly cfg: RelocalizeConfig;
  private pf: ParticleFilter | null = null;
  private streak = 0;
  steps = 0;

  constructor(cfg: Partial<RelocalizeConfig> = {}) {
    this.cfg = { ...DEFAULT_RELOCALIZE, ...cfg };
  }

  get particles(): readonly Particle[] {
    return this.pf?.particles ?? [];
  }

  /**
   * Scatter over the free space **of the map**, not of the world.
   *
   * A robot that has explored half an apartment can only relocalise into the
   * half it has seen. That is not a bug to be patched: it is the honest
   * statement of what the robot knows, and it is why the kidnap demo teleports
   * Rusty into a room it has already mapped.
   */
  scatter(grid: OccupancyGrid, cls: Uint8Array, esdf: Esdf, rng: Rng, clearance = 0.25): void {
    const free: number[] = [];
    for (let k = 0; k < cls.length; k++) {
      if (cls[k] !== FREE) continue;
      const i = k % grid.width;
      const j = (k - i) / grid.width;
      const [x, y] = grid.cellCenter(i, j);
      if (esdfAt(esdf, x, y) < clearance) continue;
      free.push(k);
    }

    const particles: Particle[] = [];
    const n = this.cfg.particles;
    if (free.length === 0) {
      this.pf = new ParticleFilter([]);
      return;
    }
    for (let m = 0; m < n; m++) {
      const k = free[Math.floor(rng.next() * free.length)];
      const i = k % grid.width;
      const j = (k - i) / grid.width;
      const [x, y] = grid.cellCenter(i, j);
      particles.push({
        state: {
          x: x + rng.uniform(-0.5, 0.5) * grid.cellSize,
          y: y + rng.uniform(-0.5, 0.5) * grid.cellSize,
          theta: rng.uniform(-Math.PI, Math.PI),
        },
        weight: 1 / n,
      });
    }
    this.pf = new ParticleFilter(particles);
    this.streak = 0;
    this.steps = 0;
  }

  /** One MCL cycle: predict with odometry, weight with the scan, resample. */
  update(
    odom: OdomDelta,
    ranges: readonly number[],
    angles: readonly number[],
    esdf: Esdf,
    rng: Rng,
  ): RelocalizeStatus {
    const pf = this.pf;
    if (!pf || pf.size === 0) {
      return { converged: false, pose: { x: 0, y: 0, theta: 0 }, spread: Infinity, clusters: 0, ess: 0, particles: [] };
    }
    this.steps++;

    pf.predict((state) => sampleMotionModelOdometry(odom, state, this.cfg.alphas, rng));
    pf.correctLog((state) => logMapLikelihood(ranges, angles, state, esdf, this.cfg.likelihood));
    // Resample only when the population has actually degenerated. Resampling a
    // healthy cloud every step throws away the diversity that is the only thing
    // keeping the second hypothesis alive long enough to be tested.
    if (pf.effectiveSampleSize() < (this.cfg.essFrac ?? 0.5) * pf.size) pf.resample(rng, 'lowVariance');

    const cov = pf.positionCovariance();
    const spread = Math.sqrt(Math.max(0, cov[0][0] + cov[1][1]));
    const clusters = countClusters(pf.particles, 0.6);
    const cluster = dominantCluster(pf.particles, 0.7);
    const ess = pf.effectiveSampleSize() / pf.size;

    const tight = spread < this.cfg.convergeSigma && clusters <= this.cfg.maxClusters && cluster.mass > 0.8;
    this.streak = tight ? this.streak + 1 : 0;

    return {
      converged: this.streak >= this.cfg.patience && this.steps >= this.cfg.minSteps,
      pose: cluster.pose,
      spread,
      clusters,
      ess,
      particles: pf.particles,
    };
  }
}
