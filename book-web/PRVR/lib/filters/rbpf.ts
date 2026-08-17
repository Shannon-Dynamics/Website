/**
 * Grid-based Rao-Blackwellized particle filtering — the *gmapping* recipe.
 *
 * Grisetti, Stachniss & Burgard, "Improved Techniques for Grid Mapping with
 * Rao-Blackwellized Particle Filters", IEEE T-RO 23(1), 2007.
 *
 * The factorization theorem says a particle should carry a *path*, not a pose,
 * and that conditioned on that path the map is no longer a random variable
 * worth sampling — it can be computed. So every particle here owns an
 * occupancy grid (Chapter 13's `OccupancyGrid`, log odds and all) which is
 * updated deterministically from the particle's own trajectory. Nothing about
 * the map is sampled; only the path is.
 *
 * Three pieces of the 2007 paper are implemented literally:
 *
 *   1. **scan matching against the particle's own map** to find the peak of the
 *      observation likelihood (`scanMatchGrid`);
 *   2. **the improved proposal**: K poses drawn around that peak, weighted by
 *      observation × motion, fitted with a Gaussian, sampled from — and the
 *      normalizer η becomes the weight increment (`GridRbpf.step`);
 *   3. **selective resampling** on N_eff, which is the difference between a
 *      filter that still has hypotheses at loop closure and one that does not.
 *
 * Maps are shared **copy-on-write** ({@link CowGrid}), the idiomatic-Rust
 * replacement for FastSLAM's shared balanced trees: cloning a particle at
 * resampling time is O(1) until one of the copies actually writes a cell.
 */

import { angleDiff, normalizeAngle, type Pose2 } from '../geom/se2';
import { logOddsToProb, OccupancyGrid, type InverseModelParams } from '../mapping/occgrid';
import {
  applyOdom,
  motionModelOdometry,
  sampleMotionModelOdometry,
  type OdomAlphas,
  type OdomDelta,
} from '../models/motion';
import { logSumExp } from '../prob/gaussian';
import { cholesky, type Mat } from '../prob/linalg';
import type { Rng } from '../prob/rng';
import { lowVarianceResample, type Particle } from './pf';

// ---------------------------------------------------------------------------
// Copy-on-write maps
// ---------------------------------------------------------------------------

interface GridBox {
  grid: OccupancyGrid;
  /** How many `CowGrid` handles point at this cell array. */
  refs: number;
}

/** Deep copy of a grid — the thing copy-on-write exists to postpone. */
export function cloneGrid(src: OccupancyGrid): OccupancyGrid {
  const out = new OccupancyGrid({
    width: src.width,
    height: src.height,
    cellSize: src.cellSize,
    origin: { ...src.origin },
    prior: logOddsToProb(src.l0),
  });
  out.logOdds.set(src.logOdds);
  return out;
}

let cowCopies = 0;
let cowShares = 0;

/** How many real map copies have happened, and how many were deferred. */
export const cowStats = () => ({ copies: cowCopies, shares: cowShares });
export const resetCowStats = () => {
  cowCopies = 0;
  cowShares = 0;
};

/**
 * A handle to a shared occupancy grid, copied only on first write.
 *
 * This is `Arc<OccGrid>` + `Arc::make_mut` in TypeScript. Resampling clones
 * particles by the dozen and most clones never diverge before they are killed
 * again, so paying for the cells at clone time is pure waste.
 */
export class CowGrid {
  private box: GridBox;

  private constructor(box: GridBox) {
    this.box = box;
  }

  static of(grid: OccupancyGrid): CowGrid {
    return new CowGrid({ grid, refs: 1 });
  }

  /** O(1) clone: both handles read the same cells until one of them writes. */
  share(): CowGrid {
    this.box.refs += 1;
    cowShares += 1;
    return new CowGrid(this.box);
  }

  /** Read-only view. Never mutate what this returns. */
  get read(): OccupancyGrid {
    return this.box.grid;
  }

  get shared(): boolean {
    return this.box.refs > 1;
  }

  /** Writable view; performs the deferred copy if anyone else is still reading. */
  write(): OccupancyGrid {
    if (this.box.refs > 1) {
      this.box.refs -= 1;
      this.box = { grid: cloneGrid(this.box.grid), refs: 1 };
      cowCopies += 1;
    }
    return this.box.grid;
  }

  /** Drop this handle's claim — call when a particle dies. */
  release(): void {
    this.box.refs -= 1;
  }
}

// ---------------------------------------------------------------------------
// Scoring a scan against a particle's own map
// ---------------------------------------------------------------------------

export interface MapScoreParams {
  /** Mixture weight on "this beam ended on something the map calls occupied". */
  zHit: number;
  /** Floor, so one contradicted beam cannot annihilate a whole hypothesis. */
  zRand: number;
  maxRange: number;
  /**
   * Radius, in cells, of an optional max-filter applied before scoring. Zero
   * means bilinear interpolation of the occupancy field instead, which is
   * sharper: a max-filter of radius 1 on 30 cm cells makes a 30 cm pose error
   * free, and a scan matcher cannot correct an error it cannot feel.
   */
  blur: number;
}

export const DEFAULT_MAP_SCORE: MapScoreParams = {
  zHit: 0.85,
  zRand: 0.08,
  maxRange: 6,
  blur: 0,
};

/**
 * Occupancy probability at an arbitrary world point.
 *
 * `blur = 0` bilinearly interpolates the four surrounding cell centres, which
 * is what gives the scan matcher a continuous surface with sub-cell gradients.
 * `blur > 0` max-filters a (2b+1)² window instead — cheaper, blunter, and
 * occasionally useful to widen the basin of attraction on a very sparse map.
 */
export function occupancyAtPoint(map: OccupancyGrid, x: number, y: number, blur: number): number {
  if (blur > 0) {
    const [ci, cj] = map.worldToCell(x, y);
    let best = 0;
    for (let dj = -blur; dj <= blur; dj++) {
      for (let di = -blur; di <= blur; di++) {
        const p = map.probAt(ci + di, cj + dj);
        if (p > best) best = p;
      }
    }
    return best;
  }

  const fi = (x - map.origin.x) / map.cellSize - 0.5;
  const fj = (y - map.origin.y) / map.cellSize - 0.5;
  const i0 = Math.floor(fi);
  const j0 = Math.floor(fj);
  const tx = fi - i0;
  const ty = fj - j0;
  return (
    (1 - tx) * (1 - ty) * map.probAt(i0, j0) +
    tx * (1 - ty) * map.probAt(i0 + 1, j0) +
    (1 - tx) * ty * map.probAt(i0, j0 + 1) +
    tx * ty * map.probAt(i0 + 1, j0 + 1)
  );
}

/**
 * log p(z_t | m^[i], x_t) for one scan against one particle's map.
 *
 * Endpoint scoring — "map matching" in Thrun et al. §6.5, and what gmapping's
 * `likelihood` does in spirit: project every beam to its endpoint and ask the
 * map how occupied that spot is. Unexplored cells sit at p = 0.5 and therefore
 * contribute a constant, which is exactly right: a hypothesis cannot be
 * rewarded or punished by territory it has never seen.
 */
export function logMapMatchScore(
  map: OccupancyGrid,
  pose: Pose2,
  ranges: number[],
  angles: number[],
  params: MapScoreParams = DEFAULT_MAP_SCORE,
): number {
  const { zHit, zRand, maxRange, blur } = params;
  let q = 0;
  for (let k = 0; k < ranges.length; k++) {
    if (ranges[k] >= maxRange - 1e-9) continue; // "I saw nothing" says nothing
    const a = pose.theta + angles[k];
    const ex = pose.x + ranges[k] * Math.cos(a);
    const ey = pose.y + ranges[k] * Math.sin(a);
    q += Math.log(zHit * occupancyAtPoint(map, ex, ey, blur) + zRand);
  }
  return q;
}

export interface ScanMatchResult {
  pose: Pose2;
  score: number;
  /** Score at the seed pose, so callers can decide whether to trust the match. */
  seedScore: number;
  evaluations: number;
}

export interface ScanMatchOptions {
  linearStep: number;
  angularStep: number;
  /** Stop once the linear step has shrunk below this. */
  tolerance: number;
  maxIterations: number;
}

export const DEFAULT_SCAN_MATCH: ScanMatchOptions = {
  linearStep: 0.18,
  angularStep: 0.06,
  tolerance: 0.02,
  maxIterations: 40,
};

/**
 * Greedy hill climb on the map-match score — gmapping's `scanMatch`.
 *
 * Six neighbours (±x, ±y, ±θ), take the best if it improves, otherwise halve
 * the step. It is not a real optimizer and does not pretend to be: its job is
 * to find the *mode* of the observation likelihood so the proposal can be
 * centred there, and it fails gracefully (returns the seed) when the map is
 * still unexplored and the score surface is flat.
 */
export function scanMatchGrid(
  map: OccupancyGrid,
  seed: Pose2,
  ranges: number[],
  angles: number[],
  params: MapScoreParams = DEFAULT_MAP_SCORE,
  opts: ScanMatchOptions = DEFAULT_SCAN_MATCH,
): ScanMatchResult {
  const score = (p: Pose2) => logMapMatchScore(map, p, ranges, angles, params);
  const seedScore = score(seed);

  let best: Pose2 = { ...seed };
  let bestScore = seedScore;
  let lin = opts.linearStep;
  let ang = opts.angularStep;
  let evaluations = 1;

  for (let iter = 0; iter < opts.maxIterations && lin > opts.tolerance; iter++) {
    const candidates: Pose2[] = [
      { ...best, x: best.x + lin },
      { ...best, x: best.x - lin },
      { ...best, y: best.y + lin },
      { ...best, y: best.y - lin },
      { ...best, theta: normalizeAngle(best.theta + ang) },
      { ...best, theta: normalizeAngle(best.theta - ang) },
    ];
    let improved = false;
    for (const c of candidates) {
      const s = score(c);
      evaluations += 1;
      if (s > bestScore) {
        bestScore = s;
        best = c;
        improved = true;
      }
    }
    if (!improved) {
      lin *= 0.5;
      ang *= 0.5;
    }
  }

  return { pose: best, score: bestScore, seedScore, evaluations };
}

// ---------------------------------------------------------------------------
// The filter
// ---------------------------------------------------------------------------

export interface GridParticle {
  pose: Pose2;
  /** The particle *is* a path — the factorization theorem in a field name. */
  path: Pose2[];
  map: CowGrid;
  logWeight: number;
  /** Index of the ancestor this particle descends from, for the diversity gauge. */
  ancestor: number;
}

export interface RbpfOptions {
  alphas: OdomAlphas;
  angles: number[];
  score: MapScoreParams;
  inverse: InverseModelParams;
  scanMatch: ScanMatchOptions;
  /** K in the improved proposal: how many poses to probe around the mode. */
  kSamples: number;
  /** Spread of those probes: [σ_xy in m, σ_θ in rad]. */
  sampleSpread: [number, number];
  /** Resample when N_eff < ratio · M. Grisetti et al. use ratio = 1/2. */
  neffRatio: number;
  /** Off = resample every step, the classic mistake w17.2 exists to show. */
  selectiveResampling: boolean;
  /** Off = FastSLAM 1.0's proposal: sample the motion model, weight by the scan. */
  improvedProposal: boolean;
  /** Minimum score gain before the scan match is trusted over raw odometry. */
  scanMatchGain: number;
  /**
   * Exponent γ ∈ (0, 1] applied to the importance weight, w ← w · p(z | ·)^γ.
   *
   * Chapter 10 said out loud that beams are not independent, so the product
   * over 20 of them over-counts the evidence: untempered, one scan can swing a
   * weight by e^10 and the first resample collapses the population to a single
   * ancestor. Every deployed range-finder filter carries a knob of this shape —
   * a sub-sampled scan, an inflated σ, an explicit exponent. We make it
   * explicit. Note it is applied to the *weight* only: inside one particle's
   * proposal the over-counting is common to every candidate pose and cancels,
   * so the proposal keeps the full, sharp likelihood.
   */
  weightTemper: number;
}

export const DEFAULT_RBPF_OPTIONS: Omit<RbpfOptions, 'angles' | 'inverse'> = {
  alphas: [0.02, 0.01, 0.03, 0.01],
  score: DEFAULT_MAP_SCORE,
  scanMatch: DEFAULT_SCAN_MATCH,
  kSamples: 12,
  sampleSpread: [0.06, 0.035],
  neffRatio: 0.5,
  selectiveResampling: true,
  improvedProposal: true,
  scanMatchGain: 1,
  weightTemper: 0.25,
};

export interface RbpfReport {
  neff: number;
  /**
   * N_eff of *this step's* weight increments alone, before they compound with
   * the accumulated weight. This is the direct read-out of proposal quality:
   * a proposal that puts its samples where the measurement likes them produces
   * near-uniform increments and a stepNeff near M.
   */
  stepNeff: number;
  resampled: boolean;
  /** How many founding universes still have descendants. */
  distinctAncestors: number;
  /** Normalized weights, in particle order. */
  weights: number[];
  bestIndex: number;
  /** How many particles trusted their scan matcher this step. */
  scanMatched: number;
}

/**
 * `low_variance_sampler` (Thrun et al., Table 4.4) applied to *indices*.
 *
 * The book already owns that sampler, and duplicating it here would risk the
 * two drifting apart — so we hand it dummy particles whose x-coordinate is the
 * index and read the survivors back out. Same comb, same guarantees.
 */
export function resampleIndices(weights: number[], rng: Rng): number[] {
  const dummies: Particle[] = weights.map((w, i) => ({
    state: { x: i, y: 0, theta: 0 },
    weight: w,
  }));
  return lowVarianceResample(dummies, rng).map((p) => Math.round(p.state.x));
}

/** N_eff = 1 / Σ w̃², on already-normalized weights. */
export function effectiveSampleSize(weights: number[]): number {
  let s = 0;
  for (const w of weights) s += w * w;
  return s > 0 ? 1 / s : 0;
}

/** Normalize log weights into probabilities without overflowing. */
export function normalizeLogWeights(logWeights: number[]): number[] {
  const lse = logSumExp(logWeights);
  if (!Number.isFinite(lse)) return logWeights.map(() => 1 / logWeights.length);
  return logWeights.map((l) => Math.exp(l - lse));
}

export class GridRbpf {
  particles: GridParticle[];
  opts: RbpfOptions;
  /** Normalized weights from the most recent step. */
  weights: number[];

  constructor(particles: GridParticle[], opts: RbpfOptions) {
    this.particles = particles;
    this.opts = opts;
    this.weights = particles.map(() => 1 / particles.length);
  }

  /**
   * M particles, all at the same known start pose, each with its own empty map.
   * SLAM fixes the origin: t = 0 is where the robot declares the world begins.
   */
  static atPose(
    m: number,
    start: Pose2,
    makeMap: () => OccupancyGrid,
    opts: RbpfOptions,
  ): GridRbpf {
    return new GridRbpf(
      Array.from({ length: m }, (_, i) => ({
        pose: { ...start },
        path: [{ ...start }],
        map: CowGrid.of(makeMap()),
        logWeight: 0,
        ancestor: i,
      })),
      opts,
    );
  }

  get size(): number {
    return this.particles.length;
  }

  /** One `gmapping_step`: propose, weight, map, then maybe select. */
  step(u: OdomDelta, ranges: number[], rng: Rng): RbpfReport {
    const { opts } = this;
    let scanMatched = 0;
    const increments: number[] = [];

    for (const p of this.particles) {
      const predicted = applyOdom(p.pose, u);
      let pose = predicted;
      let increment = 0;
      let used = false;

      if (opts.improvedProposal) {
        const match = scanMatchGrid(
          p.map.read,
          predicted,
          ranges,
          opts.angles,
          opts.score,
          opts.scanMatch,
        );
        // A flat score surface means the map has nothing to say here yet —
        // trusting the "optimum" would be trusting numerical noise.
        if (match.score - match.seedScore > opts.scanMatchGain) {
          const fit = this.fitProposal(p, u, match.pose, ranges, rng);
          pose = fit.pose;
          increment = fit.logEta;
          used = true;
          scanMatched += 1;
        }
      }

      if (!used) {
        // FastSLAM 1.0's proposal: the motion model alone. The measurement then
        // has to do all the work through the weight, which is the whole problem.
        pose = sampleMotionModelOdometry(u, p.pose, opts.alphas, rng);
        increment = logMapMatchScore(p.map.read, pose, ranges, opts.angles, opts.score);
      }

      p.pose = pose;
      p.path.push({ ...pose });
      increments.push(opts.weightTemper * increment);
      p.logWeight += opts.weightTemper * increment;
      // Mapping with known poses — Chapter 13, run inside the hypothesis.
      p.map.write().integrateScan(pose, ranges, opts.angles, opts.inverse);
    }

    return this.select(rng, scanMatched, effectiveSampleSize(normalizeLogWeights(increments)));
  }

  /**
   * The improved proposal (Derivation 5): K probes around the scan-match mode,
   * weighted by observation × motion, collapsed to a Gaussian, sampled from.
   * The weight increment is the normalizer η — and because it no longer depends
   * on *where in the proposal* the sample landed, weight variance collapses.
   */
  private fitProposal(
    p: GridParticle,
    u: OdomDelta,
    mode: Pose2,
    ranges: number[],
    rng: Rng,
  ): { pose: Pose2; logEta: number } {
    const { kSamples, sampleSpread, alphas, angles, score } = this.opts;
    const [sxy, sth] = sampleSpread;

    const probes: Pose2[] = [];
    const logW: number[] = [];
    for (let k = 0; k < kSamples; k++) {
      const x: Pose2 =
        k === 0
          ? { ...mode }
          : {
              x: mode.x + rng.normal(0, sxy),
              y: mode.y + rng.normal(0, sxy),
              theta: normalizeAngle(mode.theta + rng.normal(0, sth)),
            };
      const obs = logMapMatchScore(p.map.read, x, ranges, angles, score);
      const mot = Math.log(Math.max(motionModelOdometry(x, u, p.pose, alphas), 1e-300));
      probes.push(x);
      logW.push(obs + mot);
    }

    const lse = logSumExp(logW);
    const w = logW.map((l) => Math.exp(l - lse));

    // Weighted mean; θ on the circle, because averaging −179° and 179° must not
    // land at 0°.
    let mx = 0;
    let my = 0;
    let cs = 0;
    let sn = 0;
    for (let k = 0; k < probes.length; k++) {
      mx += w[k] * probes[k].x;
      my += w[k] * probes[k].y;
      cs += w[k] * Math.cos(probes[k].theta);
      sn += w[k] * Math.sin(probes[k].theta);
    }
    const mean: Pose2 = { x: mx, y: my, theta: Math.atan2(sn, cs) };

    const cov: Mat = [
      [1e-6, 0, 0],
      [0, 1e-6, 0],
      [0, 0, 1e-8],
    ];
    for (let k = 0; k < probes.length; k++) {
      const d = [probes[k].x - mean.x, probes[k].y - mean.y, angleDiff(probes[k].theta, mean.theta)];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) cov[i][j] += w[k] * d[i] * d[j];
      }
    }

    const l = cholesky(cov);
    const z = [rng.normal(), rng.normal(), rng.normal()];
    const pose: Pose2 = {
      x: mean.x + l[0][0] * z[0],
      y: mean.y + l[1][0] * z[0] + l[1][1] * z[1],
      theta: normalizeAngle(mean.theta + l[2][0] * z[0] + l[2][1] * z[1] + l[2][2] * z[2]),
    };

    // η = Σ_k p(z | m, x_k) p(x_k | x_{t−1}, u): the marginal likelihood of the
    // scan under this particle, up to the constant K.
    return { pose, logEta: lse - Math.log(kSamples) };
  }

  /** Normalize, measure N_eff, and resample only if the policy says so. */
  private select(rng: Rng, scanMatched: number, stepNeff: number): RbpfReport {
    const m = this.particles.length;
    const weights = normalizeLogWeights(this.particles.map((p) => p.logWeight));
    const neff = effectiveSampleSize(weights);

    let bestIndex = 0;
    for (let i = 1; i < m; i++) if (weights[i] > weights[bestIndex]) bestIndex = i;

    const shouldResample = this.opts.selectiveResampling
      ? neff < this.opts.neffRatio * m
      : true;

    let resampled = false;
    if (shouldResample && m > 1) {
      const idx = resampleIndices(weights, rng);
      const next: GridParticle[] = idx.map((src) => {
        const p = this.particles[src];
        return {
          pose: { ...p.pose },
          path: p.path.slice(),
          map: p.map.share(), // O(1): the copy waits until this clone writes
          logWeight: 0,
          ancestor: p.ancestor,
        };
      });
      for (const p of this.particles) p.map.release();
      this.particles = next;
      resampled = true;
      this.weights = next.map(() => 1 / m);
    } else {
      this.weights = weights;
    }

    return {
      neff,
      stepNeff,
      resampled,
      distinctAncestors: new Set(this.particles.map((p) => p.ancestor)).size,
      weights: this.weights,
      bestIndex: resampled ? 0 : bestIndex,
      scanMatched,
    };
  }

  /** The heaviest particle — the map a consumer of this filter would publish. */
  best(): GridParticle {
    let b = 0;
    for (let i = 1; i < this.particles.length; i++) {
      if (this.weights[i] > this.weights[b]) b = i;
    }
    return this.particles[b];
  }

  /** Particle indices sorted by descending weight, for the universe tiles. */
  rankedIndices(): number[] {
    return this.particles
      .map((_, i) => i)
      .sort((a, b) => this.weights[b] - this.weights[a]);
  }
}
