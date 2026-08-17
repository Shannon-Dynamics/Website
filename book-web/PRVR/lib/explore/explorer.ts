/**
 * The explorer: identify → select → execute — Chapter 24.
 *
 * Placed et al. (2023) name the three stages every active-SLAM system has,
 * whether or not its authors drew the boxes: *identify* the candidate actions,
 * *select* one by a utility, *execute* it while the estimate keeps running.
 * This module is that loop, with the pieces plugged in: `detect_frontiers` for
 * identification, `score_candidates` for selection, and Chapter 4's simulator
 * plus Chapter 13's grid for execution.
 *
 * The same class powers the chapter's live widget and its headless ablation
 * runs, so the curve on the page and the numbers in the table come from one
 * implementation.
 *
 * Rust counterpart: `crates/ch24_explore/src/explorer.rs`.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import {
  DEFAULT_INVERSE_MODEL,
  OccupancyGrid,
  probToLogOdds,
  type InverseModelParams,
} from '../mapping/occgrid';
import { Rng } from '../prob/rng';
import { RUSTY_LIDAR, pursuePoint, raycastScan, type LidarParams } from '../sim/rusty';
import { collides, diffDriveStep, type World } from '../sim/world';
import {
  DEFAULT_THRESHOLDS,
  NavField,
  coverage,
  detectFrontiers,
  type ClassThresholds,
  type Frontier,
  type GridIdx,
} from './frontier';
import { DEFAULT_SENSING, type SensingParams } from './info-gain';
import {
  DEFAULT_STOP,
  DEFAULT_WEIGHTS,
  scoreCandidates,
  shouldStop,
  type Candidate,
  type StopReason,
  type StopRule,
  type UtilityWeights,
} from './utility';

/**
 * A mapping model tuned for *exploration* rather than for pretty maps: one
 * beam through a cell is enough to call it free (p = 0.25 < the 0.3 threshold).
 * A timid free-space model makes frontier detection lag the sensor by several
 * scans, and the robot spends the difference re-driving corridors it has
 * already emptied.
 */
export const EXPLORE_INVERSE_MODEL: InverseModelParams = {
  ...DEFAULT_INVERSE_MODEL,
  alpha: 0.25,
  beta: (9 * Math.PI) / 180,
  maxRange: 5,
  lOcc: probToLogOdds(0.8),
  lFree: probToLogOdds(0.25),
  l0: 0,
  clamp: 8,
};

export type ExplorePolicy = 'utility' | 'nearest' | 'lawnmower';

export interface ExploreConfig {
  world: World;
  start: Pose2;
  cellSize: number;
  sensing: SensingParams;
  lidar: LidarParams;
  weights: UtilityWeights;
  thresholds: ClassThresholds;
  model: InverseModelParams;
  stop: StopRule;
  policy: ExplorePolicy;
  minFrontierSize: number;
  inflate: number;
  /** Ticks between replans — new frontiers appear while a plan is being executed. */
  replanEvery: number;
  /** Ticks between scans. Sensing every tick is free here and not on a real robot. */
  senseEvery: number;
  dt: number;
  speed: number;
  seed: number;
}

export const DEFAULT_EXPLORE_CONFIG: Omit<ExploreConfig, 'world' | 'start'> = {
  cellSize: 0.15,
  sensing: { ...DEFAULT_SENSING, maxRange: 5, nBeams: 24 },
  lidar: { ...RUSTY_LIDAR, nBeams: 48, maxRange: 5, sigmaR: 0.02, pDropout: 0.005 },
  weights: DEFAULT_WEIGHTS,
  thresholds: DEFAULT_THRESHOLDS,
  model: EXPLORE_INVERSE_MODEL,
  stop: DEFAULT_STOP,
  policy: 'utility',
  minFrontierSize: 4,
  inflate: 1,
  replanEvery: 20,
  senseEvery: 2,
  dt: 0.25,
  speed: 0.7,
  seed: 24,
};

/** One row of the run log — the raw material of every chart in this chapter. */
export interface ExploreSample {
  tick: number;
  /** Metres travelled so far. The x-axis that makes policies comparable. */
  distance: number;
  /** H(m) in bits. */
  entropy: number;
  /** Bits removed per metre over the last window — the stopping statistic. */
  gainRate: number;
  /** Fraction of cells no longer at the prior. */
  coverage: number;
}

export interface Decision {
  kind: 'goto' | 'stop';
  target?: GridIdx;
  path: { x: number; y: number }[];
  candidates: Candidate[];
  chosen: Candidate | null;
  reason: StopReason;
}

/**
 * A scripted boustrophedon over the map's bounding box — the "lawnmower" the
 * chapter opens by parodying. It is a *coverage* path, and coverage path
 * planning is a solved problem (Choset, Ch. 6) for a floorplan you already
 * have. Here the robot does not have one, so lanes run into walls and the
 * script skips them.
 */
export function lawnmowerWaypoints(
  world: World,
  laneSpacing = 1.2,
  margin = 0.6,
): { x: number; y: number }[] {
  const { minX, minY, maxX, maxY } = world.bounds;
  const out: { x: number; y: number }[] = [];
  let flip = false;
  for (let y = minY + margin; y <= maxY - margin + 1e-9; y += laneSpacing) {
    const a = flip ? maxX - margin : minX + margin;
    const b = flip ? minX + margin : maxX - margin;
    out.push({ x: a, y }, { x: b, y });
    flip = !flip;
  }
  return out;
}

export class ExploreSim {
  readonly cfg: ExploreConfig;
  readonly grid: OccupancyGrid;
  readonly rng: Rng;

  pose: Pose2;
  trail: { x: number; y: number }[];
  distance = 0;
  tick = 0;

  scan: { ranges: number[]; angles: number[] } = { ranges: [], angles: [] };
  frontiers: Frontier[] = [];
  candidates: Candidate[] = [];
  chosen: Candidate | null = null;
  path: { x: number; y: number }[] = [];
  pathIdx = 0;
  done = false;
  reason: StopReason = 'running';
  samples: ExploreSample[] = [];

  private lane = 0;
  private nav: NavField | null = null;

  constructor(cfg: ExploreConfig) {
    this.cfg = cfg;
    this.rng = new Rng(cfg.seed);
    this.grid = OccupancyGrid.forWorld(cfg.world, cfg.cellSize, 0.5);
    this.pose = { ...cfg.start };
    this.trail = [{ x: cfg.start.x, y: cfg.start.y }];
    // One scan before the first decision: with an all-unknown map there is no
    // known-free space to plan through, and therefore no reachable frontier.
    this.sense();
    this.record();
  }

  get lawnmower(): { x: number; y: number }[] {
    return lawnmowerWaypoints(this.cfg.world);
  }

  /** Integrate one LiDAR sweep into the grid — Chapter 13, unmodified. */
  private sense(): void {
    const scan = raycastScan(this.cfg.world, this.pose, this.cfg.lidar, this.rng);
    this.scan = { ranges: scan.ranges, angles: scan.angles };
    this.grid.integrateScan(this.pose, scan.ranges, scan.angles, this.cfg.model);
  }

  private record(): void {
    const entropy = this.grid.entropy();
    const prev = this.samples[this.samples.length - 1];
    // Bits per metre over the last ~2 m of travel: the instantaneous rate is
    // pure noise, and the run-average hides the moment the run goes flat.
    let gainRate = 0;
    for (let k = this.samples.length - 1; k >= 0; k--) {
      const s = this.samples[k];
      if (this.distance - s.distance >= 2 || k === 0) {
        const dd = this.distance - s.distance;
        if (dd > 1e-6) gainRate = (s.entropy - entropy) / dd;
        break;
      }
    }
    if (!prev) gainRate = 0;
    this.samples.push({
      tick: this.tick,
      distance: this.distance,
      entropy,
      gainRate: Math.max(0, gainRate),
      coverage: coverage(this.grid),
    });
  }

  /**
   * `explore_step` — identify, select, execute; one decision.
   *
   * Identification and selection are the two lines below; everything else in
   * this class is execution. That is the honest proportion for a real system
   * too: the policy is small, and keeping the estimate alive while it runs is
   * where the code goes.
   */
  decide(): Decision {
    if (this.cfg.policy === 'lawnmower') {
      const wp = this.lawnmower;
      const target = wp[this.lane % wp.length];
      return {
        kind: 'goto',
        path: [target],
        candidates: [],
        chosen: null,
        reason: 'running',
      };
    }

    const [ri, rj] = this.grid.worldToCell(this.pose.x, this.pose.y);
    this.frontiers = detectFrontiers(this.grid, {
      thresholds: this.cfg.thresholds,
      minSize: this.cfg.minFrontierSize,
    });
    this.nav = new NavField(
      this.grid,
      { i: ri, j: rj },
      { thresholds: this.cfg.thresholds, inflate: this.cfg.inflate },
    );

    // Nearest-frontier is not a different algorithm. It is this utility with
    // the information term switched off — the chapter's first quiet point.
    const weights =
      this.cfg.policy === 'nearest' ? { wI: 0, wG: 0, wC: 1 } : this.cfg.weights;

    const candidates = scoreCandidates(
      this.grid,
      this.frontiers,
      this.nav,
      this.cfg.sensing,
      weights,
    );
    const reason = shouldStop(candidates, this.cfg.stop);
    if (reason !== 'running' || candidates.length === 0) {
      return { kind: 'stop', path: [], candidates, chosen: null, reason };
    }

    const chosen = candidates[0];
    const path = this.nav.pathTo(chosen.target);
    return { kind: 'goto', target: chosen.target, path, candidates, chosen, reason: 'running' };
  }

  private replan(): void {
    const d = this.decide();
    this.candidates = d.candidates;
    this.chosen = d.chosen;
    this.path = d.path;
    this.pathIdx = 0;
    if (d.kind === 'stop') {
      this.done = true;
      this.reason = d.reason;
    }
  }

  /** Advance the robot along the current plan by one control period. */
  private drive(): void {
    if (this.path.length === 0) return;
    const goal = this.path[Math.min(this.pathIdx, this.path.length - 1)];
    if (Math.hypot(goal.x - this.pose.x, goal.y - this.pose.y) < 0.22) {
      this.pathIdx++;
      if (this.cfg.policy === 'lawnmower' && this.pathIdx >= this.path.length) {
        this.lane++;
      }
      if (this.pathIdx >= this.path.length) return;
    }

    const target = this.path[Math.min(this.pathIdx, this.path.length - 1)];
    const u = pursuePoint(this.pose, target, {
      speed: this.cfg.speed,
      gain: 2.4,
      maxOmega: 1.6,
      turnFirst: 0.6,
    });
    const next = diffDriveStep(this.pose, u.v, u.omega, this.cfg.dt);
    if (collides(this.cfg.world, this.pose, next)) {
      // A lane that runs into a wall is abandoned, which is what a scripted
      // sweep of an unknown floorplan actually does.
      if (this.cfg.policy === 'lawnmower') this.lane++;
      this.path = [];
      return;
    }
    this.distance += Math.hypot(next.x - this.pose.x, next.y - this.pose.y);
    this.pose = { ...next, theta: normalizeAngle(next.theta) };
    this.trail.push({ x: this.pose.x, y: this.pose.y });
  }

  step(): void {
    if (this.done) return;
    this.tick++;

    const needsPlan =
      this.path.length === 0 ||
      this.pathIdx >= this.path.length ||
      this.tick % this.cfg.replanEvery === 0;

    this.drive();
    if (this.tick % this.cfg.senseEvery === 0) this.sense();
    if (needsPlan) this.replan();
    this.record();
  }

  /** The frontier cells, flattened, for drawing. */
  frontierCells(): { cell: GridIdx; utility: number }[] {
    const out: { cell: GridIdx; utility: number }[] = [];
    for (const c of this.candidates) {
      for (const cell of c.frontier.cells) out.push({ cell, utility: c.utility });
    }
    return out;
  }
}

export interface RunResult {
  samples: ExploreSample[];
  distance: number;
  entropy: number;
  coverage: number;
  ticks: number;
  reason: StopReason;
}

/**
 * Run a policy to exhaustion (or to `maxTicks`) with no rendering — the
 * ablation harness. Deterministic given the seed, which is what lets the
 * chapter's table be a unit test.
 */
export function runExploration(
  cfg: ExploreConfig,
  maxTicks = 900,
): RunResult {
  const sim = new ExploreSim(cfg);
  for (let k = 0; k < maxTicks && !sim.done; k++) sim.step();
  const last = sim.samples[sim.samples.length - 1];
  return {
    samples: sim.samples,
    distance: last.distance,
    entropy: last.entropy,
    coverage: last.coverage,
    ticks: sim.tick,
    reason: sim.reason,
  };
}

/** Distance travelled by the time `frac` of the run's total information had been gathered. */
export function distanceToInformation(result: RunResult, frac = 0.95): number {
  const s = result.samples;
  if (s.length === 0) return 0;
  const h0 = s[0].entropy;
  const total = h0 - s[s.length - 1].entropy;
  if (total <= 0) return 0;
  for (const row of s) {
    if (h0 - row.entropy >= frac * total) return row.distance;
  }
  return s[s.length - 1].distance;
}
