/**
 * The capstone: an autonomy stack, running.
 *
 * Eight tasks, each a stated approximation of one intractable mission POMDP,
 * exchanging stamped beliefs on a shared clock. On native Rust each task owns a
 * thread and a `crossbeam-channel`; here — and in the WASM build of the Rust —
 * a deterministic round-robin scheduler ticks whichever tasks are due inside
 * one frame. Same task code, same message types, same seed, same mission.
 *
 *   lidar (10 Hz) → slam (10 Hz) → map (10 Hz) → esdf (5 Hz)
 *                                             ↘ frontier (1 Hz) → plan (1 Hz)
 *                                                              ↘ control (20 Hz)
 *   supervisor (20 Hz) watches all of it.
 *
 * Every algorithm here is imported, not reinvented: ICP and the voxel map from
 * Chapter 16, the occupancy grid from Chapter 13, the motion model from
 * Chapter 9, the distance transform from Chapter 19, the particle machinery
 * from Chapters 8 and 12. What this file adds is the only genuinely new content
 * in Chapter 26 — composition: rates, interfaces, detectors, and a state
 * machine that decides what to do when a detector fires.
 *
 * Rust counterpart: `crates/capstone/src/{bus,task,mission}.rs`.
 */

import {
  adjoint,
  between,
  compose,
  IDENTITY_POSE,
  inverse,
  se2Log,
  type Pose2,
} from '../geom/se2';
import { OccupancyGrid, DEFAULT_INVERSE_MODEL } from '../mapping/occgrid';
import { sampleMotionModelVelocity, odomFromPoses, type MotionAlphas } from '../models/motion';
import { inv, matAdd, matMul, transpose, type Mat } from '../prob/linalg';
import { Rng } from '../prob/rng';
import { scanToCloud, transformCloud, voxelDownsample, VoxelMap, type Pt } from '../slam/cloud';
import { icp, icpInformation } from '../slam/icp';
import {
  APARTMENT,
  beamAngles,
  diffDriveStep,
  distanceToWalls,
  simulateScan,
  type ScanParams,
  type Segment,
  type World,
} from '../sim/world';

import { classifyGrid, dijkstraCostField, astarGrid, simplifyPath, DEFAULT_PLAN, FREE, UNKNOWN, type PlanConfig } from './astar';
import { ChiSquareGate, FitnessMonitor, Watchdog, CHI2_3DOF_999 } from './detectors';
import { esdfAt, esdfFromGrid, positionSigma, safetyMargin, type Esdf } from './esdf';
import { detectFrontiers, frontiersExhausted, DEFAULT_FRONTIER, type ScoredFrontier } from './frontier';
import { Mppi, reactiveCreep, type Cmd, type MppiResult } from './mppi';
import { GlobalRelocalizer, type RelocalizeStatus } from './relocalize';

/* -------------------------------------------------------------------------- */
/* Interfaces: what travels on the bus                                         */
/* -------------------------------------------------------------------------- */

/** D26.1: nothing crosses a task boundary without a time and a frame on it. */
export type FrameId = 'map' | 'odom' | 'base' | 'sensor';

export interface Stamped<T> {
  t: number;
  frame: FrameId;
  v: T;
}

/** A pose belief: mean on SE(2), covariance in the **body** tangent. */
export interface PoseBelief {
  mean: Pose2;
  cov: Mat;
}

export type TaskName = 'lidar' | 'slam' | 'map' | 'esdf' | 'frontier' | 'plan' | 'control' | 'supervisor';

export interface TaskSpec {
  name: TaskName;
  label: string;
  /** Scheduler period, in base ticks (one base tick = {@link BASE_DT}). */
  period: number;
  /** The chapter this task's algorithm comes from. */
  chapter: number;
  chapterSlug: string;
  publishes: string;
  consumes: string;
  /** What visibly breaks when this task is switched off (widget w26.2). */
  degradation: string;
}

/** The scheduler's base period: 20 Hz, the fastest task in the stack. */
export const BASE_DT = 0.05;

export const TASKS: TaskSpec[] = [
  {
    name: 'lidar',
    label: 'LiDAR',
    period: 2,
    chapter: 4,
    chapterSlug: 'ch04-rusty-and-sensors',
    publishes: 'Scan(360°, 6 m)',
    consumes: '—',
    degradation: 'No scans. The watchdog fires in 0.3 s and the stack coasts on odometry alone.',
  },
  {
    name: 'slam',
    label: 'SLAM front end',
    period: 2,
    chapter: 16,
    chapterSlug: 'ch16-scan-matching',
    publishes: 'PoseBelief(map)',
    consumes: 'Scan, Odom',
    degradation: 'Scan matching off: the pose belief becomes raw dead reckoning and the map shears.',
  },
  {
    name: 'map',
    label: 'Occupancy mapping',
    period: 2,
    chapter: 13,
    chapterSlug: 'ch13-occupancy-grids',
    publishes: 'MapPatch(log-odds)',
    consumes: 'Scan, PoseBelief',
    degradation: 'The map stops growing, so no frontier is ever created and exploration stalls.',
  },
  {
    name: 'esdf',
    label: 'ESDF / costmap',
    period: 4,
    chapter: 19,
    chapterSlug: 'ch19-map-representations',
    publishes: 'DistanceField + margin',
    consumes: 'MapPatch, PoseBelief',
    degradation: 'Clearance information freezes; the controller steers using a stale picture of the walls.',
  },
  {
    name: 'frontier',
    label: 'Frontier explorer',
    period: 20,
    chapter: 24,
    chapterSlug: 'ch24-exploration',
    publishes: 'Vec<ScoredFrontier>',
    consumes: 'MapPatch, PoseBelief',
    degradation: 'No goals are ever proposed: Rusty idles while reachable unknown space remains.',
  },
  {
    name: 'plan',
    label: 'Global planner',
    period: 20,
    chapter: 20,
    chapterSlug: 'ch20-motion-planning',
    publishes: 'Path(map)',
    consumes: 'Frontiers, DistanceField',
    degradation: 'No path. MPPI has nothing to track, so it holds position on the obstacle cost alone.',
  },
  {
    name: 'control',
    label: 'MPPI controller',
    period: 1,
    chapter: 23,
    chapterSlug: 'ch23-mppi',
    publishes: 'Cmd(v, ω)',
    consumes: 'Path, DistanceField, PoseBelief',
    degradation: 'Zero command. Everything upstream keeps running perfectly on a robot that never moves.',
  },
  {
    name: 'supervisor',
    label: 'Supervisor',
    period: 1,
    chapter: 22,
    chapterSlug: 'ch22-pomdps',
    publishes: 'Mode, StackEvent',
    consumes: 'everything',
    degradation: 'Detectors still compute, but nothing acts on them: a kidnapped robot never recovers.',
  },
];

/* -------------------------------------------------------------------------- */
/* D26.4 — the mode                                                            */
/* -------------------------------------------------------------------------- */

export type RecoverKind = 'sensor-dropout' | 'filter-divergence';

export type Mode =
  | { kind: 'Explore' }
  | { kind: 'Navigate'; goal: [number, number] }
  | { kind: 'Relocalize' }
  | { kind: 'Recover'; why: RecoverKind }
  | { kind: 'Done' };

export const MODE_ORDER: Mode['kind'][] = ['Explore', 'Navigate', 'Relocalize', 'Recover', 'Done'];

export type EventKind =
  | 'ModeSwitch'
  | 'GoalSelected'
  | 'GoalReached'
  | 'KidnapSuspected'
  | 'RelocalizeConverged'
  | 'RelocalizeRejected'
  | 'ScanTimeout'
  | 'ScanRestored'
  | 'NisGateTripped'
  | 'NoveltyDetected'
  | 'ChaosInjected'
  | 'GoalAbandoned'
  | 'MissionComplete';

export interface StackEvent {
  t: number;
  kind: EventKind;
  detail: string;
}

/* -------------------------------------------------------------------------- */
/* Configuration                                                               */
/* -------------------------------------------------------------------------- */

export interface MissionConfig {
  seed: number;
  /** Collision-chance bound δ of Derivation F2. */
  delta: number;
  rRobot: number;
  cellSize: number;
  scan: ScanParams;
  alphas: MotionAlphas;
  plan: PlanConfig;
  vMax: number;
  /** Tasks the reader has switched off (widget w26.2). */
  disabled: Partial<Record<TaskName, boolean>>;
  /** Chapter 25's calibrated sensor model: an honest σ instead of a hopeful one. */
  calibrated: boolean;
}

export const DEFAULT_MISSION: MissionConfig = {
  seed: 42,
  delta: 0.01,
  rRobot: 0.19,
  cellSize: 0.15,
  scan: { nBeams: 72, fov: 2 * Math.PI, maxRange: 6, sigma: 0.045, zRand: 0.01 },
  alphas: [0.08, 0.02, 0.06, 0.02, 0.03, 0.015],
  plan: DEFAULT_PLAN,
  vMax: 0.62,
  disabled: {},
  calibrated: true,
};

export interface TaskStat {
  ran: number;
  lastRun: number;
  /** Age of this task's most recent publication, seconds. */
  staleness: number;
  /** Measured publications per second over the whole run. */
  hz: number;
}

export interface HistorySample {
  t: number;
  entropy: number;
  coverage: number;
  rho: number;
  nis: number;
  sigma: number;
  margin: number;
  error: number;
  odomError: number;
  /** Beams the map cannot explain — the dynamic-obstacle signal. */
  novel: number;
  mode: Mode['kind'];
}

export interface Walker {
  x: number;
  y: number;
  vx: number;
  vy: number;
  until: number;
  radius: number;
}

/* -------------------------------------------------------------------------- */
/* The stack                                                                   */
/* -------------------------------------------------------------------------- */

const START_POSE: Pose2 = { x: 5.6, y: 4.4, theta: 0 };
const NIS_DOF3_95 = 7.815;

export class AutonomyStack {
  readonly cfg: MissionConfig;
  readonly world: World = APARTMENT;
  readonly angles: number[];
  readonly rng: Rng;

  tick = 0;
  time = 0;

  // --- ground truth (simulator-only luxury) --------------------------------
  truth: Pose2 = { ...START_POSE };
  deadReckon: Pose2 = { ...START_POSE };
  contacts = 0;

  // --- beliefs and products ------------------------------------------------
  belief: PoseBelief = { mean: { ...START_POSE }, cov: [[4e-4, 0, 0], [0, 4e-4, 0], [0, 0, 4e-4]] };
  grid: OccupancyGrid;
  cls: Uint8Array;
  esdf: Esdf;
  margin: number;
  costField: Float64Array;
  frontiers: ScoredFrontier[] = [];
  goal: [number, number] | null = null;
  path: [number, number][] = [];
  cmd: Cmd = { v: 0, omega: 0 };
  lastMppi: MppiResult | null = null;

  // --- sensing -------------------------------------------------------------
  scan: Stamped<number[]> | null = null;
  novelty: [number, number][] = [];
  scanDrop = 0;

  // --- estimation internals ------------------------------------------------
  private odomAccum: Pose2 = { ...IDENTITY_POSE };
  /**
   * The scan-matching target: every keyframe sweep, in the map frame.
   *
   * Chapter 16's front end matches against a *sliding* local map and therefore
   * drifts by construction; the back end repairs it. The capstone matches
   * scan-to-map instead — the architecture SLAM Toolbox and Cartographer use —
   * because a persistent target anchors the estimate every time Rusty revisits
   * a corridor, which is a loop closure that costs nothing. The pose graph is
   * still what the Rust build runs behind it; see the honesty note in §7.
   */
  private mapCloud = new VoxelMap(0.16, 3);
  private lastKeyframe: Pose2 = { ...START_POSE };
  private mppi: Mppi;
  private processedScan = -1;
  /** Set by `slam_tick` when a genuine correction happened this tick. */
  private estimatorUpdated = false;
  private goalDeadline = 0;
  private replanAfter = 0;
  private progressPose: Pose2 = { ...START_POSE };
  private progressTime = 0;
  private backUntil = -1;

  // --- supervision ---------------------------------------------------------
  mode: Mode = { kind: 'Explore' };
  fitness = new FitnessMonitor();
  nisGate = new ChiSquareGate(NIS_DOF3_95, 4);
  grossGate = new ChiSquareGate(CHI2_3DOF_999, 2);
  watchdog: Watchdog;
  relocalizer = new GlobalRelocalizer();
  reloc: RelocalizeStatus | null = null;
  private recoverUntil = 0;
  private resumeMode: Mode = { kind: 'Explore' };

  // --- telemetry -----------------------------------------------------------
  events: StackEvent[] = [];
  history: HistorySample[] = [];
  stats: Record<TaskName, TaskStat>;
  icpFitness = 1;
  icpRmse = 0;
  nis = 0;

  // --- chaos ---------------------------------------------------------------
  walker: Walker | null = null;
  private dropoutUntil = -1;

  // --- coverage bookkeeping -------------------------------------------------
  private reachable: Uint8Array;
  private reachableCount = 1;

  constructor(cfg: Partial<MissionConfig> = {}) {
    this.cfg = { ...DEFAULT_MISSION, ...cfg };
    this.rng = new Rng(this.cfg.seed);
    this.angles = beamAngles(this.cfg.scan);
    this.grid = OccupancyGrid.forWorld(this.world, this.cfg.cellSize);
    this.cls = classifyGrid(this.grid);
    this.esdf = esdfFromGrid(this.grid);
    this.margin = safetyMargin(this.cfg.rRobot, positionSigma(this.belief.cov), this.cfg.delta);
    this.costField = new Float64Array(this.grid.width * this.grid.height).fill(Infinity);
    this.mppi = new Mppi({ vMax: this.cfg.vMax, vRef: this.cfg.vMax * 0.8, dt: 2 * BASE_DT });
    this.watchdog = new Watchdog(2 * BASE_DT, 3);

    this.stats = Object.fromEntries(
      TASKS.map((t) => [t.name, { ran: 0, lastRun: 0, staleness: 0, hz: 0 }]),
    ) as Record<TaskName, TaskStat>;

    const { reachable, count } = floodReachable(this.world, this.grid, this.cfg.rRobot, START_POSE);
    this.reachable = reachable;
    this.reachableCount = Math.max(count, 1);

    // Prime the front end with the first sweep, so ICP always has a target.
    this.senseTick();
    if (this.scan) {
      const cloud = this.cloudFromScan(this.scan.v);
      this.pushKeyframe(cloud, this.belief.mean, true);
      this.mapTick();
      this.esdfTick();
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Scheduler                                                               */
  /* ---------------------------------------------------------------------- */

  private due(name: TaskName): boolean {
    if (this.cfg.disabled[name]) return false;
    const spec = TASKS.find((t) => t.name === name) as TaskSpec;
    return this.tick % spec.period === 0;
  }

  private ran(name: TaskName): void {
    const s = this.stats[name];
    s.ran += 1;
    s.lastRun = this.time;
    s.hz = this.time > 0.2 ? s.ran / this.time : 0;
  }

  /**
   * `stack_tick` — one scheduler quantum.
   *
   * The order is the dataflow order, so a message published this tick is
   * consumed this tick and staleness only ever comes from a task's *period*,
   * never from the scheduler. That is the property that makes the browser and
   * the threaded native build agree run-for-run.
   */
  step(): void {
    this.tick += 1;
    this.time = this.tick * BASE_DT;

    this.moveWalker();
    if (this.due('lidar')) this.senseTick();
    if (this.due('slam')) this.slamTick();
    if (this.due('map')) this.mapTick();
    if (this.due('esdf')) this.esdfTick();
    if (this.due('frontier')) this.frontierTick();
    if (this.due('plan')) this.planTick();
    if (this.due('supervisor')) this.supervisorTick();
    if (this.due('control')) this.controlTick();

    this.integrateMotion();

    for (const t of TASKS) this.stats[t.name].staleness = this.time - this.stats[t.name].lastRun;
    // Sampled at the SLAM rate, so every scan's detector reading lands in the
    // record — a statistic that dips for two scans must not be sampled away.
    if (this.tick % 2 === 0) this.record();
  }

  /* ---------------------------------------------------------------------- */
  /* Tasks                                                                   */
  /* ---------------------------------------------------------------------- */

  /** The sensing world: static walls plus whatever is walking through them. */
  private sensingWorld(): World {
    if (!this.walker) return this.world;
    return { ...this.world, walls: [...this.world.walls, ...walkerSegments(this.walker)] };
  }

  private senseTick(): void {
    if (this.time < this.dropoutUntil) {
      this.scanDrop += 1;
      return;
    }
    const ranges = simulateScan(this.sensingWorld(), this.truth, this.cfg.scan, this.rng);
    this.scan = { t: this.time, frame: 'sensor', v: ranges };
    this.watchdog.touch(this.time);
    this.ran('lidar');
  }

  /** Endpoints of the beams that disagree with the map — Chapter 12's novelty test. */
  private findNovelty(ranges: number[], pose: Pose2): boolean[] {
    const max = this.cfg.scan.maxRange;
    const flags = ranges.map((z, i) => {
      if (z >= max * 0.98) return false;
      const a = pose.theta + this.angles[i];
      const px = pose.x + z * Math.cos(a);
      const py = pose.y + z * Math.sin(a);
      const [ci, cj] = this.grid.worldToCell(px, py);
      if (!this.grid.inBounds(ci, cj)) return false;
      // The map says this cell is free with confidence, and something stopped
      // the beam there. Either the map is wrong or the world moved.
      return this.cls[cj * this.grid.width + ci] === FREE && esdfAt(this.esdf, px, py) > 0.3;
    });
    this.novelty = [];
    for (let i = 0; i < flags.length; i++) {
      if (!flags[i]) continue;
      const a = pose.theta + this.angles[i];
      this.novelty.push([pose.x + ranges[i] * Math.cos(a), pose.y + ranges[i] * Math.sin(a)]);
    }
    return flags;
  }

  private cloudFromScan(ranges: number[], keep?: boolean[]): Pt[] {
    const kept = keep ? ranges.map((z, i) => (keep[i] ? this.cfg.scan.maxRange : z)) : ranges;
    const cloud = scanToCloud(kept, this.angles, this.cfg.scan.maxRange, this.time);
    return voxelDownsample(cloud.points, 0.11);
  }

  /** Add a sweep to the map cloud, but only when Rusty has actually moved. */
  private pushKeyframe(cloud: Pt[], pose: Pose2, force = false): void {
    const rel = between(this.lastKeyframe, pose);
    if (!force && this.mapCloud.size > 0 && Math.hypot(rel.x, rel.y) < 0.28 && Math.abs(rel.theta) < 0.3) {
      return;
    }
    this.mapCloud.insert(transformCloud(pose, cloud));
    this.lastKeyframe = { ...pose };
  }

  /**
   * `slam_tick` — predict with odometry, correct with a scan match, and publish
   * a belief rather than a pose.
   *
   * The covariance is carried in the body tangent (right perturbation
   * $x = \hat{x} \boxplus \xi$), so prediction is $A \Sigma A^\top + Q$ with
   * $A = \mathrm{Ad}_{\delta^{-1}}$ and correction is one addition in the
   * information form. That covariance is not decoration: it is the σ that
   * Derivation F2's margin is computed from, and the reader can watch the
   * margin breathe as it grows and shrinks.
   */
  private slamTick(): void {
    this.ran('slam');
    const delta = this.odomAccum;
    this.odomAccum = { ...IDENTITY_POSE };

    // --- prediction --------------------------------------------------------
    const predicted = compose(this.belief.mean, delta);
    const A = adjoint(inverse(delta));
    const q = processNoise(delta);
    let cov = matAdd(matMul(matMul(A, this.belief.cov), transpose(A)), q);

    const fresh = this.scan !== null && this.scan.t > this.processedScan;

    if (this.mode.kind === 'Relocalize') {
      // The Gaussian belief is known to be wrong; MCL owns the pose now.
      this.belief = { mean: predicted, cov };
      if (fresh && this.scan) {
        this.processedScan = this.scan.t;
        const status = this.relocalizer.update(
          odomFromPoses(IDENTITY_POSE, delta),
          this.scan.v,
          this.angles,
          this.esdf,
          this.rng,
        );
        this.reloc = status;
        if (status.converged) this.finishRelocalize(status);
      }
      return;
    }

    if (!fresh || !this.scan) {
      // Open loop. Nothing to correct with, so the covariance simply grows —
      // which is exactly what makes the dropout recovery visible.
      this.belief = { mean: predicted, cov };
      return;
    }
    this.processedScan = this.scan.t;

    // --- correction --------------------------------------------------------
    const novel = this.findNovelty(this.scan.v, predicted);
    const cloud = this.cloudFromScan(this.scan.v, novel);
    const result = icp(cloud, this.mapCloud, predicted, {
      maxIters: 14,
      tau: 0.45,
      variant: 'point-to-point',
      minPairs: 8,
    });

    // Scan-match **fitness**: the fraction of beam endpoints that land within a
    // quarter of a metre of something the map already knows about, evaluated at
    // the pose ICP settled on. It is deliberately stricter than ICP's own
    // inlier count (τ = 0.45 m), because the question the kidnap detector asks
    // is not "did ICP converge?" — it always converges to something — but "does
    // this sweep actually belong to this part of the map?".
    this.icpRmse = result.rmse;
    this.icpFitness = mapAgreement(transformCloud(result.pose, cloud), this.mapCloud, 0.25);

    let mean = predicted;
    if (result.inliers >= 10 && Number.isFinite(result.rmse)) {
      const pairs = result.trace[result.trace.length - 1].pairs;
      // Chapter 25's toggle bites here: an uncalibrated model claims a σ it has
      // not earned, which inflates Ω, shrinks Σ, shrinks the F2 margin, and
      // eventually trips the divergence gate. Overconfidence is a *systems*
      // failure, not just a statistical one.
      const calibrated = this.cfg.calibrated;
      const sigma = calibrated ? Math.max(this.cfg.scan.sigma, 0.02) : this.cfg.scan.sigma * 0.4;
      // `inflate` divides the effective number of *independent* beams. A
      // calibrated model knows consecutive LiDAR beams are not independent and
      // says so; an uncalibrated one takes Jᵀ J at face value, and the cap that
      // stops one match from shouting comes off with it.
      const omega = ridge(
        icpInformation(pairs, sigma, calibrated ? 14 : 5, predicted.theta),
        1.5,
        calibrated ? 2400 : 9000,
      );
      const R = inv(omega);

      const nu = se2Log(between(predicted, result.pose));
      const S = matAdd(cov, R);
      this.nis = quadForm(nu, inv(S));

      mean = result.pose;
      cov = inv(matAdd(inv(cov), omega));
    } else {
      this.nis = 0;
    }

    this.estimatorUpdated = true;
    this.belief = { mean, cov: symmetric(cov) };
    this.pushKeyframe(cloud, mean);
  }

  private mapTick(): void {
    // Mapping is suspended whenever the pose is not trustworthy. Integrating a
    // scan at a pose you have just declared wrong is how a good map dies.
    if (this.mode.kind === 'Relocalize' || !this.scan) return;
    if (this.scan.t < this.time - 4 * BASE_DT) return;
    this.ran('map');

    const flags = this.novelty.length > 0 ? this.findNovelty(this.scan.v, this.belief.mean) : null;
    let ranges = this.scan.v;
    let angles = this.angles;
    if (flags) {
      // Withhold the dynamic beams. The map never learns the walker exists.
      const keepR: number[] = [];
      const keepA: number[] = [];
      for (let i = 0; i < ranges.length; i++) {
        if (flags[i]) continue;
        keepR.push(ranges[i]);
        keepA.push(angles[i]);
      }
      ranges = keepR;
      angles = keepA;
    }
    this.grid.integrateScan(this.belief.mean, ranges, angles, {
      ...DEFAULT_INVERSE_MODEL,
      alpha: 2 * this.cfg.cellSize,
      beta: (2 * Math.PI) / this.cfg.scan.nBeams,
      maxRange: this.cfg.scan.maxRange,
      clamp: 9,
    });
  }

  private esdfTick(): void {
    this.ran('esdf');
    this.cls = classifyGrid(this.grid);
    this.esdf = esdfFromGrid(this.grid, { transient: this.novelty });
    this.margin = safetyMargin(this.cfg.rRobot, positionSigma(this.belief.cov), this.cfg.delta);
  }

  private frontierTick(): void {
    this.ran('frontier');
    const planCfg = { ...this.cfg.plan, margin: this.margin };
    this.costField = dijkstraCostField(this.grid, this.cls, this.esdf, [this.belief.mean.x, this.belief.mean.y], planCfg);
    this.frontiers = detectFrontiers(this.grid, this.cls, this.costField, DEFAULT_FRONTIER);
  }

  private planTick(): void {
    if (!this.goal) {
      this.path = [];
      return;
    }
    this.ran('plan');
    const planCfg = { ...this.cfg.plan, margin: this.margin };
    const res = astarGrid(
      this.grid,
      this.cls,
      this.esdf,
      [this.belief.mean.x, this.belief.mean.y],
      this.goal,
      planCfg,
    );
    this.path = res.found ? simplifyPath(res.path, 0.06) : [];
    if (!res.found) this.goal = null;
  }

  private controlTick(): void {
    this.ran('control');
    const m = this.mode;

    if (this.time < this.backUntil) {
      this.cmd = { v: -0.2, omega: 0.7 };
      return;
    }

    if (m.kind === 'Done' || (m.kind === 'Recover' && m.why === 'sensor-dropout')) {
      // Creep to a stop: the safe thing to do when you cannot see.
      this.cmd = { v: this.cmd.v * 0.72, omega: this.cmd.omega * 0.72 };
      if (Math.abs(this.cmd.v) < 0.01) this.cmd = { v: 0, omega: 0 };
      return;
    }

    if (m.kind === 'Relocalize') {
      // No usable map-frame pose ⇒ no map-referenced cost is meaningful.
      this.cmd = this.scan ? reactiveCreep(this.scan.v, this.angles, 0.34) : { v: 0, omega: 0 };
      return;
    }

    const speedScale = m.kind === 'Recover' ? 0.4 : 1;
    if (this.path.length < 2) {
      this.cmd = { v: 0, omega: this.cmd.omega * 0.5 };
      return;
    }
    const res = this.mppi.step(this.belief.mean, this.path, this.esdf, this.margin, this.rng, speedScale);
    this.lastMppi = res;
    this.cmd = res.cmd;
  }

  /* ---------------------------------------------------------------------- */
  /* Supervisor — D26.4                                                      */
  /* ---------------------------------------------------------------------- */

  /** Has the robot failed to move for four seconds while trying to? */
  private stuck(): boolean {
    if (Math.hypot(this.belief.mean.x - this.progressPose.x, this.belief.mean.y - this.progressPose.y) > 0.25) {
      this.progressPose = { ...this.belief.mean };
      this.progressTime = this.time;
      return false;
    }
    return this.time - this.progressTime > 4;
  }

  private supervisorTick(): void {
    this.ran('supervisor');

    // --- F4(c): the watchdog runs first, because a missing message is the one
    // failure that produces no statistic at all.
    const stale = this.watchdog.expired(this.time);
    if (stale && this.mode.kind !== 'Recover' && this.mode.kind !== 'Relocalize') {
      this.emit('ScanTimeout', `no scan for ${this.watchdog.age(this.time).toFixed(2)} s`);
      this.enter({ kind: 'Recover', why: 'sensor-dropout' }, true);
      return;
    }
    if (!stale && this.mode.kind === 'Recover' && this.mode.why === 'sensor-dropout') {
      this.emit('ScanRestored', 'scans flowing again');
      this.enter(this.resumeMode);
      return;
    }

    // Both remaining detectors are statistics *of a scan match*, so they may
    // only be fed once per scan — never once per scheduler tick, which would
    // silently halve their effective thresholds.
    if (this.estimatorUpdated) {
      this.estimatorUpdated = false;

      // --- F4(a): kidnap / mislocalisation, the slow case. The robot has ended
      // up somewhere that does not look like where it thinks it is.
      if (this.fitness.update(this.icpFitness)) {
        this.emit('KidnapSuspected', `ρ = ${this.fitness.rho.toFixed(2)} < ${this.fitness.rhoMin}`);
        this.beginRelocalize();
        return;
      }

      // …and the violent case. An innovation past the 99.9% point of χ²₃ twice
      // running is not a filter that needs damping, it is a filter tracking the
      // wrong hypothesis; the same statistic, two thresholds, two responses.
      if (this.nis > 0 && this.grossGate.update(this.nis)) {
        this.emit('KidnapSuspected', `NIS ${this.nis.toFixed(1)} past the 99.9% gate twice`);
        this.grossGate.reset();
        this.beginRelocalize();
        return;
      }

      // --- F4(b): filter divergence.
      if (this.nis > 0 && this.nisGate.update(this.nis)) {
        this.emit('NisGateTripped', `NIS ${this.nis.toFixed(1)} > χ²₃,₉₅ for ${this.nisGate.consecutive} scans`);
        this.nisGate.reset();
        this.enter({ kind: 'Recover', why: 'filter-divergence' }, true);
        this.recoverUntil = this.time + 1.5;
        // Buy back honesty: the covariance we published was too tight.
        this.belief.cov = this.belief.cov.map((r) => r.map((v) => v * 4));
        return;
      }
    }

    switch (this.mode.kind) {
      case 'Recover': {
        if (this.mode.why === 'filter-divergence' && this.time > this.recoverUntil) {
          this.enter(this.resumeMode);
        }
        break;
      }
      case 'Relocalize':
        break;
      case 'Explore': {
        const best = this.frontiers.find((f) => Number.isFinite(f.cost) && f.utility > 0);
        if (best) {
          this.goal = best.goal;
          this.planTick();
          if (this.path.length >= 2) {
            this.emit('GoalSelected', `frontier #${best.id}, gain ${best.gain}, ${best.cost.toFixed(1)} m away`);
            this.goalDeadline = this.time + 30;
            this.enter({ kind: 'Navigate', goal: best.goal });
          }
        } else if (
          frontiersExhausted(this.frontiers, DEFAULT_FRONTIER.minCells) &&
          this.entropySettled()
        ) {
          this.emit('MissionComplete', `coverage ${(this.coverage() * 100).toFixed(1)}%`);
          this.enter({ kind: 'Done' });
        }
        break;
      }
      case 'Navigate': {
        const g = this.mode.goal;
        const d = Math.hypot(this.belief.mean.x - g[0], this.belief.mean.y - g[1]);
        const done =
          d < 0.5 ||
          !this.stillFrontier(g) ||
          this.path.length < 2 ||
          this.time > this.goalDeadline ||
          this.stuck();
        if (done) {
          if (d < 0.5) this.emit('GoalReached', `within ${d.toFixed(2)} m`);
          else if (this.stuck()) {
            // Wedged. Certainty equivalence said this was drivable; the wheels
            // disagree. Back off, blacklist the goal, and let the explorer pick
            // something else — the cheapest possible form of learning from the
            // world instead of from the map.
            this.emit('GoalAbandoned', `no progress for 4 s at (${this.belief.mean.x.toFixed(1)}, ${this.belief.mean.y.toFixed(1)})`);
            this.backUntil = this.time + 0.9;
            this.progressTime = this.time;
          }
          // Retire this frontier from the cached list. Without it the greedy
          // selector re-picks the goal it has just finished with, and the
          // supervisor oscillates at the scheduler rate until the 1 Hz frontier
          // task next runs — the single most common bug in a mode machine.
          this.frontiers = this.frontiers.filter(
            (f) => Math.hypot(f.goal[0] - g[0], f.goal[1] - g[1]) > 0.7,
          );
          this.goal = null;
          this.path = [];
          this.enter({ kind: 'Explore' });
        } else if (this.time > this.replanAfter && this.noveltyOnPath()) {
          this.emit('NoveltyDetected', `${this.novelty.length} beams disagree with the map — replanning`);
          this.replanAfter = this.time + 1;
          this.planTick();
        }
        break;
      }
      case 'Done':
        break;
    }
  }

  /** Is the cell we are driving to still on the boundary of the unknown? */
  private stillFrontier(goal: [number, number]): boolean {
    const [i, j] = this.grid.worldToCell(goal[0], goal[1]);
    if (!this.grid.inBounds(i, j)) return false;
    const nx = this.grid.width;
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as [number, number][]) {
      const ni = i + di;
      const nj = j + dj;
      if (!this.grid.inBounds(ni, nj)) continue;
      if (this.cls[nj * nx + ni] === UNKNOWN) return true;
    }
    return false;
  }

  private enter(next: Mode, remember = false): void {
    const same = next.kind === this.mode.kind;
    if (same && next.kind !== 'Navigate') return;
    if (remember && this.mode.kind !== 'Recover') this.resumeMode = this.mode;
    this.emit('ModeSwitch', `${this.mode.kind} → ${next.kind}`);
    this.mode = next;
    if (!same) this.mppi.reset();
  }

  private beginRelocalize(): void {
    this.resumeMode = { kind: 'Explore' };
    this.enter({ kind: 'Relocalize' });
    this.relocalizer.scatter(this.grid, this.cls, this.esdf, this.rng);
    this.reloc = null;
    this.goal = null;
    this.path = [];
    this.fitness.reset();
  }

  /**
   * Verify before committing — the same discipline as loop closure in
   * Chapter 16. A converged particle cloud is a *hypothesis*; accepting it
   * without checking is how a robot ends up confidently in the wrong room.
   */
  private verifyRelocalize(pose: Pose2): number {
    if (!this.scan) return 0;
    return mapAgreement(transformCloud(pose, this.cloudFromScan(this.scan.v)), this.mapCloud, 0.3);
  }

  private finishRelocalize(status: RelocalizeStatus): void {
    const agreement = this.verifyRelocalize(status.pose);
    if (agreement < 0.75) {
      this.emit(
        'RelocalizeRejected',
        `hypothesis at (${status.pose.x.toFixed(1)}, ${status.pose.y.toFixed(1)}) matches only ${(agreement * 100).toFixed(0)}% of the sweep — scattering again`,
      );
      this.relocalizer.scatter(this.grid, this.cls, this.esdf, this.rng);
      return;
    }
    this.emit(
      'RelocalizeConverged',
      `spread ${status.spread.toFixed(2)} m after ${this.relocalizer.steps} scans, ESS ${(status.ess * 100).toFixed(0)}%`,
    );
    const s2 = Math.max(status.spread * status.spread, 4e-3);
    this.belief = { mean: status.pose, cov: [[s2, 0, 0], [0, s2, 0], [0, 0, 0.02]] };
    // The map survives the kidnapping — it is what we relocalised *into*.
    this.lastKeyframe = { ...status.pose };
    this.fitness.reset();
    this.nisGate.reset();
    this.grossGate.reset();
    // Everything downstream was computed for the old, wrong pose. Rebuild it
    // before the supervisor is allowed to make a decision on it.
    this.esdfTick();
    this.frontierTick();
    this.enter({ kind: 'Explore' });
  }

  /**
   * Does a *cluster* of flagged dynamic points sit close enough to the plan to
   * matter? One lonely novel beam is speckle; three is a person.
   */
  private noveltyOnPath(): boolean {
    if (this.novelty.length < 3 || this.path.length < 2) return false;
    for (const [x, y] of this.novelty) {
      for (const p of this.path) {
        if (Math.hypot(p[0] - x, p[1] - y) < 0.5) return true;
      }
    }
    return false;
  }

  /** D26.3, second half: has the map stopped getting more certain? */
  private entropySettled(hMin = 0.35, window = 48): boolean {
    if (this.history.length < window) return false;
    const tail = this.history.slice(-window);
    // Mapping is suspended during recovery, so the entropy curve is flat for
    // reasons that have nothing to do with the map being finished. A window
    // containing a recovery proves nothing and is refused.
    if (tail.some((h) => h.mode === 'Relocalize' || h.mode === 'Recover')) return false;
    const dt = tail[tail.length - 1].t - tail[0].t;
    if (dt <= 0) return false;
    return Math.abs((tail[tail.length - 1].entropy - tail[0].entropy) / dt) < hMin;
  }

  /* ---------------------------------------------------------------------- */
  /* Physics and chaos                                                       */
  /* ---------------------------------------------------------------------- */

  private integrateMotion(): void {
    const prev = { ...this.truth };
    const next = diffDriveStep(prev, this.cmd.v, this.cmd.omega, BASE_DT);
    if (distanceToWalls(this.sensingWorld(), next.x, next.y) > this.cfg.rRobot * 0.85) {
      this.truth = next;
    } else {
      // A real bump: the wheels turned, the robot did not move, and the
      // odometry cheerfully reports the motion anyway.
      this.truth = { ...prev, theta: next.theta };
      this.contacts += 1;
    }

    const noisy = sampleMotionModelVelocity(
      { v: this.cmd.v, omega: this.cmd.omega, dt: BASE_DT },
      prev,
      this.cfg.alphas,
      this.rng,
    );
    const delta = between(prev, noisy);
    this.deadReckon = compose(this.deadReckon, delta);
    this.odomAccum = compose(this.odomAccum, delta);
  }

  private moveWalker(): void {
    const w = this.walker;
    if (!w) return;
    if (this.time > w.until) {
      this.walker = null;
      return;
    }
    const nx = w.x + w.vx * BASE_DT;
    const ny = w.y + w.vy * BASE_DT;
    if (distanceToWalls(this.world, nx, ny) < w.radius + 0.1) {
      w.vx = -w.vx;
      w.vy = -w.vy;
    } else {
      w.x = nx;
      w.y = ny;
    }
  }

  /** Chaos button 1: teleport Rusty, tell nothing. */
  kidnap(): void {
    const candidates: [number, number][] = [];
    const fallback: [number, number][] = [];
    for (let k = 0; k < this.cls.length; k++) {
      if (this.cls[k] !== FREE) continue;
      const i = k % this.grid.width;
      const j = (k - i) / this.grid.width;
      const [x, y] = this.grid.cellCenter(i, j);
      if (esdfAt(this.esdf, x, y) < 0.45) continue;
      if (Math.hypot(x - this.truth.x, y - this.truth.y) < 3) continue;
      fallback.push([x, y]);
      // Prefer a *room*. Dropping the robot somewhere down the same featureless
      // corridor is a legitimate experiment — see Exercise 4 — but the scan
      // there genuinely matches the map, so no detector built on scan agreement
      // can be expected to notice. Rooms have corners, and corners are evidence.
      if (y > 3.6 && y < 5.2) continue;
      candidates.push([x, y]);
    }
    const pool = candidates.length > 0 ? candidates : fallback;
    if (pool.length === 0) return;
    const [x, y] = pool[Math.floor(this.rng.next() * pool.length)];
    this.truth = { x, y, theta: this.rng.uniform(-Math.PI, Math.PI) };
    this.emit('ChaosInjected', `kidnapped to (${x.toFixed(1)}, ${y.toFixed(1)})`);
  }

  /**
   * Chaos button 2: a person walks across Rusty's path.
   *
   * Placed one metre to one side of a point a couple of metres ahead, moving
   * across. The search over offsets is not fussiness: dropping the walker
   * inside a wall would make it bounce in place and the demo would teach
   * nothing.
   */
  spawnWalker(speed = 0.8, duration = 14): void {
    const h = this.truth.theta;
    const px = -Math.sin(h);
    const py = Math.cos(h);
    for (const ahead of [2.0, 1.5, 2.6, 1.1]) {
      for (const side of [1, -1]) {
        const x = this.truth.x + ahead * Math.cos(h) + side * px;
        const y = this.truth.y + ahead * Math.sin(h) + side * py;
        if (distanceToWalls(this.world, x, y) < 0.42) continue;
        this.walker = {
          x,
          y,
          vx: -side * speed * px,
          vy: -side * speed * py,
          until: this.time + duration,
          radius: 0.22,
        };
        this.emit('ChaosInjected', `walker crossing at ${speed.toFixed(1)} m/s`);
        return;
      }
    }
    this.emit('ChaosInjected', 'no room for a walker here — try again in the corridor');
  }

  /** Chaos button 3: the LiDAR stops talking. */
  dropSensor(seconds = 2.5): void {
    this.dropoutUntil = this.time + seconds;
    this.emit('ChaosInjected', `LiDAR dropout for ${seconds.toFixed(1)} s`);
  }

  /* ---------------------------------------------------------------------- */
  /* Telemetry                                                               */
  /* ---------------------------------------------------------------------- */

  private emit(kind: EventKind, detail: string): void {
    this.events.push({ t: this.time, kind, detail });
    if (this.events.length > 240) this.events.shift();
  }

  coverage(): number {
    let known = 0;
    for (let k = 0; k < this.cls.length; k++) {
      if (this.reachable[k] && this.cls[k] !== UNKNOWN) known++;
    }
    return known / this.reachableCount;
  }

  /** Absolute trajectory error — a luxury that exists only because we own the world. */
  error(): number {
    return Math.hypot(this.belief.mean.x - this.truth.x, this.belief.mean.y - this.truth.y);
  }

  private record(): void {
    this.history.push({
      t: this.time,
      entropy: this.grid.entropy(),
      coverage: this.coverage(),
      rho: this.fitness.rho,
      nis: this.nis,
      sigma: positionSigma(this.belief.cov),
      margin: this.margin,
      error: this.error(),
      odomError: Math.hypot(this.deadReckon.x - this.truth.x, this.deadReckon.y - this.truth.y),
      novel: this.novelty.length,
      mode: this.mode.kind,
    });
    if (this.history.length > 1400) this.history.shift();
  }
}

/* -------------------------------------------------------------------------- */
/* Small helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * Q for one accumulated odometry increment, in the body frame.
 *
 * Deliberately larger than the sampler's own α's. A filter that trusts its
 * process model exactly is a filter whose covariance is a lower bound rather
 * than an estimate — Chapter 5's second Markov repair, applied where it is
 * cheapest to apply.
 */
function processNoise(delta: Pose2): Mat {
  const trans = Math.hypot(delta.x, delta.y);
  const rot = Math.abs(delta.theta);
  const sAlong = 0.11 * trans + 0.02 * rot + 2e-4;
  const sLat = 0.045 * trans + 0.02 * rot + 2e-4;
  const sTheta = 0.06 * rot + 0.035 * trans + 3e-4;
  return [
    [sAlong * sAlong, 0, 0],
    [0, sLat * sLat, 0],
    [0, 0, sTheta * sTheta],
  ];
}

/** Fraction of a world-frame cloud that has a map point within `r`. */
function mapAgreement(cloud: readonly Pt[], map: VoxelMap, r: number): number {
  if (cloud.length === 0 || map.size === 0) return 0;
  let hit = 0;
  for (const p of cloud) if (map.nearestIndex(p, r) >= 0) hit++;
  return hit / cloud.length;
}

/** Ω + λI, keeping a corridor match from claiming infinite along-wall certainty. */
function ridge(omega: Mat, lambda = 1.5, cap = 2400): Mat {
  const out = omega.map((row, i) => row.map((v, j) => (i === j ? v + lambda : v)));
  const peak = Math.max(out[0][0], out[1][1], out[2][2]);
  if (peak > cap) {
    const s = cap / peak;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i][j] *= s;
  }
  return out;
}

const symmetric = (m: Mat): Mat => m.map((row, i) => row.map((v, j) => (v + m[j][i]) / 2));

function quadForm(v: readonly number[], m: Mat): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) for (let j = 0; j < v.length; j++) s += v[i] * m[i][j] * v[j];
  return s;
}

/**
 * Which cells could Rusty ever reach? Flood fill the *true* world at the robot
 * radius, once, at construction.
 *
 * This is the denominator of the coverage number, and it is the clearest
 * example in the book of the simulator grading its own homework: no robot on
 * real hardware knows this set. The retrospective says what replaces it.
 */
function floodReachable(
  world: World,
  grid: OccupancyGrid,
  rRobot: number,
  start: Pose2,
): { reachable: Uint8Array; count: number } {
  const nx = grid.width;
  const ny = grid.height;
  const free = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const [x, y] = grid.cellCenter(i, j);
      if (distanceToWalls(world, x, y) > rRobot * 1.05) free[j * nx + i] = 1;
    }
  }
  const reachable = new Uint8Array(nx * ny);
  const [si, sj] = grid.worldToCell(start.x, start.y);
  const stack = [sj * nx + si];
  reachable[sj * nx + si] = 1;
  let count = 1;
  while (stack.length > 0) {
    const k = stack.pop() as number;
    const i = k % nx;
    const j = (k - i) / nx;
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as [number, number][]) {
      const ni = i + di;
      const nj = j + dj;
      if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
      const nk = nj * nx + ni;
      if (reachable[nk] || !free[nk]) continue;
      reachable[nk] = 1;
      count++;
      stack.push(nk);
    }
  }
  return { reachable, count };
}

/** A person, as four short wall segments the ray caster already understands. */
export function walkerSegments(w: Walker): Segment[] {
  const r = w.radius;
  const pts: [number, number][] = [
    [w.x + r, w.y],
    [w.x, w.y + r],
    [w.x - r, w.y],
    [w.x, w.y - r],
  ];
  return pts.map((p, i) => {
    const q = pts[(i + 1) % pts.length];
    return { x1: p[0], y1: p[1], x2: q[0], y2: q[1] };
  });
}

/* -------------------------------------------------------------------------- */
/* Headless mission, for tests and for the numbers quoted in the chapter        */
/* -------------------------------------------------------------------------- */

export interface MissionReport {
  seed: number;
  ticks: number;
  simTime: number;
  coverage: number;
  ate: number;
  odomError: number;
  contacts: number;
  finalEntropy: number;
  mode: Mode['kind'];
  events: StackEvent[];
}

/**
 * `run_mission` — the deterministic regression the chapter quotes.
 *
 * Same seed ⇒ same trajectory, same events, same numbers, in the browser and
 * in `cargo test`. That is the whole reason the scheduler is round-robin rather
 * than preemptive.
 */
export function runMission(cfg: Partial<MissionConfig> = {}, maxTicks = 2400): MissionReport {
  const stack = new AutonomyStack(cfg);
  let ate = 0;
  let n = 0;
  for (let i = 0; i < maxTicks; i++) {
    stack.step();
    if (i % 4 === 0) {
      ate += stack.error() ** 2;
      n++;
    }
    if (stack.mode.kind === 'Done') break;
  }
  return {
    seed: stack.cfg.seed,
    ticks: stack.tick,
    simTime: stack.time,
    coverage: stack.coverage(),
    ate: Math.sqrt(ate / Math.max(n, 1)),
    odomError: Math.hypot(stack.deadReckon.x - stack.truth.x, stack.deadReckon.y - stack.truth.y),
    contacts: stack.contacts,
    finalEntropy: stack.grid.entropy(),
    mode: stack.mode.kind,
    events: stack.events,
  };
}
