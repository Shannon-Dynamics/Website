/**
 * RustSLAM-2D — the book's first complete SLAM system.
 *
 * Two loops, taking turns. The **front end** runs every scan: predict with
 * odometry, register the sweep against a small sliding local map with ICP, push
 * the result into the map. It is fast, greedy, and drifts. The **back end**
 * runs only when something interesting happens: a keyframe is added as a node,
 * a revisit is proposed and verified, and Chapter 15's Gauss–Newton rewrites
 * every pose in the graph at once. It is slow, global, and honest.
 *
 * That division — not any single algorithm — is what SLAM actually is, and it
 * is why this file is short: `icp.ts` supplies the front end, `posegraph.ts`
 * the back end, and the code below is mostly bookkeeping about when to call
 * which.
 *
 * Rust counterpart: `crates/ch16_slam2d/src/slam.rs`.
 */

import {
  angleDiff,
  between,
  compose,
  inverse,
  normalizeAngle,
  se2Log,
  type Pose2,
} from '../geom/se2';
import { sampleMotionModelVelocity, type MotionAlphas } from '../models/motion';
import type { Mat } from '../prob/linalg';
import { Rng } from '../prob/rng';
import {
  beamAngles,
  diffDriveStep,
  simulateScan,
  type ScanParams,
  type World,
} from '../sim/world';
import { estimateNormals, scanToCloud, transformCloud, voxelDownsample, VoxelMap, type Pt } from './cloud';
import { AdaptiveThreshold, icp, icpInformation, type IcpConfig, type IcpResult } from './icp';
import { chi2Gate, detectLoopCandidates, PoseGraph, CHI2_3DOF_95 } from './posegraph';

export interface Slam2dConfig {
  scan: ScanParams;
  /** Voxel size for downsampling the incoming sweep. */
  voxel: number;
  /** Voxel size of the local map's hash grid. */
  mapCell: number;
  /** How many recent sweeps the local map keeps. Small = honest drift. */
  localWindow: number;
  /** Distance / rotation between graph nodes. */
  keyframeDist: number;
  keyframeAngle: number;
  loopRadius: number;
  loopMinGap: number;
  /** Reject a verified loop whose ICP RMSE exceeds this. */
  loopMaxRmse: number;
  loopMinInliers: number;
  icp: Partial<IcpConfig>;
  alphas: MotionAlphas;
  vMax: number;
  wMax: number;
  dt: number;
}

export const DEFAULT_SLAM_CONFIG: Slam2dConfig = {
  scan: { nBeams: 120, fov: 2 * Math.PI, maxRange: 6, sigma: 0.05 },
  voxel: 0.12,
  mapCell: 0.25,
  localWindow: 6,
  keyframeDist: 0.4,
  keyframeAngle: 0.35,
  loopRadius: 2.0,
  loopMinGap: 22,
  loopMaxRmse: 0.16,
  loopMinInliers: 25,
  icp: { maxIters: 15, tau: 0.5, variant: 'point-to-point' },
  alphas: [0.06, 0.02, 0.05, 0.02, 0.02, 0.01],
  vMax: 0.62,
  wMax: 1.1,
  dt: 0.25,
};

export interface Keyframe {
  node: number;
  /** Sensor-frame points — re-rendered at whatever pose the graph now believes. */
  cloud: Pt[];
  ranges: number[];
  truth: Pose2;
}

export interface LoopEvent {
  from: number;
  to: number;
  chi2: number;
  rmse: number;
  inliers: number;
  accepted: boolean;
  /** ATE before → after the optimization this loop triggered. */
  ateBefore?: number;
  ateAfter?: number;
  maxShift?: number;
}

export interface SlamReport {
  t: number;
  truth: Pose2;
  /** Raw wheel odometry, integrated without any scan matching. */
  deadReckon: Pose2;
  estimate: Pose2;
  icp: IcpResult;
  tau: number;
  loop?: LoopEvent;
  keyframeAdded: boolean;
  finished: boolean;
  ate: number;
  /** RMS disagreement between the two passes over the corridor. */
  consistency: number;
  odomError: number;
}

/** Waypoint controller: turn toward the target, then drive at it. */
function driveTo(pose: Pose2, target: Pt, vMax: number, wMax: number) {
  const dx = target[0] - pose.x;
  const dy = target[1] - pose.y;
  const d = Math.hypot(dx, dy);
  const err = angleDiff(Math.atan2(dy, dx), pose.theta);
  const omega = Math.max(-wMax, Math.min(wMax, 2.2 * err));
  const v = Math.abs(err) > 0.5 ? 0 : vMax * Math.min(1, d / 0.6);
  return { v, omega, d };
}

/**
 * The lap: down the Apartment's corridor and back.
 *
 * The Apartment has no topological cycle — every room is a dead end off one
 * corridor — so Rusty revisits rather than circles. That is all a loop closure
 * has ever needed: two poses that see the same place, far apart in time.
 */
export const CORRIDOR_LAP: Pt[] = [
  [10.6, 4.42],
  [1.35, 4.42],
];

export const LAP_START: Pose2 = { x: 1.35, y: 4.42, theta: 0 };

export class Slam2d {
  readonly cfg: Slam2dConfig;
  readonly world: World;
  readonly graph = new PoseGraph();
  readonly keyframes: Keyframe[] = [];
  readonly angles: number[];

  truth: Pose2;
  deadReckon: Pose2;
  estimate: Pose2;
  t = 0;
  finished = false;
  /** Node index at which the lap reversed — the split `mapConsistency` uses. */
  turnNode = -1;

  private rng: Rng;
  private waypoint = 0;
  private tau: AdaptiveThreshold;
  private window: { cloud: Pt[]; pose: Pose2 }[] = [];
  private localMap = new VoxelMap(0.25, 4);
  private lastKeyframePose: Pose2;
  private lastLoopAttempt = -999;
  readonly events: LoopEvent[] = [];

  constructor(world: World, seed = 0xc0ffee, cfg: Partial<Slam2dConfig> = {}) {
    this.cfg = { ...DEFAULT_SLAM_CONFIG, ...cfg };
    this.world = world;
    this.rng = new Rng(seed);
    this.angles = beamAngles(this.cfg.scan);
    this.truth = { ...LAP_START };
    this.deadReckon = { ...LAP_START };
    this.estimate = { ...LAP_START };
    this.lastKeyframePose = { ...LAP_START };
    this.tau = new AdaptiveThreshold(this.cfg.icp.tau ?? 0.5, 0.03, this.cfg.scan.maxRange);
    this.localMap = new VoxelMap(this.cfg.mapCell, 4);

    // Node 0 is fixed: it defines the world frame the whole map is drawn in.
    const node = this.graph.addNode(this.estimate, true);
    const cloud = this.sense(this.truth);
    this.keyframes.push({ node, cloud: cloud.pts, ranges: cloud.ranges, truth: { ...this.truth } });
    this.pushWindow(cloud.pts, this.estimate);
  }

  private sense(pose: Pose2): { pts: Pt[]; ranges: number[] } {
    const ranges = simulateScan(this.world, pose, this.cfg.scan, this.rng);
    const cloud = scanToCloud(ranges, this.angles, this.cfg.scan.maxRange, this.t);
    return { pts: voxelDownsample(cloud.points, this.cfg.voxel), ranges };
  }

  private pushWindow(cloud: Pt[], pose: Pose2): void {
    this.window.push({ cloud, pose: { ...pose } });
    if (this.window.length > this.cfg.localWindow) this.window.shift();
    this.rebuildLocalMap();
  }

  private rebuildLocalMap(): void {
    const map = new VoxelMap(this.cfg.mapCell, 4);
    for (const w of this.window) {
      const pts = transformCloud(w.pose, w.cloud);
      map.insert(pts, estimateNormals(pts, 0.3));
    }
    this.localMap = map;
  }

  /** World points of every keyframe, drawn at the pose the graph currently holds. */
  mapPoints(): Pt[] {
    const out: Pt[] = [];
    for (const kf of this.keyframes) {
      const pose = this.graph.nodes[kf.node].pose;
      for (const p of transformCloud(pose, kf.cloud)) out.push(p);
    }
    return out;
  }

  /**
   * How badly the map disagrees with itself, in metres.
   *
   * For every point contributed by a keyframe recorded *after* the turnaround,
   * the distance to the nearest point contributed *before* it. Zero means the
   * two passes over the corridor drew the same walls; a large value means the
   * map is double-walled, which is the failure a reader can see without knowing
   * the ground truth — and therefore the one a real robot can act on.
   */
  mapConsistency(): number {
    const split = this.turnNode;
    if (split < 0) return 0;
    const early = new VoxelMap(0.2, 12);
    for (const kf of this.keyframes) {
      if (kf.node > split) continue;
      early.insert(transformCloud(this.graph.nodes[kf.node].pose, kf.cloud));
    }
    let s = 0;
    let n = 0;
    for (const kf of this.keyframes) {
      if (kf.node <= split) continue;
      for (const p of transformCloud(this.graph.nodes[kf.node].pose, kf.cloud)) {
        const idx = early.nearestIndex(p, 1.5);
        if (idx < 0) continue;
        const q = early.pts[idx];
        s += (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2;
        n += 1;
      }
    }
    return n > 0 ? Math.sqrt(s / n) : 0;
  }

  /** RMSE of the keyframe positions against ground truth, in metres. */
  ate(): number {
    if (this.keyframes.length === 0) return 0;
    let s = 0;
    for (const kf of this.keyframes) {
      const p = this.graph.nodes[kf.node].pose;
      s += (p.x - kf.truth.x) ** 2 + (p.y - kf.truth.y) ** 2;
    }
    return Math.sqrt(s / this.keyframes.length);
  }

  /** One SLAM tick: sense, register, maybe add a node, maybe close a loop. */
  step(): SlamReport {
    const { cfg } = this;

    // --- ground truth and odometry --------------------------------------
    const target = CORRIDOR_LAP[Math.min(this.waypoint, CORRIDOR_LAP.length - 1)];
    const { v, omega, d } = driveTo(this.truth, target, cfg.vMax, cfg.wMax);
    if (d < 0.3) {
      if (this.waypoint === 0) this.turnNode = this.keyframes[this.keyframes.length - 1].node;
      this.waypoint += 1;
      if (this.waypoint >= CORRIDOR_LAP.length) this.finished = true;
    }

    const prevTruth = { ...this.truth };
    this.truth = diffDriveStep(this.truth, v, omega, cfg.dt);
    const noisy = sampleMotionModelVelocity({ v, omega, dt: cfg.dt }, prevTruth, cfg.alphas, this.rng);
    const odomDelta = between(prevTruth, noisy);
    this.deadReckon = compose(this.deadReckon, odomDelta);
    const predicted = compose(this.estimate, odomDelta);

    // --- front end: register the sweep ----------------------------------
    const { pts, ranges } = this.sense(this.truth);
    const result = icp(pts, this.localMap, predicted, { ...cfg.icp, tau: this.tau.tau });
    this.estimate = result.inliers >= 8 ? result.pose : predicted;
    this.tau.update(between(predicted, this.estimate));
    this.pushWindow(pts, this.estimate);
    this.t += cfg.dt;

    // --- back end: keyframe, loop, optimize -----------------------------
    const rel = between(this.lastKeyframePose, this.estimate);
    let keyframeAdded = false;
    let loop: LoopEvent | undefined;

    if (Math.hypot(rel.x, rel.y) >= cfg.keyframeDist || Math.abs(rel.theta) >= cfg.keyframeAngle) {
      const prevNode = this.keyframes[this.keyframes.length - 1].node;
      const node = this.graph.addNode(this.estimate);
      const omegaOdom = icpInformation(
        result.trace[result.trace.length - 1].pairs,
        Math.max(cfg.scan.sigma, 0.01),
        12,
        this.graph.nodes[prevNode].pose.theta,
      );
      this.graph.addEdge(prevNode, node, rel, regularizeInformation(omegaOdom), 'odometry');
      this.keyframes.push({ node, cloud: pts, ranges, truth: { ...this.truth } });
      this.lastKeyframePose = { ...this.estimate };
      keyframeAdded = true;

      loop = this.tryCloseLoop(node);
    }

    return {
      t: this.t,
      truth: { ...this.truth },
      deadReckon: { ...this.deadReckon },
      estimate: { ...this.estimate },
      icp: result,
      tau: this.tau.tau,
      loop,
      keyframeAdded,
      finished: this.finished,
      ate: this.ate(),
      consistency: this.mapConsistency(),
      odomError: Math.hypot(this.deadReckon.x - this.truth.x, this.deadReckon.y - this.truth.y),
    };
  }

  /**
   * `detect_loop_candidates` → `verify_loop` → optimize.
   *
   * Detection is cheap and wrong often; verification is expensive and has to be
   * right. The gate is geometric (does an ICP run from the candidate actually
   * converge with a low residual and enough support?) *and* statistical (is the
   * resulting relative pose consistent with what the graph already believes,
   * at the 95% level of χ²₃?). A false positive here does not degrade the map,
   * it destroys it.
   */
  private tryCloseLoop(node: number): LoopEvent | undefined {
    if (node - this.lastLoopAttempt < 4) return undefined;
    this.lastLoopAttempt = node;
    const candidates = detectLoopCandidates(this.graph, node, this.cfg.loopRadius, this.cfg.loopMinGap);
    if (candidates.length === 0) return undefined;

    const here = this.graph.nodes[node].pose;
    candidates.sort(
      (a, b) =>
        Math.hypot(this.graph.nodes[a].pose.x - here.x, this.graph.nodes[a].pose.y - here.y) -
        Math.hypot(this.graph.nodes[b].pose.x - here.x, this.graph.nodes[b].pose.y - here.y),
    );
    const target = candidates[0];

    // A submap: the candidate keyframe and its neighbours, in the candidate's
    // own frame. One scan is too thin a target for a confident match.
    const submap = new VoxelMap(this.cfg.mapCell, 6);
    const base = this.graph.nodes[target].pose;
    for (const kf of this.keyframes) {
      if (Math.abs(kf.node - target) > 3) continue;
      const world = transformCloud(this.graph.nodes[kf.node].pose, kf.cloud);
      const local = transformCloud(inverse(base), world);
      submap.insert(local, estimateNormals(local, 0.3));
    }

    const current = this.keyframes.find((k) => k.node === node);
    if (!current) return undefined;

    const init = between(base, here);
    const match = icp(current.cloud, submap, init, {
      ...this.cfg.icp,
      maxIters: 30,
      tau: Math.max(this.cfg.loopRadius * 0.5, 0.6),
    });

    const omega = regularizeInformation(
      icpInformation(match.trace[match.trace.length - 1].pairs, Math.max(this.cfg.scan.sigma, 0.01), 25, base.theta),
    );
    // The innovation: how far the verified match moved the relative pose the
    // graph already believed in. Gate it against what the graph *could* have
    // been wrong by over `node − target` edges of odometry.
    const innovation = se2Log(between(init, match.pose));
    const gateOmega = pathUncertainty(node - target);
    const chi2 = mahalanobis(innovation, gateOmega);
    const gatePassed = chi2Gate(innovation, gateOmega, CHI2_3DOF_95);
    const geometryOk = match.rmse <= this.cfg.loopMaxRmse && match.inliers >= this.cfg.loopMinInliers;

    const event: LoopEvent = {
      from: node,
      to: target,
      chi2,
      rmse: match.rmse,
      inliers: match.inliers,
      accepted: geometryOk && gatePassed,
    };

    if (event.accepted) {
      event.ateBefore = this.ate();
      const oldLast = { ...this.graph.nodes[node].pose };
      this.graph.addEdge(target, node, match.pose, omega, 'loop');
      const report = this.graph.optimize(15, 1e-7, 12);
      event.ateAfter = this.ate();
      event.maxShift = report.maxShift;

      // The front end lives in the world frame, so it has to be told that the
      // world frame just moved under it.
      const correction = compose(this.graph.nodes[node].pose, inverse(oldLast));
      this.estimate = compose(correction, this.estimate);
      this.lastKeyframePose = { ...this.graph.nodes[node].pose };
      for (const w of this.window) w.pose = compose(correction, w.pose);
      this.rebuildLocalMap();
    }

    this.events.push(event);
    return event;
  }
}

function mahalanobis(r: [number, number, number], omega: Mat): number {
  let s = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += r[i] * omega[i][j] * r[j];
  return s;
}

/**
 * Keep an ICP-derived information matrix usable *without* breaking it.
 *
 * Along a featureless corridor JᵀJ is genuinely rank deficient, and a graph fed
 * a singular Ω will happily slide the whole trajectory along the wall. The fix
 * is a **ridge**, Ω + λI, not a clamp on the diagonal: clamping entries one at
 * a time can leave a matrix that is no longer positive semi-definite, and a
 * pose graph with an indefinite Ω does not converge slowly, it explodes. The
 * ridge says the honest thing — "this axis is constrained only by the motion
 * prior" — and the trailing rescale caps how much any single ICP match is
 * allowed to shout, which is the standing correction for ICP's belief that
 * every LiDAR beam is independent evidence.
 */
function regularizeInformation(omega: Mat, ridge = 1.0, cap = 2500): Mat {
  const out = omega.map((row, i) => row.map((v, j) => (i === j ? v + ridge : v)));
  const peak = Math.max(out[0][0], out[1][1], out[2][2]);
  if (peak > cap) {
    const s = cap / peak;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i][j] *= s;
  }
  return out;
}

/**
 * How uncertain is the graph about the relative pose between two nodes?
 *
 * Exactly: recover the joint marginal from the information matrix, which is a
 * sparse back-substitution ([Chapter 15](/chapters/ch15-factor-graphs)).
 * Cheaply and good enough for a gate: compound one edge's covariance along the
 * `gap` edges that separate the two nodes, so the ellipse the gate uses grows
 * with how far the robot drove between the two views. That growth is the whole
 * point — a candidate 40 nodes back must clear a much wider bar than one 4
 * nodes back, because 40 nodes of odometry could have put the robot anywhere.
 */
function pathUncertainty(gap: number, sigmaStep = 0.045, sigmaTheta = 0.012): Mat {
  const n = Math.max(gap, 1);
  const sd = sigmaStep * Math.sqrt(n);
  const st = sigmaTheta * Math.sqrt(n);
  return [
    [1 / (sd * sd), 0, 0],
    [0, 1 / (sd * sd), 0],
    [0, 0, 1 / (st * st)],
  ];
}

/** Angle helper re-exported so widgets need not reach into geom/se2 twice. */
export { normalizeAngle };
