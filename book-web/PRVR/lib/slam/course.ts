/**
 * The Apartment landmark course — the one scenario every Chapter 14 widget runs.
 *
 * Rusty patrols the Apartment's corridor between two clusters of beacons, with
 * roughly three metres of featureless corridor in the middle. That gap is the
 * whole design: crossing it the filter has nothing but dead reckoning, so pose
 * uncertainty balloons, the far cluster gets mapped *through* that uncertainty,
 * and the return trip's re-sighting of the near cluster is a genuine loop
 * closure — a large, highly informative innovation arriving after a long
 * excursion. That is the moment the correlation web snaps taut.
 *
 * The simulator corrupts the truth with **exactly** the noise model the filter
 * assumes (`EkfSlam.processNoise`), and correspondences can be handed over for
 * free. So when the consistency instrument still says the filter is
 * overconfident, tuning is not the explanation. Linearization is.
 */

import { angleDiff, normalizeAngle, type Pose2 } from '../geom/se2';
import { Rng } from '../prob/rng';
import { APARTMENT, rayCast, type World } from '../sim/world';
import type { VelocityCmd } from '../models/motion';
import type { RangeBearingFeature } from '../models/sensor';
import { DEFAULT_SLAM_CONFIG, EkfSlam, type Association, type SlamConfig } from './ekf-slam';

export interface Beacon {
  id: number;
  x: number;
  y: number;
  /** Which end of the corridor this beacon belongs to — only used for labels. */
  cluster: 'west' | 'east';
}

/**
 * Eight beacons in two clusters. Two of them (ids 2 and 4) sit inside rooms and
 * are visible only through a doorway, which is what keeps the geometry from
 * degenerating into a straight line of points along the corridor.
 */
export const APARTMENT_COURSE: Beacon[] = [
  { id: 0, x: 0.55, y: 3.95, cluster: 'west' },
  { id: 1, x: 0.75, y: 4.85, cluster: 'west' },
  { id: 2, x: 2.05, y: 3.15, cluster: 'west' },
  { id: 3, x: 2.3, y: 4.88, cluster: 'west' },
  { id: 4, x: 10.05, y: 3.15, cluster: 'east' },
  { id: 5, x: 10.3, y: 4.88, cluster: 'east' },
  { id: 6, x: 11.45, y: 3.95, cluster: 'east' },
  { id: 7, x: 11.3, y: 4.85, cluster: 'east' },
];

export interface SensorParams {
  maxRange: number;
  sigmaR: number;
  sigmaPhi: number;
  /** Poisson-ish rate of spurious detections per step. */
  clutterRate: number;
}

export const COURSE_SENSOR: SensorParams = {
  maxRange: 2.2,
  sigmaR: 0.06,
  sigmaPhi: 0.03,
  clutterRate: 0,
};

export interface CourseParams {
  /** West turnaround, metres. */
  westX: number;
  /** East turnaround — the chapter's "loop length" knob. */
  eastX: number;
  laneY: number;
  speed: number;
  dt: number;
}

export const DEFAULT_COURSE: CourseParams = {
  westX: 1.4,
  eastX: 11.2,
  laneY: 4.4,
  speed: 0.55,
  dt: 0.25,
};

/** A landmark counts as re-found after this many steps out of sight. */
export const LOOP_CLOSURE_GAP = 25;

// ---------------------------------------------------------------------------
// Driving
// ---------------------------------------------------------------------------

/** A proportional heading controller aimed at the current turnaround point. */
export function patrolCommand(pose: Pose2, dir: 1 | -1, p: CourseParams): VelocityCmd {
  const tx = dir === 1 ? p.eastX : p.westX;
  const err = angleDiff(Math.atan2(p.laneY - pose.y, tx - pose.x), pose.theta);
  const omega = Math.max(-1.1, Math.min(1.1, 1.6 * err));
  // Slow to a crawl while the heading error is large, so the U-turn is an arc
  // the reader can watch rather than a teleport.
  const v = p.speed * (Math.abs(err) < 0.5 ? 1 : 0.25);
  return { v, omega, dt: p.dt };
}

export function nextDirection(pose: Pose2, dir: 1 | -1, p: CourseParams): 1 | -1 {
  if (dir === 1 && pose.x >= p.eastX - 0.15) return -1;
  if (dir === -1 && pose.x <= p.westX + 0.15) return 1;
  return dir;
}

/**
 * Move the *true* robot: the exact arc, then a body-frame Gaussian kick whose
 * covariance is precisely the R_t the filter will add. Along-track, cross-track
 * and heading errors are drawn independently in the body frame and rotated out,
 * which is what makes the world's noise and the filter's model the same object.
 */
export function stepTruth(pose: Pose2, u: VelocityCmd, cfg: SlamConfig, rng: Rng): Pose2 {
  const { v, omega, dt } = u;
  const th = pose.theta;
  const nt = th + omega * dt;
  let x: number;
  let y: number;
  if (Math.abs(omega) < 1e-6) {
    x = pose.x + v * Math.cos(th) * dt;
    y = pose.y + v * Math.sin(th) * dt;
  } else {
    const r = v / omega;
    x = pose.x - r * Math.sin(th) + r * Math.sin(nt);
    y = pose.y + r * Math.cos(th) - r * Math.cos(nt);
  }

  const d = Math.abs(v) * dt;
  const turn = Math.abs(omega) * dt;
  const ea = rng.normal(0, cfg.motion.alongTrack * d);
  const ec = rng.normal(0, cfg.motion.crossTrack * d);
  const et = rng.normal(0, cfg.motion.headingPerMetre * d + cfg.motion.headingPerRadian * turn);
  const c = Math.cos(th);
  const s = Math.sin(th);
  return {
    x: x + c * ea - s * ec,
    y: y + s * ea + c * ec,
    theta: normalizeAngle(nt + et),
  };
}

// ---------------------------------------------------------------------------
// Sensing
// ---------------------------------------------------------------------------

/** In range, and with an unobstructed line of sight — a wall really does hide a beacon. */
export function visibleBeacons(
  pose: Pose2,
  beacons: Beacon[],
  sensor: SensorParams,
  world: World = APARTMENT,
): Beacon[] {
  return beacons.filter((b) => {
    const d = Math.hypot(b.x - pose.x, b.y - pose.y);
    if (d > sensor.maxRange) return false;
    const angle = Math.atan2(b.y - pose.y, b.x - pose.x);
    return rayCast(world, pose.x, pose.y, angle, sensor.maxRange) > d - 0.06;
  });
}

/**
 * Noisy (r, φ, s) for every visible beacon, plus `clutterRate` spurious
 * detections drawn uniformly over the sensor's footprint. Clutter carries
 * `s = -1`: it is not a landmark, and nothing in the world will ever confirm it.
 */
export function observe(
  pose: Pose2,
  beacons: Beacon[],
  sensor: SensorParams,
  rng: Rng,
  world: World = APARTMENT,
): RangeBearingFeature[] {
  const out: RangeBearingFeature[] = visibleBeacons(pose, beacons, sensor, world).map((b) => {
    const dx = b.x - pose.x;
    const dy = b.y - pose.y;
    return {
      r: Math.max(0.05, Math.hypot(dx, dy) + rng.normal(0, sensor.sigmaR)),
      phi: normalizeAngle(Math.atan2(dy, dx) - pose.theta + rng.normal(0, sensor.sigmaPhi)),
      s: b.id,
    };
  });

  let budget = sensor.clutterRate;
  while (budget > 0) {
    if (rng.next() < Math.min(budget, 1)) {
      out.push({
        r: rng.uniform(0.6, sensor.maxRange),
        phi: rng.uniform(-Math.PI, Math.PI),
        s: -1,
      });
    }
    budget -= 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The scenario, as one object
// ---------------------------------------------------------------------------

export interface CourseOptions {
  seed: number;
  params?: Partial<CourseParams>;
  sensor?: Partial<SensorParams>;
  config?: Partial<SlamConfig>;
  /** Hand the filter the correspondences (Table 10.1) or make it guess (Table 10.2). */
  knownCorrespondence?: boolean;
}

export interface StepResult {
  u: VelocityCmd;
  features: RangeBearingFeature[];
  associations: Association[];
  /** True beacon ids in view this step. */
  visible: number[];
  /** Slots matched after a long absence — the operational loop closure. */
  closures: number[];
  nees: number;
  positionError: number;
  headingError: number;
}

/** The initial pose prior. Small but non-zero: a floor the map can never beat. */
export const INITIAL_POSE_COV = [
  [0.0025, 0, 0],
  [0, 0.0025, 0],
  [0, 0, 0.0004],
];

export class CourseSim {
  truth: Pose2;
  dir: 1 | -1 = 1;
  laps = 0;
  t = 0;
  filter: EkfSlam;
  rng: Rng;
  params: CourseParams;
  sensor: SensorParams;
  beacons: Beacon[];
  known: boolean;
  /** Step index at which each slot was last matched — the loop-closure clock. */
  lastSeen: number[] = [];
  /** True trajectory and filtered trajectory, for the map panel. */
  truthPath: { x: number; y: number }[] = [];
  estimatePath: { x: number; y: number }[] = [];

  constructor(opts: CourseOptions) {
    this.params = { ...DEFAULT_COURSE, ...opts.params };
    this.sensor = { ...COURSE_SENSOR, ...opts.sensor };
    this.known = opts.knownCorrespondence ?? true;
    this.beacons = APARTMENT_COURSE;
    this.rng = new Rng(opts.seed);
    const cfg: SlamConfig = {
      ...DEFAULT_SLAM_CONFIG,
      ...opts.config,
      motion: { ...DEFAULT_SLAM_CONFIG.motion, ...(opts.config?.motion ?? {}) },
      sigmaR: opts.config?.sigmaR ?? this.sensor.sigmaR,
      sigmaPhi: opts.config?.sigmaPhi ?? this.sensor.sigmaPhi,
    };
    const nominal: Pose2 = { x: this.params.westX, y: this.params.laneY, theta: 0 };
    this.filter = new EkfSlam(nominal, INITIAL_POSE_COV.map((r) => r.slice()), cfg);
    // The truth is *drawn from* the prior, not set equal to its mean. A prior
    // the world does not honour would bias every consistency number downwards
    // and let an overconfident filter hide behind a lucky start.
    this.truth = {
      x: nominal.x + this.rng.normal(0, Math.sqrt(INITIAL_POSE_COV[0][0])),
      y: nominal.y + this.rng.normal(0, Math.sqrt(INITIAL_POSE_COV[1][1])),
      theta: normalizeAngle(nominal.theta + this.rng.normal(0, Math.sqrt(INITIAL_POSE_COV[2][2]))),
    };
    this.truthPath.push({ x: this.truth.x, y: this.truth.y });
    this.estimatePath.push({ x: this.truth.x, y: this.truth.y });
  }

  /** One control cycle: drive, predict, sense, correct, manage the map. */
  step(): StepResult {
    const u = patrolCommand(this.truth, this.dir, this.params);
    const prevDir = this.dir;

    this.truth = stepTruth(this.truth, u, this.filter.cfg, this.rng);
    this.dir = nextDirection(this.truth, this.dir, this.params);
    if (prevDir === -1 && this.dir === 1) this.laps += 1;

    this.filter.predict(u);

    const features = observe(this.truth, this.beacons, this.sensor, this.rng);
    const visible = features.filter((f) => (f.s ?? -1) >= 0).map((f) => f.s as number);

    const before = this.filter.count;
    const associations = this.known
      ? this.filter.correctKnown(features.filter((f) => (f.s ?? -1) >= 0))
      : this.filter.correct(features, this.t);
    while (this.lastSeen.length < this.filter.count) this.lastSeen.push(this.t);
    if (this.filter.count < before) this.lastSeen.length = this.filter.count;

    const closures: number[] = [];
    const matched: number[] = [];
    for (const a of associations) {
      if (a.kind !== 'matched') continue;
      matched.push(a.landmark);
      if (this.t - this.lastSeen[a.landmark] > LOOP_CLOSURE_GAP) closures.push(a.landmark);
      this.lastSeen[a.landmark] = this.t;
    }

    if (!this.known) {
      this.filter.updateExistence(this.expectedSlots(), matched);
      this.filter.pruneMap();
      this.filter.expireCandidates(this.t);
      while (this.lastSeen.length < this.filter.count) this.lastSeen.push(this.t);
      this.lastSeen.length = this.filter.count;
    }

    this.t += 1;
    this.truthPath.push({ x: this.truth.x, y: this.truth.y });
    const est = this.filter.pose();
    this.estimatePath.push({ x: est.x, y: est.y });
    if (this.truthPath.length > 1200) {
      this.truthPath.shift();
      this.estimatePath.shift();
    }

    return {
      u,
      features,
      associations,
      visible,
      closures,
      nees: this.filter.nees(this.truth),
      positionError: Math.hypot(est.x - this.truth.x, est.y - this.truth.y),
      headingError: Math.abs(angleDiff(est.theta, this.truth.theta)),
    };
  }

  /**
   * Slots the filter *believes* are in view right now. Landmarks it expects and
   * does not see lose existence evidence — the only way a phantom ever dies.
   */
  expectedSlots(): number[] {
    const pose = this.filter.pose();
    const out: number[] = [];
    for (let j = 0; j < this.filter.count; j++) {
      const [mx, my] = this.filter.landmarkMean(j);
      const d = Math.hypot(mx - pose.x, my - pose.y);
      if (d > this.sensor.maxRange - 0.2) continue;
      const angle = Math.atan2(my - pose.y, mx - pose.x);
      if (rayCast(APARTMENT, pose.x, pose.y, angle, this.sensor.maxRange) > d - 0.06) out.push(j);
    }
    return out;
  }

  /** Ground-truth position of the beacon a slot is tracking, if it is known. */
  truthFor(slot: number): Beacon | undefined {
    const label = this.filter.labels[slot];
    return this.beacons.find((b) => b.id === label);
  }
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

export interface MonteCarloResult {
  /** Mean NEES across runs, per step. */
  meanNees: number[];
  /** Mean absolute position error across runs, per step. */
  meanError: number[];
  /** Reported pose-position σ, averaged across runs. */
  meanSigma: number[];
  runs: number;
  steps: number;
}

/**
 * The honest instrument. A single NEES trace is a χ²(3) draw — it bounces
 * everywhere, and any story can be told with one lucky seed. Averaging R
 * independent runs shrinks the acceptance band by √R and turns "this run looks
 * bad" into a hypothesis test.
 */
export function monteCarloNees(
  runs: number,
  steps: number,
  opts: Omit<CourseOptions, 'seed'> & { seed?: number },
): MonteCarloResult {
  const meanNees = new Array<number>(steps).fill(0);
  const meanError = new Array<number>(steps).fill(0);
  const meanSigma = new Array<number>(steps).fill(0);
  const base = opts.seed ?? 1;

  for (let r = 0; r < runs; r++) {
    const sim = new CourseSim({ ...opts, seed: base + 1013 * r });
    for (let k = 0; k < steps; k++) {
      const res = sim.step();
      meanNees[k] += res.nees;
      meanError[k] += res.positionError;
      const P = sim.filter.poseCov();
      meanSigma[k] += Math.sqrt(Math.max(0.5 * (P[0][0] + P[1][1]), 0));
    }
  }
  for (let k = 0; k < steps; k++) {
    meanNees[k] /= runs;
    meanError[k] /= runs;
    meanSigma[k] /= runs;
  }
  return { meanNees, meanError, meanSigma, runs, steps };
}
