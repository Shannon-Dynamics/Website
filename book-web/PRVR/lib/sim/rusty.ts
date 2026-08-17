/**
 * Rusty: the book's differential-drive rover, its wheel encoders, and its LiDAR.
 *
 * This module is the *generative* side of the book — the physics the simulator
 * actually executes. It is deliberately **not** the same parametrization as the
 * probabilistic motion models of Chapter 9 (Thrun's α₁…α₆) or the four-way beam
 * mixture of Chapter 10. Those are *inference* models that get fitted to data
 * produced here, and the gap between the two is a lesson rather than an
 * oversight.
 *
 * Faithful port of `crates/sim/src/{robot,encoders,lidar}.rs`. Three invariants
 * hold in both implementations:
 *
 *   1. With `slipStd = 0`, `radiusBiasRight = 0` and infinite encoder
 *      resolution, dead reckoning equals ground truth exactly — the integrator
 *      and the odometry formula are the same map, run forwards and backwards.
 *   2. Integration is exact: a constant wheel pair over Δt traces a
 *      constant-curvature arc, and `boxplus` with `se2Exp` *is* that arc. No
 *      small-angle approximation appears anywhere.
 *   3. Every stochastic quantity comes from an explicit seeded `Rng` passed in
 *      by the caller. Slip and LiDAR noise must draw from *different* streams —
 *      `crates/sim` splits them from one seed in `run.rs`, and here it is the
 *      widget's job — so that adding a sensor never perturbs an existing replay.
 */

import { boxminus, boxplus, type Pose2, type Twist2 } from '../geom/se2';
import type { Rng } from '../prob/rng';
import { collides, rayCast, type World } from './world';

// ---------------------------------------------------------------------------
// The chassis
// ---------------------------------------------------------------------------

export interface RobotParams {
  /** Wheel radius r, metres. */
  wheelRadius: number;
  /** Track width ℓ — the distance between the two wheel contact points, metres. */
  track: number;
  /**
   * Body radius, metres. Used for drawing here and for the swept-disc collision
   * test in `crates/sim`; the browser port collides as a point against wall
   * segments, which is cheaper and close enough at 0.11 m in a 1.2 m corridor.
   */
  bodyRadius: number;
  /** Encoder resolution N, counts per wheel revolution. */
  ticksPerRev: number;
  /**
   * Per-wheel multiplicative slip σ. Ground travel is wheel travel × (1 + ε)
   * with ε ~ 𝒩(0, σ²): the wheel turns, the floor disagrees about how far that
   * got you. Stochastic, zero-mean, and *not* removable by calibration.
   */
  slipStd: number;
  /**
   * Systematic error: the right wheel's true effective radius is
   * r(1 + δ) while odometry keeps believing it is r. Borenstein & Feng's
   * "unequal wheel diameters". Biased, and no amount of filtering removes it.
   */
  radiusBiasRight: number;
}

/** Rusty's nominal geometry — a TurtleBot-class rover. */
export const RUSTY: RobotParams = {
  wheelRadius: 0.033,
  track: 0.16,
  bodyRadius: 0.11,
  ticksPerRev: 4096,
  slipStd: 0.02,
  radiusBiasRight: 0,
};

/** A commanded body twist: forward speed and yaw rate. */
export interface Twist {
  v: number;
  omega: number;
}

/**
 * Inverse kinematics (v, ω) → (ω_L, ω_R), the map the teleop keys go through.
 *
 *   ω_L = (v − ωℓ/2)/r,   ω_R = (v + ωℓ/2)/r
 *
 * Note there is no third equation: a diff-drive has two actuators and a
 * three-dimensional configuration space, which is the whole nonholonomic story.
 */
export function wheelSpeeds(u: Twist, p: RobotParams): [number, number] {
  const half = (u.omega * p.track) / 2;
  return [(u.v - half) / p.wheelRadius, (u.v + half) / p.wheelRadius];
}

/** Forward kinematics (ω_L, ω_R) → (v, ω). Inverse of {@link wheelSpeeds}. */
export function bodyTwist(omegaL: number, omegaR: number, p: RobotParams): Twist {
  return {
    v: (p.wheelRadius * (omegaR + omegaL)) / 2,
    omega: (p.wheelRadius * (omegaR - omegaL)) / p.track,
  };
}

/** Instantaneous centre of rotation radius R = v/ω. Infinite when driving straight. */
export function icrRadius(u: Twist): number {
  return Math.abs(u.omega) < 1e-12 ? Infinity : u.v / u.omega;
}

// ---------------------------------------------------------------------------
// diff_drive_step
// ---------------------------------------------------------------------------

/** Everything the simulator carries for the chassis between ticks. */
export interface RustyState {
  /** Ground truth — the quantity the robot itself never observes. */
  pose: Pose2;
  /** Cumulative wheel rotation in radians, [left, right]. What encoders count. */
  wheelAngles: [number, number];
}

export interface DriveOutcome extends RustyState {
  /** True when the move was refused because it would cross a wall. */
  blocked: boolean;
  /** The body twist actually realised on the floor, after slip. */
  realized: Twist;
}

/**
 * `diff_drive_step` — one tick of Rusty's physics.
 *
 * 1. commanded twist → nominal wheel speeds;
 * 2. **the wheels always turn by the commanded amount** — that is what the
 *    encoders will report, and it is why odometry is a *proprioceptive* sensor
 *    that cannot see the floor;
 * 3. the ground travel of each wheel is its rotation times its *effective*
 *    radius times a slip factor (1 + ε);
 * 4. the realised body twist is integrated exactly as x ⊞ (ṽΔt, 0, ω̃Δt).
 *
 * On collision the pose is held but the wheels still advance, which is exactly
 * what a stuck robot's encoders do — and the most vivid demonstration in the
 * chapter that odometry is not a position sensor.
 */
export function diffDriveSlipStep(
  s: RustyState,
  u: Twist,
  dt: number,
  world: World | null,
  p: RobotParams,
  rng: Rng,
): DriveOutcome {
  const [wl, wr] = wheelSpeeds(u, p);

  // What the encoders will count: the wheel turned, full stop.
  const wheelAngles: [number, number] = [s.wheelAngles[0] + wl * dt, s.wheelAngles[1] + wr * dt];

  // What the floor delivered. Two independent error sources, on purpose:
  // radiusBiasRight is a bias (systematic), slipStd is zero-mean (stochastic).
  const epsL = p.slipStd > 0 ? rng.normal(0, p.slipStd) : 0;
  const epsR = p.slipStd > 0 ? rng.normal(0, p.slipStd) : 0;
  const groundL = p.wheelRadius * wl * dt * (1 + epsL);
  const groundR = p.wheelRadius * (1 + p.radiusBiasRight) * wr * dt * (1 + epsR);

  const ds = (groundR + groundL) / 2;
  const dtheta = (groundR - groundL) / p.track;
  const realized: Twist = { v: dt > 0 ? ds / dt : 0, omega: dt > 0 ? dtheta / dt : 0 };

  // Exact arc integration. `boxplus` is p ∘ exp(τ) — Chapter 3, cashed in.
  const next = boxplus(s.pose, [ds, 0, dtheta]);

  if (world && collides(world, s.pose, next)) {
    return { pose: s.pose, wheelAngles, blocked: true, realized };
  }
  return { pose: next, wheelAngles, blocked: false, realized };
}

// ---------------------------------------------------------------------------
// Encoders and odometry
// ---------------------------------------------------------------------------

/** Cumulative quadrature counts, one integer per wheel. */
export interface EncoderTicks {
  left: number;
  right: number;
}

/**
 * `encoders_observe` — quantize the wheel angles to integer counts.
 *
 * This is the only place the simulator discards information for free: the
 * residual angle inside one count is gone, bounding the per-wheel distance
 * error at ±πr/N (about 25 µm for Rusty).
 */
export function encoderTicks(wheelAngles: [number, number], p: RobotParams): EncoderTicks {
  const perRad = p.ticksPerRev / (2 * Math.PI);
  return {
    left: Math.round(wheelAngles[0] * perRad),
    right: Math.round(wheelAngles[1] * perRad),
  };
}

/** Metres of wheel travel per encoder count: λ = 2πr/N. */
export function tickLength(p: RobotParams): number {
  return (2 * Math.PI * p.wheelRadius) / p.ticksPerRev;
}

/**
 * `odometry_delta` — encoder counts to a tangent vector, ready for ⊞.
 *
 *   Δs_i = λ · Δticks_i,   Δs = ½(Δs_L + Δs_R),   Δθ = (Δs_R − Δs_L)/ℓ
 *
 * The returned twist is (Δs, 0, Δθ): the lateral component is *structurally*
 * zero, which is the nonholonomic constraint expressed in the Lie algebra.
 */
export function odometryDelta(prev: EncoderTicks, cur: EncoderTicks, p: RobotParams): Twist2 {
  const lambda = tickLength(p);
  const dsL = lambda * (cur.left - prev.left);
  const dsR = lambda * (cur.right - prev.right);
  return [(dsR + dsL) / 2, 0, (dsR - dsL) / p.track];
}

/** Apply an odometry increment to a dead-reckoned pose: x̂ ⊞ τ. */
export function integrateOdometry(estimate: Pose2, tau: Twist2): Pose2 {
  return boxplus(estimate, tau);
}

/**
 * Dead-reckoning error, reported the way the book always reports pose error:
 * the tangent vector x̂ ⊟ x, split into a translation magnitude and a heading.
 * Subtracting angles would be wrong at the wrap, and averaging them worse.
 */
export function driftError(estimate: Pose2, truth: Pose2): { position: number; heading: number } {
  const tau = boxminus(estimate, truth);
  return { position: Math.hypot(tau[0], tau[1]), heading: Math.abs(tau[2]) };
}

/**
 * The exact chord of a constant-curvature arc, in closed form.
 *
 * For an arc of length Δs turning through Δθ, the displacement is
 * `Δs · sinc(Δθ/2)` metres in the direction `Δθ/2` — the arc's *half* heading
 * change. Written out this way it is obvious that the "midpoint" heuristic gets
 * the direction exactly right and only ever overestimates the distance, by the
 * factor 1/sinc(Δθ/2) ≈ 1 + Δθ²/24.
 */
export function arcChord(ds: number, dtheta: number): { length: number; bearing: number } {
  const h = dtheta / 2;
  const sinc = Math.abs(h) < 1e-8 ? 1 - (h * h) / 6 : Math.sin(h) / h;
  return { length: ds * sinc, bearing: h };
}

// ---------------------------------------------------------------------------
// LiDAR
// ---------------------------------------------------------------------------

export interface LidarParams {
  nBeams: number;
  /** Angular field of view, radians. 2π for a spinning planar LiDAR. */
  fov: number;
  maxRange: number;
  /** Std-dev of the range noise on a good return, metres. */
  sigmaR: number;
  /** Probability a beam returns nothing and reports z_max (glass, black felt, grazing). */
  pDropout: number;
  /** Sensor origin in the body frame — Rusty's LiDAR sits slightly forward. */
  offset: [number, number];
}

export const RUSTY_LIDAR: LidarParams = {
  nBeams: 180,
  fov: 2 * Math.PI,
  maxRange: 8,
  sigmaR: 0.02,
  pDropout: 0.01,
  offset: [0.04, 0],
};

/**
 * One sweep. Unlike the library's `simulateScan`, this keeps the noise-free
 * ranges and the dropout flags alongside the returns, because the *point* of
 * Chapter 4 is to show the reader both halves at once: what the geometry says,
 * and what the sensor said.
 */
export interface Scan {
  /** What the robot receives. This, and only this, is the data. */
  ranges: number[];
  /** z*ᵏ — the true first-hit distance. The simulator knows it; the robot does not. */
  trueRanges: number[];
  /** Beam bearings φ_k relative to the robot heading. */
  angles: number[];
  dropped: boolean[];
  maxRange: number;
}

/**
 * Evenly spaced bearings; a full circle does not repeat its first beam. Left
 * deliberately unwrapped and monotonically increasing over [−π, π), because
 * beam *index* is the axis every scan-space plot in this book uses.
 */
export function lidarBearings(p: LidarParams): number[] {
  if (p.nBeams <= 1) return [0];
  const wraps = Math.abs(p.fov - 2 * Math.PI) < 1e-9;
  const step = wraps ? p.fov / p.nBeams : p.fov / (p.nBeams - 1);
  return Array.from({ length: p.nBeams }, (_, k) => -p.fov / 2 + k * step);
}

/**
 * `raycast_scan` — the simulator's LiDAR forward model.
 *
 *   z_t^k = min(z*^k + ε, z_max),  ε ~ 𝒩(0, σ_r²),  with dropout to z_max w.p. p_drop
 *
 * Two components, not four. The short returns and uniform noise of Chapter 10's
 * mixture describe *unmodelled* things — people, chair legs, multipath — and
 * this world has none of them. Chapter 10 will fit a four-way model to data from
 * here anyway, and finding that ẑ_short → 0 is the point.
 */
export function raycastScan(world: World, pose: Pose2, p: LidarParams, rng: Rng): Scan {
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const ox = pose.x + c * p.offset[0] - s * p.offset[1];
  const oy = pose.y + s * p.offset[0] + c * p.offset[1];

  const angles = lidarBearings(p);
  const trueRanges: number[] = [];
  const ranges: number[] = [];
  const dropped: boolean[] = [];

  for (const rel of angles) {
    const zStar = rayCast(world, ox, oy, pose.theta + rel, p.maxRange);
    trueRanges.push(zStar);
    // One uniform draw per beam, always — so toggling dropout off does not
    // reshuffle the noise on the beams that survive.
    const u = rng.next();
    const noise = rng.normal(0, p.sigmaR);
    if (u < p.pDropout || zStar >= p.maxRange) {
      ranges.push(p.maxRange);
      dropped.push(true);
    } else {
      ranges.push(Math.max(0, Math.min(p.maxRange, zStar + noise)));
      dropped.push(false);
    }
  }
  return { ranges, trueRanges, angles, dropped, maxRange: p.maxRange };
}

/** Scan endpoints in the robot's own frame — the point cloud a scan matcher wants. */
export function scanPointsBody(scan: Scan): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  for (let k = 0; k < scan.ranges.length; k++) {
    if (scan.dropped[k]) continue;
    pts.push({
      x: scan.ranges[k] * Math.cos(scan.angles[k]),
      y: scan.ranges[k] * Math.sin(scan.angles[k]),
    });
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Command scripts — the deterministic half of a reproducible run
// ---------------------------------------------------------------------------

/** A control script: tick index → commanded twist. Pure, so replays are exact. */
export type Script = (tick: number) => Twist;

/** Build a script from a list of (twist, duration in ticks) segments; it loops. */
export function segments(list: { u: Twist; ticks: number }[]): Script {
  const total = list.reduce((a, b) => a + b.ticks, 0);
  return (tick: number) => {
    let k = ((tick % total) + total) % total;
    for (const seg of list) {
      if (k < seg.ticks) return seg.u;
      k -= seg.ticks;
    }
    return { v: 0, omega: 0 };
  };
}

/**
 * A proportional heading regulator that steers toward a point.
 *
 * This is a *cheat*, and the book says so where it uses it: it closes the loop
 * on ground truth, which is precisely the quantity no robot has. It exists so
 * that a demo can keep driving a sensible route for minutes while the reader
 * studies something else — the odometry drift — instead of watching an
 * open-loop script wander into a wall. Chapter 20 builds the honest version.
 */
export function pursuePoint(
  pose: Pose2,
  target: { x: number; y: number },
  opts: { speed?: number; gain?: number; maxOmega?: number; turnFirst?: number } = {},
): Twist {
  const { speed = 0.55, gain = 1.6, maxOmega = 1.2, turnFirst = 0.7 } = opts;
  const heading = Math.atan2(target.y - pose.y, target.x - pose.x);
  let e = heading - pose.theta;
  e = Math.atan2(Math.sin(e), Math.cos(e));
  const omega = Math.max(-maxOmega, Math.min(maxOmega, gain * e));
  return { v: Math.abs(e) > turnFirst ? 0 : speed, omega };
}
