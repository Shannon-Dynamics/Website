/**
 * Probabilistic motion models — Thrun et al., *Probabilistic Robotics*, Ch. 5.
 *
 * Two models, each in two flavours:
 *
 *   velocity  — "I commanded (v, ω) for Δt". Used when there is no wheel
 *               encoder to read, e.g. planning forwards in time.
 *   odometry  — "the wheels say I moved from x̄ to x̄'". Strictly more accurate
 *               after the fact, and what a real robot actually has.
 *
 * The *closed form* p(x' | u, x) is what the histogram and EKF chapters need;
 * the *sampler* is what the particle filter needs.
 *
 * All six/four α's are **variance** coefficients: `sample(b)` in the book draws
 * from a zero-mean distribution of variance b, so we pass √b as a std-dev.
 *
 * ---------------------------------------------------------------------------
 * A wart worth knowing about, because it looks like a bug and isn't.
 *
 * The closed forms below are Tables 5.1 and 5.5 *verbatim*, and they are not
 * exactly the densities of the Tables 5.3 / 5.6 samplers. Both tables score a
 * hypothesis by inverting the motion to recover the control perturbation it
 * implies, then evaluating the noise density there — but they never apply the
 * Jacobian of that inversion, so the result is a density over the control
 * deltas rather than over poses. For the odometry model the missing factor is
 * exactly 1/δ̂trans; restoring it (and taking the variances from the measured
 * u, as the sampler does, rather than from the hypothesised δ̂) makes sampler
 * and density agree to within Monte Carlo error. As written, the two disagree
 * by a few percent in total-variation distance.
 *
 * We keep the book's form deliberately: it is what the chapter derives, what
 * the Rust listing shows, and what a reader will check this code against. It is
 * also harmless in the role it actually plays — a smooth, correctly-peaked
 * scoring function for nearby hypotheses.
 * ---------------------------------------------------------------------------
 */

import { angleDiff, normalizeAngle, type Pose2 } from '../geom/se2';
import { prob } from '../prob/gaussian';
import type { Rng } from '../prob/rng';

export interface VelocityCmd {
  v: number;
  omega: number;
  dt: number;
}

/** Odometry decomposed as rotate → translate → rotate. */
export interface OdomDelta {
  rot1: number;
  trans: number;
  rot2: number;
}

/** (α₁ … α₆) for the velocity model. */
export type MotionAlphas = [number, number, number, number, number, number];

/** (α₁ … α₄) for the odometry model. */
export type OdomAlphas = [number, number, number, number];

export const DEFAULT_ALPHAS: MotionAlphas = [0.02, 0.02, 0.02, 0.02, 0.005, 0.005];
export const DEFAULT_ODOM_ALPHAS: OdomAlphas = [0.01, 0.01, 0.02, 0.01];

/** Below this |ω̂| the arc formulae are replaced by their straight-line limit. */
const STRAIGHT = 1e-6;

// ---------------------------------------------------------------------------
// Velocity model
// ---------------------------------------------------------------------------

/**
 * `sample_motion_model_velocity` — Thrun et al., **Table 5.3**.
 *
 * Perturb the command, then integrate the perturbed arc exactly. The third
 * noise term γ̂ is the one students always ask about: it is a final rotation
 * that the (v, ω) parameterisation cannot otherwise produce, and without it the
 * model's support is a 2-D surface inside 3-D pose space — a particle filter
 * built on it could never represent heading error at a fixed position.
 */
export function sampleMotionModelVelocity(
  u: VelocityCmd,
  x: Pose2,
  alphas: MotionAlphas,
  rng: Rng,
): Pose2 {
  const [a1, a2, a3, a4, a5, a6] = alphas;
  const v2 = u.v * u.v;
  const w2 = u.omega * u.omega;

  const vHat = u.v + rng.normal(0, Math.sqrt(a1 * v2 + a2 * w2));
  const wHat = u.omega + rng.normal(0, Math.sqrt(a3 * v2 + a4 * w2));
  const gHat = rng.normal(0, Math.sqrt(a5 * v2 + a6 * w2));

  const dt = u.dt;
  const nt = x.theta + wHat * dt;
  if (Math.abs(wHat) < STRAIGHT) {
    return {
      x: x.x + vHat * Math.cos(x.theta) * dt,
      y: x.y + vHat * Math.sin(x.theta) * dt,
      theta: normalizeAngle(nt + gHat * dt),
    };
  }
  const r = vHat / wHat;
  return {
    x: x.x - r * Math.sin(x.theta) + r * Math.sin(nt),
    y: x.y + r * Math.cos(x.theta) - r * Math.cos(nt),
    theta: normalizeAngle(nt + gHat * dt),
  };
}

/**
 * `motion_model_velocity` — Thrun et al., **Table 5.1**. Closed-form density
 * p(x' | u, x).
 *
 * The trick: any two poses are joined by exactly one circular arc that leaves x
 * tangentially. Its centre (x*, y*) lies where the perpendicular bisector of
 * x→x' meets the line normal to the heading, which is the μ construction below.
 * From the centre we read off the radius and swept angle, hence the (v̂, ω̂)
 * that *would* have been required — and the density is just the chance the
 * noise turned (v, ω) into that.
 *
 * When the denominator vanishes the displacement is parallel to the heading:
 * the arc is a straight line, the centre is at infinity, and we take the limit
 * directly rather than dividing by ~0.
 */
export function motionModelVelocity(
  xNext: Pose2,
  u: VelocityCmd,
  x: Pose2,
  alphas: MotionAlphas,
): number {
  const [a1, a2, a3, a4, a5, a6] = alphas;
  const dt = u.dt;
  const c = Math.cos(x.theta);
  const s = Math.sin(x.theta);

  const dx = x.x - xNext.x;
  const dy = x.y - xNext.y;

  const numer = dx * c + dy * s;
  const denom = dy * c - dx * s;

  let vHat: number;
  let wHat: number;

  if (Math.abs(denom) < 1e-9) {
    // Straight-line limit: the whole displacement is along the heading.
    wHat = 0;
    vHat = -numer / dt; // −numer = (x'−x)·(cos θ, sin θ)
  } else {
    const mu = 0.5 * (numer / denom);
    const cx = (x.x + xNext.x) / 2 + mu * dy;
    const cy = (x.y + xNext.y) / 2 - mu * dx;
    const rStar = Math.hypot(x.x - cx, x.y - cy);
    const dTheta = normalizeAngle(
      Math.atan2(xNext.y - cy, xNext.x - cx) - Math.atan2(x.y - cy, x.x - cx),
    );
    wHat = dTheta / dt;
    vHat = (dTheta / dt) * rStar;
  }

  const gHat = angleDiff(xNext.theta, x.theta) / dt - wHat;

  const v2 = u.v * u.v;
  const w2 = u.omega * u.omega;
  // ω is a *rate*, so it is differenced plainly — but ω̂ came from a wrapped
  // Δθ, so like the book this model assumes |ω Δt| < π over one step.
  return (
    prob(u.v - vHat, a1 * v2 + a2 * w2) *
    prob(u.omega - wHat, a3 * v2 + a4 * w2) *
    prob(gHat, a5 * v2 + a6 * w2)
  );
}

// ---------------------------------------------------------------------------
// Odometry model
// ---------------------------------------------------------------------------

/**
 * Decompose a pose change into rotate–translate–rotate (Thrun et al., §5.4).
 *
 * A pure in-place rotation has an undefined first rotation (atan2 of nothing),
 * so we special-case it: all of the change goes into rot2. Without this guard
 * the noise model would blow up on a stationary spin, which is a very common
 * command in the widgets.
 */
export function odomFromPoses(prev: Pose2, curr: Pose2): OdomDelta {
  const dx = curr.x - prev.x;
  const dy = curr.y - prev.y;
  const trans = Math.hypot(dx, dy);
  if (trans < 1e-9) {
    return { rot1: 0, trans: 0, rot2: angleDiff(curr.theta, prev.theta) };
  }
  const rot1 = angleDiff(Math.atan2(dy, dx), prev.theta);
  const rot2 = angleDiff(angleDiff(curr.theta, prev.theta), rot1);
  return { rot1, trans, rot2 };
}

/** Apply an (exact, noise-free) odometry delta to a pose. */
export function applyOdom(x: Pose2, u: OdomDelta): Pose2 {
  const heading = x.theta + u.rot1;
  return {
    x: x.x + u.trans * Math.cos(heading),
    y: x.y + u.trans * Math.sin(heading),
    theta: normalizeAngle(heading + u.rot2),
  };
}

/**
 * `sample_motion_model_odometry` — Thrun et al., **Table 5.6**.
 *
 * Note the α pattern: the two rotations share (α₁, α₂) because they are the
 * same physical process, and translation error grows with the rotations too
 * (α₄) because a wheel that slips while turning also mis-reports distance.
 */
export function sampleMotionModelOdometry(
  u: OdomDelta,
  x: Pose2,
  alphas: OdomAlphas,
  rng: Rng,
): Pose2 {
  const [a1, a2, a3, a4] = alphas;
  const r1sq = u.rot1 * u.rot1;
  const r2sq = u.rot2 * u.rot2;
  const tsq = u.trans * u.trans;

  const rot1 = u.rot1 - rng.normal(0, Math.sqrt(a1 * r1sq + a2 * tsq));
  const trans = u.trans - rng.normal(0, Math.sqrt(a3 * tsq + a4 * (r1sq + r2sq)));
  const rot2 = u.rot2 - rng.normal(0, Math.sqrt(a1 * r2sq + a2 * tsq));

  return applyOdom(x, { rot1, trans, rot2 });
}

/**
 * `motion_model_odometry` — Thrun et al., **Table 5.5**. Closed-form density.
 *
 * `u` is what the wheels reported (the "bar" quantities); the hatted values are
 * what the *hypothesised* pose pair x → x' implies. The density asks how likely
 * the noise is to explain the discrepancy between them.
 */
export function motionModelOdometry(
  xNext: Pose2,
  u: OdomDelta,
  x: Pose2,
  alphas: OdomAlphas,
): number {
  const [a1, a2, a3, a4] = alphas;
  const hat = odomFromPoses(x, xNext);

  const r1sq = hat.rot1 * hat.rot1;
  const r2sq = hat.rot2 * hat.rot2;
  const tsq = hat.trans * hat.trans;

  const p1 = prob(angleDiff(u.rot1, hat.rot1), a1 * r1sq + a2 * tsq);
  const p2 = prob(u.trans - hat.trans, a3 * tsq + a4 * (r1sq + r2sq));
  const p3 = prob(angleDiff(u.rot2, hat.rot2), a1 * r2sq + a2 * tsq);
  return p1 * p2 * p3;
}
