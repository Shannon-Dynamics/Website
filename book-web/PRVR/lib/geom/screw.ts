/**
 * Screw motions on SE(2): the geometry that {@link se2Exp} is hiding.
 *
 * A constant twist τ = (vₓ, v_y, ω) does not move a body along a straight
 * line — it sweeps it around a fixed point, the **instantaneous center of
 * rotation** (ICR), at constant angular rate. Chasles' theorem in the plane:
 * every rigid displacement is either a pure translation or a rotation about
 * some point. `exp` is the closed form of "follow that screw for one unit of
 * time"; the helpers here expose the center, the radius, and the two ways a
 * reader might try to interpolate between poses — one correct, one not.
 *
 * These are the geometric quantities Chapter 3 draws and Chapter 4 reuses when
 * the diff-drive arc turns out to be exactly this construction.
 */

import { boxminus, boxplus, compose, se2Exp, type Pose2, type Twist2 } from './se2';

/** Below this |ω| the ICR runs off to infinity and the motion is a straight line. */
const STRAIGHT = 1e-9;

export interface ScrewParams {
  /** Net rotation over the unit-time motion, in radians. */
  omega: number;
  /** Signed turning radius |v|/ω, or `null` for a pure translation. */
  radius: number | null;
  /** Path length traced by the body origin. */
  arcLength: number;
  /** True when |ω| is too small for an ICR to exist numerically. */
  straight: boolean;
}

/**
 * The instantaneous center of rotation, in the **body** frame.
 *
 * The body-frame velocity of the point c is v + ω × c; setting it to zero and
 * solving in 2-D gives c = (−v_y/ω, vₓ/ω). Returns `null` for a pure
 * translation, where the center is at infinity.
 */
export function icrBody(tau: Twist2): [number, number] | null {
  const [vx, vy, w] = tau;
  if (Math.abs(w) < STRAIGHT) return null;
  return [-vy / w, vx / w];
}

/** The same center expressed in the world frame, given the body's pose. */
export function icrWorld(pose: Pose2, tau: Twist2): [number, number] | null {
  const c = icrBody(tau);
  if (!c) return null;
  const cos = Math.cos(pose.theta);
  const sin = Math.sin(pose.theta);
  return [pose.x + cos * c[0] - sin * c[1], pose.y + sin * c[0] + cos * c[1]];
}

/** Rotation, turning radius, and arc length of the motion `exp(τ)`. */
export function screwParams(tau: Twist2): ScrewParams {
  const [vx, vy, w] = tau;
  const speed = Math.hypot(vx, vy);
  const straight = Math.abs(w) < STRAIGHT;
  return {
    omega: w,
    radius: straight ? null : speed / w,
    // Unit time at constant body speed, so path length is just the speed.
    arcLength: speed,
    straight,
  };
}

/**
 * Geodesic interpolation between two poses: a ⊞ s·(b ⊟ a).
 *
 * This is the *only* interpolation that a rigid body could physically execute
 * at constant twist, and it is the one every filter, spline, and optimizer in
 * this book means when it says "halfway between two poses".
 */
export function screwInterp(a: Pose2, b: Pose2, s: number): Pose2 {
  const d = boxminus(b, a);
  return boxplus(a, [d[0] * s, d[1] * s, d[2] * s]);
}

/**
 * Componentwise interpolation of the tuple (x, y, θ) — the tempting mistake.
 *
 * Kept in the library on purpose: Chapter 3 shows it side by side with
 * {@link screwInterp}, where it visibly leaves the arc and, when the two
 * headings straddle ±π, spins the body the long way around.
 */
export function naiveInterp(a: Pose2, b: Pose2, s: number): Pose2 {
  return {
    x: a.x + s * (b.x - a.x),
    // No wrap correction — that omission is the entire lesson.
    y: a.y + s * (b.y - a.y),
    theta: a.theta + s * (b.theta - a.theta),
  };
}

/**
 * Sample the screw motion `x₀ · exp(s·τ)` for s ∈ [0, 1] — the path drawn on
 * the manifold side of the Exp/Log Lens.
 */
export function screwPath(x0: Pose2, tau: Twist2, samples = 48): Pose2[] {
  const out: Pose2[] = [];
  for (let i = 0; i <= samples; i++) {
    const s = i / samples;
    out.push(compose(x0, se2Exp([tau[0] * s, tau[1] * s, tau[2] * s])));
  }
  return out;
}
