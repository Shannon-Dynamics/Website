/**
 * SE(2) — the group of planar rigid-body motions.
 *
 * A `Pose2` is a point *and* a heading, and the whole book treats it as an
 * element of a Lie group rather than a 3-vector. That distinction is the reason
 * `theta` never gets averaged naively, and why every filter that touches a pose
 * uses ⊞ / ⊟ instead of + / −.
 *
 * Tangent-vector convention: **translation first**, τ = (vₓ, v_y, ω). This
 * matches the Rust side, which stores twists as `Vector3::new(vx, vy, omega)`,
 * and it is the ordering assumed by {@link adjoint}.
 */

import { type Mat } from '../prob/linalg';

export interface Pose2 {
  x: number;
  y: number;
  theta: number;
}

/** Tangent (Lie algebra) element: (vₓ, v_y, ω). */
export type Twist2 = [number, number, number];

export const IDENTITY_POSE: Pose2 = { x: 0, y: 0, theta: 0 };

export const pose2 = (x = 0, y = 0, theta = 0): Pose2 => ({ x, y, theta });

/** Below this |ω| even the half-angle forms hit 0/0, and we switch to Taylor series. */
const SMALL_ANGLE = 1e-8;

/** Wrap to (−π, π]. */
export function normalizeAngle(a: number): number {
  let x = (a + Math.PI) % (2 * Math.PI);
  if (x <= 0) x += 2 * Math.PI;
  return x - Math.PI;
}

/** Shortest signed rotation taking `b` to `a`, in (−π, π]. */
export function angleDiff(a: number, b: number): number {
  return normalizeAngle(a - b);
}

/**
 * Exponential map exp: se(2) → SE(2).
 *
 *   R = Rot(ω),  t = V(ω) · (vₓ, v_y),
 *   V(ω) = 1/ω · [ sin ω      −(1 − cos ω) ]
 *                [ 1 − cos ω    sin ω      ]
 *
 * Geometrically V turns "how far I drove along the arc" into "where I ended
 * up". As ω → 0 both entries are 0/0, so we fall back to the Taylor series
 * sin ω/ω = 1 − ω²/6 and (1 − cos ω)/ω = ω/2 − ω³/24; the straight-line limit
 * V → I drops out of that automatically.
 *
 * `(1 − cos ω)` is written as `2 sin²(ω/2)`: the literal form cancels away up
 * to eight significant digits for the small ω a 60 Hz control loop produces,
 * and that error lands straight in the pose.
 */
export function se2Exp(tau: Twist2): Pose2 {
  const [vx, vy, w] = tau;
  let a: number; // sin ω / ω
  let b: number; // (1 − cos ω) / ω
  if (Math.abs(w) < SMALL_ANGLE) {
    const w2 = w * w;
    a = 1 - w2 / 6;
    b = w / 2 - (w * w2) / 24;
  } else {
    const sh = Math.sin(w / 2);
    a = Math.sin(w) / w;
    b = (2 * sh * sh) / w;
  }
  return {
    x: a * vx - b * vy,
    y: b * vx + a * vy,
    theta: normalizeAngle(w),
  };
}

/**
 * Logarithm map log: SE(2) → se(2), the exact inverse of {@link se2Exp} for
 * |θ| < π.
 *
 *   V⁻¹(ω) = [  c   ω/2 ]   with c = (ω/2)·cot(ω/2) = ω sin ω / (2(1 − cos ω))
 *            [ −ω/2  c  ]
 *
 * Written as the cotangent rather than the sin/(1−cos) ratio, for the same
 * cancellation reason as {@link se2Exp}. Small-ω fallback: c = 1 − ω²/12.
 */
export function se2Log(p: Pose2): Twist2 {
  const w = normalizeAngle(p.theta);
  const h = w / 2;
  let c: number;
  if (Math.abs(w) < SMALL_ANGLE) {
    c = 1 - (w * w) / 12;
  } else {
    c = h / Math.tan(h);
  }
  return [c * p.x + h * p.y, -h * p.x + c * p.y, w];
}

/** Group product a ∘ b: apply b in a's frame. */
export function compose(a: Pose2, b: Pose2): Pose2 {
  const c = Math.cos(a.theta);
  const s = Math.sin(a.theta);
  return {
    x: a.x + c * b.x - s * b.y,
    y: a.y + s * b.x + c * b.y,
    theta: normalizeAngle(a.theta + b.theta),
  };
}

/** Group inverse: the transform that undoes `p`. */
export function inverse(p: Pose2): Pose2 {
  const c = Math.cos(p.theta);
  const s = Math.sin(p.theta);
  return {
    x: -(c * p.x + s * p.y),
    y: s * p.x - c * p.y,
    theta: normalizeAngle(-p.theta),
  };
}

/** Relative pose of `b` seen from `a` — i.e. a⁻¹ ∘ b. The "odometry" quantity. */
export function between(a: Pose2, b: Pose2): Pose2 {
  return compose(inverse(a), b);
}

/** Right retraction p ⊞ τ = p ∘ exp(τ): perturb a pose in its **own** frame. */
export function boxplus(p: Pose2, tau: Twist2): Pose2 {
  return compose(p, se2Exp(tau));
}

/** Inverse of ⊞: a ⊟ b = log(b⁻¹ ∘ a), so that b ⊞ (a ⊟ b) = a. */
export function boxminus(a: Pose2, b: Pose2): Twist2 {
  return se2Log(between(b, a));
}

/**
 * Adjoint Ad_T, the 3×3 matrix satisfying T · exp(τ) · T⁻¹ = exp(Ad_T τ).
 * It is how a covariance expressed in the body frame is pushed into the world
 * frame: Σ_world = Ad Σ_body Adᵀ.
 *
 * With translation-first ordering:  [ R   (t_y, −t_x)ᵀ ]
 *                                   [ 0        1       ]
 */
export function adjoint(p: Pose2): Mat {
  const c = Math.cos(p.theta);
  const s = Math.sin(p.theta);
  return [
    [c, -s, p.y],
    [s, c, -p.x],
    [0, 0, 1],
  ];
}

/** Map a point from the pose's local frame into the world frame: R·pt + t. */
export function transformPoint(p: Pose2, pt: [number, number]): [number, number] {
  const c = Math.cos(p.theta);
  const s = Math.sin(p.theta);
  return [p.x + c * pt[0] - s * pt[1], p.y + s * pt[0] + c * pt[1]];
}

/** Map a world point into the pose's local frame: Rᵀ(pt − t). */
export function inverseTransformPoint(p: Pose2, pt: [number, number]): [number, number] {
  const c = Math.cos(p.theta);
  const s = Math.sin(p.theta);
  const dx = pt[0] - p.x;
  const dy = pt[1] - p.y;
  return [c * dx + s * dy, -s * dx + c * dy];
}

/** Euclidean distance between the translation parts. Ignores heading. */
export function poseDistance(a: Pose2, b: Pose2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Weighted circular mean of a set of poses. Averaging headings arithmetically
 * puts the mean of 179° and −179° at 0°, which is the wrong side of the map;
 * summing unit vectors puts it at 180° where it belongs.
 */
export function meanPose(poses: Pose2[], weights?: number[]): Pose2 {
  let wx = 0;
  let wy = 0;
  let cs = 0;
  let sn = 0;
  let wsum = 0;
  poses.forEach((p, i) => {
    const w = weights ? weights[i] : 1;
    wx += w * p.x;
    wy += w * p.y;
    cs += w * Math.cos(p.theta);
    sn += w * Math.sin(p.theta);
    wsum += w;
  });
  if (wsum <= 0) return { ...IDENTITY_POSE };
  return { x: wx / wsum, y: wy / wsum, theta: Math.atan2(sn, cs) };
}
