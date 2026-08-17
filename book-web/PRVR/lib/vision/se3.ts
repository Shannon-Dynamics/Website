/**
 * SO(3) and SE(3) — the 3-D sibling of `lib/geom/se2.ts`.
 *
 * Chapter 18 is the first chapter whose estimator genuinely lives in three
 * dimensions: a camera pose has six degrees of freedom, an IMU integrates
 * rotation at 200 Hz, and both need the same $\boxplus$/$\boxminus$ discipline
 * Chapter 3 established in the plane. Everything here is the direct
 * generalization of the SE(2) module:
 *
 *     R = Exp(φ)          Rodrigues, closed form
 *     T ⊞ ξ = T · Exp(ξ)  right (body-frame) perturbation — the book's convention
 *     a ⊟ b = Log(b⁻¹ a)
 *
 * Rotations are stored as 3×3 row-major matrices rather than quaternions. That
 * costs a little memory and buys a lot of readability: every formula in the
 * chapter can be typed in verbatim. Production code (and the Rust listings)
 * uses a unit quaternion, which is why `orthonormalize` exists — a matrix
 * drifts off SO(3) under repeated multiplication and a quaternion does not.
 *
 * Convention note (matters for every Jacobian in this chapter): the tangent
 * vector is ordered **translation first**, ξ = (ρ, φ) ∈ ℝ⁶, matching the SE(2)
 * twist [vₓ, v_y, ω] of `lib/geom/se2.ts`.
 */

import { matMul, transpose, type Mat } from '../prob/linalg';

export type Vec3 = [number, number, number];
/** ξ = (ρ, φ): translation part first, rotation part second. */
export type Twist3 = [number, number, number, number, number, number];

/** Below this rotation angle the exponential's series expansion is used. */
const SMALL = 1e-8;

/* -------------------------------------------------------------------------- */
/* Vector helpers                                                              */
/* -------------------------------------------------------------------------- */

export const v3 = (x = 0, y = 0, z = 0): Vec3 => [x, y, z];
export const addV = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const subV = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scaleV = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dotV = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const normV = (a: Vec3): number => Math.sqrt(dotV(a, a));

export const crossV = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export function normalizeV(a: Vec3): Vec3 {
  const n = normV(a);
  return n < 1e-15 ? [0, 0, 0] : scaleV(a, 1 / n);
}

/* -------------------------------------------------------------------------- */
/* SO(3)                                                                       */
/* -------------------------------------------------------------------------- */

export const I3 = (): Mat => [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

/** The skew-symmetric matrix with `hat(a) b = a × b`. */
export function hat(w: Vec3): Mat {
  return [
    [0, -w[2], w[1]],
    [w[2], 0, -w[0]],
    [-w[1], w[0], 0],
  ];
}

/** The inverse of {@link hat}, reading the axis back off a skew matrix. */
export function vee(s: Mat): Vec3 {
  return [s[2][1], s[0][2], s[1][0]];
}

/** R · v. */
export function apply(r: Mat, v: Vec3): Vec3 {
  return [
    r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
    r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
    r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
  ];
}

/** Rᵀ · v — the inverse rotation, without forming the transpose. */
export function applyT(r: Mat, v: Vec3): Vec3 {
  return [
    r[0][0] * v[0] + r[1][0] * v[1] + r[2][0] * v[2],
    r[0][1] * v[0] + r[1][1] * v[1] + r[2][1] * v[2],
    r[0][2] * v[0] + r[1][2] * v[1] + r[2][2] * v[2],
  ];
}

/**
 * Rodrigues' formula: `Exp(φ) = I + sin θ/θ [φ]× + (1−cos θ)/θ² [φ]×²`.
 *
 * The small-angle branch is not an optimization — at θ → 0 the closed form is
 * 0/0, and an IMU running at 200 Hz spends most of its samples there.
 */
export function expSO3(w: Vec3): Mat {
  const theta = normV(w);
  const k = hat(w);
  const k2 = matMul(k, k);
  let a: number;
  let b: number;
  if (theta < SMALL) {
    a = 1 - (theta * theta) / 6;
    b = 0.5 - (theta * theta) / 24;
  } else {
    a = Math.sin(theta) / theta;
    b = (1 - Math.cos(theta)) / (theta * theta);
  }
  return I3().map((row, i) => row.map((x, j) => x + a * k[i][j] + b * k2[i][j]));
}

/** `Log(R)`: the rotation vector whose exponential is R. */
export function logSO3(r: Mat): Vec3 {
  const tr = r[0][0] + r[1][1] + r[2][2];
  const cos = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const theta = Math.acos(cos);
  const axis: Vec3 = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
  if (theta < SMALL) {
    // sin θ / θ → 1: the skew part is already the rotation vector, doubled.
    return scaleV(axis, 0.5);
  }
  if (Math.PI - theta < 1e-5) {
    // Near π the skew part vanishes; read the axis off the symmetric part.
    const d: Vec3 = [
      Math.sqrt(Math.max((r[0][0] + 1) / 2, 0)),
      Math.sqrt(Math.max((r[1][1] + 1) / 2, 0)),
      Math.sqrt(Math.max((r[2][2] + 1) / 2, 0)),
    ];
    const largest = d.indexOf(Math.max(...d));
    const signed = d.map((x, i) => (i === largest ? x : x * Math.sign(r[largest][i] + r[i][largest]))) as Vec3;
    return scaleV(normalizeV(signed), theta);
  }
  return scaleV(axis, theta / (2 * Math.sin(theta)));
}

/**
 * The right Jacobian of SO(3): `Exp(φ + δ) ≈ Exp(φ) Exp(J_r(φ) δ)`.
 *
 * This is the object that makes preintegration honest. Adding a small rotation
 * increment to a rotation vector is *not* composing rotations; J_r is the
 * correction, and it is exactly where a naive implementation loses accuracy.
 */
export function rightJacobianSO3(w: Vec3): Mat {
  const theta = normV(w);
  const k = hat(w);
  const k2 = matMul(k, k);
  let a: number;
  let b: number;
  if (theta < SMALL) {
    a = -0.5 + (theta * theta) / 24;
    b = 1 / 6 - (theta * theta) / 120;
  } else {
    a = -(1 - Math.cos(theta)) / (theta * theta);
    b = (theta - Math.sin(theta)) / (theta * theta * theta);
  }
  return I3().map((row, i) => row.map((x, j) => x + a * k[i][j] + b * k2[i][j]));
}

/** `J_r(φ)⁻¹`, in closed form so no 3×3 inverse is needed in the hot loop. */
export function rightJacobianInvSO3(w: Vec3): Mat {
  const theta = normV(w);
  const k = hat(w);
  const k2 = matMul(k, k);
  let b: number;
  if (theta < SMALL) {
    b = 1 / 12 + (theta * theta) / 720;
  } else {
    const half = theta / 2;
    b = 1 / (theta * theta) - (1 + Math.cos(theta)) / (2 * theta * Math.sin(theta));
    // Guard the removable singularity at θ = π where sin θ → 0.
    if (!Number.isFinite(b)) b = 1 / 12 + (half * half) / 720;
  }
  return I3().map((row, i) => row.map((x, j) => x + 0.5 * k[i][j] + b * k2[i][j]));
}

/**
 * Project a drifted matrix back onto SO(3) (modified Gram–Schmidt).
 *
 * Numerical hygiene: after a few thousand products a rotation matrix is no
 * longer orthonormal, and every downstream `logSO3` inherits the error.
 */
export function orthonormalize(r: Mat): Mat {
  const c0 = normalizeV([r[0][0], r[1][0], r[2][0]]);
  let c1: Vec3 = [r[0][1], r[1][1], r[2][1]];
  c1 = normalizeV(subV(c1, scaleV(c0, dotV(c0, c1))));
  const c2 = crossV(c0, c1);
  return [
    [c0[0], c1[0], c2[0]],
    [c0[1], c1[1], c2[1]],
    [c0[2], c1[2], c2[2]],
  ];
}

export const rotX = (a: number): Mat => expSO3([a, 0, 0]);
export const rotY = (a: number): Mat => expSO3([0, a, 0]);
export const rotZ = (a: number): Mat => expSO3([0, 0, a]);

/* -------------------------------------------------------------------------- */
/* SE(3)                                                                       */
/* -------------------------------------------------------------------------- */

/** A rigid transform: `p ↦ R p + t`. */
export interface Pose3 {
  R: Mat;
  t: Vec3;
}

export const identityPose3 = (): Pose3 => ({ R: I3(), t: [0, 0, 0] });

export const pose3 = (R: Mat, t: Vec3): Pose3 => ({ R, t });

export const clonePose3 = (T: Pose3): Pose3 => ({ R: T.R.map((r) => [...r]), t: [...T.t] as Vec3 });

/** T_a ∘ T_b. */
export function composePose3(a: Pose3, b: Pose3): Pose3 {
  return { R: matMul(a.R, b.R), t: addV(apply(a.R, b.t), a.t) };
}

export function inversePose3(a: Pose3): Pose3 {
  const rt = transpose(a.R);
  return { R: rt, t: scaleV(apply(rt, a.t), -1) };
}

/** The group action: the point `p` carried through the transform. */
export const actPose3 = (T: Pose3, p: Vec3): Vec3 => addV(apply(T.R, p), T.t);

/** The left Jacobian block `V(φ)` appearing in the SE(3) exponential. */
function vMatrix(phi: Vec3): Mat {
  const theta = normV(phi);
  const k = hat(phi);
  const k2 = matMul(k, k);
  let a: number;
  let b: number;
  if (theta < SMALL) {
    a = 0.5 - (theta * theta) / 24;
    b = 1 / 6 - (theta * theta) / 120;
  } else {
    a = (1 - Math.cos(theta)) / (theta * theta);
    b = (theta - Math.sin(theta)) / (theta * theta * theta);
  }
  return I3().map((row, i) => row.map((x, j) => x + a * k[i][j] + b * k2[i][j]));
}

/** `Exp(ξ)` with ξ = (ρ, φ): rotation by Exp(φ), translation by V(φ) ρ. */
export function expSE3(xi: Twist3): Pose3 {
  const rho: Vec3 = [xi[0], xi[1], xi[2]];
  const phi: Vec3 = [xi[3], xi[4], xi[5]];
  const R = expSO3(phi);
  const t = apply(vMatrix(phi), rho);
  return { R, t };
}

/** `Log(T)`, the inverse of {@link expSE3}. */
export function logSE3(T: Pose3): Twist3 {
  const phi = logSO3(T.R);
  const v = vMatrix(phi);
  // V is well conditioned for |φ| < π; a 3×3 solve is cheap and exact enough.
  const rho = solve3(v, T.t);
  return [rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]];
}

/** The book's retraction: perturb in the body frame, `T ⊞ ξ = T · Exp(ξ)`. */
export const boxplusPose3 = (T: Pose3, xi: Twist3): Pose3 => composePose3(T, expSE3(xi));

/** The matching difference, `a ⊟ b = Log(b⁻¹ ∘ a)`. */
export const boxminusPose3 = (a: Pose3, b: Pose3): Twist3 =>
  logSE3(composePose3(inversePose3(b), a));

/** Cramer's rule on a 3×3 — small, branchless, and used inside `logSE3`. */
export function solve3(a: Mat, b: Vec3): Vec3 {
  const det =
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  if (Math.abs(det) < 1e-14) return [0, 0, 0];
  const col = (j: number, v: Vec3): Mat => a.map((row, i) => row.map((x, k) => (k === j ? v[i] : x)));
  const d = (m: Mat) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  return [d(col(0, b)) / det, d(col(1, b)) / det, d(col(2, b)) / det];
}

/**
 * A camera-to-world pose looking from `eye` at `target`.
 *
 * Camera convention throughout this chapter: +Z forward (out of the lens), +X
 * right, +Y down — the standard computer-vision frame, so that the projection
 * `(f X/Z + cₓ, f Y/Z + c_y)` lands in image coordinates with y increasing
 * downward.
 */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = [0, -1, 0]): Pose3 {
  const zc = normalizeV(subV(target, eye)); // forward
  const xc = normalizeV(crossV(up, zc)); // right
  const yc = crossV(zc, xc); // down
  return {
    R: [
      [xc[0], yc[0], zc[0]],
      [xc[1], yc[1], zc[1]],
      [xc[2], yc[2], zc[2]],
    ],
    t: [...eye] as Vec3,
  };
}
