/**
 * The camera as a probabilistic sensor: pinhole projection, its Jacobians,
 * triangulation, and the depth uncertainty that falls out of a baseline.
 *
 * Every sensor model so far in this book returned a *distance*. A camera
 * returns a direction, with the distance annihilated by the division by Z, so
 * the whole chapter follows from one measurement equation:
 *
 *     z = π(T_cw · P_w) + δ,     δ ~ N(0, σ_px² I)
 *
 * The Jacobians below are the derivative of that line, split by the chain rule
 * exactly as Chapter 18's Derivation 1 splits it. They are what a bundle
 * adjuster needs and nothing more.
 *
 * Frame convention (fixed here and used by every widget in the chapter):
 * +Z forward out of the lens, +X right, +Y down — the computer-vision frame, so
 * that pixel coordinates increase rightward and downward as an image buffer does.
 */

import { solve, transpose, type Mat } from '../prob/linalg';
import { actPose3, hat, inversePose3, type Pose3, type Vec3 } from './se3';

export interface Pinhole {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** Sensor size in pixels. Used for frustum culling and for drawing. */
  width: number;
  height: number;
}

export const pinhole = (
  fx: number,
  fy: number,
  cx: number,
  cy: number,
  width = 640,
  height = 480,
): Pinhole => ({ fx, fy, cx, cy, width, height });

/**
 * Points nearer than this to the image plane are not projected at all.
 *
 * Not a numerical nicety: the Jacobian's 1/Z² entries diverge, and a single
 * point that drifts behind the camera during an optimizer step will otherwise
 * produce an infinite gradient and destroy the whole solve. Every real system
 * gates the near frustum for exactly this reason.
 */
export const Z_MIN = 0.05;

/* -------------------------------------------------------------------------- */
/* Projection                                                                  */
/* -------------------------------------------------------------------------- */

/** π(P) for a point already expressed in the camera frame. `null` behind the lens. */
export function projectCam(cam: Pinhole, pc: Vec3): [number, number] | null {
  if (pc[2] <= Z_MIN) return null;
  return [cam.fx * (pc[0] / pc[2]) + cam.cx, cam.fy * (pc[1] / pc[2]) + cam.cy];
}

/** π(T_cw P_w): project a world point through a world-to-camera pose. */
export function projectPoint(cam: Pinhole, tcw: Pose3, pw: Vec3): [number, number] | null {
  return projectCam(cam, actPose3(tcw, pw));
}

/** Is a projected pixel inside the sensor? */
export function inImage(cam: Pinhole, px: readonly [number, number]): boolean {
  return px[0] >= 0 && px[0] <= cam.width && px[1] >= 0 && px[1] <= cam.height;
}

/** The camera centre of a world-to-camera pose: C = −Rᵀ t. */
export const cameraCenter = (tcw: Pose3): Vec3 => inversePose3(tcw).t;

/**
 * ∂π/∂P_c — the 2×3 block with the famous 1/Z and 1/Z² entries.
 *
 * Everything painful about cameras lives in this matrix. Its scale is f/Z, so
 * a distant point moves few pixels for a large displacement; its third column
 * is proportional to the image coordinate, so a point *on the optical axis*
 * (X = Y = 0) has no depth sensitivity at all in this camera. Depth information
 * is never in one view: it is in the difference between two.
 */
export function projectionJacobianCam(cam: Pinhole, pc: Vec3): Mat {
  const [X, Y, Z] = pc;
  const iz = 1 / Z;
  const iz2 = iz * iz;
  return [
    [cam.fx * iz, 0, -cam.fx * X * iz2],
    [0, cam.fy * iz, -cam.fy * Y * iz2],
  ];
}

/**
 * The two Jacobian blocks of a reprojection factor, both of ∂π (not of the
 * residual e = z − π, which flips their sign).
 *
 *   jPose  = ∂π/∂ξ  ∈ ℝ^{2×6}, for the right retraction T ⊞ ξ = T · Exp(ξ),
 *                    ξ = (ρ, φ) with the translation part first.
 *   jPoint = ∂π/∂P_w ∈ ℝ^{2×3}.
 *
 * The derivation is one line of first-order algebra. With
 * Exp(ξ) P_w ≈ P_w + ρ − [P_w]× φ,
 *
 *     P_c(ξ) = T Exp(ξ) P_w ≈ P_c + R ρ − R [P_w]× φ,
 *
 * so ∂π/∂ξ = J_π R [ I  −[P_w]× ] and ∂π/∂P_w = J_π R. Chapter 18 checks both
 * against central differences; so does `__checks_ch18__.ts`.
 */
export function projectionJacobians(
  cam: Pinhole,
  tcw: Pose3,
  pw: Vec3,
): { jPose: Mat; jPoint: Mat } {
  const pc = actPose3(tcw, pw);
  const jPi = projectionJacobianCam(cam, pc);
  const jPoint: Mat = [
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) {
      jPoint[r][c] =
        jPi[r][0] * tcw.R[0][c] + jPi[r][1] * tcw.R[1][c] + jPi[r][2] * tcw.R[2][c];
    }
  }
  const S = hat(pw); // [P_w]×
  const jPose: Mat = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
  ];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) {
      jPose[r][c] = jPoint[r][c];
      jPose[r][c + 3] = -(jPoint[r][0] * S[0][c] + jPoint[r][1] * S[1][c] + jPoint[r][2] * S[2][c]);
    }
  }
  return { jPose, jPoint };
}

/** e = z − π(T_cw P_w), or `null` when the point is not in front of the camera. */
export function reprojectionResidual(
  cam: Pinhole,
  tcw: Pose3,
  pw: Vec3,
  z: readonly [number, number],
): [number, number] | null {
  const px = projectPoint(cam, tcw, pw);
  return px ? [z[0] - px[0], z[1] - px[1]] : null;
}

/* -------------------------------------------------------------------------- */
/* Bearings, rays, parallax                                                    */
/* -------------------------------------------------------------------------- */

/** The calibrated bearing of a pixel: K⁻¹(u, v, 1), normalized to unit length. */
export function bearing(cam: Pinhole, px: readonly [number, number]): Vec3 {
  const x = (px[0] - cam.cx) / cam.fx;
  const y = (px[1] - cam.cy) / cam.fy;
  const n = Math.hypot(x, y, 1);
  return [x / n, y / n, 1 / n];
}

/** That bearing rotated into the world frame — the ray a pixel actually names. */
export function rayWorld(cam: Pinhole, tcw: Pose3, px: readonly [number, number]): Vec3 {
  const q = bearing(cam, px);
  const rwc = transpose(tcw.R);
  return [
    rwc[0][0] * q[0] + rwc[0][1] * q[1] + rwc[0][2] * q[2],
    rwc[1][0] * q[0] + rwc[1][1] * q[1] + rwc[1][2] * q[2],
    rwc[2][0] * q[0] + rwc[2][1] * q[1] + rwc[2][2] * q[2],
  ];
}

/**
 * The angle subtended at a point by two camera centres.
 *
 * This — not the distance the camera travelled — is the quantity that makes
 * depth observable. Pure rotation moves the camera a long way through the image
 * and produces exactly zero parallax, which is why every monocular system
 * refuses to triangulate until this angle passes a threshold.
 */
export function parallaxAngle(c1: Vec3, c2: Vec3, p: Vec3): number {
  const a: Vec3 = [c1[0] - p[0], c1[1] - p[1], c1[2] - p[2]];
  const b: Vec3 = [c2[0] - p[0], c2[1] - p[1], c2[2] - p[2]];
  const na = Math.hypot(a[0], a[1], a[2]);
  const nb = Math.hypot(b[0], b[1], b[2]);
  if (na < 1e-12 || nb < 1e-12) return 0;
  const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  return Math.acos(Math.max(-1, Math.min(1, c)));
}

/* -------------------------------------------------------------------------- */
/* Triangulation                                                               */
/* -------------------------------------------------------------------------- */

export interface TriangulationView {
  tcw: Pose3;
  z: readonly [number, number];
}

/**
 * Linear (DLT) triangulation from two or more views.
 *
 * Each observation says "the point lies on this ray", and each ray contributes
 * two linear constraints, x(R₃P + t₃) = R₁P + t₁ and y(R₃P + t₃) = R₂P + t₂,
 * where (x, y) are the calibrated image coordinates. Two views over-determine
 * the point, so we solve the stacked system in the least-squares sense.
 *
 * What this minimizes is *algebraic* error, not reprojection error — which is
 * exactly why it is used as an initializer and then handed to bundle
 * adjustment. On noiseless pixels the two agree and the result is exact.
 */
export function triangulateDlt(cam: Pinhole, views: readonly TriangulationView[]): Vec3 {
  const rows: number[][] = [];
  const rhs: number[] = [];
  for (const { tcw, z } of views) {
    const x = (z[0] - cam.cx) / cam.fx;
    const y = (z[1] - cam.cy) / cam.fy;
    rows.push([
      x * tcw.R[2][0] - tcw.R[0][0],
      x * tcw.R[2][1] - tcw.R[0][1],
      x * tcw.R[2][2] - tcw.R[0][2],
    ]);
    rhs.push(tcw.t[0] - x * tcw.t[2]);
    rows.push([
      y * tcw.R[2][0] - tcw.R[1][0],
      y * tcw.R[2][1] - tcw.R[1][1],
      y * tcw.R[2][2] - tcw.R[1][2],
    ]);
    rhs.push(tcw.t[1] - y * tcw.t[2]);
  }
  // Normal equations of the 2n×3 system. n is 2 or 3 here, so this is honest.
  const AtA: Mat = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const Atb = [0, 0, 0];
  for (let r = 0; r < rows.length; r++) {
    for (let i = 0; i < 3; i++) {
      Atb[i] += rows[r][i] * rhs[r];
      for (let j = 0; j < 3; j++) AtA[i][j] += rows[r][i] * rows[r][j];
    }
  }
  const p = solve(AtA, Atb);
  return [p[0], p[1], p[2]];
}

/**
 * Depth uncertainty of a fronto-parallel stereo pair: σ_Z = Z² σ_px / (f b).
 *
 * Disparity is d = f b / Z, so |dZ/dd| = Z²/(f b) and one pixel of matching
 * error becomes that many metres of depth error. Two consequences the chapter
 * leans on: depth error grows with the *square* of range, and halving the
 * baseline doubles it. The 5 m point of the worked example, seen through a
 * 400 px focal length across a 0.5 m baseline, is uncertain to 12.5 cm — from a
 * measurement that was accurate to one pixel.
 */
export function depthSigma(cam: Pinhole, baseline: number, Z: number, sigmaPx: number): number {
  if (baseline <= 0) return Number.POSITIVE_INFINITY;
  return (Z * Z * sigmaPx) / (cam.fx * baseline);
}
