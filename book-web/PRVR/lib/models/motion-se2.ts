/**
 * Motion noise on the manifold — the modern half of Chapter 9.
 *
 * `lib/models/motion.ts` holds Thrun's velocity and odometry models verbatim:
 * noise is injected into the *controls* and pushed through the kinematics. This
 * file says the same thing in the language Chapter 3 set up, and the change of
 * language buys an exact result.
 *
 * Write the exact arc update as a group operation. For a constant command
 * (v, ω) held for Δt the noise-free update is
 *
 *   x_t = x_{t-1} ∘ exp(τ),   τ = (v Δt, 0, ω Δt)ᵀ ∈ se(2)
 *
 * — not an approximation, the same three numbers (see `se2Exp` in geom/se2.ts,
 * and the `arcUpdateIsExp` check below the fold in the chapter). Thrun's
 * sampler perturbs v and ω *additively*, so the sampled twist is
 *
 *   τ̂ = τ + w,   w ~ N(0, R_u),
 *   R_u = Δt² · diag(α₁v² + α₂ω²,  0,  α₃v² + α₄ω²)
 *
 * and therefore the coordinates ξ = log(x_{t-1}⁻¹ x_t) are **exactly** Gaussian
 * whenever α₅ = α₆ = 0. The banana in (x, y, θ) is the image of an ellipsoid
 * under a nonlinear map, nothing more. That is the Long–Wolfe–Mashner–Chirikjian
 * result, and here it is a one-line consequence of the parameterisation.
 *
 * Conventions in this file, all matching geom/se2.ts:
 *   · tangent ordering is translation-first, τ = (vₓ, v_y, ω);
 *   · perturbations are **right** (body-frame): x ⊞ τ = x ∘ exp(τ);
 *   · the exponential chart is anchored at the *start* pose, so ξ = x_t ⊟ x_{t-1}.
 */

import {
  adjoint,
  boxminus,
  boxplus,
  inverse,
  normalizeAngle,
  se2Exp,
  type Pose2,
  type Twist2,
} from '../geom/se2';
import { cholesky, matMul, transpose, zerosMat, type Mat, type Vec } from '../prob/linalg';
import type { Rng } from '../prob/rng';
import type { MotionAlphas, OdomAlphas, OdomDelta, VelocityCmd } from './motion';

// ---------------------------------------------------------------------------
// The twist a velocity command asks for, and the covariance it comes with
// ---------------------------------------------------------------------------

/** τ = (v Δt, 0, ω Δt): the twist whose exponential *is* the exact arc update. */
export function commandTwist(u: VelocityCmd): Twist2 {
  return [u.v * u.dt, 0, u.omega * u.dt];
}

/**
 * R_u — the velocity model's control noise, expressed in the tangent space.
 *
 * Thrun's α's are variances of *rates*; a twist is a rate times Δt, so every
 * entry picks up a Δt². The middle (sideways) entry is exactly zero: a
 * differential drive cannot command lateral slip, which is why the raw velocity
 * model has a rank-2 noise on a 3-dof group — and why α₅, α₆ exist at all.
 */
export function velocityTangentCov(u: VelocityCmd, alphas: MotionAlphas): Mat {
  const [a1, a2, a3, a4] = alphas;
  const v2 = u.v * u.v;
  const w2 = u.omega * u.omega;
  const dt2 = u.dt * u.dt;
  return [
    [dt2 * (a1 * v2 + a2 * w2), 0, 0],
    [0, 0, 0],
    [0, 0, dt2 * (a3 * v2 + a4 * w2)],
  ];
}

/**
 * The final-rotation slack γ̂ as a tangent variance, so the on-manifold model
 * can be given the same three degrees of freedom the sampler has.
 *
 * Thrun applies γ̂ *after* the arc, as a rotation about the final position. To
 * first order that is the same as adding Δt²(α₅v² + α₆ω²) to the ω–ω entry of
 * R_u, and the widget draws both so the reader can see how small the difference
 * is at realistic noise levels.
 */
export function gammaVariance(u: VelocityCmd, alphas: MotionAlphas): number {
  const [, , , , a5, a6] = alphas;
  return u.dt * u.dt * (a5 * u.v * u.v + a6 * u.omega * u.omega);
}

/** R_u with the γ̂ slack folded into the rotational entry (a full-rank model). */
export function velocityTangentCovFull(u: VelocityCmd, alphas: MotionAlphas): Mat {
  const r = velocityTangentCov(u, alphas);
  r[2][2] += gammaVariance(u, alphas);
  return r;
}

// ---------------------------------------------------------------------------
// Sampling and scoring
// ---------------------------------------------------------------------------

/** Cholesky-transform a standard normal draw: w = L z, L Lᵀ = R. */
function sampleTangent(r: Mat, rng: Rng): Twist2 {
  const l = cholesky(r);
  const z = [rng.normal(), rng.normal(), rng.normal()];
  const out: number[] = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    let s = 0;
    for (let j = 0; j <= i; j++) s += l[i][j] * z[j];
    out[i] = s;
  }
  return [out[0], out[1], out[2]];
}

/**
 * The on-manifold motion model: x_t = x_{t-1} ⊞ (τ + w), w ~ N(0, R).
 *
 * One line, no arc formulae, no ω → 0 special case — `se2Exp` already carries
 * the straight-line limit in its Taylor branch. Compare against
 * `sampleMotionModelVelocity`: with α₅ = α₆ = 0 the two agree sample for
 * sample given the same noise draws.
 */
export function sampleTangentMotion(x: Pose2, tau: Twist2, r: Mat, rng: Rng): Pose2 {
  const w = sampleTangent(r, rng);
  return boxplus(x, [tau[0] + w[0], tau[1] + w[1], tau[2] + w[2]]);
}

/**
 * log p(x_t | u, x_{t-1}) for the on-manifold model.
 *
 * Because the chart is anchored at x_{t-1}, the density is a plain 3-D Gaussian
 * in ξ = x_t ⊟ x_{t-1} — no motion inversion, no change-of-variables debt. The
 * log-determinant of exp's Jacobian is absorbed into the base measure of the
 * group, which is the honest version of the η that Table 5.1 sweeps away.
 */
export function logProbTangentMotion(xNext: Pose2, x: Pose2, tau: Twist2, r: Mat): number {
  const xi = boxminus(xNext, x);
  const d: Vec = [xi[0] - tau[0], xi[1] - tau[1], xi[2] - tau[2]];
  const l = cholesky(r);
  // Forward-substitute rather than invert: R is near-singular by construction.
  const y = [0, 0, 0];
  let logDet = 0;
  for (let i = 0; i < 3; i++) {
    let s = d[i];
    for (let j = 0; j < i; j++) s -= l[i][j] * y[j];
    y[i] = s / l[i][i];
    logDet += 2 * Math.log(l[i][i]);
  }
  const q = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
  return -0.5 * (q + logDet + 3 * Math.log(2 * Math.PI));
}

// ---------------------------------------------------------------------------
// Intermediates the book's algorithm tables compute and then discard
//
// Tables 5.1 and 5.6 both work by producing quantities they never return — the
// arc through a hypothesised pose pair, the perturbed rot–trans–rot legs. Those
// intermediates are the *picture* the algorithms are built on, so the widgets
// need them; exposing them here keeps the widgets from re-deriving the same
// geometry beside the library that already owns it.
// ---------------------------------------------------------------------------

/**
 * The perturbed legs drawn inside `sample_motion_model_odometry` (Table 5.6,
 * lines 4–6), before they are recomposed into a pose.
 *
 * Identical noise, identical draw order, so
 * `applyOdom(x, perturbOdomDelta(u, α, rng))` is `sampleMotionModelOdometry`
 * — the Odometry Decomposer widget needs the legs themselves to animate the
 * hinged replay, and this is how it gets them without forking the sampler.
 */
export function perturbOdomDelta(u: OdomDelta, alphas: OdomAlphas, rng: Rng): OdomDelta {
  const [a1, a2, a3, a4] = alphas;
  const r1sq = u.rot1 * u.rot1;
  const r2sq = u.rot2 * u.rot2;
  const tsq = u.trans * u.trans;
  return {
    rot1: u.rot1 - rng.normal(0, Math.sqrt(a1 * r1sq + a2 * tsq)),
    trans: u.trans - rng.normal(0, Math.sqrt(a3 * tsq + a4 * (r1sq + r2sq))),
    rot2: u.rot2 - rng.normal(0, Math.sqrt(a1 * r2sq + a2 * tsq)),
  };
}

// ---------------------------------------------------------------------------
// Inverting the motion, and what the inversion throws away
// ---------------------------------------------------------------------------

export interface ArcInversion {
  /** Centre of the unique arc through both positions, tangent to the start heading. */
  center: [number, number];
  /** Signed radius; `Infinity` when the displacement is along the heading. */
  radius: number;
  /** Swept angle, wrapped to (−π, π]. */
  sweep: number;
  vHat: number;
  omegaHat: number;
  /** Heading rate the arc cannot explain — Thrun's γ̂. */
  gammaHat: number;
  /** True when the arc degenerates to a straight line. */
  straight: boolean;
}

/**
 * The geometry inside `motion_model_velocity` (Table 5.1), exposed.
 *
 * The table computes exactly these numbers and then throws them away, returning
 * only the product of three noise densities. The Arc Anatomy widget draws them,
 * because the construction — intersect the perpendicular bisector of x→x' with
 * the line normal to the start heading — is the whole idea of the closed form
 * and it is invisible in the algorithm listing.
 *
 * Note what the arc is *not* asked to do: it matches the two positions and the
 * start heading, and says nothing about the final heading. Whatever is left
 * over becomes γ̂. Compare `boxminus`, which matches position and heading both
 * and pays for it with a lateral twist component a differential drive cannot
 * execute. Two honest inversions of the same motion, disagreeing about which
 * impossible thing to blame.
 */
export function invertVelocityMotion(x: Pose2, xNext: Pose2, dt: number): ArcInversion {
  const c = Math.cos(x.theta);
  const s = Math.sin(x.theta);
  const dx = x.x - xNext.x;
  const dy = x.y - xNext.y;
  const numer = dx * c + dy * s;
  const denom = dy * c - dx * s;

  if (Math.abs(denom) < 1e-9) {
    const vHat = -numer / dt;
    return {
      center: [Infinity, Infinity],
      radius: Infinity,
      sweep: 0,
      vHat,
      omegaHat: 0,
      gammaHat: normalizeAngle(xNext.theta - x.theta) / dt,
      straight: true,
    };
  }

  const mu = 0.5 * (numer / denom);
  const cx = (x.x + xNext.x) / 2 + mu * dy;
  const cy = (x.y + xNext.y) / 2 - mu * dx;
  const radius = Math.hypot(x.x - cx, x.y - cy);
  const sweep = normalizeAngle(
    Math.atan2(xNext.y - cy, xNext.x - cx) - Math.atan2(x.y - cy, x.x - cx),
  );
  const omegaHat = sweep / dt;
  return {
    center: [cx, cy],
    radius,
    sweep,
    vHat: omegaHat * radius,
    omegaHat,
    gammaHat: normalizeAngle(xNext.theta - x.theta) / dt - omegaHat,
    straight: false,
  };
}

// ---------------------------------------------------------------------------
// Moments, measured two ways
// ---------------------------------------------------------------------------

export interface Moments {
  mean: Vec;
  cov: Mat;
}

/**
 * Sample moments in **Cartesian** coordinates (x, y, θ).
 *
 * The heading mean is circular (atan2 of the summed unit vectors) and heading
 * residuals are wrapped, so a cloud straddling ±π does not report a mean of
 * zero and a variance of π². Everything else is the textbook estimator.
 */
export function cartesianMoments(poses: Pose2[]): Moments {
  const n = poses.length;
  if (n === 0) return { mean: [0, 0, 0], cov: zerosMat(3) };
  let sx = 0;
  let sy = 0;
  let sc = 0;
  let ss = 0;
  for (const p of poses) {
    sx += p.x;
    sy += p.y;
    sc += Math.cos(p.theta);
    ss += Math.sin(p.theta);
  }
  const mean: Vec = [sx / n, sy / n, Math.atan2(ss, sc)];
  const cov = zerosMat(3);
  for (const p of poses) {
    const d = [p.x - mean[0], p.y - mean[1], normalizeAngle(p.theta - mean[2])];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
  }
  const denom = Math.max(n - 1, 1);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] /= denom;
  return { mean, cov };
}

/**
 * Sample moments in the **exponential chart anchored at `origin`**: the
 * statistics of ξ = x ⊟ origin.
 *
 * For the velocity model with α₅ = α₆ = 0 this returns τ and R_u to Monte Carlo
 * error however violent the command — that is the whole point of the chapter's
 * last section. For the odometry model it does not, and the gap is the price of
 * treating rot–trans–rot as three independent noises.
 */
export function tangentMoments(poses: Pose2[], origin: Pose2): Moments {
  const n = poses.length;
  if (n === 0) return { mean: [0, 0, 0], cov: zerosMat(3) };
  const xis = poses.map((p) => boxminus(p, origin));
  const mean: Vec = [0, 0, 0];
  for (const xi of xis) for (let i = 0; i < 3; i++) mean[i] += xi[i] / n;
  const cov = zerosMat(3);
  for (const xi of xis) {
    const d = [xi[0] - mean[0], xi[1] - mean[1], xi[2] - mean[2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
  }
  const denom = Math.max(n - 1, 1);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] /= denom;
  return { mean, cov };
}

/** Pearson correlation between components `i` and `j` of a covariance matrix. */
export function correlation(cov: Mat, i: number, j: number): number {
  const d = Math.sqrt(Math.max(cov[i][i], 0) * Math.max(cov[j][j], 0));
  return d < 1e-12 ? 0 : cov[i][j] / d;
}

// ---------------------------------------------------------------------------
// Symmetric eigen-decomposition (Jacobi) — needed to draw a tangent contour
// ---------------------------------------------------------------------------

/**
 * Eigenpairs of a small symmetric matrix by cyclic Jacobi rotations, returned
 * in descending eigenvalue order.
 *
 * `ellipse2` in prob/linalg.ts does the 2×2 case in closed form; the tangent
 * covariance is 3×3 and routinely rank-deficient (a differential drive has no
 * sideways noise), so a closed form would be a minefield of special cases.
 * Jacobi converges in a handful of sweeps at this size and never divides by a
 * quantity it has not just tested.
 */
export function symEig(a: Mat, sweeps = 12): { values: number[]; vectors: Mat } {
  const n = a.length;
  const m = a.map((row) => [...row]);
  const v: Mat = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p][q] * m[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-18) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const sn = t * c;
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p];
          const mkq = m[k][q];
          m[k][p] = c * mkp - sn * mkq;
          m[k][q] = sn * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k];
          const mqk = m[q][k];
          m[p][k] = c * mpk - sn * mqk;
          m[q][k] = sn * mpk + c * mqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - sn * vkq;
          v[k][q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => m[j][j] - m[i][i]);
  return {
    values: order.map((i) => m[i][i]),
    vectors: order.map((i) => v.map((row) => row[i])),
  };
}

/**
 * The `nSigma` contour of a Gaussian in the exponential chart at `origin`,
 * mapped onto the group and returned as poses.
 *
 * We trace the ellipse spanned by the two *largest* principal directions of the
 * tangent covariance. For a differential drive those are "drove too far" and
 * "turned too much", the third direction being the near-zero lateral one, so
 * the curve is the honest silhouette of the distribution rather than a slice
 * chosen for convenience. Pushed through `exp` it comes out bent: the banana's
 * outline, drawn from a perfectly ordinary ellipse.
 */
export function tangentContour(
  origin: Pose2,
  mean: Twist2,
  cov: Mat,
  nSigma = 2,
  segments = 96,
): Pose2[] {
  const { values, vectors } = symEig(cov);
  const s1 = Math.sqrt(Math.max(values[0], 0));
  const s2 = Math.sqrt(Math.max(values[1], 0));
  const e1 = vectors[0];
  const e2 = vectors[1];
  const out: Pose2[] = [];
  for (let k = 0; k <= segments; k++) {
    const phi = (2 * Math.PI * k) / segments;
    const a = nSigma * s1 * Math.cos(phi);
    const b = nSigma * s2 * Math.sin(phi);
    out.push(
      boxplus(origin, [
        mean[0] + a * e1[0] + b * e2[0],
        mean[1] + a * e1[1] + b * e2[1],
        mean[2] + a * e1[2] + b * e2[2],
      ]),
    );
  }
  return out;
}

// ---------------------------------------------------------------------------
// Compounding
// ---------------------------------------------------------------------------

/**
 * Propagate a tangent covariance through one more identical step.
 *
 * With right perturbations, x_k = x̄_k ⊞ ε_k and ε_k ≈ Ad_{exp(−τ)} ε_{k−1} + w,
 * because sliding a body-frame error past a body-frame motion is exactly what
 * the adjoint does. Hence
 *
 *   Σ_k = A Σ_{k−1} Aᵀ + R,   A = Ad_{exp(τ)⁻¹}
 *
 * — the Kalman prediction equation, with the adjoint standing in for the motion
 * Jacobian. First order only: the second-order corrections of Barfoot & Furgale
 * matter once the heading spread passes roughly 30°, which the compounding
 * widget is tuned to reach.
 */
export function compoundStep(sigma: Mat, tau: Twist2, r: Mat): Mat {
  const a = adjoint(inverse(se2Exp(tau)));
  const next = matMul(matMul(a, sigma), transpose(a));
  return next.map((row, i) => row.map((x, j) => x + r[i][j]));
}

/** Σ after `k` identical steps, starting from certainty. */
export function compoundCovariance(tau: Twist2, r: Mat, k: number): Mat {
  let sigma = zerosMat(3);
  for (let i = 0; i < k; i++) sigma = compoundStep(sigma, tau, r);
  return sigma;
}
