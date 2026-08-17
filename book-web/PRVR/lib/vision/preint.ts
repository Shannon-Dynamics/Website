/**
 * IMU preintegration on the manifold — Chapter 18, Derivation 4.
 *
 * The generative model of a strapdown IMU:
 *
 *     ω̃ = ω + b_g + η_g            (angular rate, body frame)
 *     ã  = Rᵀ(a − g) + b_a + η_a    (specific force, body frame)
 *
 * A gyroscope does not measure attitude and an accelerometer does not measure
 * position; both measure *rates*, and everything else is integration with
 * gravity in the loop. Naively, folding 200 Hz samples into an optimizer means
 * re-integrating them at every iteration, because the integral starts from the
 * current estimate of R_i, v_i, p_i.
 *
 * Lupton & Sukkarieh's trick, made rigorous on SO(3) by Forster et al.: factor
 * the initial state out of the integral. What is left,
 *
 *     ΔR_ij = Π Exp((ω̃−b_g)Δt),
 *     Δv_ij = Σ ΔR_ik (ã−b_a) Δt,
 *     Δp_ij = Σ [Δv_ik Δt + ½ ΔR_ik (ã−b_a) Δt²],
 *
 * depends on the measurements and the bias but **not** on the states. Compute
 * it once per keyframe interval; the optimizer then evaluates one 9-vector
 * residual per iteration instead of re-running the integral. Bias changes are
 * absorbed to first order through stored Jacobians, so even the bias does not
 * force a re-integration.
 */

import { matMul, transpose, zerosMat, type Mat } from '../prob/linalg';
import {
  I3,
  addV,
  apply,
  applyT,
  expSO3,
  hat,
  logSO3,
  orthonormalize,
  rightJacobianSO3,
  scaleV,
  subV,
  type Vec3,
} from './se3';

/** Gravity in the world frame, +Z up. */
export const GRAVITY: Vec3 = [0, 0, -9.81];

export interface ImuSample {
  /** ω̃, rad/s, body frame. */
  gyro: Vec3;
  /** ã, m/s², body frame — specific force, so an IMU at rest reads +9.81 up. */
  acc: Vec3;
}

export interface ImuBias {
  gyro: Vec3;
  acc: Vec3;
}

export const zeroBias = (): ImuBias => ({ gyro: [0, 0, 0], acc: [0, 0, 0] });

/** Continuous-time noise densities, the numbers on an IMU datasheet. */
export interface ImuNoise {
  /** rad/s/√Hz */
  gyro: number;
  /** m/s²/√Hz */
  acc: number;
}

/** A navigation state: attitude, position, velocity — the 9 DOF an IMU couples. */
export interface NavState {
  R: Mat;
  p: Vec3;
  v: Vec3;
}

/**
 * One preintegrated interval: a single Gaussian factor standing in for
 * hundreds of raw samples.
 */
export interface Preintegrated {
  /** Total elapsed time Δt_ij. */
  dt: number;
  /** Number of raw samples compressed. */
  n: number;
  dR: Mat;
  dv: Vec3;
  dp: Vec3;
  /** 9×9 covariance of (δφ, δv, δp), in that block order. */
  cov: Mat;
  /** ∂ΔR/∂b_g, ∂Δv/∂b_g, ∂Δv/∂b_a, ∂Δp/∂b_g, ∂Δp/∂b_a — the O(1) bias update. */
  jRg: Mat;
  jVg: Mat;
  jVa: Mat;
  jPg: Mat;
  jPa: Mat;
  /** The bias the deltas were integrated at. Corrections are relative to this. */
  bias: ImuBias;
}

/* -------------------------------------------------------------------------- */
/* The recursion                                                               */
/* -------------------------------------------------------------------------- */

const mul = (a: Mat, b: Mat) => matMul(a, b);
const scaleM = (a: Mat, s: number): Mat => a.map((row) => row.map((x) => x * s));
const addM = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((x, j) => x + b[i][j]));
const subM = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((x, j) => x - b[i][j]));

/**
 * `preintegrate` — Table-style: (samples, Δt, bias, noise) → one factor.
 *
 * The covariance recursion is the linearized error dynamics of the deltas,
 * Σ ← A Σ Aᵀ + B Σ_η Bᵀ, propagated alongside the means. Both the state update
 * and the Jacobian updates read the *previous* ΔR, so the statement order
 * below is load-bearing.
 */
export function preintegrate(
  samples: ImuSample[],
  dt: number,
  bias: ImuBias,
  noise: ImuNoise,
): Preintegrated {
  let dR = I3();
  let dv: Vec3 = [0, 0, 0];
  let dp: Vec3 = [0, 0, 0];
  let cov = zerosMat(9, 9);

  let jRg = zerosMat(3, 3);
  let jVg = zerosMat(3, 3);
  let jVa = zerosMat(3, 3);
  let jPg = zerosMat(3, 3);
  let jPa = zerosMat(3, 3);

  // White-noise densities become discrete covariances by dividing by Δt.
  const sg2 = (noise.gyro * noise.gyro) / dt;
  const sa2 = (noise.acc * noise.acc) / dt;

  for (const s of samples) {
    const w = subV(s.gyro, bias.gyro);
    const a = subV(s.acc, bias.acc);
    const dRk = expSO3(scaleV(w, dt));
    const jr = rightJacobianSO3(scaleV(w, dt));
    const aHat = hat(a);
    const dRa = mul(dR, aHat); // ΔR_ik [ã−b_a]×

    // ---- covariance: Σ ← A Σ Aᵀ + B Σ_η Bᵀ -----------------------------
    const A = zerosMat(9, 9);
    const dRkT = transpose(dRk);
    const blockA = [
      [dRkT, zerosMat(3, 3), zerosMat(3, 3)],
      [scaleM(dRa, -dt), I3(), zerosMat(3, 3)],
      [scaleM(dRa, -0.5 * dt * dt), scaleM(I3(), dt), I3()],
    ];
    for (let bi = 0; bi < 3; bi++) {
      for (let bj = 0; bj < 3; bj++) {
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) A[3 * bi + i][3 * bj + j] = blockA[bi][bj][i][j];
        }
      }
    }
    const B = zerosMat(9, 6);
    const blockB = [
      [scaleM(jr, dt), zerosMat(3, 3)],
      [zerosMat(3, 3), scaleM(dR, dt)],
      [zerosMat(3, 3), scaleM(dR, 0.5 * dt * dt)],
    ];
    for (let bi = 0; bi < 3; bi++) {
      for (let bj = 0; bj < 2; bj++) {
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) B[3 * bi + i][3 * bj + j] = blockB[bi][bj][i][j];
        }
      }
    }
    const noiseCov = zerosMat(6, 6);
    for (let i = 0; i < 3; i++) {
      noiseCov[i][i] = sg2;
      noiseCov[3 + i][3 + i] = sa2;
    }
    cov = addM(matMul(matMul(A, cov), transpose(A)), matMul(matMul(B, noiseCov), transpose(B)));

    // ---- bias Jacobians (previous values on the right-hand side) --------
    const dRaJrg = mul(dRa, jRg);
    jPa = subM(addM(jPa, scaleM(jVa, dt)), scaleM(dR, 0.5 * dt * dt));
    jPg = subM(addM(jPg, scaleM(jVg, dt)), scaleM(dRaJrg, 0.5 * dt * dt));
    jVa = subM(jVa, scaleM(dR, dt));
    jVg = subM(jVg, scaleM(dRaJrg, dt));
    jRg = subM(mul(dRkT, jRg), scaleM(jr, dt));

    // ---- the deltas themselves ------------------------------------------
    const dRa_ = apply(dR, a);
    dp = addV(addV(dp, scaleV(dv, dt)), scaleV(dRa_, 0.5 * dt * dt));
    dv = addV(dv, scaleV(dRa_, dt));
    dR = orthonormalize(mul(dR, dRk));
  }

  return {
    dt: samples.length * dt,
    n: samples.length,
    dR,
    dv,
    dp,
    cov,
    jRg,
    jVg,
    jVa,
    jPg,
    jPa,
    bias: { gyro: [...bias.gyro] as Vec3, acc: [...bias.acc] as Vec3 },
  };
}

/**
 * First-order bias correction — the payoff, and O(1).
 *
 *     ΔR(b+δb) ≈ ΔR(b) Exp(J_{ΔR}^{b_g} δb_g)
 *     Δv(b+δb) ≈ Δv(b) + J_{Δv}^{b_g} δb_g + J_{Δv}^{b_a} δb_a
 *     Δp(b+δb) ≈ Δp(b) + J_{Δp}^{b_g} δb_g + J_{Δp}^{b_a} δb_a
 *
 * A few 3×3 products instead of re-running the whole interval. The
 * approximation is first order in δb, which is why every VIO system re-linearizes
 * (re-integrates) once the bias estimate has moved far enough.
 */
export function biasCorrect(
  pre: Preintegrated,
  bias: ImuBias,
): { dR: Mat; dv: Vec3; dp: Vec3 } {
  const dbg = subV(bias.gyro, pre.bias.gyro);
  const dba = subV(bias.acc, pre.bias.acc);
  const dR = matMul(pre.dR, expSO3(apply(pre.jRg, dbg)));
  const dv = addV(pre.dv, addV(apply(pre.jVg, dbg), apply(pre.jVa, dba)));
  const dp = addV(pre.dp, addV(apply(pre.jPg, dbg), apply(pre.jPa, dba)));
  return { dR, dv, dp };
}

/* -------------------------------------------------------------------------- */
/* The factor                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * `imu_residual` — the 9-vector (r_ΔR, r_Δv, r_Δp) tying states i and j.
 *
 * Gravity appears *here*, not inside the deltas: that is precisely what makes
 * the deltas state-independent, and it is also why the accelerometer makes
 * roll and pitch observable while leaving global position and yaw free.
 */
export function imuResidual(
  pre: Preintegrated,
  si: NavState,
  sj: NavState,
  g: Vec3 = GRAVITY,
  bias?: ImuBias,
): number[] {
  const c = bias ? biasCorrect(pre, bias) : { dR: pre.dR, dv: pre.dv, dp: pre.dp };
  const dt = pre.dt;

  const rR = logSO3(matMul(transpose(c.dR), matMul(transpose(si.R), sj.R)));

  const dvWorld = subV(subV(sj.v, si.v), scaleV(g, dt));
  const rV = subV(applyT(si.R, dvWorld), c.dv);

  const dpWorld = subV(
    subV(subV(sj.p, si.p), scaleV(si.v, dt)),
    scaleV(g, 0.5 * dt * dt),
  );
  const rP = subV(applyT(si.R, dpWorld), c.dp);

  return [...rR, ...rV, ...rP];
}

/** Forward prediction of state j from state i through the preintegrated delta. */
export function predictState(
  si: NavState,
  pre: Preintegrated,
  g: Vec3 = GRAVITY,
  bias?: ImuBias,
): NavState {
  const c = bias ? biasCorrect(pre, bias) : { dR: pre.dR, dv: pre.dv, dp: pre.dp };
  const dt = pre.dt;
  return {
    R: orthonormalize(matMul(si.R, c.dR)),
    v: addV(addV(si.v, scaleV(g, dt)), apply(si.R, c.dv)),
    p: addV(
      addV(addV(si.p, scaleV(si.v, dt)), scaleV(g, 0.5 * dt * dt)),
      apply(si.R, c.dp),
    ),
  };
}

/** Marginal standard deviations of (δφ, δv, δp) — the diagonal of the 9×9. */
export function deltaSigmas(pre: Preintegrated): { phi: number; v: number; p: number } {
  const d = (i: number) => Math.sqrt(Math.max(pre.cov[i][i], 0));
  const rms = (a: number, b: number, c: number) => Math.sqrt((a * a + b * b + c * c) / 3);
  return {
    phi: rms(d(0), d(1), d(2)),
    v: rms(d(3), d(4), d(5)),
    p: rms(d(6), d(7), d(8)),
  };
}
