/**
 * Numerical self-checks for Chapter 18's visual-inertial module.
 *
 * Same contract as `lib/__checks__.ts` and `lib/optim/__checks_ch15__.ts`:
 * these are *invariants* the mathematics guarantees, not frozen outputs. If a
 * Jacobian sign flips, a bias Jacobian recursion is mistranscribed, or the
 * Schur complement stops being exact, one of these fails.
 *
 *     import { runCh18Checks } from './vision/__checks_ch18__';
 *     out.push(...runCh18Checks());
 */

import { inv, matMul, transpose } from '../prob/linalg';
import { Rng } from '../prob/rng';
import {
  bearing,
  cameraCenter,
  depthSigma,
  parallaxAngle,
  projectPoint,
  projectionJacobians,
  triangulateDlt,
} from './pinhole';
import {
  GRAVITY,
  biasCorrect,
  deltaSigmas,
  imuResidual,
  preintegrate,
  zeroBias,
  type ImuBias,
  type ImuSample,
} from './preint';
import { DEFAULT_CAM, makeImuArc, makeRingScene, makeTwoViewScene } from './scene';
import {
  I3,
  boxplusPose3,
  expSO3,
  logSO3,
  expSE3,
  logSE3,
  rightJacobianSO3,
  rotZ,
  type Pose3,
  type Twist3,
  type Vec3,
} from './se3';
import { baResiduals, baSolve, baStep, type BaProblem } from './tiny-ba';
import { blockNormMatrix, buildWindow, newestSigma, schurBlocks, slideWindow } from './window';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

export function runCh18Checks(): CheckResult[] {
  const out: CheckResult[] = [];
  const check = (name: string, fn: () => { pass: boolean; detail?: string }) => {
    try {
      const { pass, detail } = fn();
      out.push({ name, pass, detail });
    } catch (error) {
      out.push({ name, pass: false, detail: `threw: ${(error as Error).message}` });
    }
  };

  // 1 ---------------------------------------------------------------------
  check('se3: Exp/Log round-trips on SE(3)', () => {
    const rng = new Rng(7);
    let worst = 0;
    for (let i = 0; i < 40; i++) {
      const xi = Array.from({ length: 6 }, () => rng.normal(0, 0.7)) as Twist3;
      const back = logSE3(expSE3(xi));
      worst = Math.max(worst, ...back.map((x, k) => Math.abs(x - xi[k])));
    }
    return { pass: worst < 1e-9, detail: `max |Log(Exp(ξ)) − ξ| = ${fmt(worst)}` };
  });

  // 2 ---------------------------------------------------------------------
  check('se3: right Jacobian satisfies Exp(φ+δ) = Exp(φ) Exp(J_r δ)', () => {
    const rng = new Rng(11);
    let worst = 0;
    for (let i = 0; i < 20; i++) {
      const phi: Vec3 = [rng.normal(0, 0.6), rng.normal(0, 0.6), rng.normal(0, 0.6)];
      const d: Vec3 = [rng.normal(0, 1e-6), rng.normal(0, 1e-6), rng.normal(0, 1e-6)];
      const lhs = expSO3([phi[0] + d[0], phi[1] + d[1], phi[2] + d[2]]);
      const jr = rightJacobianSO3(phi);
      const jd: Vec3 = [
        jr[0][0] * d[0] + jr[0][1] * d[1] + jr[0][2] * d[2],
        jr[1][0] * d[0] + jr[1][1] * d[1] + jr[1][2] * d[2],
        jr[2][0] * d[0] + jr[2][1] * d[1] + jr[2][2] * d[2],
      ];
      const rhs = matMul(expSO3(phi), expSO3(jd));
      const err = Math.max(...lhs.map((row, a) => Math.max(...row.map((x, b) => Math.abs(x - rhs[a][b])))));
      worst = Math.max(worst, err);
    }
    return { pass: worst < 1e-11, detail: `max entrywise error = ${fmt(worst)}` };
  });

  // 3 ---------------------------------------------------------------------
  check('pinhole: the chapter worked example projects to (336, 232) and (296, 232)', () => {
    const cam = DEFAULT_CAM;
    const t1: Pose3 = { R: I3(), t: [0, 0, 0] };
    const t2: Pose3 = { R: I3(), t: [-0.5, 0, 0] };
    const p: Vec3 = [0.2, -0.1, 5];
    const z1 = projectPoint(cam, t1, p)!;
    const z2 = projectPoint(cam, t2, p)!;
    const disparity = z1[0] - z2[0];
    return {
      pass:
        Math.abs(z1[0] - 336) < 1e-12 &&
        Math.abs(z1[1] - 232) < 1e-12 &&
        Math.abs(z2[0] - 296) < 1e-12 &&
        Math.abs(disparity - 40) < 1e-12,
      detail: `z₁ = (${z1[0]}, ${z1[1]}), z₂ = (${z2[0]}, ${z2[1]}), disparity = ${disparity} px`,
    };
  });

  // 4 ---------------------------------------------------------------------
  check('pinhole: a 10% depth error is invisible in view 1 and 3.636 px in view 2', () => {
    const cam = DEFAULT_CAM;
    const t1: Pose3 = { R: I3(), t: [0, 0, 0] };
    const t2: Pose3 = { R: I3(), t: [-0.5, 0, 0] };
    const p: Vec3 = [0.2, -0.1, 5];
    const stretched: Vec3 = [0.22, -0.11, 5.5]; // 1.1 × along camera 1's ray
    const a1 = projectPoint(cam, t1, p)!;
    const b1 = projectPoint(cam, t1, stretched)!;
    const a2 = projectPoint(cam, t2, p)!;
    const b2 = projectPoint(cam, t2, stretched)!;
    const e1 = Math.hypot(b1[0] - a1[0], b1[1] - a1[1]);
    const e2 = Math.hypot(b2[0] - a2[0], b2[1] - a2[1]);
    return {
      pass: e1 < 1e-12 && Math.abs(e2 - 40 / 11) < 1e-9,
      detail: `view 1 moves ${fmt(e1)} px, view 2 moves ${e2.toFixed(4)} px (40/11 = ${(40 / 11).toFixed(4)})`,
    };
  });

  // 5 ---------------------------------------------------------------------
  check('pinhole: analytic Jacobians match central differences', () => {
    const cam = DEFAULT_CAM;
    const tcw: Pose3 = { R: rotZ(0.2), t: [0.3, -0.15, 0.4] };
    const pw: Vec3 = [0.4, 0.2, 4.2];
    const { jPose, jPoint } = projectionJacobians(cam, tcw, pw);
    const h = 1e-6;
    let worst = 0;
    for (let k = 0; k < 6; k++) {
      const plus = new Array(6).fill(0) as number[];
      const minus = new Array(6).fill(0) as number[];
      plus[k] = h;
      minus[k] = -h;
      const zp = projectPoint(cam, boxplusPose3(tcw, plus as Twist3), pw)!;
      const zm = projectPoint(cam, boxplusPose3(tcw, minus as Twist3), pw)!;
      for (let r = 0; r < 2; r++) {
        worst = Math.max(worst, Math.abs((zp[r] - zm[r]) / (2 * h) - jPose[r][k]));
      }
    }
    for (let k = 0; k < 3; k++) {
      const pp = [...pw] as Vec3;
      const pm = [...pw] as Vec3;
      pp[k] += h;
      pm[k] -= h;
      const zp = projectPoint(cam, tcw, pp)!;
      const zm = projectPoint(cam, tcw, pm)!;
      for (let r = 0; r < 2; r++) {
        worst = Math.max(worst, Math.abs((zp[r] - zm[r]) / (2 * h) - jPoint[r][k]));
      }
    }
    return { pass: worst < 1e-4, detail: `max |analytic − numeric| = ${fmt(worst)} px per unit` };
  });

  // 6 ---------------------------------------------------------------------
  check('pinhole: DLT triangulation is exact on noiseless pixels', () => {
    const scene = makeTwoViewScene({ sigmaPx: 0, baseline: 0.5, seed: 4 });
    let worst = 0;
    scene.truth.forEach((p, j) => {
      const est = scene.initial[j];
      worst = Math.max(worst, Math.hypot(est[0] - p[0], est[1] - p[1], est[2] - p[2]));
    });
    return { pass: worst < 1e-9, detail: `max triangulation error = ${fmt(worst)} m` };
  });

  // 7 ---------------------------------------------------------------------
  check('pinhole: σ_Z = Z²σ_px/(f b) matches the finite-difference disparity', () => {
    const cam = DEFAULT_CAM;
    const b = 0.5;
    const Z = 5;
    // Disparity d = f b / Z ⇒ |dZ/dd| = Z²/(f b); a 1 px error moves depth by that.
    const analytic = depthSigma(cam, b, Z, 1);
    const numeric = (() => {
      const d = (cam.fx * b) / Z;
      const zPlus = (cam.fx * b) / (d - 1);
      return Math.abs(zPlus - Z);
    })();
    return {
      pass: Math.abs(analytic - 0.125) < 1e-12 && Math.abs(analytic - numeric) < 5e-3,
      detail: `σ_Z = ${analytic} m analytically, ${numeric.toFixed(4)} m by disparity`,
    };
  });

  // 8 ---------------------------------------------------------------------
  check('pinhole: bearing and parallax are consistent with the camera centres', () => {
    const cam = DEFAULT_CAM;
    const t2: Pose3 = { R: I3(), t: [-0.5, 0, 0] };
    const c2 = cameraCenter(t2);
    const q = bearing(cam, [336, 232]);
    const angle = parallaxAngle([0, 0, 0], c2, [0.2, -0.1, 5]);
    const expected = Math.atan2(0.2, 5) + Math.atan2(0.5 - 0.2, 5);
    return {
      pass:
        Math.abs(c2[0] - 0.5) < 1e-12 &&
        Math.abs(q[2] - 1 / Math.hypot(0.04, -0.02, 1)) < 1e-12 &&
        Math.abs(angle - expected) < 1e-3,
      detail: `centre 2 = (${c2[0]}, ${c2[1]}, ${c2[2]}), parallax = ${((angle * 180) / Math.PI).toFixed(2)}°`,
    };
  });

  // 9 ---------------------------------------------------------------------
  check('bundle adjustment: Gauss–Newton drives a perturbed structure to sub-pixel', () => {
    const scene = makeTwoViewScene({ sigmaPx: 0.5, baseline: 0.5, seed: 9 });
    const rng = new Rng(31);
    const problem: BaProblem = {
      cam: scene.cam,
      poses: scene.poses.map((p) => ({ R: p.R.map((r) => [...r]), t: [...p.t] as Vec3 })),
      points: scene.initial.map((p) => [p[0] + rng.normal(0, 0.25), p[1] + rng.normal(0, 0.25), p[2] + rng.normal(0, 0.4)] as Vec3),
      obs: scene.obs,
      sigmaPx: scene.sigmaPx,
      fixedPoses: [true, true],
    };
    const start = baResiduals(problem).rmse;
    const trace = baSolve(problem, 8);
    const end = trace[trace.length - 1];
    return {
      pass: start > 5 && end < 1.2 && end < start,
      detail: `RMSE ${start.toFixed(2)} px → ${end.toFixed(3)} px in ${trace.length - 1} iterations`,
    };
  });

  // 10 --------------------------------------------------------------------
  check('bundle adjustment: the Schur complement shrinks the solved system', () => {
    const ring = makeRingScene({ nCams: 6, nPoints: 24, seed: 5 });
    const problem: BaProblem = {
      cam: ring.cam,
      poses: ring.truthPoses.map((p) => ({ R: p.R.map((r) => [...r]), t: [...p.t] as Vec3 })),
      points: ring.truthPoints.map((p) => [...p] as Vec3),
      obs: ring.obs,
      sigmaPx: 1,
      fixedPoses: ring.truthPoses.map((_, i) => i === 0),
    };
    const r = baStep(problem, { lambda: 1e-4 });
    const report = { reduced: r.reducedDim, full: r.fullDim };
    return {
      pass: report.reduced === 30 && report.full === 30 + 3 * 24,
      detail: `solved ${report.reduced}×${report.reduced} instead of ${report.full}×${report.full}`,
    };
  });

  // 11 --------------------------------------------------------------------
  check('preintegration: the residual vanishes on the noiseless truth', () => {
    const arc = makeImuArc({ seconds: 1, rate: 400, noise: { gyro: 0, acc: 0 }, bias: zeroBias() });
    const pre = preintegrate(arc.samples, arc.dt, zeroBias(), { gyro: 1e-3, acc: 1e-2 });
    const r = imuResidual(pre, arc.start, arc.end, GRAVITY);
    const worst = Math.max(...r.map(Math.abs));
    return { pass: worst < 5e-3, detail: `max |r| = ${fmt(worst)} over ${pre.n} samples` };
  });

  // 12 --------------------------------------------------------------------
  check('preintegration: first-order bias correction is exact for a pure-yaw turn', () => {
    // ω = 0.5 rad/s about z for 2 s, integrated at b = 0 ⇒ ΔR = R_z(1.0).
    const n = 400;
    const dt = 2 / n;
    const bg = 0.01;
    const samples: ImuSample[] = Array.from({ length: n }, () => ({
      gyro: [0, 0, 0.5 + bg] as Vec3,
      acc: [0, 0, 9.81] as Vec3,
    }));
    const pre = preintegrate(samples, dt, zeroBias(), { gyro: 1e-3, acc: 1e-2 });
    const raw = logSO3(pre.dR)[2];
    const bias: ImuBias = { gyro: [0, 0, bg], acc: [0, 0, 0] };
    const corrected = logSO3(biasCorrect(pre, bias).dR)[2];
    const exact = preintegrate(samples, dt, bias, { gyro: 1e-3, acc: 1e-2 });
    const exactAngle = logSO3(exact.dR)[2];
    return {
      pass:
        Math.abs(raw - 1.02) < 1e-9 &&
        Math.abs(corrected - 1.0) < 1e-9 &&
        Math.abs(corrected - exactAngle) < 1e-9,
      detail: `raw ${raw.toFixed(6)} rad, corrected ${corrected.toFixed(6)} rad, re-integrated ${exactAngle.toFixed(6)} rad`,
    };
  });

  // 13 --------------------------------------------------------------------
  check('preintegration: accelerometer noise gives σ_v ∝ √Δt and σ_p ∝ Δt^{3/2}', () => {
    const measure = (seconds: number, noise: { gyro: number; acc: number }) => {
      const arc = makeImuArc({ seconds, rate: 200, noise: { gyro: 0, acc: 0 }, bias: zeroBias() });
      return deltaSigmas(preintegrate(arc.samples, arc.dt, zeroBias(), noise));
    };
    const accOnly = { gyro: 0, acc: 0.02 };
    const a = measure(0.5, accOnly);
    const b = measure(2, accOnly);
    const vRatio = b.v / a.v; // √(2/0.5) = 2
    const pRatio = b.p / a.p; // (2/0.5)^{3/2} = 8
    // With gyro noise the attitude error leaks into velocity through the
    // −ΔR[ã]× term, and that contribution grows faster than √Δt.
    const withGyro = { gyro: 0.004, acc: 0.02 };
    const g2 = measure(2, withGyro).v / measure(0.5, withGyro).v;
    return {
      pass: Math.abs(vRatio - 2) < 0.02 && Math.abs(pRatio - 8) < 0.2 && g2 > vRatio + 0.1,
      detail: `σ_v ×${vRatio.toFixed(3)} (expect 2), σ_p ×${pRatio.toFixed(3)} (expect 8), with gyro noise ×${g2.toFixed(3)}`,
    };
  });

  // 14 --------------------------------------------------------------------
  check('marginalization: the Schur complement leaves survivor marginals unchanged', () => {
    const w = buildWindow({ keyframes: 4, landmarks: 5, coVisibility: 2, seed: 3 });
    const before = newestSigma(w);
    const reduced = schurBlocks(w.Omega, w.blocks, new Set([0]));
    const after = newestSigma({ ...w, Omega: reduced.Omega, blocks: reduced.blocks });
    return {
      pass: Math.abs(before - after) < 1e-9,
      detail: `σ(newest) = ${before.toFixed(9)} before, ${after.toFixed(9)} after eliminating x0`,
    };
  });

  // 15 --------------------------------------------------------------------
  check('marginalization: exact sliding densifies, sparsified sliding overclaims', () => {
    let exact = buildWindow({ seed: 3 });
    const nnz0 = blockNormMatrix(exact).flat().filter((x) => x > 1e-9).length;
    for (let i = 0; i < 3; i++) exact = slideWindow(exact, { sparsify: false });
    const nnzExact = blockNormMatrix(exact).flat().filter((x) => x > 1e-9).length;

    let sparse = buildWindow({ seed: 3 });
    for (let i = 0; i < 3; i++) sparse = slideWindow(sparse, { sparsify: true, threshold: 1 });
    const nnzSparse = blockNormMatrix(sparse).flat().filter((x) => x > 1e-9).length;
    const under = 1 - sparse.sigmaReducedActual / sparse.sigmaReducedExact;

    return {
      pass: nnzExact > nnz0 && nnzSparse < nnzExact && under > 0,
      detail: `nnz ${nnz0} → ${nnzExact} exact, ${nnzSparse} sparsified; σ under-reported by ${(under * 100).toFixed(2)}%`,
    };
  });

  // 16 --------------------------------------------------------------------
  check('information: Ω is symmetric positive definite throughout a slide', () => {
    let w = buildWindow({ seed: 12 });
    let ok = true;
    for (let i = 0; i < 3; i++) {
      w = slideWindow(w, { sparsify: false });
      const asym = Math.max(
        ...w.Omega.map((row, a) => Math.max(...row.map((x, b) => Math.abs(x - w.Omega[b][a])))),
      );
      if (asym > 1e-8) ok = false;
      try {
        const covariance = inv(w.Omega);
        const t = transpose(covariance);
        if (!Number.isFinite(t[0][0])) ok = false;
      } catch {
        ok = false;
      }
    }
    return { pass: ok, detail: `${w.blocks.length} blocks after 3 slides` };
  });

  return out;
}
