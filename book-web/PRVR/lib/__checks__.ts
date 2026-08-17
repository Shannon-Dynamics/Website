/**
 * Numerical self-checks.
 *
 * The simulations in this book *are* these algorithms — a reader who does the
 * algebra by hand should get the number the widget shows. This file pins that
 * down: each check asserts an invariant that would break if a formula were
 * mistranscribed, and `runSelfChecks()` can be called from a page (or a
 * scratch script) to see them all pass.
 *
 * These are invariants, not unit tests: they check identities the mathematics
 * guarantees (round-trips, normalisations, group axioms, agreement between two
 * derivations of the same quantity) rather than frozen expected outputs.
 */

import {
  adjoint,
  angleDiff,
  boxminus,
  boxplus,
  compose,
  inverse,
  normalizeAngle,
  se2Exp,
  se2Log,
  type Pose2,
  type Twist2,
} from './geom/se2';
import { Ekf } from './filters/ekf';
import { HistogramFilter1D, gaussianKernel } from './filters/bayes';
import { Kf, rtsSmoother, type KfRecord } from './filters/kf';
import { lowVarianceResample, type Particle } from './filters/pf';
import { unscentedTransform } from './filters/ukf';
import { OccupancyGrid, bresenham, logOddsToProb, probToLogOdds } from './mapping/occgrid';
import {
  DEFAULT_BEAM_PARAMS,
  LikelihoodField,
  beamLikelihood,
  landmarkModelKnownCorrespondence,
} from './models/sensor';
import {
  motionModelVelocity,
  odomFromPoses,
  applyOdom,
  sampleMotionModelVelocity,
  type MotionAlphas,
} from './models/motion';
import { cholesky, ellipse2, matMul, matVec, transpose, type Mat } from './prob/linalg';
import { discreteEntropy, gaussianProduct, mvnPdf } from './prob/gaussian';
import { Rng } from './prob/rng';
import {
  APARTMENT,
  HALLWAY_1D,
  beamAngles,
  diffDriveStep,
  distanceToWalls,
  hallwayMeasurementLikelihood,
  isDoorAt,
  rayCast,
  trueScan,
} from './sim/world';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

/** Largest absolute entry-wise difference between two matrices. */
const maxMatDiff = (a: Mat, b: Mat): number =>
  Math.max(...a.map((row, i) => Math.max(...row.map((v, j) => Math.abs(v - b[i][j])))));

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

export function runSelfChecks(): CheckResult[] {
  const out: CheckResult[] = [];
  const check = (name: string, fn: () => { pass: boolean; detail?: string }) => {
    try {
      const r = fn();
      out.push({ name, pass: r.pass, detail: r.detail });
    } catch (e) {
      out.push({ name, pass: false, detail: `threw: ${(e as Error).message}` });
    }
  };

  // 1 ---------------------------------------------------------------------
  check('se2: exp/log round-trip (incl. ω → 0)', () => {
    const taus: Twist2[] = [
      [1, 0, 0],
      [0.4, -0.9, 1.7],
      [2, 3, -3.0],
      [1, 1, 1e-9],
      [-0.3, 0.2, 1e-4],
    ];
    let worst = 0;
    for (const tau of taus) {
      const back = se2Log(se2Exp(tau));
      for (let i = 0; i < 3; i++) worst = Math.max(worst, Math.abs(back[i] - tau[i]));
    }
    return { pass: worst < 1e-9, detail: `max |log(exp(τ)) − τ| = ${fmt(worst)}` };
  });

  // 2 ---------------------------------------------------------------------
  check('se2: group axioms and ⊞/⊟ consistency', () => {
    const a: Pose2 = { x: 1.5, y: -0.4, theta: 2.1 };
    const b: Pose2 = { x: -2.2, y: 3.3, theta: -1.4 };
    const id = compose(a, inverse(a));
    const idErr = Math.abs(id.x) + Math.abs(id.y) + Math.abs(id.theta);
    const rebuilt = boxplus(b, boxminus(a, b));
    const bpErr =
      Math.abs(rebuilt.x - a.x) +
      Math.abs(rebuilt.y - a.y) +
      Math.abs(angleDiff(rebuilt.theta, a.theta));
    const wrap = Math.abs(normalizeAngle(3 * Math.PI) - Math.PI);
    return {
      pass: idErr < 1e-12 && bpErr < 1e-12 && wrap < 1e-12,
      detail: `a∘a⁻¹ = ${fmt(idErr)}, b ⊞ (a ⊟ b) − a = ${fmt(bpErr)}`,
    };
  });

  // 3 ---------------------------------------------------------------------
  check('se2: adjoint satisfies T exp(τ) T⁻¹ = exp(Ad_T τ)', () => {
    const T: Pose2 = { x: 0.7, y: -1.3, theta: 0.9 };
    const tau: Twist2 = [0.2, -0.15, 0.35];
    const lhs = compose(compose(T, se2Exp(tau)), inverse(T));
    const rhs = se2Exp(matVec(adjoint(T), tau) as Twist2);
    const err =
      Math.abs(lhs.x - rhs.x) + Math.abs(lhs.y - rhs.y) + Math.abs(angleDiff(lhs.theta, rhs.theta));
    return { pass: err < 1e-12, detail: `residual = ${fmt(err)}` };
  });

  // 4 ---------------------------------------------------------------------
  check('linalg: Cholesky reproduces A = L Lᵀ', () => {
    const A: Mat = [
      [4, 1, 0.5],
      [1, 3, -0.2],
      [0.5, -0.2, 2],
    ];
    const L = cholesky(A);
    const err = maxMatDiff(matMul(L, transpose(L)), A);
    return { pass: err < 1e-12, detail: `max |L Lᵀ − A| = ${fmt(err)}` };
  });

  // 5 ---------------------------------------------------------------------
  check('linalg: ellipse2 recovers known eigenvalues and tilt', () => {
    // Σ = R(30°) diag(4, 1) R(30°)ᵀ  ⇒  1σ semi-axes 2 and 1, tilted 30°.
    const t = Math.PI / 6;
    const c = Math.cos(t);
    const s = Math.sin(t);
    const R: Mat = [
      [c, -s],
      [s, c],
    ];
    const cov = matMul(matMul(R, [[4, 0], [0, 1]]), transpose(R));
    const e = ellipse2(cov, 1);
    const angErr = Math.abs(angleDiff(e.angle, t));
    return {
      pass: Math.abs(e.rx - 2) < 1e-9 && Math.abs(e.ry - 1) < 1e-9 && angErr < 1e-9,
      detail: `rx=${fmt(e.rx)} ry=${fmt(e.ry)} angle=${fmt((e.angle * 180) / Math.PI)}°`,
    };
  });

  // 6 ---------------------------------------------------------------------
  check('gaussian: mvnPdf integrates to 1 on a grid', () => {
    const cov: Mat = [
      [0.5, 0.2],
      [0.2, 0.3],
    ];
    const mean = [0, 0];
    const h = 0.04;
    let sum = 0;
    for (let x = -3.5; x <= 3.5; x += h) {
      for (let y = -3.5; y <= 3.5; y += h) {
        sum += mvnPdf([x, y], mean, cov) * h * h;
      }
    }
    return { pass: Math.abs(sum - 1) < 2e-3, detail: `∫∫ N = ${fmt(sum)}` };
  });

  // 7 ---------------------------------------------------------------------
  check('gaussian: 1-D product matches the hand-computed fusion', () => {
    // N(0,1) fused with N(1,1) ⇒ mean ½, variance ½.
    const g = gaussianProduct(0, 1, 1, 1);
    // …and an asymmetric case: N(2, 4) with N(5, 1) ⇒ mean 4.4, variance 0.8.
    const g2 = gaussianProduct(2, 4, 5, 1);
    const ok =
      Math.abs(g.mean - 0.5) < 1e-12 &&
      Math.abs(g.variance - 0.5) < 1e-12 &&
      Math.abs(g2.mean - 4.4) < 1e-12 &&
      Math.abs(g2.variance - 0.8) < 1e-12;
    return { pass: ok, detail: `μ=${fmt(g.mean)} σ²=${fmt(g.variance)} | μ₂=${fmt(g2.mean)}` };
  });

  // 8 ---------------------------------------------------------------------
  check('kf: 1-D update matches the hand-computed example', () => {
    // Prior N(0, 1); measurement z = 1 with H = 1, R = 1.
    // K = 1/(1+1) = ½ ⇒ x = ½, and Joseph gives P = ½ — the same answer as the
    // Gaussian product above, which is the point of the whole chapter.
    const kf = new Kf([0], [[1]]);
    const info = kf.updateWith([1], [[1]], [[1]]);
    kf.predictWith([[1]], [[1]]); // random walk: P should become ½ + 1 = 3/2
    const ok =
      Math.abs(info.K[0][0] - 0.5) < 1e-12 &&
      Math.abs(kf.x[0] - 0.5) < 1e-12 &&
      Math.abs(kf.P[0][0] - 1.5) < 1e-12;
    return { pass: ok, detail: `K=${fmt(info.K[0][0])} x=${fmt(kf.x[0])} P̄=${fmt(kf.P[0][0])}` };
  });

  // 9 ---------------------------------------------------------------------
  check('ekf: reduces to the KF on a linear model', () => {
    const F: Mat = [
      [1, 0.1],
      [0, 1],
    ];
    const Q: Mat = [
      [0.01, 0],
      [0, 0.02],
    ];
    const H: Mat = [[1, 0]];
    const R: Mat = [[0.25]];

    const kf = new Kf([0, 1], [[1, 0], [0, 1]]);
    const ekf = new Ekf([0, 1], [[1, 0], [0, 1]]);

    kf.predictWith(F, Q);
    ekf.predict((x) => matVec(F, x), F, Q);
    kf.updateWith([0.3], H, R);
    ekf.update([0.3], (x) => matVec(H, x), H, R);

    const dx = Math.max(Math.abs(kf.x[0] - ekf.x[0]), Math.abs(kf.x[1] - ekf.x[1]));
    const dP = maxMatDiff(kf.P, ekf.P);
    return { pass: dx < 1e-12 && dP < 1e-12, detail: `Δx = ${fmt(dx)}, ΔP = ${fmt(dP)}` };
  });

  // 10 --------------------------------------------------------------------
  check('kf: RTS smoothing never increases covariance', () => {
    const F: Mat = [[1]];
    const Q: Mat = [[0.1]];
    const H: Mat = [[1]];
    const R: Mat = [[0.5]];
    const kf = new Kf([0], [[1]]);
    const records: KfRecord[] = [];
    for (const z of [1.0, 1.2, 0.9, 1.1]) {
      kf.predictWith(F, Q);
      const xPrior = kf.x.slice();
      const PPrior = kf.P.map((r) => r.slice());
      kf.updateWith([z], H, R);
      records.push({ xPrior, PPrior, xPost: kf.x.slice(), PPost: kf.P.map((r) => r.slice()), F });
    }
    const sm = rtsSmoother(records);
    const shrunk = sm.every((s, i) => s.P[0][0] <= records[i].PPost[0][0] + 1e-12);
    const last = Math.abs(sm[sm.length - 1].P[0][0] - records[records.length - 1].PPost[0][0]);
    return {
      pass: shrunk && last < 1e-12 && sm.length === records.length,
      detail: `P₀: filtered ${fmt(records[0].PPost[0][0])} → smoothed ${fmt(sm[0].P[0][0])}`,
    };
  });

  // 11 --------------------------------------------------------------------
  check('ukf: unscented transform is exact for an affine map', () => {
    const A: Mat = [
      [2, -1],
      [0.5, 3],
    ];
    const b = [1, -2];
    const mean = [0.3, -0.7];
    const cov: Mat = [
      [0.4, 0.1],
      [0.1, 0.25],
    ];
    const ut = unscentedTransform(mean, cov, (x) => {
      const y = matVec(A, x);
      return [y[0] + b[0], y[1] + b[1]];
    });
    const expMean = matVec(A, mean).map((v, i) => v + b[i]);
    const expCov = matMul(matMul(A, cov), transpose(A));
    const dm = Math.max(...expMean.map((v, i) => Math.abs(v - ut.mean[i])));
    const dc = maxMatDiff(expCov, ut.cov);
    return {
      pass: dm < 1e-9 && dc < 1e-9 && ut.points.length === 5,
      detail: `Δmean = ${fmt(dm)}, Δcov = ${fmt(dc)}, ${ut.points.length} sigma points`,
    };
  });

  // 12 --------------------------------------------------------------------
  check('pf: low-variance resampling of [.1,.4,.4,.1] keeps the count', () => {
    const build = (): Particle[] =>
      [0.1, 0.4, 0.4, 0.1].map((w, i) => ({ state: { x: i, y: 0, theta: 0 }, weight: w }));
    let countsOk = true;
    let survivorsOk = true;
    for (let seed = 1; seed <= 64; seed++) {
      const res = lowVarianceResample(build(), new Rng(seed));
      if (res.length !== 4) countsOk = false;
      if (!res.every((p) => Math.abs(p.weight - 0.25) < 1e-12)) countsOk = false;
      const ids = res.map((p) => p.state.x);
      // Any particle with weight ≥ 1/M is guaranteed at least one copy.
      if (!ids.includes(1) || !ids.includes(2)) survivorsOk = false;
    }
    return {
      pass: countsOk && survivorsOk,
      detail: countsOk && survivorsOk ? '64 seeds: M = 4, heavy particles always survive' : 'failed',
    };
  });

  // 13 --------------------------------------------------------------------
  check('motion: sampler mean agrees with the noise-free arc', () => {
    const alphas: MotionAlphas = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001];
    const x0: Pose2 = { x: 1, y: 2, theta: 0.3 };
    const u = { v: 1, omega: 0.5, dt: 1 };
    const rng = new Rng(7);
    let sx = 0;
    let sy = 0;
    const N = 4000;
    for (let i = 0; i < N; i++) {
      const p = sampleMotionModelVelocity(u, x0, alphas, rng);
      sx += p.x;
      sy += p.y;
    }
    const truth = diffDriveStep(x0, u.v, u.omega, u.dt);
    const ex = Math.abs(sx / N - truth.x);
    const ey = Math.abs(sy / N - truth.y);
    return { pass: ex < 0.02 && ey < 0.02, detail: `|Δx| = ${fmt(ex)}, |Δy| = ${fmt(ey)}` };
  });

  // 14 --------------------------------------------------------------------
  check('motion: closed-form velocity density peaks at the noise-free pose', () => {
    const alphas: MotionAlphas = [0.02, 0.02, 0.02, 0.02, 0.01, 0.01];
    const x0: Pose2 = { x: 0, y: 0, theta: 0.2 };
    const u = { v: 1, omega: 0.6, dt: 1 };
    const truth = diffDriveStep(x0, u.v, u.omega, u.dt);
    const p0 = motionModelVelocity(truth, u, x0, alphas);
    const offsets: Pose2[] = [
      { ...truth, x: truth.x + 0.1 },
      { ...truth, y: truth.y - 0.1 },
      { ...truth, theta: truth.theta + 0.2 },
    ];
    const lower = offsets.every((p) => motionModelVelocity(p, u, x0, alphas) < p0);
    // A straight-line command must also be finite (the μ construction divides by ~0).
    const straight = motionModelVelocity(
      diffDriveStep(x0, 1, 0, 1),
      { v: 1, omega: 0, dt: 1 },
      x0,
      alphas,
    );
    return {
      pass: lower && Number.isFinite(straight) && straight > 0,
      detail: `p(mode) = ${fmt(p0)}, straight-line p = ${fmt(straight)}`,
    };
  });

  // 15 --------------------------------------------------------------------
  check('motion: odometry decomposition round-trips', () => {
    const a: Pose2 = { x: 1, y: -2, theta: 0.4 };
    const b: Pose2 = { x: 2.5, y: -1.2, theta: -0.9 };
    const back = applyOdom(a, odomFromPoses(a, b));
    const err =
      Math.abs(back.x - b.x) + Math.abs(back.y - b.y) + Math.abs(angleDiff(back.theta, b.theta));
    // Pure rotation is the case that breaks a naive atan2 decomposition.
    const spin = odomFromPoses(a, { ...a, theta: a.theta + 1.0 });
    return {
      pass: err < 1e-12 && spin.trans === 0 && Math.abs(spin.rot2 - 1.0) < 1e-12,
      detail: `round-trip = ${fmt(err)}, pure spin → rot2 = ${fmt(spin.rot2)}`,
    };
  });

  // 16 --------------------------------------------------------------------
  check('sensor: beam mixture integrates to 1 over [0, z_max]', () => {
    const params = { ...DEFAULT_BEAM_PARAMS, maxRange: 8 };
    const zStar = 3;
    const h = 0.002;
    let sum = 0;
    for (let z = h / 2; z < params.maxRange; z += h) {
      sum += beamLikelihood(z, zStar, params) * h;
    }
    // The p_max spike is a point mass, so it contributes its weight directly.
    sum += params.zMax;
    return { pass: Math.abs(sum - 1) < 3e-3, detail: `total mass = ${fmt(sum)}` };
  });

  // 17 --------------------------------------------------------------------
  check('world: rayCast returns exact analytic distances', () => {
    // Standing mid-corridor at (6, 4.4): the east exterior wall is 6 m ahead,
    // the corridor's north wall 0.6 m to the left.
    const east = rayCast(APARTMENT, 6, 4.4, 0, 20);
    const north = rayCast(APARTMENT, 6, 4.4, Math.PI / 2, 20);
    const miss = rayCast(APARTMENT, 6, 4.4, 0, 3); // capped by maxRange
    return {
      pass: Math.abs(east - 6) < 1e-12 && Math.abs(north - 0.6) < 1e-12 && miss === 3,
      detail: `east = ${fmt(east)} m, north = ${fmt(north)} m`,
    };
  });

  // 18 --------------------------------------------------------------------
  check('sensor: likelihood field is ~0 on a wall and matches exact distance', () => {
    const field = new LikelihoodField(APARTMENT, 0.05);
    const onWall = field.distanceAt(5.0, 0.0); // the south exterior wall
    const exactOnWall = distanceToWalls(APARTMENT, 5.0, 0.0);
    // Mid-corridor the nearest wall is the corridor's north side, 0.6 m up.
    const free = field.distanceAt(6.0, 4.4);
    const exactFree = distanceToWalls(APARTMENT, 6.0, 4.4);
    return {
      pass: exactOnWall < 1e-12 && onWall < 0.05 && Math.abs(free - exactFree) < 0.06,
      detail: `wall: exact 0, field ${fmt(onWall)} | free: exact ${fmt(exactFree)}, field ${fmt(free)}`,
    };
  });

  // 19 --------------------------------------------------------------------
  check('sensor: landmark model peaks at the true feature', () => {
    const pose: Pose2 = { x: 3, y: 2, theta: 0.5 };
    const lm = { x: 6, y: 5, id: 0 };
    const sigmas = { r: 0.2, phi: 0.05 };
    const r = Math.hypot(lm.x - pose.x, lm.y - pose.y);
    const phi = normalizeAngle(Math.atan2(lm.y - pose.y, lm.x - pose.x) - pose.theta);
    const p0 = landmarkModelKnownCorrespondence({ r, phi }, lm, pose, sigmas);
    const p1 = landmarkModelKnownCorrespondence({ r: r + 0.3, phi }, lm, pose, sigmas);
    // Bearing residual must wrap: φ and φ + 2π are the same measurement.
    const pWrap = landmarkModelKnownCorrespondence(
      { r, phi: phi + 2 * Math.PI },
      lm,
      pose,
      sigmas,
    );
    return {
      pass: p0 > p1 && Math.abs(p0 - pWrap) < 1e-12,
      detail: `p(true) = ${fmt(p0)} > p(+0.3 m) = ${fmt(p1)}; wrap-invariant`,
    };
  });

  // 20 --------------------------------------------------------------------
  check('bayes: histogram filter normalises and loses entropy on evidence', () => {
    const hf = new HistogramFilter1D({ length: HALLWAY_1D.length, cells: 100 });
    const h0 = hf.entropy();
    hf.correct((x) => hallwayMeasurementLikelihood(x, true));
    const h1 = hf.entropy();
    hf.predict(1.0, gaussianKernel(0.3));
    const h2 = hf.entropy();
    const mass = hf.belief().reduce((a, b) => a + b, 0);
    return {
      pass:
        Math.abs(h0 - Math.log2(100)) < 1e-9 &&
        h1 < h0 &&
        h2 > h1 &&
        Math.abs(mass - 1) < 1e-12 &&
        isDoorAt(2.0),
      detail: `H: ${fmt(h0)} → ${fmt(h1)} bits after sensing, ${fmt(h2)} after moving`,
    };
  });

  // 21 --------------------------------------------------------------------
  check('mapping: log-odds round-trip and scan integration', () => {
    const rt = logOddsToProb(probToLogOdds(0.7));
    const grid = OccupancyGrid.forWorld(APARTMENT, 0.1);
    const h0 = grid.entropy();
    const pose: Pose2 = { x: 6, y: 4.4, theta: 0 };
    const scan = { nBeams: 1, fov: 0, maxRange: 8, sigma: 0 };
    grid.integrateScan(pose, trueScan(APARTMENT, pose, scan), beamAngles(scan));
    const free = grid.probAtWorld(8.05, 4.45); // along the beam, before the wall
    const hit = grid.probAtWorld(11.95, 4.45); // the east wall, 6 m out
    const h1 = grid.entropy();
    return {
      pass: Math.abs(rt - 0.7) < 1e-12 && free < 0.5 && hit > 0.5 && h1 < h0,
      detail: `p(free) = ${fmt(free)}, p(hit) = ${fmt(hit)}, entropy ${fmt(h0)} → ${fmt(h1)} bits`,
    };
  });

  // 22 --------------------------------------------------------------------
  check('mapping: Bresenham is 8-connected and hits both endpoints', () => {
    const line = bresenham(2, 3, 11, 8);
    const first = line[0];
    const last = line[line.length - 1];
    let connected = true;
    for (let i = 1; i < line.length; i++) {
      const di = Math.abs(line[i][0] - line[i - 1][0]);
      const dj = Math.abs(line[i][1] - line[i - 1][1]);
      if (di > 1 || dj > 1 || di + dj === 0) connected = false;
    }
    const single = bresenham(4, 4, 4, 4);
    return {
      pass:
        first[0] === 2 &&
        first[1] === 3 &&
        last[0] === 11 &&
        last[1] === 8 &&
        line.length === 10 &&
        connected &&
        single.length === 1,
      detail: `${line.length} cells from (2,3) to (11,8), all 8-connected`,
    };
  });

  // 23 --------------------------------------------------------------------
  check('gaussian: discrete entropy is log₂ n when uniform, 0 when certain', () => {
    const uniform = discreteEntropy(new Array<number>(8).fill(1 / 8));
    const certain = discreteEntropy([0, 0, 1, 0]);
    const half = discreteEntropy([0.5, 0.5]);
    return {
      pass:
        Math.abs(uniform - 3) < 1e-12 && Math.abs(certain) < 1e-12 && Math.abs(half - 1) < 1e-12,
      detail: `uniform(8) = ${fmt(uniform)} bits, point mass = ${fmt(certain)}`,
    };
  });

  return out;
}

/** Convenience for a console or a status widget. */
export function selfCheckSummary(): { passed: number; total: number; failures: CheckResult[] } {
  const results = runSelfChecks();
  const failures = results.filter((r) => !r.pass);
  return { passed: results.length - failures.length, total: results.length, failures };
}
