/**
 * Bundle adjustment: Chapter 15's Gauss–Newton with the points eliminated first.
 *
 * A reprojection problem's normal equations have the arrowhead shape
 *
 *     [ H_pp   H_pl ] [ Δξ ]   [ g_p ]
 *     [ H_plᵀ  H_ll ] [ Δp ] = [ g_l ]
 *
 * with H_ll block-diagonal — one independent 3×3 per landmark, because no
 * landmark is ever connected to another landmark. Eliminating the points with
 * the Schur complement S = H_pp − H_pl H_ll⁻¹ H_plᵀ costs O(L) small inverses
 * and leaves a pose system whose size does not grow with the map. That is the
 * same `reduce` step the information filter of Chapter 14 performed and the
 * same variable elimination Chapter 15 derived; only the block sizes changed.
 *
 * This module is deliberately small and dense: the chapter's job is to show
 * that the *structure* of visual estimation is already familiar, not to ship a
 * competitive solver.
 */

import { inv, solve, zerosMat, type Mat } from '../prob/linalg';
import { projectPoint, projectionJacobians, type Pinhole } from './pinhole';
import { boxplusPose3, type Pose3, type Twist3, type Vec3 } from './se3';

/** One pixel observation: keyframe `cam` saw landmark `pt` at pixel `z`. */
export interface ReprojObs {
  cam: number;
  pt: number;
  /** The measured pixel — green in every figure in this book. */
  z: [number, number];
}

export interface BaProblem {
  cam: Pinhole;
  /** World-to-camera poses. Mutated in place by `baStep`. */
  poses: Pose3[];
  /** World points. Mutated in place by `baStep`. */
  points: Vec3[];
  obs: ReprojObs[];
  /** Pixel noise σ; whitens every residual so the cost is in units of σ². */
  sigmaPx: number;
  /**
   * Which poses are held fixed. Reprojection error is blind to a rigid motion
   * of the whole scene *and* (monocular) to its overall scale, so something has
   * to be pinned or the normal equations are singular. This is the gauge.
   */
  fixedPoses: boolean[];
  /** Huber threshold in units of σ. Leave undefined for plain least squares. */
  huber?: number;
}

export interface BaReport {
  /** ½ Σ ρ(‖r/σ‖²) — the objective actually minimized, dimensionless. */
  cost: number;
  /** Root-mean-square reprojection error in pixels: the number to report. */
  rmse: number;
  /** Observations in front of a camera this iteration. */
  used: number;
  /** ‖Δ‖ of the step just taken; 0 for a pure evaluation. */
  stepNorm: number;
  /** Size of the linear system actually solved, 6 × (free poses). */
  reducedDim: number;
  /** Size it would have been without the Schur trick, plus 3 per landmark. */
  fullDim: number;
}

/** Robust weight: 1 inside the Huber band, δ/|r| outside. */
function huberWeight(rNorm: number, delta: number | undefined): number {
  if (delta === undefined || rNorm <= delta) return 1;
  return delta / rNorm;
}

/** Evaluate cost and RMSE without touching the state. */
export function baResiduals(p: BaProblem): BaReport {
  const sigma = Math.max(p.sigmaPx, 1e-9);
  let cost = 0;
  let sq = 0;
  let used = 0;
  for (const o of p.obs) {
    const px = projectPoint(p.cam, p.poses[o.cam], p.points[o.pt]);
    if (!px) continue;
    const dx = o.z[0] - px[0];
    const dy = o.z[1] - px[1];
    const n = Math.hypot(dx, dy) / sigma;
    cost += 0.5 * huberWeight(n, p.huber) * n * n;
    sq += dx * dx + dy * dy;
    used += 1;
  }
  const free = p.fixedPoses.filter((f) => !f).length;
  return {
    cost,
    rmse: used > 0 ? Math.sqrt(sq / (2 * used)) : 0,
    used,
    stepNorm: 0,
    reducedDim: 6 * free,
    fullDim: 6 * free + 3 * p.points.length,
  };
}

export interface BaStepOptions {
  /**
   * Levenberg–Marquardt damping. Besides taming a bad linearization it quietly
   * supplies the missing gauge: a monocular problem's scale direction has
   * exactly zero curvature, and with damping the solver simply declines to move
   * along it instead of dividing by zero.
   */
  lambda?: number;
}

/**
 * `ba_solve`, one iteration: build the normal equations, Schur-eliminate the
 * points, solve for the poses, back-substitute the points.
 *
 * Reads as five blocks, and it is worth reading them in order — this is the
 * whole of bundle adjustment, and nothing in it is specific to cameras except
 * the two Jacobians it asks `pinhole.ts` for.
 */
export function baStep(p: BaProblem, opts: BaStepOptions = {}): BaReport {
  const lambda = opts.lambda ?? 1e-3;
  const sigma = Math.max(p.sigmaPx, 1e-9);

  // ---- 0. index the free poses -------------------------------------------
  const slotOf = new Map<number, number>();
  const freePose: number[] = [];
  for (let k = 0; k < p.poses.length; k++) {
    if (p.fixedPoses[k]) continue;
    slotOf.set(k, freePose.length);
    freePose.push(k);
  }
  const F = freePose.length;
  const L = p.points.length;

  const Hpp = zerosMat(6 * F, 6 * F);
  /** Hpl[s] is the 6 × 3L strip of free pose s against every landmark. */
  const Hpl: Mat[] = Array.from({ length: F }, () => zerosMat(6, 3 * L));
  const Hll: Mat[] = Array.from({ length: L }, () => zerosMat(3, 3));
  const gp = new Array(6 * F).fill(0);
  const gl = new Array(3 * L).fill(0);

  let cost = 0;
  let sq = 0;
  let used = 0;

  // ---- 1. accumulate ------------------------------------------------------
  for (const o of p.obs) {
    const T = p.poses[o.cam];
    const P = p.points[o.pt];
    const px = projectPoint(p.cam, T, P);
    if (!px) continue;
    used += 1;
    const dx = o.z[0] - px[0];
    const dy = o.z[1] - px[1];
    sq += dx * dx + dy * dy;

    // Whitened residual e = (z − π)/σ, so the Jacobians are whitened too and
    // the cost comes out in units of σ² — comparable across noise settings.
    const e = [dx / sigma, dy / sigma];
    const nrm = Math.hypot(e[0], e[1]);
    const w = huberWeight(nrm, p.huber);
    cost += 0.5 * w * nrm * nrm;

    const { jPose, jPoint } = projectionJacobians(p.cam, T, P);
    // The residual is z − π, so its Jacobian is −∂π; the sign cancels in JᵀJ
    // and survives in Jᵀe, which is why only the gradient below is negated.
    const Jl = jPoint.map((row) => row.map((v) => v / sigma));
    const slot = slotOf.get(o.cam);
    const Jp = slot === undefined ? null : jPose.map((row) => row.map((v) => v / sigma));

    const lb = 3 * o.pt;
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        Hll[o.pt][a][b] += w * (Jl[0][a] * Jl[0][b] + Jl[1][a] * Jl[1][b]);
      }
      gl[lb + a] += w * (Jl[0][a] * e[0] + Jl[1][a] * e[1]);
    }

    if (Jp && slot !== undefined) {
      const pb = 6 * slot;
      for (let a = 0; a < 6; a++) {
        for (let b = 0; b < 6; b++) {
          Hpp[pb + a][pb + b] += w * (Jp[0][a] * Jp[0][b] + Jp[1][a] * Jp[1][b]);
        }
        gp[pb + a] += w * (Jp[0][a] * e[0] + Jp[1][a] * e[1]);
        for (let b = 0; b < 3; b++) {
          Hpl[slot][a][lb + b] += w * (Jp[0][a] * Jl[0][b] + Jp[1][a] * Jl[1][b]);
        }
      }
    }
  }

  // ---- 2. damp ------------------------------------------------------------
  for (let i = 0; i < 6 * F; i++) Hpp[i][i] += lambda * (1 + Hpp[i][i]);
  for (let j = 0; j < L; j++) {
    for (let a = 0; a < 3; a++) Hll[j][a][a] += lambda * (1 + Hll[j][a][a]);
  }

  // ---- 3. Schur-eliminate every point ------------------------------------
  const S = Hpp.map((row) => [...row]);
  const rhs = [...gp];
  const W: Mat[][] = [];
  for (let j = 0; j < L; j++) {
    const Ainv = safeInv(Hll[j]);
    const lb = 3 * j;
    const Wj: Mat[] = [];
    for (let s = 0; s < F; s++) {
      const block = zerosMat(6, 3);
      for (let a = 0; a < 6; a++) {
        for (let b = 0; b < 3; b++) {
          block[a][b] =
            Hpl[s][a][lb] * Ainv[0][b] +
            Hpl[s][a][lb + 1] * Ainv[1][b] +
            Hpl[s][a][lb + 2] * Ainv[2][b];
        }
      }
      Wj.push(block);
    }
    W.push(Wj);
    for (let s = 0; s < F; s++) {
      for (let a = 0; a < 6; a++) {
        for (let b = 0; b < 3; b++) rhs[6 * s + a] -= Wj[s][a][b] * gl[lb + b];
      }
      // Here is the fill-in, in one loop: every *pair* of cameras that saw
      // point j is now directly coupled, whether or not they are neighbours in
      // time. Co-visibility is the sparsity pattern of the reduced system.
      for (let t = 0; t < F; t++) {
        for (let a = 0; a < 6; a++) {
          for (let c = 0; c < 6; c++) {
            let acc = 0;
            for (let b = 0; b < 3; b++) acc += Wj[s][a][b] * Hpl[t][c][lb + b];
            S[6 * s + a][6 * t + c] -= acc;
          }
        }
      }
    }
  }

  // ---- 4. solve the small pose system, then back-substitute the points ----
  const dPose = F > 0 ? solve(S, rhs) : [];
  let stepNorm = 0;
  for (let s = 0; s < F; s++) {
    const xi = dPose.slice(6 * s, 6 * s + 6) as Twist3;
    p.poses[freePose[s]] = boxplusPose3(p.poses[freePose[s]], xi);
    for (const v of xi) stepNorm += v * v;
  }
  for (let j = 0; j < L; j++) {
    const lb = 3 * j;
    const r = [gl[lb], gl[lb + 1], gl[lb + 2]];
    for (let s = 0; s < F; s++) {
      for (let b = 0; b < 3; b++) {
        let acc = 0;
        for (let a = 0; a < 6; a++) acc += Hpl[s][a][lb + b] * dPose[6 * s + a];
        r[b] -= acc;
      }
    }
    const d = solve(Hll[j], r);
    p.points[j] = [p.points[j][0] + d[0], p.points[j][1] + d[1], p.points[j][2] + d[2]];
    for (const v of d) stepNorm += v * v;
  }

  return {
    cost,
    rmse: used > 0 ? Math.sqrt(sq / (2 * used)) : 0,
    used,
    stepNorm: Math.sqrt(stepNorm),
    reducedDim: 6 * F,
    fullDim: 6 * F + 3 * L,
  };
}

/** A 3×3 inverse that survives a landmark seen from a single view. */
function safeInv(a: Mat): Mat {
  try {
    return inv(a);
  } catch {
    return inv(a.map((row, i) => row.map((x, j) => (i === j ? x + 1e-6 : x))));
  }
}

/**
 * Run `iters` damped Gauss–Newton steps, returning the RMSE trace.
 *
 * The trace has `iters + 1` entries: the error before the first step and after
 * every one, which is exactly what a convergence plot wants. Note that
 * `baStep` reports the error at the linearization point it *started* from, so
 * the trace re-evaluates after each step rather than reusing that number.
 */
export function baSolve(p: BaProblem, iters = 8, opts: BaStepOptions = {}): number[] {
  const trace = [baResiduals(p).rmse];
  for (let i = 0; i < iters; i++) {
    baStep(p, opts);
    trace.push(baResiduals(p).rmse);
  }
  return trace;
}

/**
 * Which pose pairs become directly coupled once the points are eliminated.
 *
 * The answer — every pair of cameras that saw a common point — is the whole
 * story of Schur fill-in, and it is why the reduced camera system of a large
 * reconstruction is dense wherever the trajectory revisits a place.
 */
export function coVisibility(obs: readonly ReprojObs[], nPoses: number): boolean[][] {
  const seen = new Map<number, number[]>();
  for (const o of obs) {
    const list = seen.get(o.pt) ?? [];
    if (!list.includes(o.cam)) list.push(o.cam);
    seen.set(o.pt, list);
  }
  const adj = Array.from({ length: nPoses }, () => new Array<boolean>(nPoses).fill(false));
  for (const cams of seen.values()) {
    for (const a of cams) for (const b of cams) adj[a][b] = true;
  }
  return adj;
}
