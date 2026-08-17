/**
 * Factor graphs and sparse nonlinear least squares — Chapter 15.
 *
 * A factor graph is the full SLAM posterior written down instead of filtered
 * away: variable nodes (poses, landmarks) and factor nodes (a prior, one per
 * control, one per observation). Taking −log of the factorized posterior turns
 * the product into a sum of squared Mahalanobis residuals, so MAP inference
 * *is* nonlinear least squares:
 *
 *     J(y) = Σₖ ρ( ‖rₖ(y)‖_{Σₖ} ),      ŷ = argmin J(y)
 *
 * Linearize on the manifold (r(y ⊞ Δ) ≈ r̄ + JΔ), stack, and the normal
 * equations Ω Δ = −b appear with Ω = Σₖ wₖ Jₖᵀ Jₖ — Chapter 6's information
 * matrix, grown to trajectory scale and, crucially, *sparse*: a factor writes
 * only into the blocks of the variables it touches.
 *
 * This module is the TypeScript port of `crates/ch15_graph`. It is deliberately
 * dense-matrix: the widgets solve a few hundred dimensions and clarity wins.
 * The sparsity that matters pedagogically — which blocks are nonzero, and what
 * elimination does to them — lives in `./ordering.ts`, where it is *counted*
 * rather than exploited.
 */

import {
  adjoint,
  boxminus,
  boxplus,
  between,
  inverseTransformPoint,
  normalizeAngle,
  type Pose2,
  type Twist2,
} from '../geom/se2';
import {
  eye,
  inv,
  matMul,
  matVec,
  solve,
  symmetrize,
  transpose,
  zeros,
  zerosMat,
  type Mat,
  type Vec,
} from '../prob/linalg';
import { kernelRho, kernelWeight, L2, type Kernel } from './kernels';

/* -------------------------------------------------------------------------- */
/* Variables                                                                   */
/* -------------------------------------------------------------------------- */

export type VarKind = 'pose' | 'landmark' | 'scalar';

/** A variable's identity: which container, which slot. */
export interface VarKey {
  kind: VarKind;
  id: number;
}

/** Tangent-space dimension of each variable kind. SE(2) poses retract; the rest add. */
export const VAR_DIM: Record<VarKind, number> = { pose: 3, landmark: 2, scalar: 1 };

export const poseKey = (id: number): VarKey => ({ kind: 'pose', id });
export const landmarkKey = (id: number): VarKey => ({ kind: 'landmark', id });
export const scalarKey = (id: number): VarKey => ({ kind: 'scalar', id });

/** Stable string form, for map keys. */
export const keyId = (k: VarKey): string => `${k.kind[0]}${k.id}`;

/** How the variable is printed in the book: x₃, m₂, y₁. */
export const keyLabel = (k: VarKey): string =>
  `${k.kind === 'pose' ? 'x' : k.kind === 'landmark' ? 'm' : 'y'}${k.id}`;

export const sameKey = (a: VarKey, b: VarKey): boolean => a.kind === b.kind && a.id === b.id;

export interface Point2 {
  x: number;
  y: number;
}

/** The estimate: every unknown in the problem, in one container. */
export interface Values {
  poses: Pose2[];
  landmarks: Point2[];
  scalars: number[];
}

export const emptyValues = (): Values => ({ poses: [], landmarks: [], scalars: [] });

export function cloneValues(v: Values): Values {
  return {
    poses: v.poses.map((p) => ({ ...p })),
    landmarks: v.landmarks.map((p) => ({ ...p })),
    scalars: v.scalars.slice(),
  };
}

/* -------------------------------------------------------------------------- */
/* Block index — the map from variables to slices of the big vector            */
/* -------------------------------------------------------------------------- */

export interface BlockIndex {
  /** Elimination/storage order. Changing this changes fill-in, not the answer. */
  order: VarKey[];
  offsets: number[];
  dims: number[];
  total: number;
  slot: Map<string, number>;
}

export function buildIndex(order: VarKey[]): BlockIndex {
  const offsets: number[] = [];
  const dims: number[] = [];
  const slot = new Map<string, number>();
  let total = 0;
  order.forEach((k, i) => {
    slot.set(keyId(k), i);
    offsets.push(total);
    dims.push(VAR_DIM[k.kind]);
    total += VAR_DIM[k.kind];
  });
  return { order, offsets, dims, total, slot };
}

/** Slot of a key, or −1 when the variable is held fixed (not in the index). */
export const slotOf = (index: BlockIndex, k: VarKey): number => index.slot.get(keyId(k)) ?? -1;

/**
 * y ← y ⊞ Δ. Poses retract through SE(2)'s exponential (Chapter 3), because a
 * heading is not a number you may add to; landmarks and scalars just add.
 */
export function retract(v: Values, delta: Vec, index: BlockIndex): Values {
  const out = cloneValues(v);
  index.order.forEach((k, i) => {
    const o = index.offsets[i];
    if (k.kind === 'pose') {
      const tau: Twist2 = [delta[o], delta[o + 1], delta[o + 2]];
      out.poses[k.id] = boxplus(v.poses[k.id], tau);
    } else if (k.kind === 'landmark') {
      out.landmarks[k.id] = { x: v.landmarks[k.id].x + delta[o], y: v.landmarks[k.id].y + delta[o + 1] };
    } else {
      out.scalars[k.id] = v.scalars[k.id] + delta[o];
    }
  });
  return out;
}

/* -------------------------------------------------------------------------- */
/* SE(2) right Jacobians                                                       */
/* -------------------------------------------------------------------------- */

/**
 * The left Jacobian J_l(τ) of SE(2), the 3×3 matrix satisfying
 * Log(Exp(τ + δτ)) ≈ ... — equivalently Exp(τ + δτ) ≈ Exp(J_l δτ) Exp(τ).
 *
 * With the book's translation-first twist ordering τ = (vₓ, v_y, ω), the adjoint
 * generator is ad_τ = [[0, −ω, v_y], [ω, 0, −vₓ], [0, 0, 0]], and summing the
 * series J_l = Σ ad_τⁿ/(n+1)! gives a closed form built from the same
 * sin ω/ω and (1 − cos ω)/ω coefficients as `se2Exp`, plus one extra column.
 */
export function se2LeftJacobian(tau: Twist2): Mat {
  const [vx, vy, w] = tau;
  let a: number; // sin ω / ω
  let b: number; // (1 − cos ω) / ω
  let m1: number; // (1 − cos ω) / ω²
  let m2: number; // (ω − sin ω) / ω²
  if (Math.abs(w) < 1e-6) {
    const w2 = w * w;
    a = 1 - w2 / 6;
    b = w / 2 - (w * w2) / 24;
    m1 = 0.5 - w2 / 24;
    m2 = w / 6 - (w * w2) / 120;
  } else {
    const w2 = w * w;
    a = Math.sin(w) / w;
    b = (1 - Math.cos(w)) / w;
    m1 = (1 - Math.cos(w)) / w2;
    m2 = (w - Math.sin(w)) / w2;
  }
  // u = (v_y, −vₓ) is the translation part of ad_τ's third column.
  const u0 = vy;
  const u1 = -vx;
  const c0 = m1 * u0 - m2 * u1;
  const c1 = m2 * u0 + m1 * u1;
  return [
    [a, -b, c0],
    [b, a, c1],
    [0, 0, 1],
  ];
}

/** J_r(τ) = J_l(−τ): the *right* Jacobian, matching the ⊞ = ∘ exp convention. */
export function se2RightJacobian(tau: Twist2): Mat {
  return se2LeftJacobian([-tau[0], -tau[1], -tau[2]]);
}

/**
 * J_r(τ)⁻¹, the factor every ⊟-residual's Jacobian carries:
 * Log(Exp(τ) Exp(δ)) ≈ τ + J_r(τ)⁻¹ δ.
 *
 * The matrix is block-triangular, so the inverse is one 2×2 inverse and one
 * matrix–vector product — no general 3×3 solve required.
 */
export function se2RightJacobianInv(tau: Twist2): Mat {
  const J = se2RightJacobian(tau);
  const [a, nb] = [J[0][0], J[0][1]];
  const b = -nb;
  const det = a * a + b * b;
  const i00 = a / det;
  const i01 = b / det;
  const i10 = -b / det;
  const i11 = a / det;
  const c0 = J[0][2];
  const c1 = J[1][2];
  return [
    [i00, i01, -(i00 * c0 + i01 * c1)],
    [i10, i11, -(i10 * c0 + i11 * c1)],
    [0, 0, 1],
  ];
}

/* -------------------------------------------------------------------------- */
/* Factors                                                                     */
/* -------------------------------------------------------------------------- */

export type FactorKind = 'prior' | 'odometry' | 'loop' | 'landmark' | 'range' | 'linear';

/** Whitened residual and Jacobian blocks, one per key, evaluated at `Values`. */
export interface FactorLinearization {
  r: Vec;
  J: Mat[];
}

export interface Factor {
  id: string;
  kind: FactorKind;
  keys: VarKey[];
  dim: number;
  kernel: Kernel;
  /**
   * Return the **whitened** residual Σ^{-1/2} r and the matching Jacobians, so
   * that the factor's contribution to the objective is exactly ρ(‖r‖).
   */
  linearize: (v: Values) => FactorLinearization;
}

export interface FactorGraph {
  factors: Factor[];
}

/** √Ω for an isotropic block: one number per dimension. */
export const sqrtInfoDiag = (sigmas: number[]): Mat =>
  sigmas.map((s, i) => sigmas.map((_, j) => (i === j ? 1 / s : 0)));

const applyWhitening = (S: Mat, r: Vec): Vec => matVec(S, r);

/**
 * Prior on a pose: r = x ⊟ x̄ = Log(x̄⁻¹ x).
 *
 * One of these is mandatory. Without it the whole problem is invariant to a
 * global rigid motion, Ω is singular along three directions, and the solver is
 * being asked where a map is in a world that has no origin.
 */
export function priorFactor(
  key: VarKey,
  prior: Pose2,
  sqrtInfo: Mat,
  opts: { id?: string; kernel?: Kernel } = {},
): Factor {
  return {
    id: opts.id ?? `prior:${keyId(key)}`,
    kind: 'prior',
    keys: [key],
    dim: 3,
    kernel: opts.kernel ?? L2,
    linearize: (v) => {
      const r = boxminus(v.poses[key.id], prior);
      const Jr = se2RightJacobianInv(r);
      return { r: applyWhitening(sqrtInfo, r), J: [matMul(sqrtInfo, Jr)] };
    },
  };
}

/**
 * Between factor — one control, or one loop closure:
 *
 *     r = (xᵢ⁻¹ xⱼ) ⊟ δ = Log(δ⁻¹ xᵢ⁻¹ xⱼ)
 *
 * Its Jacobians are the reason smoothing is manifold-correct by construction.
 * Perturbing xⱼ ← xⱼ ⊞ Δⱼ moves the residual by J_r⁻¹(r) Δⱼ; perturbing xᵢ acts
 * through the adjoint, which transports a tangent vector from j's frame to i's:
 *
 *     ∂r/∂Δᵢ = −J_r⁻¹(r) · Ad_{xⱼ⁻¹ xᵢ},   ∂r/∂Δⱼ = J_r⁻¹(r)
 *
 * Both are re-evaluated at the *current* estimate every iteration — the exact
 * thing an EKF cannot do once it has marginalized xᵢ away.
 */
export function betweenFactor(
  i: number,
  j: number,
  delta: Pose2,
  sqrtInfo: Mat,
  opts: { id?: string; kernel?: Kernel; kind?: 'odometry' | 'loop' } = {},
): Factor {
  const kind = opts.kind ?? 'odometry';
  return {
    id: opts.id ?? `${kind}:${i}->${j}`,
    kind,
    keys: [poseKey(i), poseKey(j)],
    dim: 3,
    kernel: opts.kernel ?? L2,
    linearize: (v) => {
      const xi = v.poses[i];
      const xj = v.poses[j];
      const rel = between(xi, xj); // xᵢ⁻¹ xⱼ
      const r = boxminus(rel, delta);
      const Ji = se2RightJacobianInv(r);
      const Ad = adjoint(between(xj, xi)); // Ad_{xⱼ⁻¹ xᵢ}
      const dj = matMul(sqrtInfo, Ji);
      const di = matMul(dj, Ad).map((row) => row.map((x) => -x));
      return { r: applyWhitening(sqrtInfo, r), J: [di, dj] };
    },
  };
}

/**
 * Range–bearing observation of a landmark from a pose — Chapter 10's model,
 * used verbatim as a factor. That reuse is the point: the models are not
 * rewritten for the smoother, they *are* the smoother's factors.
 */
export function bearingRangeFactor(
  poseIdx: number,
  lmIdx: number,
  z: { range: number; bearing: number },
  sqrtInfo: Mat,
  opts: { id?: string; kernel?: Kernel } = {},
): Factor {
  return {
    id: opts.id ?? `obs:${poseIdx}-${lmIdx}`,
    kind: 'landmark',
    keys: [poseKey(poseIdx), landmarkKey(lmIdx)],
    dim: 2,
    kernel: opts.kernel ?? L2,
    linearize: (v) => {
      const x = v.poses[poseIdx];
      const m = v.landmarks[lmIdx];
      const [dx, dy] = inverseTransformPoint(x, [m.x, m.y]); // landmark in body frame
      const q = Math.max(dx * dx + dy * dy, 1e-9);
      const rng = Math.sqrt(q);
      const r: Vec = [rng - z.range, normalizeAngle(Math.atan2(dy, dx) - z.bearing)];

      // ∂h/∂d for h = (‖d‖, atan2(d_y, d_x))
      const dhdd: Mat = [
        [dx / rng, dy / rng],
        [-dy / q, dx / q],
      ];
      // Right perturbation of the pose moves the body-frame point by
      // d ← d − (δₓ, δ_y) + δθ (d_y, −d_x).
      const dddx: Mat = [
        [-1, 0, dy],
        [0, -1, -dx],
      ];
      const c = Math.cos(x.theta);
      const s = Math.sin(x.theta);
      const dddm: Mat = [
        [c, s],
        [-s, c],
      ]; // Rᵀ
      return {
        r: applyWhitening(sqrtInfo, r),
        J: [matMul(sqrtInfo, matMul(dhdd, dddx)), matMul(sqrtInfo, matMul(dhdd, dddm))],
      };
    },
  };
}

/**
 * Range from a 2-D point variable to a fixed anchor: r = ‖ℓ − a‖ − z.
 *
 * One scalar residual, one 1×2 Jacobian, and a cost surface with a genuinely
 * curved valley — which is what makes it the right toy for comparing descent
 * strategies (widget w15.3).
 */
export function rangeToAnchorFactor(
  lmIdx: number,
  anchor: Point2,
  z: number,
  sigma: number,
  opts: { id?: string; kernel?: Kernel } = {},
): Factor {
  return {
    id: opts.id ?? `range:${lmIdx}@${anchor.x.toFixed(2)},${anchor.y.toFixed(2)}`,
    kind: 'range',
    keys: [landmarkKey(lmIdx)],
    dim: 1,
    kernel: opts.kernel ?? L2,
    linearize: (v) => {
      const p = v.landmarks[lmIdx];
      const dx = p.x - anchor.x;
      const dy = p.y - anchor.y;
      const d = Math.max(Math.hypot(dx, dy), 1e-9);
      return {
        r: [(d - z) / sigma],
        J: [[[dx / (d * sigma), dy / (d * sigma)]]],
      };
    },
  };
}

/**
 * A linear factor over scalar variables: r = (Σ cᵢ yᵢ − target)/σ.
 *
 * The 1-D chain in the chapter's worked example is built from four of these,
 * which is the cheapest possible way to watch Ω, b and the solve without any
 * geometry in the way.
 */
export function linearFactor(
  keys: VarKey[],
  coeffs: number[],
  target: number,
  sigma = 1,
  opts: { id?: string; kernel?: Kernel } = {},
): Factor {
  return {
    id: opts.id ?? `lin:${keys.map(keyId).join(',')}`,
    kind: 'linear',
    keys,
    dim: 1,
    kernel: opts.kernel ?? L2,
    linearize: (v) => {
      let acc = -target;
      keys.forEach((k, i) => {
        acc += coeffs[i] * v.scalars[k.id];
      });
      return { r: [acc / sigma], J: coeffs.map((c) => [[c / sigma]]) };
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Linearization and assembly                                                  */
/* -------------------------------------------------------------------------- */

export interface FactorReport {
  id: string;
  kind: FactorKind;
  /** Whitened residual norm e = ‖Σ^{-1/2} r‖ — "how many sigmas". */
  e: number;
  /** IRLS weight applied to this factor at the current estimate. */
  w: number;
  /** ρ(e): this factor's share of the objective. */
  cost: number;
}

export interface System {
  /** Ω = Σₖ wₖ Jₖᵀ Jₖ — the information matrix at trajectory scale. */
  Omega: Mat;
  /** b = Σₖ wₖ Jₖᵀ r̄ₖ; the step solves Ω Δ = −b. */
  b: Vec;
  cost: number;
  factors: FactorReport[];
}

/**
 * `linearize_graph`: evaluate every factor at `values`, whiten, apply the IRLS
 * weight, and scatter each block into Ω and b.
 *
 * Nothing here is a matrix factorization — assembly is a loop over factors that
 * touches O(1) blocks each, which is why the *structure* of Ω is a picture of
 * the graph and not a property of the algorithm.
 */
export function linearizeGraph(graph: FactorGraph, values: Values, index: BlockIndex): System {
  const n = index.total;
  const Omega = zerosMat(n, n);
  const b = zeros(n);
  const reports: FactorReport[] = [];
  let cost = 0;

  for (const f of graph.factors) {
    const { r, J } = f.linearize(values);
    let e2 = 0;
    for (const x of r) e2 += x * x;
    const e = Math.sqrt(e2);
    const w = kernelWeight(f.kernel, e);
    const rho = kernelRho(f.kernel, e);
    cost += rho;
    reports.push({ id: f.id, kind: f.kind, e, w, cost: rho });

    for (let a = 0; a < f.keys.length; a++) {
      const sa = slotOf(index, f.keys[a]);
      if (sa < 0) continue; // fixed variable: its columns simply do not exist
      const oa = index.offsets[sa];
      const Ja = J[a];
      const JaT = transpose(Ja);
      // bₐ += w Jₐᵀ r
      const contrib = matVec(JaT, r);
      for (let p = 0; p < contrib.length; p++) b[oa + p] += w * contrib[p];

      for (let c = 0; c < f.keys.length; c++) {
        const sc = slotOf(index, f.keys[c]);
        if (sc < 0) continue;
        const oc = index.offsets[sc];
        const blk = matMul(JaT, J[c]);
        for (let p = 0; p < blk.length; p++) {
          for (let q = 0; q < blk[p].length; q++) Omega[oa + p][oc + q] += w * blk[p][q];
        }
      }
    }
  }

  return { Omega, b, cost, factors: reports };
}

/** J(y) without assembling anything — used for line searches and cost surfaces. */
export function graphCost(graph: FactorGraph, values: Values): number {
  let cost = 0;
  for (const f of graph.factors) {
    const { r } = f.linearize(values);
    let e2 = 0;
    for (const x of r) e2 += x * x;
    cost += kernelRho(f.kernel, Math.sqrt(e2));
  }
  return cost;
}

/** Every variable mentioned by any factor, in first-seen order. */
export function graphVariables(graph: FactorGraph): VarKey[] {
  const seen = new Set<string>();
  const out: VarKey[] = [];
  for (const f of graph.factors) {
    for (const k of f.keys) {
      if (seen.has(keyId(k))) continue;
      seen.add(keyId(k));
      out.push(k);
    }
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Optimizers                                                                  */
/* -------------------------------------------------------------------------- */

export interface IterationRecord {
  iteration: number;
  cost: number;
  /** ‖Δ‖ of the accepted step; 0 for a rejected LM trial. */
  stepNorm: number;
  lambda: number;
  accepted: boolean;
  /** LM gain ratio ϱ = actual reduction / predicted reduction. */
  gain: number;
}

export interface OptimizerReport {
  iterations: IterationRecord[];
  initialCost: number;
  finalCost: number;
  converged: boolean;
}

export interface GnConfig {
  maxIterations?: number;
  /** Stop when the relative cost decrease falls below this. */
  tol?: number;
  /** Trust-region damping. 0 = pure Gauss–Newton. */
  lambda?: number;
}

/**
 * One Gauss–Newton (or damped, LM) step: assemble, solve, retract.
 *
 * Returned separately from the loops because every widget in Chapter 15 wants
 * to *watch* a single iteration — the springs relaxing one notch at a time.
 */
export function optimizeStep(
  graph: FactorGraph,
  values: Values,
  index: BlockIndex,
  lambda = 0,
): { values: Values; delta: Vec; system: System } {
  const system = linearizeGraph(graph, values, index);
  const n = index.total;
  const A = system.Omega.map((row) => row.slice());
  if (lambda > 0) {
    // Marquardt's scaling: damp each direction in proportion to its own
    // curvature, so the trust region is an ellipsoid shaped like Ω, not a ball.
    for (let i = 0; i < n; i++) A[i][i] += lambda * Math.max(system.Omega[i][i], 1e-9);
  } else {
    for (let i = 0; i < n; i++) A[i][i] += 1e-9; // gauge-safety only
  }
  const delta = solve(A, system.b.map((x) => -x));
  return { values: retract(values, delta, index), delta, system };
}

/** `gauss_newton`: linearize → solve → retract, until the cost stops moving. */
export function gaussNewton(
  graph: FactorGraph,
  init: Values,
  index: BlockIndex,
  cfg: GnConfig = {},
): { values: Values; report: OptimizerReport } {
  const maxIterations = cfg.maxIterations ?? 20;
  const tol = cfg.tol ?? 1e-9;
  const lambda = cfg.lambda ?? 0;
  let values = cloneValues(init);
  let cost = graphCost(graph, values);
  const initialCost = cost;
  const iterations: IterationRecord[] = [];
  let converged = false;

  for (let it = 0; it < maxIterations; it++) {
    const stepped = optimizeStep(graph, values, index, lambda);
    values = stepped.values;
    const next = graphCost(graph, values);
    const stepNorm = Math.sqrt(stepped.delta.reduce((s, x) => s + x * x, 0));
    iterations.push({ iteration: it + 1, cost: next, stepNorm, lambda, accepted: true, gain: 1 });
    const rel = Math.abs(cost - next) / Math.max(cost, 1e-12);
    cost = next;
    if (rel < tol || stepNorm < 1e-10) {
      converged = true;
      break;
    }
  }

  return { values, report: { iterations, initialCost, finalCost: cost, converged } };
}

export interface LmConfig extends GnConfig {
  lambda0?: number;
  lambdaMin?: number;
  lambdaMax?: number;
}

/**
 * `levenberg_marquardt`: Gauss–Newton with an adaptive trust region.
 *
 * The gain ratio ϱ compares the reduction the linear model promised with the
 * one the nonlinear cost delivered. Above ⅓ the model is trustworthy and λ
 * shrinks (toward Gauss–Newton); below, the step is rejected outright and λ
 * grows (toward scaled gradient descent, which is short but never wrong).
 */
export function levenbergMarquardt(
  graph: FactorGraph,
  init: Values,
  index: BlockIndex,
  cfg: LmConfig = {},
): { values: Values; report: OptimizerReport } {
  const maxIterations = cfg.maxIterations ?? 30;
  const tol = cfg.tol ?? 1e-9;
  let lambda = cfg.lambda0 ?? 1e-3;
  const lambdaMin = cfg.lambdaMin ?? 1e-9;
  const lambdaMax = cfg.lambdaMax ?? 1e8;

  let values = cloneValues(init);
  let cost = graphCost(graph, values);
  const initialCost = cost;
  const iterations: IterationRecord[] = [];
  let converged = false;

  for (let it = 0; it < maxIterations; it++) {
    const trial = lmTrial(graph, values, index, lambda, cost);
    iterations.push({
      iteration: it + 1,
      cost: trial.accepted ? trial.cost : cost,
      stepNorm: trial.accepted ? trial.stepNorm : 0,
      lambda,
      accepted: trial.accepted,
      gain: trial.gain,
    });
    if (trial.accepted) {
      const rel = Math.abs(cost - trial.cost) / Math.max(cost, 1e-12);
      values = trial.values;
      cost = trial.cost;
      lambda = Math.max(lambda * Math.max(1 / 3, 1 - Math.pow(2 * trial.gain - 1, 3)), lambdaMin);
      if (rel < tol || trial.stepNorm < 1e-10) {
        converged = true;
        break;
      }
    } else {
      lambda = Math.min(lambda * 10, lambdaMax);
      if (lambda >= lambdaMax) break;
    }
  }

  return { values, report: { iterations, initialCost, finalCost: cost, converged } };
}

/** One LM trial step, exposed so a widget can single-step the schedule. */
export function lmTrial(
  graph: FactorGraph,
  values: Values,
  index: BlockIndex,
  lambda: number,
  currentCost?: number,
): { values: Values; cost: number; accepted: boolean; gain: number; stepNorm: number; system: System } {
  const cost0 = currentCost ?? graphCost(graph, values);
  const stepped = optimizeStep(graph, values, index, lambda);
  const cost1 = graphCost(graph, stepped.values);
  const d = stepped.delta;
  // Predicted reduction from the damped quadratic model: ½ Δᵀ(λ diag(Ω) Δ − b).
  let predicted = 0;
  for (let i = 0; i < d.length; i++) {
    predicted += 0.5 * d[i] * (lambda * Math.max(stepped.system.Omega[i][i], 1e-9) * d[i] - stepped.system.b[i]);
  }
  const gain = predicted > 0 ? (cost0 - cost1) / predicted : cost0 - cost1;
  const accepted = cost1 < cost0;
  return {
    values: accepted ? stepped.values : values,
    cost: cost1,
    accepted,
    gain,
    stepNorm: Math.sqrt(d.reduce((s, x) => s + x * x, 0)),
    system: stepped.system,
  };
}

/* -------------------------------------------------------------------------- */
/* Covariance and diagnostics                                                  */
/* -------------------------------------------------------------------------- */

/**
 * The marginal covariance of one variable: the diagonal block of Ω⁻¹ at the
 * optimum — the Laplace approximation of the posterior. Inverting the whole Ω
 * is the wrong way to do this at scale (Chapter 16 covers selected inversion);
 * at widget scale it is the honest one-liner.
 */
export function marginalCovariance(system: System, index: BlockIndex, key: VarKey): Mat {
  const s = slotOf(index, key);
  if (s < 0) return eye(VAR_DIM[key.kind], 0);
  const n = index.total;
  const damped = system.Omega.map((row, i) => row.map((x, j) => (i === j ? x + 1e-9 : x)));
  const Sigma = symmetrize(inv(damped));
  const o = index.offsets[s];
  const d = index.dims[s];
  const out = zerosMat(d, d);
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) out[i][j] = Sigma[o + i][o + j];
  void n;
  return out;
}

/** Root-mean-square position error against a ground-truth trajectory. */
export function poseRmse(estimate: Pose2[], truth: Pose2[]): number {
  const n = Math.min(estimate.length, truth.length);
  if (n === 0) return 0;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const dx = estimate[i].x - truth[i].x;
    const dy = estimate[i].y - truth[i].y;
    acc += dx * dx + dy * dy;
  }
  return Math.sqrt(acc / n);
}
