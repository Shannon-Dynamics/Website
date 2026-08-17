/**
 * EKF localization with a feature map — Thrun et al., **Tables 7.2 and 7.3**.
 *
 * Nothing here is a new filter. It is Chapter 7's EKF with Chapter 9's velocity
 * motion model and Chapter 10's range–bearing landmark model substituted in,
 * plus the one genuinely new ingredient of Chapter 11: the *discrete* variable
 * cᵗ that says which landmark a feature came from.
 *
 * Two design decisions are worth stating up front because they are what the
 * chapter is about.
 *
 *  1. **Distances live in the metric of the innovation covariance.** Every
 *     comparison between a measurement and a prediction goes through
 *     Sʲ = Hʲ Σ̄ Hʲᵀ + Q, never through metres. A landmark far away in metres
 *     can be near in this metric, and vice versa.
 *  2. **The log-determinant term is kept.** Maximizing 𝒩(z; ẑʲ, Sʲ) is *not*
 *     the same as minimizing the Mahalanobis distance when the candidates have
 *     different Sʲ — the two disagree exactly when one candidate's gate is much
 *     larger than another's, which is the common case near the sensor.
 *
 * The mean is stored as a `Pose2` and the covariance as a 3×3 in (x, y, θ)
 * coordinates, matching Thrun's tables. The heading is wrapped on every write,
 * and bearing residuals go through `angleDiff` — the manifold discipline of
 * Chapter 7, applied here at the two places where the 2005 presentation is
 * silently wrong near ±π.
 */

import { angleDiff, normalizeAngle, type Pose2 } from '../geom/se2';
import type { MotionAlphas, VelocityCmd } from '../models/motion';
import type { Landmark } from '../sim/world';
import { LOG_2PI } from '../prob/gaussian';
import {
  eye,
  matAdd,
  matMul,
  matSub,
  matVec,
  symmetrize,
  transpose,
  zerosMat,
  type Mat,
  type Vec,
} from '../prob/linalg';
import { chi2Quantile } from './consistency';

/** Below this |ω| the arc formulae are 0/0 and we take the straight-line limit. */
const STRAIGHT = 1e-6;

/* -------------------------------------------------------------------------- */
/* Types                                                                       */
/* -------------------------------------------------------------------------- */

/** A Gaussian over pose: mean on SE(2), covariance in (x, y, θ). */
export interface PoseBelief {
  mu: Pose2;
  Sigma: Mat;
}

/** One detected feature. `truth` is simulation bookkeeping, never read by the filter. */
export interface Feature {
  r: number;
  phi: number;
  /** Ground-truth landmark id — used only to score association accuracy. */
  truth?: number;
}

/** Everything the filter predicts about one map landmark, before it sees anything. */
export interface LandmarkPrediction {
  /** Index into the landmark array (not the landmark's `id`). */
  index: number;
  /** ẑʲ = (√q, atan2(δy, δx) − θ). */
  zHat: [number, number];
  /** 2×3 measurement Jacobian ∂h/∂x at μ̄. */
  H: Mat;
  /** Innovation covariance Sʲ = Hʲ Σ̄ Hʲᵀ + Q. */
  S: Mat;
  /** log det Sʲ, cached because the ML score needs it every comparison. */
  logDetS: number;
  /** Where the landmark actually is, for drawing. */
  landmark: Landmark;
}

export type Association =
  | { kind: 'match'; index: number; d2: number; score: number }
  | { kind: 'outlier'; nearest: number; d2: number; score: number };

export interface UpdateInfo {
  /** ν = z ⊟ ẑ, with the bearing component wrapped. */
  innovation: Vec;
  S: Mat;
  K: Mat;
  /** νᵀS⁻¹ν — the NIS statistic, and the gate test. */
  d2: number;
  logLikelihood: number;
}

/** How a measurement is compared against a prediction. */
export type AssociationMetric = 'mahalanobis' | 'euclidean';

export interface AssociateOptions {
  /** χ² gate on d²; a feature whose best d² exceeds it is declared an outlier. */
  gate2: number;
  metric?: AssociationMetric;
  /** Keep the log det Sʲ term of the ML score. Dropping it is the common bug. */
  useLogDet?: boolean;
}

/* -------------------------------------------------------------------------- */
/* Small 2×2 helpers — the innovation space is two-dimensional and stays that   */
/* way, so closed forms beat a general solver and never hit a pivot problem.    */
/* -------------------------------------------------------------------------- */

export function det2(S: Mat): number {
  return S[0][0] * S[1][1] - S[0][1] * S[1][0];
}

/** νᵀ S⁻¹ ν for 2×2 S, by Cramer's rule. */
export function mahalanobis2(nu: Vec, S: Mat): number {
  const d = det2(S);
  if (Math.abs(d) < 1e-15) return Number.POSITIVE_INFINITY;
  const [a, b] = [S[0][0], S[0][1]];
  const [c, e] = [S[1][0], S[1][1]];
  // S⁻¹ = (1/det) [[e, −b], [−c, a]]
  const ix = (e * nu[0] - b * nu[1]) / d;
  const iy = (-c * nu[0] + a * nu[1]) / d;
  return Math.max(nu[0] * ix + nu[1] * iy, 0);
}

export function inv2(S: Mat): Mat {
  const d = det2(S);
  return [
    [S[1][1] / d, -S[0][1] / d],
    [-S[1][0] / d, S[0][0] / d],
  ];
}

/**
 * The χ² gate threshold γ = χ²_{dim, 1−ε}.
 *
 * dim = 2 for a range–bearing feature: γ ≈ 5.99 at 95%, 9.21 at 99%. The
 * signature adds a third row that is identically zero in H, so it contributes
 * no degrees of freedom to the *pose* update even when it is measured.
 */
export function gateThreshold(confidence = 0.95, dim = 2): number {
  return chi2Quantile(confidence, dim);
}

/* -------------------------------------------------------------------------- */
/* Jacobians — Chapter 9's velocity model, differentiated                       */
/* -------------------------------------------------------------------------- */

/**
 * Control-noise covariance M_t = diag(α₁v² + α₂ω², α₃v² + α₄ω²).
 *
 * Noise lives in *control* space and is rank 2, which is why it has to be
 * pushed into pose space by V_t rather than written down as a 3×3 R_t.
 */
export function controlNoise(u: VelocityCmd, alphas: MotionAlphas): Mat {
  const [a1, a2, a3, a4] = alphas;
  const v2 = u.v * u.v;
  const w2 = u.omega * u.omega;
  return [
    [a1 * v2 + a2 * w2, 0],
    [0, a3 * v2 + a4 * w2],
  ];
}

/**
 * G_t = ∂g/∂x and V_t = ∂g/∂u for the exact-arc velocity model, evaluated at
 * μ_{t−1}. Thrun et al., Table 7.2.
 *
 * The ω → 0 branch is not cosmetic: both matrices divide by ω, and a robot
 * driving straight is the most common command there is. Taking the limits
 * analytically (rather than letting v/ω blow up) is the difference between a
 * filter that survives a corridor and one that NaNs in it.
 */
export function velocityJacobians(mu: Pose2, u: VelocityCmd): { G: Mat; V: Mat } {
  const { v, omega: w, dt } = u;
  const th = mu.theta;
  const c = Math.cos(th);
  const s = Math.sin(th);

  if (Math.abs(w) < STRAIGHT) {
    return {
      G: [
        [1, 0, -v * dt * s],
        [0, 1, v * dt * c],
        [0, 0, 1],
      ],
      V: [
        [dt * c, -0.5 * v * dt * dt * s],
        [dt * s, 0.5 * v * dt * dt * c],
        [0, dt],
      ],
    };
  }

  const nt = th + w * dt;
  const cn = Math.cos(nt);
  const sn = Math.sin(nt);
  const r = v / w;

  return {
    G: [
      [1, 0, r * (-c + cn)],
      [0, 1, r * (-s + sn)],
      [0, 0, 1],
    ],
    V: [
      [(-s + sn) / w, (v * (s - sn)) / (w * w) + (v * cn * dt) / w],
      [(c - cn) / w, (-v * (c - cn)) / (w * w) + (v * sn * dt) / w],
      [0, dt],
    ],
  };
}

/** g(u, x): the noise-free exact arc. Same integration as `diffDriveStep`. */
export function velocityStep(mu: Pose2, u: VelocityCmd): Pose2 {
  const { v, omega: w, dt } = u;
  const nt = mu.theta + w * dt;
  if (Math.abs(w) < STRAIGHT) {
    return {
      x: mu.x + v * dt * Math.cos(mu.theta),
      y: mu.y + v * dt * Math.sin(mu.theta),
      theta: normalizeAngle(nt),
    };
  }
  const r = v / w;
  return {
    x: mu.x - r * Math.sin(mu.theta) + r * Math.sin(nt),
    y: mu.y + r * Math.cos(mu.theta) - r * Math.cos(nt),
    theta: normalizeAngle(nt),
  };
}

/**
 * Table 7.2, lines 2–4: μ̄ = g(u, μ), Σ̄ = G Σ Gᵀ + V M Vᵀ.
 *
 * `extraNoise` is an optional additive 3×3 floor. Real deployments always have
 * one — it is where "the wheels also slip sideways sometimes" goes, and it
 * keeps Σ̄ from becoming singular when the robot is commanded to stand still
 * (v = ω = 0 makes M identically zero).
 */
export function ekfLocalizationPredict(
  bel: PoseBelief,
  u: VelocityCmd,
  alphas: MotionAlphas,
  extraNoise?: Mat,
): PoseBelief {
  const { G, V } = velocityJacobians(bel.mu, u);
  const M = controlNoise(u, alphas);
  const R = matMul(matMul(V, M), transpose(V));
  let Sigma = matAdd(matMul(matMul(G, bel.Sigma), transpose(G)), R);
  if (extraNoise) Sigma = matAdd(Sigma, extraNoise);
  return { mu: velocityStep(bel.mu, u), Sigma: symmetrize(Sigma) };
}

/* -------------------------------------------------------------------------- */
/* Measurement prediction — Chapter 10's landmark model, differentiated         */
/* -------------------------------------------------------------------------- */

/**
 * ẑʲ, Hʲ and Sʲ for one landmark, at the current μ̄ — Table 7.2, lines 8–12.
 *
 *   H = (1/q) · [ −√q δx   −√q δy    0 ]
 *               [    δy      −δx    −q ]
 *
 * Read the second row: the bearing's sensitivity to position falls off as 1/√q
 * while its sensitivity to heading is exactly −1 whatever the range. That is
 * why a distant landmark is nearly a pure heading measurement, and a close one
 * is nearly a pure position measurement.
 */
export function predictMeasurement(
  bel: PoseBelief,
  landmark: Landmark,
  Q: Mat,
  index = 0,
): LandmarkPrediction {
  const dx = landmark.x - bel.mu.x;
  const dy = landmark.y - bel.mu.y;
  const q = dx * dx + dy * dy;
  const sq = Math.sqrt(Math.max(q, 1e-12));

  const H: Mat = [
    [-dx / sq, -dy / sq, 0],
    [dy / Math.max(q, 1e-12), -dx / Math.max(q, 1e-12), -1],
  ];
  const S = symmetrize(matAdd(matMul(matMul(H, bel.Sigma), transpose(H)), Q));

  return {
    index,
    zHat: [sq, normalizeAngle(Math.atan2(dy, dx) - bel.mu.theta)],
    H,
    S,
    logDetS: Math.log(Math.max(det2(S), 1e-300)),
    landmark,
  };
}

/** Every landmark's prediction, in map order. */
export function predictAll(bel: PoseBelief, landmarks: Landmark[], Q: Mat): LandmarkPrediction[] {
  return landmarks.map((lm, i) => predictMeasurement(bel, lm, Q, i));
}

/** ν = z ⊟ ẑ. The bearing component wraps; the range component does not. */
export function featureInnovation(z: Feature, zHat: [number, number]): Vec {
  return [z.r - zHat[0], angleDiff(z.phi, zHat[1])];
}

/* -------------------------------------------------------------------------- */
/* Data association                                                            */
/* -------------------------------------------------------------------------- */

/**
 * `ml_associate` — maximum-likelihood correspondence with a χ² gate.
 *
 * The score being minimized is −2 log 𝒩(z; ẑʲ, Sʲ) up to a constant:
 *
 *      score(j) = d²_M(z, ẑʲ) + log det Sʲ
 *
 * Thrun's Table 7.3 minimizes only the first term. That is the right thing to
 * do when all candidates share an Sʲ and the wrong thing otherwise: a landmark
 * whose gate is twice as wide in each axis carries a log det penalty of
 * 2 log 4 ≈ 2.77, which is worth more than a full standard deviation of
 * mismatch. `useLogDet: false` reproduces the book's version so the two can be
 * compared directly.
 *
 * `metric: 'euclidean'` is the straw man the chapter argues against: it scores
 * candidates by the metric distance between the measurement's projected
 * endpoint and the landmark, ignoring the shape of the uncertainty entirely.
 */
export function mlAssociate(
  z: Feature,
  preds: LandmarkPrediction[],
  opts: AssociateOptions,
): Association {
  const { gate2, metric = 'mahalanobis', useLogDet = true } = opts;

  let bestScore = Number.POSITIVE_INFINITY;
  let bestIndex = -1;
  let bestD2 = Number.POSITIVE_INFINITY;

  for (const p of preds) {
    const nu = featureInnovation(z, p.zHat);
    const d2 = mahalanobis2(nu, p.S);

    let score: number;
    if (metric === 'euclidean') {
      // Distance in the plane between where the measurement says the feature is
      // and where the landmark is. Units: metres, and radians pretending to be
      // metres — which is the whole objection.
      const dx = z.r * Math.cos(z.phi) - p.zHat[0] * Math.cos(p.zHat[1]);
      const dy = z.r * Math.sin(z.phi) - p.zHat[0] * Math.sin(p.zHat[1]);
      score = Math.hypot(dx, dy);
    } else {
      score = useLogDet ? d2 + p.logDetS : d2;
    }

    if (score < bestScore) {
      bestScore = score;
      bestIndex = p.index;
      bestD2 = d2;
    }
  }

  if (bestIndex < 0) return { kind: 'outlier', nearest: -1, d2: Infinity, score: Infinity };
  // The gate is always evaluated in the Mahalanobis metric, even when the
  // *choice* was made in metres: a gate in metres has no calibrated meaning.
  if (bestD2 > gate2) {
    return { kind: 'outlier', nearest: bestIndex, d2: bestD2, score: bestScore };
  }
  return { kind: 'match', index: bestIndex, d2: bestD2, score: bestScore };
}

/* -------------------------------------------------------------------------- */
/* The localizer                                                               */
/* -------------------------------------------------------------------------- */

export interface EkfLocalizerOptions {
  landmarks: Landmark[];
  /** 2×2 measurement noise diag(σ_r², σ_φ²). */
  Q: Mat;
  alphas: MotionAlphas;
  gate2?: number;
  metric?: AssociationMetric;
  useLogDet?: boolean;
  /**
   * Re-linearize H and re-evaluate the gates after every accepted feature.
   * Table 7.2 sums the corrections instead; sequential is strictly better and
   * costs nothing, so it is the default here and the difference is an exercise.
   */
  sequential?: boolean;
  extraNoise?: Mat;
}

/**
 * `EKF_localization` — Tables 7.2 (known correspondence) and 7.3 (ML).
 *
 * The two entry points differ by exactly one line: whether cᵗ is handed in or
 * inferred. Everything downstream is identical, which is the chapter's point.
 */
export class EkfLocalizer {
  mu: Pose2;
  Sigma: Mat;
  readonly landmarks: Landmark[];
  Q: Mat;
  alphas: MotionAlphas;
  gate2: number;
  metric: AssociationMetric;
  useLogDet: boolean;
  sequential: boolean;
  extraNoise?: Mat;

  constructor(bel: PoseBelief, opts: EkfLocalizerOptions) {
    this.mu = { ...bel.mu };
    this.Sigma = bel.Sigma.map((r) => r.slice());
    this.landmarks = opts.landmarks;
    this.Q = opts.Q;
    this.alphas = opts.alphas;
    this.gate2 = opts.gate2 ?? gateThreshold(0.95, 2);
    this.metric = opts.metric ?? 'mahalanobis';
    this.useLogDet = opts.useLogDet ?? true;
    this.sequential = opts.sequential ?? true;
    this.extraNoise = opts.extraNoise;
  }

  belief(): PoseBelief {
    return { mu: { ...this.mu }, Sigma: this.Sigma.map((r) => r.slice()) };
  }

  predict(u: VelocityCmd): void {
    const next = ekfLocalizationPredict(this.belief(), u, this.alphas, this.extraNoise);
    this.mu = next.mu;
    this.Sigma = next.Sigma;
  }

  /** Table 7.2, lines 8–15, for one feature whose correspondence is given. */
  correctKnown(z: Feature, index: number): UpdateInfo {
    const p = predictMeasurement(this.belief(), this.landmarks[index], this.Q, index);
    return this.applyUpdate(z, p);
  }

  /**
   * Table 7.3: predict every landmark, associate each feature by ML, apply the
   * accepted ones. Returns one `Association` per feature so a widget can draw
   * what was accepted, what was rejected, and how close the call was.
   */
  correct(features: Feature[]): { associations: Association[]; updates: UpdateInfo[] } {
    const associations: Association[] = [];
    const updates: UpdateInfo[] = [];
    let preds = predictAll(this.belief(), this.landmarks, this.Q);

    for (const z of features) {
      const a = mlAssociate(z, preds, {
        gate2: this.gate2,
        metric: this.metric,
        useLogDet: this.useLogDet,
      });
      associations.push(a);
      if (a.kind !== 'match') continue;

      updates.push(this.applyUpdate(z, preds[a.index]));
      // Re-linearizing here is what makes the *next* feature's gate reflect the
      // information this one just added — the reason ordering stops mattering.
      if (this.sequential) preds = predictAll(this.belief(), this.landmarks, this.Q);
    }
    return { associations, updates };
  }

  /** K = Σ̄ Hᵀ S⁻¹; μ ← μ ⊞ Kν; Σ ← (I − KH) Σ̄, symmetrized. */
  private applyUpdate(z: Feature, p: LandmarkPrediction): UpdateInfo {
    const nu = featureInnovation(z, p.zHat);
    const Ht = transpose(p.H);
    const K = matMul(matMul(this.Sigma, Ht), inv2(p.S));
    const dx = matVec(K, nu);

    this.mu = {
      x: this.mu.x + dx[0],
      y: this.mu.y + dx[1],
      theta: normalizeAngle(this.mu.theta + dx[2]),
    };
    this.Sigma = symmetrize(matMul(matSub(eye(3), matMul(K, p.H)), this.Sigma));

    const d2 = mahalanobis2(nu, p.S);
    return {
      innovation: nu,
      S: p.S,
      K,
      d2,
      logLikelihood: -0.5 * (d2 + p.logDetS + 2 * LOG_2PI),
    };
  }
}

/* -------------------------------------------------------------------------- */
/* Multi-hypothesis tracking                                                   */
/* -------------------------------------------------------------------------- */

export interface Hypothesis {
  mu: Pose2;
  Sigma: Mat;
  /** Normalized mixture weight. */
  w: number;
  /** Association history, one landmark index per accepted feature. */
  history: number[];
  /** Stable identity so a tree diagram can follow a branch across steps. */
  id: number;
  /** The hypothesis this one branched from, or −1 at the root. */
  parent: number;
}

export interface MhtOptions extends EkfLocalizerOptions {
  /** Drop hypotheses whose weight is below this fraction of the best. */
  pruneRatio?: number;
  maxHyps?: number;
  /** Merge two hypotheses when their means are closer than this in d². */
  mergeD2?: number;
  /**
   * Likelihood a hypothesis pays for explaining a feature as clutter rather
   * than as a landmark. Without it, a hypothesis that can explain *nothing*
   * survives forever at its old weight, and the mixture never collapses.
   */
  clutterDensity?: number;
}

/**
 * `mht_localization` — the belief as a Gaussian mixture over association
 * histories.
 *
 * Every ambiguous feature multiplies the number of hypotheses by the number of
 * landmarks inside its gate, so the exact posterior is a tree of size O(Jᵀ).
 * Pruning is what makes it finite, and pruning is also what makes it *wrong* —
 * the mass thrown away is gone, and if the true history was in it, no amount of
 * later evidence brings it back. That trade is the whole algorithm.
 */
export class MhtLocalizer {
  hyps: Hypothesis[];
  readonly landmarks: Landmark[];
  Q: Mat;
  alphas: MotionAlphas;
  gate2: number;
  pruneRatio: number;
  maxHyps: number;
  mergeD2: number;
  clutterDensity: number;
  extraNoise?: Mat;
  /** Total hypotheses ever created — the cost counter the widget displays. */
  born = 1;
  /** Association histories that were pruned, newest last, for the tree diagram. */
  pruned: number[][] = [];
  private nextId = 1;

  constructor(bel: PoseBelief, opts: MhtOptions) {
    this.hyps = [
      { mu: { ...bel.mu }, Sigma: bel.Sigma.map((r) => r.slice()), w: 1, history: [], id: 0, parent: -1 },
    ];
    this.landmarks = opts.landmarks;
    this.Q = opts.Q;
    this.alphas = opts.alphas;
    this.gate2 = opts.gate2 ?? gateThreshold(0.95, 2);
    this.pruneRatio = opts.pruneRatio ?? 0.01;
    this.maxHyps = opts.maxHyps ?? 8;
    this.mergeD2 = opts.mergeD2 ?? 0.2;
    this.clutterDensity = opts.clutterDensity ?? 0.02;
    this.extraNoise = opts.extraNoise;
  }

  predict(u: VelocityCmd): void {
    this.hyps = this.hyps.map((h) => {
      const next = ekfLocalizationPredict({ mu: h.mu, Sigma: h.Sigma }, u, this.alphas, this.extraNoise);
      return { ...h, mu: next.mu, Sigma: next.Sigma };
    });
  }

  /**
   * Branch on every landmark inside the gate, weight each branch by the
   * predictive likelihood 𝒩(z; ẑ, S), then normalize, prune, cap, and merge.
   */
  correct(features: Feature[]): void {
    for (const z of features) {
      const children: Hypothesis[] = [];

      for (const h of this.hyps) {
        const preds = predictAll({ mu: h.mu, Sigma: h.Sigma }, this.landmarks, this.Q);
        let branched = false;

        for (const p of preds) {
          const nu = featureInnovation(z, p.zHat);
          const d2 = mahalanobis2(nu, p.S);
          if (d2 > this.gate2) continue;

          const Ht = transpose(p.H);
          const K = matMul(matMul(h.Sigma, Ht), inv2(p.S));
          const dx = matVec(K, nu);
          const Sigma = symmetrize(matMul(matSub(eye(3), matMul(K, p.H)), h.Sigma));
          // log 𝒩(ν; 0, S) — the weight recursion, in logs to stay finite.
          const logLik = -0.5 * (d2 + p.logDetS + 2 * LOG_2PI);

          children.push({
            mu: {
              x: h.mu.x + dx[0],
              y: h.mu.y + dx[1],
              theta: normalizeAngle(h.mu.theta + dx[2]),
            },
            Sigma,
            w: h.w * Math.exp(logLik),
            history: [...h.history, p.index],
            id: this.nextId++,
            parent: h.id,
          });
          branched = true;
        }

        // A hypothesis with nothing inside its gate explains the feature as
        // clutter. It survives — a single outlier must not kill the truth — but
        // it pays the clutter density, so a hypothesis that keeps failing to
        // explain the data decays away instead of living forever.
        if (!branched) {
          children.push({ ...h, w: h.w * this.clutterDensity, id: this.nextId++, parent: h.id });
        }
      }

      this.born += children.length;
      this.hyps = this.reduce(children);
    }
  }

  /** Normalize → merge near-duplicates → prune by ratio → cap → renormalize. */
  private reduce(hyps: Hypothesis[]): Hypothesis[] {
    if (hyps.length === 0) return hyps;
    let out = normalizeWeights(hyps);

    // Merge: two hypotheses whose means agree within mergeD2 are the same
    // belief wearing two different histories.
    const merged: Hypothesis[] = [];
    for (const h of out) {
      const twin = merged.find((m) => poseD2(m, h) < this.mergeD2);
      if (twin) twin.w += h.w;
      else merged.push({ ...h });
    }
    out = merged;

    const best = out.reduce((m, h) => Math.max(m, h.w), 0);
    const kept = out.filter((h) => h.w >= best * this.pruneRatio);
    kept.sort((a, b) => b.w - a.w);
    const survivors = kept.slice(0, this.maxHyps);

    // Remember what was thrown away, so the tree can gray out the branches the
    // filter can no longer recover — the honest price of pruning.
    const alive = new Set(survivors.map((h) => h.history.join('/')));
    for (const h of out) {
      const key = h.history.join('/');
      if (!alive.has(key)) this.pruned.push(h.history);
    }
    if (this.pruned.length > 32) this.pruned = this.pruned.slice(-32);

    return normalizeWeights(survivors);
  }

  /** The mixture's dominant component — what a downstream planner would use. */
  best(): Hypothesis {
    return this.hyps.reduce((m, h) => (h.w > m.w ? h : m), this.hyps[0]);
  }
}

function normalizeWeights(hyps: Hypothesis[]): Hypothesis[] {
  const total = hyps.reduce((s, h) => s + h.w, 0);
  if (!(total > 0)) return hyps.map((h) => ({ ...h, w: 1 / hyps.length }));
  return hyps.map((h) => ({ ...h, w: h.w / total }));
}

/** Squared distance between two hypothesis means, scaled so 1 ≈ "the same". */
function poseD2(a: Hypothesis, b: Hypothesis): number {
  const dx = a.mu.x - b.mu.x;
  const dy = a.mu.y - b.mu.y;
  const dth = angleDiff(a.mu.theta, b.mu.theta);
  return dx * dx + dy * dy + dth * dth;
}

/* -------------------------------------------------------------------------- */
/* Diagnostics                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * NEES for a pose belief: eᵀ Σ⁻¹ e with e = (Δx, Δy, θ ⊟ μ_θ).
 *
 * A consistent 3-DOF filter averages 3. The heading component goes through
 * `angleDiff`, without which a filter crossing ±π reports a NEES of ~4π²/σ²
 * and looks catastrophically broken while being perfectly fine.
 */
export function poseNees(truth: Pose2, bel: PoseBelief): number {
  const e: Vec = [truth.x - bel.mu.x, truth.y - bel.mu.y, angleDiff(truth.theta, bel.mu.theta)];
  // 3×3 solve by cofactors — the matrix is tiny and always symmetric PD here.
  const S = bel.Sigma;
  const a = S[0][0];
  const b = S[0][1];
  const c = S[0][2];
  const d = S[1][1];
  const f = S[1][2];
  const g = S[2][2];
  const det = a * (d * g - f * f) - b * (b * g - f * c) + c * (b * f - d * c);
  if (Math.abs(det) < 1e-18) return Number.POSITIVE_INFINITY;
  const inv: Mat = [
    [(d * g - f * f) / det, (c * f - b * g) / det, (b * f - c * d) / det],
    [(c * f - b * g) / det, (a * g - c * c) / det, (b * c - a * f) / det],
    [(b * f - c * d) / det, (b * c - a * f) / det, (a * d - b * b) / det],
  ];
  const y = matVec(inv, e);
  return Math.max(e[0] * y[0] + e[1] * y[1] + e[2] * y[2], 0);
}

/**
 * The 2×2 position block of a pose covariance — what the map-pane ellipse draws.
 * Heading uncertainty is real but it is not a shape on the floor.
 */
export function positionBlock(Sigma: Mat): Mat {
  return [
    [Sigma[0][0], Sigma[0][1]],
    [Sigma[1][0], Sigma[1][1]],
  ];
}

/** An empty 3×3, for callers that want a zero process-noise floor. */
export const ZERO3: Mat = zerosMat(3, 3);
