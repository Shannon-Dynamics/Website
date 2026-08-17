/**
 * FastSLAM 1.0 / 2.0 for landmark maps.
 *
 * Montemerlo, Thrun, Koller & Wegbreit, AAAI-02 (1.0) and IJCAI-03 (2.0).
 *
 * The factorization theorem says that *given the path*, landmarks are
 * conditionally independent. So a particle carries a trajectory plus a bank of
 * tiny 2×2 EKFs — one per landmark — and the joint (3 + 2N)-dimensional
 * posterior of Chapter 14 is replaced by M cheap path hypotheses, each towing
 * N independent Gaussians.
 *
 * Two proposals are implemented behind one enum, because the difference between
 * them is the pedagogical heart of the chapter:
 *
 *   `motion-prior`      — FastSLAM 1.0. Sample p(x_t | x_{t−1}, u_t), then
 *                         weight by the measurement's marginal likelihood.
 *   `measurement-aware` — FastSLAM 2.0. Fold z_t into the proposal itself with
 *                         one EKF-style step, and weight by the *predictive*
 *                         likelihood, which no longer depends on where inside
 *                         the proposal the sample landed.
 *
 * Data association is per particle: each universe may match the same reading to
 * a different landmark, which is the one thing a single-hypothesis EKF
 * (Chapter 11) structurally cannot do.
 */

import { angleDiff, normalizeAngle, type Pose2 } from '../geom/se2';
import { applyOdom, sampleMotionModelOdometry, type OdomAlphas, type OdomDelta } from '../models/motion';
import type { RangeBearingFeature } from '../models/sensor';
import { logMvnPdf } from '../prob/gaussian';
import { cholesky, inv, matMul, transpose, type Mat } from '../prob/linalg';
import type { Rng } from '../prob/rng';
// Shared particle bookkeeping — introduced with the grid RBPF, reused here.
import { effectiveSampleSize, normalizeLogWeights, resampleIndices } from './rbpf';

// ---------------------------------------------------------------------------
// One landmark, one 2×2 EKF
// ---------------------------------------------------------------------------

export interface LandmarkEkf {
  /** Index inside this particle's map. Particles need not agree on it. */
  slot: number;
  mu: [number, number];
  sigma: Mat;
  /** Sightings so far — FastSLAM's cheap stand-in for existence evidence. */
  hits: number;
  /**
   * Simulation-only bookkeeping: which true landmark created this EKF. The
   * filter never reads it; the widgets use it to count association errors.
   */
  trueId: number;
}

/** Predicted range/bearing of a landmark from a pose — the h(x, m) of the EKF. */
export function predictObservation(pose: Pose2, mu: [number, number]) {
  const dx = mu[0] - pose.x;
  const dy = mu[1] - pose.y;
  const q = dx * dx + dy * dy;
  const r = Math.sqrt(Math.max(q, 1e-12));
  return {
    r,
    phi: normalizeAngle(Math.atan2(dy, dx) - pose.theta),
    dx,
    dy,
    q: Math.max(q, 1e-12),
  };
}

/** ∂h/∂m, the 2×2 Jacobian in the landmark's coordinates. */
export function landmarkJacobian(pose: Pose2, mu: [number, number]): Mat {
  const { dx, dy, q, r } = predictObservation(pose, mu);
  return [
    [dx / r, dy / r],
    [-dy / q, dx / q],
  ];
}

/** ∂h/∂x, the 2×3 Jacobian in the pose — only FastSLAM 2.0 needs this one. */
export function poseJacobian(pose: Pose2, mu: [number, number]): Mat {
  const { dx, dy, q, r } = predictObservation(pose, mu);
  return [
    [-dx / r, -dy / r, 0],
    [dy / q, -dx / q, -1],
  ];
}

/**
 * Initialize a landmark from its first sighting: invert the measurement, then
 * push the sensor covariance through the inverse Jacobian, Σ = H⁻¹ Q H⁻ᵀ.
 *
 * Note what this does *not* do: it does not represent the pose uncertainty. It
 * cannot, and it does not need to — inside this particle the pose is known
 * exactly, by construction. That is Rao-Blackwellization paying out.
 */
export function initLandmark(
  pose: Pose2,
  z: RangeBearingFeature,
  q: Mat,
  slot: number,
  trueId: number,
): LandmarkEkf {
  const bearing = normalizeAngle(pose.theta + z.phi);
  const mu: [number, number] = [
    pose.x + z.r * Math.cos(bearing),
    pose.y + z.r * Math.sin(bearing),
  ];
  const h = landmarkJacobian(pose, mu);
  const hInv = inv(h);
  return {
    slot,
    mu,
    sigma: matMul(matMul(hInv, q), transpose(hInv)),
    hits: 1,
    trueId,
  };
}

export interface LandmarkUpdate {
  /** Innovation ν = z − ẑ, with the bearing wrapped. */
  nu: [number, number];
  /** Innovation covariance S = H Σ Hᵀ + Q — Chapter 11's gate, unchanged. */
  s: Mat;
  logLikelihood: number;
}

/** The measurement's marginal likelihood under this landmark's Gaussian. */
export function landmarkInnovation(
  lm: LandmarkEkf,
  pose: Pose2,
  z: RangeBearingFeature,
  q: Mat,
): LandmarkUpdate {
  const pred = predictObservation(pose, lm.mu);
  const h = landmarkJacobian(pose, lm.mu);
  const s = addMat(matMul(matMul(h, lm.sigma), transpose(h)), q);
  const nu: [number, number] = [z.r - pred.r, angleDiff(z.phi, pred.phi)];
  return { nu, s, logLikelihood: logMvnPdf(nu, [0, 0], s) };
}

/** Standard EKF correction of one landmark, in place. */
export function updateLandmark(
  lm: LandmarkEkf,
  pose: Pose2,
  z: RangeBearingFeature,
  q: Mat,
): LandmarkUpdate {
  const upd = landmarkInnovation(lm, pose, z, q);
  const h = landmarkJacobian(pose, lm.mu);
  const k = matMul(matMul(lm.sigma, transpose(h)), inv(upd.s));
  lm.mu = [
    lm.mu[0] + k[0][0] * upd.nu[0] + k[0][1] * upd.nu[1],
    lm.mu[1] + k[1][0] * upd.nu[0] + k[1][1] * upd.nu[1],
  ];
  const kh = matMul(k, h);
  lm.sigma = [
    [
      (1 - kh[0][0]) * lm.sigma[0][0] - kh[0][1] * lm.sigma[1][0],
      (1 - kh[0][0]) * lm.sigma[0][1] - kh[0][1] * lm.sigma[1][1],
    ],
    [
      -kh[1][0] * lm.sigma[0][0] + (1 - kh[1][1]) * lm.sigma[1][0],
      -kh[1][0] * lm.sigma[0][1] + (1 - kh[1][1]) * lm.sigma[1][1],
    ],
  ];
  lm.hits += 1;
  return upd;
}

const addMat = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((v, j) => v + b[i][j]));

// ---------------------------------------------------------------------------
// Particles
// ---------------------------------------------------------------------------

export type ProposalKind = 'motion-prior' | 'measurement-aware';

export interface FsParticle {
  pose: Pose2;
  /** The particle IS a path (factorization theorem). */
  path: Pose2[];
  landmarks: LandmarkEkf[];
  logWeight: number;
  ancestor: number;
}

export interface FastSlamOptions {
  proposal: ProposalKind;
  alphas: OdomAlphas;
  /** Measurement noise Q = diag(σ_r², σ_φ²). */
  sigmaR: number;
  sigmaPhi: number;
  /** Given the true correspondence, or forced to guess it per particle. */
  knownCorrespondence: boolean;
  /**
   * p₀ in Thrun's `FastSLAM_unknown_correspondence`: the likelihood a *new*
   * landmark is credited with. Any existing landmark scoring below it loses.
   */
  newLandmarkLikelihood: number;
  /** 95% χ²₂ gate — associations beyond it are never considered. */
  gate: number;
  neffRatio: number;
  /** Off = resample every step, whatever the weights look like. */
  selectiveResampling: boolean;
}

export const DEFAULT_FASTSLAM_OPTIONS: FastSlamOptions = {
  proposal: 'measurement-aware',
  alphas: [0.02, 0.008, 0.03, 0.01],
  sigmaR: 0.12,
  sigmaPhi: 0.035,
  knownCorrespondence: false,
  newLandmarkLikelihood: 1e-3,
  gate: 9.21,
  neffRatio: 0.5,
  selectiveResampling: true,
};

export interface FsReport {
  neff: number;
  resampled: boolean;
  distinctAncestors: number;
  weights: number[];
  bestIndex: number;
  /** New landmarks created this step, summed over particles. */
  newLandmarks: number;
  /** Associations that picked an EKF built from a different true landmark. */
  associationErrors: number;
}

/**
 * Motion noise pushed into pose coordinates: R = V M Vᵀ.
 *
 * M is diagonal in (δrot1, δtrans, δrot2) — the α model of Chapter 9 — and V
 * is the Jacobian of the rotate–translate–rotate composition. This is the same
 * construction Chapter 11's EKF localization uses; FastSLAM 2.0 needs it
 * because its proposal has to know how wide the motion prior actually is.
 */
export function odometryPoseCovariance(
  pose: Pose2,
  u: OdomDelta,
  alphas: OdomAlphas,
): Mat {
  const [a1, a2, a3, a4] = alphas;
  const r1sq = u.rot1 * u.rot1;
  const r2sq = u.rot2 * u.rot2;
  const tsq = u.trans * u.trans;
  const m = [a1 * r1sq + a2 * tsq, a3 * tsq + a4 * (r1sq + r2sq), a1 * r2sq + a2 * tsq];

  const heading = pose.theta + u.rot1;
  const v: Mat = [
    [-u.trans * Math.sin(heading), Math.cos(heading), 0],
    [u.trans * Math.cos(heading), Math.sin(heading), 0],
    [1, 0, 1],
  ];

  const out: Mat = [
    [1e-6, 0, 0],
    [0, 1e-6, 0],
    [0, 0, 1e-8],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += v[i][k] * m[k] * v[j][k];
      out[i][j] += s;
    }
  }
  return out;
}

export class FastSlam {
  particles: FsParticle[];
  opts: FastSlamOptions;
  weights: number[];

  constructor(particles: FsParticle[], opts: FastSlamOptions) {
    this.particles = particles;
    this.opts = opts;
    this.weights = particles.map(() => 1 / particles.length);
  }

  static atPose(m: number, start: Pose2, opts: FastSlamOptions): FastSlam {
    return new FastSlam(
      Array.from({ length: m }, (_, i) => ({
        pose: { ...start },
        path: [{ ...start }],
        landmarks: [],
        logWeight: 0,
        ancestor: i,
      })),
      opts,
    );
  }

  get size(): number {
    return this.particles.length;
  }

  private get q(): Mat {
    const { sigmaR, sigmaPhi } = this.opts;
    return [
      [sigmaR * sigmaR, 0],
      [0, sigmaPhi * sigmaPhi],
    ];
  }

  /** `FastSLAM_1_0` / `FastSLAM_2_0` / `FastSLAM_unknown_correspondence`. */
  step(u: OdomDelta, features: (RangeBearingFeature & { id: number })[], rng: Rng): FsReport {
    const q = this.q;
    let newLandmarks = 0;
    let associationErrors = 0;

    for (const p of this.particles) {
      const r = odometryPoseCovariance(p.pose, u, this.opts.alphas);

      // --- proposal -------------------------------------------------------
      // 2.0 folds one already-mapped feature into the proposal. An unmapped
      // one carries no pose information yet, so there is nothing to fold and
      // the filter falls back to 1.0's motion prior for this step.
      let pose: Pose2;
      let anchorIndex = -1;
      if (this.opts.proposal === 'measurement-aware' && features.length > 0) {
        anchorIndex = this.associate(p, applyOdom(p.pose, u), features[0], q).index;
      }
      if (anchorIndex >= 0) {
        const fit = this.proposeMeasurementAware(p, u, r, features[0], anchorIndex, q, rng);
        pose = fit.pose;
        p.logWeight += fit.logWeight;
      } else {
        pose = sampleMotionModelOdometry(u, p.pose, this.opts.alphas, rng);
      }

      // --- weight and map -------------------------------------------------
      for (let f = 0; f < features.length; f++) {
        const z = features[f];
        const assoc = this.associate(p, pose, z, q);

        if (assoc.index < 0) {
          p.landmarks.push(initLandmark(pose, z, q, p.landmarks.length, z.id));
          newLandmarks += 1;
          // A brand-new landmark explains its own reading perfectly, so it
          // contributes the default likelihood rather than a real one.
          p.logWeight += Math.log(this.opts.newLandmarkLikelihood);
          continue;
        }

        const lm = p.landmarks[assoc.index];
        if (lm.trueId !== z.id) associationErrors += 1;

        // FastSLAM 1.0's weight is exactly this marginal likelihood; 2.0 has
        // already paid for the anchor feature inside the proposal.
        const alreadyPaid = f === 0 && anchorIndex >= 0;
        const upd = updateLandmark(lm, pose, z, q);
        if (!alreadyPaid) p.logWeight += upd.logLikelihood;
      }

      p.pose = pose;
      p.path.push({ ...pose });
    }

    return this.select(rng, newLandmarks, associationErrors);
  }

  /**
   * Per-particle maximum-likelihood data association with a χ² gate.
   *
   * Returns −1 for "this is a new landmark". Because every particle runs this
   * on its *own* map and its *own* pose, the correspondence variable is
   * effectively sampled: different universes commit to different matches, and
   * resampling later decides which commitment was right.
   */
  associate(
    p: FsParticle,
    pose: Pose2,
    z: RangeBearingFeature & { id: number },
    q: Mat,
  ): { index: number; logLikelihood: number } {
    if (this.opts.knownCorrespondence) {
      const idx = p.landmarks.findIndex((lm) => lm.trueId === z.id);
      if (idx < 0) return { index: -1, logLikelihood: Math.log(this.opts.newLandmarkLikelihood) };
      return { index: idx, logLikelihood: landmarkInnovation(p.landmarks[idx], pose, z, q).logLikelihood };
    }

    let best = -1;
    let bestLog = Math.log(this.opts.newLandmarkLikelihood);
    for (let i = 0; i < p.landmarks.length; i++) {
      const lm = p.landmarks[i];
      const upd = landmarkInnovation(lm, pose, z, q);
      const l = cholesky(upd.s);
      const y0 = upd.nu[0] / l[0][0];
      const y1 = (upd.nu[1] - l[1][0] * y0) / l[1][1];
      if (y0 * y0 + y1 * y1 > this.opts.gate) continue; // outside the 99% gate
      if (upd.logLikelihood > bestLog) {
        bestLog = upd.logLikelihood;
        best = i;
      }
    }
    return { index: best, logLikelihood: bestLog };
  }

  /**
   * FastSLAM 2.0's proposal (Derivation 4).
   *
   *   Σ_x = [H_xᵀ Q_j⁻¹ H_x + R⁻¹]⁻¹
   *   μ_x = Σ_x H_xᵀ Q_j⁻¹ (z − ẑ) + x̂
   *
   * with Q_j = Q + H_m Σ_j H_mᵀ. The weight is the predictive likelihood
   * N(z; ẑ, H_x R H_xᵀ + H_m Σ_j H_mᵀ + Q) — note it is evaluated at the motion
   * mean, not at the sample, which is precisely why its variance is lower.
   */
  private proposeMeasurementAware(
    p: FsParticle,
    u: OdomDelta,
    r: Mat,
    z: RangeBearingFeature,
    index: number,
    q: Mat,
    rng: Rng,
  ): { pose: Pose2; logWeight: number } {
    const lm = p.landmarks[index];
    const xHat = applyOdom(p.pose, u);
    const pred = predictObservation(xHat, lm.mu);
    const hm = landmarkJacobian(xHat, lm.mu);
    const hx = poseJacobian(xHat, lm.mu);

    const qj = addMat(matMul(matMul(hm, lm.sigma), transpose(hm)), q);
    const qjInv = inv(qj);
    const nu: number[] = [z.r - pred.r, angleDiff(z.phi, pred.phi)];

    // Information form: the proposal is the product of two Gaussians, so add
    // precisions and let the measurement pull the mean toward itself.
    const info = addMat(matMul(matMul(transpose(hx), qjInv), hx), inv(r));
    const sigmaX = inv(info);
    const gain = matMul(matMul(sigmaX, transpose(hx)), qjInv);
    const shift = [
      gain[0][0] * nu[0] + gain[0][1] * nu[1],
      gain[1][0] * nu[0] + gain[1][1] * nu[1],
      gain[2][0] * nu[0] + gain[2][1] * nu[1],
    ];
    const mean: Pose2 = {
      x: xHat.x + shift[0],
      y: xHat.y + shift[1],
      theta: normalizeAngle(xHat.theta + shift[2]),
    };

    const l = cholesky(sigmaX);
    const g = [rng.normal(), rng.normal(), rng.normal()];
    const pose: Pose2 = {
      x: mean.x + l[0][0] * g[0],
      y: mean.y + l[1][0] * g[0] + l[1][1] * g[1],
      theta: normalizeAngle(mean.theta + l[2][0] * g[0] + l[2][1] * g[1] + l[2][2] * g[2]),
    };

    const predictive = addMat(matMul(matMul(hx, r), transpose(hx)), qj);
    return { pose, logWeight: logMvnPdf(nu, [0, 0], predictive) };
  }

  private select(rng: Rng, newLandmarks: number, associationErrors: number): FsReport {
    const m = this.particles.length;
    const weights = normalizeLogWeights(this.particles.map((p) => p.logWeight));
    const neff = effectiveSampleSize(weights);

    let bestIndex = 0;
    for (let i = 1; i < m; i++) if (weights[i] > weights[bestIndex]) bestIndex = i;

    const shouldResample = this.opts.selectiveResampling
      ? neff < this.opts.neffRatio * m
      : true;

    let resampled = false;
    if (shouldResample && m > 1) {
      const idx = resampleIndices(weights, rng);
      this.particles = idx.map((src) => cloneParticle(this.particles[src]));
      this.weights = this.particles.map(() => 1 / m);
      resampled = true;
    } else {
      this.weights = weights;
    }

    return {
      neff,
      resampled,
      distinctAncestors: new Set(this.particles.map((p) => p.ancestor)).size,
      weights: this.weights,
      bestIndex: resampled ? 0 : bestIndex,
      newLandmarks,
      associationErrors,
    };
  }

  best(): FsParticle {
    let b = 0;
    for (let i = 1; i < this.particles.length; i++) {
      if (this.weights[i] > this.weights[b]) b = i;
    }
    return this.particles[b];
  }
}

/**
 * Deep copy of a particle at resampling time.
 *
 * This is the O(N) clone the classic FastSLAM tree exists to avoid; with a few
 * dozen landmarks it is cheaper than the pointer chasing, which is exactly the
 * constant-factor crossover Exercise 6 asks the reader to measure.
 */
function cloneParticle(p: FsParticle): FsParticle {
  return {
    pose: { ...p.pose },
    path: p.path.slice(),
    landmarks: p.landmarks.map((lm) => ({
      slot: lm.slot,
      mu: [lm.mu[0], lm.mu[1]] as [number, number],
      sigma: [lm.sigma[0].slice(), lm.sigma[1].slice()],
      hits: lm.hits,
      trueId: lm.trueId,
    })),
    logWeight: 0,
    ancestor: p.ancestor,
  };
}
