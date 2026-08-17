/**
 * EKF SLAM — Thrun et al., **Tables 10.1 and 10.2**.
 *
 * One filter over the joint state y_t = (x_t, m_1, …, m_N). The map is not a
 * separate object: it is the tail of the state vector, and the *off-diagonal*
 * blocks of Σ are what make the thing work. Observing one landmark moves every
 * landmark correlated with the pose, which is why a loop closure heals a map it
 * never re-measured.
 *
 * Two implementation facts the chapter leans on:
 *
 *  1. **The state grows at runtime.** Chapter 6's `Kf` is sized by const
 *     generics; that is impossible here, because N_t is discovered, not
 *     declared. Hence plain resizable arrays (`DVector`/`DMatrix` on the Rust
 *     side).
 *  2. **The prediction is O(N); the correction is Θ(N²).** Motion touches only
 *     Σ_xx and the pose–map strip, so it is linear. The measurement update is
 *     a rank-2 downdate Σ ← Σ − K (Σ Hᵀ)ᵀ that touches *every* entry, and no
 *     amount of bookkeeping in moments form avoids it. `entriesTouched` counts
 *     exactly those writes so a widget can plot the cost curve instead of
 *     asserting it.
 *
 * Everything here is deterministic: no randomness lives in the filter.
 */

import { angleDiff, normalizeAngle, type Pose2 } from '../geom/se2';
import { symmetrize, zerosMat, type Mat, type Vec } from '../prob/linalg';
import { nees as gaussianNees } from '../filters/consistency';
import type { VelocityCmd } from '../models/motion';
import type { RangeBearingFeature } from '../models/sensor';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * The pose process noise, expressed the way a wheeled robot actually earns it:
 * per metre driven and per radian turned, in the **body** frame. The filter
 * rotates it into the world frame before adding it, which is the only place
 * heading enters R_t.
 */
export interface MotionNoise {
  /** Along-track σ, metres per metre driven. */
  alongTrack: number;
  /** Cross-track σ, metres per metre driven. */
  crossTrack: number;
  /** Heading σ, radians per metre driven (the drift a straight run accrues). */
  headingPerMetre: number;
  /** Heading σ, radians per radian commanded (the error a turn accrues). */
  headingPerRadian: number;
}

export interface SlamConfig {
  motion: MotionNoise;
  /** Thrun's Q_t: range and bearing standard deviations. */
  sigmaR: number;
  sigmaPhi: number;
  /** γ_gate — associate only if the Mahalanobis distance is below this. */
  gateChi2: number;
  /** χ²_new (Thrun's α) — above this, the observation is a new landmark. */
  newChi2: number;
  /** Keep unmatched observations off the state vector until they repeat. */
  useProvisional: boolean;
  /** Sightings a candidate needs before it is promoted into the state. */
  promoteAfter: number;
  /** Log-odds added when a landmark is re-observed. */
  existenceUp: number;
  /** Log-odds subtracted when a landmark should have been seen and was not. */
  existenceDown: number;
  /** Retire a landmark once its existence log odds sinks below this. */
  retireBelow: number;
  /**
   * The chapter's ablation: zero every cross-block of Σ after each update, so
   * the map keeps its marginals but forgets that it is correlated. The filter
   * still runs; it just stops being SLAM.
   */
  ablateCorrelations: boolean;
}

export const DEFAULT_SLAM_CONFIG: SlamConfig = {
  // Deliberately generous — about 1° of heading drift per 14 cm step. Real
  // wheel odometry is better than this; the consistency literature inflates
  // exactly this term to compress an effect that otherwise needs kilometres
  // into a few hundred steps a reader will actually watch.
  motion: {
    alongTrack: 0.05,
    crossTrack: 0.02,
    headingPerMetre: 0.14,
    headingPerRadian: 0.28,
  },
  sigmaR: 0.06,
  sigmaPhi: 0.03,
  // χ²(2 dof) quantiles: 9.21 is 99%, 13.82 is 99.9%. Between them an
  // observation is neither trusted nor believed to be new — it is discarded.
  gateChi2: 9.21,
  newChi2: 13.82,
  useProvisional: true,
  promoteAfter: 3,
  existenceUp: 0.5,
  existenceDown: 0.35,
  retireBelow: -1.5,
  ablateCorrelations: false,
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

export type Association =
  | { kind: 'matched'; landmark: number; nis: number }
  | { kind: 'born'; landmark: number; nis: number }
  | { kind: 'candidate'; candidate: number; nis: number }
  | { kind: 'rejected'; nis: number };

export interface UpdateInfo {
  innovation: [number, number];
  /** Innovation covariance S = h Σ̄ hᵀ + Q. */
  S: Mat;
  /** Normalized innovation squared — the gate statistic π. */
  nis: number;
  /** Σ entries written by this update: the measured Θ(N²). */
  entries: number;
}

/** A sighting that has not earned a place in the state vector yet. */
export interface Candidate {
  x: number;
  y: number;
  cov: Mat;
  sightings: number;
  lastSeen: number;
}

/** Where landmark `j` lives in the state vector. */
export const landmarkIndex = (j: number): number => 3 + 2 * j;

/** The 5 state dimensions one landmark observation touches: pose + that landmark. */
export const observationSupport = (j: number): number[] => {
  const b = landmarkIndex(j);
  return [0, 1, 2, b, b + 1];
};

/** Which block a state index belongs to: −1 for the pose, else the landmark. */
export const blockOf = (i: number): number => (i < 3 ? -1 : Math.floor((i - 3) / 2));

/** Closed-form inverse of a symmetric 2×2, with a floor on the determinant. */
function inv2(m: Mat): Mat {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  const d = Math.abs(det) < 1e-14 ? (det < 0 ? -1e-14 : 1e-14) : det;
  return [
    [m[1][1] / d, -m[0][1] / d],
    [-m[1][0] / d, m[0][0] / d],
  ];
}

// ---------------------------------------------------------------------------
// The filter
// ---------------------------------------------------------------------------

export class EkfSlam {
  /** [x y θ | m₀ₓ m₀ᵧ | m₁ₓ m₁ᵧ | …] */
  mu: Vec;
  sigma: Mat;
  cfg: SlamConfig;
  /** Caller's label for each slot — the true landmark id in known-correspondence runs. */
  labels: number[];
  /** Per-landmark existence evidence: Chapter 13's binary Bayes filter, retargeted. */
  existence: number[];
  candidates: Candidate[];
  /** Σ entries written since construction. The Growth Meter plots this. */
  entriesTouched = 0;
  /** Measurement updates applied since construction. */
  updates = 0;

  constructor(pose: Pose2, poseCov: Mat, cfg: SlamConfig = DEFAULT_SLAM_CONFIG) {
    this.mu = [pose.x, pose.y, pose.theta];
    this.sigma = poseCov.map((r) => r.slice());
    this.cfg = cfg;
    this.labels = [];
    this.existence = [];
    this.candidates = [];
  }

  get dim(): number {
    return this.mu.length;
  }

  /** N_t — the number of landmarks currently in the state. */
  get count(): number {
    return (this.mu.length - 3) / 2;
  }

  pose(): Pose2 {
    return { x: this.mu[0], y: this.mu[1], theta: this.mu[2] };
  }

  poseCov(): Mat {
    return [0, 1, 2].map((i) => [this.sigma[i][0], this.sigma[i][1], this.sigma[i][2]]);
  }

  landmarkMean(j: number): [number, number] {
    const b = landmarkIndex(j);
    return [this.mu[b], this.mu[b + 1]];
  }

  landmarkCov(j: number): Mat {
    const b = landmarkIndex(j);
    return [
      [this.sigma[b][b], this.sigma[b][b + 1]],
      [this.sigma[b + 1][b], this.sigma[b + 1][b + 1]],
    ];
  }

  /** Slot holding the landmark the caller labelled `id`, or −1. */
  slotOf(id: number): number {
    return this.labels.indexOf(id);
  }

  // -------------------------------------------------------------------------
  // Prediction — Table 10.1, lines 2–5
  // -------------------------------------------------------------------------

  /**
   * Motion moves the pose and leaves every landmark exactly where it was, so
   * the projection form μ̄ = μ + Fₓᵀ δ, G = I + Fₓᵀ g Fₓ collapses to blockwise
   * algebra:
   *
   *   Σ_xx ← Gₓ Σ_xx Gₓᵀ + R      (3×3)
   *   Σ_xm ← Gₓ Σ_xm              (the strip: rotated, never erased)
   *   Σ_mm ← Σ_mm                 (untouched)
   *
   * O(N) work, and the middle line is the whole reason the map remembers the
   * robot. Erase that strip and SLAM becomes N independent little filters.
   */
  predict(u: VelocityCmd): void {
    const { v, omega, dt } = u;
    const th = this.mu[2];
    const nt = th + omega * dt;

    let dx: number;
    let dy: number;
    let dth02: number; // ∂x′/∂θ
    let dth12: number; // ∂y′/∂θ

    if (Math.abs(omega) < 1e-6) {
      dx = v * Math.cos(th) * dt;
      dy = v * Math.sin(th) * dt;
      dth02 = -v * Math.sin(th) * dt;
      dth12 = v * Math.cos(th) * dt;
    } else {
      const r = v / omega;
      dx = -r * Math.sin(th) + r * Math.sin(nt);
      dy = r * Math.cos(th) - r * Math.cos(nt);
      // NOTE the signs: this is ∂g/∂θ of the arc above. Thrun's Table 10.1
      // prints the negative of it; the derivative is what a numeric Jacobian
      // agrees with, and what a diverging filter cares about.
      dth02 = r * (Math.cos(nt) - Math.cos(th));
      dth12 = r * (Math.sin(nt) - Math.sin(th));
    }

    this.mu[0] += dx;
    this.mu[1] += dy;
    this.mu[2] = normalizeAngle(nt);

    const gx: Mat = [
      [1, 0, dth02],
      [0, 1, dth12],
      [0, 0, 1],
    ];
    const R = this.processNoise(u, th);

    const n = this.dim;
    const sigma = this.sigma;

    // M = Gₓ · Σ[0:3, :], computed against the old Σ.
    const M: Mat = zerosMat(3, n);
    for (let i = 0; i < 3; i++) {
      for (let c = 0; c < n; c++) {
        M[i][c] = gx[i][0] * sigma[0][c] + gx[i][1] * sigma[1][c] + gx[i][2] * sigma[2][c];
      }
    }

    // Σ_xx = (Gₓ Σ_xx) Gₓᵀ + R, using the first three columns of M.
    const xx: Mat = zerosMat(3, 3);
    for (let i = 0; i < 3; i++) {
      for (let k = 0; k < 3; k++) {
        xx[i][k] =
          M[i][0] * gx[k][0] + M[i][1] * gx[k][1] + M[i][2] * gx[k][2] + R[i][k];
      }
    }

    for (let c = 3; c < n; c++) {
      for (let i = 0; i < 3; i++) {
        sigma[i][c] = M[i][c];
        sigma[c][i] = M[i][c];
      }
    }
    for (let i = 0; i < 3; i++) {
      for (let k = 0; k < 3; k++) sigma[i][k] = xx[i][k];
    }
  }

  /** R_t: body-frame noise for this command, rotated into the world frame. */
  processNoise(u: VelocityCmd, theta: number): Mat {
    const { alongTrack, crossTrack, headingPerMetre, headingPerRadian } = this.cfg.motion;
    const d = Math.abs(u.v) * u.dt;
    const turn = Math.abs(u.omega) * u.dt;
    const sa = alongTrack * d;
    const sc = crossTrack * d;
    const st = headingPerMetre * d + headingPerRadian * turn;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const a2 = sa * sa;
    const c2 = sc * sc;
    // Rot(θ) diag(σ_a², σ_c²) Rot(θ)ᵀ — a banana-shaped step, not a circle.
    return [
      [c * c * a2 + s * s * c2, c * s * (a2 - c2), 0],
      [c * s * (a2 - c2), s * s * a2 + c * c * c2, 0],
      [0, 0, Math.max(st * st, 1e-12)],
    ];
  }

  // -------------------------------------------------------------------------
  // Measurement geometry
  // -------------------------------------------------------------------------

  /**
   * ẑ and the low-dimensional Jacobian hᵢ of Thrun's eq. (10.21), returned
   * against the 5 state dimensions the observation actually touches. The full
   * H = h F_{x,j} is never formed: forming it is how an implementation
   * accidentally turns an O(N) step into an O(N²) one.
   */
  expected(j: number): { z: [number, number]; h: Mat; q: number } {
    const b = landmarkIndex(j);
    const dx = this.mu[b] - this.mu[0];
    const dy = this.mu[b + 1] - this.mu[1];
    const q = Math.max(dx * dx + dy * dy, 1e-9);
    const sq = Math.sqrt(q);
    return {
      z: [sq, normalizeAngle(Math.atan2(dy, dx) - this.mu[2])],
      h: [
        [-dx / sq, -dy / sq, 0, dx / sq, dy / sq],
        [dy / q, -dx / q, -1, -dy / q, dx / q],
      ],
      q,
    };
  }

  /** S_k = h_k Σ̄ h_kᵀ + Q_t, the gate's shape for landmark k. */
  innovationCov(j: number): Mat {
    const { h } = this.expected(j);
    const idx = observationSupport(j);
    const S: Mat = [
      [0, 0],
      [0, 0],
    ];
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 2; b++) {
        let s = 0;
        for (let p = 0; p < 5; p++) {
          for (let r = 0; r < 5; r++) s += h[a][p] * this.sigma[idx[p]][idx[r]] * h[b][r];
        }
        S[a][b] = s;
      }
    }
    S[0][0] += this.cfg.sigmaR * this.cfg.sigmaR;
    S[1][1] += this.cfg.sigmaPhi * this.cfg.sigmaPhi;
    return S;
  }

  /** π_k — the Mahalanobis distance from an observation to landmark k's gate. */
  mahalanobis(j: number, f: RangeBearingFeature): number {
    const { z } = this.expected(j);
    const S = this.innovationCov(j);
    const y: Vec = [f.r - z[0], angleDiff(f.phi, z[1])];
    const Si = inv2(S);
    return (
      y[0] * (Si[0][0] * y[0] + Si[0][1] * y[1]) + y[1] * (Si[1][0] * y[0] + Si[1][1] * y[1])
    );
  }

  // -------------------------------------------------------------------------
  // Correction — Table 10.1, lines 12–20
  // -------------------------------------------------------------------------

  /**
   * One landmark observation, folded in.
   *
   * K = Σ̄ Hᵀ S⁻¹ is **dense** even though H has 5 non-zero columns: the
   * pose–map strip fans the 2-vector innovation across the whole state. The
   * covariance downdate Σ ← Σ − K (Σ̄ Hᵀ)ᵀ therefore writes every entry, which
   * is where the Θ(N²) lives.
   */
  updateLandmark(j: number, f: RangeBearingFeature): UpdateInfo {
    const n = this.dim;
    const idx = observationSupport(j);
    const { z, h } = this.expected(j);
    const y: [number, number] = [f.r - z[0], angleDiff(f.phi, z[1])];

    // PHt = Σ̄ Hᵀ, an n×2 slab built from 5 columns of Σ̄ — O(N).
    const PHt: Mat = zerosMat(n, 2);
    for (let r = 0; r < n; r++) {
      let a = 0;
      let b = 0;
      for (let p = 0; p < 5; p++) {
        const s = this.sigma[r][idx[p]];
        a += s * h[0][p];
        b += s * h[1][p];
      }
      PHt[r][0] = a;
      PHt[r][1] = b;
    }

    const S: Mat = [
      [0, 0],
      [0, 0],
    ];
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 2; b++) {
        let s = 0;
        for (let p = 0; p < 5; p++) s += h[a][p] * PHt[idx[p]][b];
        S[a][b] = s;
      }
    }
    S[0][0] += this.cfg.sigmaR * this.cfg.sigmaR;
    S[1][1] += this.cfg.sigmaPhi * this.cfg.sigmaPhi;
    const Si = inv2(S);

    // K = PHt S⁻¹ — n×2, and generically non-zero in every row.
    const K: Mat = zerosMat(n, 2);
    for (let r = 0; r < n; r++) {
      K[r][0] = PHt[r][0] * Si[0][0] + PHt[r][1] * Si[1][0];
      K[r][1] = PHt[r][0] * Si[0][1] + PHt[r][1] * Si[1][1];
    }

    for (let r = 0; r < n; r++) this.mu[r] += K[r][0] * y[0] + K[r][1] * y[1];
    this.mu[2] = normalizeAngle(this.mu[2]);

    // Σ ← Σ − K (Σ̄ Hᵀ)ᵀ. Every entry. This is the quadratic term.
    for (let r = 0; r < n; r++) {
      const kr0 = K[r][0];
      const kr1 = K[r][1];
      const row = this.sigma[r];
      for (let c = 0; c < n; c++) row[c] -= kr0 * PHt[c][0] + kr1 * PHt[c][1];
    }
    this.sigma = symmetrize(this.sigma);
    this.entriesTouched += n * n;
    this.updates += 1;

    if (this.cfg.ablateCorrelations) this.ablate();

    const nis =
      y[0] * (Si[0][0] * y[0] + Si[0][1] * y[1]) + y[1] * (Si[1][0] * y[0] + Si[1][1] * y[1]);
    return { innovation: y, S, nis, entries: n * n };
  }

  // -------------------------------------------------------------------------
  // Landmark birth — Table 10.1 line 10, done properly
  // -------------------------------------------------------------------------

  /**
   * First sighting: invert the measurement function and push the uncertainty
   * through both Jacobians.
   *
   *   m = (x + r cos(φ+θ),  y + r sin(φ+θ))
   *   Σ_mm = G_x Σ_xx G_xᵀ + G_z Q G_zᵀ
   *   Σ_mx = G_x Σ_xx          … and the same strip against every other landmark
   *
   * Thrun's table instead seeds the mean and hands the covariance an infinite
   * prior, letting the first update do this work. Both land in the same place;
   * this way the covariance is right on the very first frame, which matters
   * when a widget is drawing it.
   */
  initLandmark(f: RangeBearingFeature, label = -1): number {
    const nOld = this.dim;
    const th = this.mu[2];
    const a = normalizeAngle(f.phi + th);
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const mx = this.mu[0] + f.r * ca;
    const my = this.mu[1] + f.r * sa;

    const Gx: Mat = [
      [1, 0, -f.r * sa],
      [0, 1, f.r * ca],
    ];
    const Gz: Mat = [
      [ca, -f.r * sa],
      [sa, f.r * ca],
    ];
    const qr = this.cfg.sigmaR * this.cfg.sigmaR;
    const qp = this.cfg.sigmaPhi * this.cfg.sigmaPhi;

    // Cross = G_x · Σ[0:3, :] — the new landmark's correlation with everything
    // already in the state, inherited entirely through the pose.
    const Cross: Mat = zerosMat(2, nOld);
    for (let i = 0; i < 2; i++) {
      for (let c = 0; c < nOld; c++) {
        Cross[i][c] =
          Gx[i][0] * this.sigma[0][c] + Gx[i][1] * this.sigma[1][c] + Gx[i][2] * this.sigma[2][c];
      }
    }

    const mm: Mat = zerosMat(2, 2);
    for (let i = 0; i < 2; i++) {
      for (let k = 0; k < 2; k++) {
        const pose =
          Cross[i][0] * Gx[k][0] + Cross[i][1] * Gx[k][1] + Cross[i][2] * Gx[k][2];
        const meas = Gz[i][0] * qr * Gz[k][0] + Gz[i][1] * qp * Gz[k][1];
        mm[i][k] = pose + meas;
      }
    }

    this.mu.push(mx, my);
    for (let r = 0; r < nOld; r++) {
      this.sigma[r].push(Cross[0][r], Cross[1][r]);
    }
    this.sigma.push([...Cross[0], mm[0][0], mm[0][1]]);
    this.sigma.push([...Cross[1], mm[1][0], mm[1][1]]);

    this.labels.push(label);
    this.existence.push(this.cfg.existenceUp);
    if (this.cfg.ablateCorrelations) this.ablate();
    return this.count - 1;
  }

  /** Delete a landmark's rows, columns, and bookkeeping — state surgery. */
  retire(j: number): void {
    const b = landmarkIndex(j);
    this.mu.splice(b, 2);
    this.sigma.splice(b, 2);
    for (const row of this.sigma) row.splice(b, 2);
    this.labels.splice(j, 1);
    this.existence.splice(j, 1);
  }

  // -------------------------------------------------------------------------
  // Map management — §10.3.3
  // -------------------------------------------------------------------------

  /**
   * Existence evidence, as log odds. Seeing a landmark is positive evidence;
   * *not* seeing one that should have been in view is negative evidence, which
   * is the only way a filter ever un-believes a phantom.
   */
  updateExistence(expected: number[], observed: number[]): void {
    const seen = new Set(observed);
    for (const j of expected) {
      if (j < 0 || j >= this.count) continue;
      this.existence[j] += seen.has(j) ? this.cfg.existenceUp : -this.cfg.existenceDown;
      this.existence[j] = Math.max(-6, Math.min(6, this.existence[j]));
    }
  }

  /** Retire every landmark whose evidence has sunk below the floor. Returns how many. */
  pruneMap(): number {
    let removed = 0;
    for (let j = this.count - 1; j >= 0; j--) {
      if (this.existence[j] < this.cfg.retireBelow) {
        this.retire(j);
        removed += 1;
      }
    }
    return removed;
  }

  /** Fuse an unassociated sighting into the provisional list, promoting when ripe. */
  private handleCandidate(f: RangeBearingFeature, t: number): Association {
    const th = this.mu[2];
    const a = normalizeAngle(f.phi + th);
    const wx = this.mu[0] + f.r * Math.cos(a);
    const wy = this.mu[1] + f.r * Math.sin(a);
    // A candidate is matched in world coordinates with a generous radius: it is
    // not in the state, so it has no Σ to gate against.
    const radius = Math.max(3 * this.cfg.sigmaR, 0.35);
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < this.candidates.length; i++) {
      const c = this.candidates[i];
      const d = Math.hypot(c.x - wx, c.y - wy);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0 && bestD < radius) {
      const c = this.candidates[best];
      c.x = 0.5 * (c.x + wx);
      c.y = 0.5 * (c.y + wy);
      c.sightings += 1;
      c.lastSeen = t;
      if (c.sightings >= this.cfg.promoteAfter) {
        this.candidates.splice(best, 1);
        const j = this.initLandmark(f, f.s ?? -1);
        return { kind: 'born', landmark: j, nis: 0 };
      }
      return { kind: 'candidate', candidate: best, nis: 0 };
    }
    this.candidates.push({
      x: wx,
      y: wy,
      cov: [
        [this.cfg.sigmaR * this.cfg.sigmaR, 0],
        [0, this.cfg.sigmaR * this.cfg.sigmaR],
      ],
      sightings: 1,
      lastSeen: t,
    });
    return { kind: 'candidate', candidate: this.candidates.length - 1, nis: 0 };
  }

  /** Drop candidates nobody has seen for a while, so the list cannot grow forever. */
  expireCandidates(t: number, patience = 40): void {
    this.candidates = this.candidates.filter((c) => t - c.lastSeen <= patience);
  }

  // -------------------------------------------------------------------------
  // The two published algorithms
  // -------------------------------------------------------------------------

  /**
   * `EKF_SLAM_known_correspondences` — Thrun et al., **Table 10.1**.
   * Every feature carries its landmark id in `s`; unseen ids are born.
   */
  correctKnown(features: RangeBearingFeature[]): Association[] {
    const out: Association[] = [];
    for (const f of features) {
      const id = f.s ?? -1;
      let j = this.slotOf(id);
      if (j < 0) {
        j = this.initLandmark(f, id);
        out.push({ kind: 'born', landmark: j, nis: 0 });
        continue;
      }
      const info = this.updateLandmark(j, f);
      out.push({ kind: 'matched', landmark: j, nis: info.nis });
    }
    return out;
  }

  /**
   * `EKF_SLAM` — Thrun et al., **Table 10.2**, with the §10.3.3 map management.
   *
   * Maximum-likelihood association: score the observation against every
   * landmark's gate, take the smallest π_k, and treat a large minimum as
   * evidence of something new. The decision is *hard* — once made it is baked
   * into Σ, and no later evidence can undo it.
   */
  correct(features: RangeBearingFeature[], t = 0): Association[] {
    const out: Association[] = [];
    for (const f of features) {
      const N = this.count;
      let best = -1;
      let bestPi = Infinity;
      for (let k = 0; k < N; k++) {
        const pi = this.mahalanobis(k, f);
        if (pi < bestPi) {
          bestPi = pi;
          best = k;
        }
      }

      if (best >= 0 && bestPi < this.cfg.gateChi2) {
        const info = this.updateLandmark(best, f);
        out.push({ kind: 'matched', landmark: best, nis: info.nis });
        continue;
      }
      if (best >= 0 && bestPi < this.cfg.newChi2) {
        // In the no-man's-land between the gate and the birth threshold: too
        // far to trust, too close to call new. Throw it away — the cheapest
        // insurance a filter has.
        out.push({ kind: 'rejected', nis: bestPi });
        continue;
      }
      if (this.cfg.useProvisional) {
        out.push(this.handleCandidate(f, t));
        continue;
      }
      const j = this.initLandmark(f, f.s ?? -1);
      out.push({ kind: 'born', landmark: j, nis: bestPi });
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Instruments
  // -------------------------------------------------------------------------

  /** Zero every cross-block: the map keeps its marginals and forgets its web. */
  ablate(): void {
    const n = this.dim;
    for (let a = 0; a < n; a++) {
      const ba = blockOf(a);
      for (let b = 0; b < n; b++) {
        if (ba !== blockOf(b)) this.sigma[a][b] = 0;
      }
    }
  }

  /** ρ_ij = Σ_ij / √(Σ_ii Σ_jj) — what the heatmap actually draws. */
  correlation(): Mat {
    const n = this.dim;
    const sd = new Array<number>(n);
    for (let i = 0; i < n; i++) sd[i] = Math.sqrt(Math.max(this.sigma[i][i], 1e-15));
    const out = zerosMat(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        out[i][j] = Math.max(-1, Math.min(1, this.sigma[i][j] / (sd[i] * sd[j])));
      }
    }
    return out;
  }

  /**
   * Pose NEES: (x − μ)ᵀ Σ_xx⁻¹ (x − μ) with the heading difference wrapped.
   * Expectation 3 for a consistent filter; systematically above it is the
   * signature of a filter that believes its own linearizations.
   */
  nees(truth: Pose2): number {
    const mu = this.pose();
    const wrapped: Vec = [truth.x, truth.y, mu.theta + angleDiff(truth.theta, mu.theta)];
    return gaussianNees(wrapped, { x: [mu.x, mu.y, mu.theta], P: this.poseCov() });
  }

  /** Average of the landmark position variances — "how well is the map known?" */
  mapUncertainty(): number {
    const N = this.count;
    if (N === 0) return 0;
    let s = 0;
    for (let j = 0; j < N; j++) {
      const b = landmarkIndex(j);
      s += 0.5 * (this.sigma[b][b] + this.sigma[b + 1][b + 1]);
    }
    return s / N;
  }

  /** Mean |ρ| between distinct landmark blocks — the web's tautness. */
  meanLandmarkCorrelation(): number {
    const N = this.count;
    if (N < 2) return 0;
    const rho = this.correlation();
    let s = 0;
    let k = 0;
    for (let a = 0; a < N; a++) {
      for (let b = a + 1; b < N; b++) {
        const ia = landmarkIndex(a);
        const ib = landmarkIndex(b);
        s += Math.abs(rho[ia][ib]) + Math.abs(rho[ia + 1][ib + 1]);
        k += 2;
      }
    }
    return k > 0 ? s / k : 0;
  }
}

/**
 * Bytes a dense f64 covariance needs at N landmarks, and the multiply–adds one
 * observation update costs. Both are quadratic; the widget prints them at
 * N = 10⁵ so the reader can feel the wall rather than read about it.
 */
export function slamCost(nLandmarks: number): { dim: number; bytes: number; flops: number } {
  const dim = 3 + 2 * nLandmarks;
  return { dim, bytes: dim * dim * 8, flops: dim * dim * 2 };
}
