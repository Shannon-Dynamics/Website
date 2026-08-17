/**
 * Learning the inverse sensor model from the forward one — Thrun et al., §9.3.
 *
 * The hand-crafted model of Table 9.2 is an honest *design*, not a derivation:
 * α, β, l_occ and l_free are numbers somebody chose. §9.3 removes the choice.
 * Sample maps from the map prior, sample poses inside them, sample measurements
 * from the **forward** model p(z | x, m), read off the true occupancy of a cell,
 * and fit a classifier to the triplets. The minimiser of the cross-entropy loss
 * over all functions is exactly p(m_i | z, x) — so the fitted model *is* the
 * inverse model, automatically consistent with the sensor physics and with the
 * prior over maps.
 *
 * The classifier here is a logistic regression on a hand-built feature map. It
 * is the smallest thing that can express the three regions of Table 9.2 and
 * still disagree with them, and it has one property worth the whole file: the
 * score w·φ **is** the log odds, so `learnedInverseLogOdds` needs no sigmoid
 * and no inversion. Chapter 25 replaces it with a real network in `candle`.
 *
 * Deterministic: every sample comes from a seeded `Rng`.
 */

import { Rng } from '../prob/rng';
import { DEFAULT_BEAM_PARAMS, type BeamParams } from '../models/sensor';

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

export interface LearnOptions {
  /** Length scale for the range residual, metres. Not a model parameter — a unit. */
  alpha: number;
  /** The sensor's *physical* opening angle, radians. The model never sees it. */
  beta: number;
  maxRange: number;
  /** Thickness of a wall in the sampled maps, metres. */
  wallThickness: number;
  /** Side of the grid cell the label refers to: a cell is occupied if any
   *  surface falls inside it, which is what an occupancy grid actually means. */
  cellSize: number;
  /** Sub-rays used to cast the physical cone when sampling measurements. */
  subRays: number;
  /** Training triplets to draw. */
  samples: number;
  epochs: number;
  lr: number;
  l2: number;
  /** The forward model sampled from — Chapter 10's four-way mixture. */
  beam: BeamParams;
}

export const DEFAULT_LEARN_OPTIONS: LearnOptions = {
  alpha: 0.2,
  beta: (20 * Math.PI) / 180,
  maxRange: 5,
  wallThickness: 0.1,
  cellSize: 0.12,
  subRays: 9,
  samples: 24000,
  epochs: 3,
  lr: 0.06,
  l2: 2e-5,
  beam: {
    ...DEFAULT_BEAM_PARAMS,
    maxRange: 5,
    sigmaHit: 0.05,
    zHit: 0.87,
    zShort: 0.04,
    zMax: 0.05,
    zRand: 0.04,
  },
};

/**
 * Human-readable names, in the same order as {@link inverseFeatures}.
 *
 * Every feature is **bounded**: indicators and Gaussian bumps, never a raw
 * signed range residual. A linear term in d would let the fitted model drift
 * without limit far from the reading, which is exactly where the answer has to
 * be ℓ₀ and nothing else.
 */
export const INVERSE_FEATURE_NAMES = [
  'bias',
  'near = 1{d ≤ −½}',
  'far = 1{d ≥ +½}',
  'g₀ = exp(−d²/2)',
  'g₀ · axis',
  'g₋₂ = exp(−(d+2)²/2)',
  'g₊₂ = exp(−(d−2)²/2)',
  'g₊₄ = exp(−(d−4)²/2)',
  'axis = exp(−a²/2)',
  'wide = exp(−a²/8)',
  'near · axis',
  '1{z = z_max}',
  '1{z = z_max} · near',
  'g₀ · r/z_max',
  'near · r/z_max',
] as const;

/**
 * φ(r, ψ, z): everything the model is allowed to know about one cell.
 *
 * Only *relative* quantities appear — the cell's range and bearing in the beam
 * frame, and the reading — because the sensor's behaviour is invariant to where
 * in the world it happens to be standing (Thrun et al., eq. 9.14). The
 * cross-range offset `a = |r sin ψ|` rather than the bare bearing is what lets
 * the model discover that a far-away cone is a wide arc, so a single reading
 * spreads its evidence over more cells and convinces you less about each.
 */
export function inverseFeatures(r: number, psi: number, z: number, o: LearnOptions): number[] {
  const d = (r - z) / o.alpha;
  // Perpendicular distance from the cell to the beam *half*-axis. Using
  // |r sin ψ| alone would make a cell directly behind the sensor look as
  // on-axis as one directly in front of it.
  const along = r * Math.cos(psi);
  const a = (along >= 0 ? Math.abs(r * Math.sin(psi)) : r) / o.alpha;
  const bump = (c: number) => Math.exp(-0.5 * (d - c) * (d - c));
  const g0 = bump(0);
  const axis = Math.exp(-0.5 * a * a);
  const wide = Math.exp(-0.125 * a * a);
  const near = d <= -0.5 ? 1 : 0;
  const far = d >= 0.5 ? 1 : 0;
  const maxed = z >= o.maxRange - 1e-6 ? 1 : 0;
  const rn = r / o.maxRange;
  return [
    1,
    near,
    far,
    g0,
    g0 * axis,
    bump(-2),
    bump(2),
    bump(4),
    axis,
    wide,
    near * axis,
    maxed,
    maxed * near,
    g0 * rn,
    near * rn,
  ];
}

export const N_INVERSE_FEATURES = INVERSE_FEATURE_NAMES.length;

// ---------------------------------------------------------------------------
// Step 1–2 of §9.3.2: sample a map, then sample a measurement from it
// ---------------------------------------------------------------------------

interface Disc {
  x: number;
  y: number;
  r: number;
}

/**
 * A map drawn from the prior, in the beam's own frame: one long straight wall
 * crossing the axis at `d0`, tilted by `tau`, plus occasional freestanding
 * clutter. Straightness is a genuine part of the prior over indoor maps, and
 * the learned model will exploit it — that is the point, not a bug.
 */
interface Scene {
  d0: number;
  tau: number;
  discs: Disc[];
}

function sampleScene(rng: Rng, o: LearnOptions): Scene {
  const d0 = rng.uniform(0.7, 0.9 * o.maxRange);
  // Triangular, so near-perpendicular incidence is the common case and grazing
  // walls are the tail — which is what a robot driving down a corridor sees.
  const tau = 0.2 * (rng.uniform(-1, 1) + rng.uniform(-1, 1));
  const discs: Disc[] = [];
  if (rng.next() < 0.45) {
    const range = rng.uniform(0.35, Math.max(0.4, d0 - 0.25));
    const bearing = rng.uniform(-o.beta / 2, o.beta / 2);
    discs.push({
      x: range * Math.cos(bearing),
      y: range * Math.sin(bearing),
      r: rng.uniform(0.07, 0.22),
    });
  }
  return { d0, tau, discs };
}

/** Distance along a unit ray at bearing ψ to the first surface, or `maxRange`. */
function castScene(scene: Scene, psi: number, o: LearnOptions): number {
  const ux = Math.cos(psi);
  const uy = Math.sin(psi);
  let best = o.maxRange;

  // Wall: plane through (d0, 0) with unit normal n = (cos τ, sin τ). The sensor
  // sits at the origin, on the negative side, so s(O) = −d0 cos τ.
  const nx = Math.cos(scene.tau);
  const ny = Math.sin(scene.tau);
  const denom = ux * nx + uy * ny;
  if (denom > 1e-9) {
    const t = (scene.d0 * nx) / denom;
    if (t > 0 && t < best) best = t;
  }

  for (const disc of scene.discs) {
    // |t·u − c|² = r² ⇒ t² − 2(u·c)t + |c|² − r² = 0.
    const b = ux * disc.x + uy * disc.y;
    const c = disc.x * disc.x + disc.y * disc.y - disc.r * disc.r;
    const disc2 = b * b - c;
    if (disc2 < 0) continue;
    const t = b - Math.sqrt(disc2);
    if (t > 0 && t < best) best = t;
  }
  return best;
}

/** Ground-truth occupancy of the point at (r, ψ). `null` means "unobservable". */
function occupancyAt(scene: Scene, r: number, psi: number, rng: Rng, o: LearnOptions): 0 | 1 {
  const px = r * Math.cos(psi);
  const py = r * Math.sin(psi);
  // "Occupied" means a surface falls inside the *cell*, not that the sample
  // point lands exactly on it — the map is a grid, and this is its resolution.
  const skin = (o.wallThickness + o.cellSize) / 2;
  for (const disc of scene.discs) {
    if (Math.hypot(px - disc.x, py - disc.y) < disc.r + o.cellSize / 2) return 1;
  }
  const s = (px - scene.d0) * Math.cos(scene.tau) + py * Math.sin(scene.tau);
  if (Math.abs(s) < skin) return 1;
  // Behind the wall the map prior is all we have: a fair coin. This is what
  // teaches the model to return exactly l₀ beyond the reading — nobody had to
  // write that rule down.
  if (s > 0) return rng.next() < 0.5 ? 1 : 0;
  return 0;
}

/** Draw z ~ p(z | x, m) from the Chapter 10 mixture, given the true range. */
function sampleReading(rng: Rng, zStar: number, beam: BeamParams): number {
  const u = rng.next();
  if (u < beam.zMax) return beam.maxRange;
  if (u < beam.zMax + beam.zRand) return rng.uniform(0, beam.maxRange);
  if (u < beam.zMax + beam.zRand + beam.zShort) return rng.uniform(0, zStar);
  return Math.max(0, Math.min(beam.maxRange, zStar + rng.normal(0, beam.sigmaHit)));
}

export interface Triplet {
  r: number;
  psi: number;
  z: number;
  /** occ(m_i) — the label. */
  y: 0 | 1;
}

/**
 * Steps 1–4 of Thrun et al. §9.3.2, verbatim: sample a map, sample a pose (here
 * the origin, by the invariance above), sample a measurement from the forward
 * model, extract the true occupancy of a target cell.
 */
export function sampleInverseTrainingSet(seed: number, o: LearnOptions): Triplet[] {
  const rng = new Rng(seed);
  const out: Triplet[] = [];
  const queriesPerScene = 8;

  while (out.length < o.samples) {
    const scene = sampleScene(rng, o);
    // The cone: the reading is the nearest surface any sub-ray strikes.
    let zStar = o.maxRange;
    for (let s = 0; s < o.subRays; s++) {
      const frac = o.subRays === 1 ? 0 : s / (o.subRays - 1) - 0.5;
      zStar = Math.min(zStar, castScene(scene, frac * o.beta, o));
    }
    const z = sampleReading(rng, zStar, o.beam);

    for (let q = 0; q < queriesPerScene && out.length < o.samples; q++) {
      // Query cells are drawn well outside the cone too, so the model has to
      // learn where its own competence ends instead of being told.
      const r = rng.uniform(0.05, o.maxRange);
      const psi = rng.uniform(-2.2 * o.beta, 2.2 * o.beta);
      out.push({ r, psi, z, y: occupancyAt(scene, r, psi, rng, o) });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Step 5: fit
// ---------------------------------------------------------------------------

export interface LearnedInverseModel {
  w: Float64Array;
  opts: LearnOptions;
  /** Mean cross-entropy on the training set, in nats. Reported, not hidden. */
  loss: number;
  samples: number;
  /**
   * The model's own ℓ₀: what it answers about a cell the reading cannot see.
   *
   * Nobody sets this. It falls out of the prior over the maps the model was
   * trained on, and it is the number Table 9.1 must subtract — using 0 instead
   * would leak a constant drift into every cell of the perceptual field.
   */
  l0: number;
}

/**
 * Minimise J(W) = −Σ [ m log f + (1 − m) log(1 − f) ] — Thrun et al., eq. 9.20 —
 * by plain SGD. With a logistic link the gradient is (σ(w·φ) − y)·φ, which is
 * three lines and no autodiff.
 */
export function trainInverseSensorModel(
  seed = 0xc0ffee,
  options: Partial<LearnOptions> = {},
): LearnedInverseModel {
  const o: LearnOptions = { ...DEFAULT_LEARN_OPTIONS, ...options };
  const data = sampleInverseTrainingSet(seed, o);
  const w = new Float64Array(N_INVERSE_FEATURES);
  const rng = new Rng(seed ^ 0x5bf03635);

  let t = 0;
  for (let epoch = 0; epoch < o.epochs; epoch++) {
    // Shuffle indices so consecutive samples from one scene do not correlate.
    const order = data.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const idx of order) {
      const s = data[idx];
      const phi = inverseFeatures(s.r, s.psi, s.z, o);
      let score = 0;
      for (let k = 0; k < w.length; k++) score += w[k] * phi[k];
      const p = 1 / (1 + Math.exp(-score));
      const err = p - s.y;
      const lr = o.lr / (1 + t / 8000);
      for (let k = 0; k < w.length; k++) {
        w[k] -= lr * (err * phi[k] + o.l2 * w[k]);
      }
      t++;
    }
  }

  let loss = 0;
  for (const s of data) {
    const phi = inverseFeatures(s.r, s.psi, s.z, o);
    let score = 0;
    for (let k = 0; k < w.length; k++) score += w[k] * phi[k];
    const p = Math.min(1 - 1e-9, Math.max(1e-9, 1 / (1 + Math.exp(-score))));
    loss -= s.y === 1 ? Math.log(p) : Math.log(1 - p);
  }

  // The model's answer far beyond a short reading: no beam evidence reaches
  // there, so whatever it says is its inherited prior.
  const probe = inverseFeatures(0.92 * o.maxRange, 0, 0.2 * o.maxRange, o);
  let l0 = 0;
  for (let k = 0; k < w.length; k++) l0 += w[k] * probe[k];

  return { w, opts: o, loss: loss / data.length, samples: data.length, l0 };
}

/**
 * The learned inverse model, in log odds — ready to drop straight into line 4
 * of Table 9.1.
 *
 * No sigmoid appears. A logistic regressor's score *is* `log p/(1−p)`, so the
 * quantity occupancy grid mapping wants is the raw linear response.
 */
export function learnedInverseLogOdds(
  model: LearnedInverseModel,
  r: number,
  psi: number,
  z: number,
): number {
  const phi = inverseFeatures(r, psi, z, model.opts);
  let score = 0;
  for (let k = 0; k < model.w.length; k++) score += model.w[k] * phi[k];
  return score;
}

/** p(m_i | z, x) under the learned model. */
export function learnedInverseProbability(
  model: LearnedInverseModel,
  r: number,
  psi: number,
  z: number,
): number {
  return 1 / (1 + Math.exp(-learnedInverseLogOdds(model, r, psi, z)));
}

/**
 * The increment Table 9.1 adds: `inverse_sensor_model(...) − ℓ₀`, with the
 * model's *own* ℓ₀. Zero for any cell the measurement says nothing about.
 */
export function learnedInverseEvidence(
  model: LearnedInverseModel,
  r: number,
  psi: number,
  z: number,
): number {
  return learnedInverseLogOdds(model, r, psi, z) - model.l0;
}
