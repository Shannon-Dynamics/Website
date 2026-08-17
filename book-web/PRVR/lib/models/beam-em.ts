/**
 * Learning the beam model's intrinsic parameters from data — Thrun et al.,
 * **Table 6.2** (`learn_intrinsic_parameters`) and the derivation in §6.3.3.
 *
 * The beam model of {@link ./sensor} is a four-way mixture, and a mixture with
 * unobserved component labels is exactly what expectation–maximization was
 * invented for. Given a log of ranges `z` together with the range each beam
 * *should* have returned (`zStar`, ray-cast into the map from a known pose), EM
 * alternates:
 *
 *   E-step  responsibilities e_{i,c} ∝ z_c · p_c(z_i)      — eq. (6.28)
 *   M-step  z_c = mean_i e_{i,c},                          — eq. (6.30)
 *           σ_hit² = Σ e_{i,hit}(z_i − z_i*)² / Σ e_{i,hit} — eq. (6.31)
 *           λ_short = Σ e_{i,short} / Σ e_{i,short} z_i     — eq. (6.32)
 *
 * Every M-step is a closed form, so an iteration costs one pass over the data
 * and the log-likelihood is monotone non-decreasing. That monotonicity is the
 * thing to watch in the widget: if the curve ever dips, the implementation is
 * wrong, not the data.
 *
 * The component densities are obtained by evaluating the *existing* mixture with
 * one-hot weights rather than by re-deriving them here, so `beamComponents`
 * cannot silently drift away from `beamLikelihood`.
 */

import { beamLikelihood, type BeamParams } from './sensor';

/** The four unweighted component densities p_c(z) at one reading. */
export interface BeamComponents {
  hit: number;
  short: number;
  max: number;
  rand: number;
}

/** Component keys in the fixed order used by every array in this file. */
export const BEAM_CAUSES = ['hit', 'short', 'max', 'rand'] as const;
export type BeamCause = (typeof BEAM_CAUSES)[number];

const ONE_HOT: Record<BeamCause, Pick<BeamParams, 'zHit' | 'zShort' | 'zMax' | 'zRand'>> = {
  hit: { zHit: 1, zShort: 0, zMax: 0, zRand: 0 },
  short: { zHit: 0, zShort: 1, zMax: 0, zRand: 0 },
  max: { zHit: 0, zShort: 0, zMax: 1, zRand: 0 },
  rand: { zHit: 0, zShort: 0, zMax: 0, zRand: 1 },
};

/**
 * The four densities p_hit, p_short, p_max, p_rand evaluated separately.
 *
 * `beamLikelihood` with one-hot mixing weights *is* the component, which keeps
 * this function and the mixture definition welded together by construction.
 */
export function beamComponents(zk: number, zStar: number, params: BeamParams): BeamComponents {
  return {
    hit: beamLikelihood(zk, zStar, { ...params, ...ONE_HOT.hit }),
    short: beamLikelihood(zk, zStar, { ...params, ...ONE_HOT.short }),
    max: beamLikelihood(zk, zStar, { ...params, ...ONE_HOT.max }),
    rand: beamLikelihood(zk, zStar, { ...params, ...ONE_HOT.rand }),
  };
}

/**
 * E-step for a single beam — eq. (6.28). Returns the posterior over which of
 * the four causes produced this reading. The four numbers sum to one, so they
 * can be used directly to tint a histogram bar by cause.
 */
export function beamResponsibilities(
  zk: number,
  zStar: number,
  params: BeamParams,
): BeamComponents {
  const c = beamComponents(zk, zStar, params);
  const hit = params.zHit * c.hit;
  const short = params.zShort * c.short;
  const max = params.zMax * c.max;
  const rand = params.zRand * c.rand;
  const total = hit + short + max + rand;
  if (!(total > 0)) {
    // No cause explains this reading at all. Refusing to guess beats
    // manufacturing a responsibility out of a division by zero.
    return { hit: 0, short: 0, max: 0, rand: 0 };
  }
  return { hit: hit / total, short: short / total, max: max / total, rand: rand / total };
}

/** Mean log-likelihood per beam under `params` — the quantity EM increases. */
export function meanLogLikelihood(
  z: readonly number[],
  zStar: readonly number[],
  params: BeamParams,
): number {
  if (z.length === 0) return 0;
  let total = 0;
  for (let i = 0; i < z.length; i++) {
    total += Math.log(Math.max(beamLikelihood(z[i], zStar[i], params), 1e-300));
  }
  return total / z.length;
}

export interface EmStepResult {
  params: BeamParams;
  /** Mean log-likelihood per beam *before* this step's M-update. */
  logLik: number;
}

/** Never let a fitted σ collapse onto a bin: the model would become a spike. */
const SIGMA_FLOOR = 1e-3;
const LAMBDA_RANGE: [number, number] = [1e-3, 50];

/**
 * One EM iteration — the E-step of eq. (6.28) followed by the closed-form
 * M-step of eqs. (6.30)–(6.32).
 *
 * Degenerate clusters are guarded rather than renormalized away: if no beam
 * takes responsibility for `hit`, σ_hit is *kept*, not recomputed from an empty
 * sum. A component whose weight has genuinely gone to zero stops moving instead
 * of exploding.
 */
export function emStep(
  z: readonly number[],
  zStar: readonly number[],
  params: BeamParams,
): EmStepResult {
  const n = z.length;
  if (n === 0) return { params, logLik: 0 };

  let sHit = 0;
  let sShort = 0;
  let sMax = 0;
  let sRand = 0;
  let sqHit = 0; // Σ e_hit (z − z*)²
  let zShortSum = 0; // Σ e_short z
  let ll = 0;

  for (let i = 0; i < n; i++) {
    const e = beamResponsibilities(z[i], zStar[i], params);
    sHit += e.hit;
    sShort += e.short;
    sMax += e.max;
    sRand += e.rand;
    const d = z[i] - zStar[i];
    sqHit += e.hit * d * d;
    zShortSum += e.short * z[i];
    ll += Math.log(Math.max(beamLikelihood(z[i], zStar[i], params), 1e-300));
  }

  const sigmaHit = sHit > 1e-9 ? Math.max(Math.sqrt(sqHit / sHit), SIGMA_FLOOR) : params.sigmaHit;
  const lambdaShort =
    zShortSum > 1e-9
      ? Math.min(Math.max(sShort / zShortSum, LAMBDA_RANGE[0]), LAMBDA_RANGE[1])
      : params.lambdaShort;

  return {
    logLik: ll / n,
    params: {
      ...params,
      zHit: sHit / n,
      zShort: sShort / n,
      zMax: sMax / n,
      zRand: sRand / n,
      sigmaHit,
      lambdaShort,
    },
  };
}

export interface LearnResult {
  params: BeamParams;
  /** Mean log-likelihood per beam at the start of each iteration. */
  logLik: number[];
}

/**
 * `learn_intrinsic_parameters` — Thrun et al., **Table 6.2**.
 *
 * Runs `iters` EM iterations from `init`. The returned `logLik` trace is what
 * the widget sparklines; it is non-decreasing, and its flattening is the only
 * honest stopping criterion.
 */
export function learnIntrinsicParameters(
  z: readonly number[],
  zStar: readonly number[],
  init: BeamParams,
  iters = 20,
): LearnResult {
  let params: BeamParams = { ...init };
  const logLik: number[] = [];
  for (let k = 0; k < iters; k++) {
    const step = emStep(z, zStar, params);
    logLik.push(step.logLik);
    params = step.params;
  }
  return { params, logLik };
}
