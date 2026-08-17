/**
 * The beam mixture as a *proper density forecast* — Chapter 25.
 *
 * Chapter 10 gave us `beamLikelihood`, which answers "how plausible is this
 * reading?". Calibration asks a different question: "where in the predictive
 * distribution did this reading land?", and that needs the **CDF**, not the
 * density. This file adds the two pieces Chapter 25 needs and Chapter 10 did
 * not: the analytic mixture CDF, and a sampler that draws from exactly the
 * density the CDF integrates — so that a model evaluated against its own
 * samples is calibrated by construction, and any miscalibration the widgets
 * show is a genuine model error rather than a simulation artefact.
 *
 * It also carries the one-parameter *variance scaling* family used for
 * post-hoc calibration (Guo et al. 2017's temperature scaling, transposed from
 * softmax logits to a range density).
 */

import { normalCdf } from '../prob/gaussian';
import type { Rng } from '../prob/rng';
import { beamLikelihood, type BeamParams } from '../models/sensor';

/** A reading within this of the sensor limit *is* a max-range reading. */
const MAX_EPS = 1e-9;

/**
 * Mixture CDF F(z | z*, Θ) of the four-component beam model.
 *
 * Component by component:
 *
 *   hit    truncated normal on [0, z_max]
 *   short  truncated exponential on [0, z*]
 *   max    a point mass at z_max — the atom that makes PIT values non-uniform
 *          for dropout beams unless they are randomized (see `randomizedPit`)
 *   rand   uniform on [0, z_max]
 */
export function beamCdf(zk: number, zStar: number, params: BeamParams): number {
  const { zHit, zShort, zMax, zRand, sigmaHit, lambdaShort, maxRange } = params;
  if (zk < 0) return 0;

  // hit: N(z*, σ²) restricted to [0, z_max] and renormalized.
  let cdfHit = 0;
  const lo = normalCdf(0, zStar, sigmaHit);
  const hi = normalCdf(maxRange, zStar, sigmaHit);
  const mass = hi - lo;
  if (mass > 1e-12) {
    const c = normalCdf(Math.min(zk, maxRange), zStar, sigmaHit);
    cdfHit = Math.min(1, Math.max(0, (c - lo) / mass));
  } else {
    cdfHit = zk >= zStar ? 1 : 0;
  }

  // short: Exp(λ) restricted to [0, z*].
  let cdfShort = 0;
  if (zStar > 0) {
    const denom = 1 - Math.exp(-lambdaShort * zStar);
    cdfShort =
      denom > 1e-12
        ? Math.min(1, (1 - Math.exp(-lambdaShort * Math.min(zk, zStar))) / denom)
        : zk >= zStar
          ? 1
          : 0;
  } else {
    cdfShort = 1;
  }

  const cdfMax = zk >= maxRange - MAX_EPS ? 1 : 0;
  const cdfRand = Math.min(1, zk / maxRange);

  return zHit * cdfHit + zShort * cdfShort + zMax * cdfMax + zRand * cdfRand;
}

/**
 * Draw one reading from the beam mixture — the *forward model* of Thrun et al.
 * §9.3.2, which is what supplies training pairs to every learned model here.
 */
export function sampleBeam(zStar: number, params: BeamParams, rng: Rng): number {
  const { zHit, zShort, zMax, sigmaHit, lambdaShort, maxRange } = params;
  const u = rng.next();

  if (u < zHit) {
    // Rejection-sample the truncated normal: cheap, and exactly the density
    // `beamCdf` integrates (a clipped sample would not be).
    for (let k = 0; k < 64; k++) {
      const z = rng.normal(zStar, sigmaHit);
      if (z >= 0 && z <= maxRange) return z;
    }
    return Math.min(maxRange, Math.max(0, zStar));
  }
  if (u < zHit + zShort) {
    if (zStar <= 0) return 0;
    // Inverse CDF of the exponential truncated to [0, z*].
    const denom = 1 - Math.exp(-lambdaShort * zStar);
    return -Math.log(1 - rng.next() * denom) / lambdaShort;
  }
  if (u < zHit + zShort + zMax) return maxRange;
  return rng.uniform(0, maxRange);
}

/**
 * The post-hoc calibration family: scale the hit width by `s`, leaving the
 * mixture weights alone.
 *
 * This is temperature scaling for a range density. For a pure Gaussian it is
 * *exactly* Guo et al.'s temperature on the log-likelihood — raising a Gaussian
 * to the power κ scales its variance by 1/κ — so `scale = 1/√κ` and the two
 * knobs in this chapter (variance scale and tempering exponent) are the same
 * knob seen from opposite ends.
 */
export function scaleBeamWidth(params: BeamParams, s: number): BeamParams {
  return { ...params, sigmaHit: params.sigmaHit * s };
}

/** Log density of one beam under the mixture, floored for log-safety. */
export function beamLogDensity(zk: number, zStar: number, params: BeamParams): number {
  return Math.log(Math.max(beamLikelihood(zk, zStar, params), 1e-300));
}

/**
 * **Randomized** probability integral transform.
 *
 * The plain PIT `v = F(z)` is uniform only for continuous forecasts. The beam
 * model has an atom of mass `z_max` at the sensor limit, so every dropout beam
 * would pile up at `v = 1` and the histogram would show a spike that is an
 * artefact of the atom, not a miscalibration. The standard repair (Brockwell,
 * and the randomized PIT of the forecasting literature) spreads each atom
 * uniformly across the jump it makes:
 *
 *   v = F(z⁻) + u · (F(z) − F(z⁻)),   u ~ U(0, 1)
 *
 * With a continuous forecast F(z⁻) = F(z) and this reduces to the plain PIT.
 */
export function randomizedPit(
  zk: number,
  zStar: number,
  params: BeamParams,
  rng: Rng,
): number {
  const upper = beamCdf(zk, zStar, params);
  const atMax = zk >= params.maxRange - MAX_EPS;
  const lower = atMax ? upper - params.zMax : upper;
  return Math.min(1, Math.max(0, lower + rng.next() * (upper - lower)));
}
