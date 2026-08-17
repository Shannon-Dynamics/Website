/**
 * `test_range_measurement` — Thrun et al., **Table 8.4**: localization in
 * environments that contain things the map does not.
 *
 * The static-world assumption is the load-bearing lie of every localizer in
 * Part IV. People are the usual violation, and they have a signature: a person
 * standing between the LiDAR and a wall makes the beam come back *short*. The
 * beam model of Chapter 10 already has a component for exactly this — the
 * `z_short` exponential — so the filter can compute, per beam, the posterior
 * probability that this reading was caused by an unmodelled object, and drop
 * the ones where that probability is high.
 *
 *   p(c̄ₜᵏ = short | zₜᵏ) ≈ Σᵢ z_short · p_short(zₜᵏ | xₜ^[i], m)
 *                          ────────────────────────────────────
 *                          Σᵢ p(zₜᵏ | xₜ^[i], m)
 *
 * with the sum running over particles, because the integral over bel(xₜ) has no
 * closed form.
 *
 * **A note on the printed pseudocode.** The text says the measurement is
 * "rejected if its probability of being caused by an unexpected obstacle
 * exceeds a user-selected threshold χ", but Table 8.4 accumulates `p` as the
 * *hit* mass and then returns `accept` when `p/q ≤ χ` — which rejects precisely
 * the beams the map explains best. We implement the prose, which is also what
 * the accompanying figures show, and flag the discrepancy rather than quietly
 * choosing one.
 */

import type { Pose2 } from '../geom/se2';
import { beamLikelihood, type BeamParams } from '../models/sensor';
import { rayCast, type World } from '../sim/world';

export interface BeamVerdict {
  /** p(short | zᵏ): the posterior share of the reading owed to p_short. */
  pShort: number;
  /** p(hit | zᵏ): the share the map itself explains. */
  pHit: number;
  reject: boolean;
}

/**
 * One beam, judged against a representative sample of the belief.
 *
 * `poses` should be a *subsample* of the particle set — Thrun's X̄ₜ is
 * "representative", not exhaustive, and 50 poses estimate a ratio of two sums
 * perfectly well while costing 50 ray casts instead of 5 000.
 *
 * The component densities come from the library's `beamLikelihood` with the
 * other mixture weights zeroed: the model is a linear mixture, so zeroing three
 * of the four weights returns the fourth term exactly. No second copy of
 * Table 6.1 exists in this file, on purpose.
 */
export function testRangeMeasurement(
  zk: number,
  beamAngle: number,
  poses: readonly Pose2[],
  world: World,
  params: BeamParams,
  chiReject: number,
): BeamVerdict {
  const shortOnly: BeamParams = { ...params, zHit: 0, zMax: 0, zRand: 0 };
  const hitOnly: BeamParams = { ...params, zShort: 0, zMax: 0, zRand: 0 };

  let numShort = 0;
  let numHit = 0;
  let denom = 0;

  for (const x of poses) {
    const zStar = rayCast(world, x.x, x.y, x.theta + beamAngle, params.maxRange);
    numShort += beamLikelihood(zk, zStar, shortOnly);
    numHit += beamLikelihood(zk, zStar, hitOnly);
    denom += beamLikelihood(zk, zStar, params);
  }

  if (!(denom > 0)) return { pShort: 0, pHit: 0, reject: false };
  const pShort = numShort / denom;
  return { pShort, pHit: numHit / denom, reject: pShort > chiReject };
}

/**
 * The whole sweep, beam by beam. Returns one verdict per beam, so a widget can
 * draw the rejected ones as stubs and the reader can see *which* readings the
 * filter refused — the asymmetry is only visible beam by beam.
 *
 * Note what this filter does **not** do: a surprisingly *long* reading has no
 * p_short mass at all, so it always survives. That asymmetry is the design.
 * Long readings are how a delocalized filter discovers it is lost, and a
 * symmetric outlier rejector would throw away exactly the evidence that drives
 * recovery.
 */
export function filterScan(
  ranges: readonly number[],
  angles: readonly number[],
  poses: readonly Pose2[],
  world: World,
  params: BeamParams,
  chiReject: number,
): BeamVerdict[] {
  const out: BeamVerdict[] = [];
  for (let k = 0; k < ranges.length; k++) {
    out.push(testRangeMeasurement(ranges[k], angles[k], poses, world, params, chiReject));
  }
  return out;
}

/** Evenly spaced subsample of a particle set: cheap, and unbiased under resampling. */
export function subsamplePoses(
  particles: readonly { state: Pose2 }[],
  count: number,
): Pose2[] {
  const M = particles.length;
  if (M === 0) return [];
  const n = Math.min(count, M);
  const stride = M / n;
  return Array.from({ length: n }, (_, i) => particles[Math.floor(i * stride)].state);
}
