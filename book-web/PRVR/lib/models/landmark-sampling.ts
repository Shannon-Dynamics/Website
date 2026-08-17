/**
 * Inverting the landmark model into a sampler over poses — Thrun et al.,
 * **Table 6.5** (`sample_landmark_model_known_correspondence`).
 *
 * A range–bearing observation of a *known* landmark gives two constraints on a
 * three-dimensional pose, so it cannot localize the robot; it can only say that
 * the robot lies on a circle of radius r about the landmark, facing the
 * landmark to within φ. The sampler makes that explicit by drawing the missing
 * degree of freedom — the angle γ̂ of the robot around the landmark — uniformly,
 * and perturbing the observed range and bearing by their own noise.
 *
 * The result is the "donut" that Chapter 12 turns into a mixture proposal: draw
 * poses from a measurement instead of from the motion model, and a kidnapped
 * robot can recover in a single step.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';
import type { Rng } from '../prob/rng';
import type { Landmark } from '../sim/world';
import type { LandmarkSigmas, RangeBearingFeature } from './sensor';

/**
 * One pose drawn from p(x_t | f_t^i, c_t^i, m) under a uniform pose prior.
 *
 * Lines 3–5 of Table 6.5 exploit the symmetry of the Gaussian: perturbing the
 * *observation* by its noise is the same in law as perturbing the prediction,
 * which is what makes the inverse a two-line sampler instead of a rejection
 * loop. Line 8's `γ̂ − π − φ̂` is the heading that puts the landmark at relative
 * bearing φ̂ when the robot sits at angle γ̂ around it.
 */
export function sampleLandmarkModelKnownCorrespondence(
  feature: RangeBearingFeature,
  landmark: Landmark,
  sigmas: LandmarkSigmas,
  rng: Rng,
): Pose2 {
  const gamma = rng.uniform(0, 2 * Math.PI);
  // A range is a magnitude: a large σ_r on a small r can push it negative,
  // which would place the sample on the far side of the landmark.
  const rHat = Math.max(feature.r + rng.normal(0, sigmas.r), 0);
  const phiHat = feature.phi + rng.normal(0, sigmas.phi);
  return {
    x: landmark.x + rHat * Math.cos(gamma),
    y: landmark.y + rHat * Math.sin(gamma),
    theta: normalizeAngle(gamma - Math.PI - phiHat),
  };
}

/**
 * The same sampler, drawn `count` times — the cloud the widget plots.
 *
 * Kept separate from the single draw so the caller can decide whether it wants
 * one proposal (Chapter 12) or a picture of the whole annulus (this chapter).
 */
export function sampleLandmarkPoses(
  feature: RangeBearingFeature,
  landmark: Landmark,
  sigmas: LandmarkSigmas,
  rng: Rng,
  count: number,
): Pose2[] {
  return Array.from({ length: count }, () =>
    sampleLandmarkModelKnownCorrespondence(feature, landmark, sigmas, rng),
  );
}
