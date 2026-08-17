/**
 * Map-conditioned motion — Thrun et al., *Probabilistic Robotics*, §5.5.
 *
 *   p(x_t | u_t, x_{t-1}, m) = η · p(x_t | m) · p(x_t | u_t, x_{t-1})
 *
 * with p(x_t | m) the indicator of free space. The sampler is rejection: draw
 * from the map-free model, keep the draw if the *endpoint* is free.
 *
 * The approximation is right there in that sentence. Free space is a property
 * of a pose, so the test can only ever look at where the robot ended up — never
 * at how it got there. A sample that drove through a wall and came to rest in
 * an open room passes. Thrun says as much in a footnote; this module makes the
 * failure measurable by also computing whether the *path* was feasible, which
 * is what the Map Squeeze widget colours differently.
 *
 * Paths here are SE(2) geodesics: the constant-twist arc from one pose to
 * another, which for the velocity motion model is exactly the trajectory the
 * sampled command traced out.
 */

import { boxminus, boxplus, type Pose2 } from '../geom/se2';
import { collides, isFree, type World } from '../sim/world';

export interface MapConditioning {
  /** Robot radius: how close the centre may come to a wall. */
  clearance: number;
  /** Segments used to discretise the arc for the path test. */
  segments: number;
}

export const DEFAULT_MAP_CONDITIONING: MapConditioning = { clearance: 0.16, segments: 14 };

/**
 * The constant-twist arc joining two poses, sampled at `segments + 1` points.
 *
 * ξ = to ⊟ from is the twist that takes one pose to the other in unit time, so
 * scaling it linearly and re-exponentiating walks the arc. This is why the
 * function needs no knowledge of (v, ω): the group already holds it.
 */
export function geodesicPath(from: Pose2, to: Pose2, segments = 14): Pose2[] {
  const xi = boxminus(to, from);
  const out: Pose2[] = [];
  for (let k = 0; k <= segments; k++) {
    const s = k / segments;
    out.push(boxplus(from, [xi[0] * s, xi[1] * s, xi[2] * s]));
  }
  return out;
}

/** p(x_t | m): the free-space indicator, as a density up to a constant. */
export function mapPrior(world: World, pose: Pose2, clearance: number): number {
  return isFree(world, pose.x, pose.y, clearance) ? 1 : 0;
}

export interface MapVerdict {
  pose: Pose2;
  path: Pose2[];
  /** Thrun's test: is the final pose in free space? */
  endpointFree: boolean;
  /** The test he cannot afford: did the whole arc stay in free space? */
  pathClear: boolean;
}

/**
 * Score one candidate pose against the map, both ways.
 *
 * `endpointFree && !pathClear` is the interesting cell of the truth table: the
 * sample the model accepts and physics would not. In a corridor those are rare;
 * beside a doorway they are the majority of the accepted mass, which is exactly
 * where a localizer needs to be trusted.
 */
export function classifyAgainstMap(
  world: World,
  from: Pose2,
  pose: Pose2,
  opts: MapConditioning = DEFAULT_MAP_CONDITIONING,
): MapVerdict {
  const path = geodesicPath(from, pose, opts.segments);
  const endpointFree = isFree(world, pose.x, pose.y, opts.clearance);
  let pathClear = true;
  for (let k = 1; k < path.length && pathClear; k++) {
    if (collides(world, path[k - 1], path[k])) pathClear = false;
    else if (!isFree(world, path[k].x, path[k].y, opts.clearance)) pathClear = false;
  }
  return { pose, path, endpointFree, pathClear };
}

export interface MapSampleResult extends MapVerdict {
  /** Draws taken before one was accepted; `tries > maxTries` means we gave up. */
  tries: number;
  accepted: boolean;
}

/**
 * `sample_motion_model_with_map` — Thrun et al., **Table 5.7**.
 *
 * Rejection sampling: keep drawing from the map-free model until the endpoint
 * lands in free space. Expected cost is 1/p_acc draws, so a robot pressed
 * against a wall with generous noise can spin here for a while — the widget
 * shows the acceptance rate for exactly that reason, and `maxTries` keeps an
 * animation frame from hanging when p_acc is effectively zero.
 */
export function sampleMotionModelWithMap(
  world: World,
  from: Pose2,
  draw: () => Pose2,
  opts: MapConditioning = DEFAULT_MAP_CONDITIONING,
  maxTries = 24,
): MapSampleResult {
  let last = classifyAgainstMap(world, from, draw(), opts);
  for (let t = 1; t <= maxTries; t++) {
    if (last.endpointFree) return { ...last, tries: t, accepted: true };
    last = classifyAgainstMap(world, from, draw(), opts);
  }
  return { ...last, tries: maxTries + 1, accepted: last.endpointFree };
}
