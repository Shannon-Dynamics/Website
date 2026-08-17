/**
 * The Chapter 23 control lab: two scenarios in the Apartment.
 *
 * Both reuse the world from `lib/sim/world.ts` and the exact distance transform
 * from `lib/mapping/edt.ts` — the ESDF of Chapter 19 — so the obstacle cost the
 * controller minimizes is the same field the mapper produces. Chairs are the
 * only thing added, because clutter that was not on the map when the path was
 * planned is the entire reason a local controller exists.
 */

import type { Pose2 } from '../geom/se2';
import { exactDistanceField, type ExactDistanceField } from '../mapping/edt';
import { APARTMENT, type Point2 } from '../sim/world';
import type { Obstacle } from './mppi';

/** ESDF resolution. 6 cm keeps the doorway gaps honest without a huge grid. */
const FIELD_CELL = 0.06;

let cached: ExactDistanceField | null = null;

/**
 * The Apartment's distance field, built once and shared by every widget in the
 * chapter. It costs one linear-time transform (Felzenszwalb–Huttenlocher), and
 * every rollout step afterwards is a single array read.
 */
export function apartmentField(): ExactDistanceField {
  if (!cached) cached = exactDistanceField(APARTMENT, FIELD_CELL);
  return cached;
}

/** Where the moving clutter is allowed to wander. */
export const CORRIDOR_BAND = { minX: 8.2, maxX: 10.9, minY: 3.8, maxY: 5.0 };

export interface ControlScene {
  name: string;
  start: Pose2;
  goal: Point2;
  /** Reference path from Chapter 20; empty for the pure goal-seeking scene. */
  path: Point2[];
  obstacles: Obstacle[];
  view: { minX: number; maxX: number; minY: number; maxY: number };
}

/**
 * **The corridor run.** Rusty tracks a path down the Apartment's corridor that
 * was planned on a map with no chairs in it. Two chairs have been moved since,
 * and a third is being pushed slowly across the corridor while he drives.
 *
 * Corridor walls are at y = 3.8 and y = 5.0, so with a 0.19 m body radius the
 * drivable band is roughly y ∈ [4.0, 4.8]. Each chair leaves about 0.35 m of
 * that band open — enough to pass, not enough to ignore.
 */
export function corridorRun(): ControlScene {
  return {
    name: 'Corridor run',
    start: { x: 1.15, y: 4.4, theta: 0 },
    goal: { x: 11.35, y: 4.4 },
    path: [
      { x: 1.15, y: 4.4 },
      { x: 3.5, y: 4.4 },
      { x: 6.0, y: 4.4 },
      { x: 8.5, y: 4.4 },
      { x: 11.35, y: 4.4 },
    ],
    obstacles: [
      { x: 4.3, y: 3.99, r: 0.17 },
      { x: 7.2, y: 4.81, r: 0.17 },
      { x: 10.4, y: 3.97, r: 0.15, vx: -0.32 },
    ],
    view: { minX: 0.2, maxX: 11.9, minY: 3.2, maxY: 5.6 },
  };
}

/**
 * **The counter pocket.** Rusty is behind the kitchen counter in room B, facing
 * the wrong way; the goal is 2 m north of him, on the other side of that
 * counter. The counter is a wall stub running from x = 4 to x = 6.2 at y = 1.4,
 * so the pocket it makes has exactly one exit — around the east end, *away*
 * from the goal.
 *
 * Every direction that reduces the straight-line distance to the goal presses
 * him into the counter. The escape costs about 1.9 m of travel in the wrong
 * direction before a single metre is repaid, which makes this the chapter's
 * cleanest instrument: a controller escapes it only if its horizon reaches past
 * the counter's east end, and a controller that considers one constant-curvature
 * arc at a time cannot express the escape at any horizon.
 */
export function counterPocket(): ControlScene {
  return {
    name: 'Counter pocket',
    start: { x: 4.6, y: 0.75, theta: Math.PI },
    goal: { x: 5.0, y: 2.6 },
    path: [],
    obstacles: [],
    view: { minX: 3.7, maxX: 8.3, minY: -0.3, maxY: 4.3 },
  };
}

/** Both scenes, keyed for a widget toggle. */
export const CH23_SCENES = {
  corridor: corridorRun,
  pocket: counterPocket,
} as const;

export type SceneKey = keyof typeof CH23_SCENES;
