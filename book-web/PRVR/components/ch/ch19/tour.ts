/**
 * The Chapter 19 lab: one Apartment log, replayed into four map representations.
 *
 * Every widget in this chapter drives the same robot over the same waypoints
 * with the same seed, so the memory numbers, the contour, and the distance
 * field the reader sees in one figure are the ones the next figure consumes.
 * The poses are given (Chapter 16's optimized pose graph, in the book's story),
 * which is what makes this a *mapping* chapter and not a SLAM chapter.
 */

import { thickenWorld } from '@/lib/sim/solid-world';
import { APARTMENT, beamAngles, type ScanParams, type World } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import type { Segment } from '@/lib/sim/world';

/** Rusty's apartment with 30 cm of masonry — see `lib/sim/solid-world.ts`. */
export const SOLID_APARTMENT: World = thickenWorld(APARTMENT, 0.3);

export const SCAN_PARAMS: ScanParams = {
  nBeams: 90,
  fov: 2 * Math.PI,
  maxRange: 8,
  sigma: 0.02,
};

export const SCAN_ANGLES: number[] = beamAngles(SCAN_PARAMS);

/**
 * A lap that visits both south rooms, the corridor, the study, and the bedroom,
 * passing through every doorway. Checked: no leg crosses a wall and every
 * waypoint keeps 25 cm of clearance.
 */
const WAYPOINTS: [number, number][] = [
  [1.0, 2.0], [3.0, 2.0], [3.0, 0.9], [2.0, 0.9], [2.0, 4.4],
  [3.25, 4.4], [3.25, 6.4], [4.6, 7.4], [1.2, 7.0], [3.25, 6.0], [3.25, 4.4],
  [6.0, 4.4], [6.0, 2.2], [7.2, 2.6], [6.0, 3.2], [6.0, 4.4],
  [7.85, 4.4], [7.85, 6.4], [11.4, 6.4], [11.4, 8.2], [11.4, 6.2], [7.85, 5.6], [7.85, 4.4],
  [10.0, 4.4], [10.0, 2.2], [11.2, 1.2], [9.0, 1.2], [10.0, 2.2], [10.0, 4.4],
  [2.0, 4.4], [2.0, 2.0], [1.0, 2.0],
];

/** How many scans make one full lap. */
export const LAP = 200;

/** Pose at step `k` of a `n`-step lap, heading along the path. */
export function tourPose(k: number, n: number = LAP): Pose2 {
  const legs = WAYPOINTS.length - 1;
  const u = ((k % n) / n) * legs;
  const i = Math.min(legs - 1, Math.floor(u));
  const t = u - i;
  const a = WAYPOINTS[i];
  const b = WAYPOINTS[i + 1];
  return {
    x: a[0] + t * (b[0] - a[0]),
    y: a[1] + t * (b[1] - a[1]),
    theta: Math.atan2(b[1] - a[1], b[0] - a[0]),
  };
}

/** An axis-aligned box of wall segments — the teleporting chair. */
export function boxSegments(cx: number, cy: number, half: number): Segment[] {
  return [
    { x1: cx - half, y1: cy - half, x2: cx + half, y2: cy - half },
    { x1: cx + half, y1: cy - half, x2: cx + half, y2: cy + half },
    { x1: cx + half, y1: cy + half, x2: cx - half, y2: cy + half },
    { x1: cx - half, y1: cy + half, x2: cx - half, y2: cy - half },
  ];
}

/** Two corridor spots the chair alternates between, both clear of the path. */
export const CHAIR_SPOTS: [number, number][] = [
  [5.0, 4.7],
  [5.0, 4.1],
];

export const CHAIR_HALF = 0.12;

/** The world Rusty actually senses: walls plus wherever the chair is now. */
export function worldWithChair(spot: number): World {
  const [cx, cy] = CHAIR_SPOTS[spot % CHAIR_SPOTS.length];
  return {
    ...SOLID_APARTMENT,
    walls: [...SOLID_APARTMENT.walls, ...boxSegments(cx, cy, CHAIR_HALF)],
  };
}
