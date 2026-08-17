/**
 * Walls with an inside.
 *
 * Every earlier chapter models the Apartment as zero-thickness polylines, which
 * is exactly right for ray casting and for occupancy grids: a cell is occupied
 * or it is not, and a wall is a set of occupied cells. A *signed* distance field
 * asks a harder question — which side of the surface are you on? — and a curve
 * with no thickness has no answer. Observed from the corridor, the cell behind
 * the wall is inside; observed from the bedroom, the same cell is inside from
 * the other direction. Fusing those two claims produces a wall that cancels
 * itself out.
 *
 * So Chapter 19 gives Rusty's apartment 30 cm of masonry. Each segment becomes
 * a closed rectangle, rays stop at the near face, and the region between the
 * faces is genuinely interior — never observed, and correctly signed where it
 * is. This is the 2-D shadow of a real limitation: TSDF fusion degrades on
 * structures thinner than twice the truncation distance, which is why every
 * production system tunes τ against the thinnest thing it must reconstruct.
 */

import type { Segment, World } from './world';

/** Turn one segment into the four sides of a rectangle of the given thickness. */
function slab(s: Segment, t: number): Segment[] {
  const dx = s.x2 - s.x1;
  const dy = s.y2 - s.y1;
  const len = Math.hypot(dx, dy);
  if (len < 1e-9) return [];
  const ux = dx / len;
  const uy = dy / len;
  // Normal offset makes the two faces; the tangential extension mitres the
  // corners, so a wall junction does not leak a hairline gap a beam can enter.
  const nx = (-uy * t) / 2;
  const ny = (ux * t) / 2;
  const ex = (ux * t) / 2;
  const ey = (uy * t) / 2;

  const a: [number, number] = [s.x1 + nx - ex, s.y1 + ny - ey];
  const b: [number, number] = [s.x2 + nx + ex, s.y2 + ny + ey];
  const c: [number, number] = [s.x2 - nx + ex, s.y2 - ny + ey];
  const d: [number, number] = [s.x1 - nx - ex, s.y1 - ny - ey];
  return [
    { x1: a[0], y1: a[1], x2: b[0], y2: b[1] },
    { x1: b[0], y1: b[1], x2: c[0], y2: c[1] },
    { x1: c[0], y1: c[1], x2: d[0], y2: d[1] },
    { x1: d[0], y1: d[1], x2: a[0], y2: a[1] },
  ];
}

/**
 * A copy of `world` whose walls are solid slabs of the given thickness.
 * Landmarks and bounds are unchanged, so every other chapter's geometry still
 * lines up with it.
 */
export function thickenWorld(world: World, thickness = 0.3): World {
  return {
    ...world,
    name: `${world.name} (solid)`,
    walls: world.walls.flatMap((s) => slab(s, thickness)),
  };
}
