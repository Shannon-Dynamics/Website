/**
 * Artificial potential fields, and the wave-front that cures them.
 *
 * Choset's formulation exactly: a quadratic-then-conic attractive well at the
 * goal, plus a repulsive barrier that switches on within Q* of an obstacle. The
 * robot is a bead on the surface U = U_att + U_rep and simply rolls downhill.
 *
 * It is the most seductive planner in the book, and the one that fails most
 * often, because U_att + U_rep is a *sum of two functions built with no regard
 * for each other*: nothing forbids a point where the two gradients cancel and
 * the Hessian is positive definite. That point is a local minimum in free
 * space, and the bead dies in it.
 *
 * The cure in this file is the wave-front: replace the invented potential with
 * the cost-to-go computed by search (`plan/search.ts`), which has no spurious
 * minima by construction. That function is a value function; Chapter 21 builds
 * the same object out of rewards instead of distances.
 */

import { distanceAt, type ExactDistanceField } from '../mapping/edt';
import type { Point2 } from '../sim/world';

export interface PotentialParams {
  /** ζ — attractive gain. */
  zeta: number;
  /** d*_goal — where the attractive well turns from quadratic to conic. */
  dGoalStar: number;
  /** η — repulsive gain. */
  eta: number;
  /** Q* — the obstacle's radius of influence. Beyond it, U_rep ≡ 0. */
  qStar: number;
}

export const DEFAULT_POTENTIAL: PotentialParams = {
  zeta: 1.0,
  dGoalStar: 1.5,
  eta: 0.6,
  qStar: 1.0,
};

/**
 * U_att: quadratic near the goal (so the gradient dies smoothly at it) and
 * conic far away (so a distant goal does not produce an enormous pull).
 */
export function attractivePotential(q: Point2, goal: Point2, p: PotentialParams): number {
  const d = Math.hypot(q.x - goal.x, q.y - goal.y);
  if (d <= p.dGoalStar) return 0.5 * p.zeta * d * d;
  return p.dGoalStar * p.zeta * d - 0.5 * p.zeta * p.dGoalStar * p.dGoalStar;
}

export function attractiveGradient(q: Point2, goal: Point2, p: PotentialParams): [number, number] {
  const dx = q.x - goal.x;
  const dy = q.y - goal.y;
  const d = Math.hypot(dx, dy);
  if (d < 1e-9) return [0, 0];
  if (d <= p.dGoalStar) return [p.zeta * dx, p.zeta * dy];
  const s = (p.dGoalStar * p.zeta) / d;
  return [s * dx, s * dy];
}

/** U_rep: a barrier that blows up at contact and is exactly zero beyond Q*. */
export function repulsivePotential(clearance: number, p: PotentialParams): number {
  const d = Math.max(clearance, 1e-3);
  if (d > p.qStar) return 0;
  const t = 1 / d - 1 / p.qStar;
  return 0.5 * p.eta * t * t;
}

/**
 * ∇U_rep = η (1/Q* − 1/D) D⁻² ∇D, with ∇D read off the distance field by
 * central differences. This is the term that reuses Chapter 19's ESDF: the
 * repulsive gradient *is* the direction away from the nearest obstacle.
 */
export function repulsiveGradient(
  field: ExactDistanceField,
  q: Point2,
  p: PotentialParams,
): [number, number] {
  const h = field.cellSize;
  const d = Math.max(distanceAt(field, q.x, q.y), 1e-3);
  if (d > p.qStar) return [0, 0];
  const gx = (distanceAt(field, q.x + h, q.y) - distanceAt(field, q.x - h, q.y)) / (2 * h);
  const gy = (distanceAt(field, q.x, q.y + h) - distanceAt(field, q.x, q.y - h)) / (2 * h);
  const s = p.eta * (1 / p.qStar - 1 / d) * (1 / (d * d));
  return [s * gx, s * gy];
}

export function totalPotential(
  field: ExactDistanceField,
  q: Point2,
  goal: Point2,
  p: PotentialParams,
): number {
  return attractivePotential(q, goal, p) + repulsivePotential(distanceAt(field, q.x, q.y), p);
}

export function totalGradient(
  field: ExactDistanceField,
  q: Point2,
  goal: Point2,
  p: PotentialParams,
): [number, number] {
  const [ax, ay] = attractiveGradient(q, goal, p);
  const [rx, ry] = repulsiveGradient(field, q, p);
  return [ax + rx, ay + ry];
}

export type DescentStatus = 'moving' | 'arrived' | 'stuck';

export interface DescentState {
  q: Point2;
  status: DescentStatus;
  /** ‖∇U‖ at the current point — the number that goes to zero in a trap. */
  gradNorm: number;
}

/**
 * `gradient_descent_potential(q_s, U, step)` — Choset's Algorithm 4.
 *
 * One step of normalized steepest descent, with the two termination tests the
 * algorithm actually needs: arrival (close to the goal) and *stuckness*
 * (‖∇U‖ below tolerance while not at the goal). The second one is the entire
 * lesson: the algorithm can tell you it has stopped, but not that it has failed
 * — those are the same event.
 */
export function descentStep(
  field: ExactDistanceField,
  s: DescentState,
  goal: Point2,
  p: PotentialParams,
  stepSize = 0.06,
  tol = 1e-3,
): DescentState {
  if (s.status !== 'moving') return s;
  const [gx, gy] = totalGradient(field, s.q, goal, p);
  const n = Math.hypot(gx, gy);
  if (Math.hypot(s.q.x - goal.x, s.q.y - goal.y) < 0.15) {
    return { ...s, status: 'arrived', gradNorm: n };
  }
  if (n < tol) return { ...s, status: 'stuck', gradNorm: n };
  const q = { x: s.q.x - (stepSize * gx) / n, y: s.q.y - (stepSize * gy) / n };
  return { q, status: 'moving', gradNorm: n };
}

/**
 * Steepest descent on a wave-front field instead of an invented potential.
 *
 * The field is sampled at the eight neighbours of the current cell and the bead
 * moves toward the smallest. Because the wave-front is a cost-to-go computed by
 * Dijkstra, *every* free cell except the goal has a strictly cheaper neighbour,
 * so this loop cannot get stuck — the discrete statement of "a navigation
 * function has a unique minimum".
 */
export function wavefrontDescentStep(
  cost: Float64Array,
  grid: { nx: number; ny: number; cellSize: number; bounds: { minX: number; minY: number } },
  q: Point2,
  stepSize = 0.06,
): { q: Point2; stuck: boolean } {
  const toCell = (x: number, y: number) => {
    const i = Math.min(grid.nx - 1, Math.max(0, Math.floor((x - grid.bounds.minX) / grid.cellSize)));
    const j = Math.min(grid.ny - 1, Math.max(0, Math.floor((y - grid.bounds.minY) / grid.cellSize)));
    return j * grid.nx + i;
  };
  const here = toCell(q.x, q.y);
  const i0 = here % grid.nx;
  const j0 = Math.floor(here / grid.nx);
  let best = cost[here];
  let bx = 0;
  let by = 0;
  for (let dj = -1; dj <= 1; dj++) {
    for (let di = -1; di <= 1; di++) {
      if (di === 0 && dj === 0) continue;
      const i = i0 + di;
      const j = j0 + dj;
      if (i < 0 || j < 0 || i >= grid.nx || j >= grid.ny) continue;
      const c = cost[j * grid.nx + i];
      if (c < best) {
        best = c;
        bx = di;
        by = dj;
      }
    }
  }
  if (bx === 0 && by === 0) return { q, stuck: true };
  const n = Math.hypot(bx, by);
  return { q: { x: q.x + (stepSize * bx) / n, y: q.y + (stepSize * by) / n }, stuck: false };
}
