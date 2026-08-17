/**
 * Dubins paths: the shortest curve between two poses for a car that can only
 * drive forwards and cannot turn tighter than radius ρ.
 *
 * Dubins (1957) proved the optimum is always one of six words built from three
 * primitives — L (left at full lock), R (right at full lock), S (straight):
 *
 *     LSL   RSR   LSR   RSL   RLR   LRL
 *
 * so "shortest path" is a closed-form enumeration, not a search. The algebra
 * below is the standard normalised formulation (Shkel & Lumelsky 2001): scale
 * distances by ρ, rotate so the start-to-goal vector lies along +x, and every
 * word reduces to a two-line expression in the normalised headings α, β and the
 * normalised separation d.
 *
 * Everything here is checked the only way worth trusting: `dubinsSample`
 * integrates the word it was handed, and `__checks_ch20__.ts` asserts the
 * integration lands on the requested goal pose to 1e-9. A mistyped `sin` cannot
 * survive that.
 */

import { normalizeAngle, type Pose2 } from '../geom/se2';

/** The six words, in Dubins' own order: CSC family first, then CCC. */
export const DUBINS_WORDS = ['LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL'] as const;
export type DubinsWord = (typeof DUBINS_WORDS)[number];

export interface DubinsPath {
  word: DubinsWord;
  /** Segment parameters in *normalised* units: arc angles (rad) and straight length/ρ. */
  segments: [number, number, number];
  /** Turning radius the path was built for. */
  rho: number;
  /** Total arc length, in metres. */
  length: number;
  start: Pose2;
  /** True when the car drives the curve in reverse (see {@link reedsSheppLite}). */
  reverse: boolean;
}

/** Wrap to [0, 2π) — the normalised headings must be positive to compare words. */
export function mod2pi(a: number): number {
  const x = a % (2 * Math.PI);
  return x < 0 ? x + 2 * Math.PI : x;
}

type WordSolver = (alpha: number, beta: number, d: number) => [number, number, number] | null;

const SOLVERS: Record<DubinsWord, WordSolver> = {
  LSL: (a, b, d) => {
    const p2 = 2 + d * d - 2 * Math.cos(a - b) + 2 * d * (Math.sin(a) - Math.sin(b));
    if (p2 < 0) return null;
    const tmp = Math.atan2(Math.cos(b) - Math.cos(a), d + Math.sin(a) - Math.sin(b));
    return [mod2pi(tmp - a), Math.sqrt(p2), mod2pi(b - tmp)];
  },
  RSR: (a, b, d) => {
    const p2 = 2 + d * d - 2 * Math.cos(a - b) + 2 * d * (Math.sin(b) - Math.sin(a));
    if (p2 < 0) return null;
    const tmp = Math.atan2(Math.cos(a) - Math.cos(b), d - Math.sin(a) + Math.sin(b));
    return [mod2pi(a - tmp), Math.sqrt(p2), mod2pi(tmp - b)];
  },
  LSR: (a, b, d) => {
    const p2 = -2 + d * d + 2 * Math.cos(a - b) + 2 * d * (Math.sin(a) + Math.sin(b));
    if (p2 < 0) return null;
    const p = Math.sqrt(p2);
    const tmp =
      Math.atan2(-Math.cos(a) - Math.cos(b), d + Math.sin(a) + Math.sin(b)) - Math.atan2(-2, p);
    return [mod2pi(tmp - a), p, mod2pi(tmp - mod2pi(b))];
  },
  RSL: (a, b, d) => {
    const p2 = d * d - 2 + 2 * Math.cos(a - b) - 2 * d * (Math.sin(a) + Math.sin(b));
    if (p2 < 0) return null;
    const p = Math.sqrt(p2);
    const tmp =
      Math.atan2(Math.cos(a) + Math.cos(b), d - Math.sin(a) - Math.sin(b)) - Math.atan2(2, p);
    return [mod2pi(a - tmp), p, mod2pi(b - tmp)];
  },
  RLR: (a, b, d) => {
    const t = (6 - d * d + 2 * Math.cos(a - b) + 2 * d * (Math.sin(a) - Math.sin(b))) / 8;
    if (Math.abs(t) > 1) return null;
    const p = mod2pi(2 * Math.PI - Math.acos(t));
    const t1 = mod2pi(
      a - Math.atan2(Math.cos(a) - Math.cos(b), d - Math.sin(a) + Math.sin(b)) + mod2pi(p / 2),
    );
    return [t1, p, mod2pi(a - b - t1 + mod2pi(p))];
  },
  LRL: (a, b, d) => {
    const t = (6 - d * d + 2 * Math.cos(a - b) + 2 * d * (Math.sin(b) - Math.sin(a))) / 8;
    if (Math.abs(t) > 1) return null;
    const p = mod2pi(2 * Math.PI - Math.acos(t));
    const t1 = mod2pi(
      -a - Math.atan2(Math.cos(a) - Math.cos(b), d + Math.sin(a) - Math.sin(b)) + p / 2,
    );
    return [t1, p, mod2pi(mod2pi(b) - a - t1 + mod2pi(p))];
  },
};

/** Which primitive each character of a word denotes, as a signed curvature. */
const CURVATURE: Record<string, number> = { L: 1, S: 0, R: -1 };

/** All six words that admit a solution for this query, cheapest first. */
export function dubinsAllWords(start: Pose2, goal: Pose2, rho: number): DubinsPath[] {
  const dx = goal.x - start.x;
  const dy = goal.y - start.y;
  const D = Math.hypot(dx, dy);
  const d = D / rho;
  const theta = D < 1e-12 ? 0 : Math.atan2(dy, dx);
  const alpha = mod2pi(start.theta - theta);
  const beta = mod2pi(goal.theta - theta);

  const out: DubinsPath[] = [];
  for (const word of DUBINS_WORDS) {
    const seg = SOLVERS[word](alpha, beta, d);
    if (!seg) continue;
    const length = (seg[0] + seg[1] + seg[2]) * rho;
    if (!Number.isFinite(length)) continue;
    out.push({ word, segments: seg, rho, length, start, reverse: false });
  }
  return out.sort((p, q) => p.length - q.length);
}

/** `dubins_shortest_path(q0, q1, ρ)` — the cheapest of the six. */
export function dubinsShortestPath(start: Pose2, goal: Pose2, rho: number): DubinsPath | null {
  const all = dubinsAllWords(start, goal, rho);
  return all.length > 0 ? all[0] : null;
}

/**
 * Reeds–Shepp, restricted to the words with no direction change.
 *
 * A car allowed to reverse can do strictly better than Dubins, and the full
 * Reeds–Shepp family (48 words, up to two cusps) captures that. This function
 * implements only the easy half of the story: the *pure-reverse* words, found
 * by flipping both headings by π, planning forwards, and driving the resulting
 * curve backwards. It is enough to show why reverse helps at close range — and
 * the chapter is explicit that a genuine Reeds–Shepp planner also allows
 * switching direction mid-path, which this does not.
 */
export function reedsSheppLite(start: Pose2, goal: Pose2, rho: number): DubinsPath | null {
  const fwd = dubinsShortestPath(start, goal, rho);
  const flip = (p: Pose2): Pose2 => ({ x: p.x, y: p.y, theta: p.theta + Math.PI });
  const back = dubinsShortestPath(flip(start), flip(goal), rho);
  if (back) back.start = flip(start);
  if (!fwd) return back ? { ...back, reverse: true } : null;
  if (!back) return fwd;
  return back.length < fwd.length ? { ...back, reverse: true } : fwd;
}

/**
 * Integrate a Dubins word into a polyline.
 *
 * Each segment is a unit-curvature arc or a straight run in normalised units,
 * so the update is the exact differential-drive integration of Chapter 4 with
 * v = 1 and ω = ±1/ρ — which is exactly why a Dubins path is drivable by Rusty
 * and a straight-line RRT path is not.
 */
export function dubinsSample(path: DubinsPath, step = 0.05): Pose2[] {
  const pts: Pose2[] = [];
  let p: Pose2 = { ...path.start };
  pts.push({ ...p });

  for (let s = 0; s < 3; s++) {
    const kind = path.word[s];
    const k = CURVATURE[kind];
    const len = path.segments[s] * path.rho; // metres (arc angle × ρ, or straight × ρ)
    if (len <= 1e-12) continue;
    const n = Math.max(1, Math.ceil(len / step));
    const ds = len / n;
    for (let i = 0; i < n; i++) {
      p = advance(p, ds, k / path.rho);
      pts.push({ ...p });
    }
  }

  if (path.reverse) {
    // The curve was planned with both headings flipped; report the *car's*
    // heading, which is the tangent turned by π.
    return pts.map((q) => ({ x: q.x, y: q.y, theta: normalizeAngle(q.theta + Math.PI) }));
  }
  return pts;
}

/** One unit-speed step of arc length `ds` at signed curvature `kappa`. */
function advance(p: Pose2, ds: number, kappa: number): Pose2 {
  if (Math.abs(kappa) < 1e-12) {
    return {
      x: p.x + ds * Math.cos(p.theta),
      y: p.y + ds * Math.sin(p.theta),
      theta: p.theta,
    };
  }
  const r = 1 / kappa;
  const nt = p.theta + kappa * ds;
  return {
    x: p.x + r * (Math.sin(nt) - Math.sin(p.theta)),
    y: p.y - r * (Math.cos(nt) - Math.cos(p.theta)),
    theta: nt,
  };
}

/**
 * Endpoint of a word, by integration — the ground truth the checks compare
 * against. Reported in the *car's* frame, so a reverse word ends at the goal
 * pose rather than at the flipped pose it was planned from.
 */
export function dubinsEndpoint(path: DubinsPath): Pose2 {
  let p: Pose2 = { ...path.start };
  for (let s = 0; s < 3; s++) {
    const k = CURVATURE[path.word[s]];
    const len = path.segments[s] * path.rho;
    p = advance(p, len, k / path.rho);
  }
  const theta = path.reverse ? p.theta + Math.PI : p.theta;
  return { x: p.x, y: p.y, theta: normalizeAngle(theta) };
}
