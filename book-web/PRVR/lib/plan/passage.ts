/**
 * The narrow-passage benchmark: two rooms joined by one corridor whose width is
 * the only free parameter.
 *
 * This is the standard stress test for sampling-based planners, and the reason
 * probabilistic completeness has to be stated with a clearance δ in it. Uniform
 * sampling hits the passage with probability proportional to its *volume*, so
 * halving the width halves the hit rate — and the roadmap needs samples on both
 * sides *and* a connectable pair through the gap. The measured success curve
 * this module produces is the empirical face of that statement.
 */

import type { Rng } from '../prob/rng';
import type { Point2, Segment, World } from '../sim/world';
import { CSpace2 } from './cspace';
import { Prm, hybridSampler } from './sampling';

export const PASSAGE_BOUNDS = { minX: 0, minY: 0, maxX: 10, maxY: 6 };
export const PASSAGE_START: Point2 = { x: 1.6, y: 1.3 };
export const PASSAGE_GOAL: Point2 = { x: 8.4, y: 4.7 };
export const PASSAGE_RADIUS = 0.15;

const seg = (x1: number, y1: number, x2: number, y2: number): Segment => ({ x1, y1, x2, y2 });

/**
 * Two rooms, and a corridor of clear width `width` running between x = 4 and
 * x = 6. The corridor has real length, so a planner cannot slip through a
 * zero-thickness gap by luck.
 */
export function makePassageWorld(width: number): World {
  const half = width / 2;
  const yc = 3;
  return {
    name: `Passage(${width.toFixed(2)})`,
    bounds: PASSAGE_BOUNDS,
    walls: [
      seg(0, 0, 10, 0),
      seg(0, 6, 10, 6),
      seg(0, 0, 0, 6),
      seg(10, 0, 10, 6),
      // the two dividing walls, each with a gap
      seg(4, 0, 4, yc - half),
      seg(4, yc + half, 4, 6),
      seg(6, 0, 6, yc - half),
      seg(6, yc + half, 6, 6),
      // the corridor's own walls
      seg(4, yc - half, 6, yc - half),
      seg(4, yc + half, 6, yc + half),
    ],
  };
}

export function makePassageCSpace(width: number): CSpace2 {
  return new CSpace2(makePassageWorld(width), {
    radius: PASSAGE_RADIUS,
    cellSize: 0.025,
  });
}

/** Is this milestone inside the corridor? Used to colour the live samples. */
export function inPassage(q: Point2, width: number): boolean {
  return q.x > 4 && q.x < 6 && Math.abs(q.y - 3) < width / 2;
}

export interface TrialOptions {
  samples?: number;
  k?: number;
  maxEdgeLength?: number;
  bridge?: boolean;
}

/** Build one roadmap with the given budget and report whether it answers the query. */
export function passageTrial(cs: CSpace2, rng: Rng, opts: TrialOptions = {}): boolean {
  const { samples = 150, k = 8, maxEdgeLength = 1.6, bridge = false } = opts;
  const prm = new Prm(cs, {
    k,
    maxEdgeLength,
    sampler: bridge ? hybridSampler(cs, 0.6, 0.7) : undefined,
  });
  for (let i = 0; i < samples; i++) prm.step(rng);
  return prm.query(PASSAGE_START, PASSAGE_GOAL) !== null;
}

/**
 * Monte-Carlo success probability at one corridor width. Seeds are consecutive
 * integers so the curve is reproducible and the reader can re-derive it.
 */
export function successProbability(
  width: number,
  trials: number,
  opts: TrialOptions,
  makeRng: (seed: number) => Rng,
  cs = makePassageCSpace(width),
): number {
  let wins = 0;
  for (let t = 0; t < trials; t++) {
    if (passageTrial(cs, makeRng(1000 + t), opts)) wins++;
  }
  return wins / trials;
}
