/**
 * A planar reduction of Ferris — the model behind Chapter 18's reward mixer.
 *
 * A trotting quadruped moves, to first order, like a planar body on two virtual
 * legs half a cycle out of phase. Each leg is a spring-damper from the hip to a
 * phase-scheduled foot target, with Coulomb-limited tangential force while in
 * contact. That is enough physics for the reward terms of §18.3 to mean what
 * the chapter says they mean: effort really is force expenditure, slip really
 * is a foot sliding under load, air time really is a foot leaving the ground.
 *
 * The point is not fidelity — it is that when a reader moves a reward weight,
 * a real optimizer re-solves for a real gait, and the failures (creeping,
 * prancing, faceplanting) emerge instead of being drawn.
 */

import type { Rng } from './random';
import { gaussian } from './random';

export interface GaitParams {
  /** Step frequency (Hz). */
  frequency: number;
  /** Nominal leg length — effectively the ride height (m). */
  restLength: number;
  /** Leg spring stiffness (N/m). */
  stiffness: number;
  /** Foot clearance during swing (m). */
  clearance: number;
  /** Proportional gain on velocity error, i.e. how hard it pushes (N·s/m). */
  propulsion: number;
  /** Fraction of the cycle each leg spends in stance. */
  duty: number;
  /**
   * The speed the gait actually aims for. This is a *decision*, not the task:
   * the reward tracks the commanded speed, so choosing to go slower trades
   * tracking reward for lower effort. Without this the walker would servo to
   * the command no matter what the weights said, and no weight could ever
   * produce a creeping gait.
   */
  targetVelocity: number;
}

export const GAIT_BOUNDS: Record<keyof GaitParams, [number, number]> = {
  frequency: [1.0, 4.0],
  restLength: [0.25, 0.45],
  stiffness: [400, 3000],
  clearance: [0.01, 0.14],
  propulsion: [0, 260],
  duty: [0.35, 0.8],
  targetVelocity: [0, 1.6],
};

export interface RewardWeights {
  velocity: number;
  effort: number;
  airTime: number;
  orientation: number;
  slip: number;
}

export interface WalkerResult {
  /** Per-term reward totals, before weighting — so the UI can show the anatomy. */
  terms: Record<keyof RewardWeights, number>;
  weightedReturn: number;
  fell: boolean;
  meanSpeed: number;
  costOfTransport: number;
  /** Sampled body trajectory for drawing. */
  trace: Array<{ x: number; y: number; theta: number; contacts: [boolean, boolean] }>;
  /** Foot heights over time, for the gait diagram. */
  footHeights: Array<[number, number]>;
}

const G = 9.81;
const MASS = 12;
const INERTIA = 0.6;
const HIP_OFFSET = 0.22; // half the body length
const FRICTION = 0.8;
const DT = 0.002;

/**
 * Simulate the walker for `seconds` at a commanded forward speed.
 * Semi-implicit Euler at 500 Hz: stable for this stiffness range.
 */
export function simulateWalker(
  p: GaitParams,
  weights: RewardWeights,
  targetSpeed = 0.9,
  seconds = 3.0,
): WalkerResult {
  let x = 0;
  let y = p.restLength;
  let theta = 0;
  let vx = 0;
  let vy = 0;
  let omega = 0;

  const steps = Math.floor(seconds / DT);
  const terms: Record<keyof RewardWeights, number> = {
    velocity: 0,
    effort: 0,
    airTime: 0,
    orientation: 0,
    slip: 0,
  };

  // Per-leg airborne accumulator, credited at touchdown like the real term.
  const airborne = [0, 0];
  const wasContact = [false, false];
  let fell = false;
  const trace: WalkerResult['trace'] = [];
  const footHeights: WalkerResult['footHeights'] = [];
  let distance = 0;
  let energy = 0;

  for (let k = 0; k < steps; k++) {
    const t = k * DT;
    let fxTotal = 0;
    let fyTotal = -MASS * G;
    let torqueTotal = 0;
    const contacts: [boolean, boolean] = [false, false];
    const heights: [number, number] = [0, 0];

    for (let leg = 0; leg < 2; leg++) {
      const phase = (p.frequency * t + leg * 0.5) % 1; // trot: half a cycle apart
      const inStancePhase = phase < p.duty;

      // Hip position in world coordinates.
      const sign = leg === 0 ? 1 : -1;
      const hipX = x + sign * HIP_OFFSET * Math.cos(theta);
      const hipY = y + sign * HIP_OFFSET * Math.sin(theta);

      // Scheduled foot target: planted during stance, arcing during swing.
      let footTargetY: number;
      if (inStancePhase) {
        footTargetY = 0;
      } else {
        const swing = (phase - p.duty) / (1 - p.duty);
        footTargetY = p.clearance * Math.sin(Math.PI * swing);
      }

      const legLength = hipY - footTargetY;
      heights[leg] = footTargetY;

      // Contact when the scheduled foot is on the ground and the leg is loaded.
      const contact = inStancePhase && legLength < p.restLength && legLength > 0.05;
      contacts[leg] = contact;

      if (contact) {
        const compression = p.restLength - legLength;
        const damping = 2 * Math.sqrt(p.stiffness * MASS) * 0.12;
        const fN = Math.max(0, p.stiffness * compression - damping * vy);

        // Tangential propulsion, capped by the friction cone.
        const desired = p.propulsion * (p.targetVelocity - vx);
        const limit = FRICTION * fN;
        const fT = Math.max(-limit, Math.min(limit, desired));

        fyTotal += fN;
        fxTotal += fT;
        torqueTotal += sign * HIP_OFFSET * fN * 0.5 - 0.6 * omega;

        energy += (Math.abs(fT) + Math.abs(fN)) * Math.abs(vx) * DT;

        // Effort, scaled so a hard-working gait accumulates O(1) over an
        // episode — otherwise the weight could never trade against tracking.
        terms.effort += ((fN * fN * 0.25 + fT * fT * 4) / 1.4e4) * DT;

        // Slip is the friction cone being exceeded: the controller asked for
        // more tangential force than the contact can supply, so the foot slides.
        if (Math.abs(desired) > limit && fN > 1) {
          terms.slip += ((Math.abs(desired) - limit) / Math.max(fN, 1)) * DT * 6;
        }

        if (!wasContact[leg] && airborne[leg] > 0) {
          terms.airTime += (airborne[leg] - 0.15) * 3; // credited at touchdown
          airborne[leg] = 0;
        }
      } else {
        airborne[leg] += DT;
      }
      wasContact[leg] = contact;
    }

    // Integrate the body.
    vx += (fxTotal / MASS) * DT;
    vy += (fyTotal / MASS) * DT;
    omega += (torqueTotal / INERTIA) * DT;
    x += vx * DT;
    y += vy * DT;
    theta += omega * DT;

    // Ground stop, so a collapsed walker does not fall through the floor.
    if (y < 0.06) {
      y = 0.06;
      vy = Math.max(0, vy);
      fell = true;
    }
    if (Math.abs(theta) > 0.9) fell = true;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      fell = true;
      break;
    }

    terms.velocity += Math.exp(-((vx - targetSpeed) ** 2) / 0.25) * DT;
    // Scaled to the same O(1)-per-episode band as the other terms.
    terms.orientation += theta * theta * DT * 3;
    distance = x;

    if (k % 25 === 0) {
      trace.push({ x, y, theta, contacts });
      footHeights.push([heights[0], heights[1]]);
    }
  }

  const weightedReturn =
    weights.velocity * terms.velocity -
    weights.effort * terms.effort +
    weights.airTime * terms.airTime -
    weights.orientation * terms.orientation -
    weights.slip * terms.slip -
    (fell ? 3 : 0);

  const meanSpeed = distance / seconds;
  return {
    terms,
    weightedReturn,
    fell,
    meanSpeed,
    costOfTransport: distance > 0.05 ? energy / (MASS * G * distance) : Infinity,
    trace,
    footHeights,
  };
}

/**
 * A (mu, lambda) evolution strategy over the gait parameters.
 *
 * Small enough to run in a fraction of a second, which matters because the
 * reader changes a weight and expects a new gait, not a progress bar.
 */
export function optimizeGait(
  weights: RewardWeights,
  rng: Rng,
  opts: { generations?: number; population?: number; elites?: number; targetSpeed?: number } = {},
): { best: GaitParams; result: WalkerResult; history: number[] } {
  const { generations = 12, population = 24, elites = 6, targetSpeed = 0.9 } = opts;
  const keys = Object.keys(GAIT_BOUNDS) as Array<keyof GaitParams>;

  // Start from the middle of each range with a broad search width.
  const mean: Record<string, number> = {};
  const sigma: Record<string, number> = {};
  for (const k of keys) {
    const [lo, hi] = GAIT_BOUNDS[k];
    mean[k] = (lo + hi) / 2;
    sigma[k] = (hi - lo) / 4;
  }

  let best: GaitParams | null = null;
  let bestResult: WalkerResult | null = null;
  let bestScore = -Infinity;
  const history: number[] = [];

  for (let g = 0; g < generations; g++) {
    const scored: Array<{ score: number; p: GaitParams; r: WalkerResult }> = [];

    for (let i = 0; i < population; i++) {
      const cand = {} as GaitParams;
      for (const k of keys) {
        const [lo, hi] = GAIT_BOUNDS[k];
        cand[k] = Math.max(lo, Math.min(hi, mean[k] + sigma[k] * gaussian(rng)));
      }
      const r = simulateWalker(cand, weights, targetSpeed);
      scored.push({ score: r.weightedReturn, p: cand, r });
      if (r.weightedReturn > bestScore) {
        bestScore = r.weightedReturn;
        best = cand;
        bestResult = r;
      }
    }

    scored.sort((a, b) => b.score - a.score);
    const elite = scored.slice(0, elites);
    for (const k of keys) {
      const m = elite.reduce((acc, e) => acc + e.p[k], 0) / elite.length;
      const v = elite.reduce((acc, e) => acc + (e.p[k] - m) ** 2, 0) / elite.length;
      mean[k] = m;
      const [lo, hi] = GAIT_BOUNDS[k];
      sigma[k] = Math.max(Math.sqrt(v), (hi - lo) * 0.02);
    }
    history.push(bestScore);
  }

  return {
    best: best ?? ({} as GaitParams),
    result: bestResult ?? simulateWalker(best ?? ({} as GaitParams), weights, targetSpeed),
    history,
  };
}
