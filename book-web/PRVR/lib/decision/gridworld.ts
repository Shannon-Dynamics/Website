/**
 * Compiling worlds into MDPs — Chapter 21.
 *
 * The book's promise is that every probability has a pedigree, so none of the
 * transition models here are decreed. A gridworld's slip parameter is what you
 * get when the Chapter 9 velocity motion model is integrated over one cell
 * transit and binned: the commanded cell with probability 1 − 2s, each lateral
 * neighbour with probability s. `slipFromVelocityModel` does that integral by
 * sampling, so the number the widgets use is the number the motion model
 * predicts rather than a constant somebody liked.
 *
 * The TypeScript twin of `crates/ch21_mdp/src/gridworld.rs`.
 */

import { Rng } from '../prob/rng';
import { APARTMENT, distanceToWalls, type World } from '../sim/world';
import { condense, emptyMdp, type Mdp, type Transition } from './mdp';

/* -------------------------------------------------------------------------- */
/* Action sets                                                                 */
/* -------------------------------------------------------------------------- */

/** N, E, S, W as (dx, dy) with +y up — the order the arrow glyphs assume. */
export const MOVES4: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
];

/** The eight compass moves, counter-clockwise from north. */
export const MOVES8: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
];

export const MOVE_LABELS8 = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'];
export const MOVE_LABELS4 = ['N', 'E', 'S', 'W'];

/** A diagonal step is √2 cells long, so it costs √2 — geometry, not taste. */
export const moveLength = (d: readonly [number, number]): number => Math.hypot(d[0], d[1]);

/* -------------------------------------------------------------------------- */
/* Grid → MDP                                                                  */
/* -------------------------------------------------------------------------- */

export interface GridSpec {
  width: number;
  height: number;
  /** Walls. Blocked cells are states, but unreachable and absorbing. */
  blocked: boolean[];
  /** Payoff collected on *entering* the cell: goals positive, hazards negative. */
  payoff: number[];
  /** Absorbing cells — goals and terminal hazards. */
  terminal: boolean[];
  /** Probability of veering into *each* lateral neighbour. Intended: 1 − 2s. */
  slip: number;
  gamma: number;
  /** Cost of one cell of travel; a diagonal costs √2 of these. */
  stepCost: number;
  moves?: 4 | 8;
  /** Forbid squeezing diagonally between two blocked cells. */
  noCornerCutting?: boolean;
  /** Add a zero-motion action that still costs `stepCost` (rarely optimal). */
  allowStay?: boolean;
}

export const cellIndex = (spec: { width: number }, i: number, j: number): number =>
  j * spec.width + i;

export const cellCoords = (spec: { width: number }, s: number): [number, number] => [
  s % spec.width,
  Math.floor(s / spec.width),
];

/**
 * Build the finite MDP of a slippery gridworld.
 *
 * The transition rule in one sentence: the commanded direction happens with
 * probability 1 − 2s, each 45°/90° neighbour of it with probability s, and any
 * outcome that would leave the map or enter a wall leaves the robot where it
 * was — while still charging for the attempt, because the wheels turned.
 */
export function gridWorldMdp(spec: GridSpec): Mdp {
  const moves = spec.moves === 4 ? MOVES4 : MOVES8;
  const nDirs = moves.length;
  const nActions = nDirs + (spec.allowStay ? 1 : 0);
  const n = spec.width * spec.height;
  const s = Math.max(0, Math.min(0.49, spec.slip));

  const mdp = emptyMdp({
    nStates: n,
    nActions,
    gamma: spec.gamma,
    actionLabels: (spec.moves === 4 ? MOVE_LABELS4 : MOVE_LABELS8).concat(
      spec.allowStay ? ['stay'] : [],
    ),
  });

  const free = (i: number, j: number): boolean =>
    i >= 0 && j >= 0 && i < spec.width && j < spec.height && !spec.blocked[cellIndex(spec, i, j)];

  /** Where does direction `d` actually land you from (i, j)? */
  const land = (i: number, j: number, d: readonly [number, number]): number => {
    const ni = i + d[0];
    const nj = j + d[1];
    if (!free(ni, nj)) return cellIndex(spec, i, j);
    if (spec.noCornerCutting && d[0] !== 0 && d[1] !== 0) {
      if (!free(i + d[0], j) || !free(i, j + d[1])) return cellIndex(spec, i, j);
    }
    return cellIndex(spec, ni, nj);
  };

  for (let j = 0; j < spec.height; j++) {
    for (let i = 0; i < spec.width; i++) {
      const idx = cellIndex(spec, i, j);
      if (spec.blocked[idx] || spec.terminal[idx]) {
        mdp.absorbing[idx] = true;
        continue;
      }
      for (let a = 0; a < nActions; a++) {
        if (a === nDirs) {
          // "Stay": no motion, but the clock still runs.
          mdp.trans[idx][a] = [{ s: idx, p: 1 }];
          mdp.reward[idx][a] = -spec.stepCost;
          continue;
        }
        const outcomes: Transition[] = [];
        const push = (dir: number, p: number) => {
          if (p <= 0) return;
          outcomes.push({ s: land(i, j, moves[(dir + nDirs) % nDirs]), p });
        };
        push(a, 1 - 2 * s);
        push(a - 1, s);
        push(a + 1, s);
        const row = condense(outcomes);
        mdp.trans[idx][a] = row;

        // r(x, u) = −cost(u) + Σ p(x'|x,u) · payoff(x').
        let expectedPayoff = 0;
        for (const t of row) expectedPayoff += t.p * spec.payoff[t.s];
        mdp.reward[idx][a] = -spec.stepCost * moveLength(moves[a]) + expectedPayoff;
      }
    }
  }

  return mdp;
}

/** A blank spec sized to a world, with everything free and no payoff anywhere. */
export function blankGrid(width: number, height: number, base: Partial<GridSpec> = {}): GridSpec {
  const n = width * height;
  return {
    width,
    height,
    blocked: new Array<boolean>(n).fill(false),
    payoff: new Array<number>(n).fill(0),
    terminal: new Array<boolean>(n).fill(false),
    slip: 0.1,
    gamma: 0.98,
    stepCost: 1,
    moves: 8,
    noCornerCutting: true,
    ...base,
  };
}

/* -------------------------------------------------------------------------- */
/* The Apartment, discretized                                                  */
/* -------------------------------------------------------------------------- */

export interface ApartmentGrid extends GridSpec {
  cellSize: number;
  origin: { x: number; y: number };
  world: World;
}

/**
 * Chapter 13's occupancy grid is the state space: inflate the walls by the
 * robot radius, and every remaining cell is a state. Inflation is what makes a
 * point-robot plan safe for a robot with a body — the configuration-space trick
 * Chapter 20 introduced, arriving here as a one-line predicate.
 */
export function apartmentGrid(cellSize = 0.3, inflation = 0.16, base: Partial<GridSpec> = {}): ApartmentGrid {
  const world = APARTMENT;
  const width = Math.floor((world.bounds.maxX - world.bounds.minX) / cellSize);
  const height = Math.floor((world.bounds.maxY - world.bounds.minY) / cellSize);
  const spec = blankGrid(width, height, base) as ApartmentGrid;
  spec.cellSize = cellSize;
  spec.origin = { x: world.bounds.minX, y: world.bounds.minY };
  spec.world = world;

  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const { x, y } = cellCenter(spec, i, j);
      spec.blocked[cellIndex(spec, i, j)] = distanceToWalls(world, x, y) <= inflation;
    }
  }
  return spec;
}

export function cellCenter(
  spec: { cellSize: number; origin: { x: number; y: number }; width: number },
  i: number,
  j: number,
): { x: number; y: number } {
  return {
    x: spec.origin.x + (i + 0.5) * spec.cellSize,
    y: spec.origin.y + (j + 0.5) * spec.cellSize,
  };
}

/** World metres → grid indices, clamped to the map. */
export function cellAt(
  spec: { cellSize: number; origin: { x: number; y: number }; width: number; height: number },
  x: number,
  y: number,
): [number, number] {
  const i = Math.floor((x - spec.origin.x) / spec.cellSize);
  const j = Math.floor((y - spec.origin.y) / spec.cellSize);
  return [
    Math.max(0, Math.min(spec.width - 1, i)),
    Math.max(0, Math.min(spec.height - 1, j)),
  ];
}

/* -------------------------------------------------------------------------- */
/* Where the slip parameter comes from                                         */
/* -------------------------------------------------------------------------- */

export interface VelocitySlipParams {
  /** Commanded translational speed, m/s. */
  v: number;
  /** Chapter 9 noise parameters α₁…α₄ (α₅, α₆ affect final heading only). */
  alpha1: number;
  alpha2: number;
  alpha3: number;
  alpha4: number;
  cellSize: number;
  /**
   * Heading error at the moment the transit starts, in radians (one σ). The
   * robot aims at the next cell centre using the heading it *believes* it has,
   * so whatever the localizer is wrong by is a steering error that the wheels
   * then execute faithfully.
   */
  sigmaTheta: number;
  /**
   * Half-width of the angular sector that maps a displacement onto the
   * commanded neighbour: π/8 for the eight compass moves, π/4 for four.
   */
  sectorHalfWidth?: number;
}

/**
 * The slip parameter, derived rather than decreed.
 *
 * Drive the Chapter 9 velocity motion model (Thrun et al., Table 5.3) for one
 * cell transit, starting from a pose whose heading is uncertain, and bin the
 * displacement by *which neighbour it points at*: dead ahead if the net bearing
 * stays inside the commanded sector, one to the side otherwise. The lateral
 * fraction is 2s, so one side is half of it.
 *
 * Binning on bearing rather than on lateral distance is the honest
 * discretization: the grid asks "which cell did you end up in", and over a
 * single 0.3 m transit that question is answered almost entirely by direction.
 *
 * The chapter derives the closed form this samples. The chord of a circular arc
 * bisects the turn, so the bearing is exactly θ₀ + ½ω̂Δt; substituting
 * Δt = ℓ/v makes the speed cancel, leaving
 *
 *     s = Φ( −(π/8) / sqrt(σ_θ² + α₃ℓ²/4) )
 *
 * With α₃ = 0.05 and ℓ = 0.3 m the wheels contribute a fixed 0.034 rad and
 * everything else is `sigmaTheta`. Discretized slip is mostly a *localization*
 * error wearing a motion-model costume, which is exactly the observation
 * Chapter 22 turns into a different kind of planning problem.
 *
 * We sample rather than evaluate the formula, so that changing the motion model
 * changes the answer — and so that the one place sampling disagrees with the
 * formula stays visible: a Gaussian v̂ is negative with probability
 * Φ(−1/√α₁), and those draws reverse the bearing by π.
 */
export function slipFromVelocityModel(p: VelocitySlipParams, seed = 21, samples = 40000): number {
  const rng = new Rng(seed);
  const dt = p.cellSize / Math.max(p.v, 1e-6);
  const half = p.sectorHalfWidth ?? Math.PI / 8;
  let lateral = 0;
  for (let k = 0; k < samples; k++) {
    // The heading the robot actually has when it starts, relative to the one it
    // thinks it has (and therefore relative to the commanded direction).
    const theta0 = rng.normal(0, p.sigmaTheta);
    const vHat = p.v + rng.normal(0, Math.sqrt(p.alpha1 * p.v * p.v + p.alpha2 * 0));
    const wHat = rng.normal(0, Math.sqrt(p.alpha3 * p.v * p.v + p.alpha4 * 0));
    let dx: number;
    let dy: number;
    if (Math.abs(wHat) < 1e-9) {
      dx = vHat * Math.cos(theta0) * dt;
      dy = vHat * Math.sin(theta0) * dt;
    } else {
      const r = vHat / wHat;
      dx = -r * Math.sin(theta0) + r * Math.sin(theta0 + wHat * dt);
      dy = r * Math.cos(theta0) - r * Math.cos(theta0 + wHat * dt);
    }
    if (Math.abs(Math.atan2(dy, dx)) > half) lateral++;
  }
  // Both sides together are 2s, so one side is half of the tally.
  return Math.min(0.49, lateral / samples / 2);
}

/* -------------------------------------------------------------------------- */
/* The four-cell hallway — the chapter's hand-checkable example                */
/* -------------------------------------------------------------------------- */

export interface HallwaySsp {
  mdp: Mdp;
  labels: string[];
}

/**
 * States A, B, C, G in a line. Two actions:
 *
 *   `roll`  — one cell right with probability p, otherwise the wheels spin and
 *             the robot stays put. Costs 1.
 *   `lunge` — dumps enough current into the motors to guarantee the cell
 *             change. Costs `lungeCost`.
 *
 * γ = 1, G absorbing: a stochastic shortest path, which is what navigation
 * actually is. Every number in the chapter's worked example comes from here.
 */
export function hallwaySsp(pSuccess = 0.8, lungeCost = 2): HallwaySsp {
  const mdp = emptyMdp({ nStates: 4, nActions: 2, gamma: 1, actionLabels: ['roll', 'lunge'] });
  for (let s = 0; s < 3; s++) {
    mdp.trans[s][0] = condense([
      { s: s + 1, p: pSuccess },
      { s, p: 1 - pSuccess },
    ]);
    mdp.reward[s][0] = -1;
    mdp.trans[s][1] = [{ s: s + 1, p: 1 }];
    mdp.reward[s][1] = -lungeCost;
  }
  mdp.absorbing[3] = true;
  return { mdp, labels: ['A', 'B', 'C', 'G'] };
}

/* -------------------------------------------------------------------------- */
/* Cliff Run — two routes, one decision                                        */
/* -------------------------------------------------------------------------- */

export interface CliffConfig {
  /** Cells along the short route, which is flanked by the drop. */
  riskyLen: number;
  /** Cells along the detour, which is flanked by walls. */
  safeLen: number;
  /** Payoff for going over the edge. Positive number, applied as −penalty. */
  cliffPenalty: number;
  /** Probability of veering to *each* side per step. */
  slip: number;
}

export const RISKY = 0;
export const SAFE = 1;

export interface CliffRun {
  mdp: Mdp;
  /** State index → drawing position, in cells. */
  layout: { x: number; y: number }[];
  start: number;
  goal: number;
  /** Which route each state belongs to (−1 for start/goal). */
  route: number[];
}

/**
 * The two-route world, built as an explicit MDP rather than a grid so that its
 * value function has a closed form the reader can check.
 *
 * Both corridors are one cell wide. On the risky one a veer puts the robot over
 * the edge: it pays `cliffPenalty` and is dragged back to the start. On the
 * detour a veer only bumps a wall, costing a step. γ = 1; the goal absorbs.
 */
export function cliffRunMdp(cfg: CliffConfig): CliffRun {
  const s = Math.max(0, Math.min(0.49, cfg.slip));
  const q = 1 - 2 * s;
  const nRisky = Math.max(cfg.riskyLen - 1, 0);
  const nSafe = Math.max(cfg.safeLen - 1, 0);
  const START = 0;
  const GOAL = 1;
  const riskyBase = 2;
  const safeBase = riskyBase + nRisky;
  const nStates = safeBase + nSafe;

  const mdp = emptyMdp({ nStates, nActions: 2, gamma: 1, actionLabels: ['risky', 'safe'] });
  mdp.absorbing[GOAL] = true;

  // Risky state r_k means k moves still separate the robot from the goal.
  const riskyState = (k: number) => (k <= 0 ? GOAL : riskyBase + (k - 1));
  const safeState = (k: number) => (k <= 0 ? GOAL : safeBase + (k - 1));

  const riskyStep = (from: number, k: number) => {
    mdp.trans[from][RISKY] = condense([
      { s: riskyState(k - 1), p: q },
      { s: START, p: 2 * s },
    ]);
    mdp.reward[from][RISKY] = -1 - 2 * s * cfg.cliffPenalty;
  };
  const safeStep = (from: number, k: number, slipTo: number) => {
    mdp.trans[from][SAFE] = condense([
      { s: safeState(k - 1), p: q },
      { s: slipTo, p: 2 * s },
    ]);
    mdp.reward[from][SAFE] = -1;
  };

  riskyStep(START, cfg.riskyLen);
  safeStep(START, cfg.safeLen, START);
  for (let k = 1; k <= nRisky; k++) {
    const st = riskyState(k);
    riskyStep(st, k);
    // Once committed to a corridor there is only one thing to do; both actions
    // agree, so the max gate is a no-op away from the junction.
    mdp.trans[st][SAFE] = mdp.trans[st][RISKY];
    mdp.reward[st][SAFE] = mdp.reward[st][RISKY];
  }
  for (let k = 1; k <= nSafe; k++) {
    const st = safeState(k);
    safeStep(st, k, st);
    mdp.trans[st][RISKY] = mdp.trans[st][SAFE];
    mdp.reward[st][RISKY] = mdp.reward[st][SAFE];
  }

  const layout: { x: number; y: number }[] = new Array(nStates).fill(null).map(() => ({ x: 0, y: 0 }));
  const span = Math.max(cfg.riskyLen, cfg.safeLen);
  layout[START] = { x: 0, y: 0 };
  layout[GOAL] = { x: span, y: 0 };
  for (let k = 1; k <= nRisky; k++) {
    layout[riskyState(k)] = { x: ((cfg.riskyLen - k) / cfg.riskyLen) * span, y: 1 };
  }
  for (let k = 1; k <= nSafe; k++) {
    layout[safeState(k)] = { x: ((cfg.safeLen - k) / cfg.safeLen) * span, y: -1 };
  }
  const route = new Array<number>(nStates).fill(-1);
  for (let k = 1; k <= nRisky; k++) route[riskyState(k)] = RISKY;
  for (let k = 1; k <= nSafe; k++) route[safeState(k)] = SAFE;

  return { mdp, layout, start: START, goal: GOAL, route };
}

/** V^π(start) for the always-detour policy: −L/(1 − 2s). Closed form. */
export function safeRouteValue(cfg: CliffConfig): number {
  const q = 1 - 2 * cfg.slip;
  return -cfg.safeLen / q;
}

/**
 * V^π(start) for the always-cliff policy.
 *
 *   V = −(1 − q^L)(1 + 2sC) / (2s q^L),   q = 1 − 2s
 *
 * derived in the chapter by unrolling the corridor and solving the one linear
 * equation the restart-at-start term creates. As s → 0 it tends to −L.
 */
export function riskyRouteValue(cfg: CliffConfig): number {
  const s = cfg.slip;
  if (s < 1e-9) return -cfg.riskyLen;
  const q = 1 - 2 * s;
  const qL = Math.pow(q, cfg.riskyLen);
  return (-(1 - qL) * (1 + 2 * s * cfg.cliffPenalty)) / (2 * s * qL);
}

/**
 * The slip at which the optimal policy flips routes: the root of
 * V_risky(s) − V_safe(s). Bisection on two closed forms, not on a simulation.
 */
export function criticalSlip(cfg: CliffConfig, lo = 1e-6, hi = 0.45): number {
  const gap = (s: number) => riskyRouteValue({ ...cfg, slip: s }) - safeRouteValue({ ...cfg, slip: s });
  let a = lo;
  let b = hi;
  if (gap(a) < 0) return a; // the detour already wins at zero noise
  if (gap(b) > 0) return hi;
  for (let k = 0; k < 200; k++) {
    const m = 0.5 * (a + b);
    if (gap(m) > 0) a = m;
    else b = m;
  }
  return 0.5 * (a + b);
}
