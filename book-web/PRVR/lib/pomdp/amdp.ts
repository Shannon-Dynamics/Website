/**
 * Augmented MDPs — coastal navigation (Roy & Thrun 1999; Thrun et al. §16.5).
 *
 * The belief over a 2-D pose is infinite-dimensional, so exact belief-space
 * planning is hopeless. The AMDP compression throws almost all of it away and
 * keeps two numbers: *where the robot most likely is*, and *how sure it is*.
 * Plan in that augmented space with the Chapter 21 machinery and uncertainty-
 * aware behaviour — wall hugging, detours to relocalize — falls out of the
 * value function rather than being scripted.
 *
 * The uncertainty coordinate here is an isotropic position standard deviation
 * σ, which for a 2-D Gaussian is entropy in disguise:
 *
 *     H(b) = log₂(2πe σ²)   bits
 *
 * so "minimize entropy" and "minimize σ" are the same instruction, and σ is the
 * one the reader can see on screen as the width of a ribbon.
 *
 * Two dynamics carry the whole model:
 *
 *   1. σ evolves by a scalar Kalman filter in information form —
 *      1/σ'² = 1/(σ² + σ_m²) + ν(x'), the Chapter 6 predict-then-correct with
 *      the map's *localizability* ν as the measurement information.
 *   2. ν(x) = ν_max · exp(−(d(x)/R)²), where d is the distance to the nearest
 *      wall and R the LiDAR range: a scan is informative when there is
 *      geometry inside it. Crank R up and the whole map becomes informative,
 *      which is exactly when the coastal detour stops paying.
 *
 * This is the TypeScript twin of `crates/ch22_pomdp/src/amdp.rs`.
 */

import {
  condense,
  emptyMdp,
  valueIteration,
  type Mdp,
  type SparseDist,
  type Transition,
} from '../decision/mdp';

/* -------------------------------------------------------------------------- */
/* The world                                                                   */
/* -------------------------------------------------------------------------- */

export interface CoastalWorld {
  cols: number;
  rows: number;
  /** Metres per cell. */
  cell: number;
  /** Row-major occupancy: true = wall. */
  occ: boolean[];
  /**
   * Which walls the robot can actually localize against.
   *
   * A long, smooth, featureless wall constrains the distance to it and nothing
   * else — the along-wall direction is unobservable, exactly the degeneracy
   * Chapter 16 fights in scan matching. Only textured surfaces (door frames,
   * shelving, radiators) pin a pose down, so occupancy and *information* are
   * two different maps.
   */
  feature: boolean[];
  start: number;
  /** The doorway cell the robot must pass through. */
  goal: number;
  /** Half-width of that doorway, in metres: how well you must know your pose. */
  doorHalfWidth: number;
}

export const idx = (w: CoastalWorld, cx: number, cy: number) => cy * w.cols + cx;
export const colOf = (w: CoastalWorld, i: number) => i % w.cols;
export const rowOf = (w: CoastalWorld, i: number) => Math.floor(i / w.cols);
/** Cell centre in metres. */
export const cellCenter = (w: CoastalWorld, i: number): [number, number] => [
  (colOf(w, i) + 0.5) * w.cell,
  (rowOf(w, i) + 0.5) * w.cell,
];

/**
 * The Hallway lab of this chapter: a wide open room whose far end is closed by
 * a smooth wall with a single doorway.
 *
 * The geometry is the argument. Crossing the middle is short and blind; the
 * long way round follows the textured south wall, which the LiDAR can lock onto
 * all the way to the door frame. Nothing in the map says "prefer walls" — the
 * planner has to earn that.
 */
export function makeCoastalHall(): CoastalWorld {
  const cols = 30;
  const rows = 21;
  const cell = 0.5;
  const n = cols * rows;
  const occ = new Array<boolean>(n).fill(false);
  const feature = new Array<boolean>(n).fill(false);
  const set = (x: number, y: number, textured: boolean) => {
    occ[y * cols + x] = true;
    feature[y * cols + x] = textured;
  };

  const doorRow = 3;
  const doorCol = cols - 5;

  for (let x = 0; x < cols; x++) {
    set(x, 0, true); // south wall: shelving, radiators, skirting — feature-rich
    set(x, rows - 1, true); // north wall likewise
  }
  for (let y = 0; y < rows; y++) {
    set(0, y, true); // the wall the robot undocks from
    set(cols - 1, y, false);
  }
  // The far partition: freshly plastered, smooth, and useless for localizing —
  // except for the door frame itself, which is a corner the scan can bite on.
  for (let y = 1; y < rows - 1; y++) {
    if (y !== doorRow) set(doorCol, y, Math.abs(y - doorRow) <= 1);
  }

  const w: CoastalWorld = {
    cols,
    rows,
    cell,
    occ,
    feature,
    start: 0,
    goal: 0,
    doorHalfWidth: 0.45,
  };
  w.start = idx(w, 2, rows - 4);
  w.goal = idx(w, doorCol, doorRow);
  return w;
}

/**
 * Distance in metres from every free cell to the nearest wall of a given kind,
 * by a multi-source relaxation on the 8-connected grid. Called twice: once over
 * all walls (clearance, for the collision hazard) and once over the textured
 * ones only (for the information field). Cheap, and accurate enough that the ν
 * field is smooth — the exact Euclidean transform of Chapter 19 would change no
 * decision this widget makes.
 */
export function distanceField(w: CoastalWorld, sources: readonly boolean[]): number[] {
  const n = w.cols * w.rows;
  const dist = new Array<number>(n).fill(Infinity);
  const queue: number[] = [];
  for (let i = 0; i < n; i++) {
    if (sources[i]) {
      dist[i] = 0;
      queue.push(i);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head];
    const cx = colOf(w, i);
    const cy = rowOf(w, i);
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w.cols || ny >= w.rows) continue;
        const j = ny * w.cols + nx;
        const step = dx !== 0 && dy !== 0 ? Math.SQRT2 : 1;
        if (dist[i] + step < dist[j]) {
          dist[j] = dist[i] + step;
          queue.push(j);
        }
      }
    }
  }
  return dist.map((d) => (Number.isFinite(d) ? d * w.cell : 0));
}

/** Distance to the nearest wall of any kind — what the robot can collide with. */
export const clearanceField = (w: CoastalWorld) => distanceField(w, w.occ);
/** Distance to the nearest *textured* wall — what the robot can localize against. */
export const featureField = (w: CoastalWorld) => distanceField(w, w.feature);

/**
 * ν(x) — measurement information per step, in m⁻².
 *
 * A scan localizes you against the texture it can reach. Far from every feature,
 * a short-range LiDAR returns max-range readings that constrain nothing, and the
 * Chapter 10 likelihood field is flat. This is the one modelling choice the
 * coastal effect rests on, so it gets its own slider.
 */
export function informationField(
  featureDist: readonly number[],
  range: number,
  nuMax: number,
): number[] {
  return featureDist.map((d) => nuMax * Math.exp(-((d / range) ** 2)));
}

/* -------------------------------------------------------------------------- */
/* The uncertainty coordinate                                                  */
/* -------------------------------------------------------------------------- */

export interface CoastalParams {
  /** LiDAR range in metres — the headline parameter. */
  range: number;
  /** Information a scan delivers with a wall right next to the robot, m⁻². */
  nuMax: number;
  /** Odometry drift per step, metres. */
  motionSigma: number;
  /** The σ grid the planner discretizes onto, metres. */
  sigmaBins: number[];
  /** Probability the wheels deliver a neighbouring direction instead. */
  slip: number;
  /** Weight on the per-step clipping hazard. */
  riskWeight: number;
  goalReward: number;
  lostPenalty: number;
  gamma: number;
}

export const DEFAULT_COASTAL: CoastalParams = {
  range: 2.5,
  nuMax: 6.0,
  motionSigma: 0.1,
  sigmaBins: [0.05, 0.08, 0.13, 0.2, 0.32, 0.5, 0.8, 1.3],
  slip: 0.1,
  riskWeight: 6,
  goalReward: 60,
  lostPenalty: -40,
  gamma: 0.98,
};

/**
 * One predict-correct cycle on the uncertainty coordinate, in information form.
 * Prediction adds variance; correction adds information. Chapter 6, scalar.
 */
export function sigmaStep(sigma: number, motionSigma: number, nu: number): number {
  const predicted = sigma * sigma + motionSigma * motionSigma;
  return Math.sqrt(1 / (1 / predicted + nu));
}

/** Entropy of an isotropic 2-D Gaussian belief, in bits. */
export const sigmaEntropy = (sigma: number) => Math.log2(2 * Math.PI * Math.E * sigma * sigma);

/** Nearest σ bin — the AMDP's lossy half. */
export function binOf(bins: readonly number[], sigma: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let k = 0; k < bins.length; k++) {
    const d = Math.abs(Math.log(bins[k]) - Math.log(Math.max(sigma, 1e-6)));
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

/**
 * P(‖position error‖ > d) for an isotropic 2-D Gaussian — a Rayleigh tail, in
 * closed form. Used twice: as the per-step probability of clipping a wall at
 * clearance d, and (complemented) as the probability of fitting through the
 * doorway.
 */
export const tailBeyond = (d: number, sigma: number) => Math.exp(-(d * d) / (2 * sigma * sigma));

/* -------------------------------------------------------------------------- */
/* Augmented_MDP_value_iteration                                               */
/* -------------------------------------------------------------------------- */

/** The eight moves, in the order the widgets draw their arrows. */
export const MOVES: readonly [number, number][] = [
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [1, -1],
];

export interface CoastalMdp {
  mdp: Mdp;
  world: CoastalWorld;
  params: CoastalParams;
  clearance: number[];
  featureDist: number[];
  nu: number[];
  /** Free cells, in state order. */
  cells: number[];
  /** cell index → position in `cells`, or −1 for a wall. */
  cellSlot: number[];
  /** State id of (cell, bin). */
  stateOf: (cell: number, bin: number) => number;
  /** The absorbing state every episode ends in. */
  doneState: number;
}

export interface BuildOptions {
  /**
   * Plan as if the robot were always perfectly localized: freeze σ at the
   * smallest bin inside the model. This is the certainty-equivalent planner —
   * Chapter 21 run on the MAP estimate — and it produces the shortest path.
   */
  certaintyEquivalent?: boolean;
}

/**
 * Compile (cell, σ-bin) into a finite MDP.
 *
 * Every ingredient is already in the book: the transition over cells is
 * Chapter 21's slipping grid, the σ transition is Chapter 6's scalar filter,
 * and the rewards are expectations over the *next* state, exactly the
 * convention `lib/decision/mdp.ts` documents.
 */
export function buildCoastalMdp(
  world: CoastalWorld,
  params: CoastalParams,
  opts: BuildOptions = {},
): CoastalMdp {
  const { certaintyEquivalent = false } = opts;
  const clearance = clearanceField(world);
  const featureDist = featureField(world);
  const nu = informationField(featureDist, params.range, params.nuMax);
  const bins = params.sigmaBins;
  const nBins = bins.length;

  const cells: number[] = [];
  const cellSlot = new Array<number>(world.cols * world.rows).fill(-1);
  for (let i = 0; i < world.occ.length; i++) {
    if (!world.occ[i]) {
      cellSlot[i] = cells.length;
      cells.push(i);
    }
  }

  const nStates = cells.length * nBins + 1;
  const doneState = nStates - 1;
  const mdp = emptyMdp({
    nStates,
    nActions: MOVES.length,
    gamma: params.gamma,
    actionLabels: ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'],
  });
  mdp.absorbing[doneState] = true;

  const stateOf = (cell: number, bin: number) => cellSlot[cell] * nBins + bin;

  /** Where the wheels actually land: intended, or one of the two neighbours. */
  const outcomes = (cell: number, a: number): { cell: number; p: number }[] => {
    const cx = colOf(world, cell);
    const cy = rowOf(world, cell);
    const attempt = (dir: number) => {
      const [dx, dy] = MOVES[(dir + MOVES.length) % MOVES.length];
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= world.cols || ny >= world.rows) return cell;
      const j = ny * world.cols + nx;
      return world.occ[j] ? cell : j; // bump into a wall and you stay put
    };
    return [
      { cell: attempt(a), p: 1 - params.slip },
      { cell: attempt(a - 1), p: params.slip / 2 },
      { cell: attempt(a + 1), p: params.slip / 2 },
    ];
  };

  for (let c = 0; c < cells.length; c++) {
    const cell = cells[c];
    for (let a = 0; a < MOVES.length; a++) {
      const outs = outcomes(cell, a);
      const dist = MOVES[a][0] !== 0 && MOVES[a][1] !== 0 ? Math.SQRT2 : 1;
      const stepCost = -dist * world.cell;

      for (let bin = 0; bin < nBins; bin++) {
        const s = stateOf(cell, bin);
        const pairs: Transition[] = [];
        let reward = stepCost;

        for (const { cell: next, p } of outs) {
          if (p <= 0) continue;
          const sigmaNext = certaintyEquivalent
            ? bins[0]
            : sigmaStep(bins[bin], params.motionSigma, nu[next]);
          // Clipping hazard: the chance the true pose is outside the free space
          // the planner thinks it is in.
          reward -= p * params.riskWeight * tailBeyond(clearance[next], sigmaNext);

          if (next === world.goal) {
            // The doorway is a hypothesis test the robot either passes or fails,
            // and it is graded on the uncertainty it *aimed* with: the scan taken
            // inside the gap arrives after the robot has committed to the gap.
            const sigmaAim = certaintyEquivalent
              ? bins[0]
              : Math.hypot(bins[bin], params.motionSigma);
            const pass = 1 - tailBeyond(world.doorHalfWidth, sigmaAim);
            reward += p * (pass * params.goalReward + (1 - pass) * params.lostPenalty);
            pairs.push({ s: doneState, p });
          } else {
            pairs.push({ s: stateOf(next, binOf(bins, sigmaNext)), p });
          }
        }

        mdp.trans[s][a] = condense(pairs) as SparseDist;
        mdp.reward[s][a] = reward;
      }
    }
  }

  return { mdp, world, params, clearance, featureDist, nu, cells, cellSlot, stateOf, doneState };
}

export interface CoastalPlan extends CoastalMdp {
  policy: number[];
  v: number[];
  sweeps: number;
}

/** `Augmented_MDP_value_iteration()` — Chapter 21's solver, augmented state. */
export function solveCoastal(
  world: CoastalWorld,
  params: CoastalParams,
  opts: BuildOptions = {},
): CoastalPlan {
  const built = buildCoastalMdp(world, params, opts);
  const solved = valueIteration(built.mdp, { eps: 1e-6 });
  return { ...built, policy: solved.policy, v: solved.v, sweeps: solved.sweeps };
}

/* -------------------------------------------------------------------------- */
/* Execution — both pilots run against the same honest dynamics                */
/* -------------------------------------------------------------------------- */

export interface RandomSource {
  next(): number;
}

export type RunOutcome = 'running' | 'arrived' | 'missed-door' | 'clipped' | 'timeout';

export interface CoastalRun {
  cell: number;
  sigma: number;
  steps: number;
  /** Undiscounted realized cost/reward, for the scoreboard. */
  score: number;
  outcome: RunOutcome;
  /** Cell trail with the σ that held on arrival — the entropy ribbon. */
  trail: { cell: number; sigma: number }[];
}

export function startRun(world: CoastalWorld, sigma0 = 0.25): CoastalRun {
  return {
    cell: world.start,
    sigma: sigma0,
    steps: 0,
    score: 0,
    outcome: 'running',
    trail: [{ cell: world.start, sigma: sigma0 }],
  };
}

/**
 * One step of the *true* system, for either pilot.
 *
 * The planner's σ-bins are a fiction it uses to think with; here σ evolves
 * continuously, the wheels slip, and both hazards are sampled rather than
 * averaged. That gap — between the model the policy was computed in and the
 * world it is executed in — is the whole reason the tally at the bottom of the
 * widget is worth watching.
 */
export function stepRun(
  plan: CoastalPlan,
  run: CoastalRun,
  rng: RandomSource,
  maxSteps = 160,
): CoastalRun {
  if (run.outcome !== 'running') return run;
  const { world, params, clearance, nu } = plan;
  const bins = params.sigmaBins;

  const a = plan.policy[plan.stateOf(run.cell, binOf(bins, run.sigma))];
  const cx = colOf(world, run.cell);
  const cy = rowOf(world, run.cell);

  // Slip: the commanded direction, or a neighbour of it.
  const u = rng.next();
  const dir = u < params.slip / 2 ? a - 1 : u < params.slip ? a + 1 : a;
  const [dx, dy] = MOVES[((dir % MOVES.length) + MOVES.length) % MOVES.length];
  const nx = cx + dx;
  const ny = cy + dy;
  let next = run.cell;
  if (nx >= 0 && ny >= 0 && nx < world.cols && ny < world.rows) {
    const j = ny * world.cols + nx;
    if (!world.occ[j]) next = j;
  }

  const sigma = sigmaStep(run.sigma, params.motionSigma, nu[next]);
  const moved = dx !== 0 && dy !== 0 ? Math.SQRT2 : 1;
  let score = run.score - moved * world.cell;
  const steps = run.steps + 1;
  const trail = [...run.trail, { cell: next, sigma }];

  // Did the robot clip a wall it did not know it was near?
  if (rng.next() < tailBeyond(clearance[next], sigma)) {
    return { cell: next, sigma, steps, score: score - params.riskWeight, outcome: 'clipped', trail };
  }

  if (next === world.goal) {
    // Graded on the σ the robot aimed with, not the one the doorway's own walls
    // hand back a moment too late.
    const pass = 1 - tailBeyond(world.doorHalfWidth, Math.hypot(run.sigma, params.motionSigma));
    const ok = rng.next() < pass;
    score += ok ? params.goalReward : params.lostPenalty;
    return { cell: next, sigma, steps, score, outcome: ok ? 'arrived' : 'missed-door', trail };
  }

  return {
    cell: next,
    sigma,
    steps,
    score,
    outcome: steps >= maxSteps ? 'timeout' : 'running',
    trail,
  };
}

export interface Tally {
  runs: number;
  arrived: number;
  clipped: number;
  missed: number;
  timeout: number;
  totalScore: number;
  totalSteps: number;
  /** σ at the moment the run ended, summed — the "entropy at the goal" number. */
  totalFinalSigma: number;
}

export const emptyTally = (): Tally => ({
  runs: 0,
  arrived: 0,
  clipped: 0,
  missed: 0,
  timeout: 0,
  totalScore: 0,
  totalSteps: 0,
  totalFinalSigma: 0,
});

export function record(tally: Tally, run: CoastalRun): Tally {
  return {
    runs: tally.runs + 1,
    arrived: tally.arrived + (run.outcome === 'arrived' ? 1 : 0),
    clipped: tally.clipped + (run.outcome === 'clipped' ? 1 : 0),
    missed: tally.missed + (run.outcome === 'missed-door' ? 1 : 0),
    timeout: tally.timeout + (run.outcome === 'timeout' ? 1 : 0),
    totalScore: tally.totalScore + run.score,
    totalSteps: tally.totalSteps + run.steps,
    totalFinalSigma: tally.totalFinalSigma + run.sigma,
  };
}
