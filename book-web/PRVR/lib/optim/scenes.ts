/**
 * The datasets Chapter 15's widgets optimize.
 *
 * Three scenes, each cut down to exactly one lesson:
 *
 *  - `microChain1D` — the four-factor 1-D chain from the worked example, so a
 *    reader can check Ω, b and the solve on paper.
 *  - `apartmentLoopData` — Rusty drives out along the Apartment corridor, around
 *    room C and back, seeing corner landmarks; the odometry drifts, the loop
 *    closes, and least squares heals it.
 *  - `sparsityScene` / `rangeDuelScene` — small graphs whose *structure* (not
 *    whose realism) is the point.
 *
 * All of it is seeded: the same seed gives the same drift, every time.
 */

import { boxplus, compose, inverse, normalizeAngle, se2Exp, se2Log, type Pose2 } from '../geom/se2';
import { Rng } from '../prob/rng';
import { APARTMENT, collides } from '../sim/world';
import { L2, type Kernel } from './kernels';
import {
  bearingRangeFactor,
  betweenFactor,
  buildIndex,
  landmarkKey,
  linearFactor,
  poseKey,
  priorFactor,
  rangeToAnchorFactor,
  scalarKey,
  sqrtInfoDiag,
  type BlockIndex,
  type FactorGraph,
  type Point2,
  type Values,
} from './factor-graph';

/* -------------------------------------------------------------------------- */
/* The 1-D micro chain — the chapter's hand-checkable example                  */
/* -------------------------------------------------------------------------- */

/**
 * Three scalar variables, four unit-information factors:
 *
 *   prior  y₀ = 0        odometry  y₁ − y₀ = 1
 *   fix    y₂ = 1.5      odometry  y₂ − y₁ = 1
 *
 * The prior and the fix disagree by half a metre, and least squares splits the
 * disagreement across both odometry intervals. Ω is tridiagonal(−1, 2, −1),
 * b = (1, 0, −2.5)ᵀ, and one solve from y = 0 lands on (−0.125, 0.75, 1.625).
 */
export function microChain1D(): {
  graph: FactorGraph;
  index: BlockIndex;
  init: Values;
  expected: number[];
} {
  const keys = [scalarKey(0), scalarKey(1), scalarKey(2)];
  const graph: FactorGraph = {
    factors: [
      linearFactor([keys[0]], [1], 0, 1, { id: 'prior:y0' }),
      linearFactor([keys[0], keys[1]], [-1, 1], 1, 1, { id: 'odom:y0-y1' }),
      linearFactor([keys[1], keys[2]], [-1, 1], 1, 1, { id: 'odom:y1-y2' }),
      linearFactor([keys[2]], [1], 1.5, 1, { id: 'fix:y2' }),
    ],
  };
  return {
    graph,
    index: buildIndex(keys),
    init: { poses: [], landmarks: [], scalars: [0, 0, 0] },
    expected: [-0.125, 0.75, 1.625],
  };
}

/* -------------------------------------------------------------------------- */
/* The Apartment loop                                                          */
/* -------------------------------------------------------------------------- */

export interface OdometryEdge {
  i: number;
  j: number;
  delta: Pose2;
  kind: 'odometry' | 'loop';
}

export interface Observation {
  pose: number;
  lm: number;
  range: number;
  bearing: number;
}

export interface SlamData {
  truthPoses: Pose2[];
  truthLandmarks: Point2[];
  edges: OdometryEdge[];
  observations: Observation[];
  /** Dead-reckoned poses and triangulated landmarks: the optimizer's start. */
  init: Values;
  index: BlockIndex;
  sigmas: {
    odom: [number, number, number];
    obs: [number, number];
    prior: [number, number, number];
    loop: [number, number, number];
  };
  /** Poses whose relative transform the outlier closure lies about. */
  falseClosure: { i: number; j: number; truth: Pose2 };
}

/** Corner reflectors along Rusty's route — the landmarks the smoother estimates. */
const CORNERS: Point2[] = [
  { x: 2.5, y: 3.95 },
  { x: 5.5, y: 3.95 },
  { x: 6.5, y: 3.95 },
  { x: 9.5, y: 3.95 },
  { x: 2.8, y: 4.85 },
  { x: 3.7, y: 4.85 },
  { x: 7.4, y: 4.85 },
  { x: 8.3, y: 4.85 },
  { x: 8.25, y: 3.6 },
  { x: 11.75, y: 3.6 },
  { x: 11.75, y: 0.25 },
  { x: 8.25, y: 0.25 },
];

/**
 * Out along the corridor at y = 4.62, down into room C through its doorway,
 * once around the room, back up and home along y = 4.18 — so the last pose sits
 * half a metre from the first and a place-recognition front end can close the
 * loop.
 */
const ROUTE: Point2[] = [
  { x: 1.2, y: 4.62 },
  { x: 9.95, y: 4.62 },
  { x: 9.95, y: 3.2 },
  { x: 11.3, y: 3.2 },
  { x: 11.3, y: 0.9 },
  { x: 8.7, y: 0.9 },
  { x: 8.7, y: 3.2 },
  { x: 9.95, y: 3.2 },
  { x: 9.95, y: 4.18 },
  { x: 1.2, y: 4.18 },
];

/** Sample a polyline into poses whose heading follows the direction of travel. */
function samplePath(route: Point2[], step: number): Pose2[] {
  const poses: Pose2[] = [];
  for (let s = 0; s < route.length - 1; s++) {
    const a = route[s];
    const b = route[s + 1];
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    const heading = Math.atan2(b.y - a.y, b.x - a.x);
    const n = Math.max(1, Math.round(len / step));
    for (let k = 0; k < n; k++) {
      const t = k / n;
      poses.push({ x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y), theta: heading });
    }
  }
  const last = route[route.length - 1];
  const prev = route[route.length - 2];
  poses.push({ x: last.x, y: last.y, theta: Math.atan2(last.y - prev.y, last.x - prev.x) });
  return poses;
}

export interface ApartmentLoopOptions {
  seed?: number;
  /** Metres between consecutive poses. */
  step?: number;
  /** Odometry noise: translation σ (m per step) and rotation σ (rad per step). */
  odomSigma?: [number, number];
  /** Systematic under-rotation, as a fraction. This is what makes drift *drift*. */
  turnBias?: number;
  sensorRange?: number;
  obsSigma?: [number, number];
}

/**
 * Build the Apartment loop dataset: ground truth, noisy odometry, landmark
 * sightings, and the dead-reckoned initial estimate the optimizer starts from.
 */
export function apartmentLoopData(opts: ApartmentLoopOptions = {}): SlamData {
  const {
    seed = 42,
    step = 0.55,
    odomSigma = [0.035, 0.02],
    turnBias = 0.035,
    sensorRange = 2.6,
    obsSigma = [0.15, 0.07],
  } = opts;

  const rng = new Rng(seed);
  const truthPoses = samplePath(ROUTE, step);
  const n = truthPoses.length;

  // --- odometry: true relative transforms, corrupted in the tangent space ----
  const edges: OdometryEdge[] = [];
  const init: Values = { poses: [{ ...truthPoses[0] }], landmarks: [], scalars: [] };
  for (let i = 1; i < n; i++) {
    const rel = compose(inverse(truthPoses[i - 1]), truthPoses[i]);
    const tau = se2Log(rel);
    const scale = Math.max(Math.hypot(tau[0], tau[1]), 0.05);
    const noisy: [number, number, number] = [
      tau[0] + rng.normal(0, odomSigma[0] * scale),
      tau[1] + rng.normal(0, odomSigma[0] * scale),
      // The bias is deterministic, not random: a wheel radius that is 3.5% off
      // does not average away, and that is precisely why the loop fails to close.
      tau[2] * (1 - turnBias) + rng.normal(0, odomSigma[1]),
    ];
    const delta = se2Exp(noisy);
    edges.push({ i: i - 1, j: i, delta, kind: 'odometry' });
    init.poses.push(compose(init.poses[i - 1], delta));
  }

  // --- the true loop closure: the front end recognizes the starting place ----
  const closure = compose(inverse(truthPoses[n - 1]), truthPoses[0]);
  edges.push({
    i: n - 1,
    j: 0,
    delta: boxplus(closure, [rng.normal(0, 0.02), rng.normal(0, 0.02), rng.normal(0, 0.01)]),
    kind: 'loop',
  });

  // --- landmark sightings, with line-of-sight against the real floorplan -----
  const observations: Observation[] = [];
  const seenAt = new Map<number, number>();
  for (let p = 0; p < n; p++) {
    const x = truthPoses[p];
    CORNERS.forEach((lm, l) => {
      const dx = lm.x - x.x;
      const dy = lm.y - x.y;
      const range = Math.hypot(dx, dy);
      if (range > sensorRange || range < 0.25) return;
      if (collides(APARTMENT, { x: x.x, y: x.y }, { x: lm.x, y: lm.y })) return;
      const bearing = normalizeAngle(Math.atan2(dy, dx) - x.theta);
      const z = {
        pose: p,
        lm: l,
        range: range + rng.normal(0, obsSigma[0]),
        bearing: normalizeAngle(bearing + rng.normal(0, obsSigma[1])),
      };
      observations.push(z);
      if (!seenAt.has(l)) seenAt.set(l, observations.length - 1);
    });
  }

  // --- initialize each landmark from its first sighting, at the drifted pose -
  const lmIds = [...seenAt.keys()].sort((a, b) => a - b);
  const remap = new Map<number, number>();
  lmIds.forEach((id, k) => remap.set(id, k));
  const truthLandmarks = lmIds.map((id) => CORNERS[id]);
  for (const z of observations) z.lm = remap.get(z.lm) as number;
  init.landmarks = lmIds.map((id) => {
    const z = observations[seenAt.get(id) as number];
    const x = init.poses[z.pose];
    const a = x.theta + z.bearing;
    return { x: x.x + z.range * Math.cos(a), y: x.y + z.range * Math.sin(a) };
  });

  const index = buildIndex([
    ...truthPoses.map((_, i) => poseKey(i)),
    ...truthLandmarks.map((_, l) => landmarkKey(l)),
  ]);

  // A plausible-looking but wrong place recognition. The corridor and room C
  // are joined only through the long odometry chain, so a confident lie about
  // *their* relative pose is the one the map cannot argue with locally — which
  // is exactly why front-end false positives are the classic SLAM disaster.
  const i = Math.round(n * 0.11);
  const j = Math.round(n * 0.47);
  return {
    truthPoses,
    truthLandmarks,
    edges,
    observations,
    init,
    index,
    sigmas: {
      // Modeled slightly conservatively, as every deployed system does.
      odom: [0.04, 0.04, 0.025],
      obs: [obsSigma[0] * 1.2, obsSigma[1] * 1.2],
      prior: [0.01, 0.01, 0.005],
      loop: [0.05, 0.05, 0.03],
    },
    falseClosure: { i, j, truth: compose(inverse(truthPoses[i]), truthPoses[j]) },
  };
}

export interface GraphBuildOptions {
  /** Kernel applied to the loop closures — the factors that can be wrong. */
  kernel?: Kernel;
  /** Metres of error injected into the false closure. 0 disables the factor. */
  outlier?: number;
}

/**
 * Assemble the factor graph from the dataset: one prior, one factor per
 * control, one per sighting, one per closure. Nothing is fused, nothing is
 * marginalized — the graph *is* the data.
 */
export function buildSlamGraph(data: SlamData, opts: GraphBuildOptions = {}): FactorGraph {
  const { kernel = L2, outlier = 0 } = opts;
  const priorInfo = sqrtInfoDiag(data.sigmas.prior);
  const odomInfo = sqrtInfoDiag(data.sigmas.odom);
  const loopInfo = sqrtInfoDiag(data.sigmas.loop);
  const obsInfo = sqrtInfoDiag(data.sigmas.obs);

  const factors = [
    priorFactor(poseKey(0), data.truthPoses[0], priorInfo),
    ...data.edges.map((e) =>
      betweenFactor(e.i, e.j, e.delta, e.kind === 'loop' ? loopInfo : odomInfo, {
        kind: e.kind,
        kernel: e.kind === 'loop' ? kernel : L2,
      }),
    ),
    ...data.observations.map((z) =>
      bearingRangeFactor(z.pose, z.lm, { range: z.range, bearing: z.bearing }, obsInfo),
    ),
  ];

  if (outlier > 0) {
    const { i, j, truth } = data.falseClosure;
    factors.push(
      betweenFactor(i, j, boxplus(truth, [outlier, outlier * 0.35, 0.12]), loopInfo, {
        id: 'loop:false',
        kind: 'loop',
        kernel,
      }),
    );
  }
  return { factors };
}

/* -------------------------------------------------------------------------- */
/* A small graph whose structure is the lesson                                 */
/* -------------------------------------------------------------------------- */

export type SparsityPreset = 'chain' | 'loop' | 'hub';

export interface SparsityScene {
  graph: FactorGraph;
  values: Values;
  /** Where each variable is drawn in the graph view. */
  layout: { key: string; label: string; x: number; y: number; kind: 'pose' | 'landmark' }[];
  nPoses: number;
  nLandmarks: number;
}

/**
 * Eight poses and a handful of landmarks, laid out so the reader can see the
 * graph and the matrix at once.
 *
 *  - `chain`: a plain odometry chain. Banded Ω, no fill-in worth the name.
 *  - `loop`: the same chain with a closure — one edge, and suddenly ordering matters.
 *  - `hub`: one landmark observed from every pose. Eliminate it early and the
 *    whole trajectory becomes one dense clique; eliminate it last and nothing
 *    happens. Same posterior, wildly different cost.
 */
export function sparsityScene(preset: SparsityPreset = 'loop'): SparsityScene {
  const nPoses = 8;
  const radius = 2.2;
  const poses: Pose2[] = [];
  for (let i = 0; i < nPoses; i++) {
    const a = (2 * Math.PI * i) / nPoses;
    poses.push({ x: radius * Math.cos(a), y: radius * Math.sin(a), theta: a + Math.PI / 2 });
  }
  const landmarks: Point2[] =
    preset === 'hub'
      ? [{ x: 0, y: 0 }]
      : [
          { x: 0.9, y: 0.9 },
          { x: -0.9, y: 0.9 },
          { x: 0, y: -1.1 },
        ];

  const odomInfo = sqrtInfoDiag([0.05, 0.05, 0.03]);
  const obsInfo = sqrtInfoDiag([0.08, 0.03]);
  const factors = [priorFactor(poseKey(0), poses[0], sqrtInfoDiag([0.01, 0.01, 0.01]))];

  for (let i = 1; i < nPoses; i++) {
    factors.push(betweenFactor(i - 1, i, compose(inverse(poses[i - 1]), poses[i]), odomInfo));
  }
  if (preset !== 'chain') {
    factors.push(
      betweenFactor(nPoses - 1, 0, compose(inverse(poses[nPoses - 1]), poses[0]), odomInfo, {
        kind: 'loop',
      }),
    );
  }

  const visibility: number[][] =
    preset === 'hub'
      ? poses.map(() => [0])
      : [[0], [0], [0, 1], [1], [1], [2], [2], [0, 2]];

  visibility.forEach((lms, p) => {
    for (const l of lms) {
      const dx = landmarks[l].x - poses[p].x;
      const dy = landmarks[l].y - poses[p].y;
      factors.push(
        bearingRangeFactor(
          p,
          l,
          { range: Math.hypot(dx, dy), bearing: normalizeAngle(Math.atan2(dy, dx) - poses[p].theta) },
          obsInfo,
        ),
      );
    }
  });

  const layout = [
    ...poses.map((p, i) => ({
      key: `p${i}`,
      label: `x${i}`,
      x: p.x,
      y: p.y,
      kind: 'pose' as const,
    })),
    ...landmarks.map((m, l) => ({
      key: `l${l}`,
      label: `m${l}`,
      x: m.x,
      y: m.y,
      kind: 'landmark' as const,
    })),
  ];

  return {
    graph: { factors },
    values: { poses, landmarks, scalars: [] },
    layout,
    nPoses,
    nLandmarks: landmarks.length,
  };
}

/* -------------------------------------------------------------------------- */
/* The descent duel: one 2-D variable, a curved valley                         */
/* -------------------------------------------------------------------------- */

export interface DuelScene {
  graph: FactorGraph;
  index: BlockIndex;
  anchors: { p: Point2; r: number; sigma: number }[];
  solution: Point2;
}

/**
 * Trilateration with two strong ranges and one weak one.
 *
 * Two range circles meet at a shallow angle, so the cost has a long banana
 * valley and — at the midpoint between the anchors — a Jacobian that is nearly
 * rank-deficient across it. Gauss–Newton reads that flatness as "step a very
 * long way"; Levenberg–Marquardt reads it as "you cannot see far enough to
 * justify that". This is the smallest problem that shows the difference.
 */
export function rangeDuelScene(): DuelScene {
  const anchors = [
    { p: { x: -1.5, y: 0 }, r: 2.0, sigma: 0.05 },
    { p: { x: 1.5, y: 0 }, r: 2.0, sigma: 0.05 },
    { p: { x: 0, y: 2.6 }, r: 1.28, sigma: 0.35 },
  ];
  const graph: FactorGraph = {
    factors: anchors.map((a, i) =>
      rangeToAnchorFactor(0, a.p, a.r, a.sigma, { id: `range:a${i}` }),
    ),
  };
  return {
    graph,
    index: buildIndex([landmarkKey(0)]),
    anchors,
    solution: { x: 0, y: Math.sqrt(4 - 2.25) },
  };
}

/** A `Values` holding a single 2-D point — the duel's only unknown. */
export const duelValues = (p: Point2): Values => ({ poses: [], landmarks: [{ ...p }], scalars: [] });
