/**
 * Utilities: turning "how much would I learn" and "how far would I walk" into
 * one number a robot can argue about — Chapter 24.
 *
 *   U(a) = w_I · I_map(a)  +  w_G · Δ log det Ω(a)  −  w_C · C(a)
 *
 * The first term comes from the grid (`info-gain.ts`), the third from the
 * navigation field (`frontier.ts`), and the second from the pose graph that
 * Chapters 15 and 16 built — the same Ω, now read as a decision variable rather
 * than as a linear system.
 *
 * Rust counterpart: `crates/ch24_explore/src/utility.rs`.
 */

import { adjoint, inverse, normalizeAngle, type Pose2 } from '../geom/se2';
import type { OccupancyGrid } from '../mapping/occgrid';
import { cholesky, matMul, transpose, zerosMat, type Mat } from '../prob/linalg';
import type { PoseEdge, PoseGraph } from '../slam/posegraph';
import { expectedInfoGain, type SensingParams } from './info-gain';
import type { Frontier, GridIdx, NavField } from './frontier';

export interface UtilityWeights {
  /** Bits of map entropy are worth this much. */
  wI: number;
  /** Nats of log det Ω are worth this much. Zero recovers pure coverage. */
  wG: number;
  /** Metres of travel cost this much. Large w_C is the nearest-frontier policy. */
  wC: number;
}

export const DEFAULT_WEIGHTS: UtilityWeights = { wI: 1, wG: 0.15, wC: 0.35 };

export interface Candidate {
  frontier: Frontier;
  /** The reachable cell we would actually drive to. */
  target: GridIdx;
  /** Sensing pose: the target, facing the unknown. */
  pose: Pose2;
  /** I(a), bits. */
  gain: number;
  /** C(a), metres of known-free path. */
  cost: number;
  /** Δ_a = log det Ω_{+a} − log det Ω, nats. Zero when no graph is supplied. */
  graphGain: number;
  utility: number;
}

/**
 * `score_candidates` — one utility per frontier region.
 *
 * The sensing pose faces from the reachable target toward the frontier's
 * centroid, because a range finder pointed back down the corridor it just drove
 * learns nothing. `expectedInfoGain` is evaluated at that pose on the *current*
 * map: the estimate is a forward simulation through what the robot believes,
 * which is the only map it has.
 */
export function scoreCandidates(
  grid: OccupancyGrid,
  frontiers: Frontier[],
  nav: NavField,
  sensing: SensingParams,
  weights: UtilityWeights,
  graphGainOf?: (target: GridIdx, cost: number) => number,
): Candidate[] {
  const out: Candidate[] = [];

  for (const f of frontiers) {
    const reach = nav.nearestReachable(f.representative);
    if (!reach) continue; // unreachable through known-free space — skip, do not fail

    const [tx, ty] = grid.cellCenter(reach.cell.i, reach.cell.j);
    const heading = normalizeAngle(Math.atan2(f.centroid[1] - ty, f.centroid[0] - tx));
    const pose: Pose2 = { x: tx, y: ty, theta: heading };

    const gain = expectedInfoGain(grid, pose, sensing);
    const cost = reach.cost;
    const graphGain = graphGainOf ? graphGainOf(reach.cell, cost) : 0;
    const utility = weights.wI * gain + weights.wG * graphGain - weights.wC * cost;

    out.push({ frontier: f, target: reach.cell, pose, gain, cost, graphGain, utility });
  }

  out.sort((a, b) => b.utility - a.utility);
  return out;
}

/* -------------------------------------------------------------------------- */
/* Graph optimality criteria                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Assemble the pose graph's information matrix Ω = Σ Aᵀ Ω_ij A over the free
 * (non-gauge) nodes.
 *
 * These are exactly the Jacobians `PoseGraph.optimize` uses — ∂e/∂δ_i =
 * −Ad(Z⁻¹), ∂e/∂δ_j = I — so Ω here *is* the H matrix the back-end factors. The
 * gauge node's rows and columns are deleted rather than damped: with them in,
 * Ω is singular by construction and its determinant is zero no matter how good
 * the map is.
 */
export function poseGraphInformation(graph: PoseGraph): Mat {
  const free: number[] = [];
  const slot = new Map<number, number>();
  for (const node of graph.nodes) {
    if (node.fixed) continue;
    slot.set(node.id, free.length);
    free.push(node.id);
  }
  const dim = 3 * free.length;
  const h = zerosMat(dim, dim);
  if (dim === 0) return h;

  const identity: Mat = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const negate: Mat = [
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
  ];

  for (const e of graph.edges as PoseEdge[]) {
    const ai = matMul(negate, adjoint(inverse(e.z)));
    const blocks: [number, Mat][] = [];
    if (slot.has(e.i)) blocks.push([slot.get(e.i) as number, ai]);
    if (slot.has(e.j)) blocks.push([slot.get(e.j) as number, identity]);

    for (const [na, a] of blocks) {
      const atO = matMul(transpose(a), e.omega);
      for (const [nb, b] of blocks) {
        const hb = matMul(atO, b);
        for (let k = 0; k < 3; k++) {
          for (let l = 0; l < 3; l++) h[3 * na + k][3 * nb + l] += hb[k][l];
        }
      }
    }
  }
  return h;
}

/**
 * log det Ω, in nats, via the Cholesky factor: log det = 2 Σ log L_kk.
 *
 * On the Rust side this is one `faer` sparse Cholesky and the same diagonal
 * sum — for a planar pose graph that is roughly O(n^1.5), which is what makes
 * scoring a dozen candidate trajectories per decision affordable.
 */
export function graphLogDet(graph: PoseGraph): number {
  const h = poseGraphInformation(graph);
  if (h.length === 0) return 0;
  const l = cholesky(h);
  let logDet = 0;
  for (let k = 0; k < h.length; k++) logDet += 2 * Math.log(Math.max(l[k][k], 1e-300));
  return logDet;
}

/**
 * D-optimality in the form the active-SLAM literature settled on
 * (Carrillo et al. 2012; Placed et al. 2023):
 *
 *   Dopt(Ω) = exp( (1/n) log det Ω )
 *
 * The 1/n is not cosmetic. Raw log det grows with the number of nodes, so it
 * ranks a longer trajectory above a better-constrained one purely for being
 * longer; the geometric-mean form is the eigenvalue average that makes two
 * graphs of different sizes comparable at all.
 */
export function dOptimality(graph: PoseGraph): number {
  const dim = 3 * graph.nodes.filter((n) => !n.fixed).length;
  if (dim === 0) return 0;
  return Math.exp(graphLogDet(graph) / dim);
}

/** A-optimality: the trace of the covariance, via Ω's eigen-reciprocals. Kept for the exercise. */
export function aOptimalityProxy(graph: PoseGraph): number {
  const h = poseGraphInformation(graph);
  if (h.length === 0) return 0;
  // tr(Ω) is a cheap stand-in the chapter uses only to contrast with D-opt.
  let t = 0;
  for (let k = 0; k < h.length; k++) t += h[k][k];
  return t / h.length;
}

/* -------------------------------------------------------------------------- */
/* Stopping                                                                    */
/* -------------------------------------------------------------------------- */

export interface StopRule {
  /** Stop when max_a I(a)/C(a) drops below this, in bits per metre. */
  gainRate: number;
  /** …or when the best absolute gain is below this many bits. */
  minGain: number;
}

export const DEFAULT_STOP: StopRule = { gainRate: 0.35, minGain: 1.0 };

export type StopReason = 'no-frontiers' | 'gain-rate' | 'running';

/**
 * The survey-honest stopping test. There is no theorem here: an absolute
 * entropy threshold is map-size dependent, and a plateau detector needs a
 * window length nobody can justify. A gain-per-metre floor at least has units
 * the task can argue about — "a bit is worth a metre" is a statement about the
 * mission, not about information theory.
 */
export function shouldStop(candidates: Candidate[], rule: StopRule = DEFAULT_STOP): StopReason {
  if (candidates.length === 0) return 'no-frontiers';
  let bestRate = 0;
  let bestGain = 0;
  for (const c of candidates) {
    bestRate = Math.max(bestRate, c.gain / Math.max(c.cost, 1e-6));
    bestGain = Math.max(bestGain, c.gain);
  }
  if (bestRate < rule.gainRate || bestGain < rule.minGain) return 'gain-rate';
  return 'running';
}
