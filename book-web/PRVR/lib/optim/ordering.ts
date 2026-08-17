/**
 * Sparsity, elimination, and ordering — Chapter 15, derivations D4 and D5.
 *
 * Ω mirrors the factor graph: the block Ω_[j][k] is nonzero exactly when
 * variables j and k co-appear in some factor. Eliminating a variable (Schur
 * complement, Gaussian marginalization in information form, Thrun's
 * `EIF_reduce`) removes its node and *clique-connects its neighbors* — the
 * fill-in. How much fill-in you pay is decided entirely by the order in which
 * you eliminate, which is why ordering is an algorithm and not a detail.
 *
 * Everything here is **symbolic**: it manipulates which blocks exist, never
 * their values. That separation is the point — a reader can watch elimination
 * happen on the graph and on the matrix and see that they are one event.
 */

import { inv, matMul, matVec, transpose, zeros, zerosMat, type Mat, type Vec } from '../prob/linalg';
import {
  VAR_DIM,
  buildIndex,
  graphVariables,
  keyId,
  type BlockIndex,
  type FactorGraph,
  type System,
  type VarKey,
} from './factor-graph';

/* -------------------------------------------------------------------------- */
/* Block sparsity pattern                                                      */
/* -------------------------------------------------------------------------- */

export interface BlockPattern {
  keys: VarKey[];
  dims: number[];
  /** Neighbor slots of each variable, self excluded. */
  adjacency: Set<number>[];
}

/**
 * The pattern of Ω for a graph, in a chosen variable order.
 *
 * One pass over the factors: every pair of variables a factor touches gets an
 * edge. In SLAM that yields poses chained to their neighbors, poses starred to
 * the landmarks they saw, and **no landmark–landmark edges at all** — no factor
 * ever mentions two landmarks together.
 */
export function buildPattern(graph: FactorGraph, order: VarKey[]): BlockPattern {
  const slot = new Map<string, number>();
  order.forEach((k, i) => slot.set(keyId(k), i));
  const adjacency: Set<number>[] = order.map(() => new Set<number>());
  for (const f of graph.factors) {
    const slots = f.keys.map((k) => slot.get(keyId(k)) ?? -1).filter((s) => s >= 0);
    for (let a = 0; a < slots.length; a++) {
      for (let b = a + 1; b < slots.length; b++) {
        adjacency[slots[a]].add(slots[b]);
        adjacency[slots[b]].add(slots[a]);
      }
    }
  }
  return { keys: order, dims: order.map((k) => VAR_DIM[k.kind]), adjacency };
}

/** Scalar nonzeros in Ω: diagonal blocks plus both triangles of every edge. */
export function nnzOmega(pattern: BlockPattern): number {
  let n = 0;
  pattern.dims.forEach((d, i) => {
    n += d * d;
    for (const j of pattern.adjacency[i]) n += d * pattern.dims[j];
  });
  return n; // edges counted twice, once from each end — as stored in a full matrix
}

/* -------------------------------------------------------------------------- */
/* Orderings                                                                   */
/* -------------------------------------------------------------------------- */

export type OrderingName = 'chronological' | 'landmarks-first' | 'poses-first' | 'min-degree';

export const ORDERING_LABEL: Record<OrderingName, string> = {
  chronological: 'Chronological',
  'landmarks-first': 'Landmarks first (Schur)',
  'poses-first': 'Poses first',
  'min-degree': 'Minimum degree',
};

export const ORDERING_NOTE: Record<OrderingName, string> = {
  chronological:
    'Variables in the order the robot met them. Each pose only ever cliques with its neighbors and its landmarks, so the factor is banded.',
  'landmarks-first':
    'Eliminate the whole map, then solve for the trajectory. This is exactly EIF_reduce (Thrun, Table 11.3) — and it leaves a pose graph behind.',
  'poses-first':
    'Eliminate the trajectory first. Every landmark seen from many poses ends up joined to every other landmark those poses saw: fill-in on purpose.',
  'min-degree':
    'Greedily eliminate whichever variable currently has the fewest neighbors. Cheap, myopic, and hard to beat — the ancestor of COLAMD/AMD.',
};

/** Variables in first-observed order — poses and landmarks interleaved. */
export function chronologicalOrder(graph: FactorGraph): VarKey[] {
  return graphVariables(graph);
}

function partitioned(graph: FactorGraph, first: 'landmark' | 'pose'): VarKey[] {
  const vars = graphVariables(graph);
  const head = vars.filter((k) => k.kind === first).sort((a, b) => a.id - b.id);
  const tail = vars.filter((k) => k.kind !== first).sort((a, b) => a.id - b.id);
  return [...head, ...tail];
}

export const landmarksFirstOrder = (g: FactorGraph): VarKey[] => partitioned(g, 'landmark');
export const posesFirstOrder = (g: FactorGraph): VarKey[] => partitioned(g, 'pose');

/**
 * Greedy minimum-degree ordering, simulating fill-in as it goes.
 *
 * Optimal ordering is NP-hard; this heuristic is what every production solver
 * ships a refined version of (AMD, COLAMD, METIS nested dissection).
 */
export function minDegreeOrder(graph: FactorGraph): VarKey[] {
  const base = graphVariables(graph);
  const pattern = buildPattern(graph, base);
  const adjacency = pattern.adjacency.map((s) => new Set(s));
  const alive = new Set(base.map((_, i) => i));
  const out: VarKey[] = [];

  while (alive.size > 0) {
    let best = -1;
    let bestDegree = Number.POSITIVE_INFINITY;
    for (const i of alive) {
      const deg = adjacency[i].size;
      if (deg < bestDegree) {
        bestDegree = deg;
        best = i;
      }
    }
    out.push(base[best]);
    const nbrs = [...adjacency[best]];
    for (let a = 0; a < nbrs.length; a++) {
      for (let b = a + 1; b < nbrs.length; b++) {
        adjacency[nbrs[a]].add(nbrs[b]);
        adjacency[nbrs[b]].add(nbrs[a]);
      }
    }
    for (const j of nbrs) adjacency[j].delete(best);
    adjacency[best].clear();
    alive.delete(best);
  }
  return out;
}

export function orderVariables(graph: FactorGraph, name: OrderingName): VarKey[] {
  switch (name) {
    case 'landmarks-first':
      return landmarksFirstOrder(graph);
    case 'poses-first':
      return posesFirstOrder(graph);
    case 'min-degree':
      return minDegreeOrder(graph);
    default:
      return chronologicalOrder(graph);
  }
}

/* -------------------------------------------------------------------------- */
/* Symbolic elimination                                                        */
/* -------------------------------------------------------------------------- */

export interface EliminationEvent {
  /** Slot eliminated at this step, in the pattern's own indexing. */
  slot: number;
  key: VarKey;
  /** Neighbors still alive when this variable was eliminated. */
  neighbors: number[];
  /** Edges that did not exist before and do now: the fill-in this step created. */
  fill: [number, number][];
  /** Scalar entries this variable contributes to L (its column, diagonal included). */
  columnNnz: number;
}

export interface EliminationResult {
  events: EliminationEvent[];
  /** Every fill edge created, across all steps. */
  fill: [number, number][];
  /** Scalar nonzeros of the Cholesky factor L, diagonal included. */
  nnzL: number;
  /** Scalar nonzeros of the lower triangle of Ω, for the fill-in ratio. */
  nnzLowerOmega: number;
  /** Rough flop count of the numeric factorization this ordering implies. */
  flops: number;
}

/**
 * Run the elimination game on the block pattern, in its stored order.
 *
 * Each step: take the next variable, join all of its surviving neighbors into a
 * clique, record which of those joins are new (fill-in), and delete it. This is
 * symbolic Cholesky — and, read on the graph instead of the matrix, it is
 * exactly `EIF_reduce` applied one variable at a time.
 */
export function symbolicElimination(pattern: BlockPattern): EliminationResult {
  const n = pattern.keys.length;
  const adjacency = pattern.adjacency.map((s) => new Set(s));
  const original = pattern.adjacency.map((s) => new Set(s));
  const eliminated = new Array<boolean>(n).fill(false);
  const events: EliminationEvent[] = [];
  const fill: [number, number][] = [];
  let nnzL = 0;
  let flops = 0;

  for (let i = 0; i < n; i++) {
    const nbrs = [...adjacency[i]].filter((j) => !eliminated[j]).sort((a, b) => a - b);
    const stepFill: [number, number][] = [];
    for (let a = 0; a < nbrs.length; a++) {
      for (let b = a + 1; b < nbrs.length; b++) {
        const [p, q] = [nbrs[a], nbrs[b]];
        if (!adjacency[p].has(q)) {
          adjacency[p].add(q);
          adjacency[q].add(p);
          const edge: [number, number] = [p, q];
          stepFill.push(edge);
          fill.push(edge);
        }
      }
    }
    const d = pattern.dims[i];
    const below = nbrs.reduce((s, j) => s + pattern.dims[j], 0);
    const columnNnz = (d * (d + 1)) / 2 + below * d;
    nnzL += columnNnz;
    // Dense-block Cholesky cost of this column: factor the diagonal, solve the
    // panel, and update the trailing clique.
    flops += (d * d * d) / 3 + d * d * below + d * below * below;

    events.push({ slot: i, key: pattern.keys[i], neighbors: nbrs, fill: stepFill, columnNnz });
    eliminated[i] = true;
  }

  let nnzLowerOmega = 0;
  pattern.dims.forEach((d, i) => {
    nnzLowerOmega += (d * (d + 1)) / 2;
    for (const j of original[i]) if (j < i) nnzLowerOmega += d * pattern.dims[j];
  });

  return { events, fill, nnzL, nnzLowerOmega, flops };
}

/**
 * The block pattern after replaying the first `steps` eliminations — what the
 * spy plot shows while the reader scrubs.
 */
export function patternAfter(pattern: BlockPattern, steps: number): {
  adjacency: Set<number>[];
  eliminated: boolean[];
  fresh: Set<string>;
} {
  const adjacency = pattern.adjacency.map((s) => new Set(s));
  const eliminated = new Array<boolean>(pattern.keys.length).fill(false);
  const fresh = new Set<string>();
  const result = symbolicElimination(pattern);
  for (let s = 0; s < Math.min(steps, result.events.length); s++) {
    const ev = result.events[s];
    for (const [p, q] of ev.fill) {
      adjacency[p].add(q);
      adjacency[q].add(p);
      if (s === steps - 1) {
        fresh.add(`${p},${q}`);
        fresh.add(`${q},${p}`);
      }
    }
    eliminated[ev.slot] = true;
  }
  return { adjacency, eliminated, fresh };
}

/* -------------------------------------------------------------------------- */
/* Schur complement — `EIF_reduce`, reborn                                     */
/* -------------------------------------------------------------------------- */

export interface ReducedSystem {
  Omega: Mat;
  b: Vec;
  index: BlockIndex;
  /** Blocks removed, in the order they were eliminated. */
  eliminated: VarKey[];
}

/**
 * `schur_marginalize`: eliminate the given variables exactly.
 *
 *     Ω̃ = Ω_kk − Ω_km Ω_mm⁻¹ Ω_mk,     b̃ = b_k − Ω_km Ω_mm⁻¹ b_m
 *
 * The reduced system is the *same posterior* over the remaining variables, not
 * an approximation — which is why Thrun's Table 11.3 can throw the map away,
 * solve for the trajectory, and put the map back afterwards. What it is not is
 * free: Ω̃ is denser than Ω was, by exactly the fill-in edges the graph reading
 * predicts.
 */
export function schurMarginalize(
  system: System,
  index: BlockIndex,
  victims: VarKey[],
): ReducedSystem {
  const victimIds = new Set(victims.map(keyId));
  const keepKeys = index.order.filter((k) => !victimIds.has(keyId(k)));
  const dropKeys = index.order.filter((k) => victimIds.has(keyId(k)));

  const rows = (keys: VarKey[]): number[] => {
    const out: number[] = [];
    for (const k of keys) {
      const s = index.slot.get(keyId(k));
      if (s === undefined) continue;
      const o = index.offsets[s];
      for (let i = 0; i < index.dims[s]; i++) out.push(o + i);
    }
    return out;
  };

  const K = rows(keepKeys);
  const M = rows(dropKeys);
  const sub = (r: number[], c: number[]): Mat => r.map((i) => c.map((j) => system.Omega[i][j]));
  const pick = (r: number[]): Vec => r.map((i) => system.b[i]);

  const Okk = sub(K, K);
  const Okm = sub(K, M);
  const Omm = sub(M, M);
  const bk = pick(K);
  const bm = pick(M);

  if (M.length === 0) {
    return { Omega: Okk, b: bk, index: buildIndex(keepKeys), eliminated: [] };
  }

  const OmmInv = inv(Omm.map((row, i) => row.map((x, j) => (i === j ? x + 1e-12 : x))));
  const OkmOmmInv = matMul(Okm, OmmInv);
  const correction = matMul(OkmOmmInv, transpose(Okm));
  const Omega = Okk.map((row, i) => row.map((x, j) => x - correction[i][j]));
  const bCorr = matVec(OkmOmmInv, bm);
  const b = bk.map((x, i) => x - bCorr[i]);

  return { Omega, b, index: buildIndex(keepKeys), eliminated: dropKeys };
}

/** A zero-filled dense system of the given size — used by tests and demos. */
export const emptySystem = (n: number): { Omega: Mat; b: Vec } => ({
  Omega: zerosMat(n, n),
  b: zeros(n),
});
