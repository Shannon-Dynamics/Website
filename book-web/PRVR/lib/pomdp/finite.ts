/**
 * Finite POMDPs: belief-space dynamics, exact α-vector value iteration, and the
 * two approximations Chapter 22 measures it against (QMDP and PBVI).
 *
 * This is the TypeScript twin of `crates/ch22_pomdp/src/{model,exact,qmdp,pbvi}.rs`.
 * Everything the chapter's widgets draw — the envelope, the pruning counts, the
 * action thresholds, the tournament returns — comes out of these functions, so a
 * reader who redoes the algebra by hand gets the number on screen.
 *
 * Conventions (they differ between papers, so we fix them once):
 *   t[u][x][x']  = p(x' | x, u)          rows sum to 1
 *   o[u][x'][z]  = p(z  | x', u)         rows sum to 1   (observation follows the transition)
 *   r[x][u]      = r(x, u)               reward collected on taking u in x
 *   b[x]         = bel(x)                the Chapter 5 belief, now a planning state
 */

export interface FinitePomdp {
  name: string;
  states: string[];
  actions: string[];
  observations: string[];
  /** t[u][x][x'] = p(x' | x, u) */
  t: number[][][];
  /** o[u][x'][z] = p(z | x', u) */
  o: number[][][];
  /** r[x][u] */
  r: number[][];
  gamma: number;
}

export type Belief = number[];

/** A linear piece of the value function, tagged with the action it commits to. */
export interface AlphaVec {
  v: number[];
  action: number;
}

/** One horizon of exact value iteration, with the counts w22.3 displays. */
export interface ViStage {
  horizon: number;
  /** |U| · |Γ_{t-1}|^{|Z|} — the cross-sum count *before* any pruning. */
  raw: number;
  /** Survivors after dominance + envelope pruning. */
  kept: number;
  gamma: AlphaVec[];
  /** True when `raw` exceeded the enumeration cap and nothing was materialized. */
  truncated: boolean;
}

const TOL = 1e-9;

/* -------------------------------------------------------------------------- */
/* Belief-space dynamics — the Bayes filter of Chapter 5, in matrix form        */
/* -------------------------------------------------------------------------- */

/** b̄(x') = Σ_x p(x' | x, u) b(x). Prediction: the step that loses information. */
export function beliefPredict(m: FinitePomdp, b: Belief, u: number): Belief {
  const S = m.states.length;
  const out = new Array<number>(S).fill(0);
  for (let x = 0; x < S; x++) {
    const bx = b[x];
    if (bx === 0) continue;
    const row = m.t[u][x];
    for (let xp = 0; xp < S; xp++) out[xp] += bx * row[xp];
  }
  return out;
}

/**
 * τ(b, u, z) — the belief transition, and the evidence p(z | b, u) that comes
 * with it. The evidence is not a nuisance here: it is exactly the branching
 * probability of the belief MDP, and it is what cancels the normalizer in the
 * PWLC induction.
 */
export function beliefUpdate(
  m: FinitePomdp,
  b: Belief,
  u: number,
  z: number,
): { b: Belief; pz: number } {
  const bBar = beliefPredict(m, b, u);
  const un = bBar.map((v, xp) => v * m.o[u][xp][z]);
  const pz = un.reduce((a, c) => a + c, 0);
  if (pz <= 1e-300) return { b: bBar, pz: 0 };
  return { b: un.map((v) => v / pz), pz };
}

/** p(z | b, u) alone, without forming the posterior. */
export function obsLikelihood(m: FinitePomdp, b: Belief, u: number, z: number): number {
  const bBar = beliefPredict(m, b, u);
  let s = 0;
  for (let xp = 0; xp < bBar.length; xp++) s += bBar[xp] * m.o[u][xp][z];
  return s;
}

/** ρ(b, u) = Σ_x b(x) r(x, u) — the belief reward. Linear in b, which is the whole trick. */
export function beliefReward(m: FinitePomdp, b: Belief, u: number): number {
  let s = 0;
  for (let x = 0; x < b.length; x++) s += b[x] * m.r[x][u];
  return s;
}

export const dot = (a: readonly number[], b: readonly number[]): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/** V(b) = max_k ⟨α^(k), b⟩, plus which piece attains it and what it commands. */
export function valueAt(
  set: readonly AlphaVec[],
  b: Belief,
): { value: number; index: number; action: number } {
  let best = -Infinity;
  let idx = -1;
  for (let k = 0; k < set.length; k++) {
    const v = dot(set[k].v, b);
    if (v > best) {
      best = v;
      idx = k;
    }
  }
  return { value: best, index: idx, action: idx < 0 ? 0 : set[idx].action };
}

/** Shannon entropy of a discrete belief, in bits — the AMDP's compression coordinate. */
export function beliefEntropy(b: Belief): number {
  let h = 0;
  for (const p of b) if (p > 0) h -= p * Math.log2(p);
  return h;
}

/* -------------------------------------------------------------------------- */
/* Exact value iteration                                                       */
/* -------------------------------------------------------------------------- */

/**
 * g^{u,z}_k(x) = Σ_{x'} p(z | x', u) p(x' | x, u) α^(k)(x')
 *
 * The backup's inner object: what one α-vector is worth, seen from state x,
 * *through* the pair (u, z). Indexed [u][z][k][x].
 */
export function backprojections(m: FinitePomdp, set: readonly AlphaVec[]): number[][][][] {
  const S = m.states.length;
  const A = m.actions.length;
  const Z = m.observations.length;
  const g: number[][][][] = [];
  for (let u = 0; u < A; u++) {
    const gu: number[][][] = [];
    for (let z = 0; z < Z; z++) {
      const gz: number[][] = [];
      for (const a of set) {
        const row = new Array<number>(S).fill(0);
        for (let x = 0; x < S; x++) {
          let s = 0;
          for (let xp = 0; xp < S; xp++) s += m.t[u][x][xp] * m.o[u][xp][z] * a.v[xp];
          row[x] = s;
        }
        gz.push(row);
      }
      gu.push(gz);
    }
    g.push(gu);
  }
  return g;
}

/**
 * The exact backup: one α-vector per (action, choice of one α per observation).
 * Returns |U| · |Γ|^|Z| candidates — the combinatorial explosion, un-hidden.
 */
export function backupCandidates(m: FinitePomdp, set: readonly AlphaVec[]): AlphaVec[] {
  const S = m.states.length;
  const A = m.actions.length;
  const Z = m.observations.length;
  const K = set.length;
  const g = backprojections(m, set);
  const out: AlphaVec[] = [];

  // Odometer over Γ^|Z|: one α index per observation.
  const pick = new Array<number>(Z).fill(0);
  for (let u = 0; u < A; u++) {
    pick.fill(0);
    for (;;) {
      const v = new Array<number>(S);
      for (let x = 0; x < S; x++) {
        let acc = 0;
        for (let z = 0; z < Z; z++) acc += g[u][z][pick[z]][x];
        v[x] = m.r[x][u] + m.gamma * acc;
      }
      out.push({ v, action: u });

      let z = Z - 1;
      while (z >= 0 && ++pick[z] === K) pick[z--] = 0;
      if (z < 0) break;
    }
  }
  return out;
}

/**
 * Point-based backup (Pineau et al. 2003): keep one vector per belief point
 * instead of one per cross-sum. |B| vectors out, no matter how big Γ was.
 */
export function pointBackup(
  m: FinitePomdp,
  set: readonly AlphaVec[],
  B: readonly Belief[],
): AlphaVec[] {
  const S = m.states.length;
  const A = m.actions.length;
  const Z = m.observations.length;
  const g = backprojections(m, set);
  const out: AlphaVec[] = [];

  for (const b of B) {
    let bestVal = -Infinity;
    let bestVec: AlphaVec | null = null;
    for (let u = 0; u < A; u++) {
      const v = new Array<number>(S);
      for (let x = 0; x < S; x++) v[x] = m.r[x][u];
      for (let z = 0; z < Z; z++) {
        // argmax over Γ of ⟨g^{u,z}_k, b⟩ — the one place the belief point enters.
        let bv = -Infinity;
        let bk = 0;
        for (let k = 0; k < set.length; k++) {
          const val = dot(g[u][z][k], b);
          if (val > bv) {
            bv = val;
            bk = k;
          }
        }
        for (let x = 0; x < S; x++) v[x] += m.gamma * g[u][z][bk][x];
      }
      const val = dot(v, b);
      if (val > bestVal) {
        bestVal = val;
        bestVec = { v, action: u };
      }
    }
    if (bestVec) out.push(bestVec);
  }
  return dedupe(out);
}

/* -------------------------------------------------------------------------- */
/* Pruning                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Merge vectors that agree to `tol` in every component.
 *
 * Not cosmetic: as value iteration converges, the backup keeps re-deriving the
 * *same* linear piece through different observation branches, and without this
 * merge |Γ| grows without bound while the value function stands still. Sorting
 * lexicographically first makes near-duplicates adjacent, so the sweep is
 * O(n log n) rather than the O(n²) pairwise comparison.
 */
export function dedupe(set: readonly AlphaVec[], tol = 1e-6): AlphaVec[] {
  if (set.length <= 1) return [...set];
  const idx = set.map((_, i) => i);
  idx.sort((i, j) => {
    const a = set[i].v;
    const b = set[j].v;
    for (let k = 0; k < a.length; k++) {
      if (a[k] !== b[k]) return a[k] - b[k];
    }
    return set[i].action - set[j].action;
  });
  const out: AlphaVec[] = [];
  let last: AlphaVec | null = null;
  for (const i of idx) {
    const a = set[i];
    if (last && maxAbsDiff(a.v, last.v) <= tol) continue;
    out.push(a);
    last = a;
  }
  return out;
}

/** Drop every vector that some other vector beats at *every* state. Cheap, and exact. */
export function pruneDominated(set: readonly AlphaVec[], tol = 1e-9): AlphaVec[] {
  const keep = dedupe(set);
  const out: AlphaVec[] = [];
  for (let i = 0; i < keep.length; i++) {
    let dominated = false;
    for (let j = 0; j < keep.length && !dominated; j++) {
      if (i === j) continue;
      let all = true;
      for (let x = 0; x < keep[i].v.length; x++) {
        if (keep[j].v[x] < keep[i].v[x] - tol) {
          all = false;
          break;
        }
      }
      if (all) dominated = true;
    }
    if (!dominated) out.push(keep[i]);
  }
  return out.length > 0 ? out : keep.slice(0, 1);
}

export interface EnvelopeSegment {
  /** Belief interval [tStart, tEnd] on which this piece is the maximum, t = b(x₀). */
  tStart: number;
  tEnd: number;
  /** Index into the α-vector set. */
  index: number;
  action: number;
  slope: number;
  intercept: number;
}

/**
 * The exact upper envelope of a two-state α-vector set.
 *
 * With b = (t, 1−t), ⟨α, b⟩ = α₁ + (α₀ − α₁)·t: every α-vector is a *line* over
 * the belief segment, and V is their upper envelope. We build it with the convex
 * hull trick, which is exact — no sampling, no LP — and hand back the breakpoints,
 * because those breakpoints *are* the policy's decision thresholds.
 */
export function envelope2(set: readonly AlphaVec[], t0 = 0, t1 = 1): EnvelopeSegment[] {
  interface Line {
    m: number;
    c: number;
    index: number;
    action: number;
  }
  const lines: Line[] = set.map((a, index) => ({
    m: a.v[0] - a.v[1],
    c: a.v[1],
    index,
    action: a.action,
  }));
  lines.sort((a, b) => a.m - b.m || a.c - b.c);

  // One line per slope: among parallels only the highest can ever show.
  const uniq: Line[] = [];
  for (const l of lines) {
    const last = uniq[uniq.length - 1];
    if (last && Math.abs(last.m - l.m) < TOL) {
      if (l.c > last.c) uniq[uniq.length - 1] = l;
      continue;
    }
    uniq.push(l);
  }

  const cross = (a: Line, b: Line) => (b.c - a.c) / (a.m - b.m);
  const st: Line[] = [];
  for (const l of uniq) {
    for (;;) {
      if (st.length === 0) break;
      const top = st[st.length - 1];
      const xi = cross(top, l); // where l overtakes top
      if (st.length >= 2 && xi <= cross(st[st.length - 2], top) + TOL) {
        st.pop();
        continue;
      }
      if (xi <= t0 + TOL) {
        st.pop();
        continue;
      }
      break;
    }
    if (st.length > 0 && cross(st[st.length - 1], l) >= t1 - TOL) continue;
    st.push(l);
  }

  const out: EnvelopeSegment[] = [];
  for (let i = 0; i < st.length; i++) {
    const a = st[i].m > 0 || i > 0 ? (i === 0 ? t0 : cross(st[i - 1], st[i])) : t0;
    const b = i === st.length - 1 ? t1 : cross(st[i], st[i + 1]);
    const lo = Math.max(t0, Math.min(a, t1));
    const hi = Math.max(t0, Math.min(b, t1));
    if (hi - lo < 1e-7) continue;
    out.push({
      tStart: lo,
      tEnd: hi,
      index: st[i].index,
      action: st[i].action,
      slope: st[i].m,
      intercept: st[i].c,
    });
  }
  return out;
}

/**
 * Prune to the minimal set: dominance first (free), then a witness test.
 *
 * For two states the witness test is exact — the upper envelope of a set of
 * lines is computable in closed form. Beyond two states we fall back to a
 * witness *search* over the simplex corners plus sampled interior points, which
 * is what the book's Rust does too. The text says exactly what this misses
 * relative to the LP formulation of Thrun §16.2.4.
 */
export function prune(set: readonly AlphaVec[], nSamples = 4096, seed = 7): AlphaVec[] {
  if (set.length === 0) return [];
  // Two states: the envelope *is* the answer, and it costs O(n log n) — so we
  // run it first, before the quadratic dominance sweep would ever be reached.
  if (set[0].v.length === 2) {
    const merged = dedupe(set);
    const segs = envelope2(merged);
    const keep = [...new Set(segs.map((s) => s.index))].sort((a, b) => a - b).map((i) => merged[i]);
    return keep.length > 0 ? keep : [merged[0]];
  }

  const survivors = pruneDominated(set);
  if (survivors.length <= 1) return survivors;
  const S = survivors[0].v.length;

  const keep = new Set<number>();
  // Corners of the simplex: the cheapest witnesses, and always worth testing.
  for (let x = 0; x < S; x++) {
    const b = new Array<number>(S).fill(0);
    b[x] = 1;
    keep.add(argmaxIndex(survivors, b));
  }
  let s = seed >>> 0 || 1;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let n = 0; n < nSamples; n++) {
    const b = new Array<number>(S);
    let tot = 0;
    for (let x = 0; x < S; x++) {
      const e = -Math.log(1 - rand());
      b[x] = e;
      tot += e;
    }
    for (let x = 0; x < S; x++) b[x] /= tot;
    keep.add(argmaxIndex(survivors, b));
  }
  return [...keep].sort((a, b) => a - b).map((i) => survivors[i]);
}

function argmaxIndex(set: readonly AlphaVec[], b: Belief): number {
  let best = -Infinity;
  let idx = 0;
  for (let k = 0; k < set.length; k++) {
    const v = dot(set[k].v, b);
    if (v > best) {
      best = v;
      idx = k;
    }
  }
  return idx;
}

/* -------------------------------------------------------------------------- */
/* finite_world_POMDP — Thrun Table 16.1                                       */
/* -------------------------------------------------------------------------- */

export interface ExactViOptions {
  /** Turn pruning off to watch the explosion. */
  prune?: boolean;
  /** Stop materializing candidates past this many; `raw` is still reported. */
  cap?: number;
}

/**
 * Exact value iteration over the belief simplex.
 *
 * Stage 1 is the immediate-reward set {r(·, u)}_u; every later stage is a full
 * cross-sum backup followed (optionally) by pruning. The returned `raw` counts
 * are the *analytic* |U|·|Γ|^|Z| — they keep growing even after we stop being
 * able to enumerate them, which is the point.
 */
export function exactVi(m: FinitePomdp, horizon: number, opts: ExactViOptions = {}): ViStage[] {
  const { prune: doPrune = true, cap = 200000 } = opts;
  const A = m.actions.length;
  const Z = m.observations.length;

  let set: AlphaVec[] = [];
  for (let u = 0; u < A; u++) {
    set.push({ v: m.states.map((_, x) => m.r[x][u]), action: u });
  }
  const stage1 = doPrune ? prune(set) : set;
  const stages: ViStage[] = [
    { horizon: 1, raw: A, kept: stage1.length, gamma: stage1, truncated: false },
  ];
  set = stage1;

  let rawPrev = stages[0].kept;
  let dead = false;
  for (let h = 2; h <= horizon; h++) {
    const rawCount = A * Math.pow(doPrune ? set.length : rawPrev, Z);
    if (dead || rawCount > cap) {
      stages.push({ horizon: h, raw: rawCount, kept: 0, gamma: [], truncated: true });
      rawPrev = rawCount;
      dead = true;
      continue;
    }
    const cand = backupCandidates(m, set);
    // Without pruning we keep *everything*, duplicates included: the reported
    // count has to be the real one, or the explosion looks tamer than it is.
    const kept = doPrune ? prune(cand) : cand;
    stages.push({ horizon: h, raw: rawCount, kept: kept.length, gamma: kept, truncated: false });
    set = kept;
    rawPrev = rawCount;
  }
  return stages;
}

/**
 * Infinite-horizon solve: back up until the value function stops moving on a
 * dense sweep of the simplex. The Bellman operator is a γ-contraction, so this
 * terminates; the residual is reported so the caller can quote it honestly.
 */
export function solveInfinite(
  m: FinitePomdp,
  opts: { tol?: number; maxIters?: number } = {},
): { gamma: AlphaVec[]; iterations: number; residual: number } {
  const { tol = 1e-6, maxIters = 600 } = opts;
  const A = m.actions.length;
  let set: AlphaVec[] = prune(
    m.actions.map((_, u) => ({ v: m.states.map((_s, x) => m.r[x][u]), action: u })),
  );
  const probes = simplexProbes(m.states.length, 401);
  let prevVals = probes.map((b) => valueAt(set, b).value);
  let residual = Infinity;
  let it = 0;
  for (; it < maxIters; it++) {
    const next = prune(backupCandidates(m, set));
    const vals = probes.map((b) => valueAt(next, b).value);
    residual = 0;
    for (let i = 0; i < vals.length; i++) residual = Math.max(residual, Math.abs(vals[i] - prevVals[i]));
    set = next;
    prevVals = vals;
    if (residual < tol) {
      it += 1;
      break;
    }
  }
  void A;
  return { gamma: set, iterations: it, residual };
}

/** A sweep of belief points: the segment for two states, a lattice otherwise. */
export function simplexProbes(S: number, n: number): Belief[] {
  if (S === 2) {
    const out: Belief[] = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      out.push([t, 1 - t]);
    }
    return out;
  }
  const out: Belief[] = [];
  let s = 12345;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    const b = new Array<number>(S);
    let tot = 0;
    for (let x = 0; x < S; x++) {
      const e = -Math.log(1 - rand());
      b[x] = e;
      tot += e;
    }
    out.push(b.map((v) => v / tot));
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* QMDP — the baseline that prices information at zero                         */
/* -------------------------------------------------------------------------- */

/**
 * Value iteration on the *underlying MDP* — Chapter 21's recursion, run here on
 * the fully observable problem so QMDP has a Q* to average.
 */
export function mdpQ(m: FinitePomdp, tol = 1e-10, maxIters = 20000): number[][] {
  const S = m.states.length;
  const A = m.actions.length;
  let V = new Array<number>(S).fill(0);
  for (let it = 0; it < maxIters; it++) {
    const next = new Array<number>(S).fill(-Infinity);
    for (let x = 0; x < S; x++) {
      for (let u = 0; u < A; u++) {
        let q = m.r[x][u];
        for (let xp = 0; xp < S; xp++) q += m.gamma * m.t[u][x][xp] * V[xp];
        if (q > next[x]) next[x] = q;
      }
    }
    let d = 0;
    for (let x = 0; x < S; x++) d = Math.max(d, Math.abs(next[x] - V[x]));
    V = next;
    if (d < tol) break;
  }
  const Q: number[][] = [];
  for (let x = 0; x < S; x++) {
    const row: number[] = [];
    for (let u = 0; u < A; u++) {
      let q = m.r[x][u];
      for (let xp = 0; xp < S; xp++) q += m.gamma * m.t[u][x][xp] * V[xp];
      row.push(q);
    }
    Q.push(row);
  }
  return Q;
}

/**
 * Γ_QMDP = { Q*(·, u) }_u : one α-vector per action, no backup at all.
 *
 * Q_MDP(b, u) = Σ_x b(x) Q*(x, u) is what you get by assuming the fog lifts
 * completely after one step. The set has |U| vectors for every horizon, which is
 * both why it is free and why it can never represent the value of information.
 */
export function qmdpAlphas(m: FinitePomdp): AlphaVec[] {
  const Q = mdpQ(m);
  return m.actions.map((_, u) => ({ v: m.states.map((_s, x) => Q[x][u]), action: u }));
}

/* -------------------------------------------------------------------------- */
/* PBVI — Pineau, Gordon & Thrun 2003                                          */
/* -------------------------------------------------------------------------- */

export function pbvi(
  m: FinitePomdp,
  B: readonly Belief[],
  iters: number,
): { gamma: AlphaVec[]; perIteration: number[] } {
  let set: AlphaVec[] = m.actions.map((_, u) => ({
    v: m.states.map((_s, x) => m.r[x][u]),
    action: u,
  }));
  const perIteration: number[] = [];
  for (let i = 0; i < iters; i++) {
    set = pointBackup(m, set, B);
    perIteration.push(set.length);
  }
  return { gamma: set, perIteration };
}

/** Expand the belief set by one reachable-belief sweep (PBVI's B ← B ∪ successors). */
export function expandBeliefSet(m: FinitePomdp, B: readonly Belief[], tol = 1e-3): Belief[] {
  const out: Belief[] = B.map((b) => [...b]);
  for (const b of B) {
    for (let u = 0; u < m.actions.length; u++) {
      for (let z = 0; z < m.observations.length; z++) {
        const { b: bp, pz } = beliefUpdate(m, b, u, z);
        if (pz <= 1e-9) continue;
        if (!out.some((q) => maxAbsDiff(q, bp) < tol)) out.push(bp);
      }
    }
  }
  return out;
}

const maxAbsDiff = (a: readonly number[], b: readonly number[]) => {
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
};

/* -------------------------------------------------------------------------- */
/* Execution: running a policy honestly, with a belief and a hidden state       */
/* -------------------------------------------------------------------------- */

export interface EpisodeStep {
  x: number;
  u: number;
  z: number;
  r: number;
  belief: Belief;
}

export interface RandomSource {
  next(): number;
}

/** Sample an index from a probability row. */
export function sampleRow(row: readonly number[], rng: RandomSource): number {
  let r = rng.next();
  for (let i = 0; i < row.length; i++) {
    r -= row[i];
    if (r <= 0) return i;
  }
  return row.length - 1;
}

/**
 * Run a policy for `steps` steps: the world is in one true state, the agent only
 * ever touches its belief. Returns the discounted return and the trace.
 */
export function runEpisode(
  m: FinitePomdp,
  policy: (b: Belief) => number,
  rng: RandomSource,
  steps: number,
  b0: Belief,
  x0?: number,
): { discounted: number; total: number; trace: EpisodeStep[] } {
  let b = [...b0];
  let x = x0 ?? sampleRow(b0, rng);
  let discounted = 0;
  let total = 0;
  let g = 1;
  const trace: EpisodeStep[] = [];
  for (let k = 0; k < steps; k++) {
    const u = policy(b);
    const rew = m.r[x][u];
    discounted += g * rew;
    total += rew;
    g *= m.gamma;
    const xp = sampleRow(m.t[u][x], rng);
    const z = sampleRow(m.o[u][xp], rng);
    b = beliefUpdate(m, b, u, z).b;
    x = xp;
    trace.push({ x, u, z, r: rew, belief: b });
  }
  return { discounted, total, trace };
}

/** The greedy policy induced by an α-vector set. */
export const alphaPolicy =
  (set: readonly AlphaVec[]) =>
  (b: Belief): number =>
    valueAt(set, b).action;

/* -------------------------------------------------------------------------- */
/* The tiger: the chapter's numeric micro-example                              */
/* -------------------------------------------------------------------------- */

export const TIGER_LISTEN = 0;
export const TIGER_OPEN_LEFT = 1;
export const TIGER_OPEN_RIGHT = 2;

/**
 * Kaelbling, Littman & Cassandra (1998), with this book's discount.
 *
 * Two doors. Behind one, the charging dock (+10); behind the other, a tiger
 * (−100). Listening costs 1 and reports the correct door with probability
 * `accuracy`. Opening either door ends the round and the tiger is re-placed
 * uniformly, so the belief resets to (½, ½) and the game repeats forever.
 */
export function makeTiger(accuracy = 0.85, gamma = 0.95): FinitePomdp {
  const a = accuracy;
  const half = [0.5, 0.5];
  return {
    name: 'Tiger',
    states: ['tiger-left', 'tiger-right'],
    actions: ['listen', 'open-left', 'open-right'],
    observations: ['hear-left', 'hear-right'],
    t: [
      // listen leaves the tiger where it is
      [
        [1, 0],
        [0, 1],
      ],
      // opening a door resets the game
      [half.slice(), half.slice()],
      [half.slice(), half.slice()],
    ],
    o: [
      // listening: accurate with probability `a`
      [
        [a, 1 - a],
        [1 - a, a],
      ],
      // after opening, the growl carries nothing: the tiger has been re-placed
      [half.slice(), half.slice()],
      [half.slice(), half.slice()],
    ],
    r: [
      // r[tiger-left][listen | open-left | open-right]
      [-1, -100, 10],
      // r[tiger-right][...]
      [-1, 10, -100],
    ],
    gamma,
  };
}

export const TIGER = makeTiger();

/* -------------------------------------------------------------------------- */
/* Corridor commit: the tiger, wearing a robot                                 */
/* -------------------------------------------------------------------------- */

export const CORRIDOR_ADVANCE = 0;
export const CORRIDOR_COMMIT_A = 1;
export const CORRIDOR_COMMIT_B = 2;

/**
 * Rusty at a T-junction, MCL bimodal across two mirror-image corridors.
 *
 * States: (side ∈ {A, B}) × (junction | past-the-doorway), plus an absorbing
 * DONE. Advancing costs 1 and drives the robot past a doorway, where the LiDAR
 * either does or does not see through it — a landmark test that is right with
 * probability `q`. Committing to a corridor ends the episode: +50 if the robot
 * guessed its corridor right, −80 if it turned into the wrong one and has to be
 * rescued.
 *
 * The whole chapter in five states: the only way to earn the +50 reliably is to
 * spend rewards on an action that moves you *no closer to the goal*.
 */
export function makeCorridorCommit(q = 0.9, gamma = 0.95): FinitePomdp {
  const S = 5; // A-junction, A-past, B-junction, B-past, DONE
  const A0 = 0;
  const A1 = 1;
  const B0 = 2;
  const B1 = 3;
  const DONE = 4;

  const zeros = () => Array.from({ length: S }, () => new Array<number>(S).fill(0));
  const advance = zeros();
  advance[A0][A1] = 1;
  advance[A1][A1] = 1;
  advance[B0][B1] = 1;
  advance[B1][B1] = 1;
  advance[DONE][DONE] = 1;

  const commit = zeros();
  for (let x = 0; x < S; x++) commit[x][DONE] = 1;

  // p(see-through-doorway | x', u): only meaningful once past the doorway.
  const obsAdvance = [
    [0.5, 0.5], // A-junction — unreachable after an advance, kept for completeness
    [q, 1 - q], // A-past: the doorway is there
    [0.5, 0.5],
    [1 - q, q], // B-past: the mirrored corridor has no doorway here
    [1, 0],
  ];
  const obsCommit = Array.from({ length: S }, () => [1, 0]);

  return {
    name: 'Corridor commit',
    states: ['A · junction', 'A · past door', 'B · junction', 'B · past door', 'done'],
    actions: ['advance', 'commit-A', 'commit-B'],
    observations: ['doorway', 'wall'],
    t: [advance, commit, commit.map((r) => [...r])],
    o: [obsAdvance, obsCommit, obsCommit.map((r) => [...r])],
    r: [
      // r[x][advance, commit-A, commit-B]
      [-1, 50, -80],
      [-1, 50, -80],
      [-1, -80, 50],
      [-1, -80, 50],
      [0, 0, 0],
    ],
    gamma,
  };
}

/** The bimodal prior: Rusty is equally sure it is in either corridor. */
export const CORRIDOR_PRIOR: Belief = [0.5, 0, 0.5, 0, 0];
