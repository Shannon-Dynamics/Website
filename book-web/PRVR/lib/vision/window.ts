/**
 * Sliding-window marginalization and its fill-in tax — Chapter 18, Derivation 5.
 *
 * A visual-inertial smoother cannot keep every state forever, so it keeps a
 * window and *marginalizes* what falls out the back:
 *
 *     Ω' = Ω_ββ − Ω_βα Ω_αα⁻¹ Ω_αβ
 *
 * That is exactly Chapter 15's Schur complement (and the draft's `EIF_reduce`),
 * and it is exact — no information is thrown away. What it is not is free:
 * every pair of survivors that touched the marginalized variable is now
 * *directly* coupled. The prior densifies. Deleting those couplings to get the
 * sparsity back is what SEIF did in 2000, and it is a modelling decision with a
 * measurable price, which this module computes: the marginal covariance of the
 * newest keyframe **shrinks** even though no measurement arrived.
 *
 * The block Schur complement here is dimensioned for 3-D (6-DOF keyframes, 3-DOF
 * landmarks); `lib/optim/ordering.ts` has the SE(2)-dimensioned twin that
 * Chapter 15's widgets use.
 */

import { inv, matMul, transpose, zerosMat, type Mat } from '../prob/linalg';
import { Rng } from '../prob/rng';

export type BlockKind = 'keyframe' | 'landmark';

export interface WindowBlock {
  id: string;
  kind: BlockKind;
  dim: number;
  label: string;
  /** Which keyframes observe this landmark (empty for keyframes). */
  seenBy: string[];
}

export interface WindowState {
  blocks: WindowBlock[];
  /** The window's information matrix, dense over whatever survives. */
  Omega: Mat;
  /** Block pairs "i,j" created by the most recent marginalization. */
  fresh: Set<string>;
  /** Block pairs deleted by the most recent sparsification. */
  dropped: Set<string>;
  slides: number;
  nextKf: number;
  nextLm: number;
  rng: Rng;
  /** Marginal σ of the newest keyframe, in the window's arbitrary units. */
  sigma: number;
  /**
   * The same σ, measured on the reduced window immediately after the last
   * marginalization: `exact` keeps every link the Schur complement produced,
   * `actual` is what survived sparsification. They are equal when nothing was
   * dropped, and `actual < exact` is overconfidence, quantified.
   */
  sigmaReducedExact: number;
  sigmaReducedActual: number;
  history: {
    slide: number;
    nnz: number;
    fill: number;
    dropped: number;
    sigmaExact: number;
    sigmaActual: number;
  }[];
}

export interface WindowOptions {
  keyframes?: number;
  landmarks?: number;
  /** How many keyframes observe each landmark. */
  coVisibility?: number;
  seed?: number;
}

const KF_DIM = 6;
const LM_DIM = 3;
/** Rows a single factor contributes: two per pixel observation, six per IMU edge. */
const PIXEL_ROWS = 2;
const IMU_ROWS = 6;
/** Below this Frobenius norm a block counts as structurally zero. */
const ZERO = 1e-9;

const pairKey = (i: number, j: number) => `${Math.min(i, j)},${Math.max(i, j)}`;

/* -------------------------------------------------------------------------- */
/* Construction                                                                */
/* -------------------------------------------------------------------------- */

function offsets(blocks: WindowBlock[]): { off: number[]; total: number } {
  const off: number[] = [];
  let total = 0;
  for (const b of blocks) {
    off.push(total);
    total += b.dim;
  }
  return { off, total };
}

/**
 * Add one factor's JᵀJ contribution to Ω. The Jacobian is synthetic — the
 * *pattern* is what the widget teaches, and the magnitudes only need to be
 * plausible — but the algebra is the real thing.
 */
function addFactor(
  Omega: Mat,
  off: number[],
  dims: number[],
  a: number,
  b: number,
  weight: number,
  rng: Rng,
  rows: number,
) {
  const cols = dims[a] + dims[b];
  const J = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => rng.normal(0, 1)),
  );
  const JtJ = matMul(transpose(J), J);
  const put = (bi: number, bj: number, ri: number, ci: number) => {
    for (let i = 0; i < dims[bi]; i++) {
      for (let j = 0; j < dims[bj]; j++) {
        Omega[off[bi] + i][off[bj] + j] += weight * JtJ[ri + i][ci + j];
      }
    }
  };
  put(a, a, 0, 0);
  put(b, b, dims[a], dims[a]);
  put(a, b, 0, dims[a]);
  put(b, a, dims[a], 0);
}

/** A fresh window: a keyframe chain, landmarks observed from several frames. */
export function buildWindow(opts: WindowOptions = {}): WindowState {
  const nKf = opts.keyframes ?? 6;
  const nLm = opts.landmarks ?? 9;
  const coVis = opts.coVisibility ?? 3;
  const rng = new Rng(opts.seed ?? 1807);

  const blocks: WindowBlock[] = [];
  for (let k = 0; k < nKf; k++) {
    blocks.push({ id: `x${k}`, kind: 'keyframe', dim: KF_DIM, label: `x${k}`, seenBy: [] });
  }
  // Features are tracked across *non-adjacent* keyframes — the stride is what
  // makes co-visibility interesting. Two frames three apart share a landmark but
  // no IMU edge, so eliminating that landmark is what creates genuine fill-in.
  const stride = 2;
  for (let j = 0; j < nLm; j++) {
    const first = j % Math.max(1, nKf - stride * (coVis - 1));
    const seenBy = Array.from({ length: coVis }, (_, i) => `x${first + stride * i}`).filter(
      (id) => Number(id.slice(1)) < nKf,
    );
    blocks.push({ id: `m${j}`, kind: 'landmark', dim: LM_DIM, label: `m${j}`, seenBy });
  }

  const { off, total } = offsets(blocks);
  const dims = blocks.map((b) => b.dim);
  const Omega = zerosMat(total, total);
  // A weak prior on everything: without it the window is not positive definite
  // until enough factors arrive, and no covariance could be reported at all.
  // It is deliberately small for landmarks — their information really is
  // borrowed from the frames that observed them.
  blocks.forEach((b, i) => {
    const prior = b.kind === 'keyframe' ? 0.05 : 0.002;
    for (let k = 0; k < b.dim; k++) Omega[off[i] + k][off[i] + k] += prior;
  });

  const index = new Map(blocks.map((b, i) => [b.id, i]));
  // An IMU factor constrains all six relative DOF; a pixel observation
  // contributes exactly two rows, which is why a landmark seen three times is
  // only just determined — and why its information is mostly *borrowed* from
  // the keyframes that saw it.
  for (let k = 0; k + 1 < nKf; k++) addFactor(Omega, off, dims, k, k + 1, 1, rng, IMU_ROWS);
  for (const b of blocks) {
    if (b.kind !== 'landmark') continue;
    const lm = index.get(b.id)!;
    for (const kf of b.seenBy) addFactor(Omega, off, dims, index.get(kf)!, lm, 0.6, rng, PIXEL_ROWS);
  }

  const state: WindowState = {
    blocks,
    Omega,
    fresh: new Set(),
    dropped: new Set(),
    slides: 0,
    nextKf: nKf,
    nextLm: nLm,
    rng,
    sigma: 0,
    sigmaReducedExact: 0,
    sigmaReducedActual: 0,
    history: [],
  };
  state.sigma = newestSigma(state);
  state.sigmaReducedExact = state.sigma;
  state.sigmaReducedActual = state.sigma;
  state.history.push({
    slide: 0,
    nnz: nnzBlocks(state),
    fill: 0,
    dropped: 0,
    sigmaExact: state.sigma,
    sigmaActual: state.sigma,
  });
  return state;
}

/* -------------------------------------------------------------------------- */
/* Marginalization                                                             */
/* -------------------------------------------------------------------------- */

/** Exact block Schur complement: eliminate `victims`, keep everyone else. */
export function schurBlocks(
  Omega: Mat,
  blocks: WindowBlock[],
  victims: Set<number>,
): { Omega: Mat; blocks: WindowBlock[] } {
  const { off } = offsets(blocks);
  const keepRows: number[] = [];
  const dropRows: number[] = [];
  blocks.forEach((b, i) => {
    const target = victims.has(i) ? dropRows : keepRows;
    for (let k = 0; k < b.dim; k++) target.push(off[i] + k);
  });
  const sub = (r: number[], c: number[]): Mat => r.map((i) => c.map((j) => Omega[i][j]));
  const keepBlocks = blocks.filter((_, i) => !victims.has(i));
  if (dropRows.length === 0) return { Omega: sub(keepRows, keepRows), blocks: keepBlocks };

  const Okk = sub(keepRows, keepRows);
  const Okm = sub(keepRows, dropRows);
  const Omm = sub(dropRows, dropRows);
  const OmmInv = inv(Omm.map((row, i) => row.map((x, j) => (i === j ? x + 1e-9 : x))));
  const corr = matMul(matMul(Okm, OmmInv), transpose(Okm));
  return {
    Omega: Okk.map((row, i) => row.map((x, j) => x - corr[i][j])),
    blocks: keepBlocks,
  };
}

export interface SlideOptions {
  /** SEIF-style sparsification: delete the weakest links after marginalizing. */
  sparsify?: boolean;
  /** Fraction of the strongest *new* link below which a link is cut; 1 cuts all. */
  threshold?: number;
  coVisibility?: number;
}

/**
 * One slide of the window: drop the oldest keyframe (and any landmark that only
 * it could see), then admit a new keyframe with fresh observations.
 */
export function slideWindow(state: WindowState, opts: SlideOptions = {}): WindowState {
  const sparsify = opts.sparsify ?? false;
  const threshold = opts.threshold ?? 1;
  const coVis = opts.coVisibility ?? 3;

  const oldest = state.blocks.findIndex((b) => b.kind === 'keyframe');
  if (oldest < 0) return state;
  const oldestId = state.blocks[oldest].id;

  // Landmarks *hosted* by the departing keyframe leave with it — their inverse
  // depth is parameterized in that frame, so keeping them would mean keeping the
  // frame. This is what VINS-Mono and OKVIS do, and it is where the interesting
  // fill-in comes from: eliminating a landmark clique-connects every keyframe
  // that saw it.
  const survivingKf = new Set(
    state.blocks.filter((b) => b.kind === 'keyframe' && b.id !== oldestId).map((b) => b.id),
  );
  const victims = new Set<number>([oldest]);
  state.blocks.forEach((b, i) => {
    if (b.kind !== 'landmark') return;
    const hosted = b.seenBy[0] === oldestId;
    const orphaned = !b.seenBy.some((k) => survivingKf.has(k));
    if (hosted || orphaned) victims.add(i);
  });

  // Which pairs were connected *before* — anything new is fill-in.
  const before = edgeSet(state.Omega, state.blocks);
  const beforeIds = new Map(state.blocks.map((b, i) => [i, b.id]));
  const beforePairs = new Set(
    [...before].map((k) => {
      const [i, j] = k.split(',').map(Number);
      return pairKey2(beforeIds.get(i)!, beforeIds.get(j)!);
    }),
  );

  const reduced = schurBlocks(state.Omega, state.blocks, victims);
  let blocks = reduced.blocks.map((b) => ({
    ...b,
    seenBy: b.seenBy.filter((k) => survivingKf.has(k)),
  }));
  let Omega = reduced.Omega;

  const afterEdges = edgeSet(Omega, blocks);
  const fresh = new Set<string>();
  for (const key of afterEdges) {
    const [i, j] = key.split(',').map(Number);
    if (i === j) continue;
    if (!beforePairs.has(pairKey2(blocks[i].id, blocks[j].id))) fresh.add(key);
  }

  // ---- optional SEIF-style sparsification --------------------------------
  // SEIF's move: refuse to let the graph densify. Every link the elimination
  // just created is a candidate for deletion, weakest first; `threshold` is the
  // fraction of the strongest new link below which a link is cut. At 1 the
  // window keeps exactly the sparsity it had, and the price is visible in σ.
  const dropped = new Set<string>();
  // Measured before any link is cut: the honest posterior of the reduced window.
  const sigmaReducedExact = anchorSigma({ Omega, blocks });
  if (sparsify) {
    const { off } = offsets(blocks);
    const norms = new Map<string, number>();
    let maxNorm = 0;
    for (const key of fresh) {
      const [i, j] = key.split(',').map(Number);
      const n = blockNorm(Omega, off, blocks, i, j);
      norms.set(key, n);
      maxNorm = Math.max(maxNorm, n);
    }
    for (const [key, n] of norms) {
      if (n > ZERO && n <= threshold * maxNorm) {
        const [i, j] = key.split(',').map(Number);
        zeroBlock(Omega, off, blocks, i, j);
        dropped.add(key);
      }
    }
  }

  const sigmaReducedActual = sparsify ? anchorSigma({ Omega, blocks }) : sigmaReducedExact;

  // ---- admit the new keyframe --------------------------------------------
  const kfId = `x${state.nextKf}`;
  const newestOld = [...blocks].reverse().find((b) => b.kind === 'keyframe');
  blocks = [...blocks, { id: kfId, kind: 'keyframe', dim: KF_DIM, label: kfId, seenBy: [] }];
  const newLandmarks: WindowBlock[] = [];
  // Admit as many fresh tracks as the slide retired, so the window is
  // stationary and every change in the sparsity pattern is fill-in, not growth.
  const newLmCount = Math.max(1, victims.size - 1);
  const kfIds = blocks.filter((b) => b.kind === 'keyframe').map((b) => b.id);
  // The new landmark is tracked back across the window with the same stride.
  const tail = kfIds
    .filter((_, i) => (kfIds.length - 1 - i) % 2 === 0)
    .slice(-coVis);
  for (let n = 0; n < newLmCount; n++) {
    const id = `m${state.nextLm + n}`;
    newLandmarks.push({ id, kind: 'landmark', dim: LM_DIM, label: id, seenBy: [...tail] });
  }
  blocks = [...blocks, ...newLandmarks];

  const { off, total } = offsets(blocks);
  const grown = zerosMat(total, total);
  const oldTotal = Omega.length;
  for (let i = 0; i < oldTotal; i++) {
    for (let j = 0; j < oldTotal; j++) grown[i][j] = Omega[i][j];
  }
  blocks.forEach((b, i) => {
    if (off[i] < oldTotal) return;
    const prior = b.kind === 'keyframe' ? 0.05 : 0.002;
    for (let k = 0; k < b.dim; k++) grown[off[i] + k][off[i] + k] += prior;
  });

  const dims = blocks.map((b) => b.dim);
  const index = new Map(blocks.map((b, i) => [b.id, i]));
  if (newestOld) {
    addFactor(grown, off, dims, index.get(newestOld.id)!, index.get(kfId)!, 1, state.rng, IMU_ROWS);
  }
  for (const lm of newLandmarks) {
    for (const kf of lm.seenBy) {
      addFactor(grown, off, dims, index.get(kf)!, index.get(lm.id)!, 0.6, state.rng, PIXEL_ROWS);
    }
  }

  const next: WindowState = {
    ...state,
    blocks,
    Omega: grown,
    fresh,
    dropped,
    slides: state.slides + 1,
    nextKf: state.nextKf + 1,
    nextLm: state.nextLm + newLandmarks.length,
    sigmaReducedExact,
    sigmaReducedActual,
  };
  next.sigma = newestSigma(next);
  next.history = [
    ...state.history,
    {
      slide: next.slides,
      nnz: nnzBlocks(next),
      fill: fresh.size,
      dropped: dropped.size,
      sigmaExact: sigmaReducedExact,
      sigmaActual: sigmaReducedActual,
    },
  ].slice(-40);
  return next;
}

/* -------------------------------------------------------------------------- */
/* Readouts                                                                    */
/* -------------------------------------------------------------------------- */

function blockNorm(Omega: Mat, off: number[], blocks: WindowBlock[], i: number, j: number): number {
  let s = 0;
  for (let a = 0; a < blocks[i].dim; a++) {
    for (let b = 0; b < blocks[j].dim; b++) {
      const x = Omega[off[i] + a][off[j] + b];
      s += x * x;
    }
  }
  return Math.sqrt(s);
}

function zeroBlock(Omega: Mat, off: number[], blocks: WindowBlock[], i: number, j: number) {
  for (let a = 0; a < blocks[i].dim; a++) {
    for (let b = 0; b < blocks[j].dim; b++) {
      Omega[off[i] + a][off[j] + b] = 0;
      Omega[off[j] + b][off[i] + a] = 0;
    }
  }
}

function edgeSet(Omega: Mat, blocks: WindowBlock[]): Set<string> {
  const { off } = offsets(blocks);
  const out = new Set<string>();
  for (let i = 0; i < blocks.length; i++) {
    for (let j = i; j < blocks.length; j++) {
      if (blockNorm(Omega, off, blocks, i, j) > ZERO) out.add(pairKey(i, j));
    }
  }
  return out;
}

const pairKey2 = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`);

/** Block-level |Ω| for the spy plot: one number per block pair. */
export function blockNormMatrix(state: WindowState): number[][] {
  const { off } = offsets(state.blocks);
  const n = state.blocks.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => blockNorm(state.Omega, off, state.blocks, i, j)),
  );
}

/** Nonzero blocks in the upper triangle, diagonal included. */
export function nnzBlocks(state: WindowState): number {
  return edgeSet(state.Omega, state.blocks).size;
}

/** Scalar nonzeros — what a solver actually stores. */
export function nnzScalars(state: WindowState): number {
  const { off } = offsets(state.blocks);
  let n = 0;
  for (let i = 0; i < state.blocks.length; i++) {
    for (let j = 0; j < state.blocks.length; j++) {
      if (blockNorm(state.Omega, off, state.blocks, i, j) > ZERO) {
        n += state.blocks[i].dim * state.blocks[j].dim;
      }
    }
  }
  return n;
}

/** The marginal σ of one block: √(tr Σ_bb / dim), with Σ = Ω⁻¹. */
export function blockSigma(state: { Omega: Mat; blocks: WindowBlock[] }, idx: number): number {
  if (idx < 0 || idx >= state.blocks.length) return 0;
  const { off } = offsets(state.blocks);
  let cov: Mat;
  try {
    cov = inv(state.Omega.map((row, i) => row.map((x, j) => (i === j ? x + 1e-9 : x))));
  } catch {
    return 0;
  }
  let tr = 0;
  const d = state.blocks[idx].dim;
  for (let a = 0; a < d; a++) tr += cov[off[idx] + a][off[idx] + a];
  return Math.sqrt(Math.max(tr, 0) / d);
}

/**
 * The consistency probe: the marginal σ of the **oldest surviving keyframe**,
 * the one that inherits the marginalization prior.
 *
 * Exact marginalization leaves it alone (check 14). Deleting the links the
 * elimination created makes it *smaller* — the estimator claims certainty it
 * never earned, which is the failure mode SEIF's sparsification traded for
 * sparsity in 2000.
 */
export function anchorSigma(state: { Omega: Mat; blocks: WindowBlock[] }): number {
  return blockSigma(state, indexOfKind(state.blocks, 'keyframe', 'first'));
}

/** The live readout: how uncertain the newest keyframe is. */
export function newestSigma(state: { Omega: Mat; blocks: WindowBlock[] }): number {
  return blockSigma(state, indexOfKind(state.blocks, 'keyframe', 'last'));
}

function indexOfKind(blocks: WindowBlock[], kind: BlockKind, which: 'first' | 'last'): number {
  if (which === 'first') {
    return blocks.findIndex((b) => b.kind === kind);
  }
  for (let i = blocks.length - 1; i >= 0; i--) if (blocks[i].kind === kind) return i;
  return -1;
}
