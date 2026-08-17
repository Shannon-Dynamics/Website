/**
 * A log-odds quadtree — OctoMap (Hornung et al., 2013) with one dimension
 * removed so the pedagogy fits on a page.
 *
 * The estimator is unchanged from Chapter 13: every leaf runs the static binary
 * Bayes filter, `ℓ ← clamp(ℓ + inverse_sensor_model − ℓ₀, ℓ_min, ℓ_max)`. The
 * only new idea is that the *representation* is adaptive. Clamping bounds the
 * filter's state, so four saturated siblings carry no more information than
 * their parent does, and merging them is lossless with respect to the clamped
 * filter. Free space saturates almost everywhere, which is why the tree ends up
 * proportional to the surface rather than to the area.
 */

import type { Pose2 } from '../geom/se2';
import { bresenham, logOddsToProb, probToLogOdds } from './occgrid';

export interface QuadTreeOptions {
  /** World coordinates of the lower-left corner of the (square) root region. */
  origin: { x: number; y: number };
  /** Side length of the root region, metres. */
  size: number;
  /** Leaf resolution is size / 2^maxDepth. */
  maxDepth: number;
  /** Log-odds prior, and the evidence levels of the inverse sensor model. */
  l0?: number;
  lOcc?: number;
  lFree?: number;
  /**
   * The clamp bounds [ℓ_min, ℓ_max]. OctoMap's defaults are asymmetric —
   * p ∈ [0.12, 0.97] — and the asymmetry is deliberate: free space is allowed
   * to saturate after a handful of misses, which is precisely what makes whole
   * subtrees agree and collapse.
   */
  lMin?: number;
  lMax?: number;
  /** Obstacle thickness α: cells within α/2 of the return are called occupied. */
  alpha?: number;
}

interface QNode {
  /** Log odds. On an inner node this is the value its children agreed on. */
  l: number;
  /** Four children in Morton order: 0 = SW, 1 = SE, 2 = NW, 3 = NE. */
  kids: QNode[] | null;
}

export interface QuadLeaf {
  x: number;
  y: number;
  size: number;
  l: number;
}

/** 4-byte value + four 4-byte child indices, the layout a packed octree uses. */
export const NODE_BYTES = 20;

export class LogOddsQuadTree {
  readonly origin: { x: number; y: number };
  readonly size: number;
  readonly maxDepth: number;
  readonly resolution: number;
  readonly l0: number;
  readonly lOcc: number;
  readonly lFree: number;
  readonly lMin: number;
  readonly lMax: number;
  readonly alpha: number;
  private root: QNode;

  constructor(opts: QuadTreeOptions) {
    this.origin = { ...opts.origin };
    this.size = opts.size;
    this.maxDepth = opts.maxDepth;
    this.resolution = opts.size / 2 ** opts.maxDepth;
    this.l0 = opts.l0 ?? 0;
    // OctoMap's published defaults: p_hit = 0.7, p_miss = 0.4, clamped to
    // [0.12, 0.97].
    this.lOcc = opts.lOcc ?? probToLogOdds(0.7);
    this.lFree = opts.lFree ?? probToLogOdds(0.4);
    this.lMin = opts.lMin ?? probToLogOdds(0.12);
    this.lMax = opts.lMax ?? probToLogOdds(0.97);
    this.alpha = opts.alpha ?? 0.2;
    this.root = { l: this.l0, kids: null };
  }

  /** Cells per side at leaf resolution. */
  get cells(): number {
    return 2 ** this.maxDepth;
  }

  worldToCell(x: number, y: number): [number, number] {
    return [
      Math.floor((x - this.origin.x) / this.resolution),
      Math.floor((y - this.origin.y) / this.resolution),
    ];
  }

  /**
   * Add `delta` to the leaf containing cell (i, j), splitting on the way down.
   *
   * A split copies the parent's value into all four children — the tree is a
   * compressed representation of the same field, so refining it must not change
   * what the field says.
   */
  update(i: number, j: number, delta: number): void {
    const n = this.cells;
    if (i < 0 || j < 0 || i >= n || j >= n) return;
    let node = this.root;
    for (let depth = this.maxDepth - 1; depth >= 0; depth--) {
      if (!node.kids) {
        node.kids = [
          { l: node.l, kids: null },
          { l: node.l, kids: null },
          { l: node.l, kids: null },
          { l: node.l, kids: null },
        ];
      }
      const bit = 1 << depth;
      const q = ((j & bit) !== 0 ? 2 : 0) | ((i & bit) !== 0 ? 1 : 0);
      node = node.kids[q];
    }
    const next = node.l + delta - this.l0;
    node.l = Math.max(this.lMin, Math.min(this.lMax, next));
  }

  /** Descend to the leaf covering (x, y), optionally stopping early. */
  logOddsAt(x: number, y: number, maxDepth = this.maxDepth): number {
    const [i, j] = this.worldToCell(x, y);
    const n = this.cells;
    if (i < 0 || j < 0 || i >= n || j >= n) return this.l0;
    let node = this.root;
    for (let depth = this.maxDepth - 1; depth >= this.maxDepth - maxDepth; depth--) {
      if (!node.kids) return node.l;
      const bit = 1 << depth;
      const q = ((j & bit) !== 0 ? 2 : 0) | ((i & bit) !== 0 ? 1 : 0);
      node = node.kids[q];
    }
    return node.l;
  }

  occupancyAt(x: number, y: number, maxDepth = this.maxDepth): number {
    return logOddsToProb(this.logOddsAt(x, y, maxDepth));
  }

  /**
   * `octree_insert_scan` — Chapter 13's update, applied to leaves.
   *
   * Beams are rasterised with Bresenham at leaf resolution: every cell before
   * the return gets free evidence, the cell at the return gets occupied
   * evidence. A per-scan visited set keeps a cell from being counted once per
   * beam that happens to cross it.
   */
  insertScan(pose: Pose2, ranges: number[], angles: number[], maxRange: number): void {
    const [ri, rj] = this.worldToCell(pose.x, pose.y);
    const visited = new Set<number>();
    const n = this.cells;

    for (let b = 0; b < ranges.length; b++) {
      const z = Math.min(ranges[b], maxRange);
      const a = pose.theta + angles[b];
      const hit = ranges[b] < maxRange;
      const reach = hit ? z + this.alpha / 2 : z;
      const [ei, ej] = this.worldToCell(
        pose.x + reach * Math.cos(a),
        pose.y + reach * Math.sin(a),
      );
      const line = bresenham(ri, rj, ei, ej);

      for (let c = 0; c < line.length; c++) {
        const [i, j] = line[c];
        if (i < 0 || j < 0 || i >= n || j >= n) continue;
        const key = j * n + i;
        if (visited.has(key)) continue;

        // Distance of this cell's centre along the beam decides which of the
        // inverse model's two branches applies.
        const cx = this.origin.x + (i + 0.5) * this.resolution;
        const cy = this.origin.y + (j + 0.5) * this.resolution;
        const r = Math.hypot(cx - pose.x, cy - pose.y);
        let delta: number;
        if (hit && Math.abs(r - z) < this.alpha / 2) delta = this.lOcc;
        else if (r < z) delta = this.lFree;
        else continue; // beyond the return: the beam says nothing

        visited.add(key);
        this.update(i, j, delta);
      }
    }
  }

  /**
   * Merge any four siblings that agree, bottom-up, and report how many nodes
   * were freed.
   *
   * "Agree" means identical clamped log odds — which, once the clamp is active,
   * is the common case for both saturated free space and saturated walls. This
   * is lossless: the merged parent answers every query exactly as its children
   * would have.
   */
  prune(): number {
    let freed = 0;
    const visit = (node: QNode): void => {
      if (!node.kids) return;
      for (const kid of node.kids) visit(kid);
      const [a, b, c, d] = node.kids;
      if (a.kids || b.kids || c.kids || d.kids) return;
      const eq =
        Math.abs(a.l - b.l) < 1e-9 && Math.abs(a.l - c.l) < 1e-9 && Math.abs(a.l - d.l) < 1e-9;
      if (!eq) return;
      node.l = a.l;
      node.kids = null;
      freed += 4;
    };
    visit(this.root);
    return freed;
  }

  nodeCount(): number {
    const count = (node: QNode): number =>
      node.kids ? 1 + node.kids.reduce((s, k) => s + count(k), 0) : 1;
    return count(this.root);
  }

  leafCount(): number {
    const count = (node: QNode): number =>
      node.kids ? node.kids.reduce((s, k) => s + count(k), 0) : 1;
    return count(this.root);
  }

  memoryBytes(): number {
    return this.nodeCount() * NODE_BYTES;
  }

  /** Every leaf as a square, for drawing the tree's own subdivision. */
  leaves(): QuadLeaf[] {
    const out: QuadLeaf[] = [];
    const walk = (node: QNode, x: number, y: number, size: number): void => {
      if (!node.kids) {
        out.push({ x, y, size, l: node.l });
        return;
      }
      const h = size / 2;
      walk(node.kids[0], x, y, h);
      walk(node.kids[1], x + h, y, h);
      walk(node.kids[2], x, y + h, h);
      walk(node.kids[3], x + h, y + h, h);
    };
    walk(this.root, this.origin.x, this.origin.y, this.size);
    return out;
  }
}
