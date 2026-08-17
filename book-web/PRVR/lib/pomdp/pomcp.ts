/**
 * POMCP — Monte-Carlo tree search over histories (Silver & Veness, 2010).
 *
 * The planner never touches the model's transition matrices. It only needs a
 * *generative* model — the black box that, given a state and an action, hands
 * back a next state, an observation, and a reward. In this book that black box
 * is the Chapter 4 simulator, and the belief at every node is a bag of particles
 * exactly like the ones Chapter 8 resamples.
 *
 * The tree is stored in flat arrays rather than nested objects so that w22.4 can
 * lay it out and draw it without walking a linked structure every frame.
 */

export interface RandomSource {
  next(): number;
}

/** The black box POMCP plans with: (x, u) ↦ (x', z, r). */
export interface GenerativeModel<X> {
  nActions: number;
  gamma: number;
  step(x: X, u: number, rng: RandomSource): { x: X; z: number; r: number };
  /** Rollout policy; uniform-random unless the caller knows better. */
  rollout?(x: X, depth: number, rng: RandomSource): number;
  /** Optional absorbing check — the tiger never terminates, mazes do. */
  terminal?(x: X): boolean;
}

/** A history node: everything reached by one action–observation sequence. */
export interface BeliefNode<X> {
  id: number;
  depth: number;
  /** Visit count N(h). */
  n: number;
  /** Unweighted particles that arrived here — the belief, à la Chapter 8. */
  particles: X[];
  /** Child action node ids, indexed by action. Empty until the node is expanded. */
  children: number[];
  parentAction: number | null;
  /** The observation that led here from the parent action node. */
  obs: number | null;
}

/** An action node: the bandit arm N(h, u), Q(h, u). */
export interface ActionNode {
  id: number;
  action: number;
  depth: number;
  parent: number;
  /** Visit count N(h, u). */
  n: number;
  /** Running mean return Q(h, u). */
  q: number;
  /** Child belief node ids keyed by observation. */
  children: Map<number, number>;
}

export interface PomcpOptions {
  /** UCB1 exploration constant. Scale it to the reward magnitude, not to 1. */
  c: number;
  maxDepth: number;
  /** Cap on particles kept per node, purely to bound memory in the browser. */
  particleCap?: number;
}

export class Pomcp<X> {
  readonly beliefs: BeliefNode<X>[] = [];
  readonly actions: ActionNode[] = [];
  readonly opts: PomcpOptions;
  private model: GenerativeModel<X>;
  private rng: RandomSource;
  private root: number;
  /** Simulations run so far — the scrub axis of w22.4. */
  simulations = 0;

  constructor(model: GenerativeModel<X>, rootParticles: X[], rng: RandomSource, opts: PomcpOptions) {
    this.model = model;
    this.rng = rng;
    this.opts = { particleCap: 400, ...opts };
    this.beliefs.push({
      id: 0,
      depth: 0,
      n: 0,
      particles: [...rootParticles],
      children: [],
      parentAction: null,
      obs: null,
    });
    this.root = 0;
  }

  get rootNode(): BeliefNode<X> {
    return this.beliefs[this.root];
  }

  /** Q(root, u) for every action, or NaN for an arm never pulled. */
  rootQ(): number[] {
    const node = this.beliefs[this.root];
    const out = new Array<number>(this.model.nActions).fill(Number.NaN);
    for (const aid of node.children) {
      const a = this.actions[aid];
      out[a.action] = a.n > 0 ? a.q : Number.NaN;
    }
    return out;
  }

  rootVisits(): number[] {
    const node = this.beliefs[this.root];
    const out = new Array<number>(this.model.nActions).fill(0);
    for (const aid of node.children) out[this.actions[aid].action] = this.actions[aid].n;
    return out;
  }

  /** The recommended action: the most-visited arm, not the highest-valued one. */
  bestAction(): number {
    const visits = this.rootVisits();
    let best = 0;
    for (let u = 1; u < visits.length; u++) if (visits[u] > visits[best]) best = u;
    return best;
  }

  /** Run `n` more simulations from the root. Anytime: stop whenever you like. */
  search(n: number): void {
    const node = this.beliefs[this.root];
    for (let i = 0; i < n; i++) {
      // A simulation begins by *sampling a state from the belief* — the one
      // place the particle set enters the planner.
      const x = node.particles[Math.floor(this.rng.next() * node.particles.length) % node.particles.length];
      this.simulate(x, this.root, 0);
      this.simulations += 1;
    }
  }

  private expand(nodeId: number): void {
    const node = this.beliefs[nodeId];
    if (node.children.length > 0) return;
    for (let u = 0; u < this.model.nActions; u++) {
      const id = this.actions.length;
      this.actions.push({
        id,
        action: u,
        depth: node.depth,
        parent: nodeId,
        n: 0,
        q: 0,
        children: new Map(),
      });
      node.children.push(id);
    }
  }

  /** UCB1: exploit Q, but keep an eye on arms you have barely tried. */
  private selectAction(nodeId: number): number {
    const node = this.beliefs[nodeId];
    let best = -Infinity;
    let bestId = node.children[0];
    for (const aid of node.children) {
      const a = this.actions[aid];
      if (a.n === 0) return aid; // every arm gets one free pull
      const score = a.q + this.opts.c * Math.sqrt(Math.log(node.n + 1) / a.n);
      if (score > best) {
        best = score;
        bestId = aid;
      }
    }
    return bestId;
  }

  private rolloutValue(x0: X, depth: number): number {
    let x = x0;
    let g = 1;
    let total = 0;
    for (let d = depth; d < this.opts.maxDepth; d++) {
      const u = this.model.rollout
        ? this.model.rollout(x, d, this.rng)
        : Math.floor(this.rng.next() * this.model.nActions) % this.model.nActions;
      const s = this.model.step(x, u, this.rng);
      total += g * s.r;
      g *= this.model.gamma;
      x = s.x;
      if (this.model.terminal?.(x)) break;
    }
    return total;
  }

  private simulate(x: X, nodeId: number, depth: number): number {
    if (depth >= this.opts.maxDepth) return 0;
    const node = this.beliefs[nodeId];

    if (node.children.length === 0) {
      this.expand(nodeId);
      node.n += 1;
      // A leaf's value is a *rollout*, not zero: MCTS is only as good as the
      // default policy it falls back on.
      return this.rolloutValue(x, depth);
    }

    const aid = this.selectAction(nodeId);
    const a = this.actions[aid];
    const s = this.model.step(x, a.action, this.rng);

    let childId = a.children.get(s.z);
    if (childId === undefined) {
      childId = this.beliefs.length;
      this.beliefs.push({
        id: childId,
        depth: depth + 1,
        n: 0,
        particles: [],
        children: [],
        parentAction: aid,
        obs: s.z,
      });
      a.children.set(s.z, childId);
    }
    const child = this.beliefs[childId];
    // The child's belief is *built* out of the particles that reach it: this is
    // an unweighted particle filter running inside the search tree.
    if (child.particles.length < (this.opts.particleCap ?? 400)) child.particles.push(s.x);

    const future = this.model.terminal?.(s.x) ? 0 : this.simulate(s.x, childId, depth + 1);
    const ret = s.r + this.model.gamma * future;

    node.n += 1;
    a.n += 1;
    a.q += (ret - a.q) / a.n;
    return ret;
  }
}

/* -------------------------------------------------------------------------- */
/* Layout for the tree figure                                                  */
/* -------------------------------------------------------------------------- */

export interface LaidOutNode {
  kind: 'belief' | 'action';
  /** Which arm this is, for action nodes. */
  action: number | null;
  /** The observation that led here, for belief nodes below the root. */
  obs: number | null;
  /** Layer index: belief nodes on even rows, action nodes on odd ones. */
  row: number;
  x: number;
  /** Visit count: N(h) for beliefs, N(h, u) for arms. */
  n: number;
  /** Q(h, u); NaN on belief nodes. */
  q: number;
  /** Index into the returned array, or null at the root. */
  parent: number | null;
  particles: number;
  /** Mean of the node's particle set, when the state is a number (tiger). */
  particleMean: number;
}

/**
 * A tidy layered layout: action–observation depth on the vertical axis, leaves
 * spread evenly on the horizontal one, and every internal node centred over its
 * children. Arms with fewer than `minVisits` pulls are dropped, which is what
 * makes a 2000-simulation tree legible instead of a hairball.
 */
export function layoutTree<X>(
  t: Pomcp<X>,
  maxDepth = 2,
  minVisits = 1,
): { nodes: LaidOutNode[]; width: number } {
  const out: LaidOutNode[] = [];
  let leaf = 0;

  const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;

  const placeBelief = (id: number, depth: number, parent: number | null, obs: number | null): number => {
    const node = t.beliefs[id];
    const self = out.length;
    out.push({
      kind: 'belief',
      action: null,
      obs,
      row: depth * 2,
      x: 0,
      n: node.n,
      q: Number.NaN,
      parent,
      particles: node.particles.length,
      particleMean:
        node.particles.length > 0 && typeof node.particles[0] === 'number'
          ? mean(node.particles as unknown as number[])
          : Number.NaN,
    });
    const kidX: number[] = [];
    if (depth < maxDepth) {
      for (const aid of node.children) {
        if (t.actions[aid].n < minVisits) continue;
        kidX.push(out[placeAction(aid, depth, self)].x);
      }
    }
    out[self].x = kidX.length > 0 ? mean(kidX) : leaf++;
    return self;
  };

  const placeAction = (aid: number, depth: number, parent: number): number => {
    const a = t.actions[aid];
    const self = out.length;
    out.push({
      kind: 'action',
      action: a.action,
      obs: null,
      row: depth * 2 + 1,
      x: 0,
      n: a.n,
      q: a.q,
      parent,
      particles: 0,
      particleMean: Number.NaN,
    });
    const kidX: number[] = [];
    for (const [z, childId] of [...a.children.entries()].sort((p, q) => p[0] - q[0])) {
      if (t.beliefs[childId].n < minVisits) continue;
      kidX.push(out[placeBelief(childId, depth + 1, self, z)].x);
    }
    out[self].x = kidX.length > 0 ? mean(kidX) : leaf++;
    return self;
  };

  placeBelief(0, 0, null, null);
  return { nodes: out, width: Math.max(leaf, 1) };
}
