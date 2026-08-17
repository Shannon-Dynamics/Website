/**
 * Active localization: choosing motions that sharpen the *pose* belief —
 * Chapter 24, following Fox, Burgard and Thrun (1998).
 *
 * The rule is one line of decision theory:
 *
 *   a* = argmax_a [ H(bel) − E_{z|a}[ H(bel′) ] ] − w_C C(a)
 *
 * and it is a depth-1 POMDP backup — Chapter 22's machinery, truncated at one
 * step because the exact value function is intractable and one step is already
 * enough to make a robot take the informative route.
 *
 * With a discrete belief and a discrete measurement, no sampling is needed: the
 * expectation over z is a finite sum, and everything below is exact. That is
 * why the chapter's numbers can be checked by hand.
 *
 * Rust counterpart: `crates/ch24_explore/src/active_loc.rs`.
 */

import { discreteEntropy } from '../prob/gaussian';

/** A belief over a 1-D grid of states — Chapter 5's histogram filter, as data. */
export interface DiscreteBelief {
  centers: number[];
  p: number[];
}

/** One possible reading, with its likelihood field p(z | x) over the state grid. */
export interface Outcome {
  label: string;
  likelihood: (x: number) => number;
}

export interface ActionCandidate {
  label: string;
  /** Commanded displacement, metres. Signed: negative is a detour backwards. */
  delta: number;
}

export interface OutcomeScore {
  label: string;
  /** p(z | a) = Σ_x b̄(x) p(z | x) — how likely this reading is, before acting. */
  prob: number;
  /** H(bel′) in bits, had this reading arrived. */
  entropy: number;
}

export interface ActionWeights {
  /** Bits of expected entropy reduction are worth this much. */
  wI: number;
  /** One unit of cost is worth this many bits — the exchange rate, made explicit. */
  wC: number;
  /**
   * C(a). Defaults to |δ|, the metres driven. A *navigating* robot should
   * instead pass the remaining distance to its goal: a pure information
   * objective never arrives anywhere, because standing still and looking is
   * always the cheapest way to stay certain.
   */
  cost?: (delta: number) => number;
  /**
   * The task penalty an action leaves behind — for a navigating robot, the
   * distance still to drive after taking it. Without this term the utility has
   * no reason to ever reach the goal.
   */
  taskCost?: (delta: number) => number;
  /** Exchange rate for `taskCost`, in bits per task unit. */
  wTask?: number;
}

export interface ActionScore {
  label: string;
  delta: number;
  /** H(bel), bits — the same for every candidate, but printed for the arithmetic. */
  priorEntropy: number;
  /** H(b̄), bits — after motion, before sensing. Motion never lowers it. */
  predictedEntropy: number;
  /** E_z[H(bel′)], bits. */
  expectedPosteriorEntropy: number;
  /** I(x ; z | a) = H(b̄) − E_z[H(bel′)] ≥ 0. Always. The theorem, computed. */
  mutualInfo: number;
  /** H(bel) − E_z[H(bel′)]: what the action is worth *net of the motion it costs*. */
  netGain: number;
  /** C(a), metres. */
  cost: number;
  /** The task penalty this action leaves behind, in whatever units the task uses. */
  taskCost: number;
  utility: number;
  outcomes: OutcomeScore[];
}

const normalized = (p: number[]): number[] => {
  let s = 0;
  for (const v of p) s += v;
  if (!(s > 0)) return p.map(() => 1 / p.length);
  return p.map((v) => v / s);
};

/**
 * `active_localize` — score every candidate motion by expected entropy
 * reduction, exactly.
 *
 * For each candidate the belief is pushed through the motion model (which
 * spreads it), then each possible reading is applied in turn (which sharpens
 * it), and the resulting entropies are averaged with the reading's own
 * probability. Two numbers come back per candidate and the difference between
 * them is the whole lesson:
 *
 *   `mutualInfo` — the measurement's value, provably ≥ 0.
 *   `netGain`    — the *action's* value, which goes negative when the motion
 *                  smears the belief faster than the scan can sharpen it.
 */
export function scoreActions(
  belief: DiscreteBelief,
  actions: ActionCandidate[],
  predict: (p: number[], delta: number) => number[],
  outcomes: Outcome[],
  weights: ActionWeights = { wI: 1, wC: 0.05 },
): ActionScore[] {
  const prior = normalized(belief.p);
  const priorEntropy = discreteEntropy(prior);

  return actions.map((a) => {
    const predicted = normalized(predict(prior, a.delta));
    const predictedEntropy = discreteEntropy(predicted);

    const rows: OutcomeScore[] = [];
    let expected = 0;

    for (const z of outcomes) {
      // p(z | a) = Σ_x b̄(x) p(z | x) — Chapter 5's evidence term, reused as a
      // *forecast* rather than as a normaliser after the fact.
      let pz = 0;
      const post = new Array<number>(predicted.length);
      for (let i = 0; i < predicted.length; i++) {
        const l = Math.max(z.likelihood(belief.centers[i]), 0);
        post[i] = predicted[i] * l;
        pz += post[i];
      }
      const h = pz > 0 ? discreteEntropy(post) : predictedEntropy;
      rows.push({ label: z.label, prob: pz, entropy: h });
      expected += pz * h;
    }

    // p(z | a) sums to one only if the likelihood is a proper distribution over
    // the outcome set; renormalising keeps a widget's ad-hoc sensor honest.
    const total = rows.reduce((s, r) => s + r.prob, 0);
    const expectedPosteriorEntropy = total > 0 ? expected / total : predictedEntropy;
    const mutualInfo = Math.max(0, predictedEntropy - expectedPosteriorEntropy);
    const netGain = priorEntropy - expectedPosteriorEntropy;
    const cost = Math.abs(a.delta);
    const taskCost = weights.taskCost ? weights.taskCost(a.delta) : 0;

    return {
      label: a.label,
      delta: a.delta,
      priorEntropy,
      predictedEntropy,
      expectedPosteriorEntropy,
      mutualInfo,
      netGain,
      cost,
      taskCost,
      utility:
        weights.wI * netGain - weights.wC * cost - (weights.wTask ?? 0) * taskCost,
      outcomes: rows,
    };
  });
}

/** argmax of the utility column. */
export function bestAction(scores: ActionScore[]): number {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i].utility > scores[best].utility) best = i;
  }
  return best;
}

/**
 * The baseline policy the chapter argues against: go where the goal is, and
 * trust the filter to sort itself out. It never reads the belief's *shape* —
 * only its mean — which is exactly the mistake.
 */
export function greedyGoalAction(
  scores: ActionScore[],
  meanState: number,
  goal: number,
): number {
  let best = 0;
  let bestErr = Infinity;
  for (let i = 0; i < scores.length; i++) {
    const err = Math.abs(goal - (meanState + scores[i].delta));
    if (err < bestErr) {
      bestErr = err;
      best = i;
    }
  }
  return best;
}
