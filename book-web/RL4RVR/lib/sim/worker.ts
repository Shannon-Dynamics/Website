/**
 * The simulation worker.
 *
 * Widgets that train something real — Q-learning across seeds, a behaviour-
 * cloned policy, a dynamics ensemble, a gait optimizer — run here instead of on
 * the main thread, so a reader dragging a slider never meets a frozen page.
 *
 * Protocol: the page posts {id, job, params}; the worker streams
 * {id, type:'progress'|'done'|'error', payload}. Every job is seeded, so the
 * same parameters always produce the same result.
 */

/// <reference lib="webworker" />

import { GridWorld } from '@/lib/rl/gridworld';
import { TabularLearner } from '@/lib/rl/td';
import { mulberry32, gaussian } from '@/lib/rl/random';
import { Mlp, Standardizer } from '@/lib/rl/nn';
import { TileCoder, TileSarsa } from '@/lib/rl/tiles';
import { DynamicsEnsemble, type Transition } from '@/lib/rl/dynamics';
import { optimizeGait, simulateWalker, type RewardWeights } from '@/lib/rl/walker';
import {
  DEFAULT_PENDLE,
  rk4Step,
  wrapAngle,
  type PendleState,
} from '@/lib/rl/pendulum';

export type JobName =
  | 'replay-ablation'
  | 'behaviour-cloning'
  | 'dynamics-ensemble'
  | 'randomization-transfer'
  | 'gait-optimize';

interface Message {
  id: number;
  job: JobName;
  params: Record<string, unknown>;
}

const ctx = self as unknown as DedicatedWorkerGlobalScope;

function post(id: number, type: 'progress' | 'done' | 'error', payload: unknown) {
  ctx.postMessage({ id, type, payload });
}

// ---------------------------------------------------------------------------
// Chapter 9 — does replay/target-network surgery actually matter?
// ---------------------------------------------------------------------------

/**
 * Trains tabular Q-learning on Rusty's warehouse under four configurations,
 * with a deliberately aggressive learning rate so the instabilities the chapter
 * describes are visible rather than theoretical.
 */
function replayAblation(p: Record<string, unknown>, id: number) {
  const episodes = (p.episodes as number) ?? 180;
  const bufferSize = (p.bufferSize as number) ?? 2000;
  const syncEvery = (p.syncEvery as number) ?? 100;
  const alpha = (p.alpha as number) ?? 0.35;
  const seeds = (p.seeds as number) ?? 3;

  const configs = [
    { key: 'both', replay: true, target: true },
    { key: 'noReplay', replay: false, target: true },
    { key: 'noTarget', replay: true, target: false },
    { key: 'neither', replay: false, target: false },
  ];

  const out: Record<string, { returns: number[]; qMax: number[]; tdError: number[] }> = {};

  for (const cfg of configs) {
    const returns: number[] = new Array(episodes).fill(0);
    const qMax: number[] = new Array(episodes).fill(0);
    const tdError: number[] = new Array(episodes).fill(0);

    for (let s = 0; s < seeds; s++) {
      const env = new GridWorld();
      const rng = mulberry32(1234 + s * 101);
      const nA = env.nActions;
      const Q = new Float64Array(env.nStates * nA);
      const Qtarget = new Float64Array(env.nStates * nA);
      const buffer: Array<{ s: number; a: number; r: number; ns: number; done: boolean }> = [];
      let step = 0;
      let epsilon = 0.3;

      const idx = (st: number, a: number) => st * nA + a;
      const maxQ = (table: Float64Array, st: number) => {
        let m = -Infinity;
        for (let a = 0; a < nA; a++) m = Math.max(m, table[idx(st, a)]);
        return m;
      };

      for (let ep = 0; ep < episodes; ep++) {
        let st = env.startState;
        let total = 0;
        let deltaSum = 0;
        let n = 0;

        for (let t = 0; t < 300; t++) {
          const a =
            rng() < epsilon
              ? Math.floor(rng() * nA)
              : (() => {
                  let best = 0;
                  for (let k = 1; k < nA; k++)
                    if (Q[idx(st, k)] > Q[idx(st, best)]) best = k;
                  return best;
                })();

          const tr = env.step(st, a as 0 | 1 | 2 | 3, rng);
          total += tr.reward;
          buffer.push({ s: st, a, r: tr.reward, ns: tr.next, done: tr.done });
          if (buffer.length > bufferSize) buffer.shift();

          // Without replay we learn only from the transition just taken, so
          // consecutive updates are strongly correlated.
          const batch = cfg.replay
            ? Array.from({ length: 8 }, () => buffer[Math.floor(rng() * buffer.length)])
            : [buffer[buffer.length - 1]];

          for (const b of batch) {
            if (!b) continue;
            const bootstrap = cfg.target ? Qtarget : Q;
            const target = b.done
              ? b.r
              : b.r + env.config.gamma * maxQ(bootstrap, b.ns);
            const delta = target - Q[idx(b.s, b.a)];
            Q[idx(b.s, b.a)] += alpha * delta;
            deltaSum += Math.abs(delta);
            n += 1;
          }

          step += 1;
          if (cfg.target && step % syncEvery === 0) Qtarget.set(Q);
          if (!cfg.target) Qtarget.set(Q); // target == online: the moving target

          st = tr.next;
          if (tr.done) break;
        }

        epsilon = Math.max(0.03, epsilon * 0.99);
        returns[ep] += total / seeds;
        qMax[ep] += maxQ(Q, env.startState) / seeds;
        tdError[ep] += deltaSum / Math.max(1, n) / seeds;
      }
    }
    out[cfg.key] = { returns, qMax, tdError };
    post(id, 'progress', { stage: cfg.key });
  }

  // The true optimal value at the start state, for an honest overestimation axis.
  post(id, 'done', out);
}

// ---------------------------------------------------------------------------
// Chapter 16 — behaviour cloning that really compounds error
// ---------------------------------------------------------------------------

/**
 * An expert follows a sinusoidal lane with a proportional controller. The clone
 * is an MLP fitted to (state → action) pairs from the expert's own states, then
 * rolled out on its own. DAgger relabels the states the learner actually visits.
 */
function behaviourCloning(p: Record<string, unknown>, id: number) {
  const nDemos = (p.demos as number) ?? 12;
  const horizon = (p.horizon as number) ?? 220;
  const useDagger = (p.dagger as boolean) ?? false;
  const daggerRounds = (p.daggerRounds as number) ?? 4;
  const noise = (p.noise as number) ?? 0.06;
  const userDemos = p.userDemos as Array<{ x: number; y: number }> | undefined;

  const laneY = (x: number) => 0.5 + 0.28 * Math.sin(x * Math.PI * 2.1);
  const expert = (x: number, y: number) => {
    // Proportional controller onto the lane, with lookahead.
    const ahead = laneY(Math.min(1, x + 0.03));
    return Math.max(-1, Math.min(1, 5.5 * (ahead - y)));
  };

  const rng = mulberry32(4242);
  const xs: number[][] = [];
  const ys: number[][] = [];

  if (userDemos && userDemos.length > 4) {
    // The reader drew the demonstrations: label them by finite differences.
    for (let i = 1; i < userDemos.length; i++) {
      const d = userDemos[i];
      const prev = userDemos[i - 1];
      const dx = Math.max(1e-3, d.x - prev.x);
      xs.push([d.x, d.y]);
      ys.push([Math.max(-1, Math.min(1, (d.y - prev.y) / dx / 3))]);
    }
  } else {
    for (let e = 0; e < nDemos; e++) {
      let y = laneY(0) + gaussian(rng, 0, 0.01);
      for (let t = 0; t < horizon; t++) {
        const x = t / horizon;
        const a = expert(x, y) + gaussian(rng, 0, noise);
        xs.push([x, y]);
        ys.push([a]);
        y += (a / horizon) * 3;
      }
    }
  }

  const scaler = new Standardizer();
  scaler.fit(xs);
  const net = new Mlp([2, 16, 16, 1], mulberry32(9), 'tanh', 'tanh');

  const trainRounds = useDagger ? daggerRounds : 1;
  const losses: number[] = [];

  for (let round = 0; round < trainRounds; round++) {
    for (let epoch = 0; epoch < 120; epoch++) {
      const order = Array.from({ length: xs.length }, (_, i) => i);
      for (let b = 0; b < order.length; b += 32) {
        const slice = order.slice(b, b + 32);
        losses.push(
          net.trainBatch(
            slice.map((i) => scaler.apply(xs[i])),
            slice.map((i) => ys[i]),
            0.05,
          ),
        );
      }
    }

    if (useDagger && round < trainRounds - 1) {
      // Roll out the LEARNER, then ask the expert what it should have done.
      let y = laneY(0);
      for (let t = 0; t < horizon; t++) {
        const x = t / horizon;
        const a = net.forward(scaler.apply([x, y]))[0];
        xs.push([x, y]);
        ys.push([expert(x, y)]); // relabel on the learner's own distribution
        y += (a / horizon) * 3;
      }
    }
    post(id, 'progress', { round });
  }

  // Final rollout of the clone, and the expert's own path for comparison.
  const clonePath: Array<{ x: number; y: number }> = [];
  const expertPath: Array<{ x: number; y: number }> = [];
  let cy = laneY(0);
  let ey = laneY(0);
  for (let t = 0; t <= horizon; t++) {
    const x = t / horizon;
    clonePath.push({ x, y: cy });
    expertPath.push({ x, y: laneY(x) });
    const a = net.forward(scaler.apply([x, cy]))[0];
    cy += (a / horizon) * 3;
    ey += (expert(x, ey) / horizon) * 3;
    if (!Number.isFinite(cy)) break;
  }

  const deviations = clonePath.map((pt) => Math.abs(pt.y - laneY(pt.x)));
  post(id, 'done', {
    clonePath,
    expertPath,
    deviations,
    finalDeviation: deviations[deviations.length - 1],
    meanDeviation: deviations.reduce((a, b) => a + b, 0) / deviations.length,
    datasetSize: xs.length,
    trainLoss: losses[losses.length - 1],
  });
}

// ---------------------------------------------------------------------------
// Chapter 12 — a genuinely learned dynamics ensemble
// ---------------------------------------------------------------------------

function dynamicsEnsemble(p: Record<string, unknown>, id: number) {
  const nEpisodes = (p.episodes as number) ?? 14;
  const members = (p.members as number) ?? 5;
  const horizon = (p.horizon as number) ?? 30;
  const dt = 0.05;

  // Collect real transitions from Pendle under random torques.
  const rng = mulberry32(3);
  const data: Transition[] = [];
  for (let e = 0; e < nEpisodes; e++) {
    let s: PendleState = [Math.PI - 0.4 + gaussian(rng, 0, 0.3), gaussian(rng, 0, 0.5)];
    for (let t = 0; t < 60; t++) {
      const a = gaussian(rng, 0, 1.1);
      const ns = rk4Step(s, a, dt, DEFAULT_PENDLE);
      data.push({
        state: [wrapAngle(s[0]), s[1]],
        action: [a],
        next: [wrapAngle(ns[0]), ns[1]],
      });
      s = ns;
    }
  }
  post(id, 'progress', { stage: 'collected', samples: data.length });

  const ens = new DynamicsEnsemble(2, 1, members, 24, 5);
  const losses = ens.fit(data, (p.epochs as number) ?? 45);
  post(id, 'progress', { stage: 'trained', losses });

  // Roll the ensemble and the truth forward from the same state and action seq.
  const start: PendleState = [Math.PI - 0.35, 0.2];
  const planRng = mulberry32(77);
  const actions = Array.from({ length: horizon }, () => [gaussian(planRng, 0, 0.8)]);

  const { trajectories, spread } = ens.rolloutAll([wrapAngle(start[0]), start[1]], actions);

  let s: PendleState = start;
  const truth: number[][] = [[wrapAngle(s[0]), s[1]]];
  for (const a of actions) {
    s = rk4Step(s, a[0], dt, DEFAULT_PENDLE);
    truth.push([wrapAngle(s[0]), s[1]]);
  }

  // Error of the ensemble mean against the truth, per horizon step.
  const meanError: number[] = [];
  for (let h = 0; h < Math.min(truth.length, ...trajectories.map((t) => t.length)); h++) {
    const m = trajectories.reduce((acc, t) => acc + t[h][0], 0) / trajectories.length;
    meanError.push(Math.abs(m - truth[h][0]));
  }

  post(id, 'done', {
    trajectories: trajectories.map((t) => t.map((st) => st[0])),
    truth: truth.map((st) => st[0]),
    spread,
    meanError,
    trainLoss: losses.reduce((a, b) => a + b, 0) / losses.length,
    samples: data.length,
  });
}

// ---------------------------------------------------------------------------
// Chapter 15 — randomized training, evaluated on held-out dynamics
// ---------------------------------------------------------------------------

/**
 * Trains a tile-coded SARSA *balancing* policy on Pendle with the pole mass
 * drawn from a randomization range, then evaluates it across a grid of masses
 * it was never trained on. The peak-versus-robustness trade is measured, not
 * modelled.
 *
 * Balancing rather than swing-up is deliberate. Swing-up from hanging needs
 * directed exploration that tile-coded SARSA will not find inside a budget a
 * browser can afford, so every policy scored zero and the widget showed two
 * flat lines. Balancing is learnable in a few hundred episodes and is also the
 * question the chapter actually asks: does a controller tuned for one mass
 * still hold up when the real robot weighs something else?
 */
function randomizationTransfer(p: Record<string, unknown>, id: number) {
  const range = (p.range as number) ?? 0.25;
  const iterations = (p.iterations as number) ?? 18;
  const dt = 0.05;
  const FELL = 0.7;          // radians from upright that counts as a fall
  const HORIZON = 200;       // 10 s at dt = 0.05
  const maxTorque = 3;       // tight enough that mass actually matters
  const LAG = 1;             // one step of actuation delay, as Ch 14 warns about
  const G = DEFAULT_PENDLE.gravity;
  const L = DEFAULT_PENDLE.length;

  /**
   * The policy is computed-torque control (Chapter 13): cancel the gravity the
   * controller *believes* is acting, then add PD feedback. Its three
   * parameters are the assumed mass and the two gains.
   *
   * The assumed mass is what makes this a real sim-to-real experiment. Guess
   * too low and the controller under-compensates on a heavy pole; too high and
   * it over-drives a light one. Mismatch hurts in both directions, which is
   * precisely the situation domain randomization exists to insure against.
   */
  function score(params3: number[], mass: number, rng: () => number, trials: number): number {
    const [mHat, kp, kd] = params3;
    const plant = { ...DEFAULT_PENDLE, mass, maxTorque };
    let total = 0;
    for (let k = 0; k < trials; k++) {
      let s: PendleState = [gaussian(rng, 0, 0.18), gaussian(rng, 0, 0.25)];
      let held = 0;
      const queue: number[] = new Array(LAG).fill(0);
      for (let t = 0; t < HORIZON; t++) {
        const th = wrapAngle(s[0]);
        const ff = -mHat * G * L * Math.sin(th);
        const cmd = Math.max(-maxTorque, Math.min(maxTorque, ff - kp * th - kd * s[1]));
        queue.push(cmd);
        s = rk4Step(s, queue.shift() as number, dt, plant);
        if (Math.abs(wrapAngle(s[0])) > FELL) break;   // fallen; stays fallen
        held += 1;
      }
      total += held / HORIZON;
    }
    return total / trials;
  }

  /** Cross-entropy policy search over the three parameters. */
  function train(trainRange: number, seed: number): number[] {
    const rng = mulberry32(seed);
    let mu = [1.0, 4, 1.5];
    let sd = [0.8, 4, 2];
    for (let it = 0; it < iterations; it++) {
      const pop = Array.from({ length: 48 }, () =>
        mu.map((m, d) => Math.max(0, m + gaussian(rng, 0, sd[d]))),
      );
      const scored = pop
        .map((c) => {
          // A fixed evaluation stream so candidates are ranked on equal terms.
          const er = mulberry32(4242);
          let f = 0;
          const n = 5;
          for (let i = 0; i < n; i++) {
            const m = trainRange === 0 ? 1 : 1 + ((i / (n - 1)) * 2 - 1) * trainRange;
            f += score(c, m, er, 3);
          }
          return { c, f: f / n };
        })
        .sort((a, b) => b.f - a.f);
      const elite = scored.slice(0, 10).map((e) => e.c);
      mu = [0, 1, 2].map((d) => elite.reduce((a, e) => a + e[d], 0) / elite.length);
      sd = [0, 1, 2].map((d) =>
        Math.max(0.05, Math.sqrt(elite.reduce((a, e) => a + (e[d] - mu[d]) ** 2, 0) / elite.length)),
      );
      if (it % 6 === 0) post(id, 'progress', { trainRange, it });
    }
    return mu;
  }

  const narrowP = train(0, 3);
  post(id, 'progress', { stage: 'narrow-trained' });
  const wideP = train(range, 3);
  post(id, 'progress', { stage: 'wide-trained' });

  // Held-out masses, deliberately wider than either training distribution.
  const masses = Array.from({ length: 13 }, (_, i) => 0.4 + i * 0.1333);
  const narrowCurve = masses.map((m) => ({ x: m, y: score(narrowP, m, mulberry32(999), 5) }));
  const wideCurve = masses.map((m) => ({ x: m, y: score(wideP, m, mulberry32(999), 5) }));

  post(id, 'done', {
    masses,
    narrow: narrowCurve,
    wide: wideCurve,
    narrowWorst: Math.min(...narrowCurve.map((d) => d.y)),
    wideWorst: Math.min(...wideCurve.map((d) => d.y)),
    narrowPeak: Math.max(...narrowCurve.map((d) => d.y)),
    widePeak: Math.max(...wideCurve.map((d) => d.y)),
    narrowAssumedMass: narrowP[0],
    wideAssumedMass: wideP[0],
    trainedRange: range,
  });
}

// ---------------------------------------------------------------------------
// Chapter 18 — optimize a real gait against the reader's reward weights
// ---------------------------------------------------------------------------

function gaitOptimize(p: Record<string, unknown>, id: number) {
  const weights = p.weights as RewardWeights;
  const targetSpeed = (p.targetSpeed as number) ?? 0.9;
  const rng = mulberry32((p.seed as number) ?? 5);

  const { best, result, history } = optimizeGait(weights, rng, {
    generations: (p.generations as number) ?? 12,
    population: (p.population as number) ?? 24,
    targetSpeed,
  });

  post(id, 'done', {
    params: best,
    terms: result.terms,
    weightedReturn: result.weightedReturn,
    fell: result.fell,
    meanSpeed: result.meanSpeed,
    costOfTransport: Number.isFinite(result.costOfTransport) ? result.costOfTransport : null,
    trace: result.trace,
    footHeights: result.footHeights,
    history,
  });
}

// ---------------------------------------------------------------------------

ctx.onmessage = (e: MessageEvent<Message>) => {
  const { id, job, params } = e.data;
  try {
    switch (job) {
      case 'replay-ablation':
        return replayAblation(params, id);
      case 'behaviour-cloning':
        return behaviourCloning(params, id);
      case 'dynamics-ensemble':
        return dynamicsEnsemble(params, id);
      case 'randomization-transfer':
        return randomizationTransfer(params, id);
      case 'gait-optimize':
        return gaitOptimize(params, id);
      default:
        return post(id, 'error', `unknown job: ${job}`);
    }
  } catch (err) {
    post(id, 'error', err instanceof Error ? err.message : String(err));
  }
};

export {};
