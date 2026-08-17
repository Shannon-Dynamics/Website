'use client';

import { useCallback, useMemo, useRef, useState } from 'react';

import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { HistogramFilter1D, gaussianKernel } from '@/lib/filters/bayes';
import {
  bestAction,
  greedyGoalAction,
  scoreActions,
  type ActionCandidate,
  type ActionScore,
  type Outcome,
} from '@/lib/explore/active-loc';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w24.2 — the Disambiguation Detour.
 *
 * A corridor whose doors repeat every 2.5 m, so a belief that is wrong by
 * exactly one door spacing is *never* contradicted by anything ahead. One
 * alcove, 3.75 m behind the robot, breaks the symmetry.
 *
 * Each trial is one scored decision and then a commitment, which is the honest
 * shape of the problem: choose a motion, take it, and then drive to the goal on
 * whatever the belief now says. Every bar is `scoreActions` from
 * `lib/explore/active-loc.ts` run for real — the belief is pushed through the
 * motion model, each possible reading is applied in turn, and the resulting
 * entropies are averaged with the reading's own probability.
 *
 * The candidate displacements are whole multiples of the 2.5 m door period, so
 * each of them lands both hypotheses at the *same* offset inside a feature.
 * Their expected gain is therefore exactly zero, and the bars say so.
 */

const LENGTH = 16;
const CELLS = 320;
/**
 * The goal sits well clear of the far wall. A boundary is itself a landmark,
 * and it would quietly disambiguate the very thing this widget is about.
 */
const GOAL = 12.0;
const DOORS = [3.5, 6.0, 8.5, 11.0, 13.5];
const ALCOVE = 1.0;
const FEATURE_HALF = 0.5;
/** The two hypotheses: one door spacing apart, and indistinguishable ahead. */
const MODE_A = 4.75;
const MODE_B = 7.25;
const MODE_SIGMA = 0.12;
const MOTION_SIGMA = 0.07;
/** Utility weights: bits, metres driven, and metres still owed to the goal. */
const W_C = 0.02;
const W_TASK = 0.05;
/** A wrong-mode commitment misses by exactly one door spacing, 2.5 m. */
const SUCCESS_TOL = 1.25;

type Feature = 'alcove' | 'door' | 'wall';

function featureAt(x: number): Feature {
  if (Math.abs(x - ALCOVE) < FEATURE_HALF) return 'alcove';
  for (const d of DOORS) if (Math.abs(x - d) < FEATURE_HALF) return 'door';
  return 'wall';
}

/** p(z | x) for a three-way detector that is right with probability q. */
function likelihoodOf(z: Feature, q: number): (x: number) => number {
  const wrong = (1 - q) / 2;
  return (x: number) => (featureAt(x) === z ? q : wrong);
}

const OUTCOME_LABELS: Feature[] = ['alcove', 'door', 'wall'];

const ACTIONS: ActionCandidate[] = [
  { label: 'detour −3.75 m', delta: -3.75 },
  { label: 'back −2.5 m', delta: -2.5 },
  { label: 'hold, scan', delta: 0 },
  { label: 'ahead +2.5 m', delta: 2.5 },
  { label: 'ahead +5.0 m', delta: 5.0 },
];

type Policy = 'entropy' | 'goal';
type Phase = 'decide' | 'commit';

interface State {
  filter: HistogramFilter1D;
  rng: Rng;
  truth: number;
  phase: Phase;
  index: number;
  scores: ActionScore[];
  chosen: number;
  lastReading: Feature | null;
  outcome: 'ok' | 'wrong' | null;
  trials: number;
  correct: number;
  history: ('ok' | 'wrong')[];
}

function bimodalPrior(): number[] {
  const out: number[] = [];
  for (let i = 0; i < CELLS; i++) {
    const x = ((i + 0.5) * LENGTH) / CELLS;
    out.push(
      Math.exp(-0.5 * ((x - MODE_A) / MODE_SIGMA) ** 2) +
        Math.exp(-0.5 * ((x - MODE_B) / MODE_SIGMA) ** 2),
    );
  }
  return out;
}

function freshFilter(): HistogramFilter1D {
  const filter = new HistogramFilter1D({ length: LENGTH, cells: CELLS, wrap: false });
  filter.setBelief(bimodalPrior());
  return filter;
}

export function DisambiguationDetour() {
  const [noise, setNoise] = useState(0.06);
  const [policy, setPolicy] = useState<Policy>('entropy');

  const noiseRef = useRef(noise);
  noiseRef.current = noise;
  const policyRef = useRef(policy);
  policyRef.current = policy;

  const init = useCallback((seed: number): State => {
    const rng = new Rng(seed);
    return {
      filter: freshFilter(),
      rng,
      // Which hypothesis is true is a coin flip the robot cannot see. The
      // belief is identical either way — that is what makes this a fair test.
      truth: rng.next() < 0.5 ? MODE_A : MODE_B,
      phase: 'decide',
      index: 0,
      scores: [],
      chosen: -1,
      lastReading: null,
      outcome: null,
      trials: 0,
      correct: 0,
      history: [],
    };
  }, []);

  const step = useCallback((s: State): State => {
    const q = 1 - noiseRef.current;

    // ---- commit: drive to the goal on whatever the belief now says --------
    if (s.phase === 'commit') {
      const displacement = GOAL - s.filter.mode();
      const truth = s.truth + displacement + s.rng.normal(0, MOTION_SIGMA);
      s.filter.predict(displacement, gaussianKernel(MOTION_SIGMA));
      const ok = Math.abs(truth - GOAL) < SUCCESS_TOL;
      return {
        ...s,
        truth,
        phase: 'decide',
        outcome: ok ? 'ok' : 'wrong',
        trials: s.trials + 1,
        correct: s.correct + (ok ? 1 : 0),
        history: [...s.history, ok ? ('ok' as const) : ('wrong' as const)].slice(-28),
      };
    }

    // A finished trial is cleared before the next decision is scored.
    const finished = s.outcome !== null;
    const filter = finished ? freshFilter() : s.filter;
    const truth0 = finished ? (s.rng.next() < 0.5 ? MODE_A : MODE_B) : s.truth;
    const index = finished ? s.index + 1 : s.index;

    // ---- decide: score every candidate motion exactly ---------------------
    const mode = filter.mode();
    const outcomes: Outcome[] = OUTCOME_LABELS.map((z) => ({
      label: z,
      likelihood: likelihoodOf(z, q),
    }));

    // Forward-simulating the motion model without disturbing the live belief.
    const scratch = new HistogramFilter1D({ length: LENGTH, cells: CELLS, wrap: false });
    const predict = (p: number[], delta: number): number[] => {
      scratch.setBelief(p);
      scratch.predict(delta, gaussianKernel(MOTION_SIGMA));
      return scratch.belief();
    };

    const scores = scoreActions(
      { centers: filter.centers(), p: filter.belief() },
      ACTIONS,
      predict,
      outcomes,
      {
        wI: 1,
        wC: W_C,
        // A pure information objective never arrives anywhere. This is the
        // task talking: metres still owed to the goal, priced in bits.
        taskCost: (delta) => Math.abs(GOAL - (mode + delta)),
        wTask: W_TASK,
      },
    );

    const chosen =
      policyRef.current === 'entropy' ? bestAction(scores) : greedyGoalAction(scores, mode, GOAL);
    const delta = ACTIONS[chosen].delta;

    // Execute: the wheels deliver something slightly else, and the filter is
    // told only what was commanded.
    const truth = truth0 + delta + s.rng.normal(0, MOTION_SIGMA);
    filter.predict(delta, gaussianKernel(MOTION_SIGMA));

    // Sense: right with probability q, else uniform over the two wrong answers.
    const truthFeature = featureAt(truth);
    let reading: Feature;
    if (s.rng.next() < q) {
      reading = truthFeature;
    } else {
      const others = OUTCOME_LABELS.filter((z) => z !== truthFeature);
      reading = others[s.rng.next() < 0.5 ? 0 : 1];
    }
    filter.correct(likelihoodOf(reading, q));

    return {
      ...s,
      filter,
      truth,
      index,
      phase: 'commit',
      scores,
      chosen,
      lastReading: reading,
      outcome: null,
    };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 1.1, initialSeed: 24 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { filter, truth, lastReading, phase, outcome } = sim.state;
      const belief = filter.belief();
      const centers = filter.centers();

      const corridorY = 0.8;
      const yTop = sy(v, corridorY + 0.1);
      const yBot = sy(v, corridorY - 0.1);

      ctx.fillStyle = p.free;
      ctx.fillRect(sx(v, 0), yTop, sl(v, LENGTH), yBot - yTop);
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx(v, 0), yTop, sl(v, LENGTH), yBot - yTop);

      // Doors: identical, evenly spaced, and therefore useless on their own.
      for (const d of DOORS) {
        ctx.fillStyle = p.bg;
        ctx.fillRect(sx(v, d - FEATURE_HALF), yTop - 1, sl(v, 2 * FEATURE_HALF), yBot - yTop + 2);
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx(v, d - FEATURE_HALF), yTop);
        ctx.lineTo(sx(v, d - FEATURE_HALF), yBot);
        ctx.moveTo(sx(v, d + FEATURE_HALF), yTop);
        ctx.lineTo(sx(v, d + FEATURE_HALF), yBot);
        ctx.stroke();
      }

      // The alcove: the one thing in this corridor that is not repeated.
      ctx.save();
      ctx.globalAlpha = 0.3;
      ctx.fillStyle = p.measurement;
      ctx.fillRect(sx(v, ALCOVE - FEATURE_HALF), yTop - 13, sl(v, 2 * FEATURE_HALF), 13);
      ctx.restore();
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.6;
      ctx.strokeRect(sx(v, ALCOVE - FEATURE_HALF), yTop - 13, sl(v, 2 * FEATURE_HALF), 13);
      label(ctx, 'alcove', sx(v, ALCOVE), yTop - 20, p.measurement, { size: 10, align: 'center' });

      // The goal, in chrome teal: it is the task, not a belief.
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, GOAL), yTop - 9);
      ctx.lineTo(sx(v, GOAL), yBot + 5);
      ctx.stroke();
      ctx.setLineDash([]);
      label(ctx, 'goal', sx(v, GOAL), yTop - 16, p.accent, { size: 10, align: 'center' });

      // The true robot. It never gets to see this, and neither does the policy.
      const rx = sx(v, truth);
      const ry = (yTop + yBot) / 2;
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(rx, ry, 6, 0, Math.PI * 2);
      ctx.fill();

      if (outcome) {
        label(
          ctx,
          outcome === 'ok' ? 'arrived' : 'wrong room — off by one door spacing',
          rx + 12,
          ry,
          outcome === 'ok' ? p.posterior : p.prediction,
          { size: 10 },
        );
      } else if (lastReading) {
        label(ctx, `sensor: ${lastReading}`, rx + 12, ry, p.measurement, { size: 10 });
      }

      // The belief. Purple once it has collapsed to one hypothesis, blue while
      // it is still arguing with itself.
      const peak = Math.max(...belief, 1e-9);
      const modes = countModes(belief);
      const baseY = sy(v, 0.08);
      const topY = sy(v, 0.58);
      const h = baseY - topY;
      const barW = sl(v, LENGTH / CELLS);
      ctx.fillStyle = modes > 1 ? p.prior : p.posterior;
      for (let i = 0; i < CELLS; i++) {
        const bh = (belief[i] / peak) * h;
        ctx.fillRect(sx(v, centers[i]) - barW / 2, baseY - bh, Math.max(barW, 1), bh);
      }

      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), baseY);
      ctx.lineTo(sx(v, LENGTH), baseY);
      ctx.stroke();

      label(
        ctx,
        `${phase === 'decide' ? 'SCORING' : 'COMMITTING'} · bel(x): ${
          modes > 1 ? `${modes} hypotheses` : 'committed'
        } · H = ${filter.entropy().toFixed(2)} bits`,
        sx(v, 0.1),
        topY - 8,
        modes > 1 ? p.prior : p.posterior,
        { size: 11, weight: 600 },
      );
    },
    [sim.state],
  );

  const rate = useMemo(() => {
    const { trials, correct } = sim.state;
    return trials === 0 ? null : (100 * correct) / trials;
  }, [sim.state]);

  const peakInfo = useMemo(
    () => Math.max(...sim.state.scores.map((x) => x.mutualInfo), 1e-6),
    [sim.state.scores],
  );

  return (
    <WidgetFrame
      id="w24.2"
      title="Disambiguation Detour"
      teaches="Motion is a sensing action. When the belief is ambiguous, the informative route can be the fast route."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          The doors repeat every 2.5 m, so a belief that is wrong by exactly one door spacing is
          never contradicted by anything ahead of the robot. Every candidate below except the detour
          is a whole number of door spacings, which is why their expected gain is exactly{' '}
          <span className="font-mono">0.000</span> bits — the corridor cannot tell those two futures
          apart. Only the alcove, 3.75 m <em>behind</em>, can. Each trial is one scored decision and
          then a commitment: drive to the goal on whatever the belief now says. Switch to{' '}
          <em>goal-greedy</em> and watch the tally settle near 50%; the filter never complains,
          because nothing it saw was surprising. Then raise the sensor noise — near 0.2 the
          detour&rsquo;s expected gain falls below the metres it costs, the <em>scored</em> argmax
          flips to &ldquo;ahead&rdquo;, and the entropy-greedy policy degenerates into the one it
          had been beating.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.2, minY: 0, maxX: LENGTH + 0.2, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={3}
        padding={0}
        ariaLabel="A corridor with five evenly spaced identical doors and one alcove near the left end. Beneath it, a histogram of the robot's belief shows two peaks one door spacing apart until the robot detours past the alcove, after which a single peak remains."
      />

      <div className="border-t border-fd-border px-3 py-2.5">
        <p className="eyebrow mb-1.5">
          expected entropy reduction per candidate — H(b̄) − E&#8202;[H(bel′)]
        </p>
        <ul className="m-0 flex list-none flex-col gap-1 p-0">
          {sim.state.scores.map((s, k) => (
            <ScoreBar key={s.label} score={s} chosen={k === sim.state.chosen} peak={peakInfo} />
          ))}
          {sim.state.scores.length === 0 ? (
            <li className="font-ui text-xs text-fd-muted-foreground">Scoring…</li>
          ) : null}
        </ul>
      </div>

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="trial" value={`#${sim.state.index + 1}`} />
        <Stat label="phase" value={sim.state.phase === 'decide' ? 'scoring' : 'committing'} />
        <Stat
          label="arrived in the right place"
          value={
            rate === null ? '—' : `${sim.state.correct}/${sim.state.trials} (${rate.toFixed(0)}%)`
          }
        />
      </div>

      {sim.state.history.length > 0 ? (
        <div className="flex items-center gap-2 border-t border-fd-border px-3 py-2">
          <span className="eyebrow shrink-0">trial log</span>
          <span className="flex flex-wrap gap-1">
            {sim.state.history.map((o, k) => (
              <span
                key={k}
                title={o === 'ok' ? 'right room' : 'wrong room'}
                className="size-2.5 rounded-[1px]"
                style={{
                  background: o === 'ok' ? 'var(--pr-posterior)' : 'var(--pr-prediction)',
                  opacity: o === 'ok' ? 0.9 : 0.8,
                }}
              />
            ))}
          </span>
        </div>
      ) : null}

      <ControlPanel columns={2}>
        <Slider
          label="Sensor noise  1 − q"
          role="measurement"
          value={noise}
          min={0.02}
          max={0.4}
          step={0.01}
          onChange={setNoise}
          help="The chance the feature detector reports the wrong thing. Information is worth less when it is unreliable."
        />
        <div className="flex flex-col gap-1.5">
          <span className="font-ui text-[0.72rem] font-medium">Policy</span>
          <ButtonRow>
            <ActionButton onClick={() => setPolicy('entropy')} emphasis={policy === 'entropy'}>
              Entropy-greedy
            </ActionButton>
            <ActionButton onClick={() => setPolicy('goal')} emphasis={policy === 'goal'}>
              Goal-greedy
            </ActionButton>
          </ButtonRow>
          <span className="font-ui text-[0.68rem] text-fd-muted-foreground">
            {policy === 'entropy'
              ? 'argmax of H(bel) − E[H(bel′)] net of costs: a depth-1 POMDP backup.'
              : 'Drive toward the goal under the MAP hypothesis, and trust the filter.'}
          </span>
        </div>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function ScoreBar({ score, chosen, peak }: { score: ActionScore; chosen: boolean; peak: number }) {
  const frac = Math.max(0.012, Math.min(1, score.mutualInfo / peak));
  return (
    <li className="flex items-center gap-2">
      <span className="w-[7rem] shrink-0 font-mono text-[0.68rem] text-fd-muted-foreground">
        {score.label}
      </span>
      <span className="h-2 flex-1 overflow-hidden rounded-[1px] bg-fd-muted">
        <span
          className="block h-full"
          style={{
            width: `${frac * 100}%`,
            background: chosen ? 'var(--pr-posterior)' : 'var(--pr-measurement)',
            opacity: chosen ? 1 : 0.55,
          }}
        />
      </span>
      <span className="w-[10.5rem] shrink-0 text-end font-mono text-[0.68rem] tabular-nums text-fd-muted-foreground">
        I={score.mutualInfo.toFixed(3)} · U={score.utility.toFixed(2)}
      </span>
    </li>
  );
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}

/** How many peaks still carry real mass — the "how many hypotheses" readout. */
function countModes(belief: number[]): number {
  const peak = Math.max(...belief);
  const floor = peak * 0.3;
  let n = 0;
  for (let i = 1; i < belief.length - 1; i++) {
    if (belief[i] > floor && belief[i] >= belief[i - 1] && belief[i] > belief[i + 1]) n++;
  }
  return Math.max(n, 1);
}
