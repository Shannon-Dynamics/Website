'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import {
  TIGER_LISTEN,
  TIGER_OPEN_LEFT,
  TIGER_OPEN_RIGHT,
  type AlphaVec,
  type Belief,
  type FinitePomdp,
  beliefUpdate,
  envelope2,
  exactVi,
  makeTiger,
  qmdpAlphas,
  sampleRow,
  valueAt,
} from '@/lib/pomdp/finite';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w22.1 — the Tiger Door Console.
 *
 * Two identical doors. Behind one, Rusty's charging dock; behind the other, a
 * stairwell drop wearing a tiger costume. The robot may listen, for a price.
 *
 * The console draws the two objects the chapter is about, one above the other:
 * the *game* (doors, growls, a score) and the *value function over the belief
 * simplex* — every α-vector a line over b(tiger-left), their upper envelope in
 * bold, and the belief itself a marker sliding along it. Listening walks the
 * marker; opening a door resets it to the middle.
 *
 * Three pilots run in lockstep on the same seed stream, so the scoreboard is a
 * paired comparison rather than three separate experiments.
 */

const GAMMA = 0.95;
const PILOTS = ['optimal', 'qmdp', 'impatient'] as const;
type PilotId = (typeof PILOTS)[number];

const PILOT_LABEL: Record<PilotId, string> = {
  optimal: 'Optimal',
  qmdp: 'QMDP',
  impatient: 'Impatient',
};

interface Run {
  /** This pilot's own seeded stream, so its luck is reproducible on its own. */
  rng: Rng;
  b: Belief;
  /** The true tiger position: 0 = left, 1 = right. The pilot never sees it. */
  x: number;
  listens: number;
  games: number;
  score: number;
  mauls: number;
  opens: number;
  lastAction: number | null;
  lastObs: number | null;
  lastReward: number | null;
}

interface State {
  runs: Record<PilotId | 'you', Run>;
}

const freshRun = (seed: number): Run => {
  const rng = new Rng(seed);
  return {
    rng,
    b: [0.5, 0.5],
    x: rng.next() < 0.5 ? 0 : 1,
    listens: 0,
    games: 0,
    score: 0,
    mauls: 0,
    opens: 0,
    lastAction: null,
    lastObs: null,
    lastReward: null,
  };
};

/** Advance one pilot by one action. The run's own generator supplies the luck. */
function act(m: FinitePomdp, run: Run, u: number): Run {
  if (u === TIGER_LISTEN) {
    const z = sampleRow(m.o[TIGER_LISTEN][run.x], run.rng);
    return {
      ...run,
      b: beliefUpdate(m, run.b, TIGER_LISTEN, z).b,
      listens: run.listens + 1,
      score: run.score - 1,
      lastAction: u,
      lastObs: z,
      lastReward: -1,
    };
  }
  const reward = m.r[run.x][u];
  return {
    ...run,
    // Opening ends the round: the tiger is re-placed and the belief resets.
    b: [0.5, 0.5],
    x: run.rng.next() < 0.5 ? 0 : 1,
    listens: 0,
    games: run.games + 1,
    score: run.score + reward,
    mauls: run.mauls + (reward < 0 ? 1 : 0),
    opens: run.opens + 1,
    lastAction: u,
    lastObs: null,
    lastReward: reward,
  };
}

export function TigerDoorConsole() {
  const [accuracy, setAccuracy] = useState(0.85);
  const [pilot, setPilot] = useState<PilotId | 'you'>('optimal');

  const model = useMemo(() => makeTiger(accuracy, GAMMA), [accuracy]);

  /**
   * Exact α-vector value iteration, live. A short horizon keeps the slider
   * responsive: at accuracy 0.85 the horizon-20 set already puts the open
   * threshold within 0.001 of the converged 0.9603, and a barely-informative
   * sensor needs a shorter horizon because its Γ grows much faster.
   */
  const optimalSet: AlphaVec[] = useMemo(() => {
    const horizon = accuracy < 0.65 ? 12 : 20;
    const stages = exactVi(model, horizon, { prune: true });
    return stages[stages.length - 1].gamma;
  }, [model, accuracy]);

  const qmdpSet: AlphaVec[] = useMemo(() => qmdpAlphas(model), [model]);

  const policies = useMemo(() => {
    const impatient = (b: Belief) => (b[0] > 0.5 ? TIGER_OPEN_RIGHT : TIGER_OPEN_LEFT);
    return {
      optimal: (b: Belief) => valueAt(optimalSet, b).action,
      qmdp: (b: Belief) => valueAt(qmdpSet, b).action,
      impatient,
    } satisfies Record<PilotId, (b: Belief) => number>;
  }, [optimalSet, qmdpSet]);

  const init = useCallback(
    (seed: number): State => ({
      runs: {
        optimal: freshRun(seed * 7 + 1),
        qmdp: freshRun(seed * 7 + 2),
        impatient: freshRun(seed * 7 + 3),
        you: freshRun(seed * 7 + 4),
      },
    }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      const runs = { ...s.runs };
      for (const id of PILOTS) {
        runs[id] = act(model, runs[id], policies[id](runs[id].b));
      }
      return { runs };
    },
    [model, policies],
  );

  const sim = useSimulation<State>({ init, step, fps: 3, initialSeed: 22 });
  const { pause, setState } = sim;

  const human = useCallback(
    (u: number) => {
      pause();
      setState((s) => ({ runs: { ...s.runs, you: act(model, s.runs.you, u) } }));
    },
    [model, pause, setState],
  );

  const shown = pilot === 'you' ? sim.state.runs.you : sim.state.runs[pilot];
  const shownSet = pilot === 'qmdp' ? qmdpSet : optimalSet;
  const segments = useMemo(() => envelope2(shownSet), [shownSet]);

  /** Smallest b(tiger-left) at which this α-set commands "open the right door". */
  const openThreshold = useMemo(() => {
    const seg = segments.filter((s) => s.action === TIGER_OPEN_RIGHT);
    return seg.length > 0 ? seg[0].tStart : 1;
  }, [segments]);

  /**
   * Each pilot's decision threshold, side by side. This is the comparison the
   * widget exists to make: the three pilots differ in exactly one number, and
   * every difference in the scoreboard below is a consequence of it.
   */
  const thresholds: Record<PilotId, number> = useMemo(
    () => ({
      optimal: thresholdOf(optimalSet),
      qmdp: thresholdOf(qmdpSet),
      impatient: 0.5,
    }),
    [optimalSet, qmdpSet],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const actionColor = [p.measurement, p.prediction, p.posterior];

      // ---- the two doors ------------------------------------------------
      const doorTop = sy(v, 1.3);
      const doorBot = sy(v, 0.98);
      const doorW = sl(v, 0.26);
      for (const [k, x0] of [
        [TIGER_OPEN_LEFT, 0.1],
        [TIGER_OPEN_RIGHT, 0.64],
      ] as const) {
        const px = sx(v, x0);
        ctx.fillStyle = p.free;
        ctx.fillRect(px, doorTop, doorW, doorBot - doorTop);
        ctx.strokeStyle = shown.lastAction === k ? actionColor[k] : p.wall;
        ctx.lineWidth = shown.lastAction === k ? 3 : 2;
        ctx.strokeRect(px, doorTop, doorW, doorBot - doorTop);
        label(ctx, k === TIGER_OPEN_LEFT ? 'LEFT' : 'RIGHT', px + doorW / 2, doorTop + 14, p.truth, {
          size: 10,
          align: 'center',
        });
        // The door just opened: show what was behind it. Otherwise, nothing —
        // the robot never gets to see the true state.
        if (shown.lastAction === k && shown.lastReward !== null) {
          label(
            ctx,
            shown.lastReward > 0 ? 'dock  +10' : 'tiger  −100',
            px + doorW / 2,
            (doorTop + doorBot) / 2,
            shown.lastReward > 0 ? p.posterior : p.prediction,
            { size: 13, align: 'center', weight: 700 },
          );
        }
      }

      // The growl: drawn on the side the sensor claims to have heard.
      if (shown.lastAction === TIGER_LISTEN && shown.lastObs !== null) {
        const gx = sx(v, shown.lastObs === 0 ? 0.23 : 0.77);
        label(ctx, 'growl', gx, doorBot + 12, p.measurement, { size: 11, align: 'center', weight: 600 });
      }

      // ---- the α-vector envelope over the belief segment ----------------
      const plotTop = 0.9;
      const plotBot = 0.16;
      const probe = (t: number) => valueAt(shownSet, [t, 1 - t]).value;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i <= 200; i++) {
        const val = probe(i / 200);
        lo = Math.min(lo, val);
        hi = Math.max(hi, val);
      }
      // Individual α-vectors dive far below the envelope; clip the window to
      // the envelope's own range plus a margin, or the picture is all whitespace.
      const span = Math.max(hi - lo, 1e-6);
      const yLo = lo - 0.55 * span;
      const yHi = hi + 0.12 * span;
      const yOf = (val: number) =>
        sy(v, plotBot + ((val - yLo) / (yHi - yLo)) * (plotTop - plotBot));

      // Every α-vector, faint: the reader should see how many lines are hiding
      // under the envelope, and that each one is straight.
      ctx.strokeStyle = p.truth;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1;
      for (const a of shownSet) {
        ctx.beginPath();
        ctx.moveTo(sx(v, 0), yOf(a.v[1]));
        ctx.lineTo(sx(v, 1), yOf(a.v[0]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // The upper envelope, one bold stroke per action interval.
      for (const seg of segments) {
        ctx.strokeStyle = actionColor[seg.action];
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(sx(v, seg.tStart), yOf(seg.intercept + seg.slope * seg.tStart));
        ctx.lineTo(sx(v, seg.tEnd), yOf(seg.intercept + seg.slope * seg.tEnd));
        ctx.stroke();

        // The policy strip: which action this piece commands.
        ctx.fillStyle = actionColor[seg.action];
        ctx.globalAlpha = 0.5;
        ctx.fillRect(
          sx(v, seg.tStart),
          sy(v, 0.1),
          Math.max(sl(v, seg.tEnd - seg.tStart), 1),
          sl(v, 0.045),
        );
        ctx.globalAlpha = 1;
      }

      // ---- the belief axis ----------------------------------------------
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, 0.1));
      ctx.lineTo(sx(v, 1), sy(v, 0.1));
      ctx.stroke();
      label(ctx, 'b(tiger-left) = 0', sx(v, 0), sy(v, 0.03), p.truth, { size: 9 });
      label(ctx, '1', sx(v, 1), sy(v, 0.03), p.truth, { size: 9, align: 'right' });
      label(ctx, 'V(b)', sx(v, 0.005), yOf(yHi) + 8, p.truth, { size: 9 });

      // The belief itself: prior-blue, because that is what it is.
      const t = shown.b[0];
      const bx = sx(v, t);
      ctx.strokeStyle = p.prior;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(bx, sy(v, 0.1));
      ctx.lineTo(bx, yOf(probe(t)));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = p.prior;
      ctx.beginPath();
      ctx.arc(bx, yOf(probe(t)), 4.5, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, `b = ${t.toFixed(3)}`, bx, yOf(probe(t)) - 12, p.prior, {
        size: 10,
        align: t > 0.7 ? 'right' : 'left',
        weight: 600,
      });

      // The threshold the policy actually uses, called out by number.
      if (openThreshold < 1) {
        const tx = sx(v, openThreshold);
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(tx, sy(v, 0.06));
        ctx.lineTo(tx, sy(v, plotTop));
        ctx.stroke();
        ctx.setLineDash([]);
        label(ctx, openThreshold.toFixed(3), tx - 4, sy(v, 0.045), p.posterior, {
          size: 9,
          align: 'right',
        });
      }
    },
    [shown, shownSet, segments, openThreshold],
  );

  const scoreboard = PILOTS.map((id) => {
    const r = sim.state.runs[id];
    return {
      id,
      games: r.games,
      opens: r.opens,
      mean: r.games > 0 ? r.score / r.games : 0,
      maul: r.opens > 0 ? (100 * r.mauls) / r.opens : 0,
      threshold: thresholds[id],
    };
  });

  return (
    <WidgetFrame
      id="w22.1"
      title="The Tiger Door Console"
      teaches="Sensing is an action with a price, and the price is worth paying exactly when the belief is near the middle. Acting on the most likely state throws that away."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          The lower panel is the whole theory: each faint gray line is one α-vector, the bold
          coloured curve is their upper envelope <em>V(b)</em>, and the strip beneath it is the
          policy — green where the optimal action is <em>listen</em>, orange and purple where it is
          to open a door. The blue marker is the belief, and listening slides it. Watch it climb the
          ladder <code>0.5 → 0.85 → 0.9698</code> and only then cross the threshold at 0.9611.{' '}
          <strong>Try this:</strong> switch the pilot to QMDP. The envelope collapses to three
          straight lines and the threshold drops to exactly 0.900 — and it stays at 0.900 however far
          you drag the hearing-accuracy slider, because QMDP&rsquo;s <code>Q*</code> never looked at
          the observation model at all. At the default accuracy the two pilots nevertheless{' '}
          <em>behave</em> identically: the reachable ladder steps clean over the gap between 0.900
          and 0.9611, so a tournament run here would certify QMDP as optimal. Now drag accuracy down
          to 0.70. The ladder becomes <code>0.70 → 0.845 → 0.927 → 0.967</code>, QMDP opens one rung
          early, and over 20,000 paired rounds it is mauled 7.2% of the time against the optimal
          pilot&rsquo;s 3.2%. Then drag accuracy <em>up</em> to 0.95: now a single growl clears
          QMDP&rsquo;s fixed 0.900, it opens after one listen, and it is eaten eighteen times as
          often. A better microphone made the approximation worse.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: 1, minY: 0, maxY: 1.34 }}
        draw={draw}
        deps={[sim.tick, shown, shownSet, pilot]}
        aspect={2.1}
        padding={0.04}
        ariaLabel="Two doors above a plot of the POMDP value function over the belief that the tiger is behind the left door. Straight alpha-vector lines meet in an upper envelope; a coloured strip below shows which action is optimal at each belief."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="belief b(left)" value={shown.b[0].toFixed(4)} />
        <Stat label="open threshold" value={openThreshold >= 1 ? 'never' : openThreshold.toFixed(4)} />
        <Stat label="|Γ| vectors" value={String(shownSet.length)} />
        <Stat label="listens this round" value={String(shown.listens)} />
      </div>

      <div className="border-t border-fd-border px-3 py-2">
        <p className="eyebrow mb-1.5">Tournament — every pilot on its own seeded stream</p>
        <table className="w-full font-mono text-[0.72rem] tabular-nums">
          <thead className="text-fd-muted-foreground">
            <tr>
              <th className="text-left font-normal">pilot</th>
              <th className="text-right font-normal">opens at b ≥</th>
              <th className="text-right font-normal">rounds</th>
              <th className="text-right font-normal">mean score / round</th>
              <th className="text-right font-normal">mauled</th>
            </tr>
          </thead>
          <tbody>
            {scoreboard.map((row) => (
              <tr key={row.id} style={row.id === pilot ? { color: 'var(--pr-prior)' } : undefined}>
                <td className="text-left">{PILOT_LABEL[row.id]}</td>
                <td className="text-right">{row.threshold.toFixed(4)}</td>
                <td className="text-right">{row.games}</td>
                <td className="text-right">{row.games > 0 ? row.mean.toFixed(2) : '—'}</td>
                <td className="text-right">{row.opens === 0 ? '—' : `${row.maul.toFixed(1)}%`}</td>
              </tr>
            ))}
            <tr style={pilot === 'you' ? { color: 'var(--pr-prior)' } : undefined}>
              <td className="text-left">You</td>
              <td className="text-right">—</td>
              <td className="text-right">{sim.state.runs.you.games}</td>
              <td className="text-right">
                {sim.state.runs.you.games > 0
                  ? (sim.state.runs.you.score / sim.state.runs.you.games).toFixed(2)
                  : '—'}
              </td>
              <td className="text-right">
                {sim.state.runs.you.opens === 0
                  ? '—'
                  : `${((100 * sim.state.runs.you.mauls) / sim.state.runs.you.opens).toFixed(1)}%`}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="Hearing accuracy"
          role="measurement"
          value={accuracy}
          min={0.55}
          max={0.99}
          step={0.01}
          onChange={setAccuracy}
          help="p(hear-left | tiger-left). At 0.5 the growl says nothing and no amount of listening can ever justify opening a door."
        />
        <ButtonRow>
          {(['optimal', 'qmdp', 'impatient', 'you'] as const).map((id) => (
            <ActionButton key={id} onClick={() => setPilot(id)} emphasis={pilot === id}>
              {id === 'you' ? 'You play' : PILOT_LABEL[id]}
            </ActionButton>
          ))}
        </ButtonRow>
        {pilot === 'you' ? (
          <ButtonRow>
            <ActionButton onClick={() => human(TIGER_LISTEN)}>Listen (−1)</ActionButton>
            <ActionButton onClick={() => human(TIGER_OPEN_LEFT)}>Open left</ActionButton>
            <ActionButton onClick={() => human(TIGER_OPEN_RIGHT)}>Open right</ActionButton>
          </ButtonRow>
        ) : null}
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

/** Smallest b(tiger-left) at which an α-set commands "open the right door". */
function thresholdOf(set: readonly AlphaVec[]): number {
  const seg = envelope2(set).filter((s) => s.action === TIGER_OPEN_RIGHT);
  return seg.length > 0 ? seg[0].tStart : 1;
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
