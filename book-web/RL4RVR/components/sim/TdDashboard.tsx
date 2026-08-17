'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GridWorld } from '@/lib/rl/gridworld';
import { TabularLearner, smooth, type EpisodeStats, type LearnerKind } from '@/lib/rl/td';
import { mulberry32 } from '@/lib/rl/random';
import { GridWorldCanvas, ValueLegend } from './GridWorldCanvas';
import { Segmented, SimControls, SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';

/**
 * `ch06-train-live` — model-free control, learning in front of you.
 *
 * Rusty starts knowing nothing: no map, no transition probabilities, only the
 * rewards that arrive after he acts. The heatmap fills in from the dock
 * backwards as the TD error propagates value one step at a time — the single
 * most important animation in Part I, because it shows *bootstrapping* doing
 * its work rather than describing it.
 */
export function TdDashboard() {
  const [kind, setKind] = useState<LearnerKind>('qlearning');
  /**
   * Driving it yourself is the fastest way to feel what the agent is up
   * against: no map, slippery floor, and a −10 shelf you cannot see coming
   * from the value function you do not have.
   */
  const [driving, setDriving] = useState(false);
  const [drive, setDrive] = useState<{ s: number; ret: number; steps: number; path: number[]; done: boolean } | null>(null);
  const [bestHuman, setBestHuman] = useState<number | null>(null);
  const [alpha, setAlpha] = useState(0.15);
  const [epsilon, setEpsilon] = useState(0.2);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(12);
  const [history, setHistory] = useState<EpisodeStats[]>([]);
  const [tick, setTick] = useState(0);

  const env = useMemo(() => new GridWorld(), []);
  const rng = useRef(mulberry32(7));
  const learner = useRef<TabularLearner | null>(null);
  const raf = useRef<number | null>(null);
  const acc = useRef(0);

  const rebuild = useCallback(() => {
    rng.current = mulberry32(7);
    learner.current = new TabularLearner(env, kind, {
      alpha,
      epsilon,
      epsilonDecay: 0.995,
      epsilonMin: 0.02,
      lambda: 0.9,
    });
    setHistory([]);
    setTick((t) => t + 1);
  }, [env, kind, alpha, epsilon]);

  useEffect(() => {
    rebuild();
    setPlaying(false);
  }, [rebuild]);

  const runEpisode = useCallback(() => {
    if (!learner.current) return;
    const stats = learner.current.stepEpisode(rng.current);
    setHistory((h) => [...h, stats]);
    setTick((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const loop = (now: number) => {
      acc.current += (now - last) * (speed / 1000);
      last = now;
      let guard = 0;
      while (acc.current >= 1 && guard < 12) {
        acc.current -= 1;
        guard += 1;
        runEpisode();
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, speed, runEpisode]);

  const startDrive = useCallback(() => {
    setDriving(true);
    setDrive({ s: env.startState, ret: 0, steps: 0, path: [env.startState], done: false });
  }, [env]);

  useEffect(() => {
    if (!driving) return;
    const onKey = (e: KeyboardEvent) => {
      const map: Record<string, number> = {
        ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3,
        w: 0, d: 1, s: 2, a: 3,
      };
      const a = map[e.key];
      if (a === undefined) return;
      e.preventDefault();
      setDrive((cur) => {
        if (!cur || cur.done) return cur;
        const t = env.step(cur.s, a as 0 | 1 | 2 | 3, rng.current);
        const next = {
          s: t.next,
          ret: cur.ret + t.reward,
          steps: cur.steps + 1,
          path: [...cur.path, t.next],
          done: t.done || cur.steps + 1 >= 400,
        };
        if (next.done && t.done) {
          setBestHuman((b) => (b === null ? next.ret : Math.max(b, next.ret)));
        }
        return next;
      });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [driving, env]);

  const V = learner.current?.valueFunction();
  const policy = learner.current?.greedyPolicy();
  const last = history[history.length - 1];

  const vRange = useMemo(() => {
    if (!V) return { lo: 0, hi: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, V[s]);
      hi = Math.max(hi, V[s]);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi: Number.isFinite(hi) ? hi : 1 };
    // `tick` forces a recompute as the tables change in place.
  }, [V, env, tick]);

  const curves = useMemo(() => {
    const stride = Math.max(1, Math.floor(history.length / 240));
    const returns = smooth(history.map((h) => h.totalReward), 20);
    const deltas = smooth(history.map((h) => h.meanAbsDelta), 20);
    return {
      returns: [
        {
          id: 'episode return (smoothed)',
          data: returns.map((y, x) => ({ x: x + 1, y })).filter((_, i) => i % stride === 0),
        },
      ],
      deltas: [
        {
          id: 'mean |δ|',
          data: deltas.map((y, x) => ({ x: x + 1, y })).filter((_, i) => i % stride === 0),
        },
      ],
    };
  }, [history]);

  return (
    <SimPanel
      title="Learning without a model"
      id="ch06-train-live"
      subtitle="Rusty has no map. Every value on the heatmap was bootstrapped from experience alone."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Algorithm"
              value={kind}
              onChange={(v) => setKind(v)}
              options={[
                { value: 'qlearning', label: 'Q-learning' },
                { value: 'sarsa', label: 'SARSA' },
                { value: 'expected-sarsa', label: 'Expected SARSA' },
                { value: 'double-q', label: 'Double Q' },
              ]}
            />
            <SimControls
              playing={playing}
              onPlayPause={() => setPlaying((p) => !p)}
              onStep={runEpisode}
              onReset={() => {
                setPlaying(false);
                rebuild();
              }}
              speed={speed}
              onSpeedChange={setSpeed}
            />
            <button
              type="button"
              onClick={() => {
                if (driving) {
                  setDriving(false);
                  setDrive(null);
                } else {
                  setPlaying(false);
                  startDrive();
                }
              }}
              className="rounded-md border px-2.5 py-1.5 text-[12px] font-medium transition-colors"
              style={{
                borderColor: driving ? 'var(--series-2)' : 'var(--border-hairline)',
                background: driving
                  ? 'color-mix(in srgb, var(--series-2) 12%, transparent)'
                  : 'transparent',
                color: driving ? 'var(--text-primary)' : 'var(--text-secondary)',
              }}
            >
              {driving ? 'Stop driving' : 'Drive it yourself'}
            </button>
          </div>
          {driving && (
            <p className="rounded-md border border-hairline bg-surface-sunken px-2.5 py-1.5 text-[12px] text-ink-secondary">
              Arrow keys or WASD. You get the same slippery floor and the same −10 shelves the
              agent does — and, like the agent at episode 1, no value function to consult.
            </p>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            <Slider
              label="Learning rate α"
              value={alpha}
              min={0.01}
              max={0.6}
              step={0.01}
              onChange={setAlpha}
              hint="too large and the estimates rattle; too small and they crawl"
            />
            <Slider
              label="Initial exploration ε"
              value={epsilon}
              min={0}
              max={0.6}
              step={0.02}
              onChange={setEpsilon}
              hint="decays ×0.995 per episode toward 0.02"
            />
          </div>
        </div>
      }
      caption="Try 'Drive it yourself' before training anything: no heatmap, no arrows, just the floor and the consequences — which is exactly Rusty's situation at episode 1. Then train the agent and watch it overtake your best run. Press play and watch value bleed backwards from the dock. Early on the arrows are nonsense because every Q is zero and ties break at random; the policy sharpens only where the TD error has actually reached. Set ε = 0 and the run usually stalls — with no exploration Rusty commits to the first corridor that ever paid off."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          {V && policy ? (
            <GridWorldCanvas
              env={env}
              V={driving ? undefined : V}
              policy={driving ? undefined : policy}
              path={driving ? drive?.path : last?.path}
              agentState={driving ? drive?.s : undefined}
              cellSize={32}
            />
          ) : null}
          <div className="mt-2">
            <ValueLegend min={vRange.lo} max={vRange.hi} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <StatTile label="Episodes" value={history.length} />
            <StatTile
              label="Last return"
              value={last?.totalReward ?? 0}
              status={last && last.totalReward > 0 ? 'good' : undefined}
              hint={last ? `${last.steps} steps` : 'not started'}
            />
            <StatTile
              label="Mean |δ| (last ep.)"
              value={last?.meanAbsDelta ?? 0}
              hint="magnitude of the learning signal"
            />
            <StatTile
              label="Current ε"
              value={last?.epsilon ?? epsilon}
              hint="exploration remaining"
            />
            {(driving || bestHuman !== null) && (
              <>
                <StatTile
                  label="Your return"
                  value={drive?.ret ?? 0}
                  status={drive?.done && drive.ret > 0 ? 'good' : undefined}
                  hint={
                    drive?.done
                      ? drive.s === env.goalState
                        ? `docked in ${drive.steps} steps`
                        : 'gave up'
                      : `${drive?.steps ?? 0} steps so far`
                  }
                />
                <StatTile
                  label="Your best vs agent"
                  value={
                    bestHuman !== null && last
                      ? `${bestHuman.toFixed(0)} vs ${last.totalReward.toFixed(0)}`
                      : bestHuman !== null
                        ? bestHuman.toFixed(0)
                        : '—'
                  }
                  mono={false}
                  hint={
                    bestHuman !== null && last && bestHuman >= last.totalReward
                      ? 'you are still ahead'
                      : 'the agent has caught up'
                  }
                />
              </>
            )}
          </div>
          <LineChart
            data={curves.returns}
            height={175}
            xLegend="episode"
            yLegend="return"
            caption="Smoothed episode return."
          />
          <LineChart
            data={curves.deltas}
            height={165}
            xLegend="episode"
            yLegend="mean |δ|"
            caption="The TD error decays as the value function becomes self-consistent."
          />
        </div>
      </div>
    </SimPanel>
  );
}
