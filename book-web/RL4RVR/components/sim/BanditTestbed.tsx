'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import {
  EpsilonGreedy,
  GradientBandit,
  ThompsonSampling,
  Ucb1,
  makeBandit,
  pull,
  runBanditExperiment,
  type BanditPolicy,
} from '@/lib/rl/bandits';
import { mulberry32, type Rng } from '@/lib/rl/random';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider, Segmented } from './SimControls';
import { seriesColor } from '@/lib/theme';
import { useTheme } from '@/components/layout/ThemeProvider';
import { cn } from '@/lib/utils';

interface Algo {
  key: string;
  label: string;
  make: (k: number, eps: number) => BanditPolicy;
}

const ALGOS: Algo[] = [
  { key: 'greedy', label: 'greedy (ε=0)', make: (k) => new EpsilonGreedy(k, 0) },
  { key: 'eps', label: 'ε-greedy', make: (k, eps) => new EpsilonGreedy(k, eps) },
  {
    key: 'optimistic',
    label: 'optimistic greedy (Q₁=5)',
    make: (k) => new EpsilonGreedy(k, 0, 0.1, 5),
  },
  { key: 'ucb', label: 'UCB (c=2)', make: (k) => new Ucb1(k, 2) },
  { key: 'gradient', label: 'gradient bandit', make: (k) => new GradientBandit(k, 0.1) },
  { key: 'thompson', label: 'Thompson sampling', make: (k) => new ThompsonSampling(k) },
];

const K = 10;

/** One live arm, as the reader sees it: their own estimate and pull count. */
interface ArmView {
  pulls: number;
  mean: number;
  lastReward: number | null;
}

/**
 * `ch03-bandit-testbed` — the 10-armed testbed, with a seat at the table.
 *
 * The chapter argues that exploration is a decision with a price. Reading that
 * is not the same as paying it, so this widget puts the reader in the loop:
 * you pull the arms yourself, one at a time, against the same problem the
 * algorithms face, and your regret is plotted beside theirs.
 *
 * Almost everyone plays the same way — a few exploratory pulls, then commit —
 * and almost everyone loses to UCB, usually because they committed to an arm
 * that got lucky in its first two samples. That experience is the argument.
 */
export function BanditTestbed() {
  const { mode } = useTheme();
  const [mode_, setMode_] = useState<'watch' | 'play'>('play');
  const [epsilon, setEpsilon] = useState(0.1);
  const [runs, setRuns] = useState(120);
  const [selected, setSelected] = useState<string[]>(['eps', 'ucb']);

  // ---- The reader's own game -------------------------------------------------
  const [seed, setSeed] = useState(7);
  const problem = useMemo(() => makeBandit(mulberry32(seed), K), [seed]);
  const rngRef = useRef<Rng>(mulberry32(seed * 31 + 5));

  const [arms, setArms] = useState<ArmView[]>(() =>
    Array.from({ length: K }, () => ({ pulls: 0, mean: 0, lastReward: null })),
  );
  const [history, setHistory] = useState<{ reward: number; regret: number; arm: number }[]>([]);
  // Shadow algorithms play the identical problem, one pull per pull of yours.
  const shadowRef = useRef<{ ucb: BanditPolicy; eps: BanditPolicy } | null>(null);
  const [shadow, setShadow] = useState<{ ucb: number[]; eps: number[] }>({ ucb: [], eps: [] });

  const resetGame = useCallback(
    (newSeed?: number) => {
      const s = newSeed ?? seed;
      if (newSeed !== undefined) setSeed(newSeed);
      rngRef.current = mulberry32(s * 31 + 5);
      shadowRef.current = { ucb: new Ucb1(K, 2), eps: new EpsilonGreedy(K, 0.1) };
      setArms(Array.from({ length: K }, () => ({ pulls: 0, mean: 0, lastReward: null })));
      setHistory([]);
      setShadow({ ucb: [], eps: [] });
    },
    [seed],
  );

  // Lazily initialise the shadows on first render.
  if (!shadowRef.current) {
    shadowRef.current = { ucb: new Ucb1(K, 2), eps: new EpsilonGreedy(K, 0.1) };
  }

  const best = problem.qStar[problem.optimalAction];

  const pullArm = useCallback(
    (a: number) => {
      const rng = rngRef.current;
      const r = pull(problem, a, rng);

      setArms((prev) => {
        const next = [...prev];
        const n = next[a].pulls + 1;
        next[a] = { pulls: n, mean: next[a].mean + (r - next[a].mean) / n, lastReward: r };
        return next;
      });

      setHistory((prev) => {
        const prevRegret = prev.length ? prev[prev.length - 1].regret : 0;
        return [...prev, { reward: r, regret: prevRegret + (best - problem.qStar[a]), arm: a }];
      });

      // Advance the shadow algorithms on the same problem and the same clock.
      const sh = shadowRef.current!;
      setShadow((prev) => {
        const step = prev.ucb.length + 1;
        const out = { ucb: [...prev.ucb], eps: [...prev.eps] };
        for (const [key, pol] of [['ucb', sh.ucb], ['eps', sh.eps]] as const) {
          const choice = pol.selectAction(step, rng);
          const rr = pull(problem, choice, rng);
          pol.update(choice, rr);
          const last = prev[key].length ? prev[key][prev[key].length - 1] : 0;
          out[key].push(last + (best - problem.qStar[choice]));
        }
        return out;
      });
    },
    [problem, best],
  );

  const yourRegret = history.length ? history[history.length - 1].regret : 0;
  const ucbRegret = shadow.ucb.length ? shadow.ucb[shadow.ucb.length - 1] : 0;
  const epsRegret = shadow.eps.length ? shadow.eps[shadow.eps.length - 1] : 0;
  const totalReward = history.reduce((a, h) => a + h.reward, 0);
  const optimalPulls = history.filter((h) => h.arm === problem.optimalAction).length;

  const raceSeries = useMemo(() => {
    if (!history.length) return [];
    return [
      { id: 'you', data: history.map((h, i) => ({ x: i + 1, y: h.regret })) },
      { id: 'UCB (c=2)', data: shadow.ucb.map((y, i) => ({ x: i + 1, y })) },
      { id: 'ε-greedy (0.1)', data: shadow.eps.map((y, i) => ({ x: i + 1, y })) },
    ];
  }, [history, shadow]);

  // ---- Watch mode: the classic averaged testbed -------------------------------
  const results = useMemo(() => {
    if (mode_ !== 'watch') return [];
    return ALGOS.filter((a) => selected.includes(a.key)).map((a) =>
      runBanditExperiment((k) => a.make(k, epsilon), {
        runs,
        steps: 1000,
        k: K,
        seed: 42,
        rngFactory: mulberry32,
      }),
    );
  }, [selected, epsilon, runs, mode_]);

  const labels = useMemo(
    () => ALGOS.filter((a) => selected.includes(a.key)).map((a) => a.label),
    [selected],
  );
  const thin = <T,>(xs: T[]) => xs.filter((_, i) => i % 5 === 0);

  return (
    <SimPanel
      title="The 10-armed testbed"
      id="ch03-bandit-testbed"
      subtitle={
        mode_ === 'play'
          ? 'You pull the arms. UCB and ε-greedy play the same problem beside you, one pull for one pull.'
          : 'Each curve averages independent runs on fresh problems with q*(a) ~ N(0,1), rewards ~ N(q*(a), 1).'
      }
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Mode"
              value={mode_}
              onChange={(v) => setMode_(v)}
              options={[
                { value: 'play', label: 'Play it yourself' },
                { value: 'watch', label: 'Watch the algorithms' },
              ]}
            />
            {mode_ === 'play' ? (
              <>
                <button
                  type="button"
                  onClick={() => resetGame(Math.floor(Math.random() * 99999))}
                  className="rounded-md bg-series-1 px-2.5 py-1.5 text-[12px] font-medium text-white transition-opacity hover:opacity-90"
                >
                  New problem
                </button>
                <button
                  type="button"
                  onClick={() => resetGame()}
                  className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
                >
                  Replay this one
                </button>
              </>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {ALGOS.map((a) => {
                  const on = selected.includes(a.key);
                  return (
                    <button
                      key={a.key}
                      type="button"
                      onClick={() =>
                        setSelected((s) =>
                          s.includes(a.key) ? s.filter((x) => x !== a.key) : [...s, a.key],
                        )
                      }
                      aria-pressed={on}
                      className="flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11.5px] font-medium transition-colors"
                      style={{
                        borderColor: on
                          ? seriesColor(selected.indexOf(a.key), mode)
                          : 'var(--border-hairline)',
                        background: on
                          ? `color-mix(in srgb, ${seriesColor(selected.indexOf(a.key), mode)} 12%, transparent)`
                          : 'transparent',
                        color: on ? 'var(--text-primary)' : 'var(--text-secondary)',
                      }}
                    >
                      <span
                        aria-hidden
                        className="inline-block h-2 w-2 rounded-full"
                        style={{
                          background: on
                            ? seriesColor(selected.indexOf(a.key), mode)
                            : 'var(--text-muted)',
                        }}
                      />
                      {a.label}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          {mode_ === 'watch' && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Slider
                label="ε (for ε-greedy)"
                value={epsilon}
                min={0}
                max={0.5}
                step={0.01}
                onChange={setEpsilon}
                hint="constant ε ⇒ linear regret forever"
              />
              <Slider
                label="Independent runs"
                value={runs}
                min={20}
                max={400}
                step={20}
                onChange={setRuns}
                format={(v) => v.toFixed(0)}
                hint="more runs ⇒ smoother averages"
              />
            </div>
          )}
        </div>
      }
      caption={
        mode_ === 'play'
          ? "Notice what you do: a few pulls to look around, then commitment. That is greedy-with-a-short-exploration-phase, and its failure mode is specific — you commit to whichever arm got lucky in its first two samples, and you never find out what you missed, because you stopped sampling the alternatives. UCB does not have that failure mode: its bonus keeps rarely-tried arms in contention until their uncertainty actually shrinks. Play twice on the same problem with 'Replay this one' and you will usually do better the second time, which is exactly the information the algorithms have to earn."
          : 'Watch the greedy curve plateau below the others: with ε = 0 it locks onto whichever arm looked good first. Raise ε and early reward drops but the percentage-optimal keeps climbing — exploration paid for now, repaid later. UCB’s regret curve bends toward logarithmic; ε-greedy’s stays a straight line, as Proposition 3.1 predicts.'
      }
    >
      {mode_ === 'play' ? (
        <div className="space-y-3">
          {/* The arms */}
          <div className="grid grid-cols-5 gap-2 sm:grid-cols-10">
            {arms.map((arm, a) => {
              const isOptimal = a === problem.optimalAction;
              const revealed = history.length >= 40;
              return (
                <button
                  key={a}
                  type="button"
                  onClick={() => pullArm(a)}
                  className={cn(
                    'group flex flex-col items-center rounded-lg border px-1 py-2 transition-all',
                    'hover:-translate-y-0.5 hover:border-series-1 active:translate-y-0',
                    arm.pulls > 0 ? 'border-hairline bg-surface' : 'border-dashed border-baseline',
                  )}
                  style={
                    revealed && isOptimal
                      ? { borderColor: 'var(--status-good)', borderWidth: 2 }
                      : undefined
                  }
                  aria-label={`Pull arm ${a + 1}, pulled ${arm.pulls} times`}
                >
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-ink-muted">
                    {a + 1}
                  </span>
                  {/* Estimate bar: height is the running mean */}
                  <span className="relative my-1 flex h-12 w-full items-end justify-center">
                    <span
                      className="w-3 rounded-t-sm transition-all"
                      style={{
                        height: `${Math.max(2, Math.min(48, 24 + arm.mean * 12))}px`,
                        background:
                          arm.pulls === 0
                            ? 'var(--gridline)'
                            : arm.mean >= 0
                              ? seriesColor(0, mode)
                              : seriesColor(1, mode),
                      }}
                    />
                  </span>
                  <span className="tabular text-[11px] font-semibold text-ink">
                    {arm.pulls ? arm.mean.toFixed(2) : '—'}
                  </span>
                  <span className="tabular text-[9.5px] text-ink-muted">
                    {arm.pulls} pull{arm.pulls === 1 ? '' : 's'}
                  </span>
                  {arm.lastReward !== null && (
                    <span
                      className="tabular text-[9.5px]"
                      style={{ color: arm.lastReward >= 0 ? 'var(--status-good)' : 'var(--status-critical)' }}
                    >
                      {arm.lastReward >= 0 ? '+' : ''}
                      {arm.lastReward.toFixed(1)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <p className="text-[11.5px] text-ink-muted">
            Click an arm to pull it. Bars show <em>your</em> running estimate — not the truth. After
            40 pulls the genuinely best arm is outlined in green.
          </p>

          <div className="grid gap-3 lg:grid-cols-[1fr,215px]">
            {history.length > 1 ? (
              <LineChart
                data={raceSeries}
                height={230}
                xLegend="pulls"
                yLegend="cumulative regret"
                caption="Lower is better. Regret is what you gave up by not pulling the best arm every time."
              />
            ) : (
              <div className="grid h-[230px] place-items-center rounded-lg border border-dashed border-baseline text-[12.5px] text-ink-muted">
                Pull a few arms to start the race.
              </div>
            )}
            <div className="space-y-2">
              <StatTile
                label="Your regret"
                value={yourRegret}
                status={
                  history.length < 5
                    ? undefined
                    : yourRegret <= ucbRegret
                      ? 'good'
                      : yourRegret > ucbRegret * 1.5
                        ? 'critical'
                        : 'warning'
                }
                hint={
                  history.length < 5
                    ? 'keep pulling'
                    : yourRegret <= ucbRegret
                      ? 'you are beating UCB'
                      : `UCB is ahead by ${(yourRegret - ucbRegret).toFixed(1)}`
                }
              />
              <StatTile label="UCB regret" value={ucbRegret} hint="same problem, same pulls" />
              <StatTile label="ε-greedy regret" value={epsRegret} hint="ε = 0.1" />
              <StatTile
                label="Your total reward"
                value={totalReward}
                hint={`${history.length} pull${history.length === 1 ? '' : 's'}`}
              />
              <StatTile
                label="% optimal"
                value={history.length ? (100 * optimalPulls) / history.length : 0}
                unit="%"
                hint="how often you hit the best arm"
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <LineChart
            data={results.map((r, i) => ({
              id: labels[i],
              data: thin(r.avgReward.map((y, x) => ({ x: x + 1, y }))),
            }))}
            height={230}
            xLegend="steps"
            yLegend="average reward"
            caption="Average reward per step."
          />
          <div className="grid gap-3 lg:grid-cols-2">
            <LineChart
              data={results.map((r, i) => ({
                id: labels[i],
                data: thin(r.optimalPct.map((y, x) => ({ x: x + 1, y }))),
              }))}
              height={210}
              xLegend="steps"
              yLegend="% optimal action"
              yMin={0}
              yMax={100}
              caption="How often the best arm was chosen."
            />
            <LineChart
              data={results.map((r, i) => ({
                id: labels[i],
                data: thin(r.regret.map((y, x) => ({ x: x + 1, y }))),
              }))}
              height={210}
              xLegend="steps"
              yLegend="cumulative regret"
              caption="The quantity UCB’s bound controls."
            />
          </div>
        </div>
      )}
    </SimPanel>
  );
}
