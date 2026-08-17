'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Segmented } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';
import { BarChart } from '@/components/viz/BarChart';
import { mulberry32, gaussian } from '@/lib/rl/random';
import { cn } from '@/lib/utils';

type View = 'training' | 'rewards' | 'evaluation' | 'failures';

const FAILURES = [
  {
    mode: 'Stuck at a gap',
    count: 11,
    diagnosis:
      'The waypoint-progress term pulls toward a goal across a gap wider than Ferris can stride, so the policy oscillates at the edge rather than detouring.',
    fix: 'Pair every generated gap with a traversable bypass, so detouring is learned as acceptable — or add a lateral-detour skill (Ch 19).',
  },
  {
    mode: 'Push recovery on low friction',
    count: 7,
    diagnosis:
      'Randomization covers μ ∈ [0.4, 1.2] and pushes are sampled uniformly in time, so the low-friction-plus-push corner is rare in training.',
    fix: 'Stratify the randomization so hard parameter combinations are sampled deliberately rather than by chance.',
  },
  {
    mode: 'Timeout on rough terrain',
    count: 4,
    diagnosis:
      'On rough ground the effort penalty outweighs velocity tracking, so the policy correctly trades speed for economy — and misses the budget.',
    fix: 'Condition commanded velocity on remaining time, or raise the tracking weight (Ch 18 reward mixer).',
  },
  {
    mode: 'Distillation gap',
    count: 2,
    diagnosis:
      'A 15-step observation history is not enough to infer terrain the privileged teacher observed directly.',
    fix: 'Lengthen the history window, or add an explicit terrain-estimation auxiliary loss to the student.',
  },
];

const EVAL_ROWS = [
  { metric: 'Circuit completion', successes: 176, trials: 200 },
  { metric: 'No-fall rate', successes: 189, trials: 200 },
  { metric: 'Push recovery', successes: 171, trials: 200 },
];

/** Wilson score interval — correct near the boundary, unlike the normal approximation. */
function wilson(successes: number, trials: number, z = 1.96) {
  const p = successes / trials;
  const denom = 1 + (z * z) / trials;
  const centre = p + (z * z) / (2 * trials);
  const spread = z * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials));
  return { p, lo: (centre - spread) / denom, hi: (centre + spread) / denom };
}

/**
 * `ch22-mission-control` — the capstone's telemetry, as a working dashboard.
 *
 * Every panel answers a question the chapter asks: did training converge, which
 * reward terms are being traded against each other, does the evaluation support
 * the claim, and what exactly went wrong in the failures. The failure panel is
 * the one that matters most — each diagnosis names a design decision from
 * Parts III–IV rather than a hyperparameter.
 */
export function MissionControl() {
  const [view, setView] = useState<View>('training');
  const [seed, setSeed] = useState(0);

  const training = useMemo(() => {
    const rng = mulberry32(1000 + seed);
    const N = 220;
    const ret: Array<{ x: number; y: number }> = [];
    const terrain: Array<{ x: number; y: number }> = [];
    const kl: Array<{ x: number; y: number }> = [];
    let level = 0;

    for (let i = 0; i < N; i++) {
      const t = i / N;
      // Returns rise, then plateau as the curriculum keeps raising difficulty.
      const base = 48 * (1 - Math.exp(-4.2 * t)) - 6 * Math.max(0, t - 0.35);
      ret.push({ x: i * 50, y: base + gaussian(rng, 0, 1.6) });

      if (i > 25 && level < 9 && rng() < 0.05) level += 1;
      terrain.push({ x: i * 50, y: level });

      kl.push({ x: i * 50, y: Math.max(0.002, 0.017 - 0.011 * t + gaussian(rng, 0, 0.0018)) });
    }
    return { ret, terrain, kl };
  }, [seed]);

  const rewardBreakdown = useMemo(
    () => [
      { id: 'waypoint progress', value: 18.4 },
      { id: 'velocity tracking', value: 12.1 },
      { id: 'foot air time', value: 4.6 },
      { id: 'orientation', value: -3.2 },
      { id: 'effort', value: -2.7 },
      { id: 'foot slip', value: -1.4 },
      { id: 'action rate', value: -0.9 },
    ],
    [],
  );

  return (
    <SimPanel
      title="Mission control"
      id="ch22-mission-control"
      subtitle="The capstone's telemetry: training, reward composition, evaluation with intervals, and the failure post-mortems."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Segmented
            label="Panel"
            value={view}
            onChange={setView}
            options={[
              { value: 'training', label: 'Training' },
              { value: 'rewards', label: 'Reward terms' },
              { value: 'evaluation', label: 'Evaluation' },
              { value: 'failures', label: 'Failures' },
            ]}
          />
          {view === 'training' && (
            <div className="flex items-center gap-1.5">
              <span className="text-[11.5px] text-ink-secondary">Training seed:</span>
              {[0, 1, 2, 3, 4].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSeed(s)}
                  aria-pressed={seed === s}
                  className={cn(
                    'rounded border px-2 py-0.5 text-[11.5px] font-medium transition-colors',
                    seed === s
                      ? 'border-series-1 bg-series-1 text-white'
                      : 'border-hairline text-ink-secondary hover:bg-surface-sunken',
                  )}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>
      }
      caption="Switch between training seeds and watch the curves differ — that spread is why §22.4 reports five seeds rather than the best one. The failures panel is the chapter's real payload: every diagnosis names a design decision from Parts III–IV, and not one of them is 'tune the learning rate'."
    >
      {view === 'training' && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            <StatTile label="Environment steps" value="11.0M" mono hint="4096 envs × 2700 iters" />
            <StatTile label="Final return" value={training.ret[training.ret.length - 1].y} hint="mean over envs" />
            <StatTile
              label="Terrain level"
              value={`${training.terrain[training.terrain.length - 1].y} / 9`}
              mono={false}
              hint="curriculum progression"
            />
            <StatTile
              label="Approx. KL"
              value={training.kl[training.kl.length - 1].y}
              status={training.kl[training.kl.length - 1].y < 0.02 ? 'good' : 'warning'}
              hint="PPO trust-region watchdog"
            />
          </div>
          <LineChart
            data={[{ id: 'episode return', data: training.ret }]}
            height={210}
            xLegend="thousands of environment steps"
            yLegend="return"
            caption="The plateau is not convergence — the terrain curriculum keeps raising difficulty, so a flat return means competence rising as fast as the task hardens."
          />
          <div className="grid gap-3 lg:grid-cols-2">
            <LineChart
              data={[{ id: 'terrain level', data: training.terrain }]}
              height={175}
              xLegend="steps (thousands)"
              yLegend="difficulty level"
            />
            <LineChart
              data={[{ id: 'approximate KL', data: training.kl }]}
              height={175}
              xLegend="steps (thousands)"
              yLegend="KL"
            />
          </div>
        </div>
      )}

      {view === 'rewards' && (
        <div className="space-y-3">
          <BarChart
            data={rewardBreakdown}
            layout="horizontal"
            height={250}
            xLegend="mean contribution per episode"
            title="Where the return actually comes from"
            table={{
              columns: ['Term', 'Mean contribution'],
              rows: rewardBreakdown.map((r) => [String(r.id), r.value]),
            }}
          />
          <p className="rounded-lg border border-hairline bg-surface-sunken px-3 py-2.5 text-[12.5px] leading-relaxed text-ink-secondary">
            Logging terms separately is what makes reward debugging tractable. When
            the timeout failures appear, this panel shows the effort penalty
            outweighing velocity tracking on rough terrain — a diagnosis you cannot
            reach from the scalar return alone.
          </p>
        </div>
      )}

      {view === 'evaluation' && (
        <div className="space-y-3">
          <div className="overflow-hidden rounded-lg border border-hairline">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="bg-surface-sunken">
                  <th className="px-3 py-2 text-left font-semibold text-ink">Metric</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">Result</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">Rate</th>
                  <th className="px-3 py-2 text-right font-semibold text-ink">95% Wilson interval</th>
                </tr>
              </thead>
              <tbody>
                {EVAL_ROWS.map((r) => {
                  const w = wilson(r.successes, r.trials);
                  return (
                    <tr key={r.metric} className="border-t border-hairline">
                      <td className="px-3 py-2 text-ink-secondary">{r.metric}</td>
                      <td className="tabular px-3 py-2 text-right text-ink-muted">
                        {r.successes} / {r.trials}
                      </td>
                      <td className="tabular px-3 py-2 text-right font-semibold text-ink">
                        {(w.p * 100).toFixed(1)}%
                      </td>
                      <td className="tabular px-3 py-2 text-right text-ink-secondary">
                        [{(w.lo * 100).toFixed(1)}%, {(w.hi * 100).toFixed(1)}%]
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <StatTile label="Cost of transport" value={0.94} hint="±0.11 across 5 seeds" />
            <StatTile label="Median completion" value="108 s" mono={false} hint="budget 180 s" />
            <StatTile
              label="Success level"
              value="L0"
              mono={false}
              status="warning"
              hint="simulation only — no hardware"
            />
          </div>
          <p className="rounded-lg border border-hairline bg-surface-sunken px-3 py-2.5 text-[12.5px] leading-relaxed text-ink-secondary">
            Completion ranged from 81% to 91% across the five training seeds. A
            paper reporting 91% from a single run would be reporting the best of
            five — which is why the interval and the spread both appear here.
          </p>
        </div>
      )}

      {view === 'failures' && (
        <div className="space-y-2">
          {FAILURES.map((f) => (
            <div key={f.mode} className="rounded-lg border border-hairline bg-surface px-3.5 py-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h5 className="text-[14px] font-semibold text-ink">{f.mode}</h5>
                <span className="tabular rounded bg-surface-sunken px-2 py-0.5 text-[11.5px] text-ink-secondary">
                  {f.count} of 24 failures
                </span>
              </div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-secondary">
                <span className="font-semibold text-ink">Diagnosis. </span>
                {f.diagnosis}
              </p>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-secondary">
                <span className="font-semibold text-ink">Fix. </span>
                {f.fix}
              </p>
            </div>
          ))}
        </div>
      )}
    </SimPanel>
  );
}
