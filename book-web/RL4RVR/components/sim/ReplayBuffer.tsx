'use client';

import { useMemo } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { useDebounced, useSimulation, useWidgetState } from '@/lib/sim/useSimulation';
import { smooth } from '@/lib/rl/td';

interface Ablation {
  returns: number[];
  qMax: number[];
  tdError: number[];
}
type Result = Record<'both' | 'noReplay' | 'noTarget' | 'neither', Ablation>;

const CONFIGS = [
  { key: 'both', label: 'replay + target' },
  { key: 'noReplay', label: 'no replay' },
  { key: 'noTarget', label: 'no target network' },
  { key: 'neither', label: 'neither' },
] as const;

const DEFAULTS = {
  bufferSize: 2000,
  syncEvery: 100,
  alpha: 0.35,
  show: 'returns' as string,
};

/**
 * `ch09-replay-target` — the two pieces of surgery, measured.
 *
 * Four agents actually train on Rusty's warehouse, differing only in whether
 * they sample from a replay buffer and whether they bootstrap from a frozen
 * target. The learning rate is deliberately aggressive, because the failures
 * this chapter describes are the ones that appear when you push a method
 * slightly past where it is comfortable.
 */
export function ReplayBuffer() {
  const [state, set, reset] = useWidgetState('ch09-replay-target', DEFAULTS);
  const debounced = useDebounced(state, 400);

  const { data, running, error } = useSimulation<Result>('replay-ablation', {
    episodes: 260,
    bufferSize: debounced.bufferSize,
    syncEvery: debounced.syncEvery,
    alpha: debounced.alpha,
    seeds: 3,
  });

  const metric = state.show as 'returns' | 'qMax' | 'tdError';

  const series = useMemo(() => {
    if (!data) return [];
    return CONFIGS.map((c) => {
      const raw = data[c.key][metric];
      const sm = smooth(raw, metric === 'returns' ? 18 : 8);
      const stride = Math.max(1, Math.floor(sm.length / 180));
      return {
        id: c.label,
        data: sm
          .map((y, x) => ({ x: x + 1, y }))
          .filter((_, i) => i % stride === 0),
      };
    });
  }, [data, metric]);

  // Final-quarter averages, which is where the differences are legible.
  const summary = useMemo(() => {
    if (!data) return [];
    return CONFIGS.map((c) => {
      const r = data[c.key].returns;
      const tail = r.slice(Math.floor(r.length * 0.75));
      const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
      const varr = tail.reduce((a, b) => a + (b - mean) ** 2, 0) / tail.length;
      return { label: c.label, mean, sd: Math.sqrt(varr) };
    });
  }, [data]);

  const best = summary.length ? Math.max(...summary.map((s) => s.mean)) : 0;

  return (
    <SimPanel
      title="Replay and target networks as variance surgery"
      id="ch09-replay-target"
      subtitle="Four Q-learning agents trained on the warehouse, averaged over three seeds. The only differences are replay and the target network."
      controls={
        <div className="space-y-2.5">
          <div className="grid gap-3 sm:grid-cols-3">
            <Slider
              label="Replay buffer size"
              value={state.bufferSize}
              min={200}
              max={20000}
              step={200}
              onChange={(v) => set({ bufferSize: v })}
              format={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0))}
              hint="larger ⇒ less correlated batches"
            />
            <Slider
              label="Target sync interval"
              value={state.syncEvery}
              min={1}
              max={500}
              step={1}
              onChange={(v) => set({ syncEvery: v })}
              format={(v) => `${v.toFixed(0)} steps`}
              hint="longer ⇒ steadier, slower to track"
            />
            <Slider
              label="Learning rate α"
              value={state.alpha}
              min={0.05}
              max={0.8}
              step={0.05}
              onChange={(v) => set({ alpha: v })}
              hint="raise it to expose the instability"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {(
              [
                ['returns', 'Episode return'],
                ['qMax', 'max Q at start state'],
                ['tdError', 'mean |δ|'],
              ] as const
            ).map(([k, label]) => (
              <button
                key={k}
                type="button"
                onClick={() => set({ show: k })}
                aria-pressed={metric === k}
                className={`rounded-md border px-2 py-1 text-[11.5px] font-medium transition-colors ${
                  metric === k
                    ? 'border-series-1 bg-series-1 text-white'
                    : 'border-hairline text-ink-secondary hover:bg-surface-sunken'
                }`}
              >
                {label}
              </button>
            ))}
            <button
              type="button"
              onClick={reset}
              className="ml-auto rounded-md border border-hairline px-2.5 py-1 text-[11.5px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              Reset
            </button>
          </div>
        </div>
      }
      caption="Switch to 'max Q at start state' and watch what the target network is for: without it, the estimate chases itself upward — you are regressing toward a quantity that moves every time you update. Turn the learning rate up and the effect arrives sooner. Shrink the buffer toward 200 and the no-replay and replay curves converge, because a small buffer is nearly as correlated as no buffer at all."
    >
      {error ? (
        <p className="rounded-lg border border-hairline px-3 py-2 text-[12.5px] text-status-critical">
          Simulation failed: {error}
        </p>
      ) : (
        <div className="space-y-3">
          <LineChart
            data={series}
            height={260}
            xLegend="episode"
            yLegend={
              metric === 'returns'
                ? 'return (smoothed)'
                : metric === 'qMax'
                  ? 'max Q(s₀, ·)'
                  : 'mean |δ|'
            }
            caption={
              running
                ? 'Training four agents…'
                : 'Each curve is a mean over three seeds; the only difference between them is the surgery.'
            }
          />
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
            {summary.map((s) => (
              <StatTile
                key={s.label}
                label={s.label}
                value={s.mean}
                hint={`final-quarter return · sd ${s.sd.toFixed(1)}`}
                status={
                  s.mean >= best - 1 ? 'good' : s.mean < best - 12 ? 'critical' : 'warning'
                }
              />
            ))}
          </div>
        </div>
      )}
    </SimPanel>
  );
}
