'use client';

import { useMemo } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';
import { useDebounced, useSimulation, useWidgetState } from '@/lib/sim/useSimulation';

interface Result {
  trajectories: number[][]; // per member, theta over horizon
  truth: number[];
  spread: number[];
  meanError: number[];
  trainLoss: number;
  samples: number;
}

const DEFAULTS = { episodes: 14, members: 5, horizon: 30, epochs: 45 };

/**
 * `ch12-imagination-fan` — a genuinely learned ensemble, diverging.
 *
 * Five neural dynamics models are trained on transitions actually collected
 * from Pendle under random torques, each on its own bootstrap resample. They
 * are then rolled forward from a shared state under a shared action sequence,
 * and compared against the true dynamics. The fan is measured disagreement:
 * collect fewer episodes and it widens, because the models genuinely know less.
 */
export function ModelBiasFan() {
  const { mode } = useTheme();
  const [state, set, reset] = useWidgetState('ch12-imagination-fan', DEFAULTS);
  const debounced = useDebounced(state, 380);

  const { data, running, error } = useSimulation<Result>('dynamics-ensemble', {
    episodes: debounced.episodes,
    members: debounced.members,
    horizon: debounced.horizon,
    epochs: debounced.epochs,
  });

  const trustHorizon = useMemo(() => {
    if (!data) return -1;
    return data.spread.findIndex((s) => s > 0.35);
  }, [data]);

  const W = 430;
  const H = 200;
  const geom = useMemo(() => {
    if (!data) return null;
    const all = [...data.truth, ...data.trajectories.flat()].filter(Number.isFinite);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const pad = (hi - lo) * 0.1 || 0.5;
    const n = data.truth.length;
    return {
      sx: (i: number) => 32 + (i / Math.max(1, n - 1)) * (W - 48),
      sy: (v: number) => H - 26 - ((v - lo + pad) / (hi - lo + 2 * pad)) * (H - 48),
    };
  }, [data]);

  return (
    <SimPanel
      title="Imagination diverges from reality"
      id="ch12-imagination-fan"
      subtitle="An ensemble of neural dynamics models, trained on real Pendle transitions, rolled forward against the truth."
      controls={
        <div className="space-y-2.5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Slider
              label="Training episodes"
              value={state.episodes}
              min={3}
              max={40}
              step={1}
              onChange={(v) => set({ episodes: v })}
              format={(v) => v.toFixed(0)}
              hint="less data ⇒ a wider fan"
            />
            <Slider
              label="Ensemble members"
              value={state.members}
              min={2}
              max={8}
              step={1}
              onChange={(v) => set({ members: v })}
              format={(v) => v.toFixed(0)}
            />
            <Slider
              label="Rollout horizon"
              value={state.horizon}
              min={8}
              max={60}
              step={1}
              onChange={(v) => set({ horizon: v })}
              format={(v) => `${v.toFixed(0)} steps`}
            />
            <Slider
              label="Training epochs"
              value={state.epochs}
              min={10}
              max={120}
              step={5}
              onChange={(v) => set({ epochs: v })}
              format={(v) => v.toFixed(0)}
              hint="how well each member fits"
            />
          </div>
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-hairline px-2.5 py-1 text-[11.5px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
          >
            Reset
          </button>
        </div>
      }
      caption="Every member has small one-step error, and every member is wrong by the end. That is compounding: each prediction becomes the next input, so the rollout walks off the data it was trained on. Cut the training episodes to three and the fan opens almost immediately — the disagreement is the ensemble telling you how far it can be trusted, which is exactly the signal MBPO uses to choose a branch length."
    >
      {error ? (
        <p className="rounded-lg border border-hairline px-3 py-2 text-[12.5px] text-status-critical">
          Simulation failed: {error}
        </p>
      ) : (
        <div className="grid gap-3 lg:grid-cols-[1fr,200px]">
          <div>
            <svg
              width={W}
              height={H}
              viewBox={`0 0 ${W} ${H}`}
              className="max-w-full rounded-lg"
              style={{ background: 'var(--surface-sunken)' }}
              role="img"
              aria-label="Ensemble rollouts fanning out from a shared state against the true trajectory"
            >
              {data && geom ? (
                <>
                  {data.trajectories.map((traj, i) => (
                    <path
                      key={i}
                      d={`M${traj
                        .map((v, k) => `${geom.sx(k).toFixed(1)},${geom.sy(v).toFixed(1)}`)
                        .join(' L')}`}
                      fill="none"
                      stroke={seriesColor(0, mode)}
                      strokeWidth={1.4}
                      opacity={0.45}
                    />
                  ))}
                  <path
                    d={`M${data.truth
                      .map((v, k) => `${geom.sx(k).toFixed(1)},${geom.sy(v).toFixed(1)}`)
                      .join(' L')}`}
                    fill="none"
                    stroke={seriesColor(1, mode)}
                    strokeWidth={2.6}
                  />
                  {trustHorizon > 0 && (
                    <>
                      <line
                        x1={geom.sx(trustHorizon)}
                        y1={12}
                        x2={geom.sx(trustHorizon)}
                        y2={H - 24}
                        stroke="var(--status-warning)"
                        strokeWidth={1.5}
                        strokeDasharray="4 3"
                      />
                      <text
                        x={geom.sx(trustHorizon) + 4}
                        y={22}
                        fontSize={9.5}
                        fill="var(--status-warning)"
                      >
                        trust ends
                      </text>
                    </>
                  )}
                  <text x={32} y={H - 8} fontSize={9.5} fill="var(--text-muted)">
                    thin: ensemble members · thick: true dynamics · axis: pole angle θ
                  </text>
                </>
              ) : (
                <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={12} fill="var(--text-muted)">
                  {running ? 'collecting data and training the ensemble…' : 'no data'}
                </text>
              )}
            </svg>

            {data && (
              <LineChart
                data={[
                  {
                    id: 'ensemble disagreement',
                    data: data.spread.map((y, x) => ({ x, y })),
                  },
                  {
                    id: 'error against the truth',
                    data: data.meanError.map((y, x) => ({ x, y })),
                  },
                ]}
                height={185}
                xLegend="rollout horizon"
                yLegend="radians"
                dashed={['error against the truth']}
                caption="Disagreement tracks true error — which is why it works as a proxy when the truth is unavailable."
              />
            )}
          </div>

          <div className="space-y-2">
            <StatTile
              label="Training samples"
              value={data?.samples ?? 0}
              hint={`${debounced.episodes} episodes on Pendle`}
            />
            <StatTile
              label="One-step fit loss"
              value={data?.trainLoss ?? NaN}
              hint="normalized delta MSE"
              status={data && data.trainLoss < 0.15 ? 'good' : 'warning'}
            />
            <StatTile
              label="Usable horizon"
              value={
                trustHorizon > 0 ? `${trustHorizon} steps` : `> ${debounced.horizon} steps`
              }
              mono={false}
              hint="before disagreement exceeds 0.35 rad"
              status={trustHorizon > 0 && trustHorizon < 8 ? 'critical' : 'good'}
            />
            <StatTile
              label="Final disagreement"
              value={data?.spread[data.spread.length - 1] ?? 0}
              unit="rad"
              status={
                (data?.spread[data.spread.length - 1] ?? 0) > 1 ? 'critical' : 'warning'
              }
            />
          </div>
        </div>
      )}
    </SimPanel>
  );
}
