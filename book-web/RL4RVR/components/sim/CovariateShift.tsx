'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';
import { useDebounced, useSimulation, useWidgetState } from '@/lib/sim/useSimulation';

interface Result {
  clonePath: Array<{ x: number; y: number }>;
  expertPath: Array<{ x: number; y: number }>;
  deviations: number[];
  finalDeviation: number;
  meanDeviation: number;
  datasetSize: number;
  trainLoss: number;
}

const DEFAULTS = { demos: 12, horizon: 220, noise: 0.06, dagger: false };

/**
 * `ch16-covariate-drift` — behaviour cloning that actually fails.
 *
 * A neural network is fitted to state–action pairs from an expert following a
 * lane, then rolled out on its own. The drift is not drawn: it is what the
 * trained policy does when its own small errors carry it into states the
 * demonstrations never covered. Switching on DAgger relabels the states the
 * learner visits, and the same network stops drifting.
 *
 * The reader can also draw their own demonstrations, in which case the clone
 * is fitted to those instead — and sparse or shaky demonstrations produce
 * exactly the failure the theory predicts.
 */
export function CovariateShift() {
  const { mode } = useTheme();
  const [state, set, reset] = useWidgetState('ch16-covariate-drift', DEFAULTS);
  const [userDemos, setUserDemos] = useState<Array<{ x: number; y: number }>>([]);
  const [drawing, setDrawing] = useState(false);
  const debounced = useDebounced(state, 400);

  const { data, running, error } = useSimulation<Result>('behaviour-cloning', {
    demos: debounced.demos,
    horizon: debounced.horizon,
    noise: debounced.noise,
    dagger: debounced.dagger,
    daggerRounds: 4,
    userDemos: userDemos.length > 4 ? userDemos : undefined,
  });

  const W = 520;
  const H = 210;
  const sx = (x: number) => 10 + x * (W - 20);
  const sy = (y: number) => H - 14 - y * (H - 28);

  const handlePointer = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!drawing) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * (W / (W - 20)) - 10 / (W - 20);
    const y = 1 - ((e.clientY - r.top) / r.height);
    if (x >= 0 && x <= 1) {
      setUserDemos((d) => (d.length && x < d[d.length - 1].x ? d : [...d, { x, y }]));
    }
  };

  const boundCurves = useMemo(() => {
    const eps = debounced.noise;
    const ts = Array.from({ length: 30 }, (_, i) => (i + 1) * 10);
    return [
      { id: 'cloning: O(εT²)', data: ts.map((t) => ({ x: t, y: eps * t * t * 0.001 })) },
      { id: 'DAgger: O(εT)', data: ts.map((t) => ({ x: t, y: eps * t * 0.02 })) },
    ];
  }, [debounced.noise]);

  return (
    <SimPanel
      title="The cloned robot drifts"
      id="ch16-covariate-drift"
      subtitle="A network is fitted to the expert's state–action pairs, then rolled out on its own. Everything below is the trained policy's real behaviour."
      controls={
        <div className="space-y-2.5">
          <div className="grid gap-3 sm:grid-cols-3">
            <Slider
              label="Demonstrations"
              value={state.demos}
              min={2}
              max={40}
              step={1}
              onChange={(v) => set({ demos: v })}
              format={(v) => v.toFixed(0)}
              hint={userDemos.length > 4 ? 'ignored — using yours' : 'expert rollouts collected'}
            />
            <Slider
              label="Expert action noise ε"
              value={state.noise}
              min={0}
              max={0.3}
              step={0.01}
              onChange={(v) => set({ noise: v })}
              hint="imperfect demonstrations"
            />
            <Slider
              label="Horizon T"
              value={state.horizon}
              min={60}
              max={400}
              step={20}
              onChange={(v) => set({ horizon: v })}
              format={(v) => v.toFixed(0)}
            />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
              <input
                type="checkbox"
                checked={state.dagger}
                onChange={(e) => set({ dagger: e.target.checked })}
                className="accent-[var(--series-1)]"
              />
              Use DAgger (relabel the learner&apos;s own states)
            </label>
            <button
              type="button"
              onClick={() => {
                setDrawing((d) => !d);
                if (!drawing) setUserDemos([]);
              }}
              aria-pressed={drawing}
              className={`rounded-md border px-2.5 py-1 text-[11.5px] font-medium transition-colors ${
                drawing
                  ? 'border-series-2 bg-series-2 text-white'
                  : 'border-hairline text-ink-secondary hover:bg-surface-sunken'
              }`}
            >
              {drawing ? 'Drawing — release to finish' : 'Demonstrate it yourself'}
            </button>
            {userDemos.length > 4 && (
              <button
                type="button"
                onClick={() => setUserDemos([])}
                className="rounded-md border border-hairline px-2.5 py-1 text-[11.5px] text-ink-secondary hover:bg-surface-sunken"
              >
                Clear my demo ({userDemos.length} pts)
              </button>
            )}
            <button
              type="button"
              onClick={reset}
              className="ml-auto rounded-md border border-hairline px-2.5 py-1 text-[11.5px] text-ink-secondary hover:bg-surface-sunken"
            >
              Reset
            </button>
          </div>
        </div>
      }
      caption="Cut the demonstrations to two or three and the clone leaves the lane early — not because the network failed to fit, but because it fits only where the expert went, and its own first mistake takes it somewhere else. Tick DAgger and the same architecture, the same data budget, stays on the lane: asking the expert what to do in the states the LEARNER visits is what breaks the feedback loop. Draw your own demonstration to watch it fail on data you produced."
    >
      {error ? (
        <p className="rounded-lg border border-hairline px-3 py-2 text-[12.5px] text-status-critical">
          Simulation failed: {error}
        </p>
      ) : (
        <div className="grid gap-3 lg:grid-cols-[1fr,300px]">
          <div>
            <svg
              width={W}
              height={H}
              viewBox={`0 0 ${W} ${H}`}
              className="max-w-full touch-none rounded-lg"
              style={{ background: 'var(--surface-sunken)', cursor: drawing ? 'crosshair' : 'default' }}
              onPointerDown={(e) => drawing && handlePointer(e)}
              onPointerMove={handlePointer}
              onPointerUp={() => setDrawing(false)}
              role="img"
              aria-label="Demonstrated lane and the cloned policy's rollout"
            >
              {data && (
                <>
                  <path
                    d={`M${data.expertPath.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' L')}`}
                    fill="none"
                    stroke={seriesColor(0, mode)}
                    strokeWidth={2}
                    strokeDasharray="5 4"
                  />
                  <path
                    d={`M${data.clonePath.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' L')}`}
                    fill="none"
                    stroke={seriesColor(1, mode)}
                    strokeWidth={2.5}
                  />
                  <circle
                    cx={sx(data.clonePath[data.clonePath.length - 1].x)}
                    cy={sy(data.clonePath[data.clonePath.length - 1].y)}
                    r={6}
                    fill={seriesColor(1, mode)}
                    stroke="var(--surface-1)"
                    strokeWidth={2}
                  />
                </>
              )}
              {userDemos.length > 1 && (
                <path
                  d={`M${userDemos.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' L')}`}
                  fill="none"
                  stroke={seriesColor(2, mode)}
                  strokeWidth={2.5}
                  opacity={0.9}
                />
              )}
              <text x={10} y={14} fontSize={9.5} fill="var(--text-muted)">
                {drawing
                  ? 'drag left to right to demonstrate'
                  : '— — expert lane · —— cloned rollout'}
                {userDemos.length > 4 ? ' · green: your demonstration' : ''}
              </text>
              {running && (
                <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={12} fill="var(--text-muted)">
                  training the clone…
                </text>
              )}
            </svg>

            <div className="mt-2 grid grid-cols-3 gap-2">
              <StatTile
                label="Final deviation"
                value={data?.finalDeviation ?? 0}
                status={
                  (data?.finalDeviation ?? 0) > 0.2
                    ? 'critical'
                    : (data?.finalDeviation ?? 0) > 0.08
                      ? 'warning'
                      : 'good'
                }
                hint="distance off the lane at the end"
              />
              <StatTile
                label="Mean deviation"
                value={data?.meanDeviation ?? 0}
                hint="averaged over the rollout"
              />
              <StatTile
                label="Training pairs"
                value={data?.datasetSize ?? 0}
                hint={state.dagger ? 'grows each DAgger round' : 'from the demonstrations'}
              />
            </div>
          </div>

          <LineChart
            data={boundCurves}
            height={250}
            xLegend="horizon T"
            yLegend="worst-case regret"
            caption="Ross & Bagnell (2011): cloning's quadratic bound against DAgger's linear one."
          />
        </div>
      )}
    </SimPanel>
  );
}
