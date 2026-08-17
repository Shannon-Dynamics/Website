'use client';

import { useMemo } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { BarChart } from '@/components/viz/BarChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';
import { useDebounced, useSimulation, useWidgetState } from '@/lib/sim/useSimulation';
import type { GaitParams, RewardWeights } from '@/lib/rl/walker';

interface GaitResult {
  params: GaitParams;
  terms: Record<keyof RewardWeights, number>;
  weightedReturn: number;
  fell: boolean;
  meanSpeed: number;
  costOfTransport: number | null;
  trace: Array<{ x: number; y: number; theta: number; contacts: [boolean, boolean] }>;
  footHeights: Array<[number, number]>;
  history: number[];
}

const TERMS: Array<{
  key: keyof RewardWeights;
  label: string;
  formula: string;
  units: string;
  sign: 1 | -1;
  description: string;
}> = [
  {
    key: 'velocity',
    label: 'Velocity tracking',
    formula: 'exp(−‖v_xy − v*‖² / σ)',
    units: 'dimensionless',
    sign: 1,
    description: 'The task itself: move at the commanded speed.',
  },
  {
    key: 'effort',
    label: 'Effort',
    formula: '−‖F‖²',
    units: 'N²',
    sign: -1,
    description: 'Force expended. Raise it and the gait chooses to go slower.',
  },
  {
    key: 'airTime',
    label: 'Foot air time',
    formula: 'Σ_f (t_air − 0.15)',
    units: 's',
    sign: 1,
    description: 'Credited at touchdown, so it rewards real swings, not held feet.',
  },
  {
    key: 'orientation',
    label: 'Body orientation',
    formula: '−θ²',
    units: 'rad²',
    sign: -1,
    description: 'Keeps the body level; the main thing between you and a faceplant.',
  },
  {
    key: 'slip',
    label: 'Foot slip',
    formula: '−(|F_t| − μF_n)⁺',
    units: 'dimensionless',
    sign: -1,
    description: 'Friction cone exceeded — the foot is sliding under load.',
  },
];

const DEFAULTS = {
  velocity: 1.0,
  effort: 0.15,
  airTime: 0.3,
  orientation: 0.5,
  slip: 0.2,
  command: 0.9,
};

/**
 * `ch18-reward-mixer` — the reward function as an engineering artifact.
 *
 * The weights below are handed to an evolution strategy that optimizes a real
 * planar walker: leg spring-dampers, a friction cone, contact scheduling. When
 * a weight moves, a genuine optimizer re-solves and a different gait comes
 * back. Nothing here is a caricature — if the effort penalty makes creeping
 * the better policy, the walker creeps because that is what maximizes the
 * objective you wrote.
 */
export function RewardMixer() {
  const { mode } = useTheme();
  const [state, set, reset] = useWidgetState('ch18-reward-mixer', DEFAULTS);
  const debounced = useDebounced(state, 320);

  const weights: RewardWeights = useMemo(
    () => ({
      velocity: debounced.velocity,
      effort: debounced.effort,
      airTime: debounced.airTime,
      orientation: debounced.orientation,
      slip: debounced.slip,
    }),
    [debounced],
  );

  const { data, running, error } = useSimulation<GaitResult>('gait-optimize', {
    weights,
    targetSpeed: debounced.command,
    generations: 12,
    population: 24,
    seed: 5,
  });

  const verdict = useMemo(() => {
    if (!data) return null;
    if (data.fell) return { text: 'Falls over', status: 'critical' as const };
    if (data.meanSpeed < 0.2)
      return { text: 'Stands still', status: 'critical' as const };
    if (data.meanSpeed < debounced.command * 0.6)
      return { text: 'Creeps', status: 'warning' as const };
    if (data.terms.airTime > 6)
      return { text: 'Prancing — exaggerated swing', status: 'warning' as const };
    return { text: 'Plausible trotting gait', status: 'good' as const };
  }, [data, debounced.command]);

  const contributions = useMemo(() => {
    if (!data) return [];
    return TERMS.map((t) => ({
      id: t.label,
      value: t.sign * weights[t.key] * data.terms[t.key],
    }));
  }, [data, weights]);

  // Gait diagram: which feet are on the ground over time.
  const contactBars = data?.trace.slice(0, 90) ?? [];

  return (
    <SimPanel
      title="Reward anatomy: the weights decide the gait"
      id="ch18-reward-mixer"
      subtitle="An evolution strategy optimizes a real planar walker against the reward you specify. Move a weight and a different gait comes back."
      controls={
        <div className="space-y-2.5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {TERMS.map((t) => (
              <Slider
                key={t.key}
                label={t.label}
                value={state[t.key]}
                min={0}
                max={t.key === 'velocity' ? 2 : 1.5}
                step={0.05}
                onChange={(v) => set({ [t.key]: v } as never)}
                hint={t.units}
              />
            ))}
            <Slider
              label="Commanded speed"
              value={state.command}
              min={0.2}
              max={1.6}
              step={0.05}
              onChange={(v) => set({ command: v })}
              format={(v) => `${v.toFixed(2)} m/s`}
              hint="what the task asks for"
            />
          </div>
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-hairline px-2.5 py-1 text-[11.5px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
          >
            Reset to book defaults
          </button>
        </div>
      }
      caption="Set the velocity weight to zero and the walker stops — nothing rewards moving, so the optimizer correctly chooses not to. Raise the effort penalty and it slows down rather than tracking the command, because going slower is now worth more than tracking. Drop the orientation term and it pitches forward until it falls. None of these is an algorithm failure; each is the optimizer serving the objective you wrote."
    >
      {error ? (
        <p className="rounded-lg border border-hairline px-3 py-2 text-[12.5px] text-status-critical">
          Simulation failed: {error}
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[320px,1fr]">
          <div>
            {/* Walker, drawn from the optimized trajectory */}
            <svg
              width={320}
              height={168}
              viewBox="0 0 320 168"
              className="max-w-full rounded-lg"
              style={{ background: 'var(--surface-sunken)' }}
              role="img"
              aria-label={`Optimized gait: ${verdict?.text ?? 'computing'}`}
            >
              <line x1={0} y1={140} x2={320} y2={140} stroke="var(--baseline)" strokeWidth={1.5} />
              {data && data.trace.length > 2 ? (
                (() => {
                  // Show one stride, scaled into the frame.
                  const seg = data.trace.slice(0, Math.min(40, data.trace.length));
                  const x0 = seg[0].x;
                  const span = Math.max(0.35, seg[seg.length - 1].x - x0);
                  const sx = (x: number) => 30 + ((x - x0) / span) * 250;
                  const sy = (y: number) => 140 - y * 170;
                  const mid = seg[Math.floor(seg.length / 2)];
                  return (
                    <>
                      <path
                        d={`M${seg.map((s) => `${sx(s.x)},${sy(s.y)}`).join(' L')}`}
                        fill="none"
                        stroke={seriesColor(0, mode)}
                        strokeWidth={1.5}
                        strokeDasharray="3 3"
                        opacity={0.5}
                      />
                      <g transform={`translate(${sx(mid.x)},${sy(mid.y)}) rotate(${(mid.theta * 180) / Math.PI})`}>
                        <rect
                          x={-34}
                          y={-11}
                          width={68}
                          height={22}
                          rx={5}
                          fill={seriesColor(0, mode)}
                        />
                        {[0, 1].map((leg) => {
                          const down = mid.contacts[leg];
                          const hx = leg === 0 ? 26 : -26;
                          return (
                            <line
                              key={leg}
                              x1={hx}
                              y1={8}
                              x2={hx + (down ? 4 : 14)}
                              y2={down ? sy(0) - sy(mid.y) : sy(0) - sy(mid.y) - 18}
                              stroke={seriesColor(0, mode)}
                              strokeWidth={4}
                              strokeLinecap="round"
                              opacity={down ? 1 : 0.55}
                            />
                          );
                        })}
                      </g>
                      <text x={8} y={160} fontSize={10} fill="var(--text-muted)">
                        {data.meanSpeed.toFixed(2)} m/s · stride {(1 / data.params.frequency).toFixed(2)} s
                        · clearance {(data.params.clearance * 100).toFixed(1)} cm
                      </text>
                    </>
                  );
                })()
              ) : (
                <text x={160} y={80} textAnchor="middle" fontSize={12} fill="var(--text-muted)">
                  optimizing…
                </text>
              )}
            </svg>

            {/* Footfall diagram — the gait, as a contact schedule */}
            <div className="mt-2 rounded-lg border border-hairline bg-surface-sunken p-2.5">
              <p className="mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
                Footfall pattern
              </p>
              <svg width="100%" height={34} viewBox="0 0 290 34" role="img" aria-label="Contact schedule for the two virtual legs">
                {[0, 1].map((leg) => (
                  <g key={leg}>
                    <text x={0} y={leg * 16 + 12} fontSize={9} fill="var(--text-muted)">
                      {leg === 0 ? 'F' : 'H'}
                    </text>
                    {contactBars.map((t, i) => (
                      <rect
                        key={i}
                        x={14 + i * 3}
                        y={leg * 16 + 3}
                        width={2.4}
                        height={10}
                        fill={t.contacts[leg] ? seriesColor(0, mode) : 'var(--gridline)'}
                      />
                    ))}
                  </g>
                ))}
              </svg>
              <p className="mt-1 text-[10.5px] leading-snug text-ink-muted">
                Filled = foot loaded. Diagonal pairs alternate in a trot.
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <StatTile
                label="Outcome"
                value={running ? 'optimizing…' : (verdict?.text ?? '—')}
                mono={false}
                status={running ? undefined : verdict?.status}
              />
              <StatTile
                label="Achieved speed"
                value={data?.meanSpeed ?? 0}
                unit="m/s"
                hint={`commanded ${debounced.command.toFixed(2)}`}
              />
              <StatTile
                label="Cost of transport"
                value={data?.costOfTransport ?? NaN}
                hint="energy per unit weight-distance"
              />
              <StatTile
                label="Step frequency"
                value={data?.params.frequency ?? 0}
                unit="Hz"
                hint={data ? `duty ${(data.params.duty * 100).toFixed(0)}%` : ''}
              />
            </div>

            <BarChart
              data={contributions}
              layout="horizontal"
              height={190}
              xLegend="weighted contribution to return"
              title="Where the return actually comes from"
              table={{
                columns: ['Term', 'Contribution'],
                rows: contributions.map((c) => [String(c.id), c.value]),
              }}
            />

            <div className="space-y-1.5">
              {TERMS.map((t) => (
                <div
                  key={t.key}
                  className="rounded-lg border border-hairline bg-surface px-3 py-1.5 text-[11.5px]"
                >
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <span className="font-semibold text-ink">{t.label}</span>
                    <code className="font-mono text-[10.5px] text-ink-secondary">{t.formula}</code>
                  </div>
                  <p className="mt-0.5 leading-snug text-ink-muted">{t.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </SimPanel>
  );
}
