'use client';

import { useMemo } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { useDebounced, useSimulation, useWidgetState } from '@/lib/sim/useSimulation';

interface Result {
  masses: number[];
  narrow: Array<{ x: number; y: number }>;
  wide: Array<{ x: number; y: number }>;
  narrowWorst: number;
  wideWorst: number;
  narrowPeak: number;
  narrowAssumedMass: number;
  wideAssumedMass: number;
  trainedRange: number;
  widePeak: number;
}

const DEFAULTS = { range: 0.25, realMass: 1.35, iterations: 18 };

/**
 * `ch15-randomization-wall` — the peak/robustness trade, measured.
 *
 * Two tile-coded SARSA policies are actually trained on Pendle: one at the
 * nominal pole mass, one with the mass drawn from a randomization range each
 * episode. Both are then evaluated across a grid of masses neither was trained
 * on. The curves are held-out performance, so the trade the chapter describes
 * is something the reader can measure rather than take on faith.
 */
export function DomainRandomization() {
  const [state, set, reset] = useWidgetState('ch15-randomization-wall', DEFAULTS);
  const debounced = useDebounced(state, 500);

  const { data, running, error } = useSimulation<Result>('randomization-transfer', {
    range: debounced.range,
    iterations: debounced.iterations,
  });

  const series = useMemo(() => {
    if (!data) return [];
    return [
      { id: `randomized ±${(debounced.range * 100).toFixed(0)}%`, data: data.wide },
      { id: 'trained at nominal only', data: data.narrow },
    ];
  }, [data, debounced.range]);

  /** Nearest evaluated mass to the reader's chosen "real" robot. */
  const atReal = useMemo(() => {
    if (!data) return null;
    const nearest = (arr: Array<{ x: number; y: number }>) =>
      arr.reduce((best, p) =>
        Math.abs(p.x - debounced.realMass) < Math.abs(best.x - debounced.realMass) ? p : best,
      );
    return { wide: nearest(data.wide).y, narrow: nearest(data.narrow).y };
  }, [data, debounced.realMass]);

  return (
    <SimPanel
      title="Domain randomization: buying the worst case"
      id="ch15-randomization-wall"
      subtitle="Two computed-torque policies found by policy search — one trained at the nominal mass only, one across a randomized range — then evaluated on masses neither has seen."
      controls={
        <div className="space-y-2.5">
          <div className="grid gap-3 sm:grid-cols-3">
            <Slider
              label="Randomization range"
              value={state.range}
              min={0}
              max={0.6}
              step={0.05}
              onChange={(v) => set({ range: v })}
              format={(v) => `±${(v * 100).toFixed(0)}%`}
              hint="how much mass varies during training"
            />
            <Slider
              label="The real robot's mass"
              value={state.realMass}
              min={0.55}
              max={1.55}
              step={0.05}
              onChange={(v) => set({ realMass: v })}
              hint="unknown at training time — that is the problem"
            />
            <Slider
              label="Search iterations"
              value={state.iterations}
              min={6}
              max={30}
              step={2}
              onChange={(v) => set({ iterations: v })}
              format={(v) => v.toFixed(0)}
            />
          </div>
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-hairline px-2.5 py-1 text-[11.5px] text-ink-secondary hover:bg-surface-sunken"
          >
            Reset
          </button>
        </div>
      }
      caption="Each policy cancels the gravity it believes is acting, using an assumed mass it learned during search. Watch that assumed mass as you widen the range: training only at the nominal mass fits it close to 1.0, and the policy then under-compensates on anything heavier and falls. Randomized training deliberately over-estimates it — insurance against a parameter nobody can measure before deployment — and the worst case across the held-out masses improves substantially. The worst case is the number that ships."
    >
      {error ? (
        <p className="rounded-lg border border-hairline px-3 py-2 text-[12.5px] text-status-critical">
          Simulation failed: {error}
        </p>
      ) : (
        <div className="grid gap-3 lg:grid-cols-[1fr,215px]">
          <LineChart
            data={series}
            height={265}
            xLegend="pole mass at evaluation (kg)"
            yLegend="fraction of time upright"
            yMin={0}
            yMax={1}
            dashed={['trained at nominal only']}
            caption={
              running
                ? 'Training two policies and evaluating across the mass grid…'
                : 'Held-out evaluation: five trials per mass, deterministic control.'
            }
            table={
              data
                ? {
                    columns: ['Mass', 'Randomized', 'Nominal-only'],
                    rows: data.wide.map((p, i) => [
                      p.x.toFixed(2),
                      p.y,
                      data.narrow[i].y,
                    ]),
                  }
                : undefined
            }
          />
          <div className="space-y-2">
            <StatTile
              label="Randomized, on the real robot"
              value={atReal?.wide ?? 0}
              status={(atReal?.wide ?? 0) > 0.5 ? 'good' : 'warning'}
              hint={`at m = ${debounced.realMass.toFixed(2)} kg`}
            />
            <StatTile
              label="Nominal-only, on the real robot"
              value={atReal?.narrow ?? 0}
              status={(atReal?.narrow ?? 0) > 0.5 ? 'good' : 'critical'}
              hint="trained at m = 1.00 exactly"
            />
            <StatTile
              label="Worst case — randomized"
              value={data?.wideWorst ?? 0}
              hint="min over the whole grid"
              status="good"
            />
            <StatTile
              label="Worst case — nominal"
              value={data?.narrowWorst ?? 0}
              hint="the number that would ship"
              status="critical"
            />
            <StatTile
              label="Assumed mass — nominal-only"
              value={data?.narrowAssumedMass ?? 0}
              unit="kg"
              hint="what that policy thinks it is holding"
            />
            <StatTile
              label="Assumed mass — randomized"
              value={data?.wideAssumedMass ?? 0}
              unit="kg"
              hint="hedged upward as insurance"
            />
          </div>
        </div>
      )}
    </SimPanel>
  );
}
