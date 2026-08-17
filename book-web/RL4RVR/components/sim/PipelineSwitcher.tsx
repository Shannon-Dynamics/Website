'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Segmented, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { BarChart } from '@/components/viz/BarChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

type Arch = 'modular' | 'hybrid' | 'end-to-end';

interface Stage {
  name: string;
  learned: boolean;
  note: string;
}

const PIPELINES: Record<Arch, { stages: Stage[]; summary: string }> = {
  modular: {
    summary:
      'Classical stack: SLAM builds a map, a global planner finds a route, a local planner avoids obstacles, a tracking controller executes. Each stage is inspectable and independently testable.',
    stages: [
      { name: 'Perception / SLAM', learned: false, note: 'geometric, well understood' },
      { name: 'Global planner', learned: false, note: 'A* / RRT on the map' },
      { name: 'Local planner', learned: false, note: 'DWA, hand-tuned costs' },
      { name: 'Tracking controller', learned: false, note: 'PID or MPC' },
    ],
  },
  hybrid: {
    summary:
      'Keep the parts that classical methods do well — mapping and global routing — and learn the part where hand-tuned costs struggle: reactive local navigation among moving obstacles.',
    stages: [
      { name: 'Perception / SLAM', learned: false, note: 'geometric, well understood' },
      { name: 'Global planner', learned: false, note: 'A* / RRT on the map' },
      { name: 'Local policy', learned: true, note: 'RL from lidar + goal vector' },
      { name: 'Tracking controller', learned: false, note: 'PID or MPC' },
    ],
  },
  'end-to-end': {
    summary:
      'One network from raw sensing to velocity commands. Nothing to hand-tune and nothing to inspect: when it fails, the diagnosis is "the network did that".',
    stages: [
      { name: 'Sensor → action network', learned: true, note: 'lidar/pixels → velocity' },
    ],
  },
};

/**
 * `ch19-pipeline-switcher` — end-to-end versus modular, with the trade made explicit.
 *
 * NOTE ON FIDELITY: unlike most simulations in this book, this one does not run
 * the architectures it compares — training a SLAM stack and an end-to-end
 * policy in the browser is not feasible. The curves encode the survey's
 * qualitative findings, and the widget is labelled in the interface as a
 * conceptual comparison so no reader mistakes it for a measurement.
 *
 * Tang's survey finds neither architecture universally superior, and the reason
 * is that they fail differently. Modular stacks degrade predictably and can be
 * debugged stage by stage; end-to-end policies handle situations nobody wrote a
 * cost function for, and fail without explanation. The slider makes the
 * crossover concrete: as the environment gets less structured, the learned
 * components gain and the interpretability is what you pay.
 */
export function PipelineSwitcher() {
  const { mode } = useTheme();
  const [arch, setArch] = useState<Arch>('hybrid');
  const [clutter, setClutter] = useState(0.5);

  const metrics = useMemo(() => {
    // Success falls off with clutter at a rate that depends on how much of the
    // stack was hand-designed against structured assumptions.
    const success = {
      modular: Math.max(0.12, 0.97 - 1.15 * clutter * clutter),
      hybrid: Math.max(0.3, 0.94 - 0.55 * clutter * clutter),
      'end-to-end': Math.max(0.25, 0.86 - 0.42 * clutter * clutter - 0.06 * (1 - clutter)),
    }[arch];

    const interpretability = { modular: 1.0, hybrid: 0.62, 'end-to-end': 0.15 }[arch];
    const tuningEffort = { modular: 0.9, hybrid: 0.55, 'end-to-end': 0.25 }[arch];
    const dataNeed = { modular: 0.05, hybrid: 0.5, 'end-to-end': 1.0 }[arch];

    return { success, interpretability, tuningEffort, dataNeed };
  }, [arch, clutter]);

  const comparison = useMemo(
    () =>
      (['modular', 'hybrid', 'end-to-end'] as Arch[]).map((a) => ({
        id: a === 'end-to-end' ? 'end-to-end' : a,
        value:
          {
            modular: Math.max(0.12, 0.97 - 1.15 * clutter * clutter),
            hybrid: Math.max(0.3, 0.94 - 0.55 * clutter * clutter),
            'end-to-end': Math.max(0.25, 0.86 - 0.42 * clutter * clutter - 0.06 * (1 - clutter)),
          }[a] * 100,
      })),
    [clutter],
  );

  const pipeline = PIPELINES[arch];

  return (
    <SimPanel
      title="End-to-end or modular?"
      id="ch19-pipeline-switcher"
      subtitle="A conceptual comparison, not a simulation: the curves encode the survey's qualitative findings, because training a SLAM stack and an end-to-end policy in the browser is not feasible."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Segmented
            label="Architecture"
            value={arch}
            onChange={setArch}
            options={[
              { value: 'modular', label: 'Classical modular' },
              { value: 'hybrid', label: 'Hybrid' },
              { value: 'end-to-end', label: 'End-to-end' },
            ]}
          />
          <Slider
            className="w-56"
            label="Environment unstructuredness"
            value={clutter}
            min={0}
            max={1}
            step={0.02}
            onChange={setClutter}
            format={(v) => `${(v * 100).toFixed(0)}%`}
            hint="empty corridor → crowded, dynamic, unmapped"
          />
        </div>
      }
      caption="In a mapped, static corridor the classical stack is best and needs no data at all — reaching for RL there is a mistake. As the environment becomes crowded and dynamic, hand-tuned local planners degrade fastest, because their cost functions encode assumptions that no longer hold. The hybrid architecture wins across most of the middle, which is why it dominates deployed systems: learn the reactive layer where hand-design struggles, keep the global reasoning where it does not."
    >
      <div className="grid gap-4 lg:grid-cols-[1fr,215px]">
        <div>
          {/* Pipeline diagram */}
          <div className="rounded-lg border border-hairline bg-surface-sunken p-3">
            <div className="flex flex-wrap items-stretch gap-1.5">
              {pipeline.stages.map((s, i) => (
                <div key={s.name} className="flex items-stretch gap-1.5">
                  <div
                    className="min-w-[124px] flex-1 rounded-md border px-2.5 py-2"
                    style={{
                      borderColor: s.learned ? seriesColor(0, mode) : 'var(--border-hairline)',
                      background: s.learned
                        ? `color-mix(in srgb, ${seriesColor(0, mode)} 11%, transparent)`
                        : 'var(--surface-1)',
                    }}
                  >
                    <p className="text-[11.5px] font-semibold leading-snug text-ink">{s.name}</p>
                    <p className="mt-0.5 text-[10.5px] leading-snug text-ink-muted">{s.note}</p>
                    <span
                      className="mt-1 inline-block rounded px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wide"
                      style={{
                        color: s.learned ? seriesColor(0, mode) : 'var(--text-muted)',
                        background: s.learned
                          ? `color-mix(in srgb, ${seriesColor(0, mode)} 16%, transparent)`
                          : 'var(--surface-0)',
                      }}
                    >
                      {s.learned ? 'learned' : 'hand-designed'}
                    </span>
                  </div>
                  {i < pipeline.stages.length - 1 && (
                    <span className="self-center text-ink-muted" aria-hidden>
                      →
                    </span>
                  )}
                </div>
              ))}
            </div>
            <p className="mt-2.5 border-t border-hairline pt-2 text-[12px] leading-relaxed text-ink-secondary">
              {pipeline.summary}
            </p>
          </div>

          <BarChart
            data={comparison}
            height={185}
            xLegend="architecture"
            yLegend="success rate (%)"
            colorByIndex
            title="Success at the current unstructuredness"
            table={{
              columns: ['Architecture', 'Success %'],
              rows: comparison.map((c) => [String(c.id), c.value]),
            }}
          />
        </div>

        <div className="space-y-2">
          <StatTile
            label="Success rate"
            value={metrics.success * 100}
            unit="%"
            status={metrics.success > 0.8 ? 'good' : metrics.success > 0.5 ? 'warning' : 'critical'}
          />
          <StatTile
            label="Interpretability"
            value={metrics.interpretability}
            hint="can you tell why it failed?"
          />
          <StatTile
            label="Hand-tuning burden"
            value={metrics.tuningEffort}
            hint="cost functions, gains, thresholds"
          />
          <StatTile
            label="Data requirement"
            value={metrics.dataNeed}
            hint="training episodes needed"
          />
        </div>
      </div>
    </SimPanel>
  );
}
