'use client';

import { useState } from 'react';
import { BarChart } from '@/components/viz/BarChart';
import { SimPanel } from './SimControls';
import { cn } from '@/lib/utils';

/**
 * `ch01-success-levels` — Tang et al.'s levels of real-world success (§3.4),
 * turned into a browsable maturity map.
 *
 * The levels are the survey's own rubric for "how real is this result?", and
 * they run as an evaluation standard through the whole book: Chapter 14 adopts
 * them as the reward-design sanity check, and Chapter 22 grades its own
 * capstone against them.
 */

const LEVELS = [
  { id: 0, label: 'L0', name: 'Simulation only', desc: 'Validated only in simulation environments.' },
  { id: 1, label: 'L1', name: 'Limited lab', desc: 'Validated under limited laboratory conditions.' },
  { id: 2, label: 'L2', name: 'Diverse lab', desc: 'Validated under diverse laboratory conditions.' },
  {
    id: 3,
    label: 'L3',
    name: 'Confined real world',
    desc: 'Validated under confined real-world operational conditions.',
  },
  {
    id: 4,
    label: 'L4',
    name: 'Diverse real world',
    desc: 'Validated under diverse, representative real-world conditions.',
  },
  { id: 5, label: 'L5', name: 'Commercialized', desc: 'Deployed on commercialized products.' },
];

interface System {
  name: string;
  competency: string;
  level: number;
  note: string;
}

/** Exemplars drawn from Tang et al. §4, with the survey's own level assessments. */
const SYSTEMS: System[] = [
  {
    name: 'Quadruped locomotion (ANYmal, perceptive)',
    competency: 'Locomotion',
    level: 4,
    note: 'Zero-shot sim-to-real with teacher–student privileged learning; deployed on varied natural terrain.',
  },
  {
    name: 'Production quadrupeds (ANYbotics, Swiss-Mile, Boston Dynamics)',
    competency: 'Locomotion',
    level: 5,
    note: 'RL locomotion controllers shipping inside commercial robot products.',
  },
  {
    name: 'Champion-level drone racing',
    competency: 'Locomotion',
    level: 4,
    note: 'Kaufmann et al., Nature 2023 — beat human champions on a physical racing track.',
  },
  {
    name: 'Legged + wheeled navigation',
    competency: 'Navigation',
    level: 3,
    note: 'Robust local planning in real buildings; global reasoning still largely classical.',
  },
  {
    name: 'Social / crowd navigation',
    competency: 'Navigation',
    level: 2,
    note: 'Human behaviour is the unmodelable part — simulation fidelity caps transfer.',
  },
  {
    name: 'In-hand cube reorientation',
    competency: 'Manipulation',
    level: 2,
    note: 'Massive domain randomization + recurrent policies; impressive, but lab-bound.',
  },
  {
    name: 'Contact-rich assembly / insertion',
    competency: 'Manipulation',
    level: 3,
    note: 'Dense rewards designable a priori; impedance action spaces do the heavy lifting.',
  },
  {
    name: 'Open-world pick-and-place',
    competency: 'Manipulation',
    level: 2,
    note: 'Object and scene diversity keeps general-purpose picking below confined deployment.',
  },
  {
    name: 'Long-horizon mobile manipulation',
    competency: 'Mobile manipulation',
    level: 2,
    note: 'Skill composition works; the open question is which skills to learn at all.',
  },
  {
    name: 'Physical human–robot collaboration',
    competency: 'HRI',
    level: 1,
    note: 'Neither accurate simulation nor cheap real rollouts — the hardest data regime.',
  },
  {
    name: 'Multi-robot soccer',
    competency: 'Multi-robot',
    level: 4,
    note: 'Full-body control and coordination on physical humanoid/quadruped platforms.',
  },
  {
    name: 'Urban autonomous driving',
    competency: 'Multi-robot',
    level: 1,
    note: 'DRL-based solutions remain in simulation or strictly confined field tests.',
  },
];

const COMPETENCIES = ['Locomotion', 'Navigation', 'Manipulation', 'Mobile manipulation', 'HRI', 'Multi-robot'];

export function SuccessLevels() {
  const [selected, setSelected] = useState<string | null>(null);

  const peak = COMPETENCIES.map((c) => ({
    id: c,
    value: Math.max(...SYSTEMS.filter((s) => s.competency === c).map((s) => s.level)),
  }));

  const shown = selected ? SYSTEMS.filter((s) => s.competency === selected) : SYSTEMS;

  return (
    <SimPanel
      title="Levels of real-world success"
      id="ch01-success-levels"
      subtitle="Tang et al. (2024) §3.4 — a maturity rubric for robot RL results, from 'works in simulation' to 'shipping in a product'."
      caption="The bar chart shows the HIGHEST level any surveyed system reached per competency — not the typical one. Locomotion reaches L5 because its dynamics simulate well and its rewards shape easily; HRI sits at L1 because humans are the part nobody can simulate. That spread, more than any single algorithm, is the map this book navigates."
    >
      <div className="space-y-4">
        {/* The ladder */}
        <ol className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
          {LEVELS.map((l) => (
            <li
              key={l.id}
              className="flex gap-2.5 rounded-lg border border-hairline bg-surface-sunken px-3 py-2"
            >
              <span
                className="tabular mt-0.5 h-fit shrink-0 rounded px-1.5 py-0.5 text-[11px] font-bold text-white"
                style={{
                  background: `color-mix(in srgb, var(--series-1) ${28 + l.id * 14}%, var(--text-muted))`,
                }}
              >
                {l.label}
              </span>
              <span>
                <span className="block text-[12.5px] font-semibold text-ink">{l.name}</span>
                <span className="block text-[11.5px] leading-snug text-ink-muted">{l.desc}</span>
              </span>
            </li>
          ))}
        </ol>

        <BarChart
          data={peak}
          layout="horizontal"
          height={230}
          xLegend="highest level demonstrated"
          title="Peak demonstrated maturity, by competency"
          table={{
            columns: ['Competency', 'Peak level'],
            rows: peak.map((p) => [p.id, `L${p.value}`]),
          }}
        />

        {/* Filter row, above the list */}
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setSelected(null)}
            aria-pressed={selected === null}
            className={cn(
              'rounded-md border px-2 py-1 text-[11.5px] font-medium transition-colors',
              selected === null
                ? 'border-series-1 bg-series-1 text-white'
                : 'border-hairline text-ink-secondary hover:bg-surface-sunken',
            )}
          >
            All competencies
          </button>
          {COMPETENCIES.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setSelected(c === selected ? null : c)}
              aria-pressed={selected === c}
              className={cn(
                'rounded-md border px-2 py-1 text-[11.5px] font-medium transition-colors',
                selected === c
                  ? 'border-series-1 bg-series-1 text-white'
                  : 'border-hairline text-ink-secondary hover:bg-surface-sunken',
              )}
            >
              {c}
            </button>
          ))}
        </div>

        <ul className="space-y-1.5">
          {shown.map((s) => (
            <li
              key={s.name}
              className="flex gap-3 rounded-lg border border-hairline bg-surface px-3 py-2.5"
            >
              <span
                className="tabular h-fit shrink-0 rounded px-1.5 py-0.5 text-[11px] font-bold text-white"
                style={{
                  background: `color-mix(in srgb, var(--series-1) ${28 + s.level * 14}%, var(--text-muted))`,
                }}
              >
                L{s.level}
              </span>
              <span className="min-w-0">
                <span className="block text-[13px] font-semibold text-ink">{s.name}</span>
                <span className="block text-[12px] leading-snug text-ink-muted">{s.note}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </SimPanel>
  );
}
