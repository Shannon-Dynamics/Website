'use client';

import Link from 'next/link';
import { Fragment, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { Toggle } from '@/components/sim/controls';

/**
 * w11.3 — the Taxonomy Grid.
 *
 * Localization is not one problem. Thrun's taxonomy splits it along three axes,
 * and a Gaussian filter lives in exactly one cell of the resulting grid. This
 * diagram is the map of Part IV: click a cell to see what the prior looks like
 * there, whether one Gaussian can carry it, and which chapter pays the bill.
 */

type Knowledge = 'tracking' | 'global' | 'kidnapped';
type Environment = 'static' | 'dynamic';

interface Cell {
  knowledge: Knowledge;
  environment: Environment;
  /** Can a single Gaussian represent the belief this cell demands? */
  gaussian: 'yes' | 'partly' | 'no';
  prior: string;
  breaks: string;
  passive: string;
  active: string;
  links: { label: string; slug: string }[];
}

const KNOWLEDGE: { id: Knowledge; label: string; sub: string }[] = [
  { id: 'tracking', label: 'Position tracking', sub: 'initial pose known' },
  { id: 'global', label: 'Global localization', sub: 'initial pose unknown' },
  { id: 'kidnapped', label: 'Kidnapped robot', sub: 'initial pose wrong' },
];

const ENVIRONMENT: { id: Environment; label: string; sub: string }[] = [
  { id: 'static', label: 'Static', sub: 'only the robot moves' },
  { id: 'dynamic', label: 'Dynamic', sub: 'the world moves too' },
];

const CELLS: Cell[] = [
  {
    knowledge: 'tracking',
    environment: 'static',
    gaussian: 'yes',
    prior: 'One narrow Gaussian, centred on a pose you already know. The error is small enough that the motion and measurement models are near-linear across it.',
    breaks:
      'Only two things: a wrong data association, and a linearisation point that has drifted too far from the truth. This chapter is about the first; Chapter 7 was about the second.',
    passive: 'The EKF localizer of this chapter. Cheap, constant-time, and the right answer whenever it applies.',
    active:
      'Active tracking means steering to keep landmarks in view — trading path length for a tighter ellipse. That is a decision problem, and it waits for Chapter 22.',
    links: [
      { label: 'Ch. 7 — EKF and manifolds', slug: 'ch07-ekf-ukf-manifolds' },
      { label: 'Ch. 22 — POMDPs', slug: 'ch22-pomdps' },
    ],
  },
  {
    knowledge: 'global',
    environment: 'static',
    gaussian: 'no',
    prior: 'Uniform over every pose in the map. Nothing is more plausible than anything else, and the first door sighting produces one peak per identical door.',
    breaks:
      'A single Gaussian has one mode. Averaging three doors puts the mean in a wall — a pose the robot has never occupied and could not occupy.',
    passive:
      'Represent the belief with cells or samples: grid localization and Monte Carlo localization, both in Chapter 12.',
    active:
      'A robot that chooses where to look can collapse the ambiguity deliberately — drive to where the three hypotheses predict *different* readings.',
    links: [
      { label: 'Ch. 8 — nonparametric filters', slug: 'ch08-nonparametric-filters' },
      { label: 'Ch. 12 — global localization', slug: 'ch12-localization-global' },
    ],
  },
  {
    knowledge: 'kidnapped',
    environment: 'static',
    gaussian: 'no',
    prior: 'A narrow Gaussian around the wrong pose. Formally this is global localization; practically it is worse, because the robot does not know it has a problem.',
    breaks:
      'The gate rejects every true feature as an outlier, so the filter receives no corrections at all and stays confidently wrong. Its covariance grows slowly, from motion noise alone, and may take minutes to become wide enough to catch the truth.',
    passive:
      'Watch the evidence, not the covariance. A run of unexplained measurements is the signal; injecting fresh hypotheses is the cure (Augmented MCL, Chapter 12).',
    active: 'Deliberately drive somewhere distinctive. Recovery is faster when the robot chooses an informative place to be lost in.',
    links: [
      { label: 'Ch. 12 — kidnapping and recovery', slug: 'ch12-localization-global' },
      { label: 'Ch. 5 — the evidence term', slug: 'ch05-bayes-filter' },
    ],
  },
  {
    knowledge: 'tracking',
    environment: 'dynamic',
    gaussian: 'partly',
    prior: 'The same narrow Gaussian — but the map is now partly a lie. People, moved chairs, and open doors generate features that belong to no landmark.',
    breaks:
      'Outliers stop being independent. A gate copes with one intruder; it does not cope with a crowd walking alongside the robot, because their features are consistent with each other and inconsistent with the map.',
    passive:
      'Gate hard, and model the clutter explicitly: the beam mixture of Chapter 10 and the robust costs of Chapter 15 are the two honest answers.',
    active: 'Prefer viewpoints dominated by structure that cannot walk away — walls and corners over the middle of a lobby.',
    links: [
      { label: 'Ch. 10 — sensor models', slug: 'ch10-sensor-models' },
      { label: 'Ch. 15 — robust back-ends', slug: 'ch15-factor-graphs' },
    ],
  },
  {
    knowledge: 'global',
    environment: 'dynamic',
    gaussian: 'no',
    prior: 'Uniform, in a world where a fraction of what the sensor reports is not in the map at all.',
    breaks:
      'Both failure modes at once: the belief is multimodal *and* the likelihood is contaminated. A particle filter can represent the first; nothing but a clutter model saves it from the second.',
    passive: 'MCL with a mixture measurement model, and enough particles that the true pose is never unrepresented.',
    active: 'Move to where the map is most reliable before trying to decide where you are.',
    links: [
      { label: 'Ch. 12 — MCL', slug: 'ch12-localization-global' },
      { label: 'Ch. 10 — beam mixture', slug: 'ch10-sensor-models' },
    ],
  },
  {
    knowledge: 'kidnapped',
    environment: 'dynamic',
    gaussian: 'no',
    prior: 'Confidently wrong, in a world that keeps changing. The honest worst case, and the one every deployed robot eventually meets.',
    breaks:
      'The failure detector itself becomes unreliable: unexplained measurements now have an innocent explanation, so "I am lost" and "someone walked in front of me" look identical for a while.',
    passive:
      'Long-horizon evidence tracking, hypothesis injection, and a planner that can ask for help. This is where localization stops being an estimation problem alone.',
    active: 'Exploration and active SLAM: choose actions to maximise information gain, not just to reach a goal.',
    links: [
      { label: 'Ch. 12 — recovery', slug: 'ch12-localization-global' },
      { label: 'Ch. 24 — exploration', slug: 'ch24-exploration' },
    ],
  },
];

const VERDICT: Record<Cell['gaussian'], { text: string; color: string }> = {
  yes: { text: 'one Gaussian suffices', color: 'var(--pr-posterior)' },
  partly: { text: 'one Gaussian, with help', color: 'var(--pr-measurement)' },
  no: { text: 'one Gaussian cannot', color: 'var(--pr-prediction)' },
};

/** A 32×14 sketch of the prior the cell starts from. */
function BeliefSketch({ knowledge }: { knowledge: Knowledge }) {
  const peak = (cx: number) =>
    `M 2 12 ${Array.from({ length: 29 }, (_, i) => {
      const x = 2 + i;
      const y = 12 - 10 * Math.exp(-((x - cx) ** 2) / 4);
      return `L ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ')} L 30 12`;

  return (
    <svg viewBox="0 0 32 14" className="h-5 w-16" aria-hidden="true">
      <line x1="2" y1="12" x2="30" y2="12" stroke="var(--pr-grid)" strokeWidth="0.6" />
      {/* the true pose, always at the same place */}
      <line x1="10" y1="12" x2="10" y2="3" stroke="var(--pr-truth)" strokeWidth="0.7" strokeDasharray="1.5 1.2" />
      {knowledge === 'global' ? (
        <line x1="2" y1="7" x2="30" y2="7" stroke="var(--pr-posterior)" strokeWidth="1.1" />
      ) : (
        <path
          d={peak(knowledge === 'tracking' ? 10 : 24)}
          fill="none"
          stroke="var(--pr-posterior)"
          strokeWidth="1.1"
        />
      )}
    </svg>
  );
}

export function TaxonomyGrid() {
  const [selected, setSelected] = useState(0);
  const [active, setActive] = useState(false);
  const cell = CELLS[selected];

  return (
    <WidgetFrame
      id="w11.3"
      title="The Taxonomy Grid"
      teaches="Gaussian localization is not a solution to 'localization'. It is a solution to one cell of a six-cell grid, and it fails outside it for reasons you can name in advance."
      colorKey={['posterior', 'truth']}
      caption={
        <>
          Three axes: what the robot knows initially, whether the world holds still, and whether it
          gets to choose where to go. Each cell shows the prior belief it starts from — purple for
          the belief, gray dashed for the true pose. Click any cell. The one cell an extended Kalman
          filter can honestly claim is the top-left; the rest of Part IV is about the other five.
        </>
      }
    >
      <div className="grid gap-0 lg:grid-cols-[1.25fr_1fr] lg:divide-x lg:divide-fd-border">
        <div className="p-3">
          <div className="grid grid-cols-[5.5rem_repeat(3,minmax(0,1fr))] gap-1.5">
            <div />
            {KNOWLEDGE.map((k) => (
              <div key={k.id} className="px-1 pb-1">
                <div className="font-ui text-[0.72rem] font-semibold leading-tight">{k.label}</div>
                <div className="font-mono text-[0.6rem] text-fd-muted-foreground">{k.sub}</div>
              </div>
            ))}

            {ENVIRONMENT.map((e) => (
              <Fragment key={e.id}>
                <div className="flex flex-col justify-center pe-2">
                  <div className="font-ui text-[0.72rem] font-semibold leading-tight">{e.label}</div>
                  <div className="font-mono text-[0.6rem] text-fd-muted-foreground">{e.sub}</div>
                </div>
                {KNOWLEDGE.map((k) => {
                  const index = CELLS.findIndex(
                    (c) => c.knowledge === k.id && c.environment === e.id,
                  );
                  const c = CELLS[index];
                  const isSel = index === selected;
                  return (
                    <button
                      key={`${e.id}-${k.id}`}
                      type="button"
                      onClick={() => setSelected(index)}
                      aria-pressed={isSel}
                      className={`flex flex-col items-start gap-1 rounded-sm border px-2 py-2 text-start transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fd-primary ${
                        isSel
                          ? 'border-fd-primary bg-fd-primary/10'
                          : 'border-fd-border bg-fd-card hover:bg-fd-accent'
                      }`}
                    >
                      <BeliefSketch knowledge={k.id} />
                      <span
                        className="font-mono text-[0.6rem] leading-tight"
                        style={{ color: VERDICT[c.gaussian].color }}
                      >
                        {VERDICT[c.gaussian].text}
                      </span>
                    </button>
                  );
                })}
              </Fragment>
            ))}
          </div>

          <div className="mt-3 border-t border-fd-border pt-2.5">
            <Toggle label="Active: the robot chooses where to go" checked={active} onChange={setActive} />
          </div>
        </div>

        <div className="border-t border-fd-border p-4 lg:border-t-0">
          <h4 className="font-display text-base font-semibold">
            {KNOWLEDGE.find((k) => k.id === cell.knowledge)?.label} ·{' '}
            {ENVIRONMENT.find((e) => e.id === cell.environment)?.label.toLowerCase()}
          </h4>

          <dl className="mt-3 space-y-2.5">
            <div>
              <dt className="eyebrow">the prior</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{cell.prior}</dd>
            </div>
            <div>
              <dt className="eyebrow">what breaks</dt>
              <dd
                className="font-prose text-[0.84rem] leading-snug"
                style={{ color: 'var(--pr-prediction)' }}
              >
                {cell.breaks}
              </dd>
            </div>
            <div>
              <dt className="eyebrow">{active ? 'if the robot may choose' : 'the fix'}</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">
                {active ? cell.active : cell.passive}
              </dd>
            </div>
          </dl>

          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
            {cell.links.map((l) => (
              <Link
                key={l.slug}
                href={`/chapters/${l.slug}`}
                className="font-mono text-[0.7rem] text-fd-primary hover:underline"
              >
                {l.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </WidgetFrame>
  );
}
