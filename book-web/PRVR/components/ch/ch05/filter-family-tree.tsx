'use client';

import Link from 'next/link';
import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * w5.3 — the filter family tree.
 *
 * The Bayes filter is not implementable as written; every practical filter is a
 * choice of representation for the belief. This diagram makes the choices — and
 * what each one costs — clickable, so the reader can see the map of Part II
 * before walking it.
 */

interface Node {
  id: string;
  label: string;
  sub: string;
  x: number;
  y: number;
  chapter?: number;
  slug?: string;
  represents: string;
  assumes: string;
  cost: string;
  breaks: string;
}

const NODES: Node[] = [
  {
    id: 'bayes',
    label: 'Bayes filter',
    sub: 'the recursion',
    x: 50,
    y: 8,
    chapter: 5,
    slug: 'ch05-bayes-filter',
    represents: 'An arbitrary distribution over the state — a mathematical object, not a data structure.',
    assumes: 'Only the Markov assumption: the state is a complete summary of the past.',
    cost: 'Not computable. The prediction step is an integral over a continuous state space.',
    breaks: 'Nothing — but nothing runs it either. Everything below is an attempt to approximate it.',
  },
  {
    id: 'gauss',
    label: 'Gaussian family',
    sub: 'belief = (μ, Σ)',
    x: 26,
    y: 36,
    represents: 'One mean and one covariance. The belief is unimodal by construction.',
    assumes: 'The posterior stays approximately Gaussian under motion and measurement.',
    cost: 'Cheap and constant: a few matrix operations per step.',
    breaks: 'Ambiguity. It cannot say "one of three doors" — it will average them and be confident about a place the robot has never been.',
  },
  {
    id: 'nonparam',
    label: 'Nonparametric family',
    sub: 'belief = samples or cells',
    x: 74,
    y: 36,
    represents: 'Many hypotheses, each carrying weight. No functional form assumed.',
    assumes: 'That enough samples or cells can cover the region where the belief has mass.',
    cost: 'Scales with the number of hypotheses, and with the dimension of the state.',
    breaks: 'High dimensions. The number of samples needed to cover a space grows exponentially with it.',
  },
  {
    id: 'kf',
    label: 'Kalman filter',
    sub: 'linear + Gaussian',
    x: 10,
    y: 68,
    chapter: 6,
    slug: 'ch06-kalman-filters',
    represents: 'A Gaussian, propagated in closed form.',
    assumes: 'Linear motion and measurement models with additive Gaussian noise.',
    cost: 'O(n³) in the state dimension, but with tiny constants. Milliseconds for a rover.',
    breaks: 'Any nonlinearity at all — which includes every rotation a real robot performs.',
  },
  {
    id: 'ekf',
    label: 'EKF / UKF',
    sub: 'linearize or sample',
    x: 32,
    y: 68,
    chapter: 7,
    slug: 'ch07-ekf-ukf-manifolds',
    represents: 'A Gaussian, propagated through a locally linear approximation of the model.',
    assumes: 'The models are near-linear over the width of the belief.',
    cost: 'Same order as the KF, plus Jacobians (EKF) or 2n+1 model evaluations (UKF).',
    breaks: 'Strong nonlinearity or large uncertainty. The EKF is where most filters silently go wrong.',
  },
  {
    id: 'hist',
    label: 'Histogram filter',
    sub: 'discretize the space',
    x: 62,
    y: 68,
    chapter: 8,
    slug: 'ch08-nonparametric-filters',
    represents: 'A probability per cell of a grid over the state space.',
    assumes: 'The discretization is fine enough that within-cell variation does not matter.',
    cost: 'O(cells²) for the naive prediction. Brutal beyond three dimensions.',
    breaks: 'Dimension. A 3-D pose grid at useful resolution is already millions of cells.',
  },
  {
    id: 'pf',
    label: 'Particle filter',
    sub: 'sample the belief',
    x: 88,
    y: 68,
    chapter: 8,
    slug: 'ch08-nonparametric-filters',
    represents: 'A weighted set of state hypotheses, resampled to follow the mass.',
    assumes: 'That you can sample from the motion model and evaluate the measurement likelihood.',
    cost: 'O(M) per step in the particle count. Trivially parallel.',
    breaks: 'Particle deprivation — the true state ends up with no particles near it, and no amount of resampling brings it back.',
  },
];

const EDGES: [string, string][] = [
  ['bayes', 'gauss'],
  ['bayes', 'nonparam'],
  ['gauss', 'kf'],
  ['gauss', 'ekf'],
  ['nonparam', 'hist'],
  ['nonparam', 'pf'],
];

export function FilterFamilyTree() {
  const [selected, setSelected] = useState<string>('bayes');
  const node = NODES.find((n) => n.id === selected)!;

  return (
    <WidgetFrame
      id="w5.3"
      title="The filter family tree"
      teaches="Every practical filter is the Bayes filter plus one decision about how to represent a belief — and every decision costs something."
      caption={
        <>
          Select any filter to see what it represents, what it assumes, what it costs, and — the
          entry that matters most in practice — how it fails. Part II of this book walks this tree
          from the root down.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1.35fr_1fr] md:divide-x md:divide-fd-border">
        <svg
          viewBox="0 0 100 82"
          className="w-full"
          role="img"
          aria-label="A tree diagram: the Bayes filter branches into the Gaussian family, leading to the Kalman filter and EKF/UKF, and the nonparametric family, leading to the histogram and particle filters."
        >
          {EDGES.map(([from, to]) => {
            const a = NODES.find((n) => n.id === from)!;
            const b = NODES.find((n) => n.id === to)!;
            const onPath = selected === to || selected === from;
            return (
              <path
                key={`${from}-${to}`}
                d={`M ${a.x} ${a.y + 7} C ${a.x} ${(a.y + b.y) / 2 + 6}, ${b.x} ${(a.y + b.y) / 2 - 2}, ${b.x} ${b.y - 6}`}
                fill="none"
                stroke={onPath ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                strokeWidth={onPath ? 0.7 : 0.5}
              />
            );
          })}

          {NODES.map((n) => {
            const active = n.id === selected;
            return (
              <g
                key={n.id}
                transform={`translate(${n.x} ${n.y})`}
                onClick={() => setSelected(n.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelected(n.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={active}
                aria-label={`${n.label}: ${n.sub}`}
                className="cursor-pointer focus-visible:outline-none"
              >
                <rect
                  x={-11.5}
                  y={-6}
                  width={23}
                  height={12}
                  rx={1.5}
                  fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                  stroke={active ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                  strokeWidth={0.5}
                />
                <text
                  y={-1.2}
                  textAnchor="middle"
                  style={{ fontSize: 3.1, fontWeight: 600 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-canvas-ink)'}
                >
                  {n.label}
                </text>
                <text
                  y={3.2}
                  textAnchor="middle"
                  style={{ fontSize: 2.4 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
                >
                  {n.sub}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="border-t border-fd-border p-4 md:border-t-0">
          <div className="flex items-baseline gap-2">
            <h4 className="font-display text-base font-semibold">{node.label}</h4>
            {node.chapter ? (
              <Link
                href={`/chapters/${node.slug}`}
                className="font-mono text-[0.7rem] text-fd-primary hover:underline"
              >
                Ch. {node.chapter}
              </Link>
            ) : null}
          </div>

          <dl className="mt-3 space-y-2.5">
            {[
              ['represents', node.represents],
              ['assumes', node.assumes],
              ['costs', node.cost],
              ['breaks when', node.breaks],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="eyebrow">{k}</dt>
                <dd
                  className="font-prose text-[0.84rem] leading-snug"
                  style={k === 'breaks when' ? { color: 'var(--pr-prediction)' } : undefined}
                >
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </WidgetFrame>
  );
}
