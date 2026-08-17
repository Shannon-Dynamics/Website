'use client';

import Link from 'next/link';
import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * w7.4 — the Filter Chooser.
 *
 * The chapter's closing argument, as a map. The two axes are the two factors
 * derivation 2 identifies: how curved the model is over one step, and how wide
 * the belief is. Their *product* is the linearization error, which is why the
 * regions run diagonally rather than along either axis.
 *
 * The strip above the plane is deliberately detached. Choosing a manifold or an
 * invariant error is not a point on this chart — it changes what the axes even
 * mean, and no amount of moving right or up will substitute for it.
 */

interface Region {
  id: string;
  label: string;
  sub: string;
  /** SVG polygon points inside the 0–100 plane. */
  points: string;
  /** Where to put the label inside the region. */
  lx: number;
  ly: number;
  use: string;
  cost: string;
  breaks: string;
  chapter?: { n: number; slug: string; label: string };
}

const REGIONS: Region[] = [
  {
    id: 'kf',
    label: 'Kalman filter',
    sub: 'nothing to linearize',
    points: '0,0 26,0 26,100 0,100',
    lx: 13,
    ly: 52,
    use: 'Models that are linear, or linear enough that the Jacobian does not move over the width of the belief: constant-velocity tracking, altitude from a barometer, a wheel encoder integrated over one millisecond.',
    cost: 'One matrix inverse of the innovation dimension. Nothing else.',
    breaks: 'The moment a heading enters the state. A differential drive is nonlinear in θ at every step, however small the step.',
    chapter: { n: 6, slug: 'ch06-kalman-filters', label: 'Chapter 6' },
  },
  {
    id: 'ekf',
    label: 'EKF',
    sub: 'one tangent is enough',
    points: '26,0 100,0 100,42 26,42',
    lx: 62,
    ly: 20,
    use: 'Curved models kept honest by a tight belief — the usual case for a tracking filter running at 50 Hz with good odometry. This is where the vast majority of deployed filters live, and where they are right to.',
    cost: 'Two Jacobians per step, hand-derived or auto-differentiated. Cheapest of everything that handles nonlinearity at all.',
    breaks: 'When the belief widens: the neglected term is ½ tr(∇²g Σ), so a doubling of σ quadruples the bias. Thrun\'s rule of thumb — σ_θ beyond about 20° and the linearization is finished.',
  },
  {
    id: 'ukf',
    label: 'UKF',
    sub: 'send scouts instead',
    points: '26,42 70,42 70,100 26,100',
    lx: 47,
    ly: 72,
    use: 'A belief wide enough that one tangent misrepresents it, but a model still smooth and unimodal: initialization, a robot re-acquiring after a stop, range-only measurements at closest approach, any model whose Jacobian you cannot face deriving.',
    cost: '2d+1 evaluations of the model and one Cholesky per step. Same asymptotic order as the EKF, roughly 2d+1 times the constant.',
    breaks: 'Multi-modality, and near-linear problems where it buys nothing: on a linear model the UKF returns the KF answer, having spent seven model evaluations to do so.',
  },
  {
    id: 'iterate',
    label: 'Iterate',
    sub: 'IEKF → Gauss–Newton',
    points: '70,42 100,42 100,100 70,100',
    lx: 85,
    ly: 72,
    use: 'Strong curvature *and* a wide belief, where a single linearization point is wrong wherever you put it. Re-linearize at the posterior and repeat until it stops moving; that iteration is literally Gauss–Newton, which is why this region is the doorway to the factor-graph chapters.',
    cost: 'One EKF update per iteration, typically two to five. Convergence is not guaranteed — damp it and it becomes Levenberg–Marquardt.',
    breaks: 'It fixes the linearization point, not the belief shape. If the true posterior is a banana, an iterated filter gives you a better-centered ellipse around a banana.',
    chapter: { n: 15, slug: 'ch15-factor-graphs', label: 'Chapter 15' },
  },
];

interface PanelInfo {
  label: string;
  sub?: string;
  use: string;
  cost?: string;
  breaks?: string;
  chapter?: { n: number; slug: string; label: string };
}

const OFF_PLANE: PanelInfo = {
  label: 'Re-parameterize',
  sub: 'manifold state, invariant error',
  use: 'Not a region of this plane. If the state is a rotation, a pose, or anything else that is not a vector space, no amount of better linearization helps — the arithmetic itself is wrong. Put the mean on the manifold and the covariance in its tangent space with ⊞/⊟, and then choose your error definition: a group error η = μ⁻¹x makes the Jacobians state-independent for group-affine models, which is the invariant EKF.',
  cost: 'An exp/log pair and an adjoint. In wall-clock terms, nothing.',
  breaks: 'Nothing about it fixes curvature. An on-manifold EKF still linearizes; it just linearizes something meaningful.',
  chapter: { n: 14, slug: 'ch14-ekf-slam', label: 'Chapter 14' },
};

interface Demo {
  id: string;
  x: number;
  y: number;
  label: string;
  role: 'prediction' | 'measurement' | 'posterior' | 'prior';
  detail: string;
}

const DEMOS: Demo[] = [
  {
    id: 'quad',
    x: 14,
    y: 18,
    label: 'x²/20, σ = 0.3',
    role: 'prior',
    detail: 'w7.1 with the quadratic and a tight input: EKF bias 0.0045, UT bias 0. Nobody would notice either.',
  },
  {
    id: 'polar',
    x: 54,
    y: 28,
    label: 'polar fix, σ_θ = 15°',
    role: 'prediction',
    detail: 'The chapter\'s worked example. The EKF is 3.4 cm out at 1 m range and claims 2 cm of spread where the truth is 5 cm. Five sigma points fix both.',
  },
  {
    id: 'beacon',
    x: 63,
    y: 66,
    label: 'range beacon, σ = 0.9 m',
    role: 'measurement',
    detail: 'w7.1 with the range curve at closest approach. The EKF slope is zero there, so it reports almost no output uncertainty at all — a textbook overconfident divergence in the making.',
  },
  {
    id: 'slam',
    x: 86,
    y: 79,
    label: 'EKF SLAM, 200 steps',
    role: 'posterior',
    detail: 'Chapter 14. Every landmark re-linearized about a drifting estimate; the accumulated inconsistency is what ended the filtering era of SLAM — and what invariant filtering reframes.',
  },
];

export function FilterChooser() {
  const [selected, setSelected] = useState<string>('ekf');
  const panel: PanelInfo = (() => {
    if (selected === 'manifold') return OFF_PLANE;
    const r = REGIONS.find((x) => x.id === selected);
    if (r) {
      return { label: r.label, sub: r.sub, use: r.use, cost: r.cost, breaks: r.breaks, chapter: r.chapter };
    }
    const d = DEMOS.find((x) => x.id === selected);
    if (d) return { label: d.label, sub: 'a demo from this chapter', use: d.detail };
    return OFF_PLANE;
  })();

  return (
    <WidgetFrame
      id="w7.4"
      title="Filter Chooser"
      teaches="UKF is not better than EKF. It is better in one corner of a plane, and the corner is set by curvature times spread."
      caption={
        <>
          The two axes are the two factors in the bias term ½&nbsp;tr(∇²g&nbsp;Σ): how sharply the
          model bends over one step, and how wide the belief is. Neither one alone decides anything
          — it is the product that hurts, which is why a filter can be perfectly well behaved for an
          hour and fall apart in the ten seconds after a bad measurement widens it. Click a region
          for what it buys and what it costs, or a labelled point for one of this chapter&apos;s
          demos. The strip along the top is deliberately not part of the plane.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1.25fr_1fr] md:divide-x md:divide-fd-border">
        <svg
          viewBox="-16 -26 128 132"
          className="w-full"
          role="img"
          aria-label="A plane whose horizontal axis is model curvature and whose vertical axis is belief spread, divided into regions for the Kalman filter, the EKF, the UKF, and iterated filters, with a detached strip above it for re-parameterizing the state onto a manifold."
        >
          {/* the off-plane strip */}
          <g
            onClick={() => setSelected('manifold')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setSelected('manifold');
              }
            }}
            tabIndex={0}
            role="button"
            aria-pressed={selected === 'manifold'}
            aria-label="Re-parameterize: manifold state and invariant error"
            className="cursor-pointer focus-visible:outline-none"
          >
            <rect
              x={0}
              y={-22}
              width={100}
              height={13}
              rx={1.5}
              fill={selected === 'manifold' ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
              stroke="var(--color-fd-primary)"
              strokeWidth={0.6}
              strokeDasharray="3 2"
            />
            <text
              x={50}
              y={-16.5}
              textAnchor="middle"
              style={{ fontSize: 4, fontWeight: 600 }}
              fill={selected === 'manifold' ? 'var(--pr-canvas-bg)' : 'var(--color-fd-primary)'}
            >
              is the state even a vector space?
            </text>
            <text
              x={50}
              y={-11.5}
              textAnchor="middle"
              style={{ fontSize: 3.1 }}
              fill={selected === 'manifold' ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
            >
              ⊞ / ⊟ · error-state · invariant EKF — answer this before reading the plane
            </text>
          </g>

          {/* regions */}
          {REGIONS.map((r) => {
            const active = r.id === selected;
            return (
              <g
                key={r.id}
                onClick={() => setSelected(r.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelected(r.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={active}
                aria-label={`${r.label}: ${r.sub}`}
                className="cursor-pointer focus-visible:outline-none"
              >
                <polygon
                  points={r.points}
                  fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                  fillOpacity={active ? 0.85 : 1}
                  stroke="var(--pr-grid)"
                  strokeWidth={0.6}
                />
                <text
                  x={r.lx}
                  y={r.ly}
                  textAnchor="middle"
                  style={{ fontSize: 4.4, fontWeight: 600 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-canvas-ink)'}
                >
                  {r.label}
                </text>
                <text
                  x={r.lx}
                  y={r.ly + 4.6}
                  textAnchor="middle"
                  style={{ fontSize: 3 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
                >
                  {r.sub}
                </text>
              </g>
            );
          })}

          {/* the iso-error diagonal: curvature × spread = constant */}
          <path
            d="M 26 100 Q 55 46 100 34"
            fill="none"
            stroke="var(--pr-truth)"
            strokeWidth={0.5}
            strokeDasharray="2 2"
          />
          <text x={97} y={31} textAnchor="end" style={{ fontSize: 2.7 }} fill="var(--pr-truth)">
            constant curvature × spread
          </text>

          {/* the chapter's demos */}
          {DEMOS.map((d) => {
            const active = d.id === selected;
            return (
              <g
                key={d.id}
                transform={`translate(${d.x} ${100 - d.y})`}
                onClick={() => setSelected(d.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelected(d.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={active}
                aria-label={d.label}
                className="cursor-pointer focus-visible:outline-none"
              >
                <circle
                  r={active ? 2.4 : 1.5}
                  fill={`var(--pr-${d.role})`}
                  stroke="var(--pr-canvas-bg)"
                  strokeWidth={0.6}
                />
                <text
                  x={0}
                  y={-3.2}
                  textAnchor="middle"
                  style={{ fontSize: 2.7, fontWeight: active ? 600 : 400 }}
                  fill={`var(--pr-${d.role})`}
                >
                  {d.label}
                </text>
              </g>
            );
          })}

          {/* axes */}
          <line x1={0} y1={100} x2={104} y2={100} stroke="var(--pr-canvas-ink)" strokeWidth={0.6} />
          <line x1={0} y1={100} x2={0} y2={-4} stroke="var(--pr-canvas-ink)" strokeWidth={0.6} />
          <text x={52} y={107.5} textAnchor="middle" style={{ fontSize: 3.6 }} fill="var(--pr-canvas-ink)">
            model curvature over one step  →
          </text>
          <text
            x={-52}
            y={-6}
            transform="rotate(-90)"
            textAnchor="middle"
            style={{ fontSize: 3.6 }}
            fill="var(--pr-canvas-ink)"
          >
            belief spread  →
          </text>
        </svg>

        <div className="border-t border-fd-border p-4 md:border-t-0">
          <div className="flex items-baseline gap-2">
            <h4 className="font-display text-base font-semibold">{panel.label}</h4>
            {panel.sub ? (
              <span className="font-mono text-[0.7rem] text-fd-muted-foreground">{panel.sub}</span>
            ) : null}
            {panel.chapter ? (
              <Link
                href={`/chapters/${panel.chapter.slug}`}
                className="ms-auto font-mono text-[0.7rem] text-fd-primary hover:underline"
              >
                Ch. {panel.chapter.n}
              </Link>
            ) : null}
          </div>

          <dl className="mt-3 space-y-2.5">
            {(
              [
                ['use it when', panel.use],
                ['costs', panel.cost],
                ['breaks when', panel.breaks],
              ] as [string, string | undefined][]
            )
              .filter(([, v]) => Boolean(v))
              .map(([k, v]) => (
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
