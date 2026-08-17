'use client';

import Link from 'next/link';
import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * f18.1 — ORB-SLAM3 and VINS-Fusion, side by side.
 *
 * Real systems are described in this chapter, not dissected. The point of the
 * figure is that neither of them contains an idea this book has not already
 * taught: every block is either a measurement model (green), a piece of
 * estimation machinery (purple), or a front end we deliberately leave to the
 * computer-vision literature (gray).
 *
 * Sources for the block structure: Campos et al., *ORB-SLAM3*, IEEE T-RO 37(6),
 * 2021; Qin, Li & Shen, *VINS-Mono*, IEEE T-RO 34(4), 2018.
 */

type Kind = 'front' | 'back' | 'external';

interface Block {
  id: string;
  col: 0 | 1;
  row: number;
  label: string;
  sub: string;
  kind: Kind;
  what: string;
  book: string;
  slug?: string;
  price: string;
}

const BLOCKS: Block[] = [
  /* ---- ORB-SLAM3 ------------------------------------------------------- */
  {
    id: 'orb-feat',
    col: 0,
    row: 0,
    label: 'ORB extraction & matching',
    sub: 'front end',
    kind: 'external',
    what: 'Binary corner features, matched frame-to-frame and against the map by descriptor distance. Everything downstream treats a match as a pixel measurement with σ ≈ 1 px.',
    book: 'Deliberately not taught here — see Hartley & Zisserman or Szeliski. Chapter 25 revisits it as the place where learning enters a probabilistic stack.',
    slug: 'ch25-learning',
    price: 'Wrong matches are not Gaussian. Everything above assumes they were, so the back end needs robust kernels and a gate.',
  },
  {
    id: 'orb-track',
    col: 0,
    row: 1,
    label: 'Tracking: motion-only BA',
    sub: 'pose against a fixed map',
    kind: 'back',
    what: 'Optimize the current camera pose alone, holding the map fixed, by minimizing the reprojection error of the matched map points. Six unknowns, hundreds of residuals.',
    book: 'This chapter: the reprojection factor and its 2×6 Jacobian, solved with the Gauss–Newton of Chapter 15.',
    slug: 'ch15-factor-graphs',
    price: 'It is only as good as the map it trusts, and it inherits any drift already in it.',
  },
  {
    id: 'orb-local',
    col: 0,
    row: 2,
    label: 'Local mapping: local BA',
    sub: 'covisible keyframes + points',
    kind: 'back',
    what: 'Bundle-adjust the keyframes that share observations with the newest one, plus the points they see, freezing everything else as a fixed boundary.',
    book: 'This chapter: bundle adjustment with the points Schur-eliminated. The covisibility graph is exactly the fill-in pattern of the reduced system.',
    price: 'The boundary is a lie of convenience: frozen neighbours act as an infinitely stiff prior.',
  },
  {
    id: 'orb-imu',
    col: 0,
    row: 3,
    label: 'IMU: preintegration + MAP init',
    sub: 'inertial residuals & biases',
    kind: 'front',
    what: 'Forster-style preintegrated factors between keyframes, with an inertial-only maximum-a-posteriori initialization that solves for scale, gravity direction, and biases before the visual-inertial optimization starts.',
    book: 'This chapter, Derivation 4. The bias is in the state because the Markov assumption demanded it (Chapter 5).',
    slug: 'ch05-bayes-filter',
    price: 'Initialization needs motion with genuine excitation; a hovering drone cannot observe scale.',
  },
  {
    id: 'orb-atlas',
    col: 0,
    row: 4,
    label: 'Atlas: multiple maps',
    sub: 'active + dormant sub-maps',
    kind: 'back',
    what: 'When tracking is lost, start a new map rather than corrupting the old one; merge maps later if place recognition finds a link between them.',
    book: 'The multi-hypothesis instinct of Chapter 12, applied to maps instead of poses.',
    slug: 'ch12-localization-global',
    price: 'Merging is a graph surgery, and a wrong merge is unrecoverable.',
  },
  {
    id: 'orb-loop',
    col: 0,
    row: 5,
    label: 'Place recognition → loop & merge',
    sub: 'DBoW2 + full BA',
    kind: 'back',
    what: 'Bag-of-words candidates, verified geometrically, then a pose-graph correction followed by full bundle adjustment.',
    book: 'Chapter 16: loop closure as the constraint that converts drift into a global correction.',
    slug: 'ch16-scan-matching',
    price: 'One false positive rewrites the map. Verification, not detection, is the hard part.',
  },

  /* ---- VINS-Fusion ------------------------------------------------------ */
  {
    id: 'vins-klt',
    col: 1,
    row: 0,
    label: 'KLT feature tracking',
    sub: 'front end',
    kind: 'external',
    what: 'Shi–Tomasi corners tracked by optical flow, with outlier rejection by a fundamental-matrix RANSAC.',
    book: 'Not taught here. The measurement it emits is the same bearing this chapter models.',
    price: 'Optical flow assumes small motion and constant brightness — both fail exactly when the drone is interesting.',
  },
  {
    id: 'vins-preint',
    col: 1,
    row: 1,
    label: 'IMU preintegration',
    sub: 'one factor per keyframe pair',
    kind: 'front',
    what: 'The same Lupton/Forster compression: hundreds of samples become a nine-dimensional relative constraint with a 9×9 covariance and stored bias Jacobians.',
    book: 'This chapter, Derivation 4 and widget w18.2.',
    price: 'The bias Jacobian is a first-order model. Let the bias estimate wander far enough and you must re-integrate.',
  },
  {
    id: 'vins-init',
    col: 1,
    row: 2,
    label: 'Loosely-coupled initialization',
    sub: 'SfM ⊕ IMU alignment',
    kind: 'back',
    what: 'Run a vision-only structure-from-motion on a window, then align it with the preintegrated inertial deltas to recover metric scale, the gravity direction, velocities, and the gyro bias.',
    book: 'This chapter, Derivation 3: monocular scale is unobservable until the accelerometer supplies a metre.',
    price: 'It is a bootstrap, not an estimator; a bad initialization poisons everything that follows.',
  },
  {
    id: 'vins-window',
    col: 1,
    row: 3,
    label: 'Sliding-window optimization',
    sub: 'tightly coupled + marginalization prior',
    kind: 'back',
    what: 'One nonlinear least-squares problem over the recent keyframes, their velocities and biases, and the tracked features — with a dense prior standing in for everything already marginalized out.',
    book: 'This chapter, Derivation 5, and widget w18.3. The solver is Chapter 15’s.',
    slug: 'ch15-factor-graphs',
    price: 'The prior freezes its linearization point. Consistency needs first-estimates Jacobians.',
  },
  {
    id: 'vins-loop',
    col: 1,
    row: 4,
    label: 'Loop detection & relocalization',
    sub: 'DBoW2 + tightly-coupled retrieval',
    kind: 'back',
    what: 'Matched loop features are added to the current window as extra reprojection factors against a fixed past pose.',
    book: 'Chapter 16, in a sliding-window costume.',
    slug: 'ch16-scan-matching',
    price: 'Adding a loop factor inside the window can yank the estimate; VINS keeps the loop pose fixed to avoid it.',
  },
  {
    id: 'vins-pgo',
    col: 1,
    row: 5,
    label: '4-DOF pose graph',
    sub: 'x, y, z, yaw',
    kind: 'back',
    what: 'A global pose graph that optimizes only four degrees of freedom, because gravity has already made roll and pitch observable.',
    book: 'Chapter 15 machinery, with the observability argument of this chapter deciding which variables are even in the problem.',
    price: 'Global position and yaw remain unobservable forever; the graph fixes drift, not the gauge.',
  },
];

const COLUMNS = [
  { title: 'ORB-SLAM3', sub: 'feature-based, multi-map, T-RO 2021' },
  { title: 'VINS-Fusion', sub: 'optical-flow, sliding-window, T-RO 2018' },
];

const KIND_FILL: Record<Kind, string> = {
  front: 'var(--pr-measurement)',
  back: 'var(--pr-posterior)',
  external: 'var(--pr-truth)',
};

const KIND_LABEL: Record<Kind, string> = {
  front: 'measurement model',
  back: 'estimation machinery',
  external: 'front end, not taught here',
};

const COL_X = [26, 74];
const ROW_Y = [10, 23, 36, 49, 62, 75];
const NODE_W = 42;
const NODE_H = 10;

export function SystemArchitectures() {
  const [selected, setSelected] = useState('vins-window');
  const node = BLOCKS.find((b) => b.id === selected)!;

  return (
    <WidgetFrame
      id="f18.1"
      title="ORB-SLAM3 and VINS architectures"
      teaches="Two systems, no new ideas: both are compositions of blocks this book has already built — the engineering is in the plumbing, not in a hidden theory."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          Select any block to see what it does, which chapter taught it, and what it costs. Green
          blocks turn photons and specific force into likelihoods; purple blocks are estimation;
          gray is the feature front end this book deliberately leaves to the computer-vision
          literature. The two columns disagree about almost every engineering decision and about
          nothing in the mathematics.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1.5fr_1fr] md:divide-x md:divide-fd-border">
        <svg
          viewBox="0 0 100 92"
          className="w-full"
          role="img"
          aria-label="Two block diagrams side by side: ORB-SLAM3 with ORB extraction, tracking, local mapping, IMU preintegration with MAP initialization, the Atlas multi-map, and place recognition; VINS-Fusion with KLT tracking, IMU preintegration, loosely-coupled initialization, sliding-window optimization, loop detection, and a four-degree-of-freedom pose graph."
        >
          {COLUMNS.map((c, i) => (
            <g key={c.title}>
              <text
                x={COL_X[i]}
                y={3.4}
                textAnchor="middle"
                style={{ fontSize: 3.6, fontWeight: 700 }}
                fill="var(--pr-canvas-ink)"
              >
                {c.title}
              </text>
              <text
                x={COL_X[i]}
                y={6.6}
                textAnchor="middle"
                style={{ fontSize: 2.5 }}
                fill="var(--pr-truth)"
              >
                {c.sub}
              </text>
            </g>
          ))}

          {/* Flow arrows down each column. */}
          {COL_X.map((x, col) =>
            ROW_Y.slice(0, -1).map((y, r) => (
              <line
                key={`arrow-${col}-${r}`}
                x1={x}
                y1={y + NODE_H}
                x2={x}
                y2={ROW_Y[r + 1]}
                stroke="var(--pr-grid)"
                strokeWidth={0.5}
              />
            )),
          )}

          {BLOCKS.map((b) => {
            const active = b.id === selected;
            const x = COL_X[b.col] - NODE_W / 2;
            const y = ROW_Y[b.row];
            return (
              <g
                key={b.id}
                onClick={() => setSelected(b.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelected(b.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={active}
                aria-label={`${b.label}: ${b.sub}`}
                className="cursor-pointer focus-visible:outline-none"
              >
                <rect
                  x={x}
                  y={y}
                  width={NODE_W}
                  height={NODE_H}
                  rx={1.4}
                  fill={active ? KIND_FILL[b.kind] : 'var(--pr-canvas-bg)'}
                  fillOpacity={active ? 0.16 : 1}
                  stroke={KIND_FILL[b.kind]}
                  strokeWidth={active ? 0.9 : 0.4}
                />
                <rect x={x} y={y} width={1.1} height={NODE_H} rx={0.4} fill={KIND_FILL[b.kind]} />
                <text
                  x={x + 3}
                  y={y + 4.4}
                  style={{ fontSize: 2.9, fontWeight: 600 }}
                  fill="var(--pr-canvas-ink)"
                >
                  {b.label}
                </text>
                <text x={x + 3} y={y + 7.9} style={{ fontSize: 2.4 }} fill="var(--pr-truth)">
                  {b.sub}
                </text>
              </g>
            );
          })}

          <text
            x={50}
            y={90}
            textAnchor="middle"
            style={{ fontSize: 2.5 }}
            fill="var(--pr-truth)"
          >
            shared machinery: SE(3) (Ch. 3) · error-state filtering (Ch. 7) · sparse least squares
            (Ch. 15) · loop closure (Ch. 16)
          </text>
        </svg>

        <div className="border-t border-fd-border p-4 md:border-t-0">
          <div className="flex items-baseline gap-2">
            <h4 className="font-display text-base font-semibold">{node.label}</h4>
            <span
              className="font-mono text-[0.65rem]"
              style={{ color: KIND_FILL[node.kind] }}
            >
              {KIND_LABEL[node.kind]}
            </span>
          </div>

          <dl className="mt-3 space-y-2.5">
            <div>
              <dt className="eyebrow">what it does</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{node.what}</dd>
            </div>
            <div>
              <dt className="eyebrow">where this book teaches it</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">
                {node.book}
                {node.slug ? (
                  <>
                    {' '}
                    <Link
                      href={`/chapters/${node.slug}`}
                      className="font-mono text-[0.72rem] text-fd-primary hover:underline"
                    >
                      →
                    </Link>
                  </>
                ) : null}
              </dd>
            </div>
            <div>
              <dt className="eyebrow">what it costs</dt>
              <dd
                className="font-prose text-[0.84rem] leading-snug"
                style={{ color: 'var(--pr-prediction)' }}
              >
                {node.price}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </WidgetFrame>
  );
}
