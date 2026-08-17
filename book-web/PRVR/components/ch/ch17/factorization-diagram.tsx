'use client';

import Link from 'next/link';
import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * f17.1 — the factorization, made clickable.
 *
 * The SLAM graphical model with one switch: is the path a random variable, or
 * is it given? Marginalizing it couples every landmark to every other landmark
 * (Chapter 14's dense covariance). Conditioning on it — which is what a
 * particle does — severs those couplings and leaves N independent 2×2
 * problems. The whole chapter is this one toggle.
 */

interface Pose {
  id: string;
  label: string;
  x: number;
}

interface Landmark {
  id: string;
  label: string;
  x: number;
  y: number;
  seenFrom: string[];
  note: string;
}

const POSE_Y = 74;
const POSES: Pose[] = [
  { id: 'x0', label: 'x₀', x: 9 },
  { id: 'x1', label: 'x₁', x: 25 },
  { id: 'x2', label: 'x₂', x: 41 },
  { id: 'x3', label: 'x₃', x: 57 },
  { id: 'x4', label: 'x₄', x: 73 },
  { id: 'x5', label: 'x₅', x: 89 },
];

const LANDMARKS: Landmark[] = [
  {
    id: 'm1',
    label: 'm₁',
    x: 20,
    y: 24,
    seenFrom: ['x0', 'x1', 'x2'],
    note: 'Seen three times early in the run. Its posterior is a 2×2 Gaussian built from three range–bearing readings taken at three known (per particle) poses.',
  },
  {
    id: 'm2',
    label: 'm₂',
    x: 50,
    y: 15,
    seenFrom: ['x2', 'x3'],
    note: 'Seen from x₂ and x₃. It shares x₂ with m₁ and x₃ with m₃ — which is exactly how a *marginalized* path would correlate all three.',
  },
  {
    id: 'm3',
    label: 'm₃',
    x: 80,
    y: 25,
    seenFrom: ['x3', 'x4', 'x5'],
    note: 'Seen at the end of the run. In EKF SLAM it is nevertheless correlated with m₁, which it has never co-occurred with, purely through the shared trajectory.',
  },
];

const INDUCED: [string, string][] = [
  ['m1', 'm2'],
  ['m2', 'm3'],
  ['m1', 'm3'],
];

export function FactorizationDiagram() {
  const [conditioned, setConditioned] = useState(false);
  const [selected, setSelected] = useState<string>('m2');
  const landmark = LANDMARKS.find((l) => l.id === selected) ?? LANDMARKS[1];
  const poseOf = (id: string) => POSES.find((p) => p.id === id)!;

  return (
    <WidgetFrame
      id="f17.1"
      title="The factorization"
      teaches="Landmarks are correlated only through the robot's path. Freeze the path and the map falls apart into independent pieces."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Toggle between the two ways of handling the trajectory. <em>Marginalize</em> it — integrate
          it out, as EKF SLAM does — and every landmark becomes correlated with every other one
          (the purple arcs), giving the dense (3+2N)×(3+2N) covariance of
          <Link href="/chapters/ch14-ekf-slam"> Chapter 14</Link>. <em>Condition</em> on it — assume it
          known, as a single particle does — and the arcs vanish: the map is N independent 2×2
          problems, and the particle can solve each one in closed form. Click a landmark to see
          which poses observed it.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1.45fr_1fr] md:divide-x md:divide-fd-border">
        <svg
          viewBox="0 0 100 92"
          className="w-full"
          role="img"
          aria-label="A graphical model: six robot poses in a row joined by motion edges, with three landmarks above joined to the poses that observed them. When the path is marginalized, dashed arcs connect every pair of landmarks; when the path is conditioned on, those arcs disappear."
        >
          {/* Induced landmark–landmark couplings: present only when the path is unknown. */}
          {!conditioned &&
            INDUCED.map(([a, b]) => {
              const la = LANDMARKS.find((l) => l.id === a)!;
              const lb = LANDMARKS.find((l) => l.id === b)!;
              const lift = Math.abs(la.x - lb.x) * 0.35;
              return (
                <path
                  key={`${a}-${b}`}
                  d={`M ${la.x} ${la.y - 5} Q ${(la.x + lb.x) / 2} ${Math.min(la.y, lb.y) - lift} ${lb.x} ${lb.y - 5}`}
                  fill="none"
                  stroke="var(--pr-posterior)"
                  strokeWidth={0.6}
                  strokeDasharray="1.6 1.4"
                  opacity={0.85}
                />
              );
            })}

          {/* Motion edges. */}
          {POSES.slice(0, -1).map((p, i) => (
            <line
              key={`u${i}`}
              x1={p.x + 4}
              y1={POSE_Y}
              x2={POSES[i + 1].x - 4}
              y2={POSE_Y}
              stroke="var(--pr-prediction)"
              strokeWidth={0.8}
              opacity={conditioned ? 0.25 : 1}
            />
          ))}

          {/* Measurement edges. */}
          {LANDMARKS.flatMap((l) =>
            l.seenFrom.map((pid) => {
              const p = poseOf(pid);
              const active = l.id === selected;
              return (
                <line
                  key={`${l.id}-${pid}`}
                  x1={l.x}
                  y1={l.y + 5}
                  x2={p.x}
                  y2={POSE_Y - 4}
                  stroke="var(--pr-measurement)"
                  strokeWidth={active ? 0.85 : 0.45}
                  opacity={active ? 1 : 0.4}
                />
              );
            }),
          )}

          {/* Poses. Filled when conditioned on — the graphical-model convention. */}
          {POSES.map((p) => (
            <g key={p.id}>
              <circle
                cx={p.x}
                cy={POSE_Y}
                r={4}
                fill={conditioned ? 'var(--pr-truth)' : 'var(--pr-canvas-bg)'}
                stroke={conditioned ? 'var(--pr-truth)' : 'var(--pr-prediction)'}
                strokeWidth={0.7}
              />
              <text
                x={p.x}
                y={POSE_Y + 1.3}
                textAnchor="middle"
                style={{ fontSize: 3.4, fontWeight: 600 }}
                fill={conditioned ? 'var(--pr-canvas-bg)' : 'var(--pr-canvas-ink)'}
              >
                {p.label}
              </text>
            </g>
          ))}

          {/* Landmarks. */}
          {LANDMARKS.map((l) => {
            const active = l.id === selected;
            return (
              <g
                key={l.id}
                onClick={() => setSelected(l.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelected(l.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={active}
                aria-label={`Landmark ${l.label}`}
                className="cursor-pointer focus-visible:outline-none"
              >
                {conditioned ? (
                  <rect
                    x={l.x - 8}
                    y={l.y - 8}
                    width={16}
                    height={16}
                    rx={1.2}
                    fill="none"
                    stroke="var(--pr-posterior)"
                    strokeWidth={0.4}
                    strokeDasharray="1.2 1.2"
                    opacity={0.7}
                  />
                ) : null}
                <rect
                  x={l.x - 5}
                  y={l.y - 5}
                  width={10}
                  height={10}
                  rx={1}
                  fill={active ? 'var(--pr-posterior)' : 'var(--pr-canvas-bg)'}
                  stroke="var(--pr-posterior)"
                  strokeWidth={0.7}
                />
                <text
                  x={l.x}
                  y={l.y + 1.3}
                  textAnchor="middle"
                  style={{ fontSize: 3.4, fontWeight: 600 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-canvas-ink)'}
                >
                  {l.label}
                </text>
              </g>
            );
          })}

          <text x={2} y={POSE_Y + 14} style={{ fontSize: 3 }} fill="var(--pr-truth)">
            {conditioned
              ? 'path given  →  3 independent 2×2 posteriors'
              : 'path marginalized  →  one dense 15×15 covariance'}
          </text>
        </svg>

        <div className="border-t border-fd-border p-4 md:border-t-0">
          <div className="flex flex-wrap gap-1.5">
            {[
              { on: false, label: 'Marginalize the path' },
              { on: true, label: 'Condition on the path' },
            ].map((opt) => (
              <button
                key={String(opt.on)}
                type="button"
                onClick={() => setConditioned(opt.on)}
                aria-pressed={conditioned === opt.on}
                className={
                  conditioned === opt.on
                    ? 'rounded-sm bg-fd-primary px-2.5 py-1 font-ui text-xs font-medium text-fd-primary-foreground'
                    : 'rounded-sm border border-fd-border bg-fd-card px-2.5 py-1 font-ui text-xs font-medium transition-colors hover:bg-fd-accent'
                }
              >
                {opt.label}
              </button>
            ))}
          </div>

          <dl className="mt-3 space-y-2.5">
            <div>
              <dt className="eyebrow">what the estimator holds</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">
                {conditioned
                  ? 'One sampled trajectory plus N tiny Gaussians. Adding a landmark adds four numbers, not a row and a column to a growing matrix.'
                  : 'A single joint Gaussian over the pose and every landmark. Its off-diagonal blocks are the map: they are what make the estimate consistent, and what make it cost O(N²).'}
              </dd>
            </div>
            <div>
              <dt className="eyebrow">cost of one update</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">
                {conditioned
                  ? 'O(1) per observed landmark inside each particle — one 2×2 EKF — so O(M) for the filter, independent of how big the map has grown.'
                  : 'O(N²): the innovation touches the observed landmark, and the correction writes back to every correlated entry.'}
              </dd>
            </div>
            <div>
              <dt className="eyebrow">selected landmark</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">
                <strong>{landmark.label}</strong> — {landmark.note}
              </dd>
            </div>
            <div>
              <dt className="eyebrow">the catch</dt>
              <dd className="font-prose text-[0.84rem] leading-snug" style={{ color: 'var(--pr-prediction)' }}>
                {conditioned
                  ? 'The independence is conditional. It holds only if the particle carries the whole path — which is why a particle here is a trajectory, and why the filter can never forget one.'
                  : 'Nothing is approximated away, but the matrix grows without bound and one bad data association corrupts all of it at once.'}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </WidgetFrame>
  );
}
