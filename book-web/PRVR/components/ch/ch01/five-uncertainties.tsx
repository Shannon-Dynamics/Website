'use client';

import Link from 'next/link';
import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { APARTMENT } from '@/lib/sim/world';

/**
 * w1.3 — Five Uncertainties, One Apartment.
 *
 * Thrun's taxonomy of where uncertainty comes from, pinned to five specific
 * places in the world the rest of this book runs in. The floorplan is not a
 * drawing: it is `APARTMENT.walls` from the simulator, rendered as SVG, so the
 * figure cannot drift away from the world the widgets simulate.
 */

const W = APARTMENT.bounds.maxX;
const H = APARTMENT.bounds.maxY;

/** SVG y grows downward; the simulator's y grows up, as in the equations. */
const Y = (y: number) => H - y;

interface Source {
  id: number;
  name: string;
  /** Thrun's own one-line characterisation of the source. */
  taxonomy: string;
  /** Where it lives in this apartment, concretely. */
  here: string;
  /** What the book does about it, and where. */
  answer: string;
  chapter: string;
  slug: string;
  x: number;
  y: number;
}

const SOURCES: Source[] = [
  {
    id: 1,
    name: 'Environments',
    taxonomy: 'Physical worlds are inherently unpredictable — homes and highways far more than assembly lines.',
    here: 'Someone walks down the corridor while Rusty is scanning it. Half the beams come back short, and nothing about the map explains why.',
    answer: 'Model the surprise instead of forbidding it: a mixture component for unexpected obstacles, and a filter that can survive readings it cannot explain.',
    chapter: 'Chapter 12',
    slug: 'ch12-localization-global',
    x: 7.0,
    y: 4.4,
  },
  {
    id: 2,
    name: 'Sensors',
    taxonomy: 'Sensors are limited by physics in range and resolution, and corrupted by noise on top of that.',
    here: 'The study window. A LiDAR beam hits glass, keeps going, and reports max range — the robot reads "nothing there" exactly where a wall is.',
    answer: 'A measurement model with explicit probability mass for dropouts and random returns, rather than a Gaussian that pretends they never happen.',
    chapter: 'Chapter 10',
    slug: 'ch10-sensor-models',
    x: 1.4,
    y: 8.55,
  },
  {
    id: 3,
    name: 'Actuation',
    taxonomy: 'Robot actuation involves motors that are, at least to some extent, unpredictable — control noise, backlash, wear.',
    here: 'The rug in room A. One wheel rides up on it, slips, and Rusty turns three degrees it never commanded and never measures.',
    answer: 'Treat the commanded motion as the mean of a distribution over poses, with noise parameters α₁…α₆ you can measure from your own hardware.',
    chapter: 'Chapter 9',
    slug: 'ch09-motion-models',
    x: 2.1,
    y: 1.9,
  },
  {
    id: 4,
    name: 'Models',
    taxonomy: 'Models are abstractions of the real world; they model the underlying physics only partially, and often crudely.',
    here: 'The kitchen counter is a line segment two centimetres thick, in a map whose cells are independent of one another. Both claims are false, and the robot believes them anyway.',
    answer: 'Say out loud which assumption each algorithm consumes, then show what breaks when the world violates it — and what the honest repair costs.',
    chapter: 'Chapter 13',
    slug: 'ch13-occupancy-grids',
    x: 5.2,
    y: 1.4,
  },
  {
    id: 5,
    name: 'Computation',
    taxonomy: 'Robots are real-time systems, which bounds the computation available: most algorithms here are approximate on purpose.',
    here: 'Rusty itself. One animation frame is 16 milliseconds — enough for a few thousand particles, not for the exact posterior over every pose in the flat.',
    answer: 'Choose a representation with a known cost, then adapt the sample size to the belief instead of praying that a fixed one is enough.',
    chapter: 'Chapter 8',
    slug: 'ch08-nonparametric-filters',
    x: 3.2,
    y: 4.4,
  },
];

export function FiveUncertainties() {
  const [selected, setSelected] = useState(1);
  const source = SOURCES.find((s) => s.id === selected) ?? SOURCES[0];

  return (
    <WidgetFrame
      id="w1.3"
      title="Five uncertainties, one apartment"
      teaches="Uncertainty is not one thing that better engineering removes; it is five different things, each with its own address in the world and its own chapter in this book."
      caption={
        <>
          The floorplan is the Apartment every simulation in this book runs in, drawn straight from
          the wall list the simulator ray-casts against. Select a pin to see how Thrun characterises
          that source, where it lives in this particular flat, and which chapter answers it. Four of
          the five are
          properties of the world; the fifth is a property of your computer, and it is the one that
          decides which algorithms in this book you are allowed to use.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1.25fr_1fr] md:divide-x md:divide-fd-border">
        <div className="p-3">
          <svg
            viewBox={`-0.6 -0.6 ${W + 1.2} ${H + 1.2}`}
            className="w-full"
            role="img"
            aria-label="Floorplan of the apartment with five numbered pins: a person in the corridor, a window in the study, a rug in room A, the kitchen counter, and the robot itself."
          >
            {/* Room floors, so the walls read as a plan rather than a scribble. */}
            <rect
              x={0}
              y={0}
              width={W}
              height={H}
              fill="var(--pr-free)"
              stroke="none"
              opacity={0.6}
            />

            {/* The rug: soft, slippery, and completely invisible to the map. */}
            <rect
              x={1.2}
              y={Y(2.7)}
              width={1.9}
              height={1.4}
              fill="var(--pr-prediction)"
              opacity={0.14}
              stroke="var(--pr-prediction)"
              strokeWidth={0.05}
              strokeDasharray="0.2 0.15"
            />

            {/* The window: a stretch of exterior wall that a laser goes straight through. */}
            <line
              x1={0.6}
              y1={Y(H)}
              x2={2.2}
              y2={Y(H)}
              stroke="var(--pr-measurement)"
              strokeWidth={0.22}
              strokeLinecap="butt"
              opacity={0.55}
            />

            {APARTMENT.walls.map((s, i) => (
              <line
                key={i}
                x1={s.x1}
                y1={Y(s.y1)}
                x2={s.x2}
                y2={Y(s.y2)}
                stroke="var(--pr-wall)"
                strokeWidth={0.12}
                strokeLinecap="round"
              />
            ))}

            {/* A person crossing the corridor — the environment, moving. */}
            <g opacity={0.8}>
              <circle cx={7.0} cy={Y(4.62)} r={0.16} fill="var(--pr-truth)" />
              <path
                d={`M 7.0 ${Y(4.44)} L 7.0 ${Y(4.1)} M 6.82 ${Y(4.35)} L 7.18 ${Y(4.35)}`}
                stroke="var(--pr-truth)"
                strokeWidth={0.09}
                fill="none"
                strokeLinecap="round"
              />
            </g>

            {/* Rusty, facing down the corridor. */}
            <path
              d={`M ${3.2 + 0.34} ${Y(4.4)} L ${3.2 - 0.24} ${Y(4.4) - 0.24} L ${3.2 - 0.1} ${Y(4.4)} L ${3.2 - 0.24} ${Y(4.4) + 0.24} Z`}
              fill="var(--color-fd-primary)"
            />

            {SOURCES.map((s) => {
              const active = s.id === selected;
              return (
                <g
                  key={s.id}
                  transform={`translate(${s.x} ${Y(s.y)})`}
                  onClick={() => setSelected(s.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setSelected(s.id);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-pressed={active}
                  aria-label={`${s.name}: ${s.here}`}
                  className="cursor-pointer focus-visible:outline-none"
                >
                  {active ? (
                    <circle r={0.62} fill="var(--color-fd-primary)" opacity={0.18} />
                  ) : null}
                  <circle
                    r={0.36}
                    fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                    stroke="var(--color-fd-primary)"
                    strokeWidth={0.07}
                  />
                  <text
                    y={0.14}
                    textAnchor="middle"
                    style={{ fontSize: 0.42, fontWeight: 700 }}
                    fill={active ? 'var(--pr-canvas-bg)' : 'var(--color-fd-primary)'}
                  >
                    {s.id}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        <div className="border-t border-fd-border p-4 md:border-t-0">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-[0.7rem] text-fd-primary">{source.id}</span>
            <h4 className="font-display text-base font-semibold">{source.name}</h4>
          </div>

          <dl className="mt-3 space-y-2.5">
            <div>
              <dt className="eyebrow">the source</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{source.taxonomy}</dd>
            </div>
            <div>
              <dt className="eyebrow">in this apartment</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{source.here}</dd>
            </div>
            <div>
              <dt className="eyebrow">what we do about it</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{source.answer}</dd>
            </div>
          </dl>

          <Link
            href={`/chapters/${source.slug}`}
            className="mt-3 inline-block font-mono text-[0.72rem] text-fd-primary hover:underline"
          >
            {source.chapter} →
          </Link>
        </div>
      </div>
    </WidgetFrame>
  );
}
