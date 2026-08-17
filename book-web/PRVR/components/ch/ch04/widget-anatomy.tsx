'use client';

import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * w4.5 — Anatomy of a Book Widget.
 *
 * The chapter builds the lab; this diagram fixes the contract every later
 * chapter's figures obey. It is deliberately self-referential: the thing being
 * drawn is the frame this diagram is sitting inside.
 */

interface Part {
  id: string;
  label: string;
  /** SVG rect in the schematic, in viewBox units. */
  x: number;
  y: number;
  w: number;
  h: number;
  guarantee: string;
  why: string;
  code: string;
}

const PARTS: Part[] = [
  {
    id: 'header',
    label: 'id · title · teaches',
    x: 3,
    y: 3,
    w: 62,
    h: 9,
    guarantee:
      'A stable id (w4.1) that matches the chapter design document, a title, and one sentence naming the misconception the widget exists to kill.',
    why: 'Widgets get cited across chapters and linked to from exercises. An id that drifts breaks the book; a widget that cannot name what it teaches should not have been built.',
    code: '<WidgetFrame id teaches …>',
  },
  {
    id: 'canvas',
    label: 'world canvas',
    x: 3,
    y: 14,
    w: 62,
    h: 30,
    guarantee:
      'Device-pixel-ratio correct, redrawn on resize and on a light/dark flip, drawn in metres and radians with the pixel mapping owned by the frame.',
    why: 'Every simulation should think in world coordinates. The moment a widget starts computing pixels, its figures stop agreeing with the equations above them.',
    code: '<SimCanvas world={…} draw={(ctx, v, palette) => …} />',
  },
  {
    id: 'readout',
    label: 'live readouts',
    x: 3,
    y: 46,
    w: 62,
    h: 7,
    guarantee:
      'Numbers with units and fixed-width digits, updated every tick, showing the quantity the surrounding prose is arguing about.',
    why: 'A claim like "drift grows with distance" is a claim about a number. Put the number on screen and the reader can falsify you in ten seconds.',
    code: '<StatTile label value unit role sparkline />',
  },
  {
    id: 'panel',
    label: 'one foregrounded parameter',
    x: 3,
    y: 55,
    w: 62,
    h: 8,
    guarantee:
      'Exactly one slider carries the widget. Others may exist, but the primary one is tinted with the role color it controls.',
    why: 'Two competing knobs turn an experiment into a toy. If a widget needs a paragraph to explain its controls, it is two widgets.',
    code: '<Slider role="prediction" … />',
  },
  {
    id: 'transport',
    label: 'transport · visible seed',
    x: 3,
    y: 65,
    w: 62,
    h: 8,
    guarantee:
      'Autoplay on load, play/pause, single-step, reset, re-roll — and the seed printed in the open, so a run can be reproduced or reported.',
    why: 'A hidden seed is an unfalsifiable figure. Determinism is a claim this book makes about its own code, so the evidence sits in the chrome.',
    code: 'useSimulation({ init, step, fps, initialSeed })',
  },
  {
    id: 'legend',
    label: 'color key',
    x: 3,
    y: 75,
    w: 62,
    h: 6,
    guarantee:
      'Prior blue, prediction orange, measurement green, posterior purple, truth gray dashed — pulled from CSS custom properties, never hardcoded.',
    why: 'The same five colors mean the same five things in every equation, figure, and code comment in the book. That consistency is worth more than any single clever picture.',
    code: "colorKey={['prediction', 'truth']}",
  },
  {
    id: 'caption',
    label: 'caption: what to notice, what to try',
    x: 3,
    y: 83,
    w: 62,
    h: 9,
    guarantee:
      'Two things in prose: the observation the reader should make without touching anything, and the one change that breaks it.',
    why: 'Interaction is an invitation, not a requirement. A reader who never clicks should still leave with the lesson; a reader who does should know where to push.',
    code: 'caption={<>…</>}',
  },
];

export function WidgetAnatomy() {
  const [selected, setSelected] = useState('transport');
  const part = PARTS.find((p) => p.id === selected) ?? PARTS[0];

  return (
    <WidgetFrame
      id="w4.5"
      title="Anatomy of a Book Widget"
      teaches="Every figure in this book obeys the same seven-part contract — and the seed is part of it."
      caption={
        <>
          Select a region to see what it promises and why. This is the frame you have been looking at
          all chapter, and it is the frame every later chapter reuses: the widgets change, the
          contract does not.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1fr_1fr] md:divide-x md:divide-fd-border">
        <svg
          viewBox="0 0 100 96"
          className="w-full p-3"
          role="img"
          aria-label="A schematic of a book widget: a header with an id and title, a world canvas, a row of readouts, a control panel, a transport bar with a visible seed, a color key, and a caption."
        >
          {PARTS.map((p) => {
            const active = p.id === selected;
            return (
              <g
                key={p.id}
                onClick={() => setSelected(p.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelected(p.id);
                  }
                }}
                tabIndex={0}
                role="button"
                aria-pressed={active}
                aria-label={p.label}
                className="cursor-pointer focus-visible:outline-none"
              >
                <rect
                  x={p.x}
                  y={p.y}
                  width={p.w}
                  height={p.h}
                  rx={1}
                  fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                  fillOpacity={active ? 0.12 : 1}
                  stroke={active ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                  strokeWidth={active ? 0.9 : 0.4}
                />
                <text
                  x={p.x + 2}
                  y={p.y + p.h / 2 + 1}
                  style={{ fontSize: 2.9, fontWeight: active ? 700 : 500 }}
                  fill={active ? 'var(--color-fd-primary)' : 'var(--pr-truth)'}
                >
                  {p.label}
                </text>
                {/* Leader line out to the callout number. */}
                <line
                  x1={p.x + p.w}
                  y1={p.y + p.h / 2}
                  x2={70}
                  y2={p.y + p.h / 2}
                  stroke={active ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                  strokeWidth={0.35}
                />
                <circle
                  cx={72.5}
                  cy={p.y + p.h / 2}
                  r={2.4}
                  fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                  stroke={active ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                  strokeWidth={0.4}
                />
                <text
                  x={72.5}
                  y={p.y + p.h / 2 + 1}
                  textAnchor="middle"
                  style={{ fontSize: 2.8, fontWeight: 600 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
                >
                  {PARTS.indexOf(p) + 1}
                </text>
              </g>
            );
          })}

          {/* The five book colors, sitting inside the legend strip. */}
          {(['prior', 'prediction', 'measurement', 'posterior', 'truth'] as const).map((role, i) => (
            <rect
              key={role}
              x={40 + i * 4.5}
              y={77}
              width={3.4}
              height={2}
              rx={0.4}
              fill={`var(--pr-${role})`}
            />
          ))}
        </svg>

        <div className="border-t border-fd-border p-4 md:border-t-0">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-[0.7rem] text-fd-primary">
              {PARTS.indexOf(part) + 1}
            </span>
            <h4 className="font-display text-base font-semibold">{part.label}</h4>
          </div>

          <dl className="mt-3 space-y-2.5">
            <div>
              <dt className="eyebrow">guarantees</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{part.guarantee}</dd>
            </div>
            <div>
              <dt className="eyebrow">why it is non-negotiable</dt>
              <dd className="font-prose text-[0.84rem] leading-snug">{part.why}</dd>
            </div>
            <div>
              <dt className="eyebrow">in code</dt>
              <dd className="font-mono text-[0.72rem] leading-snug text-fd-muted-foreground">
                {part.code}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </WidgetFrame>
  );
}
