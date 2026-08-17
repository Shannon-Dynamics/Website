'use client';

import { useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';

/**
 * w1.4 — How to Read This Book.
 *
 * The chapter rhythm as a clickable strip, plus the color contract. Both are
 * promises the rest of the book keeps: every chapter runs Hook → C → F → P →
 * Lab → Exercises, and every blue thing on every page means the same thing.
 */

interface Stage {
  id: string;
  short: string;
  pass: 'C' | 'F' | 'P' | '';
  what: string;
  why: string;
  example: string;
}

const STAGES: Stage[] = [
  {
    id: 'hook',
    short: 'Hook',
    pass: '',
    what: 'An autoplaying simulation, above any prose, showing the failure the chapter exists to fix.',
    why: 'You should be able to see the problem before you can name it. If a chapter cannot show you its problem, it does not have one.',
    example: 'This chapter: Rusty drives a perfect square and ends up somewhere else (w1.2).',
  },
  {
    id: 'conceptual',
    short: 'Conceptual',
    pass: 'C',
    what: 'Widgets you can steer, with the math named but not yet derived.',
    why: 'Intuition first: a parameter you have moved yourself is one whose meaning you will remember.',
    example: 'The belief histogram sharpening under a door sighting, before the word "posterior" is defined (w1.1).',
  },
  {
    id: 'foundation',
    short: 'Foundation',
    pass: 'F',
    what: 'Definitions, stated assumptions, and derivations — three to eight named steps in the text, full algebra one click away.',
    why: 'Everything the book claims must be checkable. Collapsible derivations let you skip the algebra on a first pass and never lose the thread.',
    example: 'The argmax fallacy, then the ten-cell corridor worked out to the last decimal.',
  },
  {
    id: 'practical',
    short: 'Practical',
    pass: 'P',
    what: 'Idiomatic Rust on the crate stack the book pins: nalgebra, rand, faer, parry2d, factrs, petgraph.',
    why: 'The printed code is the code that runs. Where a chapter states a number, a test pins it, and the widget on the page computes it.',
    example: 'ch01_hello — sixty dependency-free lines that print the belief in the table above.',
  },
  {
    id: 'lab',
    short: 'Lab',
    pass: 'P',
    what: 'The integration section: the new algorithm dropped into the Hallway or the Apartment with Rusty driving.',
    why: 'An algorithm that only works on the page it was derived on has not been demonstrated. Every method meets the same two worlds.',
    example: 'From Chapter 4 onward, every chapter ends inside one of the running worlds.',
  },
  {
    id: 'exercises',
    short: 'Exercises',
    pass: '',
    what: 'Four to seven, tagged F (derive it), C (predict what a widget will do, then check), and P (write the Rust).',
    why: 'The C exercises are the ones people skip and should not: predicting before you press play is the only way to find out what you actually believe.',
    example: 'Exercise 3 below asks you to call the posterior before running the widget.',
  },
];

const COLORS: { role: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth'; name: string; math: string; means: string }[] = [
  { role: 'prior', name: 'Prior', math: 'bel(x_{t-1})', means: 'what was believed before this step' },
  { role: 'prediction', name: 'Prediction', math: 'bel-bar(x_t)', means: 'belief after motion, before sensing' },
  { role: 'measurement', name: 'Measurement', math: 'p(z_t | x_t)', means: 'what the sensor says about each state' },
  { role: 'posterior', name: 'Posterior', math: 'bel(x_t)', means: 'belief after the update' },
  { role: 'truth', name: 'Truth', math: 'x_t', means: 'ground truth, which the robot never sees' },
];

export function HowToReadThisBook() {
  const [selected, setSelected] = useState('hook');
  const stage = STAGES.find((s) => s.id === selected) ?? STAGES[0];

  const boxW = 15.2;
  const gap = 1.4;

  return (
    <WidgetFrame
      id="w1.4"
      title="How to read this book"
      teaches="Every chapter has the same shape and the same color contract, so the effort you spend learning to read one chapter is spent once."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Select a stage to see what it is for. The rhythm never changes, and neither do the colors:
          a term tinted blue in an equation is the same quantity as the blue curve in the figure
          beside it and the blue variable in the Rust listing below it. Teal — the color of this
          box, the links, and the widget ids — is deliberately not one of the five: chrome never
          borrows a data color.
        </>
      }
    >
      <div className="p-3">
        <svg
          viewBox="0 0 100 16"
          className="w-full"
          role="img"
          aria-label="The chapter rhythm: Hook, Conceptual, Foundation, Practical, Lab, Exercises, in sequence."
        >
          {STAGES.map((s, i) => {
            const x = i * (boxW + gap);
            const active = s.id === selected;
            return (
              <g
                key={s.id}
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
                aria-label={s.short}
                className="cursor-pointer focus-visible:outline-none"
              >
                <rect
                  x={x}
                  y={2}
                  width={boxW}
                  height={9}
                  rx={1}
                  fill={active ? 'var(--color-fd-primary)' : 'var(--pr-canvas-bg)'}
                  stroke={active ? 'var(--color-fd-primary)' : 'var(--pr-grid)'}
                  strokeWidth={0.4}
                />
                <text
                  x={x + boxW / 2}
                  y={6.4}
                  textAnchor="middle"
                  style={{ fontSize: 2.6, fontWeight: 600 }}
                  fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-canvas-ink)'}
                >
                  {s.short}
                </text>
                {s.pass ? (
                  <text
                    x={x + boxW / 2}
                    y={9.6}
                    textAnchor="middle"
                    style={{ fontSize: 2.1, fontWeight: 700, letterSpacing: 0.1 }}
                    fill={active ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
                  >
                    {s.pass}
                  </text>
                ) : null}
              </g>
            );
          })}

          {STAGES.slice(0, -1).map((s, i) => {
            const x = i * (boxW + gap) + boxW;
            return (
              <path
                key={`arrow-${s.id}`}
                d={`M ${x + 0.3} 6.5 L ${x + gap - 0.3} 6.5`}
                stroke="var(--pr-grid)"
                strokeWidth={0.45}
                strokeLinecap="round"
              />
            );
          })}
        </svg>

        <dl className="mt-3 grid gap-2.5 sm:grid-cols-3">
          <div>
            <dt className="eyebrow">what it is</dt>
            <dd className="font-prose text-[0.84rem] leading-snug">{stage.what}</dd>
          </div>
          <div>
            <dt className="eyebrow">why it is there</dt>
            <dd className="font-prose text-[0.84rem] leading-snug">{stage.why}</dd>
          </div>
          <div>
            <dt className="eyebrow">here, that means</dt>
            <dd className="font-prose text-[0.84rem] leading-snug">{stage.example}</dd>
          </div>
        </dl>
      </div>

      <table className="w-full table-fixed border-collapse border-t border-fd-border text-left">
        <caption className="sr-only">The book color code and what each color means</caption>
        <tbody>
          {COLORS.map((c) => (
            <tr key={c.role} className="border-b border-fd-border/50 last:border-b-0">
              <td className="w-[9rem] py-1.5 ps-3">
                <span className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className="size-2.5 shrink-0 rounded-[1px]"
                    style={{ backgroundColor: `var(--pr-${c.role})` }}
                  />
                  <span className="font-ui text-[0.72rem] font-medium">{c.name}</span>
                </span>
              </td>
              <td className="py-1.5 font-mono text-[0.7rem]" style={{ color: `var(--pr-${c.role})` }}>
                {c.math}
              </td>
              <td className="py-1.5 pe-3 font-ui text-[0.72rem] text-fd-muted-foreground">
                {c.means}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </WidgetFrame>
  );
}
