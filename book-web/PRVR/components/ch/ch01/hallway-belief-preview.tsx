'use client';

import { useCallback } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { HistogramFilter1D } from '@/lib/filters/bayes';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w1.1 — the Hallway Belief Machine, preview edition.
 *
 * The book's thesis in one loop: a ten-cell corridor, three indistinguishable
 * doors, and the canned sequence sense → move → sense from the chapter's second
 * derivation. Every number on screen comes from `HistogramFilter1D` — the same
 * class Chapter 5 derives as the Bayes filter — run once at module load, so the
 * bars, the table, and the prose are guaranteed to agree.
 *
 * The full parameter surface (noise sliders, Markov breaking, continuous state)
 * is deliberately withheld until w5.1. Here the reader gets play, pause, step.
 */

const N = 10;
const DOORS = [1, 4, 5];
const P_HIT = 0.6; // p(z = door | at a door)
const P_FALSE = 0.2; // p(z = door | not at a door)

/** Cell index of a position given in metres. Cell i spans [i, i+1). */
const cellOf = (x: number) => Math.floor(x);

/** p(z = door | x), evaluated at a cell centre. */
const doorLikelihood = (center: number) =>
  DOORS.includes(cellOf(center)) ? P_HIT : P_FALSE;

type PhaseKind = 'prior' | 'sense' | 'move' | 'sense2';

interface Phase {
  kind: PhaseKind;
  /** Colour role the belief is drawn in during this phase. */
  role: 'prior' | 'prediction' | 'posterior';
  eyebrow: string;
  headline: string;
  /** Where the true robot actually is during this phase. */
  truthCell: number;
  /** Does the measurement likelihood belong on screen? */
  showLikelihood: boolean;
}

const PHASES: Phase[] = [
  {
    kind: 'prior',
    role: 'prior',
    eyebrow: 'PRIOR',
    headline: 'bel(x) = 0.1 everywhere — the robot knows nothing',
    truthCell: 4,
    showLikelihood: false,
  },
  {
    kind: 'sense',
    role: 'posterior',
    eyebrow: 'SENSE  "door"',
    headline: 'multiply by p(z | x), normalize — three peaks, not one',
    truthCell: 4,
    showLikelihood: true,
  },
  {
    kind: 'move',
    role: 'prediction',
    eyebrow: 'MOVE  one cell right',
    headline: 'shift the whole histogram — no mass is created or destroyed',
    truthCell: 5,
    showLikelihood: false,
  },
  {
    kind: 'sense2',
    role: 'posterior',
    eyebrow: 'SENSE  "door"  again',
    headline: 'only cell 5 explains door → move → door: 9/26 ≈ 0.3462',
    truthCell: 5,
    showLikelihood: true,
  },
];

/**
 * The four beliefs, produced by running the library filter once. Module scope,
 * because nothing here is random: this is a *table*, and it must be the same
 * table every time the page loads.
 */
const ROWS: number[][] = (() => {
  const filter = new HistogramFilter1D({ length: N, cells: N, wrap: true });
  filter.setUniform();
  const rows = [filter.belief()];
  filter.correct(doorLikelihood); // sense
  rows.push(filter.belief());
  filter.predict(1); // move: no kernel ⇒ an exact one-cell cyclic shift
  rows.push(filter.belief());
  filter.correct(doorLikelihood); // sense again
  rows.push(filter.belief());
  return rows;
})();

interface State {
  phase: number;
}

export function HallwayBeliefPreview() {
  const init = useCallback((): State => ({ phase: 0 }), []);
  const step = useCallback((s: State): State => ({ phase: (s.phase + 1) % PHASES.length }), []);
  const sim = useSimulation<State>({ init, step, fps: 1 });

  const idx = sim.state.phase;
  const phase = PHASES[idx];
  const belief = ROWS[idx];
  const before = ROWS[(idx + PHASES.length - 1) % PHASES.length];

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);

      const roleColor =
        phase.role === 'prior' ? p.prior : phase.role === 'prediction' ? p.prediction : p.posterior;

      // ---- corridor -----------------------------------------------------
      // World y runs 0…4 so that the world rectangle matches the canvas
      // aspect: the drawing then fills the frame instead of floating in it.
      const yTop = sy(v, 3.78);
      const yBot = sy(v, 3.08);
      ctx.fillStyle = p.free;
      ctx.fillRect(sx(v, 0), yTop, sl(v, N), yBot - yTop);
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx(v, 0), yTop, sl(v, N), yBot - yTop);

      // Cell dividers, so "cell 5" is a thing the reader can point at.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < N; i++) {
        ctx.moveTo(sx(v, i), yTop);
        ctx.lineTo(sx(v, i), yBot);
      }
      ctx.stroke();

      // The three doors — identical, which is the entire problem.
      for (const d of DOORS) {
        ctx.fillStyle = p.bg;
        ctx.fillRect(sx(v, d + 0.12), yTop - 1, sl(v, 0.76), yBot - yTop + 2);
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx(v, d + 0.12), yTop);
        ctx.lineTo(sx(v, d + 0.12), yBot);
        ctx.moveTo(sx(v, d + 0.88), yTop);
        ctx.lineTo(sx(v, d + 0.88), yBot);
        ctx.stroke();
      }

      // Cell numbers, under the corridor.
      for (let i = 0; i < N; i++) {
        label(ctx, String(i), sx(v, i + 0.5), yBot + 11, p.truth, { size: 10, align: 'center' });
      }

      // ---- ground truth --------------------------------------------------
      const rx = sx(v, phase.truthCell + 0.5);
      const ry = (yTop + yBot) / 2;
      ctx.strokeStyle = p.truth;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(rx, ry, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(rx, ry, 3, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, 'true cell', rx + 12, ry, p.truth, { size: 10 });

      // ---- histogram ------------------------------------------------------
      const baseY = sy(v, 0.35);
      const topY = sy(v, 2.35);
      const h = baseY - topY;
      const peak = 0.4; // fixed scale: bars are comparable across phases
      const barW = sl(v, 0.72);

      // Ghost of the belief this phase started from, in the colour that belief
      // had, so the reader can see what the operation changed. The prior has no
      // "before", so it gets none.
      if (idx > 0) {
        ctx.globalAlpha = 0.22;
        ctx.fillStyle = idx === 1 ? p.prior : idx === 2 ? p.posterior : p.prediction;
        for (let i = 0; i < N; i++) {
          const bh = (before[i] / peak) * h;
          ctx.fillRect(sx(v, i + 0.5) - barW / 2, baseY - bh, barW, bh);
        }
        ctx.globalAlpha = 1;
      }

      // The belief itself.
      ctx.fillStyle = roleColor;
      for (let i = 0; i < N; i++) {
        const bh = (belief[i] / peak) * h;
        ctx.fillRect(sx(v, i + 0.5) - barW / 2, baseY - bh, barW, bh);
        label(ctx, belief[i].toFixed(4), sx(v, i + 0.5), baseY - bh - 8, roleColor, {
          size: 9,
          align: 'center',
          weight: 600,
        });
      }

      // The likelihood, drawn as a curve: it is a function of state, not a
      // distribution over it, so it never gets bars.
      if (phase.showLikelihood) {
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 1.75;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
          const l = doorLikelihood(i + 0.5);
          const y = baseY - (l / P_HIT) * h * 0.55;
          if (i === 0) ctx.moveTo(sx(v, i), y);
          else ctx.lineTo(sx(v, i), y);
          ctx.lineTo(sx(v, i + 1), y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        label(ctx, 'p(z = door | x)', sx(v, 0.1), baseY - 0.55 * h - 12, p.measurement, {
          size: 10,
        });
      }

      // Baseline and phase caption.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), baseY);
      ctx.lineTo(sx(v, N), baseY);
      ctx.stroke();

      label(ctx, phase.eyebrow, sx(v, 0), topY - 26, roleColor, { size: 11, weight: 700 });
      label(ctx, phase.headline, sx(v, 0), topY - 12, p.ink, { size: 10.5 });
    },
    [phase, belief, before, idx],
  );

  return (
    <WidgetFrame
      id="w1.1"
      title="The Hallway Belief Machine (preview)"
      teaches="A belief is a whole distribution. Sensing multiplies it, moving shifts it, and ambiguity dies from the sequence — not from any single reading."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Ten cells, three identical doors, one perfectly ordinary robot. Watch the loop twice
          before reading on. <strong>Notice</strong> that a single door sighting produces three
          equal peaks — the sensor did exactly its job and the robot still does not know where it
          is; and that after the second sighting one cell holds exactly three times the belief of
          its nearest rival, because only cell 5 is a door that is also one step right of a door.{' '}
          <strong>Try</strong> clicking a row of the table to scrub to that step, then stepping
          forward by hand. Every number in the table is computed by the same histogram filter the
          Practical section prints in Rust, and the peak really is 9/26.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: N, minY: 0, maxY: 4 }}
        draw={draw}
        deps={[idx]}
        aspect={2.5}
        padding={0.1}
        ariaLabel="A ten-cell corridor with doors at cells 1, 4 and 5, above a bar chart of the robot's belief. The belief is flat, then has three peaks after a door sighting, then shifts one cell right, then collapses onto cell 5."
      />

      <BeliefTable
        active={idx}
        onSelect={(i) => {
          // Scrubbing implies "I want to look at this", so stop the loop.
          sim.pause();
          sim.setState(() => ({ phase: i }));
        }}
      />

      {/* No seed control: nothing here is random. That is the point — the
          animation is a table, and the table is in the derivation. */}
      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
      />
    </WidgetFrame>
  );
}

const ROW_META: { label: string; role: 'prior' | 'prediction' | 'posterior' }[] = [
  { label: 'bel₀  (uniform prior)', role: 'prior' },
  { label: 'after sense "door"', role: 'posterior' },
  { label: 'after move right', role: 'prediction' },
  { label: 'after sense "door"', role: 'posterior' },
];

/**
 * The derivation's table, linked to the animation: the live row is tinted with
 * that phase's role colour, and clicking a row scrubs the widget to it.
 */
function BeliefTable({ active, onSelect }: { active: number; onSelect: (i: number) => void }) {
  return (
    <div className="overflow-x-auto border-t border-fd-border">
      <table className="w-full border-collapse text-right font-mono text-[0.68rem] tabular-nums">
        <caption className="sr-only">
          The belief over all ten cells after each step of the sense–move–sense sequence
        </caption>
        <thead>
          <tr className="border-b border-fd-border">
            <th scope="col" className="px-2 py-1 text-left font-normal text-fd-muted-foreground">
              step
            </th>
            {Array.from({ length: N }, (_, i) => (
              <th
                key={i}
                scope="col"
                className="px-1 py-1 font-normal"
                style={DOORS.includes(i) ? { color: 'var(--pr-measurement)' } : undefined}
              >
                {i}
                {DOORS.includes(i) ? '*' : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row, r) => (
            <tr
              key={ROW_META[r].label + r}
              onClick={() => onSelect(r)}
              className="cursor-pointer border-b border-fd-border/50 last:border-b-0 hover:bg-fd-accent"
              style={
                r === active
                  ? { color: `var(--pr-${ROW_META[r].role})`, fontWeight: 600 }
                  : undefined
              }
            >
              <th
                scope="row"
                className="px-2 py-1 text-left font-normal whitespace-nowrap"
                style={r === active ? undefined : { color: 'var(--color-fd-muted-foreground)' }}
              >
                {ROW_META[r].label}
              </th>
              {row.map((p, i) => (
                <td key={i} className="px-1 py-1">
                  {p.toFixed(4)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-2 pb-1.5 font-ui text-[0.65rem] text-fd-muted-foreground">
        * cells with a door. Click a row to scrub the animation to that step.
      </p>
    </div>
  );
}
