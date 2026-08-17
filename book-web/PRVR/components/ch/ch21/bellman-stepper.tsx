'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import {
  blankGrid,
  cellIndex,
  gridWorldMdp,
  MOVE_LABELS4,
  MOVES4,
  type GridSpec,
} from '@/lib/decision/gridworld';
import { backup, qValues, sweepInPlace, type Mdp } from '@/lib/decision/mdp';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w21.2 — the Bellman Stepper.
 *
 * One backup, magnified until it stops looking like linear algebra. The bars on
 * the right are Q(x, u) built the way the equation builds it: a green slab for
 * the immediate payoff r(x, u), then one blue slab per successor worth
 * γ · p(x' | x, u) · V(x'). The max gate picks a row; that row's total becomes
 * the cell's new value, in purple.
 *
 * Run it and the focus walks the grid in sweep order, which is exactly what
 * `sweepInPlace` does — so the wave crawling backwards from the goal is not an
 * illustration of value iteration, it *is* value iteration, one backup a frame.
 */

const W = 11;
const H = 8;
const GOAL_PAYOFF = 20;

interface State {
  v: number[];
  focus: number;
  backups: number;
  sweeps: number;
  /** Value before and after the most recent backup, for the delta readout. */
  last: { s: number; before: number; after: number } | null;
}

function buildSpec(slip: number, gamma: number): GridSpec {
  const spec = blankGrid(W, H, { slip, gamma, moves: 4, stepCost: 1, noCornerCutting: true });
  // A wall with one gap, so the value wave has to find the doorway.
  for (let j = 0; j < H; j++) if (j !== 5) spec.blocked[cellIndex(spec, 5, j)] = true;
  const goal = cellIndex(spec, W - 1, 1);
  spec.payoff[goal] = GOAL_PAYOFF;
  spec.terminal[goal] = true;
  return spec;
}

export function BellmanStepper() {
  const [params, setParams] = useState({ slip: 0.15, gamma: 0.95 });

  const specRef = useRef<GridSpec>(buildSpec(0.15, 0.95));
  const mdpRef = useRef<Mdp>(gridWorldMdp(specRef.current));
  const signature = `${params.slip}|${params.gamma}`;
  const lastSignature = useRef(signature);
  if (lastSignature.current !== signature) {
    lastSignature.current = signature;
    specRef.current = buildSpec(params.slip, params.gamma);
    mdpRef.current = gridWorldMdp(specRef.current);
  }
  const spec = specRef.current;

  const nextFocus = useCallback(
    (from: number): { focus: number; wrapped: boolean } => {
      const mdp = mdpRef.current;
      for (let k = 1; k <= mdp.nStates; k++) {
        const s = (from + k) % mdp.nStates;
        if (!mdp.absorbing[s]) return { focus: s, wrapped: s <= from };
      }
      return { focus: from, wrapped: true };
    },
    [],
  );

  const init = useCallback((): State => {
    const mdp = mdpRef.current;
    let focus = 0;
    while (focus < mdp.nStates && mdp.absorbing[focus]) focus++;
    return { v: new Array<number>(mdp.nStates).fill(0), focus, backups: 0, sweeps: 0, last: null };
  }, []);

  const step = useCallback(
    (s: State): State => {
      const mdp = mdpRef.current;
      const v = s.v.slice();
      const before = v[s.focus];
      const after = backup(mdp, v, s.focus).value;
      v[s.focus] = after;
      const { focus, wrapped } = nextFocus(s.focus);
      return {
        v,
        focus,
        backups: s.backups + 1,
        sweeps: s.sweeps + (wrapped ? 1 : 0),
        last: { s: s.focus, before, after },
      };
    },
    [nextFocus],
  );

  const sim = useSimulation<State>({ init, step, fps: 6, initialSeed: 21 });

  const runSweep = useCallback(() => {
    sim.setState((s) => {
      const v = s.v.slice();
      sweepInPlace(mdpRef.current, v);
      return { ...s, v, sweeps: s.sweeps + 1, backups: s.backups + mdpRef.current.nStates, last: null };
    });
  }, [sim]);

  /* ------------------------------------------------------------ grid panel */

  const drawGridPanel = useCallback(
    (ctx: CanvasRenderingContext2D, view: Viewport, p: Palette) => {
      clear(ctx, view, p);
      const { v, focus } = sim.state;
      let vMax = 1e-6;
      for (let k = 0; k < v.length; k++) if (!spec.blocked[k] && v[k] > vMax) vMax = v[k];

      for (let j = 0; j < H; j++) {
        for (let i = 0; i < W; i++) {
          const idx = cellIndex(spec, i, j);
          const px = sx(view, i);
          const py = sy(view, j + 1);
          const w = sl(view, 1) + 0.5;

          if (spec.blocked[idx]) {
            ctx.globalAlpha = 0.28;
            ctx.fillStyle = p.wall;
          } else if (spec.terminal[idx]) {
            ctx.globalAlpha = 0.85;
            ctx.fillStyle = p.measurement;
          } else {
            ctx.globalAlpha = 0.08 + 0.75 * Math.max(0, v[idx] / vMax);
            ctx.fillStyle = p.posterior;
          }
          ctx.fillRect(px, py, w, w);
          ctx.globalAlpha = 1;

          ctx.strokeStyle = p.grid;
          ctx.lineWidth = 1;
          ctx.strokeRect(px, py, w, w);

          if (!spec.blocked[idx] && !spec.terminal[idx] && Math.abs(v[idx]) > 0.005) {
            label(ctx, v[idx].toFixed(1), px + w / 2, py + w / 2, p.ink, { size: 9, align: 'center' });
          }
          if (spec.terminal[idx]) {
            label(ctx, 'GOAL', px + w / 2, py + w / 2, p.bg, { size: 8, align: 'center', weight: 700 });
          }
        }
      }

      // The focus cell and the four neighbours its backup will read.
      const fi = focus % W;
      const fj = Math.floor(focus / W);
      ctx.strokeStyle = p.prior;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 2]);
      for (const d of MOVES4) {
        const ni = fi + d[0];
        const nj = fj + d[1];
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        ctx.strokeRect(sx(view, ni), sy(view, nj + 1), sl(view, 1), sl(view, 1));
      }
      ctx.setLineDash([]);
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(sx(view, fi), sy(view, fj + 1), sl(view, 1), sl(view, 1));
    },
    [sim.state, spec],
  );

  /* --------------------------------------------------------- backup panel */

  const drawBackup = useCallback(
    (ctx: CanvasRenderingContext2D, view: Viewport, p: Palette) => {
      clear(ctx, view, p);
      const mdp = mdpRef.current;
      const { v, focus } = sim.state;
      const q = qValues(mdp, v, focus);
      const best = q.reduce((b, x, i) => (x > q[b] ? i : b), 0);

      const scale = Math.max(2, ...q.map(Math.abs), GOAL_PAYOFF * 0.35);
      const x0 = 2.4; // baseline, in panel units
      const unit = 5.2 / scale; // panel units per unit of value

      label(
        ctx,
        `Q(x, u) = r(x, u) + γ Σ p(x' | x, u) V(x')      cell (${focus % W}, ${Math.floor(focus / W)})`,
        sx(view, 0.15),
        sy(view, 4.62),
        p.ink,
        { size: 10, weight: 600 },
      );

      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(view, x0), sy(view, 4.25));
      ctx.lineTo(sx(view, x0), sy(view, 0.15));
      ctx.stroke();

      for (let a = 0; a < mdp.nActions; a++) {
        const y = 3.55 - a * 0.85;
        const h = 0.42;
        label(ctx, MOVE_LABELS4[a], sx(view, 0.15), sy(view, y + h / 2), p.ink, { size: 11 });

        // Slab 1: the immediate payoff, in measurement green.
        let cursor = x0;
        const r = mdp.reward[focus][a] * unit;
        drawSlab(ctx, view, cursor, y, r, h, p.measurement, 0.85);
        cursor += r;

        // Slabs 2..: γ p(x'|x,u) V(x'), one per successor, in prior blue —
        // because V is the *previous* iterate, the thing being folded in.
        for (const t of mdp.trans[focus][a]) {
          const contribution = mdp.gamma * t.p * v[t.s] * unit;
          if (Math.abs(contribution) < 1e-9) continue;
          drawSlab(ctx, view, cursor, y, contribution, h, p.prior, 0.28 + 0.6 * t.p);
          cursor += contribution;
        }

        // The total, and the max gate.
        const isBest = a === best;
        ctx.strokeStyle = isBest ? p.posterior : p.grid;
        ctx.lineWidth = isBest ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(sx(view, cursor), sy(view, y - 0.08));
        ctx.lineTo(sx(view, cursor), sy(view, y + h + 0.08));
        ctx.stroke();
        label(
          ctx,
          `${q[a].toFixed(2)}${isBest ? '  ← max' : ''}`,
          sx(view, Math.max(cursor, x0) + 0.12),
          sy(view, y + h / 2),
          isBest ? p.posterior : p.truth,
          { size: 10, weight: isBest ? 700 : 500 },
        );
      }

      const { last } = sim.state;
      const note = last
        ? `V(${last.s % W}, ${Math.floor(last.s / W)}) : ${last.before.toFixed(3)} → ${last.after.toFixed(3)}`
        : 'press step to commit the max into V';
      label(ctx, note, sx(view, 0.15), sy(view, 0.05), p.posterior, { size: 10, weight: 600 });
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const mdp = mdpRef.current;
    const { v, focus } = sim.state;
    const q = qValues(mdp, v, focus);
    return {
      residual: Math.abs(Math.max(...q) - v[focus]),
      backups: sim.state.backups,
      sweeps: sim.state.sweeps,
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w21.2"
      title="Bellman Stepper"
      teaches="A backup is bookkeeping, not linear algebra: add the payoff, add the discounted neighbours weighted by how likely you are to reach them, keep the biggest."
      colorKey={['prior', 'measurement', 'posterior']}
      caption={
        <>
          The purple square is the cell about to be backed up; the dashed blue squares are the
          neighbours its backup reads. On the right, each row assembles one action&apos;s{' '}
          <em>Q</em>-value: a green slab for <em>r</em>(<em>x</em>, <em>u</em>), then a blue slab per
          successor of width γ&nbsp;<em>p</em>&nbsp;<em>V</em>. The longest row wins and its total
          becomes the new value. <strong>Press step a few times</strong>, then hold play: the focus
          walks the grid in sweep order and a value wave crawls backwards from the goal, bending
          through the doorway exactly as Chapter 20&apos;s wave-front planner did. Turn the slip up
          and watch the blue slabs split three ways.
        </>
      }
    >
      <div className="grid gap-0 md:grid-cols-[1.15fr_1fr] md:divide-x md:divide-fd-border">
        <SimCanvas
          world={{ minX: 0, minY: 0, maxX: W, maxY: H }}
          draw={drawGridPanel}
          deps={[sim.tick, sim.state]}
          aspect={W / H}
          padding={0.15}
          ariaLabel="A grid world with a wall and one doorway. Cells are shaded by their current value, with the cell about to be backed up outlined and its four neighbours dashed."
        />
        <SimCanvas
          world={{ minX: 0, minY: 0, maxX: 8, maxY: 4.8 }}
          draw={drawBackup}
          deps={[sim.tick, sim.state]}
          aspect={8 / 4.8}
          padding={0.05}
          ariaLabel="Four horizontal bars, one per action, each built from a reward slab and one discounted-value slab per successor state. The longest bar is marked as the maximum."
        />
      </div>

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="backups" value={String(stats.backups)} />
        <Stat label="sweeps" value={String(stats.sweeps)} />
        <Stat label="residual at focus" value={stats.residual.toFixed(4)} />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Slip s (per side)"
          role="prediction"
          value={params.slip}
          min={0}
          max={0.35}
          step={0.01}
          onChange={(v) => setParams((p) => ({ ...p, slip: v }))}
          help="At s = 0 each action has a single successor and the backup has one blue slab."
        />
        <Slider
          label="Discount γ"
          role="posterior"
          value={params.gamma}
          min={0.5}
          max={0.99}
          step={0.01}
          onChange={(v) => setParams((p) => ({ ...p, gamma: v }))}
        />
      </ControlPanel>

      <div className="flex flex-wrap items-center gap-2 border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton onClick={runSweep}>run one full sweep</ActionButton>
        </ButtonRow>
      </div>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

/** One signed slab of a Q bar, drawn from `x` to `x + w` in panel units. */
function drawSlab(
  ctx: CanvasRenderingContext2D,
  view: Viewport,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  alpha: number,
) {
  if (Math.abs(w) < 1e-9) return;
  const x1 = Math.min(x, x + w);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fillRect(sx(view, x1), sy(view, y + h), Math.max(sl(view, Math.abs(w)), 1), sl(view, h));
  ctx.restore();
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
