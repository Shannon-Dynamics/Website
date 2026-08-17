'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, drawSegments, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import type { Point2, Segment, World } from '@/lib/sim/world';
import { CSpace2, freeCellNear, latticeFromCSpace } from '@/lib/plan/cspace';
import { wavefront } from '@/lib/plan/search';
import {
  DEFAULT_POTENTIAL,
  descentStep,
  totalPotential,
  wavefrontDescentStep,
  type DescentState,
  type PotentialParams,
} from '@/lib/plan/potential';

/**
 * w20.3 — the Potential Well.
 *
 * The hook for the whole chapter. "Drive toward the goal and steer around what
 * you hit" is the first control law anyone writes, and it is a gradient descent
 * on U_att + U_rep. Watch it die in a cup-shaped obstacle: the two gradients
 * cancel, ‖∇U‖ → 0, and the robot reports success by standing still.
 *
 * The cure is one toggle away. Replace the invented potential with the
 * cost-to-go computed by Dijkstra from the goal — Choset's wave-front planner —
 * and the same greedy descent cannot get stuck, because every free cell has a
 * strictly cheaper neighbour. That field is the reader's first value function.
 */

const BOUNDS = { minX: 0, minY: 0, maxX: 8, maxY: 5 };
const START: Point2 = { x: 0.9, y: 2.5 };
const ROBOT_RADIUS = 0.12;
const DISPLAY_CELL = 0.1;

const seg = (x1: number, y1: number, x2: number, y2: number): Segment => ({ x1, y1, x2, y2 });

/** A room with one cup-shaped obstacle, opening toward the start. */
const TRAP: World = {
  name: 'U-trap',
  bounds: BOUNDS,
  walls: [
    seg(0, 0, 8, 0),
    seg(0, 5, 8, 5),
    seg(0, 0, 0, 5),
    seg(8, 0, 8, 5),
    // the cup: back wall plus two arms reaching back toward the robot
    seg(4.6, 1.4, 4.6, 3.6),
    seg(3.0, 1.4, 4.6, 1.4),
    seg(3.0, 3.6, 4.6, 3.6),
  ],
};

interface State extends DescentState {
  trail: Point2[];
  steps: number;
}

export function PotentialWell() {
  const [eta, setEta] = useState(0.6);
  const [qStar, setQStar] = useState(1.0);
  const [useWavefront, setUseWavefront] = useState(false);
  const [goal, setGoal] = useState<Point2>({ x: 6.9, y: 2.5 });

  const params: PotentialParams = useMemo(
    () => ({ ...DEFAULT_POTENTIAL, eta, qStar }),
    [eta, qStar],
  );

  const cs = useMemo(() => new CSpace2(TRAP, { radius: ROBOT_RADIUS, cellSize: 0.05 }), []);
  const lattice = useMemo(() => latticeFromCSpace(cs, DISPLAY_CELL), [cs]);
  const costToGo = useMemo(
    () => wavefront(lattice, freeCellNear(lattice, goal.x, goal.y)),
    [lattice, goal],
  );

  // The field the reader sees: U for potential mode, the wave-front for the cure.
  const heightfield = useMemo(() => {
    const { nx, ny, cellSize } = lattice;
    const data = new Float64Array(nx * ny).fill(NaN);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (lattice.free[k] === 0) continue;
        const x = lattice.bounds.minX + (i + 0.5) * cellSize;
        const y = lattice.bounds.minY + (j + 0.5) * cellSize;
        data[k] = useWavefront ? costToGo[k] : totalPotential(cs.field, { x, y }, goal, params);
      }
    }
    // Robust normalisation: U_rep diverges at contact, so a raw max would wash
    // the whole picture out. The 95th percentile keeps the interesting range.
    const finite = Array.from(data).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
    const lo = finite[0] ?? 0;
    const hi = finite[Math.floor(finite.length * 0.95)] ?? lo + 1;
    return { data, lo, hi: Math.max(hi, lo + 1e-6) };
  }, [cs, costToGo, goal, lattice, params, useWavefront]);

  const init = useCallback(
    (): State => ({ q: { ...START }, status: 'moving', gradNorm: NaN, trail: [{ ...START }], steps: 0 }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      if (s.status !== 'moving') return s;
      if (useWavefront) {
        const { q, stuck } = wavefrontDescentStep(costToGo, lattice, s.q, 0.05);
        const arrived = Math.hypot(q.x - goal.x, q.y - goal.y) < 0.15;
        return {
          q,
          status: arrived ? 'arrived' : stuck ? 'stuck' : 'moving',
          gradNorm: NaN,
          trail: [...s.trail, q],
          steps: s.steps + 1,
        };
      }
      const next = descentStep(cs.field, s, goal, params, 0.05);
      return {
        ...next,
        trail: next.q === s.q ? s.trail : [...s.trail, next.q],
        steps: s.steps + 1,
      };
    },
    [cs, costToGo, goal, lattice, params, useWavefront],
  );

  const sim = useSimulation<State>({ init, step, fps: 30, initialSeed: 20 });

  // Any change to the field invalidates the run: restart the bead so the reader
  // always sees the trajectory that belongs to the parameters on screen.
  const resetRef = useRef(sim.reset);
  resetRef.current = sim.reset;
  const paramKey = `${eta}|${qStar}|${useWavefront}|${goal.x.toFixed(2)},${goal.y.toFixed(2)}`;
  useEffect(() => {
    resetRef.current();
  }, [paramKey]);

  const onPointer = useCallback(
    (w: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') return;
      const [x, y] = w;
      const cx = Math.min(Math.max(x, 0.3), BOUNDS.maxX - 0.3);
      const cy = Math.min(Math.max(y, 0.3), BOUNDS.maxY - 0.3);
      if (cs.isFree(cx, cy)) setGoal({ x: cx, y: cy });
    },
    [cs],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { nx, ny, cellSize } = lattice;
      const { data, lo, hi } = heightfield;
      const w = Math.ceil(sl(v, cellSize)) + 1;
      const fieldColor = useWavefront ? p.posterior : p.truth;

      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const val = data[j * nx + i];
          if (!Number.isFinite(val)) continue;
          const t = Math.min(1, Math.max(0, (val - lo) / (hi - lo)));
          ctx.globalAlpha = 0.08 + 0.62 * t;
          ctx.fillStyle = fieldColor;
          const x = lattice.bounds.minX + i * cellSize;
          const y = lattice.bounds.minY + j * cellSize;
          ctx.fillRect(sx(v, x), sy(v, y + cellSize), w, w);
        }
      }
      ctx.globalAlpha = 1;

      drawSegments(ctx, v, TRAP.walls, p.wall, 3);

      // The bead's trail — the actual executed motion, not a plan.
      const trail = sim.state.trail;
      if (trail.length > 1) {
        ctx.save();
        ctx.strokeStyle = p.prediction;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(sx(v, trail[0].x), sy(v, trail[0].y));
        for (let i = 1; i < trail.length; i++) ctx.lineTo(sx(v, trail[i].x), sy(v, trail[i].y));
        ctx.stroke();
        ctx.restore();
      }

      // Goal (draggable) and start.
      ctx.save();
      ctx.fillStyle = p.measurement;
      ctx.beginPath();
      ctx.arc(sx(v, goal.x), sy(v, goal.y), 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx(v, START.x), sy(v, START.y), 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // The bead.
      const q = sim.state.q;
      ctx.save();
      ctx.fillStyle = p.prediction;
      ctx.beginPath();
      ctx.arc(sx(v, q.x), sy(v, q.y), 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      label(
        ctx,
        useWavefront ? 'WAVE-FRONT: descend cost-to-go' : 'POTENTIAL: descend U_att + U_rep',
        14,
        18,
        useWavefront ? p.posterior : p.ink,
        { size: 11, weight: 700 },
      );

      if (sim.state.status === 'stuck') {
        const bx = sx(v, q.x) + 12;
        const by = sy(v, q.y) - 14;
        ctx.save();
        ctx.fillStyle = p.prediction;
        ctx.globalAlpha = 0.15;
        ctx.fillRect(bx - 4, by - 10, 118, 20);
        ctx.restore();
        label(ctx, 'STUCK  ‖∇U‖ ≈ 0', bx, by, p.prediction, { size: 11, weight: 700 });
      } else if (sim.state.status === 'arrived') {
        label(ctx, 'arrived', sx(v, q.x) + 12, sy(v, q.y) - 14, p.measurement, {
          size: 11,
          weight: 700,
        });
      }
    },
    [goal, heightfield, lattice, sim.state, useWavefront],
  );

  return (
    <WidgetFrame
      id="w20.3"
      title="The Potential Well"
      teaches="“Just follow the gradient” is a planner that reports success by standing still."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          The gray shading is the artificial potential <em>U</em> = <em>U</em><sub>att</sub> +{' '}
          <em>U</em><sub>rep</sub>; darker is higher. The orange bead rolls downhill from the gray
          ring toward the green goal — and stops dead inside the cup, at a point where the pull of
          the goal exactly cancels the push of the wall. Nothing is wrong with the code: that point
          really is a minimum of <em>U</em>. Now switch on <strong>wave-front mode</strong>. The
          field becomes the cost-to-go computed by Dijkstra from the goal, and the identical greedy
          descent walks straight out of the cup. Drag the green goal anywhere to re-plan, and try
          raising η until even the wave-front’s starting corridor looks tight.
        </>
      }
    >
      <SimCanvas
        world={BOUNDS}
        draw={draw}
        deps={[sim.tick, sim.state, heightfield, useWavefront, goal]}
        aspect={8 / 5}
        padding={0.15}
        onPointer={onPointer}
        cursor="crosshair"
        ariaLabel="A room containing a cup-shaped obstacle, shaded by the value of an artificial potential field. A bead rolls downhill from the start and becomes trapped inside the cup; in wave-front mode it escapes around the obstacle."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="status" value={sim.state.status} />
        <Stat
          label="‖∇U‖"
          value={Number.isFinite(sim.state.gradNorm) ? sim.state.gradNorm.toFixed(3) : '—'}
        />
        <Stat label="steps" value={String(sim.state.steps)} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Repulsive gain η"
          role="prediction"
          value={eta}
          min={0.05}
          max={2}
          step={0.05}
          onChange={setEta}
          help="How hard obstacles push back. The trap survives every value of it."
        />
        <Slider
          label="Influence radius Q*"
          value={qStar}
          min={0.3}
          max={2}
          step={0.05}
          unit="m"
          onChange={setQStar}
        />
        <Toggle
          label="Wave-front mode"
          role="posterior"
          checked={useWavefront}
          onChange={setUseWavefront}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
