'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, drawGrid, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import {
  dubinsAllWords,
  dubinsSample,
  reedsSheppLite,
  type DubinsPath,
} from '@/lib/plan/dubins';

/**
 * w20.4 — the Dubins Dial.
 *
 * Drag two poses and watch all six Dubins words compete. The misconception this
 * kills is "the shortest path for a car is turn, drive, turn": at close range
 * the three-arc words LRL and RLR win outright, and no amount of intuition
 * about straight lines predicts that. The other lesson is quieter — the winning
 * word *changes* as ρ grows, which is why a planner that steers with straight
 * lines is not planning for a car at all.
 */

const WORLD = { minX: 0, minY: 0, maxX: 8, maxY: 4.6 };
const HANDLE = 0.75;

interface Handles {
  start: Pose2;
  goal: Pose2;
}

type DragTarget = 'start-pos' | 'start-dir' | 'goal-pos' | 'goal-dir' | null;

export function DubinsDial() {
  const [poses, setPoses] = useState<Handles>({
    start: { x: 1.6, y: 2.3, theta: 0 },
    goal: { x: 4.0, y: 2.3, theta: Math.PI },
  });
  const [rho, setRho] = useState(0.7);
  const [allowReverse, setAllowReverse] = useState(false);
  const [drag, setDrag] = useState<DragTarget>(null);

  // The whole widget is one closed-form evaluation — no search anywhere.
  const words = useMemo(
    () => dubinsAllWords(poses.start, poses.goal, rho),
    [poses, rho],
  );
  const best: DubinsPath | null = useMemo(
    () =>
      allowReverse
        ? reedsSheppLite(poses.start, poses.goal, rho)
        : (words[0] ?? null),
    [allowReverse, poses, rho, words],
  );
  const bestPolyline = useMemo(() => (best ? dubinsSample(best, 0.02) : []), [best]);
  const wordPolylines = useMemo(
    () => words.map((w) => ({ word: w.word, pts: dubinsSample(w, 0.05), length: w.length })),
    [words],
  );

  const sim = useSimulation<{ phase: number }>({
    init: () => ({ phase: 0 }),
    step: (s) => ({ phase: (s.phase + 0.01) % 1 }),
    fps: 30,
    initialSeed: 20,
  });

  const onPointer = useCallback(
    (w: [number, number], phase: 'down' | 'move' | 'up') => {
      const [x, y] = w;
      if (phase === 'up') {
        setDrag(null);
        return;
      }
      const tip = (p: Pose2) => ({
        x: p.x + HANDLE * Math.cos(p.theta),
        y: p.y + HANDLE * Math.sin(p.theta),
      });
      if (phase === 'down') {
        const targets: [DragTarget, { x: number; y: number }][] = [
          ['start-dir', tip(poses.start)],
          ['goal-dir', tip(poses.goal)],
          ['start-pos', poses.start],
          ['goal-pos', poses.goal],
        ];
        let picked: DragTarget = null;
        let bestD = 0.45;
        for (const [name, p] of targets) {
          const d = Math.hypot(p.x - x, p.y - y);
          if (d < bestD) {
            bestD = d;
            picked = name;
          }
        }
        setDrag(picked);
        return;
      }
      if (!drag) return;
      setPoses((p) => {
        const clampX = Math.min(Math.max(x, 0.4), WORLD.maxX - 0.4);
        const clampY = Math.min(Math.max(y, 0.4), WORLD.maxY - 0.4);
        if (drag === 'start-pos') return { ...p, start: { ...p.start, x: clampX, y: clampY } };
        if (drag === 'goal-pos') return { ...p, goal: { ...p.goal, x: clampX, y: clampY } };
        if (drag === 'start-dir') {
          return { ...p, start: { ...p.start, theta: Math.atan2(y - p.start.y, x - p.start.x) } };
        }
        return { ...p, goal: { ...p.goal, theta: Math.atan2(y - p.goal.y, x - p.goal.x) } };
      });
    },
    [drag, poses],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 1);

      // Every word that exists for this query, drawn faintly: the enumeration
      // *is* the algorithm, so the reader should see all six candidates.
      ctx.save();
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.32;
      ctx.strokeStyle = p.prior;
      for (const w of wordPolylines) {
        if (best && !best.reverse && w.word === best.word) continue;
        ctx.beginPath();
        for (let i = 0; i < w.pts.length; i++) {
          const q = w.pts[i];
          if (i === 0) ctx.moveTo(sx(v, q.x), sy(v, q.y));
          else ctx.lineTo(sx(v, q.x), sy(v, q.y));
        }
        ctx.stroke();
      }
      ctx.restore();

      // The winner.
      if (bestPolyline.length > 1) {
        ctx.save();
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        if (best?.reverse) ctx.setLineDash([7, 4]);
        ctx.beginPath();
        for (let i = 0; i < bestPolyline.length; i++) {
          const q = bestPolyline[i];
          if (i === 0) ctx.moveTo(sx(v, q.x), sy(v, q.y));
          else ctx.lineTo(sx(v, q.x), sy(v, q.y));
        }
        ctx.stroke();
        ctx.restore();

        // A token driving the winning word, so the arcs read as *motion*.
        const k = Math.min(
          bestPolyline.length - 1,
          Math.floor(sim.state.phase * bestPolyline.length),
        );
        const car = bestPolyline[k];
        drawCar(ctx, v, car, p.prediction, best?.reverse ?? false);
      }

      drawPoseHandle(ctx, v, poses.start, p.truth, 'q₀', p);
      drawPoseHandle(ctx, v, poses.goal, p.measurement, 'q₁', p);

      // Turning circles at the start: the geometry the words are built from.
      ctx.save();
      ctx.strokeStyle = p.grid;
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      for (const s of [1, -1]) {
        const cx = poses.start.x + s * rho * Math.cos(poses.start.theta + (s * Math.PI) / 2);
        const cy = poses.start.y + s * rho * Math.sin(poses.start.theta + (s * Math.PI) / 2);
        ctx.beginPath();
        ctx.arc(sx(v, cx), sy(v, cy), sl(v, rho), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      if (best) {
        label(
          ctx,
          `${best.reverse ? 'reverse ' : ''}${best.word}   ${best.length.toFixed(3)} m`,
          14,
          18,
          p.posterior,
          { size: 12, weight: 700 },
        );
      } else {
        label(ctx, 'no path', 14, 18, p.prediction, { size: 12, weight: 700 });
      }
    },
    [best, bestPolyline, poses, rho, sim.state.phase, wordPolylines],
  );

  return (
    <WidgetFrame
      id="w20.4"
      title="The Dubins Dial"
      teaches="A car's shortest path is not turn–straight–turn: at close range the three-arc words LRL and RLR win."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Blue: every Dubins word that exists for this query. Purple: the shortest one, with an
          orange token driving it. Gray is the start pose <em>q₀</em>, green the goal <em>q₁</em>;
          drag either dot to move it, or its stalk to turn it. Notice that as the poses come
          together the answer switches from a CSC word (an arc, a straight, an arc) to a CCC word —
          and that <em>LSR</em> and <em>RSL</em> simply stop existing when the two turning circles
          overlap. Try setting ρ to its maximum with the poses close together: the car can no longer
          reach the goal without a long detour, which is exactly the constraint a straight-line RRT
          pretends does not exist.
        </>
      }
    >
      <SimCanvas
        world={WORLD}
        draw={draw}
        deps={[sim.tick, poses, rho, best, allowReverse]}
        aspect={8 / 4.6}
        padding={0.1}
        onPointer={onPointer}
        cursor={drag ? 'grabbing' : 'grab'}
        ariaLabel="Two draggable poses in the plane with all six Dubins paths drawn between them; the shortest is highlighted and a token drives along it."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-6">
        {wordPolylines.map((w) => (
          <div key={w.word} className="px-1 py-1.5">
            <div
              className="eyebrow"
              style={best && !best.reverse && w.word === best.word ? { color: 'var(--pr-posterior)' } : undefined}
            >
              {w.word}
            </div>
            <div className="font-mono text-[0.72rem] tabular-nums">{w.length.toFixed(2)}</div>
          </div>
        ))}
        {Array.from({ length: Math.max(0, 6 - wordPolylines.length) }).map((_, i) => (
          <div key={`none-${i}`} className="px-1 py-1.5">
            <div className="eyebrow">—</div>
            <div className="font-mono text-[0.72rem] text-fd-muted-foreground">n/a</div>
          </div>
        ))}
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Turning radius ρ"
          role="posterior"
          value={rho}
          min={0.15}
          max={1.6}
          step={0.05}
          unit="m"
          onChange={setRho}
          help="The tightest circle the car can drive. Everything else in the widget follows from it."
        />
        <Toggle
          label="Allow reverse (pure-reverse words only)"
          role="prediction"
          checked={allowReverse}
          onChange={setAllowReverse}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onReset={() => {
          setPoses({
            start: { x: 1.6, y: 2.3, theta: 0 },
            goal: { x: 4.0, y: 2.3, theta: normalizeAngle(Math.PI) },
          });
          sim.reset();
        }}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function drawPoseHandle(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  pose: Pose2,
  color: string,
  name: string,
  p: Palette,
) {
  const px = sx(v, pose.x);
  const py = sy(v, pose.y);
  const tx = sx(v, pose.x + HANDLE * Math.cos(pose.theta));
  const ty = sy(v, pose.y + HANDLE * Math.sin(pose.theta));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(tx, ty, 4, 0, Math.PI * 2);
  ctx.fill();
  label(ctx, name, px + 9, py - 10, p.ink, { size: 11, weight: 600 });
  ctx.restore();
}

/** A little rectangle with a nose, so heading and travel direction both read. */
function drawCar(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  pose: Pose2,
  color: string,
  reverse: boolean,
) {
  const L = sl(v, 0.34);
  const W = sl(v, 0.2);
  ctx.save();
  ctx.translate(sx(v, pose.x), sy(v, pose.y));
  ctx.rotate(-pose.theta);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(L, 0);
  ctx.lineTo(-L * 0.6, W);
  ctx.lineTo(-L * 0.6, -W);
  ctx.closePath();
  ctx.fill();
  if (reverse) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-L * 0.9, 0);
    ctx.lineTo(-L * 1.6, 0);
    ctx.stroke();
  }
  ctx.restore();
}
