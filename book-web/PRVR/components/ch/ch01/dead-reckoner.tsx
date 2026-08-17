'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz/line-chart';
import { StatTile } from '@/components/viz/stat-tile';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT, diffDriveStep } from '@/lib/sim/world';
import { sampleMotionModelVelocity, type MotionAlphas } from '@/lib/models/motion';
import { angleDiff, pose2, type Pose2 } from '@/lib/geom/se2';
import {
  clear,
  drawPath,
  drawRobot,
  drawSegments,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w1.2 — the Dead Reckoner. The book's opening argument, in one loop.
 *
 * Rusty is commanded around a 2.4 m square in the Apartment. Integrating those
 * commands gives the orange trace: a perfect square, closing exactly where it
 * started, because arithmetic never slips. The gray dashed trace is where the
 * robot actually went, produced by `sampleMotionModelVelocity` — the real
 * Thrun Table 5.3 sampler from the library, not a hand-drawn wobble.
 *
 * Nothing in the orange computation is wrong. It is simply answering a
 * different question from the one the robot needs answered.
 */

const DT = 0.1; // s per simulation step
const SPEED = 0.6; // m/s along a side
const SIDE_TICKS = 40; // 40 × 0.1 s × 0.6 m/s = 2.4 m
const TURN_TICKS = 15;
const TURN_RATE = Math.PI / 2 / (TURN_TICKS * DT); // exactly 90° per corner
const LAP = 4 * (SIDE_TICKS + TURN_TICKS);
const START: Pose2 = pose2(8.75, 0.65, 0);

/** The commanded control at a given tick: drive a side, turn a corner, repeat. */
function command(tick: number): { v: number; omega: number } {
  const leg = tick % (SIDE_TICKS + TURN_TICKS);
  return leg < SIDE_TICKS ? { v: SPEED, omega: 0 } : { v: 0, omega: TURN_RATE };
}

interface Sample {
  x: number;
  y: number;
}

interface State {
  rng: Rng;
  /** Where Rusty really is. The robot never gets to see this. */
  truth: Pose2;
  /** Where the commands say Rusty is. This is all the robot has. */
  reckoned: Pose2;
  truthPath: Sample[];
  reckonedPath: Sample[];
  /** Position error against elapsed time, for the chart. */
  errors: { x: number; y: number }[];
  distance: number;
}

export function DeadReckoner() {
  const [noise, setNoise] = useState(1);

  const alphas = useMemo<MotionAlphas>(
    // α₁…α₆ are *variance* coefficients on the commanded (v, ω): a robot that
    // drives faster is wrong faster. Zero means perfect wheels on perfect floor.
    () => [0.05 * noise, 0.05 * noise, 0.05 * noise, 0.05 * noise, 0.01 * noise, 0.01 * noise],
    [noise],
  );

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      truth: { ...START },
      reckoned: { ...START },
      truthPath: [{ x: START.x, y: START.y }],
      reckonedPath: [{ x: START.x, y: START.y }],
      errors: [{ x: 0, y: 0 }],
      distance: 0,
    }),
    [],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      const u = command(tick);

      // Ground truth: the command as the *world* executes it — wheels slip, one
      // motor is a hair stronger than the other, the floor is not quite flat.
      const truth = sampleMotionModelVelocity({ v: u.v, omega: u.omega, dt: DT }, s.truth, alphas, s.rng);
      // Dead reckoning: the command as the *robot* integrates it. Noise-free by
      // construction — this is the same arithmetic with the noise term deleted.
      const reckoned = diffDriveStep(s.reckoned, u.v, u.omega, DT);

      const err = Math.hypot(truth.x - reckoned.x, truth.y - reckoned.y);
      const t = (tick + 1) * DT;

      // Sample the error series at 1/4 rate: the chart wants shape, not every step.
      const errors =
        tick % 4 === 0 ? [...s.errors, { x: t, y: err }].slice(-140) : s.errors;

      return {
        rng: s.rng,
        truth,
        reckoned,
        truthPath: [...s.truthPath, { x: truth.x, y: truth.y }],
        reckonedPath: [...s.reckonedPath, { x: reckoned.x, y: reckoned.y }],
        errors,
        distance: s.distance + Math.abs(u.v) * DT,
      };
    },
    [alphas],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 24,
    maxTicks: LAP,
    loop: true,
    initialSeed: 13,
  });

  const { truth, reckoned, truthPath, reckonedPath, errors, distance } = sim.state;
  const posError = Math.hypot(truth.x - reckoned.x, truth.y - reckoned.y);
  const headError = Math.abs(angleDiff(truth.theta, reckoned.theta)) * (180 / Math.PI);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      // Walls only — the landmarks in the world file belong to Chapter 11.
      drawSegments(ctx, v, APARTMENT.walls, p.wall, 2.5);

      // Where the lap began, so the reader can see whether it closed.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx(v, START.x), sy(v, START.y), 5, 0, Math.PI * 2);
      ctx.stroke();

      drawPath(ctx, v, truthPath, p.truth, { dashed: true, lineWidth: 2 });
      drawPath(ctx, v, reckonedPath, p.prediction, { lineWidth: 2 });

      // The gap: what the robot would have to be told to be right again.
      if (posError > 0.03) {
        ctx.strokeStyle = p.ink;
        ctx.globalAlpha = 0.5;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx(v, truth.x), sy(v, truth.y));
        ctx.lineTo(sx(v, reckoned.x), sy(v, reckoned.y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        label(
          ctx,
          `${posError.toFixed(2)} m`,
          (sx(v, truth.x) + sx(v, reckoned.x)) / 2 + 6,
          (sy(v, truth.y) + sy(v, reckoned.y)) / 2,
          p.ink,
          { size: 10 },
        );
      }

      drawRobot(ctx, v, truth, p.truth, 0.3, { filled: false });
      drawRobot(ctx, v, reckoned, p.prediction, 0.3);

      label(ctx, 'dead-reckoned pose', sx(v, 6.35), sy(v, 6.7), p.prediction, {
        size: 11,
        weight: 600,
      });
      label(ctx, 'what Rusty believes', sx(v, 6.35), sy(v, 6.35), p.prediction, { size: 9.5 });
      label(ctx, 'true pose', sx(v, 6.35), sy(v, 5.85), p.truth, { size: 11, weight: 600 });
      label(ctx, 'where Rusty is', sx(v, 6.35), sy(v, 5.5), p.truth, { size: 9.5 });
    },
    [truthPath, reckonedPath, truth, reckoned, posError],
  );

  // The chart is expensive next to a 24 fps canvas, so it re-renders only when
  // the sampled series actually changes.
  const chart = useMemo(
    () => (
      <LineChart
        series={[{ id: 'position error', role: 'prediction', data: errors }]}
        xLabel="time (s)"
        yLabel="error (m)"
        height={170}
        yMin={0}
        legend={false}
        margin={{ left: 52, bottom: 40, top: 10, right: 16 }}
        ariaLabel="Line chart of the distance between the dead-reckoned pose and the true pose, growing without bound as the lap proceeds."
      />
    ),
    [errors],
  );

  return (
    <WidgetFrame
      id="w1.2"
      title="The Dead Reckoner"
      teaches="Knowing the commands is not knowing the position: integration turns a small, honest actuation error into an unbounded one, and no amount of care in the arithmetic fixes it."
      colorKey={['prediction', 'truth']}
      caption={
        <>
          Rusty is told to drive a 2.4 m square and does exactly as instructed. The orange trace is
          the square Rusty computes by integrating its own commands; the gray dashed trace is where
          the wheels actually took it. <strong>Notice</strong> that the orange path closes perfectly
          every lap — dead reckoning is never internally inconsistent, it is just wrong — while the
          error chart never comes back down. <strong>Try</strong> setting actuation noise to zero
          and watch the two traces fuse; then re-roll the seed a few times at noise 1 and ask
          yourself whether you could predict which way the drift will go. You cannot: that is the
          whole reason this book exists.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={1.35}
        padding={0.3}
        ariaLabel="A floorplan of the apartment. An orange square path shows the pose Rusty computes from its commands; a gray dashed path shows where the robot actually travels, drifting away from the square."
      />

      <div className="grid grid-cols-1 gap-3 border-t border-fd-border p-3 sm:grid-cols-3">
        <StatTile
          label="distance driven"
          value={distance}
          unit="m"
          precision={2}
          role="truth"
        />
        <StatTile
          label="position error"
          value={posError}
          unit="m"
          precision={3}
          role="prediction"
          sparkline={errors.map((e) => e.y)}
        />
        <StatTile label="heading error" value={headError} unit="°" precision={1} role="prediction" />
      </div>

      <div className="border-t border-fd-border px-3 py-2">{chart}</div>

      <ControlPanel columns={1}>
        <Slider
          label="Actuation noise ×"
          role="prediction"
          value={noise}
          min={0}
          max={3}
          step={0.1}
          onChange={setNoise}
          help="Scales α₁…α₆, the variance coefficients of the velocity motion model. Zero is a robot on perfect wheels — a robot that does not exist."
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}
