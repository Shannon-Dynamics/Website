'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { between, normalizeAngle, se2Log, type Pose2 } from '@/lib/geom/se2';
import { APARTMENT, beamAngles, simulateScan } from '@/lib/sim/world';
import {
  estimateNormals,
  scanToCloud,
  transformCloud,
  voxelDownsample,
  VoxelMap,
  type Pt,
} from '@/lib/slam/cloud';
import { icp, type IcpVariant } from '@/lib/slam/icp';
import { clear, drawSegments, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w16.1 — the ICP Stepper.
 *
 * Two real LiDAR sweeps of the Apartment's room A, taken 0.6 m and 20° apart.
 * The blue cloud is the target (what the map already contains), the orange one
 * is the source dragged to wherever the reader's initial guess puts it, and
 * every frame is one iteration of the *actual* `icp` from lib/slam — the same
 * routine the SLAM lab runs.
 *
 * Room A is 4.0 m × 3.8 m. That near-squareness is the point: rotate the
 * initial guess far enough and ICP converges, confidently and with a small
 * residual, onto the room turned a quarter turn.
 */

const SCAN = { nBeams: 180, fov: 2 * Math.PI, maxRange: 6, sigma: 0.012 };
const ANGLES = beamAngles(SCAN);
const POSE_A: Pose2 = { x: 2.0, y: 1.9, theta: 0 };
const POSE_B: Pose2 = { x: 2.5, y: 1.55, theta: 0.35 };
const VIEW = { minX: -0.35, minY: -0.35, maxX: 4.35, maxY: 4.15 };
/** Below this the answer is the right minimum; above it, ICP found another one. */
const SUCCESS_M = 0.12;

interface Scene {
  target: Pt[];
  map: VoxelMap;
  source: Pt[];
  truth: Pose2;
}

function buildScene(): Scene {
  const rng = new Rng(0x1c9);
  const a = scanToCloud(simulateScan(APARTMENT, POSE_A, SCAN, rng), ANGLES, SCAN.maxRange);
  const b = scanToCloud(simulateScan(APARTMENT, POSE_B, SCAN, rng), ANGLES, SCAN.maxRange);
  // The target lives in A's frame; the source in B's. What ICP has to recover
  // is exactly the relative pose between the two sensor positions.
  const target = voxelDownsample(a.points, 0.06);
  const map = new VoxelMap(0.25, 8);
  map.insert(target, estimateNormals(target, 0.3));
  return { target, map, source: voxelDownsample(b.points, 0.09), truth: between(POSE_A, POSE_B) };
}

export function IcpStepper() {
  const scene = useMemo(buildScene, []);
  const [headingOffset, setHeadingOffset] = useState(25);
  const [drag, setDrag] = useState<{ dx: number; dy: number }>({ dx: 0.55, dy: 0.45 });
  const [variant, setVariant] = useState<IcpVariant>('point-to-point');
  const [tau, setTau] = useState(0.7);

  const init = useMemo<Pose2>(
    () => ({
      x: scene.truth.x + drag.dx,
      y: scene.truth.y + drag.dy,
      theta: normalizeAngle(scene.truth.theta + (headingOffset * Math.PI) / 180),
    }),
    [scene.truth, drag, headingOffset],
  );

  const result = useMemo(
    () => icp(scene.source, scene.map, init, { variant, tau, maxIters: 30 }),
    [scene, init, variant, tau],
  );

  const simInit = useCallback(() => ({ i: 0 }), []);
  const simStep = useCallback((s: { i: number }) => ({ i: s.i + 1 }), []);
  const sim = useSimulation<{ i: number }>({ init: simInit, step: simStep, fps: 4 });

  // A new initial guess restarts the animation from iteration 0, so the reader
  // always sees the convergence they just asked for from its beginning.
  const resetRef = useRef(sim.reset);
  resetRef.current = sim.reset;
  useEffect(() => {
    resetRef.current();
  }, [init, variant, tau]);

  const nIter = result.trace.length;
  // Replay, hold for a beat, replay: the loop is the autoplay behaviour.
  const step = Math.min(sim.state.i % (nIter + 5), nIter - 1);
  const frame = result.trace[step];

  const finalError = useMemo(() => {
    const e = se2Log(between(scene.truth, result.pose));
    return { pos: Math.hypot(e[0], e[1]), rot: (Math.abs(e[2]) * 180) / Math.PI };
  }, [scene.truth, result.pose]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawSegments(ctx, v, APARTMENT.walls, p.grid, 2);

      const dot = (pt: Pt, color: string, r = 2.1, alpha = 1) => {
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx(v, pt[0]), sy(v, pt[1]), r, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      };

      // Ground truth: where the source cloud belongs.
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1;
      for (const pt of transformCloud(scene.truth, scene.source)) {
        ctx.beginPath();
        ctx.arc(sx(v, pt[0]), sy(v, pt[1]), 3.2, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      // The target cloud — the map ICP is matching against.
      for (const pt of scene.target) dot(pt, p.prior, 1.9, 0.85);

      // Correspondence lines for this iteration.
      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      for (const pr of frame.pairs) {
        ctx.moveTo(sx(v, pr.src[0]), sy(v, pr.src[1]));
        ctx.lineTo(sx(v, pr.dst[0]), sy(v, pr.dst[1]));
      }
      ctx.stroke();
      ctx.restore();

      // The source cloud at this iteration: orange while moving, purple once settled.
      const settled = step >= nIter - 1;
      const color = settled ? p.posterior : p.prediction;
      for (const pt of transformCloud(frame.pose, scene.source)) dot(pt, color, 2.4);

      // Sensor origins.
      const origin = (pose: Pose2, c: string) => {
        ctx.strokeStyle = c;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(sx(v, pose.x), sy(v, pose.y), sl(v, 0.11), 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx(v, pose.x), sy(v, pose.y));
        ctx.lineTo(sx(v, pose.x + 0.3 * Math.cos(pose.theta)), sy(v, pose.y + 0.3 * Math.sin(pose.theta)));
        ctx.stroke();
      };
      origin({ x: 0, y: 0, theta: 0 }, p.prior);
      origin(frame.pose, color);

      label(ctx, `iteration ${step} / ${nIter - 1}`, 10, 16, p.ink, { size: 11, weight: 600 });
      label(
        ctx,
        `rmse ${Number.isFinite(frame.rmse) ? frame.rmse.toFixed(3) : '—'} m · ${frame.pairs.length} pairs`,
        10,
        32,
        p.measurement,
        { size: 10 },
      );
      if (settled) {
        const ok = finalError.pos < SUCCESS_M;
        label(
          ctx,
          ok ? 'converged to the right minimum' : `WRONG MINIMUM — off by ${finalError.pos.toFixed(2)} m, ${finalError.rot.toFixed(0)}°`,
          10,
          v.height - 12,
          ok ? p.posterior : p.prediction,
          { size: 11, weight: 700 },
        );
      }
    },
    [scene, frame, step, nIter, finalError],
  );

  const costSeries = useMemo(
    () => [
      {
        id: 'RMSE',
        role: 'posterior' as const,
        data: result.trace.map((it, i) => ({ x: i, y: Number.isFinite(it.rmse) ? it.rmse : 0 })),
      },
    ],
    [result],
  );

  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up', e: React.PointerEvent) => {
      if (phase === 'move' && e.buttons === 0) return;
      setDrag({ dx: world[0] - scene.truth.x, dy: world[1] - scene.truth.y });
    },
    [scene.truth],
  );

  return (
    <WidgetFrame
      id="w16.1"
      title="ICP Stepper"
      teaches="ICP does not find the alignment; it finds the nearest one. Start it far enough away and it converges, confidently, onto the wrong wall."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Blue is the target sweep already in the map; orange is the new sweep, placed by your
          initial guess; green lines are the current correspondences; gray circles show where the
          new sweep actually belongs. Each frame is one iteration of the real <code>icp</code>{' '}
          routine. <strong>What to notice:</strong> correspondences change discretely and the
          residual falls in steps, not smoothly — that is the piecewise cost of{' '}
          <a href="#w16.3">w16.3</a>. <strong>What to try:</strong> push the heading offset past
          about 45°. Room A is 4.0 × 3.8 m, so a quarter turn maps it almost onto itself and ICP
          settles into that wrong minimum with a residual small enough to look convincing. Then
          switch to point-to-plane and count iterations: the same answer arrives in a third of them,
          because sliding <em>along</em> a wall costs nothing and the solver stops fighting it.
        </>
      }
    >
      <div className="grid gap-0 lg:grid-cols-[1.4fr_1fr] lg:divide-x lg:divide-fd-border">
        <SimCanvas
          world={VIEW}
          draw={draw}
          deps={[step, result, finalError]}
          aspect={1.24}
          padding={0}
          ariaLabel="Two overlapping LiDAR scans of a rectangular room. Green lines join each source point to its nearest target point, and the source cloud rotates and slides onto the target as the iterations advance."
          onPointer={onPointer}
          cursor="crosshair"
        />
        <div className="flex flex-col gap-2 border-t border-fd-border p-3 lg:border-t-0">
          <p className="eyebrow m-0">residual per iteration</p>
          <LineChart
            series={costSeries}
            xLabel="iteration"
            yLabel="RMSE (m)"
            height={168}
            yMin={0}
            markers={[{ axis: 'x', value: step, label: 'now', role: 'prediction' }]}
            legend={false}
          />
          <div className="grid grid-cols-2 gap-2 text-center">
            <Readout label="iterations to converge" value={String(nIter - 1)} />
            <Readout
              label="error vs truth"
              value={`${finalError.pos.toFixed(2)} m`}
              tone={finalError.pos < SUCCESS_M ? 'posterior' : 'prediction'}
            />
          </div>
        </div>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Initial heading offset"
          role="prediction"
          value={headingOffset}
          min={-100}
          max={100}
          step={1}
          unit="°"
          format={(v) => v.toFixed(0)}
          onChange={setHeadingOffset}
          help="How badly the prediction guessed the rotation. This is the parameter that decides which minimum ICP falls into."
        />
        <Slider
          label="Rejection radius τ"
          role="measurement"
          value={tau}
          min={0.1}
          max={1.6}
          step={0.05}
          unit="m"
          onChange={setTau}
          help="Correspondences longer than τ are thrown away."
        />
        <Toggle
          label="Point-to-plane"
          role="posterior"
          checked={variant === 'point-to-plane'}
          onChange={(on) => setVariant(on ? 'point-to-plane' : 'point-to-point')}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={step}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function Readout({
  label: l,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: 'posterior' | 'prediction';
}) {
  return (
    <div className="rounded-sm border border-fd-border px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={tone ? { color: `var(--pr-${tone})` } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
