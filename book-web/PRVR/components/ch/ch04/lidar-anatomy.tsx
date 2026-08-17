'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT } from '@/lib/sim/world';
import { RUSTY, RUSTY_LIDAR, raycastScan, type LidarParams } from '@/lib/sim/rusty';
import {
  clear,
  drawGrid,
  drawRobot,
  drawWorld,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import type { Pose2 } from '@/lib/geom/se2';

/**
 * w4.2 — LiDAR Anatomy.
 *
 * A frozen scene: Rusty sits in the kitchen looking north through the doorway
 * into the corridor. The sweep reveals one beam at a time, and the strip below
 * plots what the algorithms downstream will actually receive — a vector of
 * numbers indexed by beam, with no geometry attached to it at all.
 */

const POSE: Pose2 = { x: 6.0, y: 2.7, theta: Math.PI / 2 };
const N_BEAMS = 180;
/** The visible slice of the Apartment: room B, its doorway, and the corridor. */
const SCENE = { minX: 3.7, maxX: 8.3, minY: 0.9, maxY: 5.5 };

interface Clock {
  k: number;
}

export function LidarAnatomy() {
  const [sigmaR, setSigmaR] = useState(0.04);
  const [pDropout, setPDropout] = useState(0.02);
  const [showAll, setShowAll] = useState(false);

  const lidar: LidarParams = useMemo(
    () => ({ ...RUSTY_LIDAR, nBeams: N_BEAMS, sigmaR, pDropout }),
    [sigmaR, pDropout],
  );

  const init = useCallback((): Clock => ({ k: 0 }), []);
  const step = useCallback((s: Clock): Clock => ({ k: (s.k + 1) % N_BEAMS }), []);
  const sim = useSimulation<Clock>({ init, step, fps: 24, initialSeed: 7 });

  // The scene never moves, so the scan is computed once per (seed, σ_r, p_drop)
  // rather than every frame. Re-rolling the seed draws a fresh set of returns
  // from exactly the same geometry — which is the entire point of a noise model.
  const scan = useMemo(
    () => raycastScan(APARTMENT, POSE, lidar, new Rng(sim.seed)),
    [lidar, sim.seed],
  );

  const k = sim.state.k;
  const visible = showAll ? N_BEAMS : k + 1;

  const drawScene = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 0.5);
      drawWorld(ctx, v, APARTMENT, p);

      const c = Math.cos(POSE.theta);
      const s = Math.sin(POSE.theta);
      const ox = POSE.x + c * RUSTY_LIDAR.offset[0] - s * RUSTY_LIDAR.offset[1];
      const oy = POSE.y + s * RUSTY_LIDAR.offset[0] + c * RUSTY_LIDAR.offset[1];
      const px = sx(v, ox);
      const py = sy(v, oy);

      // Revealed beams, drawn to the *reported* range — the thing the robot has.
      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.globalAlpha = 0.2;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < visible; i++) {
        const a = POSE.theta + scan.angles[i];
        const r = scan.ranges[i];
        ctx.moveTo(px, py);
        ctx.lineTo(sx(v, ox + r * Math.cos(a)), sy(v, oy + r * Math.sin(a)));
      }
      ctx.stroke();
      ctx.restore();

      // True hit points (what the geometry says) under the returns (what came back).
      ctx.save();
      for (let i = 0; i < visible; i++) {
        const a = POSE.theta + scan.angles[i];
        const zStar = scan.trueRanges[i];
        if (zStar < scan.maxRange) {
          ctx.fillStyle = p.truth;
          ctx.globalAlpha = 0.55;
          ctx.beginPath();
          ctx.arc(sx(v, ox + zStar * Math.cos(a)), sy(v, oy + zStar * Math.sin(a)), 1.4, 0, Math.PI * 2);
          ctx.fill();
        }
        if (!scan.dropped[i]) {
          const r = scan.ranges[i];
          ctx.fillStyle = p.measurement;
          ctx.globalAlpha = 0.95;
          ctx.beginPath();
          ctx.arc(sx(v, ox + r * Math.cos(a)), sy(v, oy + r * Math.sin(a)), 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();

      // The beam under the cursor, in full.
      const a = POSE.theta + scan.angles[k];
      ctx.save();
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(sx(v, ox + scan.ranges[k] * Math.cos(a)), sy(v, oy + scan.ranges[k] * Math.sin(a)));
      ctx.stroke();
      ctx.restore();

      drawRobot(ctx, v, POSE, p.truth, RUSTY.bodyRadius, { filled: true, alpha: 0.9 });

      label(ctx, 'kitchen (room B)', sx(v, 4.6), sy(v, 1.8), p.truth, { size: 9.5 });
      label(ctx, 'doorway', sx(v, 6.0), sy(v, 3.55), p.truth, { size: 9.5, align: 'center' });
      label(ctx, 'corridor', sx(v, 7.4), sy(v, 4.4), p.truth, { size: 9.5, align: 'center' });

      const info = scan.dropped[k]
        ? `beam ${k}:  φ = ${deg(scan.angles[k])}  ·  z = z_max (dropout)`
        : `beam ${k}:  φ = ${deg(scan.angles[k])}  ·  z* = ${scan.trueRanges[k].toFixed(3)} m  ·  z = ${scan.ranges[k].toFixed(3)} m`;
      label(ctx, info, 10, 14, p.accent, { size: 10, weight: 600 });
      label(
        ctx,
        `residual ε = ${(scan.ranges[k] - scan.trueRanges[k]).toFixed(3)} m`,
        10,
        28,
        p.measurement,
        { size: 10 },
      );
    },
    [scan, k, visible],
  );

  const drawStrip = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);

      // Axes.
      ctx.save();
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, 0));
      ctx.lineTo(sx(v, N_BEAMS - 1), sy(v, 0));
      ctx.stroke();
      ctx.restore();

      // z_max: the ceiling every dropout is pinned to.
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, scan.maxRange));
      ctx.lineTo(sx(v, N_BEAMS - 1), sy(v, scan.maxRange));
      ctx.stroke();
      ctx.restore();
      label(ctx, 'z_max = 8 m', sx(v, 1), sy(v, scan.maxRange) - 8, p.truth, { size: 9.5 });

      // The truth: a piecewise-smooth function of bearing with hard edges at
      // every corner. Nothing downstream ever gets to see this curve.
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (let i = 0; i < visible; i++) {
        const y = sy(v, Math.min(scan.trueRanges[i], scan.maxRange));
        if (i === 0) ctx.moveTo(sx(v, i), y);
        else ctx.lineTo(sx(v, i), y);
      }
      ctx.stroke();
      ctx.restore();

      // The data.
      ctx.save();
      for (let i = 0; i < visible; i++) {
        ctx.fillStyle = p.measurement;
        ctx.globalAlpha = scan.dropped[i] ? 0.45 : 0.95;
        ctx.beginPath();
        ctx.arc(sx(v, i), sy(v, scan.ranges[i]), scan.dropped[i] ? 2.6 : 1.9, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Cursor.
      ctx.save();
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(sx(v, k), sy(v, 0));
      ctx.lineTo(sx(v, k), sy(v, scan.maxRange + 0.4));
      ctx.stroke();
      ctx.restore();

      for (const [idx, text] of [
        [0, '−180°'],
        [45, '−90°'],
        [90, '0° (ahead)'],
        [135, '+90°'],
        [179, '+178°'],
      ] as [number, string][]) {
        label(ctx, text, sx(v, idx), sy(v, -0.55), p.truth, { size: 9, align: 'center' });
      }
      label(ctx, 'range z (m)', 8, 12, p.truth, { size: 9.5 });
    },
    [scan, k, visible],
  );

  return (
    <WidgetFrame
      id="w4.2"
      title="LiDAR Anatomy"
      teaches="A scan is not geometry. It is a vector of noisy numbers indexed by beam, and everything after this is inference."
      colorKey={['measurement', 'truth']}
      caption={
        <>
          Above: the scene, with the true first-hit points in gray and the returned points in green.
          Below: the same scan as the robot receives it — range against beam index. Watch the two
          hard edges at beams 78 and 103, where the doorway ends and the beam lands on the near wall
          (1.18 m) instead of the far one (2.47 m); those cliffs are the door frame, and recovering
          them is what Chapter 16&apos;s scan matcher does. Now raise the dropout rate: every dropped
          beam is pinned to z_max, so the sensor reports its <em>largest</em> number precisely when
          it knows <em>least</em>. A model that treats 8 m as evidence of an 8 m wall will localize
          onto glass.
        </>
      }
    >
      <SimCanvas
        world={SCENE}
        draw={drawScene}
        deps={[scan, k, visible]}
        aspect={1.55}
        padding={0.15}
        ariaLabel="Rusty in a kitchen firing LiDAR beams north through a doorway into a corridor. Green dots mark the returned range of each beam; gray dots mark the true distance to the wall."
      />
      <div className="border-t border-fd-border">
        <SimCanvas
          world={{ minX: -4, maxX: N_BEAMS + 3, minY: -1.0, maxY: 8.8 }}
          draw={drawStrip}
          deps={[scan, k, visible]}
          aspect={2.2}
          padding={0}
          ariaLabel="A plot of range against beam index: a gray dashed curve for the true ranges and green dots for the noisy returns, with dropouts pinned to the eight metre maximum."
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Range noise σ_r"
          role="measurement"
          value={sigmaR}
          min={0}
          max={0.3}
          step={0.005}
          unit="m"
          onChange={setSigmaR}
          help="Std-dev of the Gaussian added to every good return."
        />
        <Slider
          label="Dropout p_drop"
          value={pDropout}
          min={0}
          max={0.2}
          step={0.005}
          format={(x) => `${(x * 100).toFixed(1)}%`}
          onChange={setPDropout}
          help="Chance a beam comes back with nothing and reports z_max. Glass, black felt, grazing incidence."
        />
        <Slider
          label="Beam index k"
          value={k}
          min={0}
          max={N_BEAMS - 1}
          step={1}
          format={(x) => String(Math.round(x))}
          onChange={(v) => {
            sim.pause();
            sim.setState(() => ({ k: Math.round(v) }));
          }}
        />
        <Toggle label="Show the whole scan at once" checked={showAll} onChange={setShowAll} />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(0)}°`;
