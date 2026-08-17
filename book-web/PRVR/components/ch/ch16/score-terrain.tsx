'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { Rng } from '@/lib/prob/rng';
import type { Pose2 } from '@/lib/geom/se2';
import { APARTMENT, beamAngles, simulateScan } from '@/lib/sim/world';
import { scanToCloud, transformCloud, voxelDownsample, VoxelMap } from '@/lib/slam/cloud';
import { buildNdt } from '@/lib/slam/ndt';
import { label, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w16.3 — Score Terrain.
 *
 * The same scan pair, the same translation offsets, two objectives. On the left
 * ICP's truncated point-to-point cost, computed by the same nearest-neighbour
 * query the matcher uses; on the right NDT's Gaussian-mixture score. Both are
 * evaluated on a real 49 × 49 grid of (Δx, Δy) with the heading held fixed.
 *
 * Flip to the corridor scene and both surfaces develop a valley: no translation
 * along a pair of parallel walls changes anything, and no amount of iterating
 * will invent the information that is missing.
 */

// A deliberately short-sighted sweep: at 3 m the corridor really is two
// parallel walls and nothing else, which is what makes the degeneracy visible
// instead of merely arguable.
const SCAN = { nBeams: 180, fov: 2 * Math.PI, maxRange: 3, sigma: 0.01 };
const ANGLES = beamAngles(SCAN);
const N = 49;
const SPAN = 1.3;

const SCENES: Record<string, { pose: Pose2; blurb: string }> = {
  corner: {
    pose: { x: 2.0, y: 1.9, theta: 0 },
    blurb: 'Room A — four walls at two orientations. Every direction is constrained.',
  },
  corridor: {
    pose: { x: 6.0, y: 4.42, theta: 0 },
    blurb: 'The corridor — two parallel walls. Sliding along them costs nothing.',
  },
};

interface Terrain {
  /** Fit quality in [0, 1]; 1 is the best cell on that surface. */
  icp: Float64Array;
  ndt: Float64Array;
  /** Raw values, for the hover readout. */
  icpRaw: Float64Array;
  ndtRaw: Float64Array;
  icpArgmax: [number, number];
  ndtArgmax: [number, number];
}

function buildTerrain(sceneKey: string, tau: number): Terrain {
  const { pose } = SCENES[sceneKey];
  const rng = new Rng(0x5c0);
  const sweep = scanToCloud(simulateScan(APARTMENT, pose, SCAN, rng), ANGLES, SCAN.maxRange);
  const cloud = voxelDownsample(sweep.points, 0.08);
  const world = transformCloud(pose, cloud);

  const map = new VoxelMap(0.3, 8);
  map.insert(world);
  const ndt = buildNdt(world, 0.55, 4);

  const icpRaw = new Float64Array(N * N);
  const ndtRaw = new Float64Array(N * N);
  const t2 = tau * tau;

  for (let j = 0; j < N; j++) {
    const dy = -SPAN + (2 * SPAN * j) / (N - 1);
    for (let i = 0; i < N; i++) {
      const dx = -SPAN + (2 * SPAN * i) / (N - 1);
      const shifted: Pose2 = { x: pose.x + dx, y: pose.y + dy, theta: pose.theta };
      let cost = 0;
      for (const p of cloud) {
        const wx = shifted.x + Math.cos(shifted.theta) * p[0] - Math.sin(shifted.theta) * p[1];
        const wy = shifted.y + Math.sin(shifted.theta) * p[0] + Math.cos(shifted.theta) * p[1];
        const idx = map.nearestIndex([wx, wy], tau);
        if (idx < 0) {
          // The truncation is what makes the ICP cost flat far from the answer:
          // beyond τ every correspondence is rejected and the cost stops caring.
          cost += t2;
        } else {
          const q = map.pts[idx];
          cost += (q[0] - wx) ** 2 + (q[1] - wy) ** 2;
        }
      }
      icpRaw[j * N + i] = cost / cloud.length;
      ndtRaw[j * N + i] = ndt.score(cloud, shifted) / cloud.length;
    }
  }

  const norm = (raw: Float64Array, invert: boolean) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of raw) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo || 1;
    const out = new Float64Array(raw.length);
    for (let k = 0; k < raw.length; k++) {
      const t = (raw[k] - lo) / span;
      out[k] = invert ? 1 - t : t;
    }
    return out;
  };

  const icp = norm(icpRaw, true);
  const ndtQ = norm(ndtRaw, false);
  const argmax = (q: Float64Array): [number, number] => {
    let best = -1;
    let bi = 0;
    for (let k = 0; k < q.length; k++) {
      if (q[k] > best) {
        best = q[k];
        bi = k;
      }
    }
    return [bi % N, Math.floor(bi / N)];
  };

  return {
    icp,
    ndt: ndtQ,
    icpRaw,
    ndtRaw,
    icpArgmax: argmax(icp),
    ndtArgmax: argmax(ndtQ),
  };
}

export function ScoreTerrain() {
  const [scene, setScene] = useState<keyof typeof SCENES>('corner');
  const [tau, setTau] = useState(0.6);
  const [hover, setHover] = useState<{ panel: 0 | 1; i: number; j: number } | null>(null);

  const terrain = useMemo(() => buildTerrain(scene, tau), [scene, tau]);

  const cellToOffset = (k: number) => -SPAN + (2 * SPAN * k) / (N - 1);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      ctx.clearRect(0, 0, v.width, v.height);
      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, v.width, v.height);

      const gap = 14;
      const top = 22;
      const bottom = 16;
      const side = (v.width - gap) / 2;
      const size = Math.min(side, v.height - top - bottom);
      const panels: { x0: number; q: Float64Array; title: string; argmax: [number, number] }[] = [
        { x0: (side - size) / 2, q: terrain.icp, title: 'ICP  ·  truncated point-to-point', argmax: terrain.icpArgmax },
        {
          x0: side + gap + (side - size) / 2,
          q: terrain.ndt,
          title: 'NDT  ·  Gaussian mixture',
          argmax: terrain.ndtArgmax,
        },
      ];

      const cw = size / N;
      panels.forEach((panel, pi) => {
        label(ctx, panel.title, panel.x0, 12, pi === 0 ? p.prediction : p.posterior, {
          size: 10,
          weight: 600,
        });
        for (let j = 0; j < N; j++) {
          for (let i = 0; i < N; i++) {
            const q = panel.q[j * N + i];
            ctx.globalAlpha = 0.06 + 0.94 * q * q;
            ctx.fillStyle = pi === 0 ? p.prediction : p.posterior;
            ctx.fillRect(
              panel.x0 + i * cw,
              top + size - (j + 1) * cw,
              Math.ceil(cw) + 0.5,
              Math.ceil(cw) + 0.5,
            );
          }
        }
        ctx.globalAlpha = 1;

        // Ground truth sits at zero offset, by construction.
        const cx = panel.x0 + size / 2;
        const cy = top + size / 2;
        ctx.strokeStyle = p.truth;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 2]);
        ctx.beginPath();
        ctx.moveTo(cx - 7, cy);
        ctx.lineTo(cx + 7, cy);
        ctx.moveTo(cx, cy - 7);
        ctx.lineTo(cx, cy + 7);
        ctx.stroke();
        ctx.setLineDash([]);

        // Where this surface actually peaks.
        const [ai, aj] = panel.argmax;
        ctx.strokeStyle = pi === 0 ? p.prediction : p.posterior;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(panel.x0 + (ai + 0.5) * cw, top + size - (aj + 0.5) * cw, 4.5, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.strokeRect(panel.x0, top, size, size);
      });

      label(ctx, `Δx →   ±${SPAN.toFixed(1)} m`, panels[0].x0, top + size + 11, p.truth, { size: 9 });
      label(ctx, `Δy ↑   ±${SPAN.toFixed(1)} m`, panels[1].x0, top + size + 11, p.truth, { size: 9 });

      if (hover) {
        const panel = panels[hover.panel];
        const raw = hover.panel === 0 ? terrain.icpRaw : terrain.ndtRaw;
        const val = raw[hover.j * N + hover.i];
        const text =
          hover.panel === 0
            ? `Δ=(${cellToOffset(hover.i).toFixed(2)}, ${cellToOffset(hover.j).toFixed(2)}) · cost ${val.toFixed(4)} m²`
            : `Δ=(${cellToOffset(hover.i).toFixed(2)}, ${cellToOffset(hover.j).toFixed(2)}) · score ${val.toFixed(3)}`;
        ctx.strokeStyle = p.ink;
        ctx.lineWidth = 1;
        ctx.strokeRect(panel.x0 + hover.i * cw, top + size - (hover.j + 1) * cw, cw, cw);
        label(ctx, text, v.width / 2, v.height - 4, p.ink, { size: 10, align: 'center' });
      }
    },
    [terrain, hover],
  );

  const sliceSeries = useMemo(() => {
    const mid = Math.floor((N - 1) / 2);
    const xs = Array.from({ length: N }, (_, i) => cellToOffset(i));
    return [
      {
        id: 'ICP fit',
        role: 'prediction' as const,
        data: xs.map((x, i) => ({ x, y: terrain.icp[mid * N + i] })),
      },
      {
        id: 'NDT fit',
        role: 'posterior' as const,
        data: xs.map((x, i) => ({ x, y: terrain.ndt[mid * N + i] })),
      },
    ];
  }, [terrain]);

  const onPointer = useCallback((world: [number, number]) => {
    // World is 2 × 1 with the two panels side by side; convert straight back.
    const panel: 0 | 1 = world[0] < 1 ? 0 : 1;
    const u = (world[0] - panel) * 1.0;
    const i = Math.round(u * (N - 1));
    const j = Math.round(world[1] * (N - 1));
    if (i < 0 || i >= N || j < 0 || j >= N) setHover(null);
    else setHover({ panel, i, j });
  }, []);

  return (
    <WidgetFrame
      id="w16.3"
      title="Score Terrain"
      teaches="The registration cost is not a bowl. ICP's is piecewise with plateaus; NDT's is smooth; and in a corridor both are flat along one whole direction."
      colorKey={['prediction', 'posterior', 'truth']}
      caption={
        <>
          Both surfaces are the real objectives, evaluated on a 49 × 49 grid of translation offsets
          with the heading held fixed; brighter is a better fit. The gray cross is the true
          alignment, the ring is where that surface actually peaks.{' '}
          <strong>What to notice:</strong> ICP's terrain is built of flat terraces separated by
          ridges — the terraces are where correspondences stop changing, and a gradient method
          sitting on one has nothing to descend. NDT's is a smooth basin, which is why it can be
          optimized with Newton's method. <strong>What to try:</strong> switch to the corridor. Both
          terrains stretch into a valley along Δx, because translating a scan along two parallel
          walls produces an identical scan. No matcher recovers information the geometry does not
          contain; all it can do is notice, and that is what the information matrix is for.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: 2, maxY: 1 }}
        draw={draw}
        deps={[terrain, hover]}
        aspect={2}
        padding={0}
        ariaLabel="Two heat maps of registration fit quality over translation offsets. The ICP surface is terraced; the NDT surface is a smooth basin. In the corridor scene both stretch into a long valley."
        onPointer={onPointer}
        cursor="crosshair"
      />

      <div className="border-t border-fd-border p-3">
        <p className="eyebrow mb-1 m-0">horizontal slice at Δy = 0 — normalized fit</p>
        <LineChart
          series={sliceSeries}
          xLabel="Δx (m)"
          yLabel="fit"
          height={170}
          yMin={0}
          yMax={1.05}
          markers={[{ axis: 'x', value: 0, label: 'truth', role: 'truth' }]}
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Rejection radius τ"
          role="measurement"
          value={tau}
          min={0.15}
          max={1.2}
          step={0.05}
          unit="m"
          onChange={setTau}
          help="ICP's truncation distance. Larger τ widens the basin and flattens the terraces."
        />
        <div className="flex flex-col gap-1.5">
          <span className="font-ui text-[0.72rem] font-medium">Scene</span>
          <ButtonRow>
            <ActionButton onClick={() => setScene('corner')} emphasis={scene === 'corner'}>
              Room corner
            </ActionButton>
            <ActionButton onClick={() => setScene('corridor')} emphasis={scene === 'corridor'}>
              Corridor
            </ActionButton>
          </ButtonRow>
          <span className="font-ui text-[0.68rem] text-fd-muted-foreground">
            {SCENES[scene].blurb}
          </span>
        </div>
      </ControlPanel>
    </WidgetFrame>
  );
}
