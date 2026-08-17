'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Toggle, Transport } from '@/components/sim/controls';
import { Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { beamAngles, simulateScan, type Segment, type World } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import { OccupancyGrid, probToLogOdds, type InverseModelParams } from '@/lib/mapping/occgrid';
import { fuseGrids, occupancyGridMappingAllCells } from '@/lib/mapping/map-occgrid';
import {
  clear,
  drawRobot,
  drawSegments,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w13.4 — Fusion Fumble.
 *
 * Thrun et al. §9.2.1 in one picture. A sonar ring at table height and a LiDAR
 * at shin height map the same room. They do not answer the same question: the
 * sonar sees a table top, the LiDAR sees four thin legs and a great deal of
 * free space where the table is. Adding their log odds — which is what running
 * one Bayes filter per cell over both sensors does — lets whichever sensor is
 * polled more often win, and the table is deleted. Eq. (9.9) keeps it.
 */

const BOUNDS = { minX: 0, minY: 0, maxX: 5, maxY: 4 };
const CELL = 0.08;
const MAX_RANGE = 6;

const ROOM: Segment[] = [
  { x1: 0.15, y1: 0.15, x2: 4.85, y2: 0.15 },
  { x1: 4.85, y1: 0.15, x2: 4.85, y2: 3.85 },
  { x1: 4.85, y1: 3.85, x2: 0.15, y2: 3.85 },
  { x1: 0.15, y1: 3.85, x2: 0.15, y2: 0.15 },
];

/** The table top, 1.5 m × 0.9 m in the middle of the room. */
const TABLE = { x0: 1.75, y0: 1.55, x1: 3.25, y1: 2.45 };

const tableTop: Segment[] = [
  { x1: TABLE.x0, y1: TABLE.y0, x2: TABLE.x1, y2: TABLE.y0 },
  { x1: TABLE.x1, y1: TABLE.y0, x2: TABLE.x1, y2: TABLE.y1 },
  { x1: TABLE.x1, y1: TABLE.y1, x2: TABLE.x0, y2: TABLE.y1 },
  { x1: TABLE.x0, y1: TABLE.y1, x2: TABLE.x0, y2: TABLE.y0 },
];

/** The same table at shin height: four 6 cm legs and a lot of nothing. */
const tableLegs: Segment[] = [
  [TABLE.x0, TABLE.y0],
  [TABLE.x1, TABLE.y0],
  [TABLE.x1, TABLE.y1],
  [TABLE.x0, TABLE.y1],
].flatMap(([lx, ly]) => {
  const h = 0.03;
  return [
    { x1: lx - h, y1: ly - h, x2: lx + h, y2: ly - h },
    { x1: lx + h, y1: ly - h, x2: lx + h, y2: ly + h },
    { x1: lx + h, y1: ly + h, x2: lx - h, y2: ly + h },
    { x1: lx - h, y1: ly + h, x2: lx - h, y2: ly - h },
  ];
});

const SONAR_WORLD: World = { name: 'room (table height)', bounds: BOUNDS, walls: [...ROOM, ...tableTop] };
const LIDAR_WORLD: World = { name: 'room (shin height)', bounds: BOUNDS, walls: [...ROOM, ...tableLegs] };

const POSES: Pose2[] = [
  { x: 0.8, y: 0.8, theta: 0.6 },
  { x: 4.2, y: 0.8, theta: 2.3 },
  { x: 4.2, y: 3.2, theta: 3.7 },
  { x: 0.8, y: 3.2, theta: 5.4 },
  { x: 2.5, y: 0.6, theta: 1.57 },
  { x: 2.5, y: 3.4, theta: 4.71 },
];

const SONAR_SCAN = { nBeams: 24, fov: 2 * Math.PI, maxRange: MAX_RANGE, sigma: 0.08 };
const LIDAR_SCAN = { nBeams: 120, fov: 2 * Math.PI, maxRange: MAX_RANGE, sigma: 0.02 };

const SONAR_MODEL: InverseModelParams = {
  alpha: 0.35,
  beta: (20 * Math.PI) / 180,
  maxRange: MAX_RANGE,
  lOcc: probToLogOdds(0.7),
  lFree: probToLogOdds(0.35),
  l0: 0,
  clamp: 8,
};

const LIDAR_MODEL: InverseModelParams = {
  alpha: 0.14,
  beta: (4 * Math.PI) / 180,
  maxRange: MAX_RANGE,
  lOcc: probToLogOdds(0.8),
  lFree: probToLogOdds(0.25),
  l0: 0,
  clamp: 8,
};

interface State {
  rng: Rng;
  sonar: OccupancyGrid;
  lidar: OccupancyGrid;
  visited: number;
}

/** Mean occupancy probability over the table's footprint. */
function tableProb(probs: Float64Array, grid: OccupancyGrid): number {
  let sum = 0;
  let n = 0;
  const [i0, j0] = grid.worldToCell(TABLE.x0 + 0.1, TABLE.y0 + 0.1);
  const [i1, j1] = grid.worldToCell(TABLE.x1 - 0.1, TABLE.y1 - 0.1);
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      sum += probs[j * grid.width + i];
      n++;
    }
  }
  return n > 0 ? sum / n : 0.5;
}

export function FusionFumble() {
  const [useMax, setUseMax] = useState(false);
  const useMaxRef = useRef(useMax);
  useMaxRef.current = useMax;

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      sonar: OccupancyGrid.forBounds(BOUNDS, CELL, 0.5),
      lidar: OccupancyGrid.forBounds(BOUNDS, CELL, 0.5),
      visited: 0,
    }),
    [],
  );

  const step = useCallback((s: State, tick: number): State => {
    const pose = POSES[tick % POSES.length];

    // The sonar's cone is wider than its beam spacing, so it needs the literal
    // Table 9.1 sweep; the LiDAR's pencil beams are exactly what Bresenham
    // traversal was built for.
    const sonarRanges = simulateScan(SONAR_WORLD, pose, SONAR_SCAN, s.rng);
    occupancyGridMappingAllCells(
      s.sonar,
      pose,
      sonarRanges,
      beamAngles(SONAR_SCAN),
      SONAR_MODEL,
    );

    const lidarRanges = simulateScan(LIDAR_WORLD, pose, LIDAR_SCAN, s.rng);
    s.lidar.integrateScan(pose, lidarRanges, beamAngles(LIDAR_SCAN), LIDAR_MODEL);

    return { ...s, visited: s.visited + 1 };
  }, []);

  const sim = useSimulation<State>({
    init,
    step,
    fps: 1.6,
    maxTicks: POSES.length,
    loop: true,
    initialSeed: 13,
  });

  const fused = useMemo(
    () => fuseGrids([sim.state.sonar, sim.state.lidar], useMax ? 'max' : 'sum'),
    [sim.state, useMax],
  );

  const paint = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      v: Viewport,
      p: Palette,
      probs: Float64Array,
      width: number,
      height: number,
      title: string,
      truth: Segment[],
    ) => {
      clear(ctx, v, p);
      const w = Math.ceil(sl(v, CELL)) + 1;
      for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
          const pr = probs[j * width + i];
          const shade = Math.round(255 * (1 - pr));
          ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
          ctx.fillRect(sx(v, i * CELL), sy(v, (j + 1) * CELL), w, w);
        }
      }
      ctx.save();
      ctx.globalAlpha = 0.5;
      drawSegments(ctx, v, truth, p.truth, 1.2);
      ctx.restore();
      for (let k = 0; k <= Math.min(sim.tick, POSES.length) - 1; k++) {
        drawRobot(ctx, v, POSES[k], p.measurement, 0.14, { alpha: 0.7 });
      }
      label(ctx, title, sx(v, 0.2), sy(v, 3.65), p.ink, { size: 11, weight: 700 });
    },
    [sim.tick],
  );

  const g = sim.state.sonar;
  const drawSonar = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) =>
      paint(
        ctx,
        v,
        p,
        sim.state.sonar.getProbabilityArray(),
        g.width,
        g.height,
        'sonar · table height',
        SONAR_WORLD.walls,
      ),
    [paint, sim.state, g.width, g.height],
  );
  const drawLidar = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) =>
      paint(
        ctx,
        v,
        p,
        sim.state.lidar.getProbabilityArray(),
        g.width,
        g.height,
        'LiDAR · shin height',
        LIDAR_WORLD.walls,
      ),
    [paint, sim.state, g.width, g.height],
  );
  const drawFused = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) =>
      paint(
        ctx,
        v,
        p,
        fused.getProbabilityArray(),
        g.width,
        g.height,
        useMaxRef.current ? 'fused · max (eq. 9.9)' : 'fused · Σ log odds',
        SONAR_WORLD.walls,
      ),
    [paint, fused, g.width, g.height],
  );

  const stats = useMemo(() => {
    const sonarP = tableProb(sim.state.sonar.getProbabilityArray(), g);
    const lidarP = tableProb(sim.state.lidar.getProbabilityArray(), g);
    const sumP = tableProb(fuseGrids([sim.state.sonar, sim.state.lidar], 'sum').getProbabilityArray(), g);
    const maxP = tableProb(fuseGrids([sim.state.sonar, sim.state.lidar], 'max').getProbabilityArray(), g);
    return { sonarP, lidarP, sumP, maxP };
  }, [sim.state, g]);

  return (
    <WidgetFrame
      id="w13.4"
      title="Fusion Fumble"
      teaches="More sensors plus more Bayes does not automatically mean a better map: adding log odds is only valid when every sensor answers the same question."
      colorKey={['measurement', 'truth']}
      wide
      caption={
        <>
          The same room, mapped by two sensors that disagree about what an obstacle is. The sonar
          rides at table height and sees a solid table; the LiDAR rides at shin height, threads
          between four 6 cm legs, and reports the table&rsquo;s footprint as free — <em>correctly</em>,
          for its own slice of the world. Adding their log odds is what a single per-cell Bayes filter
          fed both streams would do, and the LiDAR&rsquo;s 120 beams outvote the sonar&rsquo;s 24: the
          table dissolves, and a planner routes the robot straight through it. Flip to{' '}
          <strong>max fusion</strong> — one grid per modality, combined pessimistically — and the
          table comes back. The bias is deliberate: a map that hallucinates obstacles wastes a path,
          a map that deletes them wastes a robot.
        </>
      }
    >
      <div className="grid grid-cols-1 md:grid-cols-3 md:divide-x md:divide-fd-border">
        <SimCanvas
          world={BOUNDS}
          draw={drawSonar}
          deps={[sim.tick, sim.state]}
          aspect={5 / 4}
          padding={0.08}
          ariaLabel="Occupancy map from the sonar: the room outline and a solid dark rectangle where the table is."
        />
        <SimCanvas
          world={BOUNDS}
          draw={drawLidar}
          deps={[sim.tick, sim.state]}
          aspect={5 / 4}
          padding={0.08}
          ariaLabel="Occupancy map from the LiDAR: the room outline, four tiny dark dots for the table legs, and free space where the table top is."
        />
        <SimCanvas
          world={BOUNDS}
          draw={drawFused}
          deps={[sim.tick, sim.state, useMax]}
          aspect={5 / 4}
          padding={0.08}
          ariaLabel="The fused map, either summing log odds or taking the per-cell maximum probability."
        />
      </div>

      <div className="px-3 py-3">
        <Dashboard columns={4}>
          <StatTile label="p(table) · sonar" value={stats.sonarP} role="measurement" precision={3} />
          <StatTile label="p(table) · LiDAR" value={stats.lidarP} role="measurement" precision={3} />
          <StatTile label="p(table) · Σ log odds" value={stats.sumP} precision={3} />
          <StatTile label="p(table) · max" value={stats.maxP} role="posterior" precision={3} />
        </Dashboard>
      </div>

      <ControlPanel columns={1}>
        <Toggle
          label="Max fusion (one grid per modality, eq. 9.9)"
          role="posterior"
          checked={useMax}
          onChange={setUseMax}
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
