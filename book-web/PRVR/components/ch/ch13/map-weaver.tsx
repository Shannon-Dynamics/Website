'use client';

import Link from 'next/link';
import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT, type Segment, type World } from '@/lib/sim/world';
import {
  RUSTY,
  diffDriveSlipStep,
  encoderTicks,
  integrateOdometry,
  odometryDelta,
  pursuePoint,
  raycastScan,
  type EncoderTicks,
  type LidarParams,
  type RustyState,
} from '@/lib/sim/rusty';
import {
  DEFAULT_INVERSE_MODEL,
  OccupancyGrid,
  probToLogOdds,
  type InverseModelParams,
} from '@/lib/mapping/occgrid';
import type { Pose2 } from '@/lib/geom/se2';
import {
  clear,
  drawPath,
  drawRobot,
  drawScan,
  drawSegments,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w13.1 — Map Weaver.
 *
 * The chapter's flagship and its integration lab in one box. Rusty drives a
 * fixed tour of the Apartment; every scan is folded into an `OccupancyGrid` by
 * the real `integrateScan` — Table 9.1 with Bresenham traversal — and the map
 * develops out of uniform gray like film.
 *
 * Two grids are maintained on every tick, not one. They see identical scans and
 * differ only in the pose each is told to believe: one gets the oracle's ground
 * truth (the assumption this whole chapter runs on), the other gets Rusty's
 * dead-reckoned estimate. Flipping between them is the chapter's closing
 * confession, made visible — and Chapter 14's hook.
 */

const CELL = 0.1;
const MAP_PRIOR = 0.5;

const LIDAR: LidarParams = {
  nBeams: 60,
  fov: 2 * Math.PI,
  maxRange: 6,
  sigmaR: 0.02,
  pDropout: 0.01,
  offset: [0.04, 0],
};

/** A tour of the corridor and the three south rooms. The north rooms stay gray. */
const TOUR: { x: number; y: number }[] = [
  { x: 1.2, y: 4.4 },
  { x: 5.0, y: 4.4 },
  { x: 6.0, y: 4.3 },
  { x: 6.0, y: 1.9 },
  { x: 6.0, y: 4.4 },
  { x: 9.9, y: 4.3 },
  { x: 9.9, y: 1.9 },
  { x: 9.9, y: 4.4 },
  { x: 6.0, y: 4.4 },
  { x: 2.0, y: 4.3 },
  { x: 2.0, y: 1.9 },
  { x: 2.0, y: 4.4 },
];

/**
 * Three probe cells, chosen to make the three states of a map legible:
 * a wall the robot passes repeatedly, a stretch of corridor it drives through,
 * and a closet in the north bedroom the tour never reaches.
 */
const PROBES = [
  { name: 'wall cell (4.5, 3.8)', x: 4.5, y: 3.82, role: 'posterior' as const },
  { name: 'corridor cell (7.0, 4.4)', x: 7.0, y: 4.4, role: 'measurement' as const },
  { name: 'closet cell (10.3, 8.8)', x: 10.25, y: 8.8, role: 'prior' as const },
];

const PERSON_RADIUS = 0.22;

/** A walker in the corridor, as four short wall segments the LiDAR can strike. */
function personWalls(t: number): Segment[] {
  const cx = 2.5 + 6.5 * (0.5 - 0.5 * Math.cos(t / 55));
  const cy = 4.35;
  const r = PERSON_RADIUS;
  return [
    { x1: cx - r, y1: cy - r, x2: cx + r, y2: cy - r },
    { x1: cx + r, y1: cy - r, x2: cx + r, y2: cy + r },
    { x1: cx + r, y1: cy + r, x2: cx - r, y2: cy + r },
    { x1: cx - r, y1: cy + r, x2: cx - r, y2: cy - r },
  ];
}

interface Params {
  /** The headline knob: how loudly one beam is allowed to speak. */
  strength: number;
  clamp: number;
  oracle: boolean;
  person: boolean;
}

interface Sample {
  t: number;
  wall: number;
  corridor: number;
  closet: number;
}

interface State {
  rng: Rng;
  truth: RustyState;
  odom: Pose2;
  prevTicks: EncoderTicks;
  /** Mapped from the oracle's pose — the chapter's stated assumption. */
  oracleGrid: OccupancyGrid;
  /** Mapped from Rusty's own dead reckoning — the assumption, dropped. */
  odomGrid: OccupancyGrid;
  scan: { ranges: number[]; angles: number[] };
  trail: { x: number; y: number }[];
  entropy: number[];
  probes: Sample[];
  waypoint: number;
  driftPos: number;
}

export function MapWeaver() {
  const [params, setParams] = useState<Params>({
    strength: 1,
    clamp: 12,
    oracle: true,
    person: false,
  });
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const model = useMemo<InverseModelParams>(
    () => ({
      ...DEFAULT_INVERSE_MODEL,
      maxRange: LIDAR.maxRange,
      beta: (7 * Math.PI) / 180,
      lOcc: params.strength * probToLogOdds(0.75),
      lFree: params.strength * probToLogOdds(0.35),
      l0: probToLogOdds(MAP_PRIOR),
      clamp: params.clamp,
    }),
    [params.strength, params.clamp],
  );
  const modelRef = useRef(model);
  modelRef.current = model;

  const probeIdx = useMemo(() => {
    const g = OccupancyGrid.forWorld(APARTMENT, CELL, MAP_PRIOR);
    return PROBES.map((p) => {
      const [i, j] = g.worldToCell(p.x, p.y);
      return g.index(i, j);
    });
  }, []);

  const init = useCallback((seed: number): State => {
    const truth: RustyState = {
      pose: { x: 1.2, y: 4.4, theta: 0 },
      wheelAngles: [0, 0],
    };
    return {
      rng: new Rng(seed),
      truth,
      odom: { ...truth.pose },
      prevTicks: encoderTicks(truth.wheelAngles, RUSTY),
      oracleGrid: OccupancyGrid.forWorld(APARTMENT, CELL, MAP_PRIOR),
      odomGrid: OccupancyGrid.forWorld(APARTMENT, CELL, MAP_PRIOR),
      scan: { ranges: [], angles: [] },
      trail: [{ ...truth.pose }],
      entropy: [],
      probes: [],
      waypoint: 1,
      driftPos: 0,
    };
  }, []);

  const step = useCallback(
    (s: State, tick: number): State => {
      const p = paramsRef.current;
      const m = modelRef.current;

      // ---- drive ------------------------------------------------------
      let waypoint = s.waypoint;
      const goal = TOUR[waypoint % TOUR.length];
      if (Math.hypot(goal.x - s.truth.pose.x, goal.y - s.truth.pose.y) < 0.28) {
        waypoint = (waypoint + 1) % TOUR.length;
      }
      const u = pursuePoint(s.truth.pose, TOUR[waypoint % TOUR.length], {
        speed: 0.55,
        gain: 2.2,
        maxOmega: 1.4,
        turnFirst: 0.55,
      });
      // A systematic right-wheel bias: the drift is a slow curve, not a jitter,
      // which is what makes a dead-reckoned map smear rather than blur.
      const drive = diffDriveSlipStep(
        s.truth,
        u,
        0.25,
        APARTMENT,
        { ...RUSTY, radiusBiasRight: 0.006, slipStd: 0.012 },
        s.rng,
      );
      const truth: RustyState = { pose: drive.pose, wheelAngles: drive.wheelAngles };

      // ---- odometry: what Rusty thinks it did -------------------------
      const ticks = encoderTicks(truth.wheelAngles, RUSTY);
      const odom = integrateOdometry(s.odom, odometryDelta(s.prevTicks, ticks, RUSTY));

      // ---- sense ------------------------------------------------------
      const world: World = p.person
        ? { ...APARTMENT, walls: [...APARTMENT.walls, ...personWalls(tick)] }
        : APARTMENT;
      const scan = raycastScan(world, truth.pose, LIDAR, s.rng);

      // ---- map: the same scan, twice, under two different pose stories --
      s.oracleGrid.integrateScan(truth.pose, scan.ranges, scan.angles, m);
      s.odomGrid.integrateScan(odom, scan.ranges, scan.angles, m);

      const grid = p.oracle ? s.oracleGrid : s.odomGrid;
      const entropy = [...s.entropy, grid.entropy()].slice(-160);
      const probes = [
        ...s.probes,
        {
          t: tick,
          wall: grid.logOdds[probeIdx[0]],
          corridor: grid.logOdds[probeIdx[1]],
          closet: grid.logOdds[probeIdx[2]],
        },
      ].slice(-160);
      const trail = [...s.trail, { x: truth.pose.x, y: truth.pose.y }].slice(-500);

      return {
        ...s,
        truth,
        odom,
        prevTicks: ticks,
        scan: { ranges: scan.ranges, angles: scan.angles },
        trail,
        entropy,
        probes,
        waypoint,
        driftPos: Math.hypot(odom.x - truth.pose.x, odom.y - truth.pose.y),
      };
    },
    [probeIdx],
  );

  const sim = useSimulation<State>({ init, step, fps: 12, initialSeed: 0xc0ffee & 0xffff });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;
      const showOracle = paramsRef.current.oracle;
      const grid = showOracle ? s.oracleGrid : s.odomGrid;

      // ---- the map ----------------------------------------------------
      // Grayscale, not the book palette: white free, black occupied, and the
      // background gray of "no evidence" is the whole point of the picture.
      const probs = grid.getProbabilityArray();
      const w = Math.ceil(sl(v, grid.cellSize)) + 1;
      ctx.fillStyle = p.unknown;
      ctx.fillRect(sx(v, grid.origin.x), sy(v, grid.origin.y + grid.height * grid.cellSize), sl(v, grid.width * grid.cellSize), sl(v, grid.height * grid.cellSize));
      for (let j = 0; j < grid.height; j++) {
        for (let i = 0; i < grid.width; i++) {
          const pr = probs[j * grid.width + i];
          if (Math.abs(pr - 0.5) < 0.015) continue;
          const shade = Math.round(255 * (1 - pr));
          ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
          ctx.fillRect(
            sx(v, grid.origin.x + i * grid.cellSize),
            sy(v, grid.origin.y + (j + 1) * grid.cellSize),
            w,
            w,
          );
        }
      }

      // The truth, faint, so the reader can audit the map against it.
      ctx.save();
      ctx.globalAlpha = 0.3;
      drawSegments(ctx, v, APARTMENT.walls, p.truth, 1.2);
      ctx.restore();

      if (paramsRef.current.person) {
        drawSegments(ctx, v, personWalls(sim.tick), p.prediction, 2);
      }

      drawPath(ctx, v, s.trail, p.truth, { dashed: true, alpha: 0.5, lineWidth: 1.2 });

      const shown = showOracle ? s.truth.pose : s.odom;
      drawScan(ctx, v, shown, s.scan.ranges, s.scan.angles, p.measurement, LIDAR.maxRange);
      drawRobot(ctx, v, s.truth.pose, p.truth, 0.26, { filled: showOracle });
      if (!showOracle) drawRobot(ctx, v, s.odom, p.prediction, 0.26);

      for (let k = 0; k < PROBES.length; k++) {
        const probe = PROBES[k];
        // Same three role colors the probe chart uses, so the boxes and the
        // curves below them are obviously the same three cells.
        ctx.strokeStyle = k === 0 ? p.posterior : k === 1 ? p.measurement : p.prior;
        ctx.lineWidth = 1.6;
        ctx.strokeRect(sx(v, probe.x - CELL / 2) - 1, sy(v, probe.y + CELL / 2) - 1, sl(v, CELL) + 2, sl(v, CELL) + 2);
      }

      label(
        ctx,
        showOracle ? 'poses: oracle (ground truth)' : 'poses: dead reckoning — Ch. 14 territory',
        sx(v, APARTMENT.bounds.minX) + 6,
        sy(v, APARTMENT.bounds.maxY) + 12,
        showOracle ? p.truth : p.prediction,
        { size: 10, weight: 600 },
      );
    },
    [sim.state, sim.tick],
  );

  const s = sim.state;
  const grid = params.oracle ? s.oracleGrid : s.odomGrid;
  const totalCells = grid.width * grid.height;
  const lastEntropy = s.entropy[s.entropy.length - 1] ?? totalCells;
  const resolved = useMemo(() => {
    let n = 0;
    for (let i = 0; i < grid.logOdds.length; i++) if (Math.abs(grid.logOdds[i]) > 2.2) n++;
    return (100 * n) / totalCells;
  }, [grid, totalCells, sim.tick]);

  const probeSeries = useMemo(
    () => [
      { id: PROBES[0].name, role: PROBES[0].role, data: s.probes.map((h) => ({ x: h.t, y: h.wall })) },
      { id: PROBES[1].name, role: PROBES[1].role, data: s.probes.map((h) => ({ x: h.t, y: h.corridor })) },
      { id: PROBES[2].name, role: PROBES[2].role, data: s.probes.map((h) => ({ x: h.t, y: h.closet })) },
    ],
    [s.probes],
  );

  const last = s.probes[s.probes.length - 1];
  const pOf = (l: number) => 1 - 1 / (1 + Math.exp(l));

  return (
    <WidgetFrame
      id="w13.1"
      title="Map Weaver"
      teaches="A map is not a drawing. It is one binary Bayes filter per cell, run in parallel — and gray means ignorance, not emptiness."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty tours the corridor and the three south rooms. The north rooms are never entered, and
          what the LiDAR cannot see through a doorway stays exactly the gray it started at —{' '}
          <em>p</em> = 0.5, which means &ldquo;no idea&rdquo;, not &ldquo;empty&rdquo;. Watch a single
          wall: one pass makes it a smudge, three passes make it a line. Below the map, three named
          cells plot their own private log odds: the wall cell climbs, the corridor cell falls, and
          the closet cell in the north bedroom sits flat on ℓ₀ = 0 for the whole run. Now try the two
          knobs. Push <strong>evidence strength</strong> to 3 and the walls thicken and the corridor
          punches through them — one confident beam is enough to be wrong with. Then turn the{' '}
          <strong>pose oracle off</strong>: the identical scans, integrated at Rusty&rsquo;s own
          dead-reckoned poses, smear the whole apartment into a double exposure. That is the fiction
          this chapter is built on, and it is why{' '}
          <Link href="/chapters/ch14-ekf-slam">Chapter 14</Link> exists.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state, params.oracle, params.person]}
        aspect={12 / 9}
        padding={0.3}
        ariaLabel="An occupancy grid map of the apartment developing out of uniform gray as the robot drives the corridor: walls darken, the corridor whitens, and the unvisited north rooms stay gray."
      />

      <div className="px-3 pt-3">
        <Dashboard columns={4}>
          <StatTile
            label="map entropy H(m)"
            value={lastEntropy / 1000}
            unit="kbit"
            role="posterior"
            precision={2}
            sparkline={s.entropy}
            trend={s.entropy.length > 1 ? s.entropy[s.entropy.length - 1] - s.entropy[s.entropy.length - 2] : undefined}
            trendLabel="bits this scan"
          />
          <StatTile label="cells resolved" value={resolved} unit="%" precision={1} />
          <StatTile
            label="p(wall probe)"
            value={pOf(last?.wall ?? 0)}
            role="posterior"
            precision={3}
          />
          <StatTile
            label="pose error |x̂ − x|"
            value={params.oracle ? 0 : s.driftPos}
            unit="m"
            role="truth"
            precision={3}
          />
        </Dashboard>
      </div>

      <div className="px-3 pb-3 pt-3">
        <LineChart
          series={probeSeries}
          xLabel="scan t"
          yLabel="log odds ℓ_{t,i}"
          height={200}
          markers={[{ axis: 'y', value: 0, label: 'ℓ₀ = 0 (ignorance)', role: 'prior' }]}
          ariaLabel="Log odds over time for three cells: a wall cell climbing positive, a corridor cell falling negative, and an unobserved closet cell flat at zero."
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Evidence strength"
          role="measurement"
          value={params.strength}
          min={0.2}
          max={3}
          step={0.05}
          onChange={(v) => setParams((q) => ({ ...q, strength: v }))}
          help="Scales l_occ and l_free together. At 1.0 one hit is worth log(0.75/0.25) = 1.10 nats."
        />
        <Slider
          label="Clamp ℓmax"
          value={params.clamp}
          min={1}
          max={40}
          step={0.5}
          onChange={(v) => setParams((q) => ({ ...q, clamp: v }))}
          help="Nav2 practice: bound the confidence a cell may accumulate so it can still change its mind."
        />
        <Toggle
          label="Pose oracle (known poses)"
          role="truth"
          checked={params.oracle}
          onChange={(v) => setParams((q) => ({ ...q, oracle: v }))}
        />
        <Toggle
          label="Person in the corridor"
          role="prediction"
          checked={params.person}
          onChange={(v) => setParams((q) => ({ ...q, person: v }))}
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
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
