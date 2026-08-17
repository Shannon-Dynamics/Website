'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { BarChart, Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { rayCast, type Segment, type World } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import { DEFAULT_BEAM_PARAMS, type BeamParams } from '@/lib/models/sensor';
import { OccupancyGrid, probToLogOdds, type InverseModelParams } from '@/lib/mapping/occgrid';
import {
  beamIncidence,
  beamLogLikelihood,
  bestFlip,
  buildConeBeams,
  mapLogPosterior,
  occupancyGridMappingAllCells,
  type ConeBeam,
} from '@/lib/mapping/map-occgrid';
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
 * w13.2 — the Independence Trap.
 *
 * Left: the standard log-odds mapper, run as Table 9.1 actually writes it —
 * every cell in the perceptual field, not just the cells on a beam's raster.
 * With a sonar-width cone the occupied evidence lands on the whole arc, two
 * poses disagree about the doorway, and the filter "resolves" the conflict by
 * counting votes.
 *
 * Right: Table 9.3. Binary maps scored by the *forward* beam model, hill-climbed
 * one flip at a time from the all-free map. A reading is explained as soon as
 * something in its cone blocks it, so the optimizer never needs the whole arc —
 * and the doorway survives.
 */

const CELL = 0.1;
const BOUNDS = { minX: 0, minY: 0, maxX: 4.2, maxY: 3.2 };
const MAX_RANGE = 4.5;
const BETA = (22 * Math.PI) / 180;
const SUB_RAYS = 5;
const N_BEAMS = 13;
const FOV = (100 * Math.PI) / 180;
/** Table 9.3 needs a prior below one half, or the all-occupied map wins. */
const MAP_PRIOR = 0.3;

const BEAM_MODEL: BeamParams = {
  ...DEFAULT_BEAM_PARAMS,
  maxRange: MAX_RANGE,
  sigmaHit: 0.14,
  zHit: 0.85,
  zShort: 0.05,
  zMax: 0.05,
  zRand: 0.05,
};

/** Eight sonar shots from the left half of the room, all facing the wall. */
const POSES: Pose2[] = [0.7, 1.5].flatMap((x) =>
  [0.6, 1.3, 2.0, 2.7].map((y) => ({ x, y, theta: 0 })),
);

type SceneKey = 'doorway' | 'pole' | 'corner';

const SCENE_LABEL: Record<SceneKey, string> = {
  doorway: 'Open doorway',
  pole: 'Thin pole',
  corner: 'Outside corner',
};

const SHELL: Segment[] = [
  { x1: 0.1, y1: 0.1, x2: 4.0, y2: 0.1 },
  { x1: 0.1, y1: 3.1, x2: 4.0, y2: 3.1 },
  { x1: 4.0, y1: 0.1, x2: 4.0, y2: 3.1 },
];

function sceneWalls(key: SceneKey): Segment[] {
  if (key === 'doorway') {
    return [
      ...SHELL,
      // A wall at x = 2.2 with a 0.6 m opening — Thrun's Figure 9.8 case.
      { x1: 2.2, y1: 0.1, x2: 2.2, y2: 1.25 },
      { x1: 2.2, y1: 1.85, x2: 2.2, y2: 3.1 },
    ];
  }
  if (key === 'pole') {
    // A pole two cells wide: thinner than the cone, so every beam that clips it
    // reports its range and the factored model smears it across the whole arc.
    return [
      ...SHELL,
      { x1: 2.1, y1: 1.5, x2: 2.3, y2: 1.5 },
      { x1: 2.3, y1: 1.5, x2: 2.3, y2: 1.7 },
      { x1: 2.3, y1: 1.7, x2: 2.1, y2: 1.7 },
      { x1: 2.1, y1: 1.7, x2: 2.1, y2: 1.5 },
    ];
  }
  return [
    ...SHELL,
    { x1: 2.2, y1: 0.1, x2: 2.2, y2: 1.6 },
    { x1: 2.2, y1: 1.6, x2: 3.6, y2: 1.6 },
  ];
}

/**
 * A sonar reading: the nearest surface **any** ray in the cone strikes.
 *
 * This is the physics that makes the inverse model's arc a lie. The sensor is
 * telling you "something in this 22° wedge is 1.6 m away", and the wedge is
 * exactly what Table 9.2 has to smear across.
 */
function coneCast(world: World, pose: Pose2, bearing: number): number {
  let best = MAX_RANGE;
  for (let s = 0; s < SUB_RAYS; s++) {
    const frac = s / (SUB_RAYS - 1) - 0.5;
    const r = rayCast(world, pose.x, pose.y, pose.theta + bearing + frac * BETA, MAX_RANGE);
    if (r < best) best = r;
  }
  return best;
}

interface Scene {
  key: SceneKey;
  world: World;
  angles: number[];
  scans: { ranges: number[]; angles: number[] }[];
  beams: ConeBeam[];
  incidence: number[][];
  cellCount: number;
  width: number;
  height: number;
}

/** Everything that depends only on the geometry: the data, and its compilation. */
function buildScene(key: SceneKey): Scene {
  const world: World = { name: `trap-${key}`, bounds: BOUNDS, walls: sceneWalls(key) };
  const angles = Array.from({ length: N_BEAMS }, (_, k) => -FOV / 2 + (k * FOV) / (N_BEAMS - 1));
  const scans = POSES.map((pose) => ({
    ranges: angles.map((a) => coneCast(world, pose, a)),
    angles,
  }));

  const grid = OccupancyGrid.forBounds(BOUNDS, CELL, MAP_PRIOR);
  const geom = {
    width: grid.width,
    height: grid.height,
    cellSize: grid.cellSize,
    origin: grid.origin,
  };
  const beams = buildConeBeams(geom, POSES, scans, {
    beta: BETA,
    subRays: SUB_RAYS,
    maxRange: MAX_RANGE,
  });
  const cellCount = grid.width * grid.height;
  return {
    key,
    world,
    angles,
    scans,
    beams,
    incidence: beamIncidence(beams, cellCount),
    cellCount,
    width: grid.width,
    height: grid.height,
  };
}

/** The factored filter's answer: batch-run Table 9.1 over all three scans. */
function buildStandardMap(scene: Scene, alpha: number): OccupancyGrid {
  const inverse: InverseModelParams = {
    alpha,
    beta: BETA,
    maxRange: MAX_RANGE,
    lOcc: probToLogOdds(0.7),
    lFree: probToLogOdds(0.3),
    l0: 0,
    clamp: 12,
  };
  const grid = OccupancyGrid.forBounds(BOUNDS, CELL, 0.5);
  POSES.forEach((pose, t) => {
    occupancyGridMappingAllCells(grid, pose, scene.scans[t].ranges, scene.scans[t].angles, inverse);
  });
  return grid;
}

interface FlipRecord {
  cell: number;
  gain: number;
  to: 0 | 1;
  perBeam: { beam: number; delta: number }[];
}

interface State {
  occ: Uint8Array;
  logPost: number;
  flips: number;
  last: FlipRecord | null;
  converged: boolean;
  history: number[];
}

export function IndependenceTrap() {
  const [sceneKey, setSceneKey] = useState<SceneKey>('doorway');
  const [alpha, setAlpha] = useState(0.35);
  const scene = useMemo(() => buildScene(sceneKey), [sceneKey]);
  const standard = useMemo(() => buildStandardMap(scene, alpha), [scene, alpha]);
  const setupRef = useRef(scene);
  setupRef.current = scene;

  const l0 = useMemo(() => probToLogOdds(MAP_PRIOR), []);

  const init = useCallback(
    (): State => {
      const s = setupRef.current;
      const occ = new Uint8Array(s.cellCount);
      return {
        occ,
        logPost: mapLogPosterior(s.beams, occ, BEAM_MODEL, l0),
        flips: 0,
        last: null,
        converged: false,
        history: [],
      };
    },
    [l0],
  );

  /** One iteration of the hill climb: take the single most profitable flip. */
  const advance = useCallback(
    (state: State): State => {
      const s = setupRef.current;
      if (state.converged) return state;
      const f = bestFlip(s.beams, state.occ, s.incidence, BEAM_MODEL, l0);
      if (!f) {
        return { ...state, converged: true };
      }

      // Attribute the gain to the individual measurements it improved — the
      // bars below the map. Only beams whose cone crosses the cell can move.
      const before = f.beams.map((b) => beamLogLikelihood(s.beams[b], state.occ, BEAM_MODEL));
      state.occ[f.cell] = f.to;
      const perBeam = f.beams.map((b, i) => ({
        beam: b,
        delta: beamLogLikelihood(s.beams[b], state.occ, BEAM_MODEL) - before[i],
      }));

      const logPost = state.logPost + f.gain;
      return {
        occ: state.occ,
        logPost,
        flips: state.flips + 1,
        last: { cell: f.cell, gain: f.gain, to: f.to, perBeam },
        converged: false,
        history: [...state.history, logPost].slice(-200),
      };
    },
    [l0],
  );

  const sim = useSimulation<State>({
    init,
    step: (s) => advance(s),
    fps: 8,
    autoplay: true,
    loop: false,
  });
  const { reset, setState, pause } = sim;

  // A new scene is a different landscape, so the hill climb starts over. α is
  // *not* in this list: it is a parameter of the inverse model only, and Table
  // 9.3 does not have one.
  const resetRef = useRef(reset);
  resetRef.current = reset;
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    resetRef.current();
  }, [scene]);

  const runToConvergence = useCallback(() => {
    pause();
    setState((s) => {
      let next = s;
      for (let n = 0; n < 600 && !next.converged; n++) next = advance(next);
      return next;
    });
  }, [advance, pause, setState]);

  // ---- drawing ----------------------------------------------------------

  const drawScene = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette, highlight: boolean) => {
      const s = setupRef.current;
      ctx.save();
      ctx.globalAlpha = 0.55;
      drawSegments(ctx, v, s.world.walls, p.truth, 1.6);
      ctx.restore();
      for (const pose of POSES) drawRobot(ctx, v, pose, p.measurement, 0.16);
      if (!highlight) return;
      // A few beam axes, so the reader can see which cones overlap.
      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      POSES.forEach((pose, t) => {
        s.scans[t].ranges.forEach((r, k) => {
          const a = pose.theta + s.angles[k];
          ctx.moveTo(sx(v, pose.x), sy(v, pose.y));
          ctx.lineTo(sx(v, pose.x + r * Math.cos(a)), sy(v, pose.y + r * Math.sin(a)));
        });
      });
      ctx.stroke();
      ctx.restore();
    },
    [],
  );

  const drawStandard = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      // The same grayscale ramp `drawOccupancyGrid` uses everywhere in the book:
      // white free, black occupied, mid-gray "no idea".
      const probs = standard.getProbabilityArray();
      const w = Math.ceil(sl(v, CELL)) + 1;
      for (let j = 0; j < standard.height; j++) {
        for (let i = 0; i < standard.width; i++) {
          const pr = probs[j * standard.width + i];
          const shade = Math.round(255 * (1 - pr));
          ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
          ctx.fillRect(sx(v, i * CELL), sy(v, (j + 1) * CELL), w, w);
        }
      }
      drawScene(ctx, v, p, true);
      label(ctx, 'log-odds filter (Table 9.1)', sx(v, 0.12), sy(v, 3.0), p.ink, {
        size: 11,
        weight: 700,
      });
    },
    [drawScene, standard],
  );

  const drawMap = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { occ } = sim.state;
      const s = setupRef.current;
      const w = Math.ceil(sl(v, CELL)) + 1;
      // A MAP estimate is a *binary* map: there is no gray to draw.
      ctx.fillStyle = p.free;
      ctx.fillRect(sx(v, 0), sy(v, s.height * CELL), sl(v, s.width * CELL), sl(v, s.height * CELL));
      ctx.fillStyle = p.occupied;
      for (let j = 0; j < s.height; j++) {
        for (let i = 0; i < s.width; i++) {
          if (occ[j * s.width + i] !== 1) continue;
          ctx.fillRect(sx(v, i * CELL), sy(v, (j + 1) * CELL), w, w);
        }
      }
      const last = sim.state.last;
      if (last) {
        const i = last.cell % s.width;
        const j = Math.floor(last.cell / s.width);
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx(v, i * CELL) - 1.5, sy(v, (j + 1) * CELL) - 1.5, w + 3, w + 3);
      }
      drawScene(ctx, v, p, false);
      label(ctx, 'MAP with forward models (Table 9.3)', sx(v, 0.12), sy(v, 3.0), p.ink, {
        size: 11,
        weight: 700,
      });
    },
    [drawScene, sim.state],
  );

  // ---- readouts ---------------------------------------------------------

  const barSeries = useMemo(() => {
    const last = sim.state.last;
    if (!last) return [];
    const rows = last.perBeam
      .slice()
      .sort((a, b) => b.delta - a.delta)
      .slice(0, 8)
      .map((r) => ({
        x: `pose ${Math.floor(r.beam / N_BEAMS) + 1} · beam ${(r.beam % N_BEAMS) + 1}`,
        y: r.delta,
      }));
    return [{ id: 'Δ log p(z | x, m)', role: 'measurement' as const, data: rows }];
  }, [sim.state.last]);

  const occupiedCount = useMemo(() => {
    let n = 0;
    for (let i = 0; i < sim.state.occ.length; i++) n += sim.state.occ[i];
    return n;
  }, [sim.state]);

  const doorwayProb = useMemo(() => {
    // The three cells in the middle of the opening: the number that decides
    // whether a planner will ever route the robot through this door.
    let sum = 0;
    for (let k = 0; k < 3; k++) sum += standard.probAtWorld(2.25, 1.4 + 0.1 * k);
    return sum / 3;
  }, [standard]);

  return (
    <WidgetFrame
      id="w13.2"
      title="The Independence Trap"
      teaches="Per-cell independence is not harmless bookkeeping: it manufactures conflicts the data never contained, and closes doors that are open."
      colorKey={['measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Eight noise-free sonar shots from the left, at a wall with a 0.6 m opening. On the left,
          the factored filter: every reading is painted across its whole 22° arc, so beams that graze
          the door post vote &ldquo;occupied&rdquo; over the very cells that the beam through the
          opening has already carved free. The filter cannot say &ldquo;these readings are
          consistent&rdquo;; it can only add and subtract, and the door ends up sealed by a
          one-cell wall with carved-out free space behind it — a map that is not merely wrong but
          <em> impossible</em>. On the right, the same eight scans scored by the <em>forward</em>
          model. Press play: the hill climb places obstacles one cell at a time, each flip annotated
          with the measurements whose likelihood it raised. Notice how few cells it needs, and that
          the ones it puts behind the doorway sit on the far wall — seen through the opening. That
          sparseness is not a rendering failure, it is the answer: a reading is explained as soon as
          <em> something</em> in its cone blocks it, so the joint posterior never demands the whole
          arc. Try the other scenes, and try widening α: the factored map degrades smoothly, and the
          MAP map does not move at all, because Table 9.3 has no α to widen.
        </>
      }
    >
      <div className="grid grid-cols-1 md:grid-cols-2 md:divide-x md:divide-fd-border">
        <SimCanvas
          world={BOUNDS}
          draw={drawStandard}
          deps={[standard]}
          aspect={4.2 / 3.2}
          padding={0.1}
          ariaLabel="Occupancy grid built by the standard log-odds filter: the wall is drawn but the doorway is filled with conflicting gray."
        />
        <SimCanvas
          world={BOUNDS}
          draw={drawMap}
          deps={[sim.tick, sim.state, scene]}
          aspect={4.2 / 3.2}
          padding={0.1}
          ariaLabel="Binary map found by maximum a posteriori occupancy mapping: obstacles placed one cell at a time, with the doorway left open."
        />
      </div>

      <div className="px-3 pt-3">
        <Dashboard columns={4}>
          <StatTile
            label="log posterior"
            value={sim.state.logPost}
            role="posterior"
            precision={1}
            sparkline={sim.state.history}
            trend={sim.state.last?.gain}
            trendLabel="gain of last flip"
          />
          <StatTile label="cells flipped" value={sim.state.flips} />
          <StatTile label="cells occupied" value={occupiedCount} />
          <StatTile
            label="p(doorway) — factored"
            value={doorwayProb}
            role="truth"
            precision={3}
          />
        </Dashboard>
      </div>

      <div className="px-3 pb-1 pt-3">
        {barSeries.length > 0 ? (
          <BarChart
            series={barSeries}
            xLabel="measurement"
            yLabel="Δ log-likelihood"
            height={190}
            ariaLabel="Bar chart of the change in each affected beam's log-likelihood caused by the most recent cell flip."
          />
        ) : (
          <p className="px-1 py-6 text-center font-ui text-xs text-fd-muted-foreground">
            Press play, or step once, to see which measurements the next flip explains.
          </p>
        )}
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Obstacle thickness α"
          role="measurement"
          value={alpha}
          min={0.1}
          max={0.8}
          step={0.05}
          unit="m"
          onChange={setAlpha}
          help="How thick the inverse model believes an obstacle is. Only the left pane has this knob — Table 9.3 has no α."
        />
        <div className="flex flex-col gap-1.5">
          <span className="eyebrow">Scene</span>
          <ButtonRow>
            {(Object.keys(SCENE_LABEL) as SceneKey[]).map((k) => (
              <ActionButton key={k} onClick={() => setSceneKey(k)} emphasis={k === sceneKey}>
                {SCENE_LABEL[k]}
              </ActionButton>
            ))}
            <ActionButton onClick={runToConvergence}>Run to convergence</ActionButton>
          </ButtonRow>
        </div>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.state.flips}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
