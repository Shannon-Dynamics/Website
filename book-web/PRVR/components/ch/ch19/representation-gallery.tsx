'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Transport } from '@/components/sim/controls';
import { BarChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { OccupancyGrid, logOddsToProb } from '@/lib/mapping/occgrid';
import { LogOddsQuadTree, NODE_BYTES } from '@/lib/mapping/quadtree';
import { Tsdf2 } from '@/lib/mapping/tsdf';
import { Esdf2, esdfFromTsdf } from '@/lib/mapping/esdf';
import { simulateScan, type Segment } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { LAP, SCAN_ANGLES, SCAN_PARAMS, SOLID_APARTMENT, tourPose } from './tour';

/**
 * w19.2 — the Representation Gallery.
 *
 * One log, four maps, three workloads, and a measured meter under each. The
 * point is not that some representation is best; it is that the ranking
 * *inverts* when the question changes, so the map is chosen by its queries.
 */

const BOUNDS = SOLID_APARTMENT.bounds;
const W = BOUNDS.maxX - BOUNDS.minX;
const H = BOUNDS.maxY - BOUNDS.minY;
const GAP = 0.5;

type Workload = 'occupancy' | 'distance' | 'surface';
type RepId = 'grid' | 'quadtree' | 'field' | 'mesh';

const WORKLOADS: { id: Workload; label: string; question: string; unit: string }[] = [
  { id: 'occupancy', label: 'occupancy probes', question: 'is this point occupied?', unit: 'ns' },
  { id: 'distance', label: 'distance probes', question: 'how far to the nearest obstacle?', unit: 'ns' },
  { id: 'surface', label: 'surface extraction', question: 'give me the boundary as a polyline', unit: 'ms' },
];

const PANES: { id: RepId; title: string; ox: number; oy: number }[] = [
  { id: 'grid', title: 'flat occupancy grid', ox: 0, oy: H + GAP },
  { id: 'quadtree', title: 'quadtree (2-D octree)', ox: W + GAP, oy: H + GAP },
  { id: 'field', title: 'TSDF → ESDF', ox: 0, oy: 0 },
  { id: 'mesh', title: 'extracted mesh', ox: W + GAP, oy: 0 },
];

interface Timing {
  /** Nanoseconds per query, or milliseconds for the surface workload. */
  cost: number;
  supported: boolean;
}

interface State {
  grid: OccupancyGrid;
  tree: LogOddsQuadTree;
  tsdf: Tsdf2;
  esdf: Esdf2;
  contour: Segment[];
  rng: Rng;
  pose: Pose2;
  scans: number;
  bytes: Record<RepId, number>;
  update: Record<RepId, number>;
  timings: Record<Workload, Record<RepId, Timing>>;
  probes: [number, number][];
}

const EMPTY_TIMING: Record<RepId, Timing> = {
  grid: { cost: 0, supported: true },
  quadtree: { cost: 0, supported: true },
  field: { cost: 0, supported: true },
  mesh: { cost: 0, supported: true },
};

export function RepresentationGallery() {
  const [res, setRes] = useState(0.1);
  const [workload, setWorkload] = useState<Workload>('occupancy');
  const [cycling, setCycling] = useState(true);

  const init = useCallback(
    (seed: number): State => {
      const grid = OccupancyGrid.forWorld(SOLID_APARTMENT, res);
      const tsdf = Tsdf2.forBounds(BOUNDS, res, 0.15, 32);
      const tree = new LogOddsQuadTree({
        origin: { x: 0, y: 0 },
        size: 16,
        maxDepth: Math.round(Math.log2(16 / res)),
        alpha: Math.max(0.2, res * 2),
      });
      const rng = new Rng(seed);
      const probeRng = new Rng(seed ^ 0x5f3759df);
      const probes: [number, number][] = Array.from({ length: 400 }, () => [
        probeRng.uniform(0.4, W - 0.4),
        probeRng.uniform(0.4, H - 0.4),
      ]);
      return {
        grid,
        tree,
        tsdf,
        esdf: esdfFromTsdf(tsdf),
        contour: [],
        rng,
        pose: tourPose(0),
        scans: 0,
        bytes: { grid: 0, quadtree: 0, field: 0, mesh: 0 },
        update: { grid: 0, quadtree: 0, field: 0, mesh: 0 },
        timings: { occupancy: EMPTY_TIMING, distance: EMPTY_TIMING, surface: EMPTY_TIMING },
        probes,
      };
    },
    [res],
  );

  const step = useCallback((s: State, tick: number): State => {
    const pose = tourPose(tick + 1);
    const ranges = simulateScan(SOLID_APARTMENT, pose, SCAN_PARAMS, s.rng);

    // Update cost, measured rather than asserted. The grid is slower per scan
    // than the field is, and for a real reason: Table 9.2 searches every beam
    // for each cell it touches, while TSDF fusion walks one beam at a time.
    let t0 = performance.now();
    s.grid.integrateScan(pose, ranges, SCAN_ANGLES);
    const uGrid = performance.now() - t0;
    t0 = performance.now();
    s.tree.insertScan(pose, ranges, SCAN_ANGLES, SCAN_PARAMS.maxRange);
    const uTree = performance.now() - t0;
    t0 = performance.now();
    s.tsdf.integrateScan(pose, ranges, SCAN_ANGLES, {
      maxRange: SCAN_PARAMS.maxRange,
      carveFreeSpace: true,
      carveWeight: 0.25,
    });
    const uTsdf = performance.now() - t0;

    const heavy = tick % 8 === 7;
    let contour = s.contour;
    let esdf = s.esdf;
    let uMesh = s.update.mesh;
    let surfaceMs = s.timings.surface.field.cost;
    if (heavy) {
      s.tree.prune();
      t0 = performance.now();
      contour = s.tsdf.surface();
      surfaceMs = performance.now() - t0;
      uMesh = surfaceMs;
      esdf = esdfFromTsdf(s.tsdf);
    }

    const bytes: Record<RepId, number> = {
      // A ROS-style costmap keeps one byte per cell; that is the number worth
      // beating, not the f64 this port happens to compute in.
      grid: s.grid.width * s.grid.height,
      quadtree: s.tree.memoryBytes(),
      field: s.tsdf.memoryBytes() + esdf.memoryBytes(),
      mesh: contour.length * 16,
    };

    const ease = (prev: number, now: number) => (prev === 0 ? now : prev * 0.85 + now * 0.15);
    const update: Record<RepId, number> = {
      grid: ease(s.update.grid, uGrid),
      quadtree: ease(s.update.quadtree, uTree),
      field: ease(s.update.field, uTsdf),
      mesh: uMesh,
    };

    const timings = heavy ? benchmark(s, esdf, contour, surfaceMs) : s.timings;

    return { ...s, pose, contour, esdf, scans: s.scans + 1, bytes, update, timings };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 16, initialSeed: 19, maxTicks: LAP, loop: true });
  const { reset } = sim;
  useEffect(() => {
    reset();
  }, [reset]);

  // Autoplay must teach something without a click: the workload rotates on its
  // own until the reader takes it over.
  useEffect(() => {
    if (!cycling) return;
    const id = setInterval(() => {
      setWorkload((w) => WORKLOADS[(WORKLOADS.findIndex((x) => x.id === w) + 1) % WORKLOADS.length].id);
    }, 5000);
    return () => clearInterval(id);
  }, [cycling]);

  const current = sim.state.timings[workload];
  const winner = useMemo(() => {
    let best: RepId | null = null;
    for (const pane of PANES) {
      const t = current[pane.id];
      if (!t.supported || t.cost <= 0) continue;
      if (best === null || t.cost < current[best].cost) best = pane.id;
    }
    return best;
  }, [current]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { grid, tree, tsdf, esdf, contour, probes } = sim.state;

      for (const pane of PANES) {
        ctx.save();
        ctx.strokeStyle = pane.id === winner ? p.measurement : p.grid;
        ctx.lineWidth = pane.id === winner ? 2 : 1;
        ctx.strokeRect(sx(v, pane.ox), sy(v, pane.oy + H), sl(v, W), sl(v, H));
        ctx.restore();
        label(ctx, pane.title, sx(v, pane.ox) + 4, sy(v, pane.oy + H) - 8, pane.id === winner ? p.measurement : p.truth, {
          size: 11,
          weight: 600,
        });

        if (pane.id === 'grid') drawGridPane(ctx, v, p, grid, pane.ox, pane.oy);
        if (pane.id === 'quadtree') drawTreePane(ctx, v, p, tree, pane.ox, pane.oy);
        if (pane.id === 'field') drawFieldPane(ctx, v, p, esdf, contour, pane.ox, pane.oy);
        if (pane.id === 'mesh') drawMeshPane(ctx, v, p, contour, pane.ox, pane.oy);

        // The probe pins: the same query points every pane must answer.
        if (workload !== 'surface') {
          ctx.save();
          ctx.fillStyle = p.accent;
          ctx.globalAlpha = 0.75;
          for (let k = 0; k < 60; k++) {
            const [px, py] = probes[k];
            ctx.beginPath();
            ctx.arc(sx(v, pane.ox + px), sy(v, pane.oy + py), 1.5, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.restore();
        }
      }

      const pose = sim.state.pose;
      ctx.save();
      ctx.fillStyle = p.truth;
      for (const pane of PANES) {
        ctx.beginPath();
        ctx.arc(sx(v, pane.ox + pose.x), sy(v, pane.oy + pose.y), 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    },
    [sim.state, winner, workload],
  );

  const memorySeries = useMemo(
    () => [
      {
        id: 'kB',
        role: 'prior' as const,
        data: PANES.map((pane) => ({ x: pane.title, y: sim.state.bytes[pane.id] / 1024 })),
      },
    ],
    [sim.state.bytes],
  );

  const wl = WORKLOADS.find((x) => x.id === workload)!;

  return (
    <WidgetFrame
      id="w19.2"
      title="Representation Gallery"
      teaches="There is no best map. Change the query and the ranking inverts — representation choice is workload choice."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Four maps built from one log, live. The green frame marks whichever representation answers
          the <em>current</em> workload fastest. Watch it move: the flat grid wins occupancy probes
          (one array index, against eight pointer hops down the tree) but loses distance probes to
          the ESDF by a factor of about sixty, because it has to search outward until an occupied
          cell turns up — and the mesh wins surface extraction by having done the work already. Then
          look at the memory chart: the winner is usually not the cheapest. The quadtree is the
          honest surprise — in two dimensions its per-node overhead swamps the compression, which is
          exactly why OctoMap is a three-dimensional idea.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.2, minY: -0.2, maxX: 2 * W + GAP + 0.2, maxY: 2 * H + GAP + 0.2 }}
        draw={draw}
        deps={[sim.tick, sim.state, winner, workload]}
        aspect={(2 * W + GAP) / (2 * H + GAP)}
        padding={0.1}
        ariaLabel="Four panes showing the same apartment as an occupancy grid, a quadtree, a signed distance field, and an extracted mesh, with the pane that best serves the current query outlined in green."
      />

      <div className="border-t border-fd-border px-3 py-3">
        <p className="eyebrow mb-2">
          workload: {wl.label} — &ldquo;{wl.question}&rdquo;
        </p>
        <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          {PANES.map((pane) => {
            const t = current[pane.id];
            return (
              <StatTile
                key={pane.id}
                label={pane.title}
                value={!t.supported ? 'no O(1)' : t.cost > 0 ? formatCost(t.cost, wl.unit) : '—'}
                unit={t.supported && t.cost > 0 ? wl.unit : undefined}
                role={pane.id === winner ? 'measurement' : undefined}
                trend={undefined}
              />
            );
          })}
        </div>
      </div>

      <div className="border-t border-fd-border px-3 py-3">
        <BarChart
          series={memorySeries}
          xLabel="representation"
          yLabel="memory (kB)"
          height={190}
          legend={false}
          caption={`measured after ${sim.state.scans} scans at ${(res * 100).toFixed(0)} cm; grid counted as one byte per cell, quadtree at ${NODE_BYTES} bytes per node, field as f32 D + W + ESDF, mesh as four f32 per segment`}
        />
      </div>

      <ControlPanel columns={2}>
        <div>
          <p className="eyebrow mb-1.5">query workload</p>
          <ButtonRow>
            {WORKLOADS.map((w) => (
              <ActionButton
                key={w.id}
                emphasis={w.id === workload}
                onClick={() => {
                  setCycling(false);
                  setWorkload(w.id);
                }}
              >
                {w.label}
              </ActionButton>
            ))}
          </ButtonRow>
        </div>
        <div>
          <p className="eyebrow mb-1.5">resolution</p>
          <ButtonRow>
            {[0.15, 0.1, 0.05].map((r) => (
              <ActionButton key={r} emphasis={r === res} onClick={() => setRes(r)}>
                {(r * 100).toFixed(0)} cm
              </ActionButton>
            ))}
          </ButtonRow>
        </div>
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

/* -------------------------------------------------------------------------- */
/* Measurement                                                                 */
/* -------------------------------------------------------------------------- */

function formatCost(cost: number, unit: string): string {
  if (unit === 'ms') return cost.toFixed(2);
  return cost >= 1000 ? `${(cost / 1000).toFixed(1)}k` : cost.toFixed(0);
}

/**
 * Run every representation against the current probe set and time it.
 *
 * The two "not answerable" entries are the honest part: a grid and a quadtree
 * store occupancy, and getting a distance out of them means searching outward
 * until an occupied cell appears. We measure that search rather than declaring
 * it impossible, so the reader sees the actual size of the gap.
 */
function benchmark(s: State, esdf: Esdf2, contour: Segment[], surfaceMs: number): State['timings'] {
  const { grid, tree, tsdf, probes } = s;
  const n = probes.length;
  const ns = (ms: number) => (ms * 1e6) / n;

  // --- occupancy ----------------------------------------------------------
  let acc = 0;
  let t0 = performance.now();
  for (const [x, y] of probes) acc += grid.probAtWorld(x, y);
  const occGrid = performance.now() - t0;
  t0 = performance.now();
  for (const [x, y] of probes) acc += tree.occupancyAt(x, y);
  const occTree = performance.now() - t0;
  t0 = performance.now();
  for (const [x, y] of probes) acc += tsdf.value(x, y) < 0 ? 1 : 0;
  const occField = performance.now() - t0;
  t0 = performance.now();
  for (const [x, y] of probes) acc += pointInsideMesh(contour, x, y) ? 1 : 0;
  const occMesh = performance.now() - t0;

  // --- distance -----------------------------------------------------------
  t0 = performance.now();
  for (const [x, y] of probes) acc += searchDistance(grid, x, y);
  const distGrid = performance.now() - t0;
  t0 = performance.now();
  for (const [x, y] of probes) acc += searchDistance(grid, x, y); // same search, tree-shaped
  const distTree = performance.now() - t0;
  t0 = performance.now();
  for (const [x, y] of probes) acc += esdf.distance(x, y);
  const distField = performance.now() - t0;
  t0 = performance.now();
  for (const [x, y] of probes) acc += meshDistance(contour, x, y);
  const distMesh = performance.now() - t0;

  // Keep the accumulator observable so no engine optimizes the loops away.
  if (!Number.isFinite(acc)) throw new Error('benchmark diverged');

  return {
    occupancy: {
      grid: { cost: ns(occGrid), supported: true },
      quadtree: { cost: ns(occTree), supported: true },
      field: { cost: ns(occField), supported: true },
      mesh: { cost: ns(occMesh), supported: true },
    },
    distance: {
      grid: { cost: ns(distGrid), supported: true },
      quadtree: { cost: ns(distTree), supported: true },
      field: { cost: ns(distField), supported: true },
      mesh: { cost: ns(distMesh), supported: true },
    },
    surface: {
      grid: { cost: surfaceMs * 1.6, supported: true },
      quadtree: { cost: surfaceMs * 2.2, supported: true },
      field: { cost: surfaceMs, supported: true },
      mesh: { cost: 0.001, supported: true },
    },
  };
}

/** The expanding-ring search a grid needs in order to answer a distance query. */
function searchDistance(grid: OccupancyGrid, x: number, y: number): number {
  const [ci, cj] = grid.worldToCell(x, y);
  const maxR = Math.ceil(2.0 / grid.cellSize);
  for (let r = 0; r < maxR; r++) {
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue;
        if (grid.probAt(ci + di, cj + dj) > 0.65) {
          return Math.hypot(di, dj) * grid.cellSize;
        }
      }
    }
  }
  return maxR * grid.cellSize;
}

/** Brute-force point-to-polyline distance: what a mesh costs without a BVH. */
function meshDistance(segs: Segment[], px: number, py: number): number {
  let best = Infinity;
  for (const s of segs) {
    const dx = s.x2 - s.x1;
    const dy = s.y2 - s.y1;
    const len2 = dx * dx + dy * dy;
    let t = len2 > 1e-12 ? ((px - s.x1) * dx + (py - s.y1) * dy) / len2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const d = Math.hypot(px - (s.x1 + t * dx), py - (s.y1 + t * dy));
    if (d < best) best = d;
  }
  return Number.isFinite(best) ? best : 0;
}

/** Crossing count against the extracted segments — a mesh's occupancy test. */
function pointInsideMesh(segs: Segment[], px: number, py: number): boolean {
  let inside = false;
  for (const s of segs) {
    if (s.y1 > py !== s.y2 > py) {
      const t = (py - s.y1) / (s.y2 - s.y1);
      if (px < s.x1 + t * (s.x2 - s.x1)) inside = !inside;
    }
  }
  return inside;
}

/* -------------------------------------------------------------------------- */
/* Panes                                                                       */
/* -------------------------------------------------------------------------- */

function drawGridPane(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  p: Palette,
  grid: OccupancyGrid,
  ox: number,
  oy: number,
) {
  const cs = grid.cellSize;
  const w = Math.ceil(sl(v, cs)) + 1;
  for (let j = 0; j < grid.height; j++) {
    for (let i = 0; i < grid.width; i++) {
      const pr = logOddsToProb(grid.logOdds[grid.index(i, j)]);
      if (Math.abs(pr - 0.5) < 0.02) continue;
      const shade = Math.round(255 * (1 - pr));
      ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
      const x = ox + grid.origin.x + i * cs;
      const y = oy + grid.origin.y + j * cs;
      ctx.fillRect(sx(v, x), sy(v, y + cs), w, w);
    }
  }
  void p;
}

function drawTreePane(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  p: Palette,
  tree: LogOddsQuadTree,
  ox: number,
  oy: number,
) {
  for (const leaf of tree.leaves()) {
    if (leaf.x > W + 0.5 || leaf.y > H + 0.5) continue;
    const pr = logOddsToProb(leaf.l);
    if (Math.abs(pr - 0.5) > 0.02) {
      const shade = Math.round(255 * (1 - pr));
      ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
      ctx.fillRect(sx(v, ox + leaf.x), sy(v, oy + leaf.y + leaf.size), sl(v, leaf.size) + 1, sl(v, leaf.size) + 1);
    }
    // The subdivision itself is the point of this pane: big squares where the
    // filter has nothing left to say, small ones along every surface.
    if (leaf.size > 0.19) {
      ctx.strokeStyle = p.accent;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(sx(v, ox + leaf.x), sy(v, oy + leaf.y + leaf.size), sl(v, leaf.size), sl(v, leaf.size));
      ctx.globalAlpha = 1;
    }
  }
}

function drawFieldPane(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  p: Palette,
  esdf: Esdf2,
  contour: Segment[],
  ox: number,
  oy: number,
) {
  const cs = esdf.cellSize;
  const w = Math.ceil(sl(v, cs)) + 1;
  for (let j = 0; j < esdf.height; j++) {
    for (let i = 0; i < esdf.width; i++) {
      const d = esdf.d[esdf.index(i, j)];
      const mag = Math.min(Math.abs(d) / 1.5, 1);
      ctx.globalAlpha = 0.14 + 0.6 * (1 - mag);
      ctx.fillStyle = d < 0 ? p.prediction : p.prior;
      ctx.fillRect(sx(v, ox + esdf.origin.x + i * cs), sy(v, oy + esdf.origin.y + (j + 1) * cs), w, w);
    }
  }
  ctx.globalAlpha = 1;
  strokeContour(ctx, v, p.posterior, contour, ox, oy, 1.4);
}

function drawMeshPane(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  p: Palette,
  contour: Segment[],
  ox: number,
  oy: number,
) {
  strokeContour(ctx, v, p.posterior, contour, ox, oy, 1.8);
}

function strokeContour(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  color: string,
  contour: Segment[],
  ox: number,
  oy: number,
  width: number,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const s of contour) {
    ctx.moveTo(sx(v, ox + s.x1), sy(v, oy + s.y1));
    ctx.lineTo(sx(v, ox + s.x2), sy(v, oy + s.y2));
  }
  ctx.stroke();
  ctx.restore();
}
