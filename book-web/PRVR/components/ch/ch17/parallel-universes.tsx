'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { GridRbpf, type RbpfReport } from '@/lib/filters/rbpf';
import { OccupancyGrid } from '@/lib/mapping/occgrid';
import { applyOdom, odomFromPoses, type OdomDelta } from '@/lib/models/motion';
import { APARTMENT, simulateScan } from '@/lib/sim/world';
import { Rng } from '@/lib/prob/rng';
import type { Pose2 } from '@/lib/geom/se2';
import {
  clear,
  drawOccupancyGrid,
  drawPath,
  drawRobot,
  drawSegments,
  drawWorld,
  fitViewport,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import {
  BEAM_ANGLES,
  INVERSE_MODEL,
  LAP,
  MAP_OPTS,
  RBPF_BASE,
  SCAN_PARAMS,
  START_POSE,
  driveStep,
  odometryReading,
} from './apartment-lap';

/**
 * w17.1 — Parallel Universes.
 *
 * The chapter's flagship. Rusty drives the Apartment corridor and back while a
 * grid RBPF runs live: every particle carries its *own* occupancy map, built
 * from its *own* sampled trajectory. The tiles are those maps. There is no
 * "the map" anywhere in this widget, and that absence is the lesson.
 *
 * Everything on screen is the real `GridRbpf` from `lib/filters/rbpf.ts` — the
 * same scan matcher, improved proposal, and selective resampling the prose
 * derives.
 */

const TILES = 12;
const UNIT = { minX: 0, minY: 0, maxX: 1, maxY: 1 };

interface State {
  filter: GridRbpf;
  rng: Rng;
  world: Rng;
  truth: Pose2;
  dead: Pose2;
  report: RbpfReport;
  ranges: number[];
  error: number;
  deadError: number;
  resamples: number;
}

const EMPTY_REPORT: RbpfReport = {
  neff: 0,
  stepNeff: 0,
  resampled: false,
  distinctAncestors: 0,
  weights: [],
  bestIndex: 0,
  scanMatched: 0,
};

export function ParallelUniverses() {
  const [m, setM] = useState(16);
  const [improved, setImproved] = useState(true);
  const paramsRef = useRef({ m, improved });
  paramsRef.current = { m, improved };
  const mRef = useRef(m);
  mRef.current = m;

  const init = useCallback((seed: number): State => {
    const filter = GridRbpf.atPose(
      mRef.current,
      START_POSE,
      () => new OccupancyGrid(MAP_OPTS),
      {
        ...RBPF_BASE,
        angles: BEAM_ANGLES,
        inverse: INVERSE_MODEL,
        improvedProposal: paramsRef.current.improved,
      },
    );
    return {
      filter,
      rng: new Rng(seed),
      world: new Rng(seed ^ 0x5eed),
      truth: { ...START_POSE },
      dead: { ...START_POSE },
      report: { ...EMPTY_REPORT, neff: mRef.current, weights: filter.weights },
      ranges: [],
      error: 0,
      deadError: 0,
      resamples: 0,
    };
  }, []);

  const step = useCallback((s: State, tick: number): State => {
    const prev = s.truth;
    const truth = driveStep(prev, tick);
    // One odometry stream, shared by every particle — and it lies the same way
    // for all of them. Diversity comes from the proposal, not from the reading.
    const u: OdomDelta = odometryReading(odomFromPoses(prev, truth));
    const dead = applyOdom(s.dead, u);
    const ranges = simulateScan(APARTMENT, truth, SCAN_PARAMS, s.world);
    const report = s.filter.step(u, ranges, s.rng);
    const best = s.filter.best().pose;
    return {
      ...s,
      truth,
      dead,
      ranges,
      report,
      error: Math.hypot(best.x - truth.x, best.y - truth.y),
      deadError: Math.hypot(dead.x - truth.x, dead.y - truth.y),
      resamples: s.resamples + (report.resampled ? 1 : 0),
    };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 6, initialSeed: 3, maxTicks: LAP, loop: true });
  const { reset } = sim;

  // Changing the universe count restarts the lap: a particle set cannot be
  // resized mid-run without inventing maps for the newcomers.
  useEffect(() => {
    reset();
  }, [m, improved, reset]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { filter, truth, dead, report } = sim.state;
      const W = v.width;
      const H = v.height;

      // ---- top band: the world, the truth, the hypotheses -----------------
      const bandH = H * 0.36;
      const world = fitViewport(APARTMENT.bounds, W, bandH, 0.15);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W, bandH);
      ctx.clip();
      drawWorld(ctx, world, APARTMENT, p);

      // Raw odometry, for scale: this is what the robot would believe alone.
      drawRobot(ctx, world, dead, p.prediction, 0.3, { filled: false });
      label(ctx, 'odometry', sx(world, dead.x) + 8, sy(world, dead.y) - 10, p.prediction, { size: 9 });

      // Every particle's current pose, sized by weight.
      const wMax = Math.max(...report.weights, 1e-9);
      for (let i = 0; i < filter.particles.length; i++) {
        const q = filter.particles[i];
        const alpha = 0.22 + 0.7 * (report.weights[i] / wMax);
        drawRobot(ctx, world, q.pose, p.posterior, 0.24, { alpha });
      }
      drawRobot(ctx, world, truth, p.truth, 0.3);
      label(ctx, 'Rusty (truth)', sx(world, truth.x) + 9, sy(world, truth.y) + 11, p.truth, { size: 9 });
      ctx.restore();

      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, bandH);
      ctx.lineTo(W, bandH);
      ctx.stroke();

      // ---- the universes ---------------------------------------------------
      const cols = 4;
      const rows = 3;
      const gap = 4;
      const gridTop = bandH + 20;
      const tileW = (W - gap * (cols + 1)) / cols;
      const tileH = (H - gridTop - gap * (rows + 1)) / rows;
      const tileView = fitViewport(APARTMENT.bounds, tileW, tileH, 0.1);
      const ranked = filter.rankedIndices().slice(0, TILES);

      label(
        ctx,
        report.resampled
          ? `RESAMPLE — ${filter.size} universes redrawn from the survivors`
          : `${filter.size} universes, ${TILES} shown, sorted by weight`,
        gap + 2,
        bandH + 11,
        report.resampled ? p.posterior : p.truth,
        { size: 10, weight: report.resampled ? 700 : 500 },
      );

      ranked.forEach((idx, slot) => {
        const col = slot % cols;
        const row = Math.floor(slot / cols);
        const x0 = gap + col * (tileW + gap);
        const y0 = gridTop + row * (tileH + gap);
        const q = filter.particles[idx];
        const w = report.weights[idx] ?? 0;

        ctx.save();
        ctx.translate(x0, y0);
        ctx.beginPath();
        ctx.rect(0, 0, tileW, tileH);
        ctx.clip();
        ctx.fillStyle = p.bg;
        ctx.fillRect(0, 0, tileW, tileH);

        // This universe's private map, and the trajectory that produced it.
        drawOccupancyGrid(ctx, tileView, q.map.read, p);
        // Ground truth, faint: the walls the universe is trying to be.
        ctx.globalAlpha = 0.35;
        ctx.setLineDash([2, 3]);
        drawSegments(ctx, tileView, APARTMENT.walls, p.truth, 1);
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        drawPath(ctx, tileView, q.path, p.posterior, { lineWidth: 1.2, alpha: 0.9 });
        ctx.restore();

        // Border = weight. Bright green means "this universe explains the scan".
        ctx.save();
        ctx.strokeStyle = p.truth;
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, tileW - 1, tileH - 1);
        ctx.globalAlpha = Math.min(1, 0.18 + 0.82 * (w / wMax));
        ctx.strokeStyle = report.resampled ? p.posterior : p.measurement;
        ctx.lineWidth = report.resampled ? 2.5 : 2;
        ctx.strokeRect(x0 + 1, y0 + 1, tileW - 2, tileH - 2);
        ctx.restore();

        label(ctx, `w=${w.toFixed(3)}`, x0 + 4, y0 + 9, p.ink, { size: 9 });
      });
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const s = sim.state;
    return {
      neff: s.report.neff,
      ancestors: s.report.distinctAncestors,
      error: s.error,
      dead: s.deadError,
      resamples: s.resamples,
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w17.1"
      title="Parallel Universes"
      teaches="The filter does not have a map. It has M competing maps, and loop closure is natural selection among them."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Each tile is one particle&apos;s <em>private</em> occupancy grid, drawn on the trajectory
          that particle sampled, with the true floorplan dashed in gray behind it. Watch three
          things. First, the tiles disagree — the same corridor is drawn at different angles in
          different universes, because each one integrated its scans at a different pose. Second,
          the <strong style={{ color: 'var(--pr-measurement)' }}>green borders</strong> breathe:
          they are the normalized weights, and they surge when a scan agrees with that
          universe&apos;s own walls. Third, when the effective sample size falls below <em>M</em>/2 the
          frame flashes <strong style={{ color: 'var(--pr-posterior)' }}>purple</strong> and the
          losing universes are deleted and replaced with copies of the winners. Compare the
          hollow <strong style={{ color: 'var(--pr-prediction)' }}>orange</strong> robot — raw
          odometry, which is what any single universe would believe without its map — against the
          gray truth. Then turn off the measurement-aware proposal and drop <em>M</em> to 4: the
          tiles tear, because now the only way a particle can find the right pose is to guess it
          from the wheels.
        </>
      }
    >
      <SimCanvas
        world={UNIT}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={1.55}
        padding={0}
        ariaLabel="Top: the apartment floorplan with the true robot, the drifting odometry estimate, and the particle poses. Below: a grid of twelve small maps, one per particle, each built from that particle's own trajectory, with border brightness showing its weight."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-5">
        <Stat label="N_eff" value={stats.neff.toFixed(1)} alert={stats.neff < m / 2} />
        <Stat label="lineages" value={String(stats.ancestors)} />
        <Stat label="resamples" value={String(stats.resamples)} />
        <Stat label="best error" value={`${stats.error.toFixed(2)} m`} />
        <Stat label="odometry error" value={`${stats.dead.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Universes M"
          role="posterior"
          value={m}
          min={4}
          max={40}
          step={4}
          onChange={setM}
          format={(v) => String(Math.round(v))}
          help="How many hypotheses the filter can afford. Changing it restarts the lap."
        />
        <Toggle
          label="Measurement-aware proposal (gmapping)"
          role="measurement"
          checked={improved}
          onChange={setImproved}
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

function Stat({ label: l, value, alert }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={alert ? { color: 'var(--pr-prediction)' } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
