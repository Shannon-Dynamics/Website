'use client';

import { useCallback, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { ParticleFilter } from '@/lib/filters/pf';
import { dominantCluster } from '@/lib/localize/augmented-mcl';
import { GridLocalizer, coarsenedSigma } from '@/lib/localize/grid-localization';
import {
  DEFAULT_BEAM_PARAMS,
  LikelihoodField,
  logLikelihoodFieldRangeFinderModel,
} from '@/lib/models/sensor';
import { odomFromPoses, sampleMotionModelOdometry, type OdomAlphas } from '@/lib/models/motion';
import { RUSTY, RUSTY_LIDAR, diffDriveSlipStep, pursuePoint, raycastScan } from '@/lib/sim/rusty';
import { APARTMENT } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import { Rng } from '@/lib/prob/rng';
import {
  clear,
  drawRobot,
  drawWorld,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w12.2 — Grid vs. Cloud.
 *
 * The same logged run, the same scans, the same motion, decoded two ways:
 * `Grid_localization` (Table 8.1) on the left, `MCL` (Table 8.2) on the right.
 * The mathematics is identical — both are the Bayes filter of Chapter 5 — and
 * the economics is not remotely.
 *
 * The lesson lives in the numbers under each panel: the grid's cost is fixed by
 * the map and the resolution and is paid whether or not the robot is lost,
 * while MCL's cost is set by M and its error is stochastic rather than
 * quantized. Drag the resolution one notch finer and watch the cell count go up
 * by roughly 3× while the millisecond meter follows it.
 */

const FIELD = new LikelihoodField(APARTMENT, 0.05);
const LIDAR = { ...RUSTY_LIDAR, nBeams: 24, sigmaR: 0.03, pDropout: 0.02 };
const BEAM = { ...DEFAULT_BEAM_PARAMS, sigmaHit: 0.18, maxRange: LIDAR.maxRange };
const ALPHAS: OdomAlphas = [0.03, 0.02, 0.04, 0.02];
const DT = 0.4;
/** The grid gets every third beam; MCL every second. Both are subsampling. */
const GRID_STRIDE = 3;
const MCL_STRIDE = 2;

const RESOLUTIONS = [
  { cellSize: 0.8, nTheta: 8 },
  { cellSize: 0.6, nTheta: 12 },
  { cellSize: 0.4, nTheta: 16 },
  { cellSize: 0.3, nTheta: 24 },
];

const ROUTE = [
  { x: 2.05, y: 2.4 },
  { x: 2.05, y: 4.35 },
  { x: 5.6, y: 4.35 },
  { x: 7.85, y: 4.35 },
  { x: 7.85, y: 6.4 },
  { x: 7.85, y: 4.35 },
  { x: 3.0, y: 4.35 },
];

interface Params {
  res: number;
  logM: number;
}

interface State {
  rng: Rng;
  world: Rng;
  grid: GridLocalizer;
  resIndex: number;
  pf: ParticleFilter;
  truth: Pose2;
  dead: Pose2;
  wheels: [number, number];
  waypoint: number;
  marginal: Float64Array;
  gridErr: number;
  mclErr: number;
  gridMs: number;
  mclMs: number;
  estimate: Pose2;
  gridEstimate: Pose2;
}

function makeState(seed: number, resIndex: number, m: number): State {
  const rng = new Rng(seed);
  const spec = RESOLUTIONS[resIndex];
  const grid = new GridLocalizer(APARTMENT, spec);
  const truth: Pose2 = { x: 2.05, y: 2.4, theta: Math.PI / 2 };
  return {
    rng,
    world: new Rng(seed * 104729 + 5),
    grid,
    resIndex,
    pf: ParticleFilter.uniformInWorld(m, APARTMENT, rng, 0.2),
    truth,
    dead: { ...truth },
    wheels: [0, 0],
    waypoint: 1,
    marginal: grid.marginalXY(),
    gridErr: 0,
    mclErr: 0,
    gridMs: 0,
    mclMs: 0,
    estimate: { ...truth },
    gridEstimate: { ...truth },
  };
}

export function GridVsCloud() {
  const [params, setParams] = useState<Params>({ res: 1, logM: 10 });
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const init = useCallback(
    (seed: number) => makeState(seed, paramsRef.current.res, 2 ** paramsRef.current.logM),
    [],
  );

  const step = useCallback((s: State, _tick: number): State => {
    const p = paramsRef.current;
    const M = 2 ** p.logM;
    const next: State = { ...s };
    const { rng, world } = s;

    // A resolution change rebuilds the grid from scratch — which is itself the
    // point: you cannot re-resolve a histogram filter without throwing the
    // belief away, while a particle set is resized in one line.
    if (s.resIndex !== p.res) {
      next.grid = new GridLocalizer(APARTMENT, RESOLUTIONS[p.res]);
      next.resIndex = p.res;
    }
    if (s.pf.size !== M) {
      const w = s.pf.particles.map((q) => q.weight);
      s.pf.particles = Array.from({ length: M }, () => {
        const src = s.pf.particles[rng.choice(w)];
        return { state: { ...src.state }, weight: 1 / M };
      });
    }

    // ---- one logged step of the world -------------------------------------
    const wp = ROUTE[next.waypoint % ROUTE.length];
    const u = pursuePoint(s.truth, wp, { speed: 0.5, gain: 1.8, maxOmega: 1.1 });
    const out = diffDriveSlipStep(
      { pose: s.truth, wheelAngles: s.wheels },
      u,
      DT,
      APARTMENT,
      RUSTY,
      world,
    );
    next.truth = out.pose;
    next.wheels = out.wheelAngles;
    if (Math.hypot(out.pose.x - wp.x, out.pose.y - wp.y) < 0.45 || out.blocked) {
      next.waypoint = (s.waypoint + 1) % ROUTE.length;
    }
    const odom = odomFromPoses(s.truth, out.pose);

    const scan = raycastScan(APARTMENT, out.pose, LIDAR, world);
    const pack = (stride: number) => {
      const z: number[] = [];
      const angles: number[] = [];
      for (let k = 0; k < scan.ranges.length; k += stride) {
        z.push(scan.ranges[k]);
        angles.push(scan.angles[k]);
      }
      return { z, angles };
    };

    // ---- grid localization -------------------------------------------------
    const gridPack = pack(GRID_STRIDE);
    const cell = RESOLUTIONS[next.resIndex].cellSize;
    // Both models run with their noise inflated by the cell's own extent.
    const gridBeam = { ...BEAM, sigmaHit: coarsenedSigma(BEAM.sigmaHit, cell) };
    const g0 = now();
    next.grid.predict(odom, { sigmaTrans: 0.04, sigmaRot: 0.03 });
    next.grid.correctLog((pose) =>
      logLikelihoodFieldRangeFinderModel(
        gridPack.z,
        pose,
        FIELD,
        gridBeam,
        gridPack.angles,
        LIDAR.offset,
      ),
    );
    next.gridMs = 0.7 * s.gridMs + 0.3 * (now() - g0);
    next.marginal = next.grid.marginalXY();
    const gArg = next.grid.argmax();
    next.gridEstimate = gArg.pose;
    next.gridErr = Math.hypot(gArg.pose.x - out.pose.x, gArg.pose.y - out.pose.y);

    // ---- Monte Carlo localization -----------------------------------------
    const mclPack = pack(MCL_STRIDE);
    const m0 = now();
    s.pf.predict((x) => sampleMotionModelOdometry(odom, x, ALPHAS, rng));
    s.pf.correctLog((x) =>
      logLikelihoodFieldRangeFinderModel(mclPack.z, x, FIELD, BEAM, mclPack.angles, LIDAR.offset),
    );
    const cluster = dominantCluster(s.pf.particles, 0.7);
    s.pf.resample(rng, 'lowVariance');
    next.mclMs = 0.7 * s.mclMs + 0.3 * (now() - m0);
    next.estimate = cluster.pose;
    next.mclErr = Math.hypot(cluster.pose.x - out.pose.x, cluster.pose.y - out.pose.y);

    return next;
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 4, initialSeed: 21 });
  const s = sim.state;

  const drawGridPanel = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const g = s.grid;
      const m = s.marginal;
      let max = 0;
      for (const value of m) if (value > max) max = value;
      if (max > 0) {
        const w = Math.ceil(sl(v, g.cellSize)) + 1;
        ctx.save();
        ctx.fillStyle = p.posterior;
        for (let j = 0; j < g.ny; j++) {
          for (let i = 0; i < g.nx; i++) {
            const value = m[j * g.nx + i];
            if (value <= 0) continue;
            // √ of the normalized mass: a linear ramp hides everything but the
            // peak once the belief has condensed.
            ctx.globalAlpha = Math.min(1, Math.sqrt(value / max)) * 0.85;
            const [cx, cy] = g.cellCenterXY(i, j);
            ctx.fillRect(sx(v, cx - g.cellSize / 2), sy(v, cy + g.cellSize / 2), w, w);
          }
        }
        ctx.restore();
      }
      drawWorld(ctx, v, APARTMENT, p);
      drawRobot(ctx, v, s.gridEstimate, p.posterior, 0.28);
      ctx.save();
      ctx.setLineDash([4, 3]);
      drawRobot(ctx, v, s.truth, p.truth, 0.32, { filled: false });
      ctx.restore();
      label(ctx, `Grid_localization · ${g.cellSize} m × ${(360 / g.nTheta).toFixed(0)}°`, 8, 12, p.ink, {
        size: 10,
        weight: 600,
      });
    },
    [s],
  );

  const drawCloudPanel = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawWorld(ctx, v, APARTMENT, p);
      const particles = s.pf.particles;
      const stride = Math.max(1, Math.ceil(particles.length / 1200));
      ctx.save();
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.5;
      for (let i = 0; i < particles.length; i += stride) {
        const q = particles[i];
        ctx.beginPath();
        ctx.arc(sx(v, q.state.x), sy(v, q.state.y), 1.3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      drawRobot(ctx, v, s.estimate, p.posterior, 0.28);
      ctx.save();
      ctx.setLineDash([4, 3]);
      drawRobot(ctx, v, s.truth, p.truth, 0.32, { filled: false });
      ctx.restore();
      label(ctx, `MCL · M = ${particles.length.toLocaleString()}`, 8, 12, p.ink, {
        size: 10,
        weight: 600,
      });
    },
    [s],
  );

  const gridBytes = s.grid.bytes;
  const mclBytes = s.pf.size * 32; // x, y, θ, w as f64
  const fine = { cellSize: 0.15, nTheta: 72 };
  const fineCells =
    Math.ceil(12 / fine.cellSize) * Math.ceil(9 / fine.cellSize) * fine.nTheta;

  return (
    <WidgetFrame
      id="w12.2"
      title="Grid vs. Cloud"
      teaches="Grids and particles run the same recursion, but a grid pays for the whole map at every step while a particle set pays only for the hypotheses it is still entertaining."
      colorKey={['posterior', 'truth']}
      wide
      caption={
        <>
          One logged run, decoded twice. Left: the belief as a histogram over{' '}
          <em>pose</em> cells, marginalized over heading for drawing; right: the same belief as
          particles. Both hold the two-room ambiguity, both resolve it at the same moment, and both
          cost wildly different amounts to do so.
          <br />
          <strong>What to notice.</strong> The grid's error never goes below half a cell — that is
          its resolution, not its noise — and its bill is the same whether the robot is lost or
          perfectly tracked. MCL's error is stochastic and shrinks with the belief, and its bill is
          whatever you set M to. <strong>What to try.</strong> Take the resolution to 0.3 m and
          watch cells and milliseconds rise together; then ask what the literature's 15 cm × 5°
          grid — {fineCells.toLocaleString()} cells, {(fineCells * 8 / 1e6).toFixed(1)} MB at f64 —
          would cost in this browser.
        </>
      }
    >
      <div className="grid grid-cols-1 gap-px bg-fd-border sm:grid-cols-2">
        <SimCanvas
          world={APARTMENT.bounds}
          draw={drawGridPanel}
          deps={[sim.tick, s]}
          aspect={12 / 9}
          padding={0.2}
          ariaLabel="Grid localization: the apartment floorplan shaded by the probability of each pose cell."
        />
        <SimCanvas
          world={APARTMENT.bounds}
          draw={drawCloudPanel}
          deps={[sim.tick, s]}
          aspect={12 / 9}
          padding={0.2}
          ariaLabel="Monte Carlo localization: the same belief drawn as a cloud of particles."
        />
      </div>

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-6">
        <Stat label="cells" value={s.grid.cellCount.toLocaleString()} />
        <Stat label="grid memory" value={`${(gridBytes / 1024).toFixed(0)} KB`} />
        <Stat label="grid ms" value={s.gridMs.toFixed(1)} />
        <Stat label="particles" value={s.pf.size.toLocaleString()} />
        <Stat label="MCL memory" value={`${(mclBytes / 1024).toFixed(0)} KB`} />
        <Stat label="MCL ms" value={s.mclMs.toFixed(1)} />
      </div>
      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="grid |argmax − truth|" value={`${s.gridErr.toFixed(2)} m`} />
        <Stat label="MCL |est − truth|" value={`${s.mclErr.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Grid resolution"
          role="posterior"
          value={params.res}
          min={0}
          max={RESOLUTIONS.length - 1}
          step={1}
          onChange={(v) => setParams((q) => ({ ...q, res: v }))}
          format={(v) =>
            `${RESOLUTIONS[v].cellSize} m / ${(360 / RESOLUTIONS[v].nTheta).toFixed(0)}°`
          }
          help="The headline knob. Cost grows as the cube of 1/resolution, because pose space is three-dimensional."
        />
        <Slider
          label="Particles M = 2^k"
          value={params.logM}
          min={7}
          max={12}
          step={1}
          onChange={(v) => setParams((q) => ({ ...q, logM: v }))}
          format={(v) => (2 ** v).toLocaleString()}
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

const now = () => (typeof performance !== 'undefined' ? performance.now() : 0);

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
