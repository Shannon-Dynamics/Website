'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import type { Pose2 } from '@/lib/geom/se2';
import { APARTMENT, beamAngles, simulateScan, type Segment, type World } from '@/lib/sim/world';
import {
  DEFAULT_BEAM_PARAMS,
  LikelihoodField,
  logBeamRangeFinderModel,
  logLikelihoodFieldRangeFinderModel,
  type BeamParams,
} from '@/lib/models/sensor';
import { distanceAt, exactDistanceField } from '@/lib/mapping/edt';
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
 * w10.2 — the Likelihood Field Explorer.
 *
 * The same scan, scored two ways over the same patch of floor. The beam model
 * ray-casts every hypothesis and produces a surface full of cliffs; the
 * likelihood field looks each endpoint up in a precomputed distance transform
 * and produces a surface you could roll a marble down. Neither is "right" — the
 * beam model is the one with a physical story, and the field is the one a
 * filter can actually search. The chip in the corner is the other half of the
 * argument: cost per evaluation, measured live, as the room fills with clutter.
 */

const MAX_RANGE = 8;
const N_BEAMS = 40;
const CELL = 0.06;
/** Heat-map grid over a window centred on the true pose. */
const GX = 68;
const GY = 50;
const WIN = 1.7; // half-width of the window, metres
const PROFILE_N = 241;

const TRUTH: Pose2 = { x: 6.0, y: 2.05, theta: 0.35 };
const SCAN_PARAMS = {
  nBeams: N_BEAMS,
  fov: 2 * Math.PI,
  maxRange: MAX_RANGE,
  sigma: 0.05,
  zRand: 0.01,
  zMax: 0.02,
};

/** Chair and table legs: the clutter Thrun's "lack of smoothness" argument is about. */
function withClutter(seed: number, n: number): World {
  if (n === 0) return APARTMENT;
  const rng = new Rng(seed);
  const walls: Segment[] = [...APARTMENT.walls];
  for (let k = 0; k < n; k++) {
    const x = rng.uniform(4.3, 7.7);
    const y = rng.uniform(0.4, 3.4);
    const a = rng.uniform(0, Math.PI);
    const l = 0.11;
    walls.push({
      x1: x - l * Math.cos(a),
      y1: y - l * Math.sin(a),
      x2: x + l * Math.cos(a),
      y2: y + l * Math.sin(a),
    });
  }
  return { ...APARTMENT, walls };
}

interface Surface {
  grid: Float64Array;
  lo: number;
  hi: number;
  profile: number[];
  profileX: number[];
  /** Largest change in log-likelihood between neighbouring profile samples. */
  maxStep: number;
  evalsPerMs: number;
}

interface State {
  phase: number;
}

export function LikelihoodFieldExplorer() {
  const [useField, setUseField] = useState(true);
  const [sigmaHit, setSigmaHit] = useState(0.16);
  const [clutter, setClutter] = useState(false);

  const world = useMemo(() => withClutter(9, clutter ? 26 : 0), [clutter]);
  const angles = useMemo(() => beamAngles(SCAN_PARAMS), []);
  const scan = useMemo(() => simulateScan(world, TRUTH, SCAN_PARAMS, new Rng(4)), [world]);
  const field = useMemo(() => new LikelihoodField(world, CELL), [world]);

  const params: BeamParams = useMemo(
    () => ({ ...DEFAULT_BEAM_PARAMS, maxRange: MAX_RANGE, sigmaHit }),
    [sigmaHit],
  );

  /** How far the exact transform and the library's chamfer sweep disagree. */
  const edtGap = useMemo(() => {
    const exact = exactDistanceField(world, CELL);
    let max = 0;
    let sum = 0;
    let n = 0;
    for (let j = 0; j < field.ny; j += 2) {
      for (let i = 0; i < field.nx; i += 2) {
        const [x, y] = field.cellCenter(i, j);
        const d = Math.abs(field.data[j * field.nx + i] - distanceAt(exact, x, y));
        max = Math.max(max, d);
        sum += d;
        n++;
      }
    }
    return { max, mean: sum / n };
  }, [world, field]);

  const surface: Surface = useMemo(() => {
    const score = (pose: Pose2) =>
      useField
        ? logLikelihoodFieldRangeFinderModel(scan, pose, field, params, angles)
        : logBeamRangeFinderModel(scan, pose, world, params, angles);

    const grid = new Float64Array(GX * GY);
    let lo = Infinity;
    let hi = -Infinity;
    const t0 = performance.now();
    for (let j = 0; j < GY; j++) {
      const y = TRUTH.y - WIN + (2 * WIN * j) / (GY - 1);
      for (let i = 0; i < GX; i++) {
        const x = TRUTH.x - WIN + (2 * WIN * i) / (GX - 1);
        const v = score({ x, y, theta: TRUTH.theta });
        grid[j * GX + i] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const ms = performance.now() - t0;

    const profileX: number[] = [];
    const profile: number[] = [];
    for (let i = 0; i < PROFILE_N; i++) {
      const x = TRUTH.x - WIN + (2 * WIN * i) / (PROFILE_N - 1);
      profileX.push(x);
      profile.push(score({ x, y: TRUTH.y, theta: TRUTH.theta }));
    }
    let maxStep = 0;
    for (let i = 1; i < profile.length; i++) {
      maxStep = Math.max(maxStep, Math.abs(profile[i] - profile[i - 1]));
    }

    return { grid, lo, hi, profile, profileX, maxStep, evalsPerMs: (GX * GY) / Math.max(ms, 1e-6) };
  }, [useField, scan, field, params, angles, world]);

  // The animation: a hypothesis sliding along the profile line, so the reader
  // sees *why* the score falls — the endpoints slide off the walls.
  const init = useCallback((): State => ({ phase: 0 }), []);
  const step = useCallback((s: State): State => ({ phase: s.phase + 0.045 }), []);
  const sim = useSimulation<State>({ init, step, fps: 24, initialSeed: 1 });

  const hypothesis: Pose2 = useMemo(() => {
    const u = Math.sin(sim.state.phase);
    return { x: TRUTH.x + u * WIN * 0.85, y: TRUTH.y, theta: TRUTH.theta };
  }, [sim.state.phase]);

  const hypScore = useMemo(() => {
    const i = Math.round(
      ((hypothesis.x - (TRUTH.x - WIN)) / (2 * WIN)) * (PROFILE_N - 1),
    );
    return surface.profile[Math.min(PROFILE_N - 1, Math.max(0, i))];
  }, [hypothesis, surface]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, pal: Palette) => {
      clear(ctx, v, pal);
      const { grid, lo, hi, profile, profileX } = surface;

      // ---- likelihood surface as a heat map ---------------------------
      const cw = (2 * WIN) / (GX - 1);
      const ch = (2 * WIN) / (GY - 1);
      const wpx = Math.ceil(sl(v, cw)) + 1;
      const hpx = Math.ceil(sl(v, ch)) + 1;
      // Everything more than DYN nats below the peak is indistinguishable
      // rubbish; spending ink on it would hide the structure near the mode, and
      // clamping keeps the two models on the same visual scale.
      const DYN = 45;
      const floor = Math.max(lo, hi - DYN);
      const span = Math.max(hi - floor, 1e-9);
      for (let j = 0; j < GY; j++) {
        const y = TRUTH.y - WIN + ch * j;
        for (let i = 0; i < GX; i++) {
          const x = TRUTH.x - WIN + cw * i;
          const t = Math.max(0, (grid[j * GX + i] - floor) / span);
          const shaped = Math.pow(t, 2.4);
          if (shaped < 0.02) continue;
          ctx.globalAlpha = 0.06 + 0.82 * shaped;
          ctx.fillStyle = pal.measurement;
          ctx.fillRect(sx(v, x - cw / 2), sy(v, y + ch / 2), wpx, hpx);
        }
      }
      ctx.globalAlpha = 1;

      // ---- the map on top ---------------------------------------------
      drawSegments(ctx, v, world.walls, pal.wall, 2);

      // ---- profile line and its curve ---------------------------------
      ctx.strokeStyle = pal.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, TRUTH.x - WIN), sy(v, TRUTH.y));
      ctx.lineTo(sx(v, TRUTH.x + WIN), sy(v, TRUTH.y));
      ctx.stroke();

      const pLo = Math.min(...profile);
      const pHi = Math.max(...profile);
      const pSpan = Math.max(pHi - pLo, 1e-9);
      const curveBase = TRUTH.y - WIN + 0.12;
      const curveH = WIN * 0.62;
      ctx.strokeStyle = pal.measurement;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < profile.length; i++) {
        const yy = curveBase + ((profile[i] - pLo) / pSpan) * curveH;
        if (i === 0) ctx.moveTo(sx(v, profileX[i]), sy(v, yy));
        else ctx.lineTo(sx(v, profileX[i]), sy(v, yy));
      }
      ctx.stroke();
      label(
        ctx,
        useField ? 'log p(z | x, m) — likelihood field' : 'log p(z | x, m) — beam model',
        sx(v, TRUTH.x - WIN + 0.06),
        sy(v, curveBase + curveH) - 10,
        pal.measurement,
        { size: 10, weight: 600 },
      );

      // ---- the hypothesis and its projected endpoints ------------------
      ctx.fillStyle = pal.measurement;
      for (let k = 0; k < scan.length; k++) {
        if (scan[k] >= MAX_RANGE - 1e-9) continue;
        const a = hypothesis.theta + angles[k];
        const ex = hypothesis.x + scan[k] * Math.cos(a);
        const ey = hypothesis.y + scan[k] * Math.sin(a);
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.arc(sx(v, ex), sy(v, ey), 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      drawRobot(ctx, v, TRUTH, pal.truth, 0.24, { filled: false });
      drawRobot(ctx, v, hypothesis, pal.posterior, 0.2);

      // The marker on the curve for where the hypothesis currently sits.
      const yy = curveBase + ((hypScore - pLo) / pSpan) * curveH;
      ctx.fillStyle = pal.posterior;
      ctx.beginPath();
      ctx.arc(sx(v, hypothesis.x), sy(v, yy), 3.5, 0, Math.PI * 2);
      ctx.fill();
    },
    [surface, world, scan, angles, hypothesis, hypScore, useField],
  );

  return (
    <WidgetFrame
      id="w10.2"
      title="The Likelihood Field Explorer"
      teaches="The physically faithful model is not the one you want inside a filter. Smoothness beats fidelity when something has to search."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          A single 40-beam scan taken at the gray robot, scored at every pose in a 3.4 m window
          (heading held fixed). Green is high likelihood. The purple robot slides along the middle
          line and its scan endpoints slide with it; the curve is the same score plotted as a
          profile. Switch to the <strong>beam model</strong> and the surface breaks into ridges and
          the profile grows cliffs — those are the poses where a beam slips past a door edge and its
          predicted range jumps by metres. Now turn on <strong>clutter</strong>: the beam model gets
          rougher <em>and</em> slower, because every evaluation ray-casts against every chair leg,
          while the field does not notice. That is the whole argument for the field, in two numbers.
        </>
      }
    >
      <SimCanvas
        world={{
          minX: TRUTH.x - WIN,
          maxX: TRUTH.x + WIN,
          minY: TRUTH.y - WIN,
          maxY: TRUTH.y + WIN,
        }}
        draw={draw}
        deps={[sim.tick, surface, hypothesis]}
        aspect={1.55}
        padding={0.02}
        ariaLabel="A heat map of the measurement likelihood over robot positions. With the likelihood field the bright region is a smooth blob; with the beam model it breaks into jagged ridges."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="model" value={useField ? 'field' : 'beam'} />
        <Stat label="evals / ms" value={surface.evalsPerMs.toFixed(0)} />
        <Stat label="largest step" value={`${surface.maxStep.toFixed(2)} nats`} />
        <Stat label="chamfer vs exact EDT" value={`${(edtGap.mean * 100).toFixed(1)} cm`} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="σ_hit"
          role="measurement"
          value={sigmaHit}
          min={0.04}
          max={0.6}
          step={0.01}
          unit="m"
          onChange={setSigmaHit}
          help="Blurs the field. Too small and the surface is a needle no search can find; too large and every pose looks the same."
        />
        <Toggle label="Likelihood field" role="measurement" checked={useField} onChange={setUseField} />
        <Toggle label="Clutter (chair legs)" checked={clutter} onChange={setClutter} />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        seed={sim.seed}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
