'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { Tsdf2, type FusionEvent } from '@/lib/mapping/tsdf';
import { simulateScan, type Segment } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import { clear, drawRobot, drawScan, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import {
  CHAIR_HALF,
  CHAIR_SPOTS,
  LAP,
  SCAN_ANGLES,
  SCAN_PARAMS,
  SOLID_APARTMENT,
  boxSegments,
  tourPose,
  worldWithChair,
} from './tour';

/**
 * w19.1 — the TSDF Sculptor.
 *
 * Rusty drives a lap; every scan fuses a projective distance into the cells of
 * a 2-D truncated signed distance field. The inspector under the map opens one
 * cell and shows the three numbers the update actually uses — prior, incoming
 * sample, posterior — which is the whole of Derivation 2 made literal.
 */

const RES = 0.1;
const TAU = 0.15;
const BOUNDS = SOLID_APARTMENT.bounds;

interface Params {
  wMax: number;
}

interface State {
  tsdf: Tsdf2;
  rng: Rng;
  pose: Pose2;
  ranges: number[];
  contour: Segment[];
  chair: number;
  chairMovedAt: number | null;
  log: FusionEvent[];
  scans: number;
}

export function TsdfSculptor() {
  const [params, setParams] = useState<Params>({ wMax: 24 });
  const [probe, setProbe] = useState<[number, number]>([5.0, 4.7]);
  const probeRef = useRef(probe);
  probeRef.current = probe;
  const chairRef = useRef(0);
  const wMaxRef = useRef(params.wMax);
  wMaxRef.current = params.wMax;

  const init = useCallback((seed: number): State => {
    chairRef.current = 0;
    const tsdf = Tsdf2.forBounds(BOUNDS, RES, TAU, wMaxRef.current);
    return {
      tsdf,
      rng: new Rng(seed),
      pose: tourPose(0),
      ranges: [],
      contour: [],
      chair: 0,
      chairMovedAt: null,
      log: [],
      scans: 0,
    };
  }, []);

  const step = useCallback((s: State, tick: number): State => {
    const { tsdf, rng } = s;

    // The weight clamp is live: lowering it must also forget the evidence the
    // field has already banked, or the slider would only affect the future.
    if (tsdf.wMax !== wMaxRef.current) {
      tsdf.wMax = wMaxRef.current;
      for (let k = 0; k < tsdf.w.length; k++) {
        if (tsdf.w[k] > tsdf.wMax) tsdf.w[k] = tsdf.wMax;
      }
    }

    const [px, py] = probeRef.current;
    const [pi, pj] = tsdf.worldToCell(px, py);
    const idx = tsdf.inBounds(pi, pj) ? tsdf.index(pi, pj) : null;
    if (tsdf.probe !== idx) {
      tsdf.probe = idx;
      tsdf.probeLog = [];
    }

    const chair = chairRef.current;
    const pose = tourPose(tick + 1);
    const world = worldWithChair(chair);
    const ranges = simulateScan(world, pose, SCAN_PARAMS, rng);
    tsdf.integrateScan(pose, ranges, SCAN_ANGLES, {
      maxRange: SCAN_PARAMS.maxRange,
      carveFreeSpace: true,
      carveWeight: 0.25,
    });

    // Marching squares is cheap but not free; the contour only needs to be
    // fresh, not instantaneous.
    const contour = tick % 4 === 0 ? tsdf.surface() : s.contour;

    return {
      ...s,
      pose,
      ranges,
      contour,
      chair,
      chairMovedAt: chair !== s.chair ? tick : s.chairMovedAt,
      log: [...tsdf.probeLog],
      scans: s.scans + 1,
    };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 14, initialSeed: 19, maxTicks: LAP, loop: true });
  const { setState } = sim;

  const moveChair = useCallback(() => {
    chairRef.current = (chairRef.current + 1) % CHAIR_SPOTS.length;
    setState((s) => ({ ...s, chair: chairRef.current }));
  }, [setState]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { tsdf, contour, pose, ranges, chair } = sim.state;
      const cell = Math.ceil(sl(v, RES)) + 1;

      // ---- the field ---------------------------------------------------
      // A diverging ramp, and the one place this book uses one: orange for
      // negative (behind the surface), blue for positive (free), transparent at
      // the zero crossing. Opacity also carries the weight, so a cell seen once
      // looks tentative next to one seen forty times.
      for (let j = 0; j < tsdf.height; j++) {
        for (let i = 0; i < tsdf.width; i++) {
          const k = tsdf.index(i, j);
          const w = tsdf.w[k];
          if (w <= 0) continue;
          const d = tsdf.d[k];
          const mag = Math.min(Math.abs(d) / TAU, 1);
          const conf = Math.min(w / 6, 1);
          ctx.globalAlpha = 0.1 + 0.62 * mag * conf;
          ctx.fillStyle = d < 0 ? p.prediction : p.prior;
          const [cx, cy] = tsdf.cellCenter(i, j);
          ctx.fillRect(sx(v, cx - RES / 2), sy(v, cy + RES / 2), cell, cell);
        }
      }
      ctx.globalAlpha = 1;

      // ---- ground truth, drawn faintly underneath the estimate ----------
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      for (const s of SOLID_APARTMENT.walls) {
        ctx.moveTo(sx(v, s.x1), sy(v, s.y1));
        ctx.lineTo(sx(v, s.x2), sy(v, s.y2));
      }
      const [ccx, ccy] = CHAIR_SPOTS[chair];
      for (const s of boxSegments(ccx, ccy, CHAIR_HALF)) {
        ctx.moveTo(sx(v, s.x1), sy(v, s.y1));
        ctx.lineTo(sx(v, s.x2), sy(v, s.y2));
      }
      ctx.stroke();
      ctx.restore();

      // ---- the extracted zero level set ---------------------------------
      ctx.save();
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (const s of contour) {
        ctx.moveTo(sx(v, s.x1), sy(v, s.y1));
        ctx.lineTo(sx(v, s.x2), sy(v, s.y2));
      }
      ctx.stroke();
      ctx.restore();

      if (ranges.length > 0) {
        drawScan(ctx, v, pose, ranges, SCAN_ANGLES, p.measurement, SCAN_PARAMS.maxRange);
      }
      drawRobot(ctx, v, pose, p.truth, 0.24);

      // ---- the inspected cell -------------------------------------------
      const [pxw, pyw] = probeRef.current;
      const [pi, pj] = tsdf.worldToCell(pxw, pyw);
      if (tsdf.inBounds(pi, pj)) {
        const [cx, cy] = tsdf.cellCenter(pi, pj);
        ctx.save();
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx(v, cx - RES / 2), sy(v, cy + RES / 2), cell + 2, cell + 2);
        ctx.beginPath();
        ctx.arc(sx(v, cx), sy(v, cy), sl(v, 0.34), 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
        label(ctx, 'inspected cell', sx(v, cx) + 14, sy(v, cy) - 14, p.accent, { size: 10 });
      }

      label(ctx, `${sim.state.scans} scans fused`, sx(v, BOUNDS.minX + 0.15), sy(v, BOUNDS.maxY - 0.3), p.truth, {
        size: 11,
      });
    },
    [sim.state],
  );

  const last = sim.state.log[sim.state.log.length - 1];
  const [pi, pj] = sim.state.tsdf.worldToCell(probe[0], probe[1]);
  const [dNow, wNow] = sim.state.tsdf.voxel(pi, pj);

  const stats = useMemo(
    () => ({
      segments: sim.state.contour.length,
      observed: sim.state.tsdf.observedCells(),
      bytes: sim.state.tsdf.memoryBytes(),
    }),
    [sim.state],
  );

  return (
    <WidgetFrame
      id="w19.1"
      title="TSDF Sculptor"
      teaches="TSDF fusion is not a graphics trick: every cell runs a one-dimensional Kalman filter, and W_max is its memory."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Blue is positive distance (free space in front of a surface), orange is negative (behind
          it), and the purple line is the zero level set that marching squares pulls out of the
          numbers. Click any cell to inspect it: the strip below shows the prior, the projective
          sample the newest scan delivered, and the fused posterior, with the gain
          <em> K = w / (W + w)</em> that produced it. Now press <strong>move the chair</strong>. At
          W<sub>max</sub> = 64 its ghost lingers for hundreds of scans; drag the slider down to 4
          and the map forgets it almost immediately — and gets visibly noisier everywhere else.
          That trade is the same one Chapter 6 called the noise ratio.
        </>
      }
    >
      <SimCanvas
        world={BOUNDS}
        draw={draw}
        deps={[sim.tick, sim.state, probe]}
        aspect={12 / 9}
        padding={0.2}
        cursor="crosshair"
        onPointer={(world, phase) => {
          if (phase === 'down') setProbe([world[0], world[1]]);
        }}
        ariaLabel="A floor plan of the apartment being carved into a signed distance field: blue free space, orange interiors, and a purple contour marking the reconstructed walls."
      />

      <VoxelInspector d={dNow} w={wNow} wMax={params.wMax} tau={TAU} event={last} />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="contour segments" value={String(stats.segments)} />
        <Stat label="cells observed" value={stats.observed.toLocaleString()} />
        <Stat label="D + W storage" value={`${(stats.bytes / 1024).toFixed(0)} kB`} />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Weight clamp W_max"
          role="posterior"
          value={params.wMax}
          min={2}
          max={64}
          step={1}
          onChange={(v) => setParams({ wMax: v })}
          help="How much evidence a cell is allowed to bank. Large = stable and stubborn; small = responsive and noisy."
        />
        <ButtonRow>
          <ActionButton onClick={moveChair} emphasis>
            Move the chair
          </ActionButton>
          <ActionButton onClick={() => setProbe([5.0, 4.7])}>Inspect the chair cell</ActionButton>
        </ButtonRow>
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

/**
 * The three-term update on one number line. Prior blue, incoming sample green,
 * posterior purple — the same colors the equations use.
 */
function VoxelInspector({
  d,
  w,
  wMax,
  tau,
  event,
}: {
  d: number;
  w: number;
  wMax: number;
  tau: number;
  event?: FusionEvent;
}) {
  const toX = (v: number) => 6 + ((v + tau) / (2 * tau)) * 88;
  const clampX = (v: number) => Math.max(4, Math.min(98, toX(v)));

  return (
    <div className="grid gap-3 border-t border-fd-border px-3 py-3 sm:grid-cols-[1.4fr_1fr]">
      <div>
        <p className="eyebrow mb-1">the inspected cell&rsquo;s filter</p>
        <svg viewBox="0 0 104 30" className="w-full" role="img" aria-label="Number line from minus tau to plus tau showing the prior, the incoming observation, and the fused posterior for the inspected cell.">
          <line x1={6} y1={18} x2={94} y2={18} stroke="var(--pr-grid)" strokeWidth={0.8} />
          <line x1={50} y1={13} x2={50} y2={23} stroke="var(--pr-truth)" strokeWidth={0.6} />
          <text x={6} y={28} style={{ fontSize: 3.4 }} fill="var(--pr-truth)">
            −τ
          </text>
          <text x={47} y={28} style={{ fontSize: 3.4 }} fill="var(--pr-truth)">
            0
          </text>
          <text x={90} y={28} style={{ fontSize: 3.4 }} fill="var(--pr-truth)">
            +τ
          </text>

          {event ? (
            <>
              <circle cx={clampX(event.before)} cy={18} r={2.2} fill="var(--pr-prior)" />
              <text x={clampX(event.before)} y={11} textAnchor="middle" style={{ fontSize: 3.2 }} fill="var(--pr-prior)">
                prior
              </text>
              <circle cx={clampX(event.observation)} cy={18} r={2.2} fill="var(--pr-measurement)" />
              <text
                x={clampX(event.observation)}
                y={7}
                textAnchor="middle"
                style={{ fontSize: 3.2 }}
                fill="var(--pr-measurement)"
              >
                scan d
              </text>
              <circle cx={clampX(event.after)} cy={18} r={2.6} fill="var(--pr-posterior)" />
              <text
                x={clampX(event.after)}
                y={25}
                textAnchor="middle"
                style={{ fontSize: 3.2 }}
                fill="var(--pr-posterior)"
              >
                posterior
              </text>
            </>
          ) : (
            <>
              <circle cx={clampX(d)} cy={18} r={2.4} fill="var(--pr-posterior)" />
              <text x={52} y={7} style={{ fontSize: 3.2 }} fill="var(--pr-truth)">
                waiting for a scan to touch this cell
              </text>
            </>
          )}
        </svg>
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[0.72rem] tabular-nums">
        <dt className="text-fd-muted-foreground">D</dt>
        <dd className="text-right">{d.toFixed(3)} m</dd>
        <dt className="text-fd-muted-foreground">W</dt>
        <dd className="text-right">
          {w.toFixed(1)} / {wMax}
        </dd>
        <dt className="text-fd-muted-foreground">gain K</dt>
        <dd className="text-right">{event ? event.gain.toFixed(3) : '—'}</dd>
        <dt className="text-fd-muted-foreground">updates</dt>
        <dd className="text-right">{event ? 'live' : 'idle'}</dd>
      </dl>
    </div>
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
