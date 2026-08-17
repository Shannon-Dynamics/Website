'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { Tsdf2 } from '@/lib/mapping/tsdf';
import { Esdf2, esdfFromTsdf, maxDistance } from '@/lib/mapping/esdf';
import { simulateScan, type Segment } from '@/lib/sim/world';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { LAP, SCAN_ANGLES, SCAN_PARAMS, SOLID_APARTMENT, tourPose } from './tour';

/**
 * w19.3 — Wavefront.
 *
 * The same field, asked two different questions. The TSDF knows the distance to
 * the surface only inside its truncation band and only along the beams that
 * measured it; the ESDF knows the true Euclidean distance everywhere, because a
 * two-pass distance transform propagated it. Watching the wavefront sweep is
 * watching that propagation; watching the probe climb is watching the gradient
 * Chapter 20's planners and Chapter 23's MPPI will actually consume.
 */

const RES = 0.1;
const TAU = 0.3;
const BOUNDS = SOLID_APARTMENT.bounds;
/** Scans folded in before the widget starts: enough to close the apartment. */
const PREFUSE = 64;
/** How far the isoline sweep advances per tick, in metres. */
const WAVE_STEP = 0.06;
/** Isolines every this many metres. */
const BAND = 0.25;

interface Probe {
  x: number;
  y: number;
  trail: [number, number][];
  /** Clearance at the current point, metres. */
  clearance: number;
  /** Best clearance reached so far, and how many steps have failed to beat it. */
  best: number;
  stalled: number;
  settled: boolean;
  /** Settled because the field is flat, not because it reached a ridge. */
  frozen: boolean;
}

interface State {
  tsdf: Tsdf2;
  esdf: Esdf2;
  contour: Segment[];
  dMax: number;
  wave: number;
  probe: Probe | null;
}

/** Where the default probe is dropped: the tight south-west corner of room A. */
const START: [number, number] = [0.7, 0.7];

export function Wavefront() {
  const [euclidean, setEuclidean] = useState(true);
  const modeRef = useRef(euclidean);
  modeRef.current = euclidean;
  const dropRef = useRef<[number, number] | null>(null);

  const init = useCallback((seed: number): State => {
    // One short lap, fused once. This widget is about what happens *after* the
    // map is built, so the map is built up front rather than animated again.
    const tsdf = Tsdf2.forBounds(BOUNDS, RES, TAU, 40);
    const rng = new Rng(seed);
    for (let k = 0; k < PREFUSE; k++) {
      const pose = tourPose(Math.round((k * LAP) / PREFUSE));
      const ranges = simulateScan(SOLID_APARTMENT, pose, SCAN_PARAMS, rng);
      tsdf.integrateScan(pose, ranges, SCAN_ANGLES, {
        maxRange: SCAN_PARAMS.maxRange,
        carveFreeSpace: true,
        carveWeight: 0.25,
      });
    }
    const esdf = esdfFromTsdf(tsdf);
    return {
      tsdf,
      esdf,
      contour: tsdf.surface(),
      dMax: Math.min(maxDistance(esdf), 3),
      wave: 0,
      probe: null,
    };
  }, []);

  /** The field the reader has selected, sampled and differentiated the same way. */
  const sample = useCallback((s: State, x: number, y: number): number => {
    return modeRef.current ? s.esdf.distance(x, y) : s.tsdf.value(x, y);
  }, []);

  const grad = useCallback(
    (s: State, x: number, y: number): [number, number] => {
      const h = RES;
      return [
        (sample(s, x + h, y) - sample(s, x - h, y)) / (2 * h),
        (sample(s, x, y + h) - sample(s, x, y - h)) / (2 * h),
      ];
    },
    [sample],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      const wave = Math.min(s.wave + WAVE_STEP, s.dMax + BAND);

      // A drop request arrives from the pointer handler; the simulation owns the
      // integration so a click never mutates state outside the step function.
      let probe = s.probe;
      const dropped = dropRef.current;
      const drop = (at: [number, number]): Probe => ({
        x: at[0],
        y: at[1],
        trail: [at],
        clearance: sample(s, at[0], at[1]),
        best: sample(s, at[0], at[1]),
        stalled: 0,
        settled: false,
        frozen: false,
      });

      if (dropped) {
        dropRef.current = null;
        probe = drop(dropped);
      } else if (probe === null && tick === Math.round(1.6 / WAVE_STEP)) {
        // Autoplay must teach without a click: drop one in the corner.
        probe = drop(START);
      }

      if (probe && !probe.settled) {
        // Gradient *ascent* on clearance: the planner's obstacle term is −d, so
        // rolling downhill in cost means climbing the distance field.
        const [gx, gy] = grad(s, probe.x, probe.y);
        const n = Math.hypot(gx, gy);
        if (n < 0.02) {
          // A flat field has no direction to offer. On the raw TSDF, that is
          // every point more than τ from a wall.
          probe = { ...probe, settled: true, frozen: true };
        } else {
          const x = Math.min(BOUNDS.maxX - 0.05, Math.max(BOUNDS.minX + 0.05, probe.x + 0.05 * (gx / n)));
          const y = Math.min(BOUNDS.maxY - 0.05, Math.max(BOUNDS.minY + 0.05, probe.y + 0.05 * (gy / n)));
          const clearance = sample(s, x, y);
          const trail = [...probe.trail, [x, y] as [number, number]].slice(-400);
          // A bilinear field is bumpy, and a fixed step size dithers around the
          // ridge instead of stopping on it. Settle when six steps in a row fail
          // to beat the best clearance seen — that ridge is the medial axis.
          const improved = clearance > probe.best + 1e-3;
          const stalled = improved ? 0 : probe.stalled + 1;
          probe = {
            x,
            y,
            trail,
            clearance,
            best: Math.max(probe.best, clearance),
            stalled,
            settled: stalled >= 6,
            frozen: false,
          };
        }
      }

      return { ...s, wave, probe };
    },
    [grad, sample],
  );

  const sim = useSimulation<State>({ init, step, fps: 24, initialSeed: 19, maxTicks: 400, loop: true });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;
      const field = euclidean ? s.esdf : s.tsdf;
      const cell = Math.ceil(sl(v, RES)) + 1;
      const reached = s.wave;

      // ---- the field, revealed as the transform sweeps outward -------------
      for (let j = 0; j < field.height; j++) {
        for (let i = 0; i < field.width; i++) {
          const d = euclidean ? s.esdf.d[s.esdf.index(i, j)] : s.tsdf.d[s.tsdf.index(i, j)];
          const a = Math.abs(d);
          if (a > reached) continue; // not yet swept: the wavefront has not arrived
          const mag = Math.min(a / s.dMax, 1);
          // Blue for free space, orange behind a surface — the same two roles the
          // TSDF used in w19.1, so the reader can compare the two fields directly.
          ctx.fillStyle = d < 0 ? p.prediction : p.prior;
          ctx.globalAlpha = 0.5 * (1 - mag) + 0.08;
          const [cx, cy] = euclidean
            ? [s.esdf.origin.x + (i + 0.5) * RES, s.esdf.origin.y + (j + 0.5) * RES]
            : s.tsdf.cellCenter(i, j);
          ctx.fillRect(sx(v, cx - RES / 2), sy(v, cy + RES / 2), cell, cell);

          // Isolines: a cell whose distance sits within half a cell of a multiple
          // of BAND is on a contour ring.
          const near = Math.abs(a - Math.round(a / BAND) * BAND);
          if (near < RES * 0.35 && a > 0.02) {
            ctx.globalAlpha = Math.abs(a - reached) < 0.12 ? 0.95 : 0.3;
            ctx.fillStyle = p.prior;
            ctx.fillRect(sx(v, cx - RES / 2), sy(v, cy + RES / 2), cell, cell);
          }
        }
      }
      ctx.globalAlpha = 1;

      // ---- ground truth ---------------------------------------------------
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.setLineDash([4, 4]);
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const w of SOLID_APARTMENT.walls) {
        ctx.moveTo(sx(v, w.x1), sy(v, w.y1));
        ctx.lineTo(sx(v, w.x2), sy(v, w.y2));
      }
      ctx.stroke();
      ctx.restore();

      // ---- the extracted surface, the seed set of the transform -------------
      ctx.save();
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (const seg of s.contour) {
        ctx.moveTo(sx(v, seg.x1), sy(v, seg.y1));
        ctx.lineTo(sx(v, seg.x2), sy(v, seg.y2));
      }
      ctx.stroke();
      ctx.restore();

      // ---- gradient quiver, faded in once the sweep is done -----------------
      const quiverAlpha = Math.min(Math.max((s.wave - 0.8 * s.dMax) / (0.4 * s.dMax), 0), 1);
      if (quiverAlpha > 0.02) {
        ctx.save();
        ctx.strokeStyle = p.prediction;
        ctx.globalAlpha = 0.75 * quiverAlpha;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let y = BOUNDS.minY + 0.5; y < BOUNDS.maxY; y += 0.55) {
          for (let x = BOUNDS.minX + 0.5; x < BOUNDS.maxX; x += 0.55) {
            const [gx, gy] = grad(s, x, y);
            const n = Math.hypot(gx, gy);
            if (n < 0.15) continue; // flat field: no direction to draw
            const len = 0.24;
            const ex = x + (gx / n) * len;
            const ey = y + (gy / n) * len;
            ctx.moveTo(sx(v, x), sy(v, y));
            ctx.lineTo(sx(v, ex), sy(v, ey));
            // arrowhead
            const ang = Math.atan2(gy, gx);
            for (const off of [2.6, -2.6]) {
              ctx.moveTo(sx(v, ex), sy(v, ey));
              ctx.lineTo(sx(v, ex + 0.08 * Math.cos(ang + off)), sy(v, ey + 0.08 * Math.sin(ang + off)));
            }
          }
        }
        ctx.stroke();
        ctx.restore();
      }

      // ---- the probe --------------------------------------------------------
      const probe = s.probe;
      if (probe) {
        ctx.save();
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx(v, probe.trail[0][0]), sy(v, probe.trail[0][1]));
        for (const [tx, ty] of probe.trail) ctx.lineTo(sx(v, tx), sy(v, ty));
        ctx.stroke();

        // The clearance disc: the probe's own answer to "how far to anything?"
        ctx.globalAlpha = 0.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(sx(v, probe.x), sy(v, probe.y), Math.max(sl(v, Math.abs(probe.clearance)), 1), 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;

        ctx.fillStyle = p.posterior;
        ctx.beginPath();
        ctx.arc(sx(v, probe.x), sy(v, probe.y), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        label(
          ctx,
          probe.frozen
            ? `frozen: ‖∇d‖ ≈ 0, nothing to follow`
            : probe.settled
              ? `on the medial axis: clearance ${probe.clearance.toFixed(2)} m`
              : `climbing: clearance ${probe.clearance.toFixed(2)} m`,
          sx(v, probe.x) + 9,
          sy(v, probe.y) - 11,
          p.posterior,
          { size: 10 },
        );
      }

      label(
        ctx,
        euclidean
          ? `ESDF — wavefront at ${s.wave.toFixed(2)} m`
          : `raw TSDF — everything past τ = ${TAU.toFixed(2)} m is flat`,
        sx(v, BOUNDS.minX + 0.15),
        sy(v, BOUNDS.maxY - 0.3),
        euclidean ? p.prior : p.prediction,
        { size: 11, weight: 600 },
      );
    },
    [sim.state, euclidean, grad],
  );

  const stats = useMemo(() => {
    const s = sim.state;
    // The eikonal check: away from the medial axis a true distance field has
    // ‖∇d‖ = 1 exactly. The median over a probe grid is the honest summary —
    // the mean is dragged down by medial-axis points, where the gradient really
    // does vanish. A truncated field fails the check almost everywhere.
    const mags: number[] = [];
    for (let y = 0.6; y < BOUNDS.maxY - 0.6; y += 0.4) {
      for (let x = 0.6; x < BOUNDS.maxX - 0.6; x += 0.4) {
        const [gx, gy] = grad(s, x, y);
        mags.push(Math.hypot(gx, gy));
      }
    }
    mags.sort((a, b) => a - b);
    return {
      eikonal: mags.length > 0 ? mags[Math.floor(mags.length / 2)] : 0,
      reach: euclidean ? s.dMax : TAU,
      clearance: s.probe?.clearance ?? 0,
    };
  }, [sim.state, grad, euclidean]);

  return (
    <WidgetFrame
      id="w19.3"
      title="Wavefront"
      teaches="A TSDF is not a distance field with a smaller number in it. Past the truncation band it is flat, and flat fields have no gradients to plan with."
      colorKey={['prior', 'prediction', 'posterior', 'truth']}
      caption={
        <>
          The purple contour is the surface marching squares extracted; the blue rings are the
          distance transform propagating outward from it, one isoline every 25 cm. Once the sweep
          finishes, the orange arrows show <em>∇d</em> — the direction of increasing clearance — and
          a probe is dropped in the tight corner of the south-west room, where it climbs to the
          middle of the floor and stops on the medial axis, the safest point it can reach. Now switch
          the toggle off. The raw TSDF is identical near the walls and <em>completely flat</em>
          beyond <em>τ</em> = 30 cm: the rings stop, the arrows vanish, the median ‖∇d‖ collapses,
          and the probe cannot move at all. Click anywhere to drop your own probe.
        </>
      }
    >
      <SimCanvas
        world={BOUNDS}
        draw={draw}
        deps={[sim.tick, sim.state, euclidean]}
        aspect={12 / 9}
        padding={0.2}
        cursor="crosshair"
        onPointer={(world, phase) => {
          if (phase === 'down') dropRef.current = [world[0], world[1]];
        }}
        ariaLabel="The apartment's distance field: concentric rings expanding away from the reconstructed walls, with arrows pointing toward increasing clearance and a probe climbing to the middle of a corridor."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="mean ‖∇d‖" value={stats.eikonal.toFixed(2)} />
        <Stat label="field reaches" value={`${stats.reach.toFixed(2)} m`} />
        <Stat label="probe clearance" value={`${stats.clearance.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={2}>
        <Toggle
          label="True Euclidean distance (ESDF)"
          role="prior"
          checked={euclidean}
          onChange={(v) => {
            setEuclidean(v);
            // Re-drop the probe so the switch answers the question immediately:
            // on the ESDF it climbs, on the TSDF it cannot move at all.
            dropRef.current = [START[0], START[1]];
          }}
        />
        <ButtonRow>
          <ActionButton
            emphasis
            onClick={() => {
              dropRef.current = [START[0], START[1]];
            }}
          >
            Drop a probe in the doorway
          </ActionButton>
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

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
