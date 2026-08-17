'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { setValue, useExplorable } from '@/lib/explorable/store';
import { HistogramFilter1D, gaussianKernel } from '@/lib/filters/bayes';
import { HALLWAY_1D } from '@/lib/sim/world';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w5.1 — the Hallway Belief Machine.
 *
 * The whole Bayes filter, visible at once: the robot drives down a corridor
 * with three identical doors, and the belief over its position is drawn as a
 * histogram beneath it. Sensing multiplies (and sharpens); moving convolves
 * (and smears). Everything here runs the real HistogramFilter1D from the
 * library, so the curve on screen is literally Table 2.1 executing.
 */

const CELLS = 120;
const { length: L, doors: DOORS, doorWidth: DOOR_W } = HALLWAY_1D;

interface Params {
  sensorNoise: number;
  motionNoise: number;
  stepSize: number;
  sensing: boolean;
  moving: boolean;
}

interface Phase {
  kind: 'prior' | 'predicted' | 'corrected';
  /** Snapshot taken before the most recent operation, for the ghost curve. */
  before: number[];
  /** Per-cell measurement likelihood from the last correction. */
  likelihood: number[] | null;
  sawDoor: boolean;
}

interface State {
  filter: HistogramFilter1D;
  rng: Rng;
  truth: number;
  belief: number[];
  phase: Phase;
  history: { entropy: number; error: number }[];
}

/** Is position `x` in front of a door? */
const atDoor = (x: number) => DOORS.some((d) => Math.abs(x - d) < DOOR_W / 2);

export function HallwayBeliefMachine() {
  const [params, setParams] = useState<Params>({
    sensorNoise: 0.18,
    motionNoise: 0.12,
    stepSize: 0.35,
    sensing: true,
    moving: true,
  });

  // The prose above this figure carries draggable numbers. Whichever the reader
  // last touched wins; until then these fall back to the slider values, so the
  // widget is complete on its own.
  const scrubbedMotion = useExplorable('ch05.motionNoise', params.motionNoise);
  const scrubbedSensor = useExplorable('ch05.sensorNoise', params.sensorNoise);
  const motionNoise = scrubbedMotion;
  const sensorNoise = scrubbedSensor;

  const init = useCallback((seed: number): State => {
    const filter = new HistogramFilter1D({ length: L, cells: CELLS, wrap: true });
    filter.setUniform(); // Global localization: the robot knows nothing.
    return {
      filter,
      rng: new Rng(seed),
      truth: 0.4,
      belief: filter.belief(),
      phase: { kind: 'prior', before: filter.belief(), likelihood: null, sawDoor: false },
      history: [],
    };
  }, []);

  const step = useCallback(
    (s: State, tick: number): State => {
      const { filter, rng } = s;
      // Alternate: move on even ticks, sense on odd ones, so the reader can see
      // each operation act on the belief in isolation.
      const doMove = tick % 2 === 0;
      const before = filter.belief();
      let truth = s.truth;
      let phase: Phase;

      if (doMove && params.moving) {
        // The robot commands a step; the wheels deliver something slightly else.
        truth = (truth + params.stepSize + rng.normal(0, motionNoise) + L) % L;
        filter.predict(params.stepSize, gaussianKernel(Math.max(motionNoise, 1e-3)));
        phase = { kind: 'predicted', before, likelihood: null, sawDoor: false };
      } else if (!doMove && params.sensing) {
        // The door detector: noisy, and — crucially — it cannot tell the three
        // doors apart. That ambiguity is what keeps the belief multi-modal.
        const truthAtDoor = atDoor(truth);
        const detected = rng.next() < (truthAtDoor ? 0.88 : 0.12);
        const pDoor = detected ? 0.88 : 0.12;
        const pNoDoor = 1 - pDoor;
        const likelihoodAt = (x: number) => (atDoor(x) ? pDoor : pNoDoor);

        filter.correct(likelihoodAt);
        phase = {
          kind: 'corrected',
          before,
          likelihood: filter.centers().map(likelihoodAt),
          sawDoor: detected,
        };
      } else {
        phase = s.phase;
      }

      const belief = filter.belief();
      const mean = filter.mean();
      const error = Math.min(Math.abs(mean - truth), L - Math.abs(mean - truth));
      const history = [...s.history, { entropy: filter.entropy(), error }].slice(-120);
      return { filter, rng, truth, belief, phase, history };
    },
    [params, motionNoise],
  );

  const sim = useSimulation<State>({ init, step, fps: 3, initialSeed: 11 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { belief, phase, truth } = sim.state;

      const corridorY = 0.78; // world-y of the corridor strip
      const histTop = 0.62;
      const histBottom = 0.06;

      // ---- corridor ---------------------------------------------------
      const yTop = sy(v, corridorY + 0.09);
      const yBot = sy(v, corridorY - 0.09);
      ctx.fillStyle = p.free;
      ctx.fillRect(sx(v, 0), yTop, sl(v, L), yBot - yTop);
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx(v, 0), yTop, sl(v, L), yBot - yTop);

      // Doors — the three landmarks the robot cannot tell apart.
      for (const d of DOORS) {
        ctx.fillStyle = p.bg;
        ctx.fillRect(sx(v, d - DOOR_W / 2), yTop - 1, sl(v, DOOR_W), yBot - yTop + 2);
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sx(v, d - DOOR_W / 2), yTop);
        ctx.lineTo(sx(v, d - DOOR_W / 2), yBot);
        ctx.moveTo(sx(v, d + DOOR_W / 2), yTop);
        ctx.lineTo(sx(v, d + DOOR_W / 2), yBot);
        ctx.stroke();
      }

      // The true robot — gray, because the robot itself never gets to see this.
      const rx = sx(v, truth);
      const ry = (yTop + yBot) / 2;
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(rx, ry, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(rx + 13, ry);
      ctx.stroke();

      ctx.globalAlpha = 1;

      if (phase.kind === 'corrected') {
        label(
          ctx,
          phase.sawDoor ? 'sensor: DOOR' : 'sensor: no door',
          rx + 18,
          ry,
          p.measurement,
          { size: 10 },
        );
      }

      // ---- histogram --------------------------------------------------
      const centers = sim.state.filter.centers();
      const peak = Math.max(...belief, ...phase.before, 1e-6);
      const barW = sl(v, L / CELLS);
      const baseY = sy(v, histBottom);
      const topY = sy(v, histTop);
      const h = baseY - topY;

      // Ghost of the belief before this operation, so the reader sees the change.
      ctx.fillStyle = phase.kind === 'corrected' ? p.prediction : p.prior;
      ctx.globalAlpha = 0.28;
      for (let i = 0; i < CELLS; i++) {
        const bh = (phase.before[i] / peak) * h;
        ctx.fillRect(sx(v, centers[i]) - barW / 2, baseY - bh, barW, bh);
      }
      ctx.globalAlpha = 1;

      // The current belief.
      const currentColor =
        phase.kind === 'corrected' ? p.posterior : phase.kind === 'predicted' ? p.prediction : p.prior;
      ctx.fillStyle = currentColor;
      for (let i = 0; i < CELLS; i++) {
        const bh = (belief[i] / peak) * h;
        ctx.fillRect(sx(v, centers[i]) - barW / 2, baseY - bh, Math.max(barW - 0.5, 1), bh);
      }

      // The measurement likelihood, drawn as a curve rather than mass: it is a
      // function of state, not a distribution over it.
      if (phase.likelihood) {
        const lMax = Math.max(...phase.likelihood);
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 1.75;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (let i = 0; i < CELLS; i++) {
          const y = baseY - (phase.likelihood[i] / lMax) * h * 0.92;
          if (i === 0) ctx.moveTo(sx(v, centers[i]), y);
          else ctx.lineTo(sx(v, centers[i]), y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Baseline + the operation label. Chrome never dims with the data.
      ctx.globalAlpha = 1;
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), baseY);
      ctx.lineTo(sx(v, L), baseY);
      ctx.stroke();

      const caption =
        phase.kind === 'corrected'
          ? 'CORRECT  ×  measurement  →  posterior'
          : phase.kind === 'predicted'
            ? 'PREDICT  ∗  motion  →  prediction'
            : 'PRIOR';
      label(ctx, caption, sx(v, 0.1), topY - 10, currentColor, { size: 11, weight: 600 });
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const last = sim.state.history[sim.state.history.length - 1];
    return {
      entropy: last?.entropy ?? Math.log2(CELLS),
      error: last?.error ?? 0,
      modes: countModes(sim.state.belief),
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w5.1"
      title="The Hallway Belief Machine"
      teaches="Sensing sharpens the belief; moving smears it. Neither one ever tells you where the robot is."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          The robot starts knowing nothing — a flat belief over the whole corridor. Watch the two
          operations alternate. A door sighting multiplies the belief by the green likelihood,
          producing three peaks, because three doors explain the reading equally well. Then motion
          convolves it and every peak widens. Ambiguity is resolved not by any single reading but by
          the <em>sequence</em>: only one hypothesis stays consistent with door, gap, gap, door.
          Try turning motion noise up past 0.3 and watch the filter lose the plot entirely.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.3, maxX: L + 0.3, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={2.35}
        padding={0}
        ariaLabel="A corridor with three identical doors above a histogram showing the robot's belief about its position. The belief alternately sharpens when the door sensor fires and spreads out when the robot moves."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="entropy" value={`${stats.entropy.toFixed(2)} bits`} />
        <Stat label="modes" value={String(stats.modes)} />
        <Stat label="|mean − truth|" value={`${stats.error.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Sensor reliability"
          role="measurement"
          value={sensorNoise}
          min={0.02}
          max={0.45}
          step={0.01}
          onChange={(v) => {
            setParams((p) => ({ ...p, sensorNoise: v }));
            setValue('ch05.sensorNoise', v);
          }}
          help="Higher means a less trustworthy door detector."
        />
        <Slider
          label="Motion noise σ"
          role="prediction"
          value={motionNoise}
          min={0.01}
          max={0.5}
          step={0.01}
          unit="m"
          onChange={(v) => {
            setParams((p) => ({ ...p, motionNoise: v }));
            setValue('ch05.motionNoise', v);
          }}
          help="How badly the wheels betray the commanded step."
        />
        <Slider
          label="Step size"
          value={params.stepSize}
          min={0.05}
          max={1}
          step={0.05}
          unit="m"
          onChange={(v) => setParams((p) => ({ ...p, stepSize: v }))}
        />
        <Toggle
          label="Sensing enabled"
          role="measurement"
          checked={params.sensing}
          onChange={(v) => setParams((p) => ({ ...p, sensing: v }))}
        />
        <Toggle
          label="Motion enabled"
          role="prediction"
          checked={params.moving}
          onChange={(v) => setParams((p) => ({ ...p, moving: v }))}
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

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}

/** Count local maxima that carry real mass — the "how many hypotheses" readout. */
function countModes(belief: number[]): number {
  const peak = Math.max(...belief);
  const floor = peak * 0.25;
  let n = 0;
  for (let i = 0; i < belief.length; i++) {
    const prev = belief[(i - 1 + belief.length) % belief.length];
    const next = belief[(i + 1) % belief.length];
    if (belief[i] > floor && belief[i] >= prev && belief[i] > next) n++;
  }
  return n;
}
