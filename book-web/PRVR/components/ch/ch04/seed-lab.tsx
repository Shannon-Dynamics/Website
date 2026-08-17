'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import {
  RUSTY,
  diffDriveSlipStep,
  segments,
  type RobotParams,
  type RustyState,
} from '@/lib/sim/rusty';
import {
  clear,
  drawGrid,
  drawPath,
  drawRobot,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import type { Pose2 } from '@/lib/geom/se2';

/**
 * w4.4 — the Seed Lab.
 *
 * One command script, forty-eight seeds, no walls: what a *distribution over
 * futures* looks like when you grow it from real actuation noise rather than
 * assuming a shape for it. This is why every benchmark in this book reports
 * over many seeds, and it is the empirical ancestor of the banana-shaped
 * motion-model posterior in Chapter 9.
 */

const N = 48;
const DT = 0.1;
const TICKS = 96;
const START: Pose2 = { x: 0, y: 0, theta: 0 };

/** Straight, then a long left arc, then straight again. Nine seconds, 5.8 m. */
const SCRIPT = segments([
  { u: { v: 0.6, omega: 0 }, ticks: 30 },
  { u: { v: 0.6, omega: 0.7 }, ticks: 36 },
  { u: { v: 0.6, omega: 0 }, ticks: 30 },
]);

interface Spread {
  t: number;
  along: number;
  cross: number;
}

interface State {
  robots: RustyState[];
  rngs: Rng[];
  /** The σ = 0 run: the trajectory the command script *intends*. */
  nominal: RustyState;
  trails: { x: number; y: number }[][];
  nominalTrail: { x: number; y: number }[];
  spread: Spread[];
  finalErrors: number[];
}

export function SeedLab() {
  const [slipStd, setSlipStd] = useState(0.035);
  const [rightOnly, setRightOnly] = useState(0);

  const params: RobotParams = useMemo(
    () => ({ ...RUSTY, slipStd, radiusBiasRight: rightOnly }),
    [slipStd, rightOnly],
  );

  const init = useCallback((seed: number): State => {
    const master = new Rng(seed);
    return {
      robots: Array.from({ length: N }, () => ({ pose: { ...START }, wheelAngles: [0, 0] })),
      // One independent stream per universe, forked from the master seed: the
      // whole ensemble is reproducible from a single integer.
      rngs: Array.from({ length: N }, () => master.fork()),
      nominal: { pose: { ...START }, wheelAngles: [0, 0] },
      trails: Array.from({ length: N }, () => [{ x: START.x, y: START.y }]),
      nominalTrail: [{ x: START.x, y: START.y }],
      spread: [{ t: 0, along: 0, cross: 0 }],
      finalErrors: new Array<number>(N).fill(0),
    };
  }, []);

  const step = useCallback(
    (s: State, tick: number): State => {
      const u = SCRIPT(tick);
      const clean: RobotParams = { ...params, slipStd: 0, radiusBiasRight: 0 };
      const nominalOut = diffDriveSlipStep(s.nominal, u, DT, null, clean, s.rngs[0]);
      const nominal: RustyState = { pose: nominalOut.pose, wheelAngles: nominalOut.wheelAngles };

      const robots: RustyState[] = [];
      const finalErrors: number[] = [];
      const c = Math.cos(nominal.pose.theta);
      const sn = Math.sin(nominal.pose.theta);
      let sumAlong2 = 0;
      let sumCross2 = 0;

      for (let i = 0; i < N; i++) {
        const out = diffDriveSlipStep(s.robots[i], u, DT, null, params, s.rngs[i]);
        robots.push({ pose: out.pose, wheelAngles: out.wheelAngles });
        s.trails[i].push({ x: out.pose.x, y: out.pose.y });

        const dx = out.pose.x - nominal.pose.x;
        const dy = out.pose.y - nominal.pose.y;
        const along = dx * c + dy * sn;
        const cross = -dx * sn + dy * c;
        sumAlong2 += along * along;
        sumCross2 += cross * cross;
        finalErrors.push(Math.hypot(dx, dy));
      }

      s.nominalTrail.push({ x: nominal.pose.x, y: nominal.pose.y });

      return {
        ...s,
        robots,
        nominal,
        spread: [
          ...s.spread,
          {
            t: (tick + 1) * DT,
            along: Math.sqrt(sumAlong2 / N),
            cross: Math.sqrt(sumCross2 / N),
          },
        ],
        finalErrors,
      };
    },
    [params],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 10,
    maxTicks: TICKS,
    loop: true,
    initialSeed: 2026,
  });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      const s = sim.state;
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 0.5);

      for (const trail of s.trails) {
        drawPath(ctx, v, trail, p.prediction, { lineWidth: 1, alpha: 0.3 });
      }
      drawPath(ctx, v, s.nominalTrail, p.truth, { dashed: true, lineWidth: 1.8 });

      for (const r of s.robots) {
        drawRobot(ctx, v, r.pose, p.prediction, 0.1, { filled: false, alpha: 0.5 });
      }
      drawRobot(ctx, v, s.nominal.pose, p.truth, 0.13, { filled: true, alpha: 0.95 });

      label(ctx, `${N} seeds · one command script`, 10, 14, p.truth, { size: 10, weight: 600 });
      label(ctx, 'commanded (noise-free)', sx(v, START.x) + 8, sy(v, START.y) + 16, p.truth, {
        size: 9.5,
      });
    },
    [sim.state],
  );

  // Downsampled: a Nivo redraw is a full SVG rebuild, and thirteen points say
  // everything ninety-seven would.
  const chart = useMemo(() => {
    const all = sim.state.spread;
    const pts = all.filter((_, i) => i % 8 === 0 || i === all.length - 1);
    return [
      {
        id: 'cross-track σ',
        role: 'prediction' as const,
        data: pts.map((d) => ({ x: d.t, y: d.cross })),
      },
      { id: 'along-track σ', role: 'prior' as const, data: pts.map((d) => ({ x: d.t, y: d.along })) },
    ];
  }, [sim.state.spread]);

  const stats = useMemo(() => {
    const e = [...sim.state.finalErrors].sort((a, b) => a - b);
    const mean = e.reduce((a, b) => a + b, 0) / Math.max(e.length, 1);
    const p95 = e[Math.min(e.length - 1, Math.floor(0.95 * e.length))] ?? 0;
    const last = sim.state.spread[sim.state.spread.length - 1];
    return { mean, p95, worst: e[e.length - 1] ?? 0, ratio: last.along > 1e-9 ? last.cross / last.along : 0 };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w4.4"
      title="The Seed Lab"
      teaches="One simulation run tells you nothing. A run is a sample; what you want is the distribution it came from."
      colorKey={['prior', 'prediction', 'truth']}
      caption={
        <>
          Every orange trace obeys the <em>same</em> command script. The only difference between
          them is the seed. Watch where the fan opens: barely at all while Rusty drives straight, then
          violently through the arc, because a heading error is free until you translate under it —
          and once you do, the position error grows with the <em>square</em> of distance, not its
          square root. Notice the shape too: cross-track spread outruns along-track spread by roughly
          three to one, and that anisotropy is the banana of Chapter 9, grown here from nothing but
          per-wheel slip. Try δ ≠ 0 on the second slider: the fan stops being centred on the dashed
          line at all. That is bias, and averaging over more seeds will never remove it.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -1.5, maxX: 4.0, minY: -0.9, maxY: 3.6 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={1.7}
        padding={0.1}
        ariaLabel="Forty-eight thin orange trajectories fan out from a common start point around a gray dashed noise-free trajectory, spreading most where the path curves."
      />

      <div className="border-t border-fd-border p-3">
        <Dashboard columns={4}>
          <StatTile label="mean |error|" value={stats.mean} unit="m" precision={3} role="prediction" />
          <StatTile label="95th percentile" value={stats.p95} unit="m" precision={3} role="prediction" />
          <StatTile label="worst seed" value={stats.worst} unit="m" precision={3} />
          <StatTile label="cross / along" value={stats.ratio} unit="×" precision={2} />
          <DashboardPanel title="Cross-seed spread vs. time" span="full">
            <LineChart
              series={chart}
              xLabel="t (s)"
              yLabel="σ (m)"
              height={200}
              yMin={0}
              caption="Root-mean-square deviation from the commanded trajectory, resolved into the nominal heading frame."
            />
          </DashboardPanel>
        </Dashboard>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Wheel slip σ_slip"
          role="prediction"
          value={slipStd}
          min={0}
          max={0.1}
          step={0.005}
          onChange={setSlipStd}
          help="Zero-mean, per wheel, independent every tick. The stochastic half."
        />
        <Slider
          label="Right-wheel radius error δ"
          role="truth"
          value={rightOnly}
          min={-0.02}
          max={0.02}
          step={0.002}
          format={(x) => `${(x * 100).toFixed(1)}%`}
          onChange={setRightOnly}
          help="A constant bias on one wheel. Watch the whole fan leave the dashed line together."
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
