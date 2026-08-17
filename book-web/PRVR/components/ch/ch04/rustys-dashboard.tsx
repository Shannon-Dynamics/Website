'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT } from '@/lib/sim/world';
import {
  RUSTY,
  diffDriveSlipStep,
  driftError,
  encoderTicks,
  integrateOdometry,
  odometryDelta,
  pursuePoint,
  type EncoderTicks,
  type RobotParams,
  type RustyState,
  type Twist,
} from '@/lib/sim/rusty';
import {
  clear,
  drawGrid,
  drawPath,
  drawRobot,
  drawWorld,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import type { Pose2 } from '@/lib/geom/se2';

/**
 * w4.1 — Rusty's Dashboard.
 *
 * The chapter's front door, and the lab every later chapter runs in. Two poses
 * are drawn: the ground truth Rusty (gray, dashed trail) and the pose you get
 * by integrating the wheel encoders (orange ghost, orange trail). They start
 * identical. They never come back together.
 *
 * Everything here is the real library: `diffDriveSlipStep` for the physics,
 * `encoderTicks` + `odometryDelta` for the dead reckoning, `boxplus`/`se2Exp`
 * underneath both.
 */

const DT = 0.1;
const START: Pose2 = { x: 2.2, y: 4.4, theta: 0 };
/** The two ends of the corridor. The scripted tour paces between them. */
const WAYPOINTS = [
  { x: 9.8, y: 4.4 },
  { x: 2.2, y: 4.4 },
];

const ROOMS: { x: number; y: number; text: string }[] = [
  { x: 2.0, y: 2.6, text: 'A · living' },
  { x: 6.4, y: 2.8, text: 'B · kitchen' },
  { x: 10.2, y: 2.6, text: 'C · bath' },
  { x: 3.4, y: 6.6, text: 'D · study' },
  { x: 8.6, y: 6.2, text: 'E · bedroom' },
  { x: 6.0, y: 4.4, text: 'corridor' },
];

interface State {
  rusty: RustyState;
  /** The dead-reckoned pose: everything the robot itself is entitled to believe. */
  estimate: Pose2;
  prevTicks: EncoderTicks;
  truthTrail: { x: number; y: number }[];
  drTrail: { x: number; y: number }[];
  /** Path length actually travelled on the floor, metres. */
  distance: number;
  blocked: boolean;
  /** Which end of the corridor the scripted tour is currently heading for. */
  waypoint: number;
  rng: Rng;
  history: number[];
}

export function RustysDashboard() {
  const [slipStd, setSlipStd] = useState(0.02);
  const [radiusBias, setRadiusBias] = useState(0);
  const [manual, setManual] = useState(false);

  const keys = useRef<Set<string>>(new Set());
  const manualRef = useRef(false);

  const params: RobotParams = useMemo(
    () => ({ ...RUSTY, slipStd, radiusBiasRight: radiusBias }),
    [slipStd, radiusBias],
  );

  const init = useCallback(
    (seed: number): State => ({
      rusty: { pose: { ...START }, wheelAngles: [0, 0] },
      estimate: { ...START },
      prevTicks: { left: 0, right: 0 },
      truthTrail: [{ x: START.x, y: START.y }],
      drTrail: [{ x: START.x, y: START.y }],
      distance: 0,
      blocked: false,
      waypoint: 0,
      rng: new Rng(seed),
      history: [0],
    }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      const target = WAYPOINTS[s.waypoint];
      const reached =
        Math.hypot(target.x - s.rusty.pose.x, target.y - s.rusty.pose.y) < 0.4;
      const waypoint = reached ? 1 - s.waypoint : s.waypoint;
      const u: Twist = manualRef.current
        ? readKeys(keys.current)
        : pursuePoint(s.rusty.pose, WAYPOINTS[waypoint]);

      // 1. Physics. The wheels turn by the commanded amount; the floor decides
      //    how far that actually got us.
      const out = diffDriveSlipStep(s.rusty, u, DT, APARTMENT, params, s.rng);

      // 2. Sensing. Quantize the wheel angles, difference them, and integrate.
      const ticks = encoderTicks(out.wheelAngles, params);
      const tau = odometryDelta(s.prevTicks, ticks, params);
      const estimate = integrateOdometry(s.estimate, tau);

      const moved = Math.hypot(out.pose.x - s.rusty.pose.x, out.pose.y - s.rusty.pose.y);
      const err = driftError(estimate, out.pose);

      const push = (trail: { x: number; y: number }[], p: Pose2) => {
        const last = trail[trail.length - 1];
        if (Math.hypot(p.x - last.x, p.y - last.y) < 0.02) return trail;
        return [...trail, { x: p.x, y: p.y }].slice(-1400);
      };

      return {
        rusty: { pose: out.pose, wheelAngles: out.wheelAngles },
        estimate,
        prevTicks: ticks,
        truthTrail: push(s.truthTrail, out.pose),
        drTrail: push(s.drTrail, estimate),
        distance: s.distance + moved,
        blocked: out.blocked,
        waypoint,
        rng: s.rng,
        history: [...s.history, err.position].slice(-90),
      };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 18, initialSeed: 4 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      const s = sim.state;
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 1);
      drawWorld(ctx, v, APARTMENT, p);

      for (const r of ROOMS) {
        label(ctx, r.text, sx(v, r.x), sy(v, r.y), p.truth, { size: 9.5, align: 'center' });
      }

      drawPath(ctx, v, s.truthTrail, p.truth, { dashed: true, lineWidth: 1.6, alpha: 0.95 });
      drawPath(ctx, v, s.drTrail, p.prediction, { lineWidth: 1.8, alpha: 0.95 });

      // The error vector, drawn as the thing it is: a gap the robot cannot see.
      ctx.save();
      ctx.strokeStyle = p.prediction;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, s.rusty.pose.x), sy(v, s.rusty.pose.y));
      ctx.lineTo(sx(v, s.estimate.x), sy(v, s.estimate.y));
      ctx.stroke();
      ctx.restore();

      drawRobot(ctx, v, s.estimate, p.prediction, RUSTY.bodyRadius, { filled: false });
      drawRobot(ctx, v, s.rusty.pose, p.truth, RUSTY.bodyRadius, { filled: true, alpha: 0.9 });

      if (s.blocked) {
        label(
          ctx,
          'BLOCKED — wheels still turning',
          sx(v, s.rusty.pose.x),
          sy(v, s.rusty.pose.y) - sl(v, 0.35),
          p.measurement,
          { size: 10, align: 'center', weight: 600 },
        );
      }

      label(
        ctx,
        manual ? 'MANUAL — you have the wheel' : 'AUTOPILOT — corridor patrol',
        10,
        14,
        p.truth,
        { size: 9.5, weight: 600 },
      );
    },
    [sim.state, manual],
  );

  const stats = useMemo(() => {
    const err = driftError(sim.state.estimate, sim.state.rusty.pose);
    return {
      distance: sim.state.distance,
      position: err.position,
      heading: (err.heading * 180) / Math.PI,
      relative: sim.state.distance > 0.25 ? (100 * err.position) / sim.state.distance : 0,
      history: sim.state.history,
    };
  }, [sim.state]);

  // Arrow keys and WASD drive Rusty. Pressing any of them takes the wheel.
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!DRIVE_KEYS.has(e.key)) return;
    e.preventDefault();
    keys.current.add(e.key);
    if (!manualRef.current) {
      manualRef.current = true;
      setManual(true);
    }
  };
  const onKeyUp = (e: React.KeyboardEvent) => {
    if (!DRIVE_KEYS.has(e.key)) return;
    e.preventDefault();
    keys.current.delete(e.key);
  };

  useEffect(() => {
    const held = keys.current;
    return () => held.clear();
  }, []);

  return (
    <WidgetFrame
      id="w4.1"
      title="Rusty's Dashboard"
      teaches="Odometry is not a position sensor. It is a velocity sensor, integrated hopefully."
      colorKey={['prediction', 'truth']}
      caption={
        <>
          Gray dashed is where Rusty <em>is</em>; orange is where its wheel encoders <em>say</em> it
          is. They begin identical and drift apart forever — the dashed connector is an error no
          onboard sensor can measure. Notice that the gap grows fastest through the turns at each end
          of the corridor, and that it never once shrinks. Then click the map and drive with the
          arrow keys: park Rusty against a wall and hold the throttle, and watch the orange ghost
          sail on through the wall while the real robot sits still. Finally set slip to zero — the
          two traces coincide, which is exactly why the second slider matters: a right wheel 1%
          larger than the model believes is a <em>systematic</em> error, and averaging never removes
          it.
          <br />
          <span className="opacity-75">
            Two honest disclosures. The autopilot steers on ground truth, a luxury no real robot has;
            it is here so Rusty stays in the corridor while you study the drift, and Chapter 20 builds
            the version that does not cheat. And σ_slip = 0.02 per 100 ms tick is several times worse
            than a decent indoor rover — set so the lesson arrives in ten seconds rather than ten
            minutes.
          </span>
        </>
      }
    >
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
      <div
        tabIndex={0}
        onKeyDown={onKeyDown}
        onKeyUp={onKeyUp}
        onBlur={() => keys.current.clear()}
        aria-label="Drive Rusty with the arrow keys or WASD"
        className="relative outline-none focus-visible:ring-2 focus-visible:ring-fd-primary"
      >
        <SimCanvas
          world={APARTMENT.bounds}
          draw={draw}
          deps={[sim.tick, sim.state, manual]}
          aspect={1.5}
          ariaLabel="A floorplan of the Apartment with a corridor and five rooms. A gray robot follows a dashed gray trail down the corridor while an orange ghost robot, integrated from the wheel encoders, follows a trail that slowly separates from it."
        />
        <p className="pointer-events-none absolute right-2 bottom-2 m-0 rounded-sm bg-fd-card/80 px-1.5 py-0.5 font-ui text-[0.65rem] text-fd-muted-foreground">
          click, then ↑ ↓ ← → or WASD
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 sm:grid-cols-4">
        <StatTile label="distance driven" value={stats.distance} unit="m" precision={2} />
        <StatTile
          label="position drift"
          value={stats.position}
          unit="m"
          role="prediction"
          precision={3}
          sparkline={stats.history}
        />
        <StatTile
          label="heading drift"
          value={stats.heading}
          unit="°"
          role="prediction"
          precision={2}
        />
        <StatTile label="drift / distance" value={stats.relative} unit="%" precision={2} />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Wheel slip σ_slip"
          role="prediction"
          value={slipStd}
          min={0}
          max={0.12}
          step={0.002}
          onChange={setSlipStd}
          help="Stochastic: each wheel's ground travel is its rotation times (1 + ε), ε ~ N(0, σ²)."
        />
        <Slider
          label="Right-wheel radius error δ"
          role="truth"
          value={radiusBias}
          min={-0.03}
          max={0.03}
          step={0.002}
          format={(x) => `${(x * 100).toFixed(1)}%`}
          onChange={setRadiusBias}
          help="Systematic: the true right wheel is r(1+δ) while odometry still divides by r."
        />
      </ControlPanel>

      <div className="flex flex-wrap items-center gap-2 border-t border-fd-border px-3 py-2">
        <ActionButton
          onClick={() => {
            manualRef.current = false;
            keys.current.clear();
            setManual(false);
          }}
        >
          Resume scripted tour
        </ActionButton>
        <span className="font-ui text-[0.7rem] text-fd-muted-foreground">
          {manual ? 'You have the wheel.' : 'Rusty is patrolling the corridor.'}
        </span>
      </div>

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

const DRIVE_KEYS = new Set([
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'w',
  'a',
  's',
  'd',
  'W',
  'A',
  'S',
  'D',
]);

/** Held keys → a body twist. Note there is no strafe key: Rusty is nonholonomic. */
function readKeys(held: Set<string>): Twist {
  const fwd = held.has('ArrowUp') || held.has('w') || held.has('W');
  const back = held.has('ArrowDown') || held.has('s') || held.has('S');
  const left = held.has('ArrowLeft') || held.has('a') || held.has('A');
  const right = held.has('ArrowRight') || held.has('d') || held.has('D');
  return {
    v: (fwd ? 0.6 : 0) - (back ? 0.4 : 0),
    omega: (left ? 1.2 : 0) - (right ? 1.2 : 0),
  };
}
