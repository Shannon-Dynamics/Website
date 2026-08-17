'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import { Slam2d, DEFAULT_SLAM_CONFIG, type LoopEvent, type SlamReport } from '@/lib/slam/slam2d';
import type { MotionAlphas } from '@/lib/models/motion';
import {
  clear,
  drawPath,
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
 * w16.2 — Loop Snap.
 *
 * The whole system, running: Rusty drives the Apartment's corridor east and
 * back while a scan-matching front end estimates the trajectory and a pose
 * graph collects it. When the return leg gets close enough to an old node, the
 * loop is proposed, verified, and — if it survives the gate — handed to the
 * Gauss–Newton back end, which rewrites *every* pose in the graph at once.
 *
 * The correction is not applied to "where I am". It is applied to where the
 * robot was, thirty seconds ago, and the map redraws itself accordingly.
 */

const BASE_ALPHAS = DEFAULT_SLAM_CONFIG.alphas;

interface State {
  slam: Slam2d;
  report: SlamReport | null;
  /** Front-end-only trajectory: the poses as they were before any optimization. */
  raw: { x: number; y: number }[];
  truth: { x: number; y: number }[];
  odom: { x: number; y: number }[];
  events: LoopEvent[];
  ateHistory: number[];
  preLoopAte: number;
  flash: number;
}

export function LoopSnap() {
  const [noise, setNoise] = useState(1);
  const [closing, setClosing] = useState(true);

  const init = useCallback(
    (seed: number): State => {
      const alphas = BASE_ALPHAS.map((a) => a * noise) as MotionAlphas;
      const slam = new Slam2d(APARTMENT, seed, {
        alphas,
        loopRadius: closing ? DEFAULT_SLAM_CONFIG.loopRadius : 0,
      });
      return {
        slam,
        report: null,
        raw: [{ x: slam.estimate.x, y: slam.estimate.y }],
        truth: [{ x: slam.truth.x, y: slam.truth.y }],
        odom: [{ x: slam.deadReckon.x, y: slam.deadReckon.y }],
        events: [],
        ateHistory: [],
        preLoopAte: 0,
        flash: 0,
      };
    },
    [noise, closing],
  );

  const step = useCallback((s: State): State => {
    const report = s.slam.step();
    const events = report.loop ? [...s.events, report.loop] : s.events;
    const accepted = events.filter((e) => e.accepted).length;
    return {
      slam: s.slam,
      report,
      raw: [...s.raw, { x: report.estimate.x, y: report.estimate.y }],
      truth: [...s.truth, { x: report.truth.x, y: report.truth.y }],
      odom: [...s.odom, { x: report.deadReckon.x, y: report.deadReckon.y }],
      events,
      ateHistory: [...s.ateHistory, report.ate].slice(-160),
      preLoopAte: accepted === 0 ? report.ate : s.preLoopAte,
      flash: report.loop?.accepted ? 8 : Math.max(0, s.flash - 1),
    };
  }, []);

  const sim = useSimulation<State>({
    init,
    step,
    fps: 12,
    maxTicks: 131,
    loop: true,
    initialSeed: 12648430,
  });

  const resetRef = useRef(sim.reset);
  resetRef.current = sim.reset;
  useEffect(() => {
    resetRef.current();
  }, [init]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { slam, report, flash } = sim.state;
      drawSegments(ctx, v, APARTMENT.walls, p.wall, 2);

      // The map, drawn from the keyframe clouds at whatever pose the graph now
      // believes. Before a loop closes it is visibly double-walled.
      ctx.fillStyle = p.prior;
      ctx.globalAlpha = 0.55;
      for (const pt of slam.mapPoints()) {
        ctx.fillRect(sx(v, pt[0]) - 0.9, sy(v, pt[1]) - 0.9, 1.8, 1.8);
      }
      ctx.globalAlpha = 1;

      drawPath(ctx, v, sim.state.truth, p.truth, { dashed: true, lineWidth: 1.6 });
      drawPath(ctx, v, sim.state.odom, p.prediction, { lineWidth: 1.6, alpha: 0.9 });
      // The front end's own answer, never re-optimized: the ghost the graph
      // pulls away from every time a loop is accepted.
      drawPath(ctx, v, sim.state.raw, p.posterior, { lineWidth: 1, alpha: 0.3 });

      // The pose graph: nodes, odometry edges, and any verified loop factors.
      const nodes = slam.graph.nodes;
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (const e of slam.graph.edges) {
        if (e.kind !== 'odometry') continue;
        ctx.moveTo(sx(v, nodes[e.i].pose.x), sy(v, nodes[e.i].pose.y));
        ctx.lineTo(sx(v, nodes[e.j].pose.x), sy(v, nodes[e.j].pose.y));
      }
      ctx.stroke();

      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = flash > 0 ? 3 : 1.8;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (const e of slam.graph.edges) {
        if (e.kind !== 'loop') continue;
        ctx.moveTo(sx(v, nodes[e.i].pose.x), sy(v, nodes[e.i].pose.y));
        ctx.lineTo(sx(v, nodes[e.j].pose.x), sy(v, nodes[e.j].pose.y));
      }
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = p.posterior;
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(sx(v, n.pose.x), sy(v, n.pose.y), n.fixed ? 3.4 : 2.1, 0, Math.PI * 2);
        ctx.fill();
      }

      if (report) {
        drawRobot(ctx, v, report.truth, p.truth, 0.26, { filled: false });
        drawRobot(ctx, v, report.deadReckon, p.prediction, 0.24, { filled: false, alpha: 0.8 });
        drawRobot(ctx, v, report.estimate, p.posterior, 0.26);
      }

      const accepted = sim.state.events.filter((e) => e.accepted);
      label(
        ctx,
        accepted.length > 0
          ? `${accepted.length} loop factor${accepted.length > 1 ? 's' : ''} verified`
          : 'front end only — no loop yet',
        10,
        16,
        accepted.length > 0 ? p.measurement : p.prediction,
        { size: 11, weight: 600 },
      );
      if (flash > 0) {
        const last = accepted[accepted.length - 1];
        if (last) {
          label(
            ctx,
            `optimizing: node ${last.from} → ${last.to},  history moved ${(last.maxShift ?? 0).toFixed(2)} m`,
            10,
            32,
            p.measurement,
            { size: 10 },
          );
        }
      }
      label(ctx, `scale: ${sl(v, 1).toFixed(0)} px = 1 m`, v.width - 10, v.height - 10, p.truth, {
        size: 9,
        align: 'right',
      });
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const s = sim.state;
    const accepted = s.events.filter((e) => e.accepted);
    const last = accepted[accepted.length - 1];
    return {
      odom: s.report?.odomError ?? 0,
      ate: s.report?.ate ?? 0,
      consistency: s.report?.consistency ?? 0,
      preLoopAte: s.preLoopAte,
      loops: accepted.length,
      chi2: s.events[s.events.length - 1]?.chi2 ?? 0,
      shift: last?.maxShift ?? 0,
      spark: s.ateHistory,
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w16.2"
      title="Loop Snap"
      teaches="A loop closure does not fix where the robot is. It rewrites where the robot was — and the map with it."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Gray dashed is the truth, orange is raw wheel odometry, purple is the pose graph, blue
          dots are the keyframe scans drawn at the poses the graph currently holds.{' '}
          <strong>What to notice:</strong> odometry leaves the building within one length of the
          corridor while the scan-matching front end stays within a metre — and yet the map is
          visibly double-walled on the return leg, because the two passes disagree. When a green
          loop factor lands, the whole purple trajectory shifts, <em>including poses from thirty
          seconds ago</em>, and the doubled walls collapse onto one another.{' '}
          <strong>What to try:</strong> turn loop closure off and watch the map stay double for
          good. Then push odometry noise up: the front end absorbs a surprising amount of it,
          because the scan is what fixes the pose and the odometry only has to get ICP into the
          right basin.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={12 / 9}
        padding={0.25}
        ariaLabel="A floorplan with three trajectories: ground truth, a wheel-odometry estimate that drifts badly, and a pose-graph estimate that stays close. Green dashed loop factors appear on the return leg and the whole estimated trajectory shifts when they are optimized."
      />

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 sm:grid-cols-4">
        <StatTile label="odometry error" value={stats.odom} unit="m" role="prediction" precision={2} />
        <StatTile
          label="ATE (graph)"
          value={stats.ate}
          unit="m"
          role="posterior"
          precision={2}
          sparkline={stats.spark}
          trend={stats.loops > 0 ? stats.ate - stats.preLoopAte : undefined}
          trendLabel="since first loop"
        />
        <StatTile
          label="map disagreement"
          value={stats.consistency}
          unit="m"
          role="prior"
          precision={3}
        />
        <StatTile
          label="loops verified"
          value={stats.loops}
          role="measurement"
          precision={0}
          trend={stats.chi2}
          trendLabel="last χ² (gate 7.81)"
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Odometry noise ×"
          role="prediction"
          value={noise}
          min={0.25}
          max={4}
          step={0.25}
          onChange={setNoise}
          help="Scales all six α parameters of the velocity motion model."
        />
        <Toggle
          label="Loop closure enabled"
          role="measurement"
          checked={closing}
          onChange={setClosing}
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
