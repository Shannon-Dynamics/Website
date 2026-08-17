'use client';

import { useCallback, useMemo } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { Transport } from '@/components/sim/controls';
import { Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import { OccupancyGrid, DEFAULT_INVERSE_MODEL } from '@/lib/mapping/occgrid';
import { Slam2d, type LoopEvent, type SlamReport } from '@/lib/slam/slam2d';
import {
  clear,
  drawOccupancyGrid,
  drawPath,
  drawRobot,
  drawSegments,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w16.4 — the RustSLAM-2D Control Room.
 *
 * The integration lab. Everything on this panel is one `Slam2d` instance: the
 * occupancy map is integrated from the keyframe scans at the poses the graph
 * currently holds, the error plot compares the graph against raw odometry, and
 * the log is the front end and the back end taking turns.
 *
 * The map is built incrementally — one keyframe at a time, the cheap way — and
 * **rebuilt from scratch whenever a loop closure is accepted**, because the
 * poses those scans were written at have just changed. That rebuild is not an
 * implementation detail; it is why Thrun's draft insisted that a robot mapping
 * a cycle has to keep its raw scans rather than only the grid.
 */

const CELL = 0.15;
const MAX_TICKS = 131;

interface LogLine {
  t: number;
  text: string;
  kind: 'front' | 'back' | 'reject';
}

interface State {
  slam: Slam2d;
  grid: OccupancyGrid;
  report: SlamReport | null;
  integrated: number;
  truth: { x: number; y: number }[];
  odom: { x: number; y: number }[];
  series: { t: number; graph: number; odom: number }[];
  log: LogLine[];
  events: LoopEvent[];
}

const INVERSE_MODEL = { ...DEFAULT_INVERSE_MODEL, maxRange: 6, alpha: 0.22 };

function integrate(grid: OccupancyGrid, slam: Slam2d, from: number): void {
  for (let k = from; k < slam.keyframes.length; k++) {
    const kf = slam.keyframes[k];
    grid.integrateScan(slam.graph.nodes[kf.node].pose, kf.ranges, slam.angles, INVERSE_MODEL);
  }
}

export function RustSlamControlRoom() {
  const init = useCallback((seed: number): State => {
    const slam = new Slam2d(APARTMENT, seed);
    const grid = OccupancyGrid.forWorld(APARTMENT, CELL);
    integrate(grid, slam, 0);
    return {
      slam,
      grid,
      report: null,
      integrated: slam.keyframes.length,
      truth: [{ x: slam.truth.x, y: slam.truth.y }],
      odom: [{ x: slam.deadReckon.x, y: slam.deadReckon.y }],
      series: [],
      log: [{ t: 0, text: 'node 0 fixed — this is the world frame', kind: 'back' }],
      events: [],
    };
  }, []);

  const step = useCallback((s: State): State => {
    const report = s.slam.step();
    const log = [...s.log];
    let grid = s.grid;
    let integrated = s.integrated;

    if (report.keyframeAdded) {
      const node = s.slam.keyframes[s.slam.keyframes.length - 1].node;
      log.push({
        t: report.t,
        text: `node ${node} · icp ${report.icp.rmse.toFixed(3)} m over ${report.icp.inliers} pairs · τ ${report.tau.toFixed(2)} m`,
        kind: 'front',
      });
    }

    if (report.loop) {
      const e = report.loop;
      if (e.accepted) {
        log.push({
          t: report.t,
          text: `loop ${e.from}→${e.to} VERIFIED · χ² ${e.chi2.toFixed(2)} < 7.81 · rmse ${e.rmse.toFixed(3)} m`,
          kind: 'back',
        });
        log.push({
          t: report.t,
          text: `optimize · history moved ${(e.maxShift ?? 0).toFixed(2)} m · ATE ${(e.ateBefore ?? 0).toFixed(2)} → ${(e.ateAfter ?? 0).toFixed(2)} m · map rebuilt`,
          kind: 'back',
        });
        // Every past pose changed, so every past scan is in the wrong place.
        grid = OccupancyGrid.forWorld(APARTMENT, CELL);
        integrate(grid, s.slam, 0);
        integrated = s.slam.keyframes.length;
      } else {
        log.push({
          t: report.t,
          text: `loop ${e.from}→${e.to} rejected · χ² ${e.chi2.toFixed(2)} · rmse ${e.rmse.toFixed(3)} m · ${e.inliers} pairs`,
          kind: 'reject',
        });
      }
    }

    if (integrated < s.slam.keyframes.length) {
      integrate(grid, s.slam, integrated);
      integrated = s.slam.keyframes.length;
    }

    return {
      slam: s.slam,
      grid,
      report,
      integrated,
      truth: [...s.truth, { x: report.truth.x, y: report.truth.y }],
      odom: [...s.odom, { x: report.deadReckon.x, y: report.deadReckon.y }],
      series: [...s.series, { t: report.t, graph: report.ate, odom: report.odomError }],
      log: log.slice(-9),
      events: report.loop ? [...s.events, report.loop] : s.events,
    };
  }, []);

  const sim = useSimulation<State>({
    init,
    step,
    fps: 10,
    maxTicks: MAX_TICKS,
    loop: true,
    initialSeed: 12648430,
  });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { slam, grid, report } = sim.state;
      drawOccupancyGrid(ctx, v, grid, p);
      drawSegments(ctx, v, APARTMENT.walls, p.grid, 1);

      const nodes = slam.graph.nodes;
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (const e of slam.graph.edges) {
        if (e.kind !== 'odometry') continue;
        ctx.moveTo(sx(v, nodes[e.i].pose.x), sy(v, nodes[e.i].pose.y));
        ctx.lineTo(sx(v, nodes[e.j].pose.x), sy(v, nodes[e.j].pose.y));
      }
      ctx.stroke();

      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      for (const e of slam.graph.edges) {
        if (e.kind !== 'loop') continue;
        ctx.moveTo(sx(v, nodes[e.i].pose.x), sy(v, nodes[e.i].pose.y));
        ctx.lineTo(sx(v, nodes[e.j].pose.x), sy(v, nodes[e.j].pose.y));
      }
      ctx.stroke();
      ctx.restore();

      drawPath(ctx, v, sim.state.truth, p.truth, { dashed: true, lineWidth: 1.4 });
      drawPath(ctx, v, sim.state.odom, p.prediction, { lineWidth: 1.4, alpha: 0.75 });

      if (report) {
        drawRobot(ctx, v, report.truth, p.truth, 0.26, { filled: false });
        drawRobot(ctx, v, report.estimate, p.posterior, 0.26);
      }

      label(ctx, `t = ${(report?.t ?? 0).toFixed(1)} s`, 10, 14, p.ink, { size: 10, weight: 600 });
      label(
        ctx,
        `${slam.keyframes.length} keyframes · ${slam.graph.edges.length} factors`,
        10,
        28,
        p.posterior,
        { size: 10 },
      );
    },
    [sim.state],
  );

  const chart = useMemo(() => {
    const s = sim.state.series;
    return [
      { id: 'raw odometry', role: 'prediction' as const, data: s.map((d) => ({ x: d.t, y: d.odom })) },
      { id: 'pose graph', role: 'posterior' as const, data: s.map((d) => ({ x: d.t, y: d.graph })) },
    ];
  }, [sim.state.series]);

  const stats = useMemo(() => {
    const s = sim.state;
    const accepted = s.events.filter((e) => e.accepted);
    return {
      ate: s.report?.ate ?? 0,
      consistency: s.report?.consistency ?? 0,
      odom: s.report?.odomError ?? 0,
      rmse: s.report?.icp.rmse ?? 0,
      tau: s.report?.tau ?? 0,
      nodes: s.slam.graph.nodes.length,
      factors: s.slam.graph.edges.length,
      loops: accepted.length,
      rejected: s.events.length - accepted.length,
      spark: s.series.slice(-80).map((d) => d.graph),
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w16.4"
      title="RustSLAM-2D Control Room"
      teaches="SLAM is not one algorithm. It is a fast greedy front end and a slow honest back end taking turns, and you can watch the hand-off."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Everything here is one <code>Slam2d</code> instance. The occupancy map is integrated from
          the keyframe scans at the poses the graph currently holds; the plot compares the graph's
          absolute trajectory error against raw wheel odometry; the log is the two halves of the
          system talking. <strong>What to notice:</strong> the front end speaks every keyframe and
          the back end says nothing at all until a loop is proposed — then it rewrites the whole
          trajectory and the map is thrown away and rebuilt, because the poses those scans were
          written at just moved. <strong>What to notice second:</strong> rejected candidates. A
          proposal that fails the χ² gate costs a few milliseconds; one that passes when it should
          not have costs you the map.
        </>
      }
    >
      <div className="p-3">
        <Dashboard columns={3}>
          <DashboardPanel title="world · map · graph" span={2}>
            <SimCanvas
              world={APARTMENT.bounds}
              draw={draw}
              deps={[sim.tick, sim.state]}
              aspect={12 / 9}
              padding={0.2}
              ariaLabel="An occupancy grid map of the apartment being built as the robot drives, with the pose graph drawn over it. When a loop closure is accepted the map is rebuilt and the doubled walls collapse."
            />
          </DashboardPanel>

          <DashboardPanel title="event log">
            <ol className="m-0 list-none space-y-1 p-0 font-mono text-[0.66rem] leading-snug">
              {sim.state.log.map((line, i) => (
                <li
                  key={`${line.t}-${i}`}
                  className="flex gap-1.5"
                  style={{
                    color:
                      line.kind === 'back'
                        ? 'var(--pr-measurement)'
                        : line.kind === 'reject'
                          ? 'var(--pr-prediction)'
                          : undefined,
                  }}
                >
                  <span className="shrink-0 tabular-nums opacity-60">{line.t.toFixed(1)}s</span>
                  <span>{line.text}</span>
                </li>
              ))}
            </ol>
          </DashboardPanel>

          <DashboardPanel title="position error vs time" span="full">
            <LineChart
              series={chart}
              xLabel="time (s)"
              yLabel="error (m)"
              height={190}
              yMin={0}
              curve="linear"
            />
          </DashboardPanel>
        </Dashboard>

        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <StatTile
            label="ATE"
            value={stats.ate}
            unit="m"
            role="posterior"
            precision={2}
            sparkline={stats.spark}
          />
          <StatTile label="odometry error" value={stats.odom} unit="m" role="prediction" precision={2} />
          <StatTile
            label="map disagreement"
            value={stats.consistency}
            unit="m"
            role="prior"
            precision={3}
          />
          <StatTile
            label="graph"
            value={`${stats.nodes}n / ${stats.factors}f`}
            trend={stats.loops}
            trendLabel={`loops · ${stats.rejected} rejected`}
            precision={0}
          />
        </div>
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
