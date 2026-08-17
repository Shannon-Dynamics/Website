'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import {
  clear,
  drawCovariance,
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
import { ellipse2 } from '@/lib/prob/linalg';
import { landmarkIndex } from '@/lib/slam/ekf-slam';
import { APARTMENT_COURSE, CourseSim, type StepResult } from '@/lib/slam/course';
import { Stat, fitPanel } from './shared';

/**
 * w14.1 — the Correlation Web.
 *
 * Two views of one object. On the left, Rusty patrolling the Apartment
 * landmark course: purple ellipses per landmark, threads between landmark pairs
 * whose width tracks |ρ|. On the right, the same Σ as a correlation heatmap
 * with its block grid showing. Hover either and the other lights up.
 *
 * The headline is the toggle. "Diagonal-only" zeroes every cross-block of Σ
 * after each update — the map keeps its marginals and forgets that it is one
 * correlated object. The filter does not crash. It gets *more confident*, by a
 * factor of a hundred, about a map that is more wrong than before.
 */

const ASPECT = 2.35;
const RUN = 460;

interface Params {
  sigmaR: number;
  ablate: boolean;
  showThreads: boolean;
}

interface State {
  sim: CourseSim;
  last: StepResult | null;
  flash: number;
  closures: number;
}

const MAP_PANEL = { x0: 0.02, y0: 0.04, w: 1.4, h: 0.92 };
const HEAT_PANEL = { x0: 1.5, y0: 0.1, w: 0.8, h: 0.8 };

export function CorrelationWeb() {
  const [params, setParams] = useState<Params>({ sigmaR: 0.06, ablate: false, showThreads: true });
  const [hover, setHover] = useState<number | null>(null);

  const init = useCallback(
    (seed: number): State => ({
      sim: new CourseSim({ seed, knownCorrespondence: true }),
      last: null,
      flash: 0,
      closures: 0,
    }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      // Live knobs are pushed into the running filter rather than rebuilt, so
      // the reader can turn the ablation on mid-lap and watch Σ change shape.
      s.sim.sensor.sigmaR = params.sigmaR;
      s.sim.filter.cfg.sigmaR = params.sigmaR;
      s.sim.filter.cfg.ablateCorrelations = params.ablate;
      const last = s.sim.step();
      return {
        sim: s.sim,
        last,
        flash: last.closures.length > 0 ? 14 : Math.max(0, s.flash - 1),
        closures: s.closures + last.closures.length,
      };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 16, maxTicks: RUN, loop: true, initialSeed: 42 });

  /* ---------------------------------------------------------------- */
  /* Readouts                                                          */
  /* ---------------------------------------------------------------- */

  const stats = useMemo(() => {
    const f = sim.state.sim.filter;
    let sq = 0;
    let k = 0;
    for (let j = 0; j < f.count; j++) {
      const truth = sim.state.sim.truthFor(j);
      if (!truth) continue;
      const [mx, my] = f.landmarkMean(j);
      sq += (mx - truth.x) ** 2 + (my - truth.y) ** 2;
      k += 1;
    }
    const P = f.poseCov();
    return {
      n: f.count,
      claimed: Math.sqrt(f.mapUncertainty()),
      actual: k > 0 ? Math.sqrt(sq / k) : 0,
      rho: f.meanLandmarkCorrelation(),
      poseSigma: Math.sqrt(Math.max(0.5 * (P[0][0] + P[1][1]), 0)),
      closures: sim.state.closures,
    };
  }, [sim.state]);

  /* ---------------------------------------------------------------- */
  /* Drawing                                                           */
  /* ---------------------------------------------------------------- */

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { sim: course, last, flash } = sim.state;
      const f = course.filter;
      const m = fitPanel(MAP_PANEL, APARTMENT.bounds);
      const px = (x: number) => sx(v, x);
      const py = (y: number) => sy(v, y);
      const wx = (x: number) => px(m.toX(x));
      const wy = (y: number) => py(m.toY(y));

      /* ---- the Apartment ------------------------------------------- */
      drawSegments(
        ctx,
        v,
        APARTMENT.walls.map((w) => ({
          x1: m.toX(w.x1),
          y1: m.toY(w.y1),
          x2: m.toX(w.x2),
          y2: m.toY(w.y2),
        })),
        p.wall,
        1.6,
      );

      // Ground truth: where the beacons really are, and where Rusty really went.
      drawPath(
        ctx,
        v,
        course.truthPath.map((q) => ({ x: m.toX(q.x), y: m.toY(q.y) })),
        p.truth,
        { dashed: true, lineWidth: 1.2, alpha: 0.75 },
      );
      drawPath(
        ctx,
        v,
        course.estimatePath.map((q) => ({ x: m.toX(q.x), y: m.toY(q.y) })),
        p.posterior,
        { lineWidth: 1.2, alpha: 0.55 },
      );

      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.2;
      for (const b of APARTMENT_COURSE) {
        const bx = wx(b.x);
        const by = wy(b.y);
        ctx.beginPath();
        ctx.moveTo(bx - 3.5, by);
        ctx.lineTo(bx + 3.5, by);
        ctx.moveTo(bx, by - 3.5);
        ctx.lineTo(bx, by + 3.5);
        ctx.stroke();
      }
      ctx.restore();

      /* ---- the web: threads between correlated landmark pairs -------- */
      const rho = f.correlation();
      if (params.showThreads) {
        ctx.save();
        ctx.strokeStyle = p.posterior;
        for (let a = 0; a < f.count; a++) {
          for (let b = a + 1; b < f.count; b++) {
            const ia = landmarkIndex(a);
            const ib = landmarkIndex(b);
            const strength = 0.5 * (Math.abs(rho[ia][ib]) + Math.abs(rho[ia + 1][ib + 1]));
            if (strength < 0.08) continue;
            const [ax, ay] = f.landmarkMean(a);
            const [bx, by] = f.landmarkMean(b);
            ctx.globalAlpha = 0.1 + 0.55 * strength;
            ctx.lineWidth = 0.4 + 2.6 * strength;
            ctx.beginPath();
            ctx.moveTo(wx(ax), wy(ay));
            ctx.lineTo(wx(bx), wy(by));
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      /* ---- measurement rays ----------------------------------------- */
      if (last) {
        ctx.save();
        ctx.strokeStyle = p.measurement;
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 1;
        for (const feat of last.features) {
          const a = course.truth.theta + feat.phi;
          ctx.beginPath();
          ctx.moveTo(wx(course.truth.x), wy(course.truth.y));
          ctx.lineTo(wx(course.truth.x + feat.r * Math.cos(a)), wy(course.truth.y + feat.r * Math.sin(a)));
          ctx.stroke();
        }
        ctx.restore();
      }

      /* ---- landmark estimates --------------------------------------- */
      for (let j = 0; j < f.count; j++) {
        const [mx, my] = f.landmarkMean(j);
        const e = ellipse2(f.landmarkCov(j), 2);
        const highlighted = hover === j;
        drawCovariance(
          ctx,
          v,
          [m.toX(mx), m.toY(my)],
          { rx: e.rx * m.scale, ry: e.ry * m.scale, angle: e.angle },
          highlighted ? p.accent : p.posterior,
          { alpha: highlighted ? 1 : 0.85, lineWidth: highlighted ? 2.2 : 1.4 },
        );
        ctx.fillStyle = highlighted ? p.accent : p.posterior;
        ctx.beginPath();
        ctx.arc(wx(mx), wy(my), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      /* ---- the robot, twice ----------------------------------------- */
      const est = f.pose();
      const pe = ellipse2(f.poseCov(), 2);
      drawCovariance(
        ctx,
        v,
        [m.toX(est.x), m.toY(est.y)],
        { rx: pe.rx * m.scale, ry: pe.ry * m.scale, angle: pe.angle },
        p.posterior,
        { alpha: 0.9, lineWidth: 1.6 },
      );
      drawRobot(
        ctx,
        v,
        { x: m.toX(course.truth.x), y: m.toY(course.truth.y), theta: course.truth.theta },
        p.truth,
        0.42 * m.scale,
      );
      drawRobot(
        ctx,
        v,
        { x: m.toX(est.x), y: m.toY(est.y), theta: est.theta },
        p.posterior,
        0.42 * m.scale,
        { filled: false },
      );

      if (flash > 0) {
        ctx.save();
        ctx.globalAlpha = Math.min(1, flash / 8);
        label(ctx, 'LOOP CLOSURE', px(m.toX(6)), py(m.toY(9.3)), p.accent, {
          size: 12,
          weight: 700,
          align: 'center',
        });
        ctx.restore();
      }

      /* ---- the same Σ, as a correlation heatmap ---------------------- */
      const n = f.dim;
      const cell = Math.min(HEAT_PANEL.w / n, HEAT_PANEL.h / n);
      const gx0 = HEAT_PANEL.x0 + (HEAT_PANEL.w - cell * n) / 2;
      const gy0 = HEAT_PANEL.y0 + (HEAT_PANEL.h - cell * n) / 2;
      const cw = sl(v, cell) + 0.6;

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const r = rho[i][j];
          const a = Math.min(1, Math.abs(r));
          if (a < 0.02) continue;
          ctx.globalAlpha = 0.12 + 0.88 * a;
          ctx.fillStyle = r >= 0 ? p.prediction : p.prior;
          // Row 0 of the matrix is drawn at the top, so the y axis is flipped.
          ctx.fillRect(px(gx0 + j * cell), py(gy0 + (n - 1 - i) * cell + cell), cw, cw);
        }
      }
      ctx.globalAlpha = 1;

      // Block grid: the pose block, then one 2×2 per landmark.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= f.count; k++) {
        const at = 3 + 2 * k > n ? n : 3 + 2 * k;
        const t = k === 0 ? 3 : at;
        ctx.moveTo(px(gx0 + t * cell), py(gy0));
        ctx.lineTo(px(gx0 + t * cell), py(gy0 + n * cell));
        ctx.moveTo(px(gx0), py(gy0 + (n - t) * cell));
        ctx.lineTo(px(gx0 + n * cell), py(gy0 + (n - t) * cell));
      }
      ctx.stroke();
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 1;
      ctx.strokeRect(px(gx0), py(gy0 + n * cell), sl(v, n * cell), sl(v, n * cell));

      if (hover !== null && hover < f.count) {
        const b = landmarkIndex(hover);
        ctx.save();
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 1.8;
        ctx.strokeRect(px(gx0 + b * cell), py(gy0 + n * cell), sl(v, 2 * cell), sl(v, n * cell));
        ctx.strokeRect(px(gx0), py(gy0 + (n - b) * cell), sl(v, n * cell), sl(v, 2 * cell));
        ctx.restore();
      }

      label(ctx, 'Σ as correlation ρ', px(gx0), py(gy0 + n * cell) - 10, p.ink, { size: 10, weight: 600 });
      label(ctx, 'x y θ', px(gx0), py(gy0) + 11, p.ink, { size: 9 });
      label(ctx, `${f.count} landmark blocks`, px(gx0 + 3 * cell), py(gy0) + 11, p.ink, { size: 9 });
      label(ctx, 'orange +   blue −', px(gx0 + n * cell), py(gy0) + 24, p.ink, { size: 9, align: 'right' });
    },
    [sim.state, hover, params.showThreads],
  );

  /* ---------------------------------------------------------------- */
  /* Hover linking                                                     */
  /* ---------------------------------------------------------------- */

  const onPointer = useCallback(
    (world: [number, number]) => {
      const f = sim.state.sim.filter;
      const m = fitPanel(MAP_PANEL, APARTMENT.bounds);
      const [u, w] = world;

      if (u < HEAT_PANEL.x0) {
        const gx = m.fromX(u);
        const gy = m.fromY(w);
        let best: number | null = null;
        let bestD = 0.6;
        for (let j = 0; j < f.count; j++) {
          const [mx, my] = f.landmarkMean(j);
          const d = Math.hypot(mx - gx, my - gy);
          if (d < bestD) {
            bestD = d;
            best = j;
          }
        }
        setHover(best);
        return;
      }

      const n = f.dim;
      const cell = Math.min(HEAT_PANEL.w / n, HEAT_PANEL.h / n);
      const gx0 = HEAT_PANEL.x0 + (HEAT_PANEL.w - cell * n) / 2;
      const col = Math.floor((u - gx0) / cell);
      if (col < 3 || col >= n) {
        setHover(null);
        return;
      }
      setHover(Math.floor((col - 3) / 2));
    },
    [sim.state],
  );

  return (
    <WidgetFrame
      id="w14.1"
      title="The Correlation Web"
      teaches="The off-diagonal blocks of Σ are not bookkeeping — they are the map. Delete them and the filter becomes certain of a map that is wrong."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty patrols the Apartment corridor between two clusters of beacons with three and a half
          metres of featureless corridor in the middle. Left: the map, with purple 2σ ellipses and threads
          whose width is <em>|ρ|</em> between landmark pairs. Right: the same Σ, normalized to
          correlations, with the 3×3 pose block and one 2×2 block per landmark. Hover either view to
          light up the other. Watch the crossing: with nothing in sight the pose ellipse balloons,
          the far cluster gets mapped <em>through</em> that uncertainty, and the whole east block of
          the matrix goes orange. Then the robot comes back and re-sees a west beacon it has not
          measured in a hundred steps — the pose σ collapses by about 5×, and the correction reaches
          landmarks the robot did not measure, because Σ said they shared the error.
          Now switch on <strong>diagonal-only</strong>. The threads vanish, the ellipses shrink to
          dots, and the claimed map σ falls to a few millimetres while the true map error climbs.
          That is a filter that has stopped knowing what it does not know.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: ASPECT, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state, hover, params.showThreads]}
        aspect={ASPECT}
        padding={0}
        cursor="crosshair"
        onPointer={onPointer}
        ariaLabel="Left: the Apartment floorplan with Rusty, eight beacon estimates drawn as purple ellipses, and threads between correlated landmark pairs. Right: the filter's covariance matrix rendered as a correlation heat map with a visible block structure."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-5">
        <Stat label="landmarks N" value={String(stats.n)} />
        <Stat label="pose σ" value={`${stats.poseSigma.toFixed(3)} m`} />
        <Stat label="claimed map σ" value={`${stats.claimed.toFixed(3)} m`} />
        <Stat
          label="actual map RMSE"
          value={`${stats.actual.toFixed(3)} m`}
          alarm={stats.actual > 3 * Math.max(stats.claimed, 1e-6)}
        />
        <Stat label="mean |ρ| landmarks" value={stats.rho.toFixed(2)} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="measurement noise σ_r"
          role="measurement"
          value={params.sigmaR}
          min={0.02}
          max={0.4}
          step={0.01}
          unit="m"
          onChange={(v) => setParams((q) => ({ ...q, sigmaR: v }))}
          help="Range noise for both the sensor and the filter's Q. Louder sensing means slower map convergence — and a floor it can never beat."
        />
        <Toggle
          label="diagonal-only ablation"
          checked={params.ablate}
          onChange={(v) => setParams((q) => ({ ...q, ablate: v }))}
        />
        <Toggle
          label="show correlation threads"
          role="posterior"
          checked={params.showThreads}
          onChange={(v) => setParams((q) => ({ ...q, showThreads: v }))}
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
