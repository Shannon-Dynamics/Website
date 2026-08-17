'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import { chi2Envelope } from '@/lib/filters/consistency';
import { ellipse2 } from '@/lib/prob/linalg';
import {
  clear,
  drawCovariance,
  drawPath,
  drawRobot,
  drawSegments,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import { APARTMENT_COURSE, monteCarloNees } from '@/lib/slam/course';
import { Stat, captureRun, fitPanel } from './shared';

/**
 * w14.2 — the Consistency Autopsy.
 *
 * The filter is handed every advantage: correspondences are known, the process
 * noise it assumes is *exactly* the noise the world applies, and the prior is
 * calibrated. It still ends up overconfident, and the longer the excursion the
 * worse it gets. The instrument is NEES against its χ² band; the mechanism is
 * visible in the second chart, where the reported heading σ sinks below the
 * heading error it is supposed to describe.
 *
 * The run is recorded once (the covariance movie) and replayed, so the scrubber
 * can rewind to the moment of death without re-simulating.
 */

const ASPECT = 2.45;
const STEPS = 320;
const RUNS = 40;
const MAP_PANEL = { x0: 0.03, y0: 0.06, w: 2.39, h: 0.88 };

interface Params {
  eastX: number;
  monteCarlo: boolean;
}

export function ConsistencyAutopsy() {
  const [params, setParams] = useState<Params>({ eastX: 11.2, monteCarlo: false });

  const sim = useSimulation<number>({
    init: () => 0,
    step: (i) => i + 1,
    fps: 26,
    maxTicks: STEPS - 1,
    loop: true,
    initialSeed: 42,
  });

  const movie = useMemo(
    () =>
      captureRun(STEPS, {
        seed: sim.seed,
        knownCorrespondence: true,
        params: { eastX: params.eastX },
      }),
    [sim.seed, params.eastX],
  );

  const mc = useMemo(
    () =>
      params.monteCarlo
        ? monteCarloNees(RUNS, STEPS, {
            seed: sim.seed,
            knownCorrespondence: true,
            params: { eastX: params.eastX },
          })
        : null,
    [params.monteCarlo, sim.seed, params.eastX],
  );

  // A single NEES is one χ²(3) draw; averaging R runs shrinks the band by √R.
  // Switching the toggle switches *both* the series and the band it is judged
  // against — comparing a single run to the averaged band would be cheating.
  const band = useMemo(() => chi2Envelope(3, params.monteCarlo ? RUNS : 1), [params.monteCarlo]);

  const series = useMemo(() => {
    const source = mc ? mc.meanNees : movie.frames.map((f) => f.nees);
    const data = source
      .map((y, x) => ({ x, y }))
      .filter((_, i) => i % 2 === 0);
    return [
      {
        id: mc ? `mean NEES over ${RUNS} runs` : 'NEES, single run',
        role: mc ? ('posterior' as const) : ('prediction' as const),
        data,
      },
    ];
  }, [mc, movie]);

  const headingSeries = useMemo(
    () => [
      {
        id: 'reported σ_θ',
        role: 'posterior' as const,
        data: movie.frames
          .map((f, x) => ({ x, y: f.headingSigma }))
          .filter((_, i) => i % 2 === 0),
      },
      {
        id: '|heading error|',
        role: 'truth' as const,
        data: movie.frames
          .map((f, x) => ({ x, y: f.headingError }))
          .filter((_, i) => i % 2 === 0),
      },
    ],
    [movie],
  );

  const escape = useMemo(() => {
    const source = mc ? mc.meanNees : movie.frames.map((f) => f.nees);
    for (let i = 20; i < source.length; i++) if (source[i] > band.hi) return i;
    return -1;
  }, [mc, movie, band.hi]);

  const stats = useMemo(() => {
    const i = Math.min(sim.state, movie.frames.length - 1);
    const f = movie.frames[i];
    const source = mc ? mc.meanNees : movie.frames.map((fr) => fr.nees);
    const tail = source.slice(Math.floor(source.length / 2));
    const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
    return {
      frame: f,
      i,
      mean,
      inside: mean >= band.lo && mean <= band.hi,
      mapRmse: movie.mapRmse[i],
    };
  }, [sim.state, movie, mc, band]);

  /* ---------------------------------------------------------------- */

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const f = stats.frame;
      const m = fitPanel(MAP_PANEL, APARTMENT.bounds);
      const wx = (x: number) => sx(v, m.toX(x));
      const wy = (y: number) => sy(v, m.toY(y));

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

      drawPath(ctx, v, f.truthPath.map((q) => ({ x: m.toX(q.x), y: m.toY(q.y) })), p.truth, {
        dashed: true,
        lineWidth: 1.3,
      });
      drawPath(ctx, v, f.estPath.map((q) => ({ x: m.toX(q.x), y: m.toY(q.y) })), p.posterior, {
        lineWidth: 1.3,
        alpha: 0.8,
      });

      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.2;
      for (const b of APARTMENT_COURSE) {
        ctx.beginPath();
        ctx.moveTo(wx(b.x) - 3.5, wy(b.y));
        ctx.lineTo(wx(b.x) + 3.5, wy(b.y));
        ctx.moveTo(wx(b.x), wy(b.y) - 3.5);
        ctx.lineTo(wx(b.x), wy(b.y) + 3.5);
        ctx.stroke();
      }
      ctx.restore();

      for (const lm of f.landmarks) {
        const e = ellipse2(lm.cov, 2);
        drawCovariance(
          ctx,
          v,
          [m.toX(lm.x), m.toY(lm.y)],
          { rx: e.rx * m.scale, ry: e.ry * m.scale, angle: e.angle },
          p.posterior,
          { alpha: 0.8, lineWidth: 1.3 },
        );
      }

      const pe = ellipse2(f.poseCov, 2);
      drawCovariance(
        ctx,
        v,
        [m.toX(f.est.x), m.toY(f.est.y)],
        { rx: pe.rx * m.scale, ry: pe.ry * m.scale, angle: pe.angle },
        p.posterior,
        { alpha: 1, lineWidth: 1.8 },
      );
      drawRobot(
        ctx,
        v,
        { x: m.toX(f.truth.x), y: m.toY(f.truth.y), theta: f.truth.theta },
        p.truth,
        0.42 * m.scale,
      );
      drawRobot(
        ctx,
        v,
        { x: m.toX(f.est.x), y: m.toY(f.est.y), theta: f.est.theta },
        p.posterior,
        0.42 * m.scale,
        { filled: false },
      );

      // The turnaround point — the parameter the reader is moving.
      ctx.save();
      ctx.strokeStyle = p.prediction;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(wx(params.eastX), wy(3.85));
      ctx.lineTo(wx(params.eastX), wy(4.95));
      ctx.stroke();
      ctx.restore();
      label(ctx, 'turnaround', wx(params.eastX), wy(3.6), p.prediction, {
        size: 9,
        align: 'center',
      });

      label(
        ctx,
        `t = ${stats.i}${f.closure ? '   ·   LOOP CLOSURE' : ''}`,
        wx(0.2),
        wy(9.35),
        f.closure ? p.accent : p.ink,
        { size: 10, weight: 600 },
      );
      label(
        ctx,
        `error ${f.error.toFixed(3)} m   ·   claimed σ ${Math.sqrt(
          0.5 * (f.poseCov[0][0] + f.poseCov[1][1]),
        ).toFixed(3)} m`,
        wx(11.8),
        wy(9.35),
        p.ink,
        { size: 10, align: 'right' },
      );
    },
    [stats, params.eastX],
  );

  return (
    <WidgetFrame
      id="w14.2"
      title="The Consistency Autopsy"
      teaches="A small reported covariance is not evidence of a good estimate. EKF SLAM grows confident and wrong at the same time."
      colorKey={['prediction', 'posterior', 'truth']}
      wide
      caption={
        <>
          Everything is in the filter&rsquo;s favour: correspondences are handed to it, the process
          noise it assumes is <em>exactly</em> the noise the simulator applies, and the initial prior
          is honest. The only approximation left is linearization. Move the{' '}
          <strong>loop length</strong> slider: below about six metres the robot never loses sight of
          the west cluster and the filter stays inside its band. Past that it crosses open corridor
          on dead reckoning, maps the far cluster through an inflated pose, and the NEES average
          climbs out of the band and stays out. The second chart is the mechanism — the reported
          heading σ keeps falling while the true heading error does not, because the filter treats
          the rotation of its own map as fresh metric information. Switch to{' '}
          <strong>{RUNS}-run Monte Carlo</strong> and the band tightens by √{RUNS}: a single trace
          bouncing above 9 proves nothing, an average of forty sitting above 3.8 proves everything.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: ASPECT, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.state, movie, params.eastX]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="The Apartment corridor with Rusty's true path dashed in gray and the filter's estimated path in purple, with covariance ellipses for the pose and every mapped beacon."
      />

      <div className="grid grid-cols-1 gap-3 border-t border-fd-border p-3 lg:grid-cols-2">
        <LineChart
          series={series}
          xLabel="step"
          yLabel="NEES"
          height={230}
          yMin={0}
          yMax={params.monteCarlo ? 10 : 22}
          markers={[
            { axis: 'y', value: 3, label: 'E[NEES] = 3', role: 'truth' },
            { axis: 'y', value: band.hi, label: '95% limit', role: 'prediction' },
            ...(escape >= 0
              ? [{ axis: 'x' as const, value: escape, label: 'escape', role: 'prediction' as const }]
              : []),
          ]}
          caption={
            params.monteCarlo
              ? `Mean NEES over ${RUNS} seeded runs. The acceptance band for an average of ${RUNS} χ²(3) statistics is [${band.lo.toFixed(2)}, ${band.hi.toFixed(2)}].`
              : `One run. The single-sample band is [${band.lo.toFixed(2)}, ${band.hi.toFixed(2)}] — wide enough to hide almost any misbehaviour.`
          }
        />
        <LineChart
          series={headingSeries}
          xLabel="step"
          yLabel="radians"
          height={230}
          yMin={0}
          caption="The claimed heading uncertainty against the heading error actually made. When purple sits below gray for long stretches, the filter is asserting a precision it does not have — and every landmark it maps inherits that lie."
        />
      </div>

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-5">
        <Stat label="frame" value={`${stats.i} / ${STEPS - 1}`} />
        <Stat label="pose error" value={`${stats.frame.error.toFixed(3)} m`} />
        <Stat label="map RMSE" value={`${stats.mapRmse.toFixed(3)} m`} />
        <Stat
          label={params.monteCarlo ? `mean NEES (${RUNS} runs)` : 'mean NEES (2nd half)'}
          value={stats.mean.toFixed(2)}
          alarm={!stats.inside}
        />
        <Stat
          label="escape step"
          value={escape < 0 ? 'never' : String(escape)}
          alarm={escape >= 0}
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="loop length — turnaround x"
          role="prediction"
          value={params.eastX}
          min={3.5}
          max={11.2}
          step={0.1}
          unit="m"
          onChange={(v) => setParams((q) => ({ ...q, eastX: v }))}
          help="How far east Rusty drives before turning back. Longer means more corridor crossed with no beacon in view."
        />
        <Slider
          label="scrub to the moment of death"
          value={stats.i}
          min={0}
          max={STEPS - 1}
          step={1}
          format={(v) => String(Math.round(v))}
          onChange={(v) => {
            sim.pause();
            sim.setState(() => Math.round(v));
          }}
        />
        <Toggle
          label={`${RUNS}-run Monte Carlo`}
          role="posterior"
          checked={params.monteCarlo}
          onChange={(v) => setParams((q) => ({ ...q, monteCarlo: v }))}
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
