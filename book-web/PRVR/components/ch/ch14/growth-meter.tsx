'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { LineChart, StatTile, type LineChartSeries } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { solve } from '@/lib/prob/linalg';
import { DEFAULT_SLAM_CONFIG, EkfSlam, slamCost } from '@/lib/slam/ekf-slam';

/**
 * w14.3 — the Growth Meter.
 *
 * Not an assertion about complexity: a measurement of it. Each tick builds a
 * real `EkfSlam` with N landmarks and times real observation updates in this
 * browser, then fits a·N² + b·N + c to whatever came out. The quadratic term
 * wins, and the extrapolation to a city-scale map is the number that ended the
 * filtering era.
 */

const LADDER = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];

interface Sample {
  n: number;
  dim: number;
  ms: number;
}

interface State {
  samples: Sample[];
}

/** Build a filter with N landmarks in a ring and time `iters` observation updates. */
function benchmark(n: number): Sample {
  const f = new EkfSlam(
    { x: 0, y: 0, theta: 0 },
    [
      [0.02, 0, 0],
      [0, 0.02, 0],
      [0, 0, 0.004],
    ],
    DEFAULT_SLAM_CONFIG,
  );
  for (let j = 0; j < n; j++) {
    f.initLandmark({ r: 2 + (j % 7) * 0.25, phi: ((j + 0.5) / n) * 2 * Math.PI - Math.PI, s: j }, j);
  }
  const dim = f.dim;
  // Aim for a few tens of milliseconds of work per point: long enough to beat
  // timer granularity, short enough that the widget stays interactive.
  const iters = Math.max(6, Math.min(400, Math.round(4e6 / (dim * dim))));
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const j = i % n;
    f.updateLandmark(j, { r: 2 + (j % 7) * 0.25, phi: ((j + 0.5) / n) * 2 * Math.PI - Math.PI, s: j });
  }
  return { n, dim, ms: (performance.now() - t0) / iters };
}

/** Least-squares a·N² + b·N + c through the measurements. */
function quadraticFit(samples: Sample[]): [number, number, number] {
  if (samples.length < 3) return [0, 0, 0];
  const A = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhs = [0, 0, 0];
  for (const s of samples) {
    const basis = [s.n * s.n, s.n, 1];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) A[i][j] += basis[i] * basis[j];
      rhs[i] += basis[i] * s.ms;
    }
  }
  const x = solve(A, rhs);
  return [x[0], x[1], x[2]];
}

export function GrowthMeter() {
  const [target, setTarget] = useState(100_000);

  const init = useCallback((): State => ({ samples: [] }), []);
  const step = useCallback((s: State, tick: number): State => {
    const n = LADDER[Math.min(tick, LADDER.length - 1)];
    return { samples: [...s.samples, benchmark(n)] };
  }, []);

  const sim = useSimulation<State>({
    init,
    step,
    fps: 3,
    maxTicks: LADDER.length,
    loop: false,
    initialSeed: 1,
  });

  const samples = sim.state.samples;
  const fit = useMemo(() => quadraticFit(samples), [samples]);

  const series = useMemo(() => {
    const measured = samples.map((s) => ({ x: s.n, y: s.ms }));
    const [a, b, c] = fit;
    const model =
      samples.length >= 3
        ? Array.from({ length: 40 }, (_, i) => {
            const x = (i / 39) * LADDER[LADDER.length - 1];
            return { x, y: Math.max(a * x * x + b * x + c, 0) };
          })
        : [];
    const out: LineChartSeries[] = [
      { id: 'measured ms/update', role: 'measurement', data: measured },
    ];
    if (model.length) out.push({ id: 'aN² + bN + c', role: 'prediction', data: model });
    return out;
  }, [samples, fit]);

  const stats = useMemo(() => {
    const [a, b, c] = fit;
    const at = (n: number) => Math.max(a * n * n + b * n + c, 0);
    const cost = slamCost(target);
    const last = samples[samples.length - 1];
    // Share of the predicted cost that the quadratic term alone accounts for.
    const share = samples.length >= 3 && at(target) > 0 ? (a * target * target) / at(target) : 0;
    return {
      last,
      msAtTarget: at(target),
      bytes: cost.bytes,
      dim: cost.dim,
      share,
      perEntry: last ? (last.ms * 1e6) / (last.dim * last.dim) : 0,
      history: samples.map((s) => s.ms),
    };
  }, [fit, samples, target]);

  const n = stats.last?.n ?? 0;
  const dim = stats.last?.dim ?? 3;

  return (
    <WidgetFrame
      id="w14.3"
      title="The Growth Meter"
      teaches="N² is not a footnote. Measured in your own browser, EKF SLAM's update cost curves away from linear before the map has fifty landmarks."
      colorKey={['prediction', 'measurement', 'posterior']}
      caption={
        <>
          Each point is a real measurement: a real <code>EkfSlam</code> with N landmarks, timed over
          real observation updates, right here. The orange curve is a least-squares
          a·N²&nbsp;+&nbsp;bN&nbsp;+&nbsp;c through whatever your machine produced. Nothing is
          asserted — the quadratic is <em>fitted</em>, and it wins because the covariance downdate
          Σ&nbsp;←&nbsp;Σ&nbsp;−&nbsp;K(Σ̄Hᵀ)ᵀ touches every one of the (3&nbsp;+&nbsp;2N)² entries
          for a measurement that names only five of them. Then move the extrapolation slider to a
          city-scale map and read the two numbers underneath. A faster language buys you a constant
          factor; a constant factor is exactly what a quadratic does not care about.
        </>
      }
    >
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[1fr_16rem]">
        <LineChart
          series={series}
          xLabel="landmarks N"
          yLabel="ms per observation update"
          height={260}
          yMin={0}
          caption={
            samples.length < LADDER.length
              ? `Measuring… ${samples.length} of ${LADDER.length} points.`
              : `Fit: ${fit[0].toExponential(2)}·N² + ${fit[1].toExponential(2)}·N + ${fit[2].toExponential(2)} ms.`
          }
        />
        <BlockMatrix n={n} dim={dim} />
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 sm:grid-cols-4">
        <StatTile
          label="state dimension 3 + 2N"
          value={dim}
          role="posterior"
          sparkline={stats.history}
        />
        <StatTile
          label="ns per Σ entry"
          value={stats.perEntry}
          precision={1}
          role="measurement"
        />
        <StatTile
          label={`update at N = ${target.toLocaleString()}`}
          value={formatDuration(stats.msAtTarget)}
          role="prediction"
        />
        <StatTile
          label={`dense Σ at N = ${target.toLocaleString()}`}
          value={formatBytes(stats.bytes)}
          role="prediction"
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="extrapolate to a map of N landmarks"
          role="prediction"
          value={Math.log10(target)}
          min={2}
          max={6}
          step={0.1}
          format={(v) => Math.round(10 ** v).toLocaleString()}
          onChange={(v) => setTarget(Math.round(10 ** v))}
          help="A room is 10². A building is 10³. A city block of visual features is 10⁵ and up."
        />
        <div className="font-ui text-[0.72rem] leading-snug text-fd-muted-foreground">
          The quadratic term accounts for{' '}
          <strong style={{ color: 'var(--pr-prediction)' }}>
            {(stats.share * 100).toFixed(0)}%
          </strong>{' '}
          of the predicted cost at N&nbsp;=&nbsp;{target.toLocaleString()}. Prediction stays O(N) —
          motion only touches the pose block and its strip — so this is the correction step alone.
        </div>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

/**
 * The state vector's covariance as a block schematic: the 3×3 pose block, the
 * pose–map strips motion rotates, and the map–map region that no motion ever
 * touches and every measurement rewrites.
 */
function BlockMatrix({ n, dim }: { n: number; dim: number }) {
  const posePct = (3 / Math.max(dim, 1)) * 100;
  const grid = n > 0 && n <= 24;
  const lines = grid
    ? Array.from({ length: n + 1 }, (_, k) => ((3 + 2 * k) / dim) * 100)
    : [];

  return (
    <figure className="not-prose m-0 flex flex-col gap-2">
      <svg viewBox="0 0 100 100" className="w-full" role="img" aria-label={`Block structure of a ${dim} by ${dim} covariance matrix with a 3 by 3 pose block and ${n} landmark blocks.`}>
        <rect x={0} y={0} width={100} height={100} style={{ fill: 'var(--pr-posterior)', opacity: 0.16 }} />
        <rect
          x={posePct}
          y={0}
          width={100 - posePct}
          height={posePct}
          style={{ fill: 'var(--pr-prediction)', opacity: 0.4 }}
        />
        <rect
          x={0}
          y={posePct}
          width={posePct}
          height={100 - posePct}
          style={{ fill: 'var(--pr-prediction)', opacity: 0.4 }}
        />
        <rect x={0} y={0} width={posePct} height={posePct} style={{ fill: 'var(--pr-prior)', opacity: 0.75 }} />
        {lines.map((p) => (
          <g key={p}>
            <line x1={p} y1={0} x2={p} y2={100} style={{ stroke: 'var(--color-fd-border)', strokeWidth: 0.4 }} />
            <line x1={0} y1={p} x2={100} y2={p} style={{ stroke: 'var(--color-fd-border)', strokeWidth: 0.4 }} />
          </g>
        ))}
        <rect
          x={0}
          y={0}
          width={100}
          height={100}
          style={{ fill: 'none', stroke: 'var(--color-fd-border)', strokeWidth: 1 }}
        />
      </svg>
      <figcaption className="font-mono text-[0.7rem] text-fd-muted-foreground">
        Σ is {dim}×{dim}. Blue: pose. Orange: the pose–map strips (O(N), rotated by every motion).
        Purple: the map block — untouched by motion, rewritten by every measurement.
      </figcaption>
    </figure>
  );
}

function formatBytes(bytes: number): string {
  const units = ['B', 'kB', 'MB', 'GB', 'TB', 'PB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

function formatDuration(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 3_600_000) return `${(ms / 60_000).toFixed(1)} min`;
  return `${(ms / 3_600_000).toFixed(1)} h`;
}
