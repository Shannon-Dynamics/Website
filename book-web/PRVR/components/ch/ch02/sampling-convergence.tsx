'use client';

import Link from 'next/link';
import { useCallback, useMemo } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { normalPdf } from '@/lib/prob/gaussian';
import { BarChart, Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';

/**
 * w2.3 — Sampling Convergence.
 *
 * The third representation of a distribution, after "formula" and "moments":
 * a bag of samples. The histogram fills in toward the true density as draws
 * accumulate, and the log-log panel shows the price — error falls like
 * 1/√N and not one bit faster, which is why Chapter 8 argues about particle
 * counts for an entire chapter.
 *
 * Everything is seeded. The same seed reproduces this figure exactly, in the
 * browser and in the Rust test suite alike.
 */

const LADDER = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000];
const LO = -3.6;
const HI = 3.6;
const BINS = 24;
const BIN_W = (HI - LO) / BINS;

interface State {
  rng: Rng;
  n: number;
  counts: number[];
  sum: number;
  sumSq: number;
  history: { n: number; err: number }[];
}

function fresh(seed: number): State {
  return {
    rng: new Rng(seed),
    n: 0,
    counts: new Array<number>(BINS).fill(0),
    sum: 0,
    sumSq: 0,
    history: [],
  };
}

/** Draw until the sample count reaches `target`. Accumulates; never restarts. */
function drawTo(s: State, target: number): State {
  if (target <= s.n) return s;
  const counts = s.counts.slice();
  let { sum, sumSq, n } = s;
  while (n < target) {
    const x = s.rng.normal();
    const b = Math.floor((x - LO) / BIN_W);
    if (b >= 0 && b < BINS) counts[b] += 1;
    sum += x;
    sumSq += x * x;
    n += 1;
  }
  return {
    ...s,
    counts,
    sum,
    sumSq,
    n,
    history: [...s.history, { n, err: Math.abs(sum / n) }],
  };
}

export function SamplingConvergence() {
  const init = useCallback((seed: number): State => drawTo(fresh(seed), LADDER[0]), []);

  const step = useCallback(
    (s: State, tick: number): State => drawTo(s, LADDER[Math.min(tick + 1, LADDER.length - 1)]),
    [],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 1.5,
    maxTicks: LADDER.length - 1,
    loop: true,
    initialSeed: 2026,
  });

  const { n, counts, sum, sumSq, history } = sim.state;
  const mean = sum / n;
  const variance = Math.max(sumSq / n - mean * mean, 0);

  const histogram = useMemo(() => {
    const empirical = counts.map((c, i) => ({
      x: (LO + (i + 0.5) * BIN_W).toFixed(1),
      y: c / (n * BIN_W),
    }));
    const exact = counts.map((_, i) => ({
      x: (LO + (i + 0.5) * BIN_W).toFixed(1),
      y: normalPdf(LO + (i + 0.5) * BIN_W, 0, 1),
    }));
    // Stable series ids: Nivo keys its colors and animations off them, and a
    // label that changed every tick would re-mount the chart each step.
    return [
      { id: 'sampled', role: 'prior' as const, data: empirical },
      { id: 'true density', role: 'truth' as const, data: exact },
    ];
  }, [counts, n]);

  const convergence = useMemo(() => {
    const measured = history.map((h) => ({ x: Math.log10(h.n), y: Math.log10(Math.max(h.err, 1e-6)) }));
    const envelope = LADDER.map((m) => ({ x: Math.log10(m), y: Math.log10(1 / Math.sqrt(m)) }));
    return [
      { id: '|x̄ − μ|', role: 'prior' as const, data: measured },
      { id: 'σ/√N', role: 'truth' as const, data: envelope },
    ];
  }, [history]);

  /** How many standard errors the current sample mean happens to be off by. */
  const zScore = Math.abs(mean) * Math.sqrt(n);

  const setN = useCallback(
    (target: number) => {
      sim.pause();
      // Rebuilding from the seed keeps the draw sequence identical, so jumping
      // to N = 5000 shows the same 5000 numbers the sweep would have produced.
      sim.setState(() => drawTo(fresh(sim.seed), Math.round(target)));
    },
    [sim],
  );

  return (
    <WidgetFrame
      id="w2.3"
      title="Sampling Convergence"
      teaches="A cloud of samples is a legitimate representation of a distribution — but its error only falls like 1/√N, so 'add more particles' is a slow fix."
      colorKey={['prior', 'truth']}
      caption={
        <>
          Ten samples look nothing like a bell curve, and the reflex is to conclude that sampling
          is broken. It is not: the histogram is an unbiased estimate the whole time, just a noisy
          one. Watch the left panel fill in as N climbs the ladder 10 → 100 000. Then read the
          right panel, which is the same run on log–log axes: the blue error rides the gray{' '}
          <code>σ/√N</code> line, a slope of −½. Buying one more decimal digit of accuracy costs a
          hundred times the samples. That single fact sets the particle counts in{' '}
          <Link href="/chapters/ch08-nonparametric-filters">Chapter 8</Link> and explains why nobody runs
          Monte Carlo localization in six dimensions without a very good proposal distribution.
          Re-roll the seed and watch the blue curve take a different random walk under the same
          envelope.
        </>
      }
    >
      <div className="p-3">
        <Dashboard columns={4}>
          <StatTile label="samples N" value={n} role="prior" />
          <StatTile label="sample mean" value={mean} precision={4} role="prior" />
          <StatTile label="sample variance" value={variance} precision={4} role="prior" />
          <StatTile
            label="|x̄ − μ| · √N   (≈ 1 by the CLT)"
            value={zScore}
            precision={2}
            role="truth"
            sparkline={history.map((h) => h.err * Math.sqrt(h.n))}
          />

          <DashboardPanel title="Histogram against the true density" span={2}>
            <BarChart
              series={histogram}
              xLabel="x"
              yLabel="density"
              height={230}
              groupMode="grouped"
              ariaLabel="A histogram of the samples drawn so far, in blue, next to the exact standard normal density in gray. As the sample count grows the two converge bin by bin."
            />
          </DashboardPanel>

          <DashboardPanel title="Error against 1/√N (log–log)" span={2}>
            <LineChart
              series={convergence}
              xLabel="log₁₀ N"
              yLabel="log₁₀ error"
              height={230}
              curve="linear"
              ariaLabel="A log-log plot of the absolute error of the sample mean against the number of samples, riding a straight reference line of slope minus one half."
            />
          </DashboardPanel>
        </Dashboard>
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="Sample count N"
          role="prior"
          value={n}
          min={10}
          max={100000}
          step={10}
          format={(v) => v.toFixed(0)}
          onChange={setN}
          help="Climbs the ladder 10 → 100 000 on its own. Drag to jump straight to a count; press reset to hand the sweep back."
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
      />
    </WidgetFrame>
  );
}
