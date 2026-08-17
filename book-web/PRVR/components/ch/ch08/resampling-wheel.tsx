'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { BarChart, Dashboard, DashboardPanel, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { lowVarianceResample, multinomialResample, type Particle } from '@/lib/filters/pf';
import { Rng } from '@/lib/prob/rng';

/**
 * w8.2 — the Resampling Wheel.
 *
 * Thrun's roulette wheel on the left, the low-variance comb on the right, both
 * spun on the same weights. The wheels are the *same* two library functions the
 * particle filter calls, so the histograms below are measurements of the real
 * resamplers, not of a re-implementation for the figure.
 *
 * Reproducing the pointer geometry deserves a note. `multinomialResample` calls
 * `rng.next()` exactly M times and `lowVarianceResample` calls `rng.uniform`
 * exactly once (with an interval of width 1/M, since we hand it normalized
 * weights). Both generators are seeded, so replaying the same seed reproduces
 * the exact pointer positions the resampler used — no shadow implementation.
 */

/** The chapter's worked example. The prose, this widget, and the Rust test share these numbers. */
const CHAPTER_WEIGHTS = [0.1, 0.3, 0.05, 0.4, 0.15];
const DEGENERATE_WEIGHTS = [0.96, 0.01, 0.01, 0.01, 0.01];
const UNIFORM_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2];

const N = 5;

interface Trial {
  /** Offspring count per particle. */
  counts: number[];
  /** Where each of the M pointers landed, in [0, 1). */
  pointers: number[];
}

interface State {
  rng: Rng;
  trials: number;
  sumMult: number[];
  sumSqMult: number[];
  sumComb: number[];
  sumSqComb: number[];
  lastMult: Trial;
  lastComb: Trial;
}

const normalize = (w: number[]): number[] => {
  const s = w.reduce((a, b) => a + b, 0) || 1;
  return w.map((x) => x / s);
};

/** Index-carrying particles: the state slot holds i, so offspring are countable. */
const asParticles = (w: number[]): Particle[] =>
  w.map((weight, i) => ({ state: { x: i, y: 0, theta: 0 }, weight }));

const tally = (out: Particle[], n: number): number[] => {
  const c = new Array<number>(n).fill(0);
  for (const p of out) c[Math.round(p.state.x)] += 1;
  return c;
};

function spinMultinomial(w: number[], m: number, seed: number): Trial {
  const particles = asParticles(w);
  // One draw is one pointer, so M draws replay as M pointers.
  const drawn = multinomialResample(padTo(particles, m), new Rng(seed));
  const replay = new Rng(seed);
  const pointers = Array.from({ length: m }, () => replay.next());
  return { counts: tally(drawn, w.length), pointers };
}

function spinComb(w: number[], m: number, seed: number): Trial {
  const particles = asParticles(w);
  const drawn = lowVarianceResample(padTo(particles, m), new Rng(seed));
  // The comb's single random number: r ~ U[0, 1/M). Everything else is fixed.
  const r = new Rng(seed).uniform(0, 1 / m);
  const pointers = Array.from({ length: m }, (_, k) => r + k / m);
  return { counts: tally(drawn, w.length), pointers };
}

/**
 * Both library resamplers return as many particles as they are given, so to
 * draw M offspring from N weighted hypotheses we present the set padded with
 * zero-weight placeholders. Zero weight means zero offspring, so the
 * distribution is untouched.
 */
function padTo(particles: Particle[], m: number): Particle[] {
  if (m <= particles.length) return particles;
  const pad: Particle[] = Array.from({ length: m - particles.length }, () => ({
    state: { x: particles.length - 1, y: 0, theta: 0 },
    weight: 0,
  }));
  return [...particles, ...pad];
}

export function ResamplingWheel() {
  const [weights, setWeights] = useState<number[]>(CHAPTER_WEIGHTS);
  const [m, setM] = useState(5);
  const w = useMemo(() => normalize(weights), [weights]);

  // The trial loop is created once; parameters reach it through refs so that
  // dragging a weight does not restart the statistics mid-experiment.
  const weightsRef = useRef(weights);
  const mRef = useRef(m);
  weightsRef.current = weights;
  mRef.current = m;

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      trials: 0,
      sumMult: new Array<number>(N).fill(0),
      sumSqMult: new Array<number>(N).fill(0),
      sumComb: new Array<number>(N).fill(0),
      sumSqComb: new Array<number>(N).fill(0),
      lastMult: { counts: new Array<number>(N).fill(0), pointers: [] },
      lastComb: { counts: new Array<number>(N).fill(0), pointers: [] },
    }),
    [],
  );

  const runTrials = useCallback(
    (s: State, n: number, wNow: number[], mNow: number): State => {
      const { rng } = s;
      const sumMult = s.sumMult.slice();
      const sumSqMult = s.sumSqMult.slice();
      const sumComb = s.sumComb.slice();
      const sumSqComb = s.sumSqComb.slice();
      let lastMult = s.lastMult;
      let lastComb = s.lastComb;

      for (let t = 0; t < n; t++) {
        const seedA = Math.floor(rng.next() * 2 ** 31);
        const seedB = Math.floor(rng.next() * 2 ** 31);
        lastMult = spinMultinomial(wNow, mNow, seedA);
        lastComb = spinComb(wNow, mNow, seedB);
        for (let i = 0; i < N; i++) {
          sumMult[i] += lastMult.counts[i];
          sumSqMult[i] += lastMult.counts[i] ** 2;
          sumComb[i] += lastComb.counts[i];
          sumSqComb[i] += lastComb.counts[i] ** 2;
        }
      }

      return { ...s, trials: s.trials + n, sumMult, sumSqMult, sumComb, sumSqComb, lastMult, lastComb };
    },
    [],
  );

  const step = useCallback(
    (s: State): State => runTrials(s, 1, normalize(weightsRef.current), mRef.current),
    [runTrials],
  );

  const sim = useSimulation<State>({ init, step, fps: 6, initialSeed: 15 });
  const { setState, reset } = sim;

  const batch = useCallback(
    (n: number) => setState((s) => runTrials(s, n, normalize(weightsRef.current), mRef.current)),
    [setState, runTrials, weightsRef, mRef],
  );

  const applyPreset = useCallback(
    (preset: number[]) => {
      setWeights(preset);
      reset();
    },
    [reset],
  );

  const s = sim.state;
  const trials = Math.max(s.trials, 1);

  const stats = useMemo(() => {
    const meanMult = s.sumMult.map((x) => x / trials);
    const meanComb = s.sumComb.map((x) => x / trials);
    const varMult = s.sumSqMult.map((x, i) => Math.max(0, x / trials - meanMult[i] ** 2));
    const varComb = s.sumSqComb.map((x, i) => Math.max(0, x / trials - meanComb[i] ** 2));
    const expected = w.map((x) => m * x);
    const theoryMult = w.map((x) => m * x * (1 - x));
    const frac = expected.map((x) => x - Math.floor(x));
    const theoryComb = frac.map((f) => f * (1 - f));
    return {
      meanMult,
      meanComb,
      varMult,
      varComb,
      expected,
      theoryMult,
      theoryComb,
      totalVarMult: varMult.reduce((a, b) => a + b, 0),
      totalVarComb: varComb.reduce((a, b) => a + b, 0),
    };
  }, [s, trials, w, m]);

  return (
    <WidgetFrame
      id="w8.2"
      title="Resampling Wheel: roulette vs. comb"
      teaches="Both resamplers have the same expectation. Only the variance differs — and variance was the enemy all along."
      colorKey={['prediction', 'posterior', 'truth']}
      wide
      caption={
        <>
          The default weights are the chapter&apos;s worked example,{' '}
          <code>[0.10, 0.30, 0.05, 0.40, 0.15]</code>, so the prose, this widget, and the Rust test
          all argue over the same five numbers. On the left, <em>M</em> independent spins of the
          roulette wheel. On the right, <strong>one</strong> spin and then a comb of <em>M</em>{' '}
          evenly-spaced teeth. Watch the first chart: the mean offspring bars converge onto the gray{' '}
          <em>M·wᵢ</em> line for <em>both</em> schemes — the comb is not biased, it is merely less
          random. Now watch the second chart: the comb&apos;s variance is a small fraction of the
          wheel&apos;s, and it is exactly zero wherever <em>M·wᵢ</em> lands on a whole number. Try
          the <em>degenerate</em> preset and ask which scheme can lose particle 1 altogether.
        </>
      }
    >
      <div className="grid gap-0 border-b border-fd-border sm:grid-cols-2 sm:divide-x sm:divide-fd-border">
        <Wheel
          title="Multinomial (roulette)"
          subtitle={`${m} independent spins`}
          weights={w}
          trial={s.lastMult}
          color="var(--pr-prediction)"
        />
        <Wheel
          title="Low-variance (comb)"
          subtitle="1 spin, M evenly spaced teeth"
          weights={w}
          trial={s.lastComb}
          color="var(--pr-posterior)"
        />
      </div>

      <div className="px-3 py-3">
        <Dashboard columns={4}>
          <StatTile
            label="trials"
            value={s.trials}
            precision={0}
          />
          <StatTile
            label="Σ var, roulette"
            value={stats.totalVarMult}
            role="prediction"
            precision={3}
          />
          <StatTile
            label="Σ var, comb"
            value={stats.totalVarComb}
            role="posterior"
            precision={3}
          />
          <StatTile
            label="variance ratio"
            value={stats.totalVarComb > 1e-9 ? stats.totalVarMult / stats.totalVarComb : 0}
            unit="×"
            precision={2}
          />

          <DashboardPanel title="Mean offspring per particle — unbiasedness" span={2}>
            <BarChart
              series={[
                {
                  id: 'roulette',
                  role: 'prediction',
                  data: stats.meanMult.map((y, i) => ({ x: i + 1, y })),
                },
                {
                  id: 'comb',
                  role: 'posterior',
                  data: stats.meanComb.map((y, i) => ({ x: i + 1, y })),
                },
                {
                  id: 'M·wᵢ',
                  role: 'truth',
                  data: stats.expected.map((y, i) => ({ x: i + 1, y })),
                },
              ]}
              xLabel="particle"
              yLabel="mean offspring"
              height={230}
              ariaLabel="Grouped bar chart comparing measured mean offspring counts for the roulette and comb resamplers against the expected value M times w."
            />
          </DashboardPanel>

          <DashboardPanel title="Offspring variance — where they differ" span={2}>
            <BarChart
              series={[
                {
                  id: 'roulette',
                  role: 'prediction',
                  data: stats.varMult.map((y, i) => ({ x: i + 1, y })),
                },
                {
                  id: 'comb',
                  role: 'posterior',
                  data: stats.varComb.map((y, i) => ({ x: i + 1, y })),
                },
                {
                  id: 'theory',
                  role: 'truth',
                  data: stats.theoryMult.map((y, i) => ({ x: i + 1, y })),
                },
              ]}
              xLabel="particle"
              yLabel="var(offspring)"
              height={230}
              ariaLabel="Grouped bar chart of measured offspring variance for both resamplers, with the multinomial theoretical variance M w (1 minus w) shown for comparison."
            />
          </DashboardPanel>
        </Dashboard>
      </div>

      <ControlPanel columns={3} title="weights (renormalized automatically)">
        {weights.map((value, i) => (
          <Slider
            key={i}
            label={`w${i + 1}`}
            value={value}
            min={0.01}
            max={1}
            step={0.01}
            onChange={(v) =>
              setWeights((prev) => {
                const next = prev.slice();
                next[i] = v;
                return next;
              })
            }
          />
        ))}
        <Slider
          label="Draws M"
          value={m}
          min={5}
          max={100}
          step={1}
          onChange={(v) => setM(Math.round(v))}
          format={(v) => String(Math.round(v))}
          help="How many offspring the resampler must produce."
        />
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton onClick={() => batch(1000)} emphasis>
            Spin ×1000
          </ActionButton>
          <ActionButton onClick={() => applyPreset(CHAPTER_WEIGHTS)}>Chapter example</ActionButton>
          <ActionButton onClick={() => applyPreset(DEGENERATE_WEIGHTS)}>Degenerate</ActionButton>
          <ActionButton onClick={() => applyPreset(UNIFORM_WEIGHTS)}>Uniform</ActionButton>
        </ButtonRow>
      </div>

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

/* -------------------------------------------------------------------------- */
/* The wheel itself                                                            */
/* -------------------------------------------------------------------------- */

const R_OUT = 40;
const R_IN = 25;

/** Polar → cartesian on the wheel, with 0 at twelve o'clock going clockwise. */
function polar(t: number, r: number): [number, number] {
  const a = 2 * Math.PI * t - Math.PI / 2;
  return [50 + r * Math.cos(a), 50 + r * Math.sin(a)];
}

function arcPath(from: number, to: number): string {
  const [x1, y1] = polar(from, R_OUT);
  const [x2, y2] = polar(to, R_OUT);
  const [x3, y3] = polar(to, R_IN);
  const [x4, y4] = polar(from, R_IN);
  const large = to - from > 0.5 ? 1 : 0;
  return [
    `M ${x1} ${y1}`,
    `A ${R_OUT} ${R_OUT} 0 ${large} 1 ${x2} ${y2}`,
    `L ${x3} ${y3}`,
    `A ${R_IN} ${R_IN} 0 ${large} 0 ${x4} ${y4}`,
    'Z',
  ].join(' ');
}

/** Five steps of one hue would read as a heat scale; segments use tinted chrome. */
const SEGMENT_ALPHA = [0.9, 0.72, 0.54, 0.36, 0.2];

function Wheel({
  title,
  subtitle,
  weights,
  trial,
  color,
}: {
  title: string;
  subtitle: string;
  weights: number[];
  trial: Trial;
  color: string;
}) {
  const edges = useMemo(() => {
    const out = [0];
    let acc = 0;
    for (const x of weights) {
      acc += x;
      out.push(acc);
    }
    return out;
  }, [weights]);

  return (
    <div className="px-3 py-3">
      <p className="eyebrow m-0" style={{ color }}>
        {title}
      </p>
      <p className="m-0 font-ui text-[0.7rem] text-fd-muted-foreground">{subtitle}</p>

      <svg viewBox="0 0 100 100" className="mx-auto block w-full max-w-[220px]" role="img"
        aria-label={`${title}: a donut whose arcs are proportional to the five weights, with the pointers of the most recent draw marked on the rim.`}
      >
        {weights.map((_, i) => (
          <path
            key={i}
            d={arcPath(edges[i], edges[i + 1])}
            fill="var(--color-fd-primary)"
            fillOpacity={SEGMENT_ALPHA[i % SEGMENT_ALPHA.length]}
            stroke="var(--pr-canvas-bg)"
            strokeWidth={0.6}
          />
        ))}

        {trial.pointers.map((u, k) => {
          const t = ((u % 1) + 1) % 1;
          const [xa, ya] = polar(t, R_IN - 3);
          const [xb, yb] = polar(t, R_OUT + 5);
          return (
            <line key={k} x1={xa} y1={ya} x2={xb} y2={yb} stroke={color} strokeWidth={1.1} strokeLinecap="round" />
          );
        })}

        {weights.map((_, i) => {
          const [tx, ty] = polar((edges[i] + edges[i + 1]) / 2, (R_IN + R_OUT) / 2);
          return (
            <text
              key={i}
              x={tx}
              y={ty + 1.4}
              textAnchor="middle"
              style={{ fontSize: 4.4, fontWeight: 600 }}
              fill="var(--pr-canvas-bg)"
            >
              {i + 1}
            </text>
          );
        })}
      </svg>

      <div className="mt-1 flex justify-center gap-3 font-mono text-[0.7rem] tabular-nums">
        {trial.counts.map((c, i) => (
          <span key={i} style={{ color: c === 0 ? 'var(--pr-truth)' : color }}>
            {i + 1}:{c}
          </span>
        ))}
      </div>
      <p className="mt-1 text-center font-ui text-[0.65rem] text-fd-muted-foreground">
        offspring counts, last trial
      </p>
    </div>
  );
}
