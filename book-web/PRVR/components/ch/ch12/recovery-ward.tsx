'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile, type ChartMarker } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { ParticleFilter } from '@/lib/filters/pf';
import { SurpriseDetector, augmentedResample, dominantCluster } from '@/lib/localize/augmented-mcl';
import { prob } from '@/lib/prob/gaussian';
import { HALLWAY_1D } from '@/lib/sim/world';
import { Rng } from '@/lib/prob/rng';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w12.3 — the Recovery Ward.
 *
 * The w_fast / w_slow detector, alone in a room with nothing to distract from
 * it. The Hallway of Chapter 5 is one-dimensional, so the average measurement
 * likelihood is a number the reader can read off the chart and check against
 * the arithmetic in the text.
 *
 * Two buttons make the case. **Kidnap** teleports the robot: the likelihood
 * collapses and stays collapsed, w_fast dives under w_slow, and injection turns
 * on until the cloud has rebuilt itself. **Glitch** corrupts a single reading:
 * w_fast dips for one step and climbs straight back, and the filter — correctly
 * — does almost nothing. One event is evidence of divergence; the other is
 * noise, and the only thing separating them is the timescale.
 */

const { length: L, doors: DOORS } = HALLWAY_1D;
const M = 500;
const HORIZON = 90;
const SIGMA_Z = 0.12;
const STEP_U = 0.22;
const MOTION_SIGMA = 0.05;

/** Distance to the nearest doorway — the one thing this robot can sense. */
const doorRange = (x: number) => Math.min(...DOORS.map((d) => Math.abs(x - d)));

/**
 * p(z | x) for the door-proximity sensor: a sharp Gaussian on a uniform floor.
 *
 * The floor is not decoration. Without a z_rand term a single wild reading
 * drives every weight to *exactly* zero, w_avg underflows, and the detector
 * saturates at p_inject = 1 — the filter throws away its entire belief because
 * of one bad number. Every sensor model in Chapter 10 has this floor for the
 * same reason.
 */
const Z_HIT = 0.85;
const Z_RAND = 0.15;
const Z_MAX = 5;
const likelihoodAt = (x: number, z: number) =>
  Z_HIT * prob(z - doorRange(x), SIGMA_Z * SIGMA_Z) + Z_RAND / Z_MAX;

interface Params {
  alphaFast: number;
  alphaSlow: number;
}

interface Sample {
  t: number;
  wFast: number;
  wSlow: number;
  pInject: number;
  error: number;
}

interface State {
  rng: Rng;
  world: Rng;
  pf: ParticleFilter;
  detector: SurpriseDetector;
  truth: number;
  z: number;
  wAvg: number;
  pInject: number;
  injected: number;
  error: number;
  history: Sample[];
  events: { t: number; kind: 'kidnap' | 'glitch' }[];
  t: number;
}

const wrap = (x: number) => ((x % L) + L) % L;

function makeState(seed: number, params: Params): State {
  const rng = new Rng(seed);
  const pf = new ParticleFilter(
    Array.from({ length: M }, (_, i) => ({
      state: { x: wrap(((i + 0.5) / M) * L), y: 0, theta: 0 },
      weight: 1 / M,
    })),
  );
  return {
    rng,
    world: new Rng(seed * 31 + 7),
    pf,
    detector: new SurpriseDetector({ alphaFast: params.alphaFast, alphaSlow: params.alphaSlow }),
    truth: 1.0,
    z: 0,
    wAvg: 0,
    pInject: 0,
    injected: 0,
    error: 0,
    history: [],
    events: [],
    t: 0,
  };
}

export function RecoveryWard() {
  const [params, setParams] = useState<Params>({ alphaFast: 0.5, alphaSlow: 0.05 });
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const pendingRef = useRef<'kidnap' | 'glitch' | null>(null);

  const init = useCallback((seed: number) => makeState(seed, paramsRef.current), []);

  const step = useCallback((s: State, _tick: number): State => {
    const p = paramsRef.current;
    const { pf, rng, world, detector } = s;
    detector.params = { alphaFast: p.alphaFast, alphaSlow: p.alphaSlow };

    const next: State = { ...s, t: s.t + 1, injected: 0 };
    const events = [...s.events];

    // ---- the world -----------------------------------------------------
    let truth = wrap(s.truth + STEP_U + world.normal(0, MOTION_SIGMA));
    let glitch = false;
    if (pendingRef.current === 'kidnap') {
      truth = wrap(truth + L / 2 + world.uniform(-1, 1));
      events.push({ t: next.t, kind: 'kidnap' });
    } else if (pendingRef.current === 'glitch') {
      glitch = true;
      events.push({ t: next.t, kind: 'glitch' });
    }
    pendingRef.current = null;
    next.truth = truth;
    next.events = events.filter((e) => e.t > next.t - HORIZON);

    // ---- predict --------------------------------------------------------
    pf.predict((x) => ({ ...x, x: wrap(x.x + STEP_U + rng.normal(0, MOTION_SIGMA)) }));

    // ---- weight ---------------------------------------------------------
    // A single wild reading is exactly what the slow filter must survive.
    const z = glitch ? 3.2 : Math.max(0, doorRange(truth) + world.normal(0, SIGMA_Z));
    next.z = z;
    pf.correct((x) => likelihoodAt(x.x, z));

    // w_avg is the mean *unnormalized* weight — the empirical p(z | z₁:ₜ₋₁).
    // `ParticleFilter.correct` renormalizes, so we recompute the raw mean here.
    let wAvg = 0;
    for (const q of pf.particles) wAvg += likelihoodAt(q.state.x, z);
    wAvg /= pf.size;
    next.wAvg = wAvg;
    next.pInject = detector.update(wAvg);

    // The estimate is the dominant *cluster*, not the mean of the whole set:
    // with three door hypotheses alive the mean sits between the doors, where
    // the filter assigns almost no probability at all.
    const est = dominantCluster(pf.particles, 0.4).pose.x;
    next.error = Math.min(Math.abs(est - truth), L - Math.abs(est - truth));

    // ---- resample, with injection ---------------------------------------
    if (next.pInject > 0) {
      const out = augmentedResample(pf.particles, rng, next.pInject, (r) => ({
        x: r.uniform(0, L),
        y: 0,
        theta: 0,
      }));
      pf.particles = out.particles;
      next.injected = out.injected;
    } else {
      pf.resample(rng, 'lowVariance');
    }

    next.history = [
      ...s.history,
      {
        t: next.t,
        wFast: detector.wFast,
        wSlow: detector.wSlow,
        pInject: next.pInject,
        error: next.error,
      },
    ].slice(-HORIZON);
    return next;
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 6, initialSeed: 12 });
  const s = sim.state;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const yTop = sy(v, 0.72);
      const yBot = sy(v, 0.28);

      ctx.fillStyle = p.free;
      ctx.fillRect(sx(v, 0), yTop, sl(v, L), yBot - yTop);
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx(v, 0), yTop, sl(v, L), yBot - yTop);

      for (const d of DOORS) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(sx(v, d), yTop);
        ctx.lineTo(sx(v, d), yBot);
        ctx.stroke();
      }

      // The cloud. Injected particles are indistinguishable once drawn — the
      // point is the density, not the individuals.
      ctx.save();
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.5;
      for (const q of s.pf.particles) {
        ctx.beginPath();
        ctx.arc(sx(v, q.state.x), (yTop + yBot) / 2 + (q.weight * M - 1) * 2, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Truth, and the reading it produced: a ring of radius z around it.
      const tx = sx(v, s.truth);
      const mid = (yTop + yBot) / 2;
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(tx, yTop - 6);
      ctx.lineTo(tx, yBot + 6);
      ctx.stroke();
      ctx.restore();

      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(tx, mid);
      ctx.lineTo(sx(v, s.truth + s.z), mid);
      ctx.stroke();
      label(ctx, `z = ${s.z.toFixed(2)} m to nearest door`, tx + 6, mid - 12, p.measurement, {
        size: 10,
      });
      label(
        ctx,
        s.pInject > 0 ? `injecting ${(s.pInject * 100).toFixed(0)}% of M` : 'no injection',
        8,
        12,
        s.pInject > 0 ? p.prior : p.ink,
        { size: 10, weight: 600 },
      );
    },
    [s],
  );

  const series = useMemo(
    () => [
      {
        id: 'w_fast',
        role: 'measurement' as const,
        data: s.history.map((h) => ({ x: h.t, y: h.wFast })),
      },
      {
        id: 'w_slow',
        role: 'prior' as const,
        data: s.history.map((h) => ({ x: h.t, y: h.wSlow })),
      },
    ],
    [s.history],
  );

  const markers = useMemo<ChartMarker[]>(
    () =>
      s.events.map((e) => ({
        axis: 'x' as const,
        value: e.t,
        label: e.kind,
        role: e.kind === 'kidnap' ? ('prediction' as const) : ('truth' as const),
      })),
    [s.events],
  );

  const ratio = s.detector.wSlow > 0 ? s.detector.wFast / s.detector.wSlow : 1;

  return (
    <WidgetFrame
      id="w12.3"
      title="Recovery Ward"
      teaches="The kidnapping detector needs no map-specific threshold: it is a ratio of the same statistic measured at two speeds, and it switches itself off once the filter recovers."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          A one-dimensional robot with a door-proximity sensor. The green line is w<sub>fast</sub>,
          the average measurement likelihood over the last handful of steps; the blue line is
          w<sub>slow</sub>, the same quantity remembered over tens of steps. When the green line
          sits above the blue one, the filter is doing at least as well as it usually does, and the
          injection probability is exactly zero. The green line jitters even when nothing is wrong —
          how well a door-proximity reading can be explained genuinely depends on where the robot is
          — and the detector does occasionally spend a few percent of the cloud on that jitter. That
          is the honest price of having no tuned threshold.
          <br />
          <strong>What to try.</strong> Press <em>Kidnap</em>: the green line falls off a cliff, the
          gap opens, particles rain in uniformly, and the gap closes by itself once the cloud has
          found the robot again — no threshold was tuned anywhere. Now press <em>Glitch</em>, which
          corrupts one single reading. Green dips hard for one step and then climbs straight back,
          so the injection is a one-step spike instead of a sustained flood: the smoothing bounds
          the damage of an outlier, it does not eliminate it, which is why a deployed system also
          rejects outliers <em>before</em> the filter sees them. Finally, set
          α<sub>fast</sub> = α<sub>slow</sub> and try both again. The detector is now blind, because
          the two lines are the same line.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.2, maxX: L + 0.2, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, s]}
        aspect={4.2}
        padding={0}
        ariaLabel="A one-dimensional corridor with three doors, a particle cloud, and the true robot position marked with a dashed line."
      />

      <div className="px-3 pt-3">
        <LineChart
          series={series}
          xLabel="filter step"
          yLabel="average likelihood"
          height={210}
          markers={markers}
          yMin={0}
          ariaLabel="Two exponential moving averages of the measurement likelihood: a fast one that dives when the robot is kidnapped, and a slow one that barely moves."
        />
      </div>

      <div className="px-3 pb-3">
        <Dashboard columns={4}>
          <StatTile
            label="w_fast"
            value={s.detector.wFast}
            role="measurement"
            precision={3}
            sparkline={s.history.map((h) => h.wFast)}
          />
          <StatTile
            label="w_slow"
            value={s.detector.wSlow}
            role="prior"
            precision={3}
            sparkline={s.history.map((h) => h.wSlow)}
          />
          <StatTile
            label="1 − w_fast/w_slow"
            value={`${(s.pInject * 100).toFixed(0)}%`}
            unit={`ratio ${ratio.toFixed(2)}`}
            role="posterior"
          />
          <StatTile
            label="|cluster − truth|"
            value={s.error}
            unit="m"
            precision={2}
            role="truth"
            sparkline={s.history.map((h) => h.error)}
          />
        </Dashboard>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="α_fast"
          role="measurement"
          value={params.alphaFast}
          min={0.05}
          max={0.9}
          step={0.05}
          onChange={(v) => setParams((q) => ({ ...q, alphaFast: v }))}
          help="How quickly the short-term average forgets. This is the headline knob: it sets how fast surprise is noticed."
        />
        <Slider
          label="α_slow"
          role="prior"
          value={params.alphaSlow}
          min={0.005}
          max={0.5}
          step={0.005}
          onChange={(v) => setParams((q) => ({ ...q, alphaSlow: v }))}
          help="Thrun's algorithm requires 0 ≤ α_slow ≪ α_fast. Set them equal and the detector goes blind."
        />
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2.5">
        <ButtonRow>
          <ActionButton onClick={() => (pendingRef.current = 'kidnap')} emphasis>
            Kidnap!
          </ActionButton>
          <ActionButton onClick={() => (pendingRef.current = 'glitch')}>
            Glitch one reading
          </ActionButton>
          <span className="font-ui text-[0.7rem] text-fd-muted-foreground">
            injected last step: {s.injected}
          </span>
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
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
