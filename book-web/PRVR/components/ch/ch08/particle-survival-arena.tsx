'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { ParticleFilter, type Particle } from '@/lib/filters/pf';
import { countOccupiedBins1D, kldResampleIndices, kldSampleSize } from '@/lib/filters/kld';
import { HALLWAY_1D, hallwayMeasurementLikelihood, isDoorAt } from '@/lib/sim/world';
import { Rng } from '@/lib/prob/rng';
import { clear, drawParticles, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w8.1 — the Particle Survival Arena.
 *
 * The whole particle filter, one phase at a time, on the Hallway from
 * Chapter 5. Dust everywhere → three clouds (one per door) → one cloud. The
 * three phases are separated on the clock so the reader can watch propagate
 * smear, weight sort, and resample select — and see that the last of those is
 * the one that throws information away.
 *
 * Everything here is the real `ParticleFilter` from `lib/filters/pf.ts` and the
 * real KLD bound from `lib/filters/kld.ts`; the arena only draws.
 */

const { length: L, doors: DOORS, doorWidth: DOOR_W } = HALLWAY_1D;
const DENSITY_BINS = 120;
/** Bin width for the KLD occupancy count, in metres. */
const KLD_BIN = 0.25;

type Phase = 'propagate' | 'weight' | 'resample';

interface Params {
  /** The headline parameter. */
  m: number;
  /** p(reports door | at door): 0.5 is a coin, 0.99 is a razor. */
  sharpness: number;
  motionNoise: number;
  kld: boolean;
}

interface Snapshot {
  ess: number;
  m: number;
  error: number;
}

interface State {
  pf: ParticleFilter;
  rng: Rng;
  truth: number;
  phase: Phase;
  sawDoor: boolean | null;
  ess: number;
  resampled: boolean;
  kldTarget: number | null;
  occupied: number;
  history: Snapshot[];
  kidnaps: number;
}

const wrap = (x: number) => ((x % L) + L) % L;

/** Global localization: M particles spread evenly over the corridor, then jittered. */
function scatter(m: number, rng: Rng): Particle[] {
  return Array.from({ length: m }, (_, i) => ({
    state: { x: wrap(((i + 0.5) / m) * L + rng.normal(0, 0.04)), y: 0, theta: 0 },
    weight: 1 / m,
  }));
}

export function ParticleSurvivalArena() {
  const [params, setParams] = useState<Params>({
    m: 400,
    sharpness: 0.85,
    motionNoise: 0.14,
    kld: false,
  });
  // The step function reads parameters through a ref so that dragging a slider
  // never restarts the run — the reader can change M mid-flight.
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const init = useCallback((seed: number): State => {
    const rng = new Rng(seed);
    const pf = new ParticleFilter(scatter(paramsRef.current.m, rng));
    return {
      pf,
      rng,
      truth: 0.55,
      phase: 'resample',
      sawDoor: null,
      ess: pf.size,
      resampled: false,
      kldTarget: null,
      occupied: countOccupiedBins1D(pf.particles.map((p) => p.state.x), KLD_BIN),
      history: [],
      kidnaps: 0,
    };
  }, []);

  const step = useCallback((s: State, tick: number): State => {
    const p = paramsRef.current;
    const { pf, rng } = s;
    const phase: Phase = (['propagate', 'weight', 'resample'] as const)[tick % 3];

    let truth = s.truth;
    let sawDoor = s.sawDoor;
    let resampled = false;
    let kldTarget = s.kldTarget;

    if (phase === 'propagate') {
      // A fixed-size filter honours the M slider here, by drawing a new
      // population of the requested size from the current weighted cloud.
      if (!p.kld && pf.size !== p.m) {
        const w = pf.particles.map((q) => q.weight);
        pf.particles = Array.from({ length: p.m }, () => {
          const src = pf.particles[rng.choice(w)];
          return { state: { ...src.state }, weight: 1 / p.m };
        });
      }

      const u = 0.42;
      truth = wrap(truth + u + rng.normal(0, p.motionNoise));
      // Prediction = sampling the motion model once per particle. This is the
      // proposal distribution; nothing about z has been used yet.
      pf.predict((st) => ({ ...st, x: wrap(st.x + u + rng.normal(0, p.motionNoise)) }));
      sawDoor = null;
    } else if (phase === 'weight') {
      const truthAtDoor = isDoorAt(truth);
      sawDoor = rng.next() < (truthAtDoor ? p.sharpness : 1 - p.sharpness);
      // The importance weight *is* the measurement likelihood — the single
      // cancellation the derivation in this chapter turns on.
      pf.correct((st) => hallwayMeasurementLikelihood(st.x, sawDoor as boolean, p.sharpness));
    } else {
      const ess = pf.effectiveSampleSize();
      if (p.kld) {
        // KLD sampling: keep drawing until the bound for the number of
        // *occupied* bins is met. The population size is an output, not a knob.
        const w = pf.particles.map((q) => q.weight);
        const idx = kldResampleIndices(
          w,
          (i) => Math.floor(pf.particles[i].state.x / KLD_BIN),
          () => rng.choice(w),
          { minParticles: 30, maxParticles: p.m },
        );
        pf.particles = idx.map((i) => ({
          state: { ...pf.particles[i].state },
          weight: 1 / idx.length,
        }));
        kldTarget = idx.length;
        resampled = true;
      } else if (ess < pf.size / 2) {
        // Thrun Table 4.4. Resample only when the weights have degenerated:
        // resampling a healthy population destroys diversity for nothing.
        pf.resample(rng, 'lowVariance');
        resampled = true;
      }
    }

    const ess = pf.effectiveSampleSize();
    const mean = pf.mean().x;
    const error = Math.min(Math.abs(mean - truth), L - Math.abs(mean - truth));
    const occupied = countOccupiedBins1D(pf.particles.map((q) => q.state.x), KLD_BIN);
    const history = [...s.history, { ess, m: pf.size, error }].slice(-90);

    return {
      pf,
      rng,
      truth,
      phase,
      sawDoor,
      ess,
      resampled,
      kldTarget,
      occupied,
      history,
      kidnaps: s.kidnaps,
    };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 3, initialSeed: 8 });
  const { setState } = sim;

  const kidnap = useCallback(() => {
    setState((s) => ({ ...s, truth: wrap(s.truth + 4.3), kidnaps: s.kidnaps + 1 }));
  }, [setState]);

  const deprivationPreset = useCallback(() => {
    setParams({ m: 30, sharpness: 0.99, motionNoise: 0.14, kld: false });
  }, []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { pf, truth, phase, sawDoor, resampled } = sim.state;

      // ---- corridor -----------------------------------------------------
      const yTop = sy(v, 0.96);
      const yBot = sy(v, 0.8);
      ctx.fillStyle = p.free;
      ctx.fillRect(sx(v, 0), yTop, sl(v, L), yBot - yTop);
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx(v, 0), yTop, sl(v, L), yBot - yTop);
      for (const d of DOORS) {
        ctx.fillStyle = p.bg;
        ctx.fillRect(sx(v, d - DOOR_W / 2), yTop - 1, sl(v, DOOR_W), yBot - yTop + 2);
      }

      const ry = (yTop + yBot) / 2;
      ctx.fillStyle = p.truth;
      ctx.beginPath();
      ctx.arc(sx(v, truth), ry, 5.5, 0, Math.PI * 2);
      ctx.fill();

      // The weighted mean: a number the filter can report that no particle
      // need be anywhere near. On a trimodal belief it points at empty floor.
      const mean = pf.mean().x;
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, mean), yTop - 4);
      ctx.lineTo(sx(v, mean), yBot + 4);
      ctx.stroke();
      ctx.setLineDash([]);
      label(ctx, 'weighted mean', sx(v, mean) + 5, yTop - 8, p.posterior, { size: 9 });

      if (sawDoor !== null) {
        label(
          ctx,
          sawDoor ? 'z = DOOR' : 'z = no door',
          sx(v, truth) + 9,
          ry,
          p.measurement,
          { size: 10, weight: 600 },
        );
      }

      // ---- the particle band --------------------------------------------
      const phaseColor =
        phase === 'propagate' ? p.prediction : phase === 'weight' ? p.measurement : p.posterior;

      // Lay the cloud out as a band: x is the hypothesis, y is a deterministic
      // golden-ratio offset so overlapping hypotheses stay countable.
      const band = pf.particles.map((q, i) => ({
        state: { x: q.state.x, y: 0.56 + (((i * 0.6180339887) % 1) - 0.5) * 0.16, theta: 0 },
        weight: q.weight,
      }));
      drawParticles(ctx, v, band, phaseColor, { showHeading: false, maxRadius: 3.4 });

      // ---- weighted density ----------------------------------------------
      const bins = new Array<number>(DENSITY_BINS).fill(0);
      for (const q of pf.particles) {
        const b = Math.min(DENSITY_BINS - 1, Math.floor((q.state.x / L) * DENSITY_BINS));
        bins[b] += q.weight;
      }
      const peak = Math.max(...bins, 1e-9);
      const baseY = sy(v, 0.06);
      const topY = sy(v, 0.44);
      const h = baseY - topY;
      const barW = sl(v, L / DENSITY_BINS);
      ctx.fillStyle = phaseColor;
      for (let i = 0; i < DENSITY_BINS; i++) {
        if (bins[i] <= 0) continue;
        const bh = (bins[i] / peak) * h;
        ctx.fillRect(sx(v, ((i + 0.5) / DENSITY_BINS) * L) - barW / 2, baseY - bh, Math.max(barW - 0.6, 1), bh);
      }

      // The likelihood the weights came from, as a curve — a function of state,
      // not a distribution over it.
      if (sawDoor !== null) {
        const pk = paramsRef.current.sharpness;
        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (let i = 0; i < DENSITY_BINS; i++) {
          const x = ((i + 0.5) / DENSITY_BINS) * L;
          const lk = hallwayMeasurementLikelihood(x, sawDoor, pk) / Math.max(pk, 1 - pk);
          const y = baseY - lk * h * 0.9;
          if (i === 0) ctx.moveTo(sx(v, x), y);
          else ctx.lineTo(sx(v, x), y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), baseY);
      ctx.lineTo(sx(v, L), baseY);
      ctx.stroke();

      const caption =
        phase === 'propagate'
          ? 'PROPAGATE  —  sample the motion model (the proposal)'
          : phase === 'weight'
            ? 'WEIGHT  —  w ∝ p(z | x)  (the importance ratio)'
            : resampled
              ? 'RESAMPLE  —  survival of the fittest'
              : 'RESAMPLE SKIPPED  —  M_eff still above M/2';
      label(ctx, caption, sx(v, 0.05), topY - 12, phaseColor, { size: 11, weight: 600 });
      label(ctx, `${pf.size} particles`, sx(v, L - 0.05), topY - 12, p.truth, {
        size: 10,
        align: 'right',
      });
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const s = sim.state;
    return {
      ess: s.ess,
      count: s.pf.size,
      error: s.history[s.history.length - 1]?.error ?? 0,
      occupied: s.occupied,
      bound: kldSampleSize(s.occupied, { minParticles: 30, maxParticles: params.m }),
    };
  }, [sim.state, params.m]);

  return (
    <WidgetFrame
      id="w8.1"
      title="Particle Survival Arena"
      teaches="A particle set is a distribution, not a list of guesses — and resampling is selection pressure, with all the side effects that implies."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Three phases, one per tick, so you can see each in isolation.{' '}
          <strong style={{ color: 'var(--pr-prediction)' }}>Propagate</strong> smears the cloud
          through the motion model. <strong style={{ color: 'var(--pr-measurement)' }}>Weight</strong>{' '}
          multiplies each particle by the green likelihood — nothing moves, the ticks just change
          size. <strong style={{ color: 'var(--pr-posterior)' }}>Resample</strong> deletes the small
          ones and clones the big ones, and only fires when the effective sample size has fallen
          below <em>M</em>/2. Notice that the purple weighted mean spends the early ticks pointing at
          bare corridor between the three door hypotheses: the mean of a multimodal belief is a
          place the robot has never been. Then press <em>Deprivation preset</em> — 30 particles and a
          razor-sharp sensor — and watch the filter converge, confidently, onto the wrong door.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.25, maxX: L + 0.25, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state, params.sharpness]}
        aspect={2.3}
        padding={0}
        ariaLabel="A corridor with three doors, a band of particles beneath it, and a weighted density histogram at the bottom. The particle cloud spreads when the robot moves, is reweighted by the door sensor, and collapses toward the surviving hypotheses after each resample."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="particles M" value={String(stats.count)} />
        <Stat
          label="M_eff"
          value={stats.ess.toFixed(1)}
          alert={stats.ess < stats.count / 2}
        />
        <Stat label="occupied bins k" value={String(stats.occupied)} />
        <Stat
          label={params.kld ? 'KLD bound' : '|mean − truth|'}
          value={params.kld ? String(stats.bound) : `${stats.error.toFixed(2)} m`}
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Particle count M"
          value={params.m}
          min={20}
          max={2000}
          step={10}
          onChange={(v) => setParams((q) => ({ ...q, m: v }))}
          format={(v) => String(Math.round(v))}
          help="The headline knob. In KLD mode this becomes the ceiling, not the target."
        />
        <Slider
          label="Sensor sharpness"
          role="measurement"
          value={params.sharpness}
          min={0.55}
          max={0.99}
          step={0.01}
          onChange={(v) => setParams((q) => ({ ...q, sharpness: v }))}
          help="p(reports door | at a door). Sharper sensors kill more particles per step."
        />
        <Slider
          label="Motion noise σ"
          role="prediction"
          value={params.motionNoise}
          min={0.01}
          max={0.5}
          step={0.01}
          unit="m"
          onChange={(v) => setParams((q) => ({ ...q, motionNoise: v }))}
          help="The only source of new diversity once resampling has started deleting it."
        />
        <Toggle
          label="KLD-adaptive M"
          checked={params.kld}
          onChange={(v) => setParams((q) => ({ ...q, kld: v }))}
        />
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton onClick={kidnap}>Kidnap Rusty</ActionButton>
          <ActionButton onClick={deprivationPreset}>Deprivation preset</ActionButton>
          <ActionButton onClick={() => setParams({ m: 400, sharpness: 0.85, motionNoise: 0.14, kld: false })}>
            Reset parameters
          </ActionButton>
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

function Stat({ label: l, value, alert }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={alert ? { color: 'var(--pr-prediction)' } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
