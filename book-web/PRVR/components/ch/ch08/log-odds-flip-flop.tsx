'use client';

import Link from 'next/link';
import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart, StatTile, Dashboard } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';

/**
 * w8.4 — the Log-Odds Flip-Flop.
 *
 * One binary cell — "is this bit of the Hallway occupied?" — fed a stream of
 * noisy detections. In probability view the update is a product that crawls
 * asymptotically toward 1. In log-odds view the identical evidence is literal
 * addition of signed increments, which is why occupancy grid mapping in
 * Chapter 13 is a `+=` per cell and nothing more.
 *
 * The flip-flop: the world changes at t = 40. Without clamping, a saturated
 * cell needs almost as many contradicting readings as it took to saturate.
 * With clamping it recovers in a handful.
 */

const HORIZON = 70;
const FLIP_AT = 40;
/** Reliability of the detector: p(reports correctly). */
const SENSOR = 0.9;

interface Params {
  /** The headline knob: p(occupied | one hit) under the inverse model. */
  confidence: number;
  clamp: boolean;
  clampBound: number;
  probabilityView: boolean;
}

interface Sample {
  t: number;
  l: number;
  p: number;
}

interface State {
  rng: Rng;
  l: number;
  increment: number;
  lastZ: 'hit' | 'miss' | null;
  history: Sample[];
  manual: number;
}

const probOf = (l: number) => 1 - 1 / (1 + Math.exp(l));

export function LogOddsFlipFlop() {
  const [params, setParams] = useState<Params>({
    confidence: 0.7,
    clamp: false,
    clampBound: 4,
    probabilityView: false,
  });
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      // ℓ₀ = log(0.5 / 0.5) = 0: a cell we know nothing about.
      l: 0,
      increment: 0,
      lastZ: null,
      history: [{ t: 0, l: 0, p: 0.5 }],
      manual: 0,
    }),
    [],
  );

  /** One evidence update. `hit` is the reading, not the truth. */
  const fold = useCallback((s: State, hit: boolean, t: number): State => {
    const p = paramsRef.current;
    const c = p.confidence;
    // The *inverse* model p(x | z), which is the natural parameterization when
    // the measurement is richer than the binary state.
    const pOcc = hit ? c : 1 - c;
    // ℓₜ = ℓₜ₋₁ + log(p(x|z)/(1−p(x|z))) − ℓ₀, and ℓ₀ = 0 for a uniform prior.
    const increment = Math.log(pOcc / (1 - pOcc));
    let l = s.l + increment;
    if (p.clamp) l = Math.max(-p.clampBound, Math.min(p.clampBound, l));
    const history = [...s.history, { t, l, p: probOf(l) }].slice(-HORIZON - 1);
    return { ...s, l, increment, lastZ: hit ? 'hit' : 'miss', history };
  }, []);

  const step = useCallback(
    (s: State, tick: number): State => {
      // The world itself flips at t = 40: the cell was occupied, now it is free.
      const occupied = tick < FLIP_AT;
      const correct = s.rng.next() < SENSOR;
      const hit = occupied === correct;
      return fold(s, hit, tick + 1);
    },
    [fold],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 8,
    maxTicks: HORIZON,
    loop: true,
    initialSeed: 5,
  });
  const { setState, pause } = sim;

  const crank = useCallback(
    (hit: boolean) => {
      pause();
      setState((s) => fold(s, hit, (s.history[s.history.length - 1]?.t ?? 0) + 1));
    },
    [pause, setState, fold],
  );

  const s = sim.state;
  const last = s.history[s.history.length - 1];

  const series = useMemo(
    () => [
      {
        id: params.probabilityView ? 'bel(x) = p(occupied)' : 'log odds ℓₜ',
        role: 'posterior' as const,
        data: s.history.map((h) => ({ x: h.t, y: params.probabilityView ? h.p : h.l })),
      },
    ],
    [s.history, params.probabilityView],
  );

  const markers = useMemo(() => {
    if (params.probabilityView) {
      return [{ axis: 'y' as const, value: 0.5, label: 'ignorance', role: 'truth' as const }];
    }
    const out = [{ axis: 'y' as const, value: 0, label: 'ℓ = 0', role: 'truth' as const }];
    if (params.clamp) {
      out.push(
        { axis: 'y' as const, value: params.clampBound, label: '+ℓmax', role: 'truth' as const },
        { axis: 'y' as const, value: -params.clampBound, label: '−ℓmax', role: 'truth' as const },
      );
    }
    return out;
  }, [params.probabilityView, params.clamp, params.clampBound]);

  return (
    <WidgetFrame
      id="w8.4"
      title="Log-Odds Flip-Flop"
      teaches="A static state still needs a filter — and in log odds that filter is one addition, which is why occupancy mapping is cheap."
      colorKey={['posterior', 'truth']}
      caption={
        <>
          One cell, one binary question, a noisy detector. The world is occupied until{' '}
          <code>t = 40</code> and free afterwards. In log-odds view every reading contributes the{' '}
          <em>same</em> signed increment, ±log(c / (1 − c)) — the curve is a random walk with drift,
          and the drift reverses cleanly at the flip. Switch to probability view and the identical
          numbers look like a saturating product that is pinned to 1 long before the flip, which is
          precisely the numerical problem log odds exists to remove. Now turn{' '}
          <strong>clamping off</strong> and re-run: the cell accumulates 40 units of confidence and
          then refuses to change its mind for another 40. Clamp it at ±4 and it recovers in five
          readings. Run one of these per grid cell and you have occupancy grid mapping —{' '}
          <Link href="/chapters/ch13-occupancy-grids">Chapter 13</Link> is this box, tiled.
        </>
      }
    >
      <div className="px-3 pt-3">
        <LineChart
          series={series}
          xLabel="t"
          yLabel={params.probabilityView ? 'bel(x)' : 'log odds ℓₜ'}
          height={250}
          markers={markers}
          yMin={params.probabilityView ? 0 : 'auto'}
          yMax={params.probabilityView ? 1 : 'auto'}
          ariaLabel="A time series of the log odds of one binary cell, rising while the detector reports occupied and falling after the world flips to free."
        />
      </div>

      <div className="px-3 pb-3">
        <Dashboard columns={4}>
          <StatTile
            label={params.probabilityView ? 'ℓₜ' : 'bel(x)'}
            value={params.probabilityView ? (last?.l ?? 0) : (last?.p ?? 0.5)}
            role="posterior"
            precision={3}
            sparkline={s.history.map((h) => (params.probabilityView ? h.l : h.p))}
          />
          <StatTile
            label="last increment"
            value={s.increment}
            precision={3}
            trend={s.increment}
            trendLabel={s.lastZ ?? 'no reading yet'}
          />
          <StatTile
            label="odds"
            value={Math.exp(last?.l ?? 0)}
            unit=": 1"
            precision={2}
          />
          <StatTile
            label="world"
            value={sim.tick < FLIP_AT ? 'occupied' : 'free'}
            role="truth"
          />
        </Dashboard>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Inverse-model confidence c"
          role="measurement"
          value={params.confidence}
          min={0.52}
          max={0.95}
          step={0.01}
          onChange={(v) => setParams((q) => ({ ...q, confidence: v }))}
          help="p(occupied | one hit). Every reading is worth ±log(c/(1−c)) — no more, no less."
        />
        <Slider
          label="Clamp ℓmax"
          value={params.clampBound}
          min={1}
          max={20}
          step={0.5}
          onChange={(v) => setParams((q) => ({ ...q, clampBound: v }))}
          help="The practical guard Chapter 13 inherits: bound the confidence a cell may accumulate."
        />
        <Toggle
          label="Clamping on"
          checked={params.clamp}
          onChange={(v) => setParams((q) => ({ ...q, clamp: v }))}
        />
        <Toggle
          label="Probability view"
          role="posterior"
          checked={params.probabilityView}
          onChange={(v) => setParams((q) => ({ ...q, probabilityView: v }))}
        />
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton onClick={() => crank(true)}>Feed a hit</ActionButton>
          <ActionButton onClick={() => crank(false)}>Feed a miss</ActionButton>
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
