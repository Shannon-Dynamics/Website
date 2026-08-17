'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { Transport } from '@/components/sim/controls';
import { LineChart, type ChartMarker, type LineChartSeries } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import type { Palette, Viewport } from '@/lib/sim/draw';
import { AutonomyStack, BASE_DT } from '@/lib/capstone/stack';
import { drawScene, modeLabel } from './shared';

/**
 * w26.3 — Failure Theater.
 *
 * One tab per row of the assumption-to-failure table in Derivation F1. Each
 * runs the real stack from a warm start, injects one sabotage at a fixed time,
 * and plots the statistic that is supposed to notice. Nothing is recorded and
 * nothing is faked: if the detector fails to fire, the reader sees it fail.
 */

type Scenario = 'Kidnap' | 'Walker' | 'Dropout';
const SCENARIOS: Scenario[] = ['Kidnap', 'Walker', 'Dropout'];

/**
 * Ticks of warm-up so the robot has a map worth breaking, run eight at a time
 * during the first couple of seconds rather than all at once on mount — a
 * blocking fast-forward is a visible stutter, and watching the map appear at
 * speed is more informative anyway.
 */
const WARMUP = 680;
const WARMUP_BATCH = 8;
const INJECT_TICK = 800;
const END_TICK = 1700;
const MAX_CALLS = Math.ceil(WARMUP / WARMUP_BATCH) + (END_TICK - WARMUP);

interface Meta {
  assumption: string;
  breaks: string;
  detector: string;
  seed: number;
}

const META: Record<Scenario, Meta> = {
  Kidnap: {
    assumption: 'The pose belief brackets the truth — the filter is tracking, not lost.',
    breaks: 'A teleport puts the truth outside every contour of the belief. No Gaussian correction can walk back from that.',
    detector: 'F4(a): dual-EMA scan-match fitness ratio ρ = w_fast / w_slow, alarm below 0.80.',
    seed: 42,
  },
  Walker: {
    assumption: 'The world is static, so every measurement is evidence about the map.',
    breaks: 'A person crosses in front of the LiDAR. Beams that used to reach a wall stop early, and a naive mapper writes the person into the floorplan permanently.',
    detector: 'Novelty: a beam whose endpoint lands in a cell the map calls confidently free. Those beams are withheld from mapping and injected into the controller’s distance field instead.',
    seed: 3,
  },
  Dropout: {
    assumption: 'Messages arrive. Every downstream task consumes a belief that is at most one period old.',
    breaks: 'The LiDAR stops for three seconds. Nothing produces an error; the estimator simply runs open loop, and σ_pose — and with it the F2 margin — starts to grow.',
    detector: 'F4(c): a watchdog on message age, tripping at three nominal periods (0.30 s). Watch both curves flatten once Rusty halts: stopping is the action that bounds σ.',
    seed: 11,
  },
};

interface State {
  stack: AutonomyStack;
  trailEst: { x: number; y: number }[];
  trailTruth: { x: number; y: number }[];
}

export function FailureTheater() {
  const [scenario, setScenario] = useState<Scenario>('Kidnap');
  const meta = META[scenario];

  const init = useCallback(
    (seed: number): State => ({ stack: new AutonomyStack({ seed }), trailEst: [], trailTruth: [] }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      const budget = s.stack.tick < WARMUP ? WARMUP_BATCH : 1;
      for (let n = 0; n < budget; n++) {
        if (s.stack.tick === INJECT_TICK) {
          if (scenario === 'Kidnap') s.stack.kidnap();
          else if (scenario === 'Walker') s.stack.spawnWalker(0.8, 18);
          else s.stack.dropSensor(3);
        }
        s.stack.step();
      }
      if (s.stack.tick % 4 === 0) {
        s.trailEst.push({ x: s.stack.belief.mean.x, y: s.stack.belief.mean.y });
        s.trailTruth.push({ x: s.stack.truth.x, y: s.stack.truth.y });
        if (s.trailEst.length > 500) {
          s.trailEst.shift();
          s.trailTruth.shift();
        }
      }
      return { ...s };
    },
    [scenario],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 40,
    initialSeed: meta.seed,
    maxTicks: MAX_CALLS,
    loop: true,
  });
  const st = sim.state.stack;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      drawScene(ctx, v, p, sim.state.stack, {
        showTruth: true,
        showRollouts: true,
        showFrontiers: false,
        showScan: true,
        showPath: true,
        trailEst: sim.state.trailEst,
        trailTruth: sim.state.trailTruth,
      });
    },
    [sim.state],
  );

  // Nivo is not asked to re-render at forty frames a second; the chart refreshes
  // about five times a second, which is faster than anyone can read it.
  const bucket = Math.floor(sim.tick / 8);
  const { series, markers, yLabel } = useMemo(() => {
    const h = st.history.filter((s) => s.t > (INJECT_TICK - 160) * BASE_DT);
    if (scenario === 'Kidnap') {
      return {
        series: [{ id: 'ρ = w_fast / w_slow', role: 'posterior', data: h.map((s) => ({ x: s.t, y: s.rho })) }] as LineChartSeries[],
        markers: [{ axis: 'y', value: 0.8, label: 'alarm', role: 'prediction' }] as ChartMarker[],
        yLabel: 'fitness ratio ρ',
      };
    }
    if (scenario === 'Walker') {
      return {
        series: [{ id: 'novel beams', role: 'measurement', data: h.map((s) => ({ x: s.t, y: s.novel })) }] as LineChartSeries[],
        markers: [{ axis: 'y', value: 3, label: 'cluster threshold', role: 'prediction' }] as ChartMarker[],
        yLabel: 'beams the map cannot explain',
      };
    }
    return {
      series: [
        { id: 'σ_pose', role: 'posterior', data: h.map((s) => ({ x: s.t, y: s.sigma })) },
        { id: 'F2 margin', role: 'prediction', data: h.map((s) => ({ x: s.t, y: s.margin })) },
      ] as LineChartSeries[],
      markers: [] as ChartMarker[],
      yLabel: 'metres',
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario, bucket, st]);

  const pins = useMemo(
    () =>
      st.events
        .filter((e) => e.kind !== 'ModeSwitch' && e.kind !== 'GoalSelected' && e.kind !== 'GoalReached')
        .slice(-5),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [bucket, st],
  );

  const injectIn = Math.max(0, (INJECT_TICK - st.tick) * BASE_DT).toFixed(1);

  return (
    <WidgetFrame
      id="w26.3"
      title="Failure Theater"
      teaches="Recovery is not luck and not magic: it is a named statistic crossing a threshold, followed by a mode that has a plan."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Each tab breaks one assumption from the table in the Foundation section, and plots the
          statistic that is meant to notice. The run loops, so you can watch the same failure
          several times; the ground-truth overlay is on here (gray dashed) precisely so you can see
          how wrong the robot is before it knows. Watch the <em>Kidnap</em> tab twice: once looking
          at the canvas, once looking only at ρ. The second viewing is the one that matters, because
          on real hardware the canvas does not exist.
        </>
      }
    >
      <div className="flex border-b border-fd-border">
        {SCENARIOS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => {
              setScenario(s);
              sim.reseed(META[s].seed);
            }}
            aria-pressed={scenario === s}
            className={`flex-1 px-2 py-2 font-ui text-xs font-medium transition-colors ${
              scenario === s ? 'bg-fd-accent text-fd-foreground' : 'text-fd-muted-foreground hover:bg-fd-accent/50'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:divide-x lg:divide-fd-border">
        <div>
          <SimCanvas
            world={APARTMENT.bounds}
            draw={draw}
            deps={[sim.tick, sim.state, scenario]}
            aspect={12 / 9}
            padding={0.2}
            ariaLabel="The mission scene during a deliberately injected failure, with ground truth shown as a dashed gray outline beside the robot's own belief."
          />
          <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
            <Cell label="mode" value={modeLabel(st.mode).split(' ')[0]} />
            <Cell label="|est − truth|" value={`${st.error().toFixed(2)} m`} />
            <Cell
              label="sabotage"
              value={st.tick < INJECT_TICK ? `in ${injectIn} s` : 'injected'}
            />
          </div>
        </div>

        <div className="border-t border-fd-border lg:border-t-0">
          <div className="px-3 pt-3">
            <dl className="space-y-2">
              <Item k="assumption" v={meta.assumption} />
              <Item k="what breaks it" v={meta.breaks} />
              <Item k="detector" v={meta.detector} accent />
            </dl>
          </div>

          <div className="px-1 pt-1">
            <LineChart
              series={series}
              xLabel="simulation time (s)"
              yLabel={yLabel}
              markers={markers}
              height={190}
              legend={series.length > 1}
              caption={null}
            />
          </div>

          <div className="border-t border-fd-border px-3 py-2">
            <p className="eyebrow mb-1">events</p>
            <ul className="space-y-0.5 font-mono text-[0.66rem] leading-tight">
              {pins.length === 0 ? (
                <li className="text-fd-muted-foreground">nothing yet</li>
              ) : (
                pins.map((e, i) => (
                  <li key={`${e.t}-${i}`} className="flex gap-2">
                    <span className="shrink-0 opacity-60">{e.t.toFixed(1)}</span>
                    <span
                      className="shrink-0"
                      style={{
                        color:
                          e.kind === 'ChaosInjected'
                            ? 'var(--pr-prediction)'
                            : e.kind === 'RelocalizeConverged'
                              ? 'var(--pr-measurement)'
                              : 'var(--pr-posterior)',
                      }}
                    >
                      {e.kind}
                    </span>
                    <span className="truncate opacity-60">{e.detail}</span>
                  </li>
                ))
              )}
            </ul>
          </div>
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

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{label}</div>
      <div className="font-mono text-[0.78rem] tabular-nums">{value}</div>
    </div>
  );
}

function Item({ k, v, accent }: { k: string; v: string; accent?: boolean }) {
  return (
    <div>
      <dt className="eyebrow">{k}</dt>
      <dd
        className="font-prose text-[0.82rem] leading-snug"
        style={accent ? { color: 'var(--pr-prediction)' } : undefined}
      >
        {v}
      </dd>
    </div>
  );
}
