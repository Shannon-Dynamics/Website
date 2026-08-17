'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, drawSegments, label, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { Rng } from '@/lib/prob/rng';
import type { Point2 } from '@/lib/sim/world';
import { Prm, hybridSampler, type PlanResult } from '@/lib/plan/sampling';
import {
  PASSAGE_BOUNDS,
  PASSAGE_GOAL,
  PASSAGE_START,
  inPassage,
  makePassageCSpace,
  makePassageWorld,
  successProbability,
} from '@/lib/plan/passage';

/**
 * w20.2 — the Narrow Passage.
 *
 * Probabilistic completeness says the failure probability decays exponentially
 * in the number of samples — *for a query with clearance δ > 0*. This widget is
 * that caveat made physical: the exponent is proportional to the volume of the
 * passage, so shrinking the corridor does not make the planner slower, it makes
 * it fail. The bridge-test toggle shows that the fix is not more samples but
 * *better-placed* ones.
 */

const BUDGET = 180;
const PER_TICK = 10;
const SWEEP_WIDTHS = [0.35, 0.5, 0.65, 0.8, 1.0, 1.3, 1.7];
const SWEEP_TRIALS = 30;

interface State {
  prm: Prm;
  rng: Rng;
  drawn: number;
  phase: 'building' | 'scoring';
  hold: number;
  solution: PlanResult | null;
  trials: number;
  wins: number;
  trial: number;
}

export function NarrowPassage() {
  const [width, setWidth] = useState(0.65);
  const [bridge, setBridge] = useState(false);

  const cs = useMemo(() => makePassageCSpace(width), [width]);
  const world = useMemo(() => makePassageWorld(width), [width]);

  const init = useCallback(
    (seed: number): State => ({
      prm: new Prm(cs, {
        k: 8,
        maxEdgeLength: 1.6,
        sampler: bridge ? hybridSampler(cs, 0.6, 0.7) : undefined,
      }),
      rng: new Rng(seed),
      drawn: 0,
      phase: 'building',
      hold: 0,
      solution: null,
      trials: 0,
      wins: 0,
      trial: 0,
    }),
    [cs, bridge],
  );

  const step = useCallback(
    (s: State): State => {
      if (s.phase === 'building') {
        for (let i = 0; i < PER_TICK; i++) s.prm.step(s.rng);
        const drawn = s.drawn + PER_TICK;
        if (drawn < BUDGET) return { ...s, drawn };
        // Budget spent: answer the query and score the trial.
        const solution = s.prm.query(PASSAGE_START, PASSAGE_GOAL);
        return {
          ...s,
          drawn,
          phase: 'scoring',
          hold: 0,
          solution,
          trials: s.trials + 1,
          wins: s.wins + (solution ? 1 : 0),
        };
      }
      if (s.hold < 12) return { ...s, hold: s.hold + 1 };
      // Fresh roadmap, fresh seed: this is one Monte-Carlo trial of many.
      const trial = s.trial + 1;
      return {
        ...s,
        prm: new Prm(cs, {
          k: 8,
          maxEdgeLength: 1.6,
          sampler: bridge ? hybridSampler(cs, 0.6, 0.7) : undefined,
        }),
        rng: new Rng(2000 + trial),
        drawn: 0,
        phase: 'building',
        hold: 0,
        solution: null,
        trial,
      };
    },
    [cs, bridge],
  );

  const sim = useSimulation<State>({ init, step, fps: 14, initialSeed: 2000 });

  const resetRef = useRef(sim.reset);
  resetRef.current = sim.reset;
  useEffect(() => {
    resetRef.current();
  }, [width, bridge]);

  // The sweep is ~300 ms of Monte Carlo, so it runs after first paint.
  const [sweep, setSweep] = useState<{ uniform: number[]; bridge: number[] } | null>(null);
  useEffect(() => {
    let cancelled = false;
    const id = window.setTimeout(() => {
      const u: number[] = [];
      const b: number[] = [];
      for (const w of SWEEP_WIDTHS) {
        const space = makePassageCSpace(w);
        u.push(successProbability(w, SWEEP_TRIALS, { samples: BUDGET, bridge: false }, (s) => new Rng(s), space));
        b.push(successProbability(w, SWEEP_TRIALS, { samples: BUDGET, bridge: true }, (s) => new Rng(s), space));
      }
      if (!cancelled) setSweep({ uniform: u, bridge: b });
    }, 60);
    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { prm, solution } = sim.state;

      // Roadmap edges first, so milestones sit on top of them.
      ctx.save();
      ctx.strokeStyle = p.prior;
      ctx.globalAlpha = 0.28;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < prm.nodes.length; i++) {
        for (const j of prm.adj[i]) {
          if (j < i) continue;
          ctx.moveTo(sx(v, prm.nodes[i].x), sy(v, prm.nodes[i].y));
          ctx.lineTo(sx(v, prm.nodes[j].x), sy(v, prm.nodes[j].y));
        }
      }
      ctx.stroke();
      ctx.restore();

      // Milestones. The ones that landed inside the corridor are the whole game.
      ctx.save();
      for (const q of prm.nodes) {
        const inside = inPassage(q, width);
        ctx.fillStyle = inside ? p.measurement : p.prior;
        ctx.globalAlpha = inside ? 1 : 0.65;
        ctx.beginPath();
        ctx.arc(sx(v, q.x), sy(v, q.y), inside ? 3.4 : 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      if (solution) {
        ctx.save();
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(sx(v, solution.path[0].x), sy(v, solution.path[0].y));
        for (const q of solution.path.slice(1)) ctx.lineTo(sx(v, q.x), sy(v, q.y));
        ctx.stroke();
        ctx.restore();
      }

      drawSegments(ctx, v, world.walls, p.wall, 3);
      terminal(ctx, v, PASSAGE_START, p.truth, false);
      terminal(ctx, v, PASSAGE_GOAL, p.measurement, true);

      const status =
        sim.state.phase === 'building'
          ? `sampling  ${sim.state.drawn}/${BUDGET}`
          : solution
            ? `connected — cost ${solution.cost.toFixed(2)} m`
            : 'no path found at this budget';
      label(ctx, status, 14, 18, solution ? p.posterior : p.ink, { size: 11, weight: 700 });
    },
    [sim.state, width, world],
  );

  const rate = sim.state.trials > 0 ? sim.state.wins / sim.state.trials : 0;

  const series = useMemo(() => {
    if (!sweep) return [];
    return [
      {
        id: 'uniform sampling',
        role: 'prior' as const,
        data: SWEEP_WIDTHS.map((w, i) => ({ x: w, y: sweep.uniform[i] })),
      },
      {
        id: 'bridge sampling',
        role: 'measurement' as const,
        data: SWEEP_WIDTHS.map((w, i) => ({ x: w, y: sweep.bridge[i] })),
      },
    ];
  }, [sweep]);

  return (
    <WidgetFrame
      id="w20.2"
      title="The Narrow Passage"
      teaches="Uniform random sampling does not see everything: it sees a passage with probability proportional to the passage's volume."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Each run scatters {BUDGET} milestones (blue), wires each to its eight nearest neighbours,
          and then asks for a path from the gray start to the green goal. Milestones that landed
          inside the corridor are drawn green — they are the only ones that can possibly bridge the
          two rooms, and at a 0.35 m corridor there are usually none. The chart is the same
          experiment repeated {SWEEP_TRIALS} times at each width. Watch the blue curve fall off a
          cliff while the green one holds: <strong>bridge sampling</strong> keeps only the midpoints
          of segments whose two endpoints are both in collision, which is a geometric signature of a
          passage rather than a bet on its volume. Try the narrowest setting with bridge sampling
          off, then on.
        </>
      }
    >
      <SimCanvas
        world={PASSAGE_BOUNDS}
        draw={draw}
        deps={[sim.tick, sim.state, width]}
        aspect={10 / 6}
        padding={0.2}
        ariaLabel="Two rooms joined by a narrow corridor, with probabilistic roadmap milestones scattered across them. Milestones inside the corridor are highlighted; a path appears when the roadmap connects the two rooms."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="corridor" value={`${width.toFixed(2)} m`} />
        <Stat label="clear width" value={`${(width - 0.3).toFixed(2)} m`} />
        <Stat label="trials" value={String(sim.state.trials)} />
        <Stat label="live success" value={`${(100 * rate).toFixed(0)}%`} />
      </div>

      <div className="border-t border-fd-border px-3 py-3">
        {series.length > 0 ? (
          <LineChart
            series={series}
            xLabel="corridor width (m)"
            yLabel="P(roadmap answers the query)"
            height={220}
            yMin={0}
            yMax={1}
            markers={[{ axis: 'x', value: width, role: 'truth', label: 'you are here' }]}
            ariaLabel="Success probability against corridor width, for uniform sampling and bridge sampling. The uniform curve collapses toward zero as the corridor narrows while the bridge curve stays high until the corridor is nearly closed."
          />
        ) : (
          <p className="py-8 text-center font-ui text-xs text-fd-muted-foreground">
            running {SWEEP_WIDTHS.length * SWEEP_TRIALS * 2} seeded trials…
          </p>
        )}
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Corridor width"
          role="measurement"
          value={width}
          min={0.35}
          max={1.7}
          step={0.05}
          unit="m"
          onChange={setWidth}
          help="The robot's diameter is 0.30 m, so the clear width is this minus 0.30."
        />
        <Toggle label="Bridge-test sampling" role="prior" checked={bridge} onChange={setBridge} />
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

function terminal(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  q: Point2,
  color: string,
  filled: boolean,
) {
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(sx(v, q.x), sy(v, q.y), 7, 0, Math.PI * 2);
  if (filled) ctx.fill();
  else ctx.stroke();
  ctx.restore();
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
