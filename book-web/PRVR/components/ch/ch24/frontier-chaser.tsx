'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import {
  DEFAULT_EXPLORE_CONFIG,
  ExploreSim,
  type ExplorePolicy,
} from '@/lib/explore/explorer';
import type { Candidate } from '@/lib/explore/utility';
import {
  clear,
  drawOccupancyGrid,
  drawPath,
  drawRobot,
  drawScan,
  drawSegments,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w24.1 — Frontier Chaser.
 *
 * Rusty is dropped into the apartment with an empty log-odds map and no goal
 * but its own ignorance. Every replan runs the real `ExploreSim` from
 * `lib/explore/explorer.ts`: `detectFrontiers` identifies the candidates,
 * `expectedInfoGain` scores them in bits, a Dijkstra field prices them in
 * metres, and the argmax of U(a) = w_I·I(a) − w_C·C(a) is the star on the map.
 *
 * The one slider that matters is w_C. Slide it up and the utility collapses to
 * "go to the nearest unknown"; slide it down and Rusty crosses the apartment
 * for a big room. Watch the bits-per-metre readout rather than the map: the
 * greedy policy is not the efficient one.
 */

const START = { x: 6.0, y: 4.4, theta: 0 };
const HOLD_TICKS = 45;

interface State {
  sim: ExploreSim;
  seed: number;
  hold: number;
  v: number;
}

const POLICIES: { id: ExplorePolicy; label: string; blurb: string }[] = [
  { id: 'lawnmower', label: 'Lawnmower', blurb: 'a scripted boustrophedon sweep — no map in the loop at all' },
  { id: 'nearest', label: 'Nearest frontier', blurb: 'w_I = 0: whichever frontier is closest wins' },
  { id: 'utility', label: 'Utility', blurb: 'w_I·I(a) − w_C·C(a): bits weighed against metres' },
];

export function FrontierChaser() {
  const [policy, setPolicy] = useState<ExplorePolicy>('utility');
  const [wC, setWC] = useState(0.35);

  const wCRef = useRef(wC);
  wCRef.current = wC;

  const build = useCallback(
    (seed: number, p: ExplorePolicy) =>
      new ExploreSim({
        ...DEFAULT_EXPLORE_CONFIG,
        world: APARTMENT,
        start: START,
        policy: p,
        seed,
        weights: { ...DEFAULT_EXPLORE_CONFIG.weights, wC: wCRef.current },
      }),
    [],
  );

  const init = useCallback(
    (seed: number): State => ({ sim: build(seed, policy), seed, hold: 0, v: 0 }),
    [build, policy],
  );

  const step = useCallback((s: State): State => {
    // The slider is live: the weights the explorer argues with are read fresh
    // every tick, so re-ranking the candidates costs no restart.
    s.sim.cfg.weights = { ...s.sim.cfg.weights, wC: wCRef.current };

    if (s.sim.done) {
      // Let the finished map sit for a beat, then replay the same seed — the
      // run is deterministic, so this is a loop, not a new experiment.
      if (s.hold >= HOLD_TICKS) {
        return { sim: build(s.seed, s.sim.cfg.policy), seed: s.seed, hold: 0, v: s.v + 1 };
      }
      return { ...s, hold: s.hold + 1, v: s.v + 1 };
    }
    s.sim.step();
    return { ...s, v: s.v + 1 };
  }, [build]);

  const sim = useSimulation<State>({ init, step, fps: 20, initialSeed: 24 });

  // Switching policy is a different experiment, so it restarts the run.
  const resetRef = useRef(sim.reset);
  resetRef.current = sim.reset;
  useEffect(() => {
    resetRef.current();
  }, [policy]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state.sim;

      // The map Rusty actually has. Everything still at p = 0.5 stays the
      // background colour: ignorance is drawn as absence, not as gray paint.
      drawOccupancyGrid(ctx, v, s.grid, p);

      // Ground truth, faint and dashed — the reader's orientation, never the
      // robot's information.
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.setLineDash([4, 4]);
      drawSegments(ctx, v, APARTMENT.walls, p.truth, 1.2);
      ctx.restore();

      // Frontier regions, filled by utility: the darker the purple, the more
      // the explorer wants to be there.
      const cands = s.candidates;
      if (cands.length > 0) {
        const uMin = Math.min(...cands.map((c) => c.utility));
        const uMax = Math.max(...cands.map((c) => c.utility));
        const span = Math.max(uMax - uMin, 1e-6);
        const w = Math.ceil(sl(v, s.grid.cellSize)) + 1;
        ctx.save();
        ctx.fillStyle = p.posterior;
        for (const c of cands) {
          ctx.globalAlpha = 0.2 + 0.65 * ((c.utility - uMin) / span);
          for (const cell of c.frontier.cells) {
            const [cx, cy] = s.grid.cellCenter(cell.i, cell.j);
            ctx.fillRect(
              sx(v, cx - s.grid.cellSize / 2),
              sy(v, cy + s.grid.cellSize / 2),
              w,
              w,
            );
          }
        }
        ctx.restore();
      }

      // The plan, then the road actually travelled.
      drawPath(ctx, v, s.path, p.posterior, { dashed: true, lineWidth: 1.6, alpha: 0.9 });
      drawPath(ctx, v, s.trail, p.truth, { lineWidth: 1.4, alpha: 0.75 });

      if (s.chosen) {
        const [tx, ty] = s.grid.cellCenter(s.chosen.target.i, s.chosen.target.j);
        star(ctx, sx(v, tx), sy(v, ty), 7, p.posterior);
      }

      drawScan(ctx, v, s.pose, s.scan.ranges, s.scan.angles, p.measurement, s.cfg.lidar.maxRange);
      drawRobot(ctx, v, s.pose, p.truth, 0.26);

      const caption = s.done
        ? s.reason === 'no-frontiers'
          ? 'DONE — no reachable frontier left'
          : 'DONE — gain per metre below threshold'
        : s.cfg.policy === 'lawnmower'
          ? 'LAWNMOWER — driving lanes, ignoring the map'
          : `IDENTIFY → SELECT → EXECUTE   (${cands.length} candidates)`;
      label(ctx, caption, 10, 14, s.done ? p.posterior : p.measurement, { size: 11, weight: 600 });
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const s = sim.state.sim;
    const rows = s.samples;
    const last = rows[rows.length - 1];
    const h0 = rows[0]?.entropy ?? 1;
    const tail = rows.slice(-90);
    const every = Math.max(1, Math.floor(tail.length / 30));
    return {
      entropy: last?.entropy ?? 0,
      entropySpark: tail.filter((_, i) => i % every === 0).map((r) => r.entropy),
      distance: last?.distance ?? 0,
      coverage: (last?.coverage ?? 0) * 100,
      rate: last?.gainRate ?? 0,
      rateSpark: tail.filter((_, i) => i % every === 0).map((r) => r.gainRate),
      perMetre: last && last.distance > 0.5 ? (h0 - last.entropy) / last.distance : 0,
      candidates: s.candidates.slice(0, 5),
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w24.1"
      title="Frontier Chaser"
      teaches="Exploration is not “go to the nearest unknown”. Nearest is one setting of a utility, and usually the wrong one."
      colorKey={['measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty starts in the corridor with a blank log-odds map and no goal except its own
          ignorance. Purple patches are frontier regions, shaded by utility; the star is the chosen
          target and the dashed line is the plan. Watch <em>bits per metre</em>, not the map — that
          is the number the policies actually differ on. Then try the other two policies at the same
          seed. The lawnmower drives 93 m of lanes and still leaves 4% of the apartment unmapped.
          Nearest-frontier is worse and stranger: it is not frozen when the distance readout stops
          moving — it has targeted a frontier under its own wheels, which costs nothing to reach, so
          it stands there re-scanning until the map pushes the boundary away. Finally, push{' '}
          <span className="font-mono">w_C</span> above ~0.9 and the utility policy quietly{' '}
          <em>becomes</em> nearest-frontier, stall and all.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: 12, maxY: 9 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={1.5}
        padding={0.3}
        ariaLabel="An apartment floorplan being mapped by an exploring robot. Purple patches mark frontier regions between mapped free space and unknown space, shaded by their utility; a star marks the chosen target and a dashed line the planned path."
      />

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 sm:grid-cols-4">
        <StatTile
          label="map entropy H(m)"
          value={stats.entropy}
          unit="bits"
          precision={0}
          role="posterior"
          sparkline={stats.entropySpark}
        />
        <StatTile label="distance travelled" value={stats.distance} unit="m" precision={1} />
        <StatTile label="cells resolved" value={stats.coverage} unit="%" precision={1} />
        <StatTile
          label="bits per metre"
          value={stats.perMetre}
          unit="b/m"
          precision={1}
          role="measurement"
          sparkline={stats.rateSpark}
        />
      </div>

      <div className="border-t border-fd-border px-3 py-2.5">
        <p className="eyebrow mb-1.5">candidate utilities — U(a) = w_I·I(a) − w_C·C(a)</p>
        {stats.candidates.length === 0 ? (
          <p className="font-ui text-xs text-fd-muted-foreground">
            {sim.state.sim.cfg.policy === 'lawnmower'
              ? 'The lawnmower never scores anything. It is a path, not a policy.'
              : 'No scored candidates this tick.'}
          </p>
        ) : (
          <ul className="m-0 flex list-none flex-col gap-1 p-0">
            {stats.candidates.map((c, k) => (
              <CandidateBar
                key={`${c.target.i}-${c.target.j}`}
                c={c}
                best={stats.candidates[0].utility}
                chosen={k === 0}
              />
            ))}
          </ul>
        )}
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Cost weight w_C  (metres vs. bits)"
          value={wC}
          min={0.05}
          max={1.2}
          step={0.05}
          onChange={setWC}
          format={(x) => x.toFixed(2)}
          help="At 0.05 the explorer is info-greedy and will cross the apartment. Above ~0.9 it degenerates to nearest-frontier."
        />
        <div className="flex flex-col gap-1.5">
          <span className="font-ui text-[0.72rem] font-medium">Policy</span>
          <ButtonRow>
            {POLICIES.map((p) => (
              <ActionButton key={p.id} onClick={() => setPolicy(p.id)} emphasis={policy === p.id}>
                {p.label}
              </ActionButton>
            ))}
          </ButtonRow>
          <span className="font-ui text-[0.68rem] text-fd-muted-foreground">
            {POLICIES.find((p) => p.id === policy)?.blurb}
          </span>
        </div>
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

function CandidateBar({ c, best, chosen }: { c: Candidate; best: number; chosen: boolean }) {
  const span = Math.max(Math.abs(best), 1e-6);
  const frac = Math.max(0.02, Math.min(1, c.utility / span));
  return (
    <li className="flex items-center gap-2">
      <span className="h-2 flex-1 overflow-hidden rounded-[1px] bg-fd-muted">
        <span
          className="block h-full"
          style={{
            width: `${frac * 100}%`,
            background: chosen ? 'var(--pr-posterior)' : 'var(--pr-prior)',
            opacity: chosen ? 1 : 0.55,
          }}
        />
      </span>
      <span className="w-[13.5rem] shrink-0 text-end font-mono text-[0.68rem] tabular-nums text-fd-muted-foreground">
        I={c.gain.toFixed(1)} b · C={c.cost.toFixed(1)} m · U={c.utility.toFixed(1)}
      </span>
    </li>
  );
}

/** The chosen target. A five-pointed star reads as "here" at any zoom. */
function star(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, color: string) {
  ctx.save();
  ctx.beginPath();
  for (let k = 0; k < 10; k++) {
    const a = (Math.PI / 5) * k - Math.PI / 2;
    const rad = k % 2 === 0 ? r : r * 0.44;
    const x = cx + rad * Math.cos(a);
    const y = cy + rad * Math.sin(a);
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();
}
