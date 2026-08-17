'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
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
import { APARTMENT_COURSE, CourseSim, type StepResult } from '@/lib/slam/course';
import { Stat, fitPanel } from './shared';

/**
 * w14.4 — the Poisoned Web.
 *
 * The same course with the correspondences taken away. Every observation now
 * has to be *claimed* by a landmark, and the claim is a hard decision: once the
 * update is applied it is baked into Σ, and nothing downstream can revise it.
 *
 * Two failure modes are on display and they have opposite cures. Clutter births
 * phantom landmarks — the provisional list stops almost all of them for free.
 * Confusion between two real landmarks poisons the web permanently, and no list
 * helps, because the observation looked perfectly reasonable at the time.
 */

const ASPECT = 2.45;
const RUN = 500;
const MAP_PANEL = { x0: 0.03, y0: 0.06, w: 2.39, h: 0.88 };

interface Params {
  clutter: number;
  provisional: boolean;
  gate: number;
}

interface State {
  sim: CourseSim;
  last: StepResult | null;
  wrong: number;
  correct: number;
  phantomsBorn: number;
}

export function PoisonedWeb() {
  const [params, setParams] = useState<Params>({ clutter: 0.5, provisional: true, gate: 9.21 });

  const init = useCallback(
    (seed: number): State => ({
      sim: new CourseSim({ seed, knownCorrespondence: false }),
      last: null,
      wrong: 0,
      correct: 0,
      phantomsBorn: 0,
    }),
    [],
  );

  const step = useCallback((s: State): State => {
    // Knobs are pushed into the running filter, never used to rebuild it: a
    // filter that has already swallowed a false landmark cannot be repaired by
    // switching the guard on, and the widget should show that.
    s.sim.sensor.clutterRate = params.clutter;
    s.sim.filter.cfg.useProvisional = params.provisional;
    s.sim.filter.cfg.gateChi2 = params.gate;
    const last = s.sim.step();
    let wrong = s.wrong;
    let right = s.correct;
    let phantoms = s.phantomsBorn;
    last.associations.forEach((a, i) => {
      const truthId = last.features[i]?.s ?? -1;
      if (a.kind === 'matched') {
        // The label is ground-truth bookkeeping only — the filter never reads
        // it. It is here so the widget can score the decision the filter made.
        if (s.sim.filter.labels[a.landmark] === truthId && truthId >= 0) right += 1;
        else wrong += 1;
      } else if (a.kind === 'born' && truthId < 0) {
        phantoms += 1;
      }
    });
    return { sim: s.sim, last, wrong, correct: right, phantomsBorn: phantoms };
  }, [params]);

  const sim = useSimulation<State>({ init, step, fps: 15, maxTicks: RUN, loop: true, initialSeed: 21 });

  const stats = useMemo(() => {
    const f = sim.state.sim.filter;
    const seen = new Map<number, number>();
    for (const l of f.labels) seen.set(l, (seen.get(l) ?? 0) + 1);
    let phantom = 0;
    for (let j = 0; j < f.count; j++) if (f.labels[j] < 0) phantom += 1;
    // Distinct beacons represented, versus extra copies of them.
    const distinct = [...seen.keys()].filter((l) => l >= 0).length;
    const duplicate = f.count - phantom - distinct;
    let sq = 0;
    let k = 0;
    for (let j = 0; j < f.count; j++) {
      const b = sim.state.sim.truthFor(j);
      if (!b) continue;
      const [mx, my] = f.landmarkMean(j);
      sq += (mx - b.x) ** 2 + (my - b.y) ** 2;
      k += 1;
    }
    return {
      n: f.count,
      phantom,
      duplicate,
      real: distinct,
      candidates: f.candidates.length,
      rmse: k > 0 ? Math.sqrt(sq / k) : 0,
      wrong: sim.state.wrong,
      correct: sim.state.correct,
    };
  }, [sim.state]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const course = sim.state.sim;
      const f = course.filter;
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
      drawPath(ctx, v, course.truthPath.map((q) => ({ x: m.toX(q.x), y: m.toY(q.y) })), p.truth, {
        dashed: true,
        lineWidth: 1.2,
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

      // Observations, colored by the decision the filter made about them.
      const last = sim.state.last;
      if (last) {
        last.associations.forEach((a, i) => {
          const feat = last.features[i];
          if (!feat) return;
          const ang = course.truth.theta + feat.phi;
          const color =
            a.kind === 'matched'
              ? p.posterior
              : a.kind === 'born'
                ? p.prediction
                : a.kind === 'candidate'
                  ? p.accent
                  : p.measurement;
          ctx.save();
          ctx.strokeStyle = color;
          ctx.globalAlpha = a.kind === 'rejected' ? 0.45 : 0.75;
          ctx.lineWidth = 1.1;
          if (a.kind === 'rejected') ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(wx(course.truth.x), wy(course.truth.y));
          ctx.lineTo(wx(course.truth.x + feat.r * Math.cos(ang)), wy(course.truth.y + feat.r * Math.sin(ang)));
          ctx.stroke();
          ctx.restore();
        });
      }

      // Provisional candidates: seen, but not yet allowed into Σ.
      ctx.save();
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 1.2;
      for (const c of f.candidates) {
        ctx.beginPath();
        ctx.arc(wx(c.x), wy(c.y), 3.5, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      const seen = new Map<number, number>();
      for (const l of f.labels) seen.set(l, (seen.get(l) ?? 0) + 1);

      for (let j = 0; j < f.count; j++) {
        const [mx, my] = f.landmarkMean(j);
        const lab = f.labels[j];
        const bad = lab < 0 || (seen.get(lab) ?? 0) > 1;
        const e = ellipse2(f.landmarkCov(j), 2);
        drawCovariance(
          ctx,
          v,
          [m.toX(mx), m.toY(my)],
          { rx: e.rx * m.scale, ry: e.ry * m.scale, angle: e.angle },
          bad ? p.prediction : p.posterior,
          { alpha: bad ? 0.95 : 0.7, lineWidth: bad ? 1.6 : 1.2 },
        );
      }

      const est = f.pose();
      const pe = ellipse2(f.poseCov(), 2);
      drawCovariance(
        ctx,
        v,
        [m.toX(est.x), m.toY(est.y)],
        { rx: pe.rx * m.scale, ry: pe.ry * m.scale, angle: pe.angle },
        p.posterior,
        { alpha: 1, lineWidth: 1.8 },
      );
      drawRobot(
        ctx,
        v,
        { x: m.toX(course.truth.x), y: m.toY(course.truth.y), theta: course.truth.theta },
        p.truth,
        0.42 * m.scale,
      );

      label(ctx, 'purple: matched   orange: new landmark   teal: provisional   green: gated out', wx(0.2), wy(9.35), p.ink, { size: 9 });
    },
    [sim.state],
  );

  return (
    <WidgetFrame
      id="w14.4"
      title="The Poisoned Web"
      teaches="Association errors do not average out. In a filter they are structural: one wrong claim is written into Σ and can never be withdrawn."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The same course with correspondences withheld. Each ray is colored by the decision the
          filter made: purple was claimed by an existing landmark, orange was born as a new one,
          teal went to the provisional list, green fell in the no-man&rsquo;s-land between the gate
          and the birth threshold and was thrown away. Orange ellipses are landmarks that should not
          exist — phantoms born from clutter, or a second copy of a beacon the filter failed to
          recognize. Turn the <strong>provisional list</strong> off and watch the map fill with
          debris within one lap; turn it back on and almost all of it disappears, because clutter
          does not repeat and a candidate that is never seen twice never enters Σ. Then look at the
          duplicates, which the list does <em>not</em> prevent: those are the filter failing to
          re-recognize its own landmarks after the blind crossing, because its gate — sized by a
          covariance it is too confident about — has become smaller than the error it is actually
          making. Flaw (b) causes flaw (a)&rsquo;s cousin.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: ASPECT, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="The Apartment corridor with Rusty and its landmark estimates. Rays from the robot are colored by association outcome, and landmark ellipses turn orange when they are phantoms or duplicates."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-5">
        <Stat label="landmarks in Σ" value={String(stats.n)} alarm={stats.n > 12} />
        <Stat label="real (of 8)" value={String(stats.real)} />
        <Stat
          label="phantom + duplicate"
          value={`${stats.phantom} + ${stats.duplicate}`}
          alarm={stats.phantom + stats.duplicate > 0}
        />
        <Stat label="provisional list" value={String(stats.candidates)} />
        <Stat
          label="wrong associations"
          value={String(stats.wrong)}
          alarm={stats.wrong > 0}
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="clutter rate"
          role="measurement"
          value={params.clutter}
          min={0}
          max={2}
          step={0.1}
          unit="/step"
          onChange={(v) => setParams((q) => ({ ...q, clutter: v }))}
          help="Expected spurious detections per step, drawn uniformly over the sensor footprint. They never repeat in the same place."
        />
        <Toggle
          label="provisional list"
          role="posterior"
          checked={params.provisional}
          onChange={(v) => setParams((q) => ({ ...q, provisional: v }))}
        />
        <Slider
          label="gate γ (χ², 2 dof)"
          value={params.gate}
          min={2}
          max={30}
          step={0.5}
          onChange={(v) => setParams((q) => ({ ...q, gate: v }))}
          help="5.99 is 95%, 9.21 is 99%, 13.8 is 99.9%. Too tight and the filter fails to recognize its own landmarks; too loose and it confuses them."
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
