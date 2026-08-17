'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { graphCost, optimizeStep, type Values } from '@/lib/optim/factor-graph';
import { duelValues, rangeDuelScene } from '@/lib/optim/scenes';

/**
 * w15.3 — the Descent Duel.
 *
 * One 2-D variable, three ranges, and a cost surface with a curved valley. The
 * three step rules are the *same* linear system with three amounts of damping:
 * λ = 0 is Gauss–Newton, λ → ∞ is scaled gradient descent, and in between is
 * Levenberg–Marquardt. Watching them race makes it obvious that LM is not "GN
 * with insurance" — it is a continuum, and λ chooses where on it you stand.
 */

const VIEW = { minX: -3.2, maxX: 3.2, minY: -1.6, maxY: 3.4 };
const GRID_X = 96;
const GRID_Y = 72;
const STEPS = 7;

/** λ values the autoplay sweeps through, log-spaced. */
const SWEEP = Array.from({ length: 36 }, (_, i) => Math.pow(10, -4 + (8 * i) / 35));

interface Params {
  lambda: number;
  manual: boolean;
  start: { x: number; y: number };
}

function runPath(
  graph: ReturnType<typeof rangeDuelScene>['graph'],
  index: ReturnType<typeof rangeDuelScene>['index'],
  start: { x: number; y: number },
  lambda: number,
): { path: { x: number; y: number }[]; costs: number[]; firstStep: number } {
  let v: Values = duelValues(start);
  const path = [{ ...start }];
  const costs = [graphCost(graph, v)];
  let firstStep = 0;
  for (let i = 0; i < STEPS; i++) {
    const s = optimizeStep(graph, v, index, lambda);
    const norm = Math.sqrt(s.delta.reduce((a, x) => a + x * x, 0));
    if (i === 0) firstStep = norm;
    v = s.values;
    const p = v.landmarks[0];
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) break;
    path.push({ ...p });
    costs.push(graphCost(graph, v));
  }
  return { path, costs, firstStep };
}

export function DescentDuel() {
  const scene = useMemo(() => rangeDuelScene(), []);
  const [params, setParams] = useState<Params>({ lambda: 1, manual: false, start: { x: -2.5, y: 0.2 } });
  const pressed = useRef(false);

  const sim = useSimulation<{ i: number }>({
    init: () => ({ i: 0 }),
    step: (s) => ({ i: (s.i + 1) % SWEEP.length }),
    fps: 3,
  });

  const lambda = params.manual ? params.lambda : SWEEP[sim.state.i];

  // The cost surface never changes, so it is sampled once.
  const field = useMemo(() => {
    const cells = new Float64Array(GRID_X * GRID_Y);
    let max = 0;
    for (let j = 0; j < GRID_Y; j++) {
      const y = VIEW.minY + ((j + 0.5) / GRID_Y) * (VIEW.maxY - VIEW.minY);
      for (let i = 0; i < GRID_X; i++) {
        const x = VIEW.minX + ((i + 0.5) / GRID_X) * (VIEW.maxX - VIEW.minX);
        const c = graphCost(scene.graph, duelValues({ x, y }));
        cells[j * GRID_X + i] = c;
        if (c > max) max = c;
      }
    }
    return { cells, max };
  }, [scene]);

  const runs = useMemo(() => {
    const gn = runPath(scene.graph, scene.index, params.start, 0);
    const lm = runPath(scene.graph, scene.index, params.start, lambda);
    const gd = runPath(scene.graph, scene.index, params.start, 1e4);
    return { gn, lm, gd };
  }, [scene, params.start, lambda]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);

      // ---- cost surface, as log-spaced bands ------------------------------
      const cw = (VIEW.maxX - VIEW.minX) / GRID_X;
      const ch = (VIEW.maxY - VIEW.minY) / GRID_Y;
      const top = Math.log10(1 + field.max);
      for (let j = 0; j < GRID_Y; j++) {
        for (let i = 0; i < GRID_X; i++) {
          const t = Math.log10(1 + field.cells[j * GRID_X + i]) / top;
          const band = (t * 9) % 1;
          const alpha = (0.05 + 0.4 * (1 - t)) * (0.55 + 0.45 * band);
          ctx.fillStyle = p.truth;
          ctx.globalAlpha = alpha;
          const x0 = VIEW.minX + i * cw;
          const y0 = VIEW.minY + j * ch;
          ctx.fillRect(sx(v, x0), sy(v, y0 + ch), Math.ceil(sl(v, cw)) + 1, Math.ceil(sl(v, ch)) + 1);
        }
      }
      ctx.globalAlpha = 1;

      // ---- the anchors and the true solution ------------------------------
      for (const a of scene.anchors) {
        ctx.strokeStyle = p.measurement;
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(sx(v, a.p.x), sy(v, a.p.y), sl(v, a.r), 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = p.measurement;
        ctx.beginPath();
        ctx.arc(sx(v, a.p.x), sy(v, a.p.y), 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(sx(v, scene.solution.x), sy(v, scene.solution.y), 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // ---- three descent paths --------------------------------------------
      const drawRun = (
        run: { path: { x: number; y: number }[] },
        color: string,
        name: string,
        dashed = false,
      ) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        if (dashed) ctx.setLineDash([4, 3]);
        ctx.beginPath();
        run.path.forEach((pt, i) => {
          const X = sx(v, pt.x);
          const Y = sy(v, pt.y);
          if (i === 0) ctx.moveTo(X, Y);
          else ctx.lineTo(X, Y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        run.path.forEach((pt, i) => {
          if (i === 0) return;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(sx(v, pt.x), sy(v, pt.y), 2.6, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.globalAlpha = 1;
        const end = run.path[run.path.length - 1];
        label(ctx, name, sx(v, end.x) + 7, sy(v, end.y), color, { size: 10, weight: 600 });
        ctx.restore();
      };

      drawRun(runs.gd, p.prior, 'gradient descent (λ→∞)', true);
      drawRun(runs.gn, p.prediction, 'Gauss–Newton (λ=0)');
      drawRun(runs.lm, p.posterior, `LM (λ=${lambda.toExponential(0)})`);

      ctx.fillStyle = p.accent;
      ctx.beginPath();
      ctx.arc(sx(v, params.start.x), sy(v, params.start.y), 4.5, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, 'start — drag me', sx(v, params.start.x) + 8, sy(v, params.start.y) + 12, p.accent, {
        size: 10,
      });
    },
    [field, runs, scene, lambda, params.start],
  );

  const stats = useMemo(
    () => ({
      gnFirst: runs.gn.firstStep,
      lmFirst: runs.lm.firstStep,
      gdFirst: runs.gd.firstStep,
      gnCost: runs.gn.costs[runs.gn.costs.length - 1],
      lmCost: runs.lm.costs[runs.lm.costs.length - 1],
      gdCost: runs.gd.costs[runs.gd.costs.length - 1],
    }),
    [runs],
  );

  return (
    <WidgetFrame
      id="w15.3"
      title="Descent Duel"
      teaches="Levenberg–Marquardt is not Gauss–Newton with insurance: λ slides continuously between a step that trusts the linear model completely and one that barely trusts it at all."
      caption={
        <>
          The gray bands are iso-contours of the true nonlinear cost for one 2-D position measured by
          three ranges (green circles); the dashed gray ring is the optimum. All three paths solve the
          same normal equations — they differ only in λ. Gauss–Newton (orange) takes the step the
          linearized model recommends, and where the valley curves that step can be enormous: watch
          its first move leave the picture entirely. Gradient descent (blue, dashed) never leaves the
          surface and never gets anywhere. LM (purple) is neither: as the sweep raises λ its steps
          shorten and rotate toward the gradient. <strong>Drag the start point</strong> into the flat
          ridge between the anchors — where the Jacobian is nearly rank-deficient — and watch the
          first Gauss–Newton step become absurd while LM stays sane.
        </>
      }
    >
      <SimCanvas
        world={VIEW}
        draw={draw}
        deps={[runs, field, lambda, params.start]}
        aspect={1.5}
        padding={0.1}
        onPointer={(world, phase) => {
          if (phase === 'down') pressed.current = true;
          if (phase === 'up') {
            pressed.current = false;
            return;
          }
          if (!pressed.current) return; // hovering must not move the start point
          setParams((p) => ({ ...p, start: { x: world[0], y: world[1] } }));
        }}
        ariaLabel="Contours of a nonlinear least-squares cost with three descent paths overlaid: Gauss-Newton overshoots the curved valley, gradient descent creeps, and Levenberg-Marquardt interpolates between them."
      />

      <div className="px-3 pt-3">
        <Dashboard columns={3}>
          <StatTile
            label="‖Δ‖ first step — GN"
            value={stats.gnFirst}
            role="prediction"
            precision={2}
            trendLabel={`final J = ${stats.gnCost.toExponential(1)}`}
          />
          <StatTile
            label={`‖Δ‖ first step — LM`}
            value={stats.lmFirst}
            role="posterior"
            precision={2}
            trendLabel={`final J = ${stats.lmCost.toExponential(1)}`}
          />
          <StatTile
            label="‖Δ‖ first step — gradient"
            value={stats.gdFirst}
            role="prior"
            precision={4}
            trendLabel={`final J = ${stats.gdCost.toExponential(1)}`}
          />
        </Dashboard>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="LM damping λ"
          role="posterior"
          value={Math.log10(lambda)}
          min={-4}
          max={4}
          step={0.1}
          format={(v) => `10^${v.toFixed(1)}`}
          onChange={(v) => {
            sim.pause();
            setParams((p) => ({ ...p, lambda: Math.pow(10, v), manual: true }));
          }}
          help="λ → 0 recovers Gauss-Newton; λ → ∞ recovers a very short gradient step."
        />
        <ButtonRow>
          <ActionButton
            emphasis={!params.manual}
            onClick={() => {
              setParams((p) => ({ ...p, manual: false }));
              sim.play();
            }}
          >
            Resume λ sweep
          </ActionButton>
          <ActionButton onClick={() => setParams((p) => ({ ...p, start: { x: 0, y: 0.05 } }))}>
            Start on the ridge
          </ActionButton>
          <ActionButton onClick={() => setParams((p) => ({ ...p, start: { x: -2.5, y: 0.2 } }))}>
            Reset start
          </ActionButton>
        </ButtonRow>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
