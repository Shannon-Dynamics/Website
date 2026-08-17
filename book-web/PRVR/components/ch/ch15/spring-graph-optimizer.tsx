'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import {
  clear,
  drawPath,
  drawRobot,
  drawSegments,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import {
  cloneValues,
  graphCost,
  linearizeGraph,
  lmTrial,
  poseRmse,
  type FactorGraph,
  type System,
  type Values,
} from '@/lib/optim/factor-graph';
import type { Kernel } from '@/lib/optim/kernels';
import { apartmentLoopData, buildSlamGraph, type SlamData } from '@/lib/optim/scenes';

/**
 * w15.1 — the Spring-Graph Optimizer.
 *
 * Every factor is a spring: its stiffness is its information, its extension is
 * its residual, and MAP inference is the graph relaxing to minimum energy. One
 * animation tick is one Levenberg–Marquardt trial of the library's real solver
 * on the real Apartment dataset — the same iterations the Rust example prints.
 */

type KernelChoice = 'l2' | 'huber' | 'cauchy' | 'geman';

const KERNEL_LABEL: Record<KernelChoice, string> = {
  l2: 'L2',
  huber: 'Huber',
  cauchy: 'Cauchy',
  geman: 'Geman–McClure',
};

interface Params {
  kernel: KernelChoice;
  scale: number;
  outlier: number;
}

interface State {
  data: SlamData;
  graph: FactorGraph;
  values: Values;
  lambda: number;
  cost: number;
  system: System;
  trace: number[];
  accepted: boolean;
  converged: boolean;
  iterations: number;
}

function makeKernel(choice: KernelChoice, scale: number): Kernel {
  switch (choice) {
    case 'huber':
      return { type: 'huber', k: scale };
    case 'cauchy':
      return { type: 'cauchy', c: scale };
    case 'geman':
      return { type: 'geman', c: scale };
    default:
      return { type: 'l2' };
  }
}

/** Alpha ramp for a spring drawn by how hard it is pulling. */
const stress = (e: number) => Math.min(0.16 + 0.84 * (e / 6), 1);

export function SpringGraphOptimizer() {
  const [params, setParams] = useState<Params>({ kernel: 'l2', scale: 1.0, outlier: 0 });
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const dragRef = useRef<number | null>(null);

  const buildOpts = useCallback(
    () => ({
      kernel: makeKernel(paramsRef.current.kernel, paramsRef.current.scale),
      outlier: paramsRef.current.outlier,
    }),
    [],
  );

  const init = useCallback(
    (seed: number): State => {
      const data = apartmentLoopData({ seed });
      const graph = buildSlamGraph(data, buildOpts());
      const values = cloneValues(data.init);
      return {
        data,
        graph,
        values,
        lambda: 1e-3,
        cost: graphCost(graph, values),
        system: linearizeGraph(graph, values, data.index),
        trace: [],
        accepted: true,
        converged: false,
        iterations: 0,
      };
    },
    [buildOpts],
  );

  const step = useCallback(
    (s: State): State => {
      // Rebuilding the graph each step keeps it in sync with the controls; it
      // is a few hundred closures, which costs nothing next to the solve.
      const graph = buildSlamGraph(s.data, buildOpts());
      const trial = lmTrial(graph, s.values, s.data.index, s.lambda);
      const values = trial.values;
      const cost = trial.accepted ? trial.cost : s.cost;
      // "Converged" is a state, not a failure: once J stops moving, stop
      // escalating λ so that a drag or a control change can still be answered.
      const rel = Math.abs(s.cost - cost) / Math.max(s.cost, 1e-12);
      const converged = rel < 1e-9 && trial.stepNorm < 1e-5;
      const lambda = converged
        ? Math.min(s.lambda, 1e-3)
        : trial.accepted
          ? Math.max(s.lambda * Math.max(1 / 3, 1 - Math.pow(2 * trial.gain - 1, 3)), 1e-9)
          : Math.min(s.lambda * 10, 1e8);
      return {
        ...s,
        graph,
        values,
        lambda,
        cost,
        system: linearizeGraph(graph, values, s.data.index),
        trace: [...s.trace, cost].slice(-80),
        accepted: trial.accepted,
        converged,
        iterations: s.iterations + (trial.accepted ? 1 : 0),
      };
    },
    [buildOpts],
  );

  const sim = useSimulation<State>({ init, step, fps: 3, initialSeed: 42 });
  const { setState } = sim;

  // Changing a control must show up immediately, even while paused.
  useEffect(() => {
    setState((s) => {
      const graph = buildSlamGraph(s.data, {
        kernel: makeKernel(params.kernel, params.scale),
        outlier: params.outlier,
      });
      return {
        ...s,
        graph,
        cost: graphCost(graph, s.values),
        system: linearizeGraph(graph, s.values, s.data.index),
        lambda: Math.max(s.lambda, 1e-4),
        converged: false,
      };
    });
  }, [params, setState]);

  /* ---- direct manipulation: grab a pose, and the springs pull it back ---- */
  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') {
        dragRef.current = null;
        return;
      }
      if (phase === 'down') {
        let best = -1;
        let bestD = 0.5;
        sim.state.values.poses.forEach((p, i) => {
          const d = Math.hypot(p.x - world[0], p.y - world[1]);
          if (d < bestD) {
            bestD = d;
            best = i;
          }
        });
        dragRef.current = best >= 0 ? best : null;
      }
      const idx = dragRef.current;
      if (idx === null) return;
      setState((s) => {
        const values = cloneValues(s.values);
        values.poses[idx] = { ...values.poses[idx], x: world[0], y: world[1] };
        return { ...s, values, lambda: Math.max(s.lambda, 1e-4), converged: false };
      });
    },
    [sim.state, setState],
  );

  /* ---- drawing ----------------------------------------------------------- */
  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { values, system, data, graph, cost, accepted, converged } = sim.state;
      drawSegments(ctx, v, APARTMENT.walls, p.wall, 1.6);

      // Ground truth, which the robot never gets to see.
      drawPath(ctx, v, data.truthPoses, p.truth, { dashed: true, lineWidth: 1.6, alpha: 0.85 });
      ctx.save();
      ctx.globalAlpha = 0.7;
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.2;
      for (const m of data.truthLandmarks) {
        const cx = sx(v, m.x);
        const cy = sy(v, m.y);
        ctx.beginPath();
        ctx.moveTo(cx - 4, cy - 4);
        ctx.lineTo(cx + 4, cy + 4);
        ctx.moveTo(cx + 4, cy - 4);
        ctx.lineTo(cx - 4, cy + 4);
        ctx.stroke();
      }
      ctx.restore();

      // Where the robot thought it was: the odometric initial guess.
      drawPath(ctx, v, data.init.poses, p.prediction, { lineWidth: 1.2, alpha: 0.4 });

      // ---- springs --------------------------------------------------------
      const reports = system.factors;
      const point = (kind: string, id: number) =>
        kind === 'pose' ? values.poses[id] : values.landmarks[id];
      ctx.save();
      graph.factors.forEach((f, k) => {
        if (f.keys.length < 2) return;
        const a = point(f.keys[0].kind, f.keys[0].id);
        const b = point(f.keys[1].kind, f.keys[1].id);
        if (!a || !b) return;
        const e = reports[k]?.e ?? 0;
        const w = reports[k]?.w ?? 1;
        const isLoop = f.kind === 'loop';
        const isFalse = f.id === 'loop:false';
        ctx.strokeStyle = f.kind === 'landmark' ? p.measurement : isLoop ? p.posterior : p.prediction;
        ctx.globalAlpha = f.kind === 'landmark' ? 0.12 + 0.45 * stress(e) : stress(e);
        // A robust kernel makes a spring physically slacker — so draw its weight.
        ctx.lineWidth = isLoop ? 1.2 + 3 * Math.sqrt(w) : f.kind === 'landmark' ? 0.9 : 1.8;
        if (isFalse) ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, a.x), sy(v, a.y));
        ctx.lineTo(sx(v, b.x), sy(v, b.y));
        ctx.stroke();
        ctx.setLineDash([]);
        if (isFalse) {
          ctx.globalAlpha = 1;
          label(
            ctx,
            `false closure  e=${e.toFixed(0)}σ  w=${w.toFixed(2)}`,
            sx(v, (a.x + b.x) / 2) + 7,
            sy(v, (a.y + b.y) / 2),
            p.posterior,
            { size: 10, weight: 600 },
          );
        }
      });
      ctx.restore();

      // ---- the estimate ---------------------------------------------------
      drawPath(ctx, v, values.poses, p.posterior, { lineWidth: 2 });
      ctx.save();
      ctx.fillStyle = p.posterior;
      for (const m of values.landmarks) {
        ctx.beginPath();
        ctx.arc(sx(v, m.x), sy(v, m.y), 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      for (let i = 0; i < values.poses.length; i += 6) {
        drawRobot(ctx, v, values.poses[i], p.posterior, 0.2, { filled: false, alpha: 0.85 });
      }

      const grabbed = dragRef.current;
      if (grabbed !== null && values.poses[grabbed]) {
        const g = values.poses[grabbed];
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx(v, g.x), sy(v, g.y), sl(v, 0.35), 0, Math.PI * 2);
        ctx.stroke();
      }

      label(ctx, `J = ${cost.toFixed(1)}`, 10, 16, p.posterior, { size: 12, weight: 700 });
      label(
        ctx,
        converged
          ? 'converged — J is stationary'
          : accepted
            ? 'LM step accepted'
            : 'step rejected — λ increased',
        10,
        32,
        converged ? p.posterior : p.prediction,
        { size: 10 },
      );
    },
    [sim.state],
  );

  const stats = useMemo(() => {
    const { values, data, system } = sim.state;
    return {
      rmse: poseRmse(values.poses, data.truthPoses),
      odoRmse: poseRmse(data.init.poses, data.truthPoses),
      bad: system.factors.find((f) => f.id === 'loop:false'),
    };
  }, [sim.state]);

  return (
    <WidgetFrame
      id="w15.1"
      title="Spring-Graph Optimizer"
      teaches="MAP inference is not a black box — it is a graph of springs relaxing. And least squares does not average an outlier away: it lets the outlier pull."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Orange is the dead-reckoned guess the optimizer starts from, gray dashed is the truth,
          purple is the current estimate. Thin green springs are landmark sightings, orange springs
          are controls, and the thick purple spring is the loop closure. Watch the first three
          iterations: nearly all of the correction happens there, because every Jacobian — including
          the one for the first pose, fifty steps ago — is re-evaluated at the newest estimate, which
          is exactly what <Link href="/chapters/ch14-ekf-slam">Chapter 14</Link>&apos;s filter structurally
          cannot do. Then <strong>drag a pose and let go</strong>: the springs pull it back, because
          the MAP estimate <em>is</em> that equilibrium. Now press <strong>inject a false loop
          closure</strong>. Under L2 one bad spring bends the whole map. Switch to Huber and the bad
          spring is drawn thinner (its IRLS weight is printed beside it) — but the map only partly
          recovers, because Huber caps influence without ever releasing it. Geman–McClure lets go
          entirely, and the map snaps back to the clean solution.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: 12, maxY: 9 }}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={1.62}
        padding={0.35}
        onPointer={onPointer}
        cursor="grab"
        ariaLabel="The Apartment floorplan with Rusty's drifted odometry path in orange, the ground-truth path dashed in gray, and the current least-squares estimate in purple, drawn as a graph of springs joining poses and landmarks."
      />

      <div className="px-3 pt-3">
        <Dashboard columns={4}>
          <StatTile
            label="objective J"
            value={sim.state.cost}
            role="posterior"
            precision={1}
            sparkline={sim.state.trace}
          />
          <StatTile label="RMSE vs truth" value={stats.rmse} unit="m" role="posterior" precision={3} />
          <StatTile label="odometry RMSE" value={stats.odoRmse} unit="m" role="prediction" precision={3} />
          <StatTile
            label={stats.bad ? 'false-closure weight w' : 'LM damping λ'}
            value={stats.bad ? stats.bad.w : sim.state.lambda}
            precision={stats.bad ? 3 : 6}
            role={stats.bad ? 'measurement' : undefined}
          />
        </Dashboard>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Kernel scale k (σ)"
          role="measurement"
          value={params.scale}
          min={0.3}
          max={6}
          step={0.1}
          onChange={(v) => setParams((p) => ({ ...p, scale: v }))}
          help="Residuals below k sigmas count as ordinary noise; above it the kernel starts discounting them."
        />
        <Slider
          label="False closure error"
          role="prediction"
          value={params.outlier}
          min={0}
          max={3}
          step={0.1}
          unit="m"
          onChange={(v) => setParams((p) => ({ ...p, outlier: v }))}
          help="How badly the front end lies about the relative pose of two places."
        />
        <div className="flex flex-col gap-1.5">
          <span className="eyebrow">Robust kernel</span>
          <ButtonRow>
            {(['l2', 'huber', 'cauchy', 'geman'] as KernelChoice[]).map((k) => (
              <ActionButton
                key={k}
                emphasis={params.kernel === k}
                onClick={() => setParams((p) => ({ ...p, kernel: k }))}
              >
                {KERNEL_LABEL[k]}
              </ActionButton>
            ))}
          </ButtonRow>
        </div>
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton
            emphasis={params.outlier === 0}
            onClick={() => setParams((p) => ({ ...p, outlier: p.outlier > 0 ? 0 : 1.6 }))}
          >
            {params.outlier > 0 ? 'Remove the false closure' : 'Inject a false loop closure'}
          </ActionButton>
          <ActionButton onClick={() => sim.reseed()}>New odometry drift</ActionButton>
        </ButtonRow>
      </div>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.state.iterations}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
