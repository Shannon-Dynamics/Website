'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import {
  cameraCenter,
  depthSigma,
  parallaxAngle,
  projectPoint,
  rayWorld,
} from '@/lib/vision/pinhole';
import { baResiduals, baStep, type BaProblem } from '@/lib/vision/tiny-ba';
import { makeTwoViewScene, type TwoViewScene } from '@/lib/vision/scene';
import type { Pose3, Vec3 } from '@/lib/vision/se3';

/**
 * w18.1 — the Reprojection Playground.
 *
 * The one thing a camera does not measure is depth. Slide the estimate of a
 * point along the ray the first camera saw it on and that camera's residual
 * stays *exactly* zero — the pixel is a bearing, and every depth on the ray
 * explains it perfectly. The second camera is the only witness that objects,
 * and how loudly it objects is set by the baseline.
 *
 * Everything on screen runs the real library: `makeTwoViewScene` builds the
 * seeded scene, `projectPoint` places every dot, and the button hands the
 * problem to `baStep` — the same Schur-eliminating Gauss–Newton iteration the
 * chapter derives.
 */

/** The deep, nearly centred point: the depth-blindness poster child. */
const SEL = 0;
const N_POINTS = 7;

interface Params {
  baseline: number;
  sigmaPx: number;
}

interface State {
  est: Vec3[];
  /** True while the reader is driving; the autoplay sweep stands down. */
  held: boolean;
}

/* -------------------------------------------------------------------------- */
/* Canvas layout — one canvas, three panels                                    */
/* -------------------------------------------------------------------------- */

const ASPECT = 2.35;
const SCENE_BOX = { x0: 0.03, y0: 0.03, x1: 1.62, y1: 0.97 };
const PANE = [
  { x0: 1.7, y0: 0.515, x1: 2.32, y1: 0.975, name: 'camera 0 — the reference view' },
  { x0: 1.7, y0: 0.025, x1: 2.32, y1: 0.485, name: 'camera 1 — the witness' },
];
const Z_RANGE: [number, number] = [-0.7, 8.2];
const X_RANGE: [number, number] = [-2.1, 2.1];
const S = Math.min(
  (SCENE_BOX.x1 - SCENE_BOX.x0) / (Z_RANGE[1] - Z_RANGE[0]),
  (SCENE_BOX.y1 - SCENE_BOX.y0) / (X_RANGE[1] - X_RANGE[0]),
);
const PAD_X = (SCENE_BOX.x1 - SCENE_BOX.x0 - S * (Z_RANGE[1] - Z_RANGE[0])) / 2;
const PAD_Y = (SCENE_BOX.y1 - SCENE_BOX.y0 - S * (X_RANGE[1] - X_RANGE[0])) / 2;

/** Top-down view: depth runs right, lateral offset runs up. */
const sceneW = (zc: number, xc: number): [number, number] => [
  SCENE_BOX.x0 + PAD_X + (zc - Z_RANGE[0]) * S,
  SCENE_BOX.y0 + PAD_Y + (xc - X_RANGE[0]) * S,
];

/** Inverse of `sceneW`, for pointer drags. */
const sceneInv = (wx: number, wy: number): [number, number] => [
  (wx - SCENE_BOX.x0 - PAD_X) / S + Z_RANGE[0],
  (wy - SCENE_BOX.y0 - PAD_Y) / S + X_RANGE[0],
];

const paneW = (
  i: number,
  u: number,
  v: number,
  width: number,
  height: number,
): [number, number] => {
  const p = PANE[i];
  return [p.x0 + (u / width) * (p.x1 - p.x0), p.y1 - (v / height) * (p.y1 - p.y0)];
};

/* -------------------------------------------------------------------------- */

function problemFrom(scene: TwoViewScene, est: Vec3[]): BaProblem {
  return {
    cam: scene.cam,
    poses: scene.poses.map((p): Pose3 => ({ R: p.R.map((r) => [...r]), t: [...p.t] as Vec3 })),
    points: est.map((p) => [...p] as Vec3),
    obs: scene.obs,
    sigmaPx: Math.max(scene.sigmaPx, 0.1),
    // Both cameras are the gauge here: this widget is about structure, so the
    // poses are known and only the points move.
    fixedPoses: [true, true],
  };
}

export function ReprojectionPlayground() {
  const [params, setParams] = useState<Params>({ baseline: 0.5, sigmaPx: 1 });
  /** The seed of the feature detector's pixel noise; re-roll draws new matches. */
  const [seed, setSeed] = useState(18);

  const scene = useMemo(
    () =>
      makeTwoViewScene({
        baseline: params.baseline,
        sigmaPx: params.sigmaPx,
        seed,
        nPoints: N_POINTS,
      }),
    [params.baseline, params.sigmaPx, seed],
  );

  const init = useCallback(
    (): State => ({
      est: scene.initial.map((p) => [...p] as Vec3),
      held: false,
    }),
    [scene],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      if (s.held) return s;
      const z0 = scene.obs.find((o) => o.cam === 0 && o.pt === SEL)?.z;
      if (!z0) return s;
      // Sweep the estimate along the ray camera 0 actually measured. Camera 0's
      // residual is identically zero for every depth on this line — that is the
      // entire point of the widget.
      const dir = rayWorld(scene.cam, scene.poses[0], z0);
      const c0 = cameraCenter(scene.poses[0]);
      const truth = scene.truth[SEL];
      const trueDepth = Math.hypot(truth[0] - c0[0], truth[1] - c0[1], truth[2] - c0[2]);
      const depth = trueDepth * (1 + 0.4 * Math.sin((tick / 48) * Math.PI * 2));
      const est = s.est.map((p, j) =>
        j === SEL
          ? ([c0[0] + dir[0] * depth, c0[1] + dir[1] * depth, c0[2] + dir[2] * depth] as Vec3)
          : p,
      );
      return { ...s, est };
    },
    [scene],
  );

  const sim = useSimulation<State>({ init, step, fps: 24, initialSeed: 18 });

  // A new scene (different baseline, noise, or seed) means a new front-end
  // initialization: re-triangulate rather than keep stale structure.
  const setSimState = sim.setState;
  useEffect(() => {
    setSimState((s) => ({ ...s, est: scene.initial.map((p) => [...p] as Vec3), held: false }));
  }, [scene, setSimState]);

  const stats = useMemo(() => {
    const problem = problemFrom(scene, sim.state.est);
    const report = baResiduals(problem);
    const resid = [0, 1].map((c) => {
      const px = projectPoint(scene.cam, scene.poses[c], sim.state.est[SEL]);
      const z = scene.obs.find((o) => o.cam === c && o.pt === SEL)?.z;
      if (!px || !z) return 0;
      return Math.hypot(z[0] - px[0], z[1] - px[1]);
    });
    const Z = sim.state.est[SEL][2];
    return {
      rmse: report.rmse,
      r0: resid[0],
      r1: resid[1],
      sigmaZ: depthSigma(scene.cam, params.baseline, scene.truth[SEL][2], params.sigmaPx),
      parallax:
        (parallaxAngle(
          cameraCenter(scene.poses[0]),
          cameraCenter(scene.poses[1]),
          scene.truth[SEL],
        ) *
          180) /
        Math.PI,
      depthError: Z - scene.truth[SEL][2],
    };
  }, [scene, sim.state.est, params.baseline, params.sigmaPx]);

  const gnStep = useCallback(() => {
    sim.pause();
    sim.setState((s) => {
      const problem = problemFrom(scene, s.est);
      baStep(problem, { lambda: 1e-4 });
      return { ...s, est: problem.points, held: true };
    });
  }, [sim, scene]);

  const release = useCallback(() => {
    sim.setState((s) => ({ ...s, held: false }));
    sim.play();
  }, [sim]);

  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') return;
      const [zc, xc] = sceneInv(world[0], world[1]);
      if (zc < Z_RANGE[0] || zc > Z_RANGE[1] || xc < X_RANGE[0] || xc > X_RANGE[1]) return;
      sim.pause();
      sim.setState((s) => ({
        ...s,
        held: true,
        est: s.est.map((p, j) => (j === SEL ? ([xc, p[1], Math.max(zc, 0.4)] as Vec3) : p)),
      }));
    },
    [sim],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const est = sim.state.est;
      const cam = scene.cam;
      const centres = scene.poses.map((T) => cameraCenter(T));
      const W = (wx: number, wy: number): [number, number] => [sx(v, wx), sy(v, wy)];

      /* ---- top-down scene ------------------------------------------------ */
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(
        sx(v, SCENE_BOX.x0),
        sy(v, SCENE_BOX.y1),
        sl(v, SCENE_BOX.x1 - SCENE_BOX.x0),
        sl(v, SCENE_BOX.y1 - SCENE_BOX.y0),
      );
      label(
        ctx,
        'the scene, seen from above',
        sx(v, SCENE_BOX.x0 + 0.02),
        sy(v, SCENE_BOX.y1 - 0.04),
        p.truth,
        { size: 10 },
      );

      // The two measured bearings. Where they cross is the only depth both
      // cameras can agree on.
      for (let c = 0; c < 2; c++) {
        const z = scene.obs.find((o) => o.cam === c && o.pt === SEL)?.z;
        if (!z) continue;
        const dir = rayWorld(cam, scene.poses[c], z);
        const a = sceneW(centres[c][2], centres[c][0]);
        const b = sceneW(centres[c][2] + dir[2] * 9, centres[c][0] + dir[0] * 9);
        ctx.strokeStyle = p.measurement;
        ctx.globalAlpha = c === 0 ? 0.9 : 0.45;
        ctx.lineWidth = c === 0 ? 1.9 : 1.2;
        ctx.beginPath();
        ctx.moveTo(...W(a[0], a[1]));
        ctx.lineTo(...W(b[0], b[1]));
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Cameras: a field-of-view wedge and a body.
      for (let c = 0; c < 2; c++) {
        const cc = centres[c];
        const half = Math.atan2(cam.width / 2, cam.fx);
        const apex = sceneW(cc[2], cc[0]);
        for (const sgn of [-1, 1]) {
          const e = sceneW(cc[2] + 1.7 * Math.cos(half), cc[0] + sgn * 1.7 * Math.sin(half));
          ctx.strokeStyle = p.grid;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(...W(apex[0], apex[1]));
          ctx.lineTo(...W(e[0], e[1]));
          ctx.stroke();
        }
        ctx.fillStyle = c === 0 ? p.accent : p.prior;
        ctx.beginPath();
        ctx.arc(...W(apex[0], apex[1]), 5, 0, Math.PI * 2);
        ctx.fill();
        label(
          ctx,
          `cam ${c}`,
          sx(v, apex[0]) - 8,
          sy(v, apex[1]) + (c === 0 ? 13 : -13),
          c === 0 ? p.accent : p.prior,
          { size: 9.5, align: 'right' },
        );
      }

      // Truth (gray dashed) and current estimates (purple; blue for the rest).
      for (let j = 0; j < scene.truth.length; j++) {
        const t = sceneW(scene.truth[j][2], scene.truth[j][0]);
        ctx.strokeStyle = p.truth;
        ctx.setLineDash([2, 2]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(...W(t[0], t[1]), 4.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        const e = sceneW(est[j][2], est[j][0]);
        ctx.fillStyle = j === SEL ? p.posterior : p.prior;
        ctx.globalAlpha = j === SEL ? 1 : 0.7;
        ctx.beginPath();
        ctx.arc(...W(e[0], e[1]), j === SEL ? 5.5 : 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // The depth error of the selected point, drawn along the ray.
      {
        const t = sceneW(scene.truth[SEL][2], scene.truth[SEL][0]);
        const e = sceneW(est[SEL][2], est[SEL][0]);
        ctx.strokeStyle = p.posterior;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(...W(t[0], t[1]));
        ctx.lineTo(...W(e[0], e[1]));
        ctx.stroke();
        ctx.globalAlpha = 1;
        label(
          ctx,
          `${stats.depthError >= 0 ? '+' : ''}${stats.depthError.toFixed(2)} m of depth error`,
          sx(v, e[0]) + 10,
          sy(v, e[1]) - 11,
          p.posterior,
          { size: 9.5 },
        );
      }

      /* ---- the two image planes ------------------------------------------ */
      for (let c = 0; c < 2; c++) {
        const pane = PANE[c];
        ctx.fillStyle = p.free;
        ctx.fillRect(
          sx(v, pane.x0),
          sy(v, pane.y1),
          sl(v, pane.x1 - pane.x0),
          sl(v, pane.y1 - pane.y0),
        );
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.strokeRect(
          sx(v, pane.x0),
          sy(v, pane.y1),
          sl(v, pane.x1 - pane.x0),
          sl(v, pane.y1 - pane.y0),
        );
        label(ctx, pane.name, sx(v, pane.x0 + 0.015), sy(v, pane.y1 - 0.032), p.truth, {
          size: 9,
        });

        for (let j = 0; j < scene.truth.length; j++) {
          const z = scene.obs.find((o) => o.cam === c && o.pt === j)?.z;
          const px = projectPoint(cam, scene.poses[c], est[j]);
          if (!z || !px) continue;
          const a = paneW(c, z[0], z[1], cam.width, cam.height);
          const b = paneW(c, px[0], px[1], cam.width, cam.height);
          const sel = j === SEL;

          // The residual whisker: measurement (green) to prediction (purple).
          ctx.strokeStyle = p.posterior;
          ctx.globalAlpha = sel ? 0.95 : 0.35;
          ctx.lineWidth = sel ? 2 : 1;
          ctx.beginPath();
          ctx.moveTo(...W(a[0], a[1]));
          ctx.lineTo(...W(b[0], b[1]));
          ctx.stroke();
          ctx.globalAlpha = 1;

          const [mx, my] = W(a[0], a[1]);
          const r = sel ? 4.5 : 3;
          ctx.strokeStyle = p.measurement;
          ctx.lineWidth = sel ? 1.8 : 1.2;
          ctx.beginPath();
          ctx.moveTo(mx - r, my);
          ctx.lineTo(mx + r, my);
          ctx.moveTo(mx, my - r);
          ctx.lineTo(mx, my + r);
          ctx.stroke();

          ctx.fillStyle = p.posterior;
          ctx.globalAlpha = sel ? 1 : 0.45;
          ctx.beginPath();
          ctx.arc(...W(b[0], b[1]), sel ? 3.4 : 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }

        const res = c === 0 ? stats.r0 : stats.r1;
        label(
          ctx,
          `residual on the tracked point: ${res.toFixed(2)} px`,
          sx(v, pane.x1 - 0.015),
          sy(v, pane.y0 + 0.032),
          res < 0.05 ? p.measurement : p.posterior,
          { size: 9.5, align: 'right' },
        );
      }
    },
    [scene, sim.state.est, stats],
  );

  return (
    <WidgetFrame
      id="w18.1"
      title="Reprojection Playground"
      teaches="A pixel is a bearing, not a position: depth is only ever inferred from parallax, and the baseline sets the exchange rate."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The purple estimate of the tracked point slides along the ray camera&nbsp;0 actually
          measured. Watch camera&nbsp;0&rsquo;s residual: it stays pinned at 0.00&nbsp;px at every
          depth, because every point on that ray produces the identical pixel. Camera&nbsp;1 is the
          only witness that objects. Now pull the <strong>baseline</strong> down to 0.1&nbsp;m: the
          same metre of depth error costs a few pixels instead of tens, and the reported
          &sigma;<sub>Z</sub> jumps by the same factor. Drag the purple point <em>sideways</em> off
          the green ray and camera&nbsp;0 complains instantly. Press{' '}
          <strong>Gauss&ndash;Newton step</strong> to hand the whole structure to the real solver.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: ASPECT, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, sim.state, scene, stats]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="Left: a top-down view of two cameras observing seven points, with green measurement rays. Right: the two image planes, showing measured pixels as green crosses and predicted projections as purple dots joined by residual whiskers. As the estimated point slides along the first camera's ray, that camera's residual stays exactly zero while the second camera's residual grows."
        onPointer={onPointer}
        cursor="crosshair"
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="cam 0 residual" value={`${stats.r0.toFixed(2)} px`} tone="measurement" />
        <Stat label="cam 1 residual" value={`${stats.r1.toFixed(2)} px`} tone="posterior" />
        <Stat label="scene RMSE" value={`${stats.rmse.toFixed(2)} px`} />
        <Stat
          label="σ_Z from 1 px of noise"
          value={`${(stats.sigmaZ * 100).toFixed(1)} cm`}
          tone="posterior"
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Baseline between the cameras"
          role="prior"
          value={params.baseline}
          min={0.05}
          max={1.2}
          step={0.05}
          unit="m"
          onChange={(b) => setParams((q) => ({ ...q, baseline: b }))}
          help="The foreground parameter: parallax is the only thing that makes depth observable."
        />
        <Slider
          label="Pixel noise σ"
          role="measurement"
          value={params.sigmaPx}
          min={0.2}
          max={3}
          step={0.1}
          unit="px"
          onChange={(s) => setParams((q) => ({ ...q, sigmaPx: s }))}
        />
        <ButtonRow>
          <ActionButton onClick={gnStep} emphasis>
            Gauss–Newton step
          </ActionButton>
          <ActionButton onClick={release}>Resume the depth sweep</ActionButton>
        </ButtonRow>
        <p className="self-center font-ui text-[0.72rem] text-fd-muted-foreground">
          Parallax at the tracked point: {stats.parallax.toFixed(2)}°
        </p>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => setSeed(Math.floor(Math.random() * 100000))}
        seed={seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function Stat({
  label: l,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: 'measurement' | 'posterior' | 'prior';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={tone ? { color: `var(--pr-${tone})` } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
