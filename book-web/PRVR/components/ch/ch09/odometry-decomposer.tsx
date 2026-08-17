'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import {
  applyOdom,
  odomFromPoses,
  sampleMotionModelOdometry,
  type OdomAlphas,
  type OdomDelta,
} from '@/lib/models/motion';
import { perturbOdomDelta } from '@/lib/models/motion-se2';
import {
  clear,
  drawGrid,
  drawParticles,
  drawPath,
  drawRobot,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w9.3 — the Odometry Decomposer.
 *
 * Any pose change is rotate, drive, rotate. The widget replays that linkage in
 * slow motion and then perturbs each hinge independently, which is precisely
 * the fiction `sample_motion_model_odometry` tells. Watching 250 ghosts replay
 * makes the consequence obvious: an error in the *first* hinge is multiplied by
 * everything that comes after it, so a long drive turns a degree of rotational
 * slop into tens of centimetres of lateral error.
 *
 * That is also the answer to "why not just add noise to (x, y, θ)?". Additive
 * Cartesian noise would put the same spread around a 10 cm nudge as around a
 * 3 m dash; the decomposition makes the error scale with what the robot did.
 */

const START: Pose2 = { x: 1.0, y: 1.55, theta: 0 };
const GHOSTS = 250;
const LINKAGES = 14;
const PHASES = 90;

interface State {
  phase: number;
}

/** Where the linkage is at animation fraction s: rotate, then drive, then rotate. */
function linkagePose(from: Pose2, d: OdomDelta, s: number): Pose2 {
  if (s < 0.28) {
    return { ...from, theta: normalizeAngle(from.theta + d.rot1 * (s / 0.28)) };
  }
  if (s < 0.82) {
    const f = (s - 0.28) / 0.54;
    return applyOdom(from, { rot1: d.rot1, trans: d.trans * f, rot2: 0 });
  }
  const f = (s - 0.82) / 0.18;
  return applyOdom(from, { rot1: d.rot1, trans: d.trans, rot2: d.rot2 * f });
}

function legName(s: number): string {
  if (s < 0.28) return 'δrot1 — turn to face the destination';
  if (s < 0.82) return 'δtrans — drive there';
  return 'δrot2 — turn to the final heading';
}

/** 1σ spread of the endpoint perpendicular to the direction of travel. */
function crossTrackSigma(poses: Pose2[], heading: number): number {
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  let mean = 0;
  const proj = poses.map((p) => -p.x * s + p.y * c);
  for (const q of proj) mean += q / proj.length;
  let acc = 0;
  for (const q of proj) acc += (q - mean) * (q - mean);
  return Math.sqrt(acc / Math.max(proj.length - 1, 1));
}

export function OdometryDecomposer() {
  const [target, setTarget] = useState({ x: 4.3, y: 2.35 });
  const [finalHeading, setFinalHeading] = useState(0.55);
  const [a1, setA1] = useState(0.06);
  const [a2, setA2] = useState(0.02);
  const [a3, setA3] = useState(0.05);
  const [a4, setA4] = useState(0.02);

  const alphas: OdomAlphas = useMemo(() => [a1, a2, a3, a4], [a1, a2, a3, a4]);
  const end: Pose2 = useMemo(
    () => ({ x: target.x, y: target.y, theta: finalHeading }),
    [target, finalHeading],
  );
  const delta = useMemo(() => odomFromPoses(START, end), [end]);

  const init = useCallback((): State => ({ phase: 0 }), []);
  const step = useCallback((_: State, tick: number): State => ({ phase: (tick % PHASES) / PHASES }), []);
  const sim = useSimulation<State>({ init, step, fps: 26, maxTicks: PHASES, loop: true, initialSeed: 31 });

  // The legs, not just the endpoints: the linkage replay needs the hinges the
  // sampler drew, so this uses the library's own perturbation rather than a
  // second copy of it.
  const ghosts = useMemo(() => {
    const rng = new Rng(sim.seed);
    const out: { pose: Pose2; delta: OdomDelta }[] = [];
    for (let i = 0; i < GHOSTS; i++) {
      const perturbed = perturbOdomDelta(delta, alphas, rng);
      out.push({ pose: applyOdom(START, perturbed), delta: perturbed });
    }
    return out;
  }, [delta, alphas, sim.seed]);

  /** Sanity anchor: the library sampler must land in the same cloud. */
  const librarySpread = useMemo(() => {
    const rng = new Rng(31);
    const poses: Pose2[] = [];
    for (let i = 0; i < 900; i++) poses.push(sampleMotionModelOdometry(delta, START, alphas, rng));
    return crossTrackSigma(poses, START.theta + delta.rot1);
  }, [delta, alphas]);

  /** How the lateral spread grows as the same rotation is followed by a longer drive. */
  const sweep = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let k = 0; k <= 9; k++) {
      const trans = 0.4 + (k * 4.6) / 9;
      const rng = new Rng(77);
      const poses: Pose2[] = [];
      for (let i = 0; i < 700; i++) {
        poses.push(sampleMotionModelOdometry({ ...delta, trans }, START, alphas, rng));
      }
      pts.push({ x: trans, y: crossTrackSigma(poses, START.theta + delta.rot1) });
    }
    return pts;
  }, [delta, alphas]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 0.5);
      const s = sim.state.phase;

      // A handful of ghost linkages, replaying in lockstep with the nominal one.
      for (let i = 0; i < LINKAGES; i++) {
        const g = ghosts[i];
        const via = applyOdom(START, { rot1: g.delta.rot1, trans: g.delta.trans, rot2: 0 });
        drawPath(ctx, v, [START, via, linkagePose(START, g.delta, s)], p.prediction, {
          lineWidth: 1,
          alpha: 0.22,
        });
      }

      // The nominal linkage: rotate, drive, rotate.
      const via = applyOdom(START, { rot1: delta.rot1, trans: delta.trans, rot2: 0 });
      drawPath(ctx, v, [START, via], p.truth, { dashed: true, lineWidth: 1.75 });

      // The first hinge, drawn as a wedge from the start heading.
      const rad = sl(v, 0.45);
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(
        sx(v, START.x),
        sy(v, START.y),
        rad,
        -Math.max(START.theta, START.theta + delta.rot1),
        -Math.min(START.theta, START.theta + delta.rot1),
      );
      ctx.stroke();
      label(ctx, `δrot1 = ${delta.rot1.toFixed(2)}`, sx(v, START.x) + 8, sy(v, START.y) - 26, p.truth, {
        size: 10,
      });
      label(
        ctx,
        `δtrans = ${delta.trans.toFixed(2)} m`,
        sx(v, (START.x + via.x) / 2),
        sy(v, (START.y + via.y) / 2) - 12,
        p.truth,
        { size: 10 },
      );
      label(ctx, `δrot2 = ${delta.rot2.toFixed(2)}`, sx(v, end.x) + 10, sy(v, end.y) + 18, p.truth, {
        size: 10,
      });

      // Every ghost endpoint: the cloud the three independent hinges produce.
      drawParticles(
        ctx,
        v,
        ghosts.map((g) => ({ state: g.pose, weight: 1 })),
        p.prediction,
        { showHeading: false, maxRadius: 1.6 },
      );

      drawRobot(ctx, v, linkagePose(START, delta, s), p.prediction, 0.2);
      drawRobot(ctx, v, end, p.truth, 0.2, { filled: false });
      drawRobot(ctx, v, START, p.prior, 0.2);

      label(ctx, legName(s), sx(v, 0.35), sy(v, 3.0), p.prediction, { size: 11, weight: 600 });
      label(ctx, 'drag the destination', sx(v, end.x) + 12, sy(v, end.y) - 14, p.truth, { size: 10 });
    },
    [sim.state.phase, ghosts, delta, end],
  );

  return (
    <WidgetFrame
      id="w9.3"
      title="The Odometry Decomposer"
      teaches="Odometry error is not additive in (x, y, θ). It enters as three independent hinges, and the first hinge is multiplied by everything after it."
      colorKey={['prior', 'prediction', 'truth']}
      caption={
        <>
          The wheels report that the robot went from the blue pose to the gray outline. Any such
          change is exactly one rotate–drive–rotate sequence, and the animation walks it. Now the
          model&rsquo;s one real assumption: each of those three legs is noisy{' '}
          <em>independently</em>. Watch the ghost linkages fan out and note where the fanning comes
          from — a wobble in δrot1 is a lever, and the drive that follows is its arm, so the endpoint
          cloud is an <em>arc</em>, not a blob. Drag the destination further away without touching a
          single slider and the cloud stretches sideways far faster than it stretches along the
          drive. That is the quadratic growth in the chart below, and it is why a robot that turns
          before a long corridor run is in much more trouble than one that turns after it.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0.2, maxX: 5.6, minY: 0.2, maxY: 3.2 }}
        draw={draw}
        deps={[sim.tick, ghosts, delta, end]}
        aspect={2.05}
        padding={0}
        cursor="grab"
        ariaLabel="A robot at the left, a draggable destination at the right, and a rotate-drive-rotate linkage replaying between them with a fan of noisy ghost linkages."
        onPointer={(world, phase) => {
          if (phase === 'down' || phase === 'move') {
            setTarget({ x: Math.max(1.3, world[0]), y: world[1] });
          }
        }}
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Readout label="σ(δrot1)" value={`${Math.sqrt(a1 * delta.rot1 ** 2 + a2 * delta.trans ** 2).toFixed(3)} rad`} />
        <Readout
          label="σ(δtrans)"
          value={`${Math.sqrt(a3 * delta.trans ** 2 + a4 * (delta.rot1 ** 2 + delta.rot2 ** 2)).toFixed(3)} m`}
        />
        <Readout label="cross-track σ" value={`${librarySpread.toFixed(3)} m`} />
        <Readout label="lever arm δtrans" value={`${delta.trans.toFixed(2)} m`} />
      </div>

      <div className="border-t border-fd-border px-3 py-3">
        <LineChart
          series={[{ id: 'cross-track σ of the endpoint', role: 'prediction', data: sweep }]}
          xLabel="δtrans — length of the drive (m)"
          yLabel="cross-track σ (m)"
          height={190}
          markers={[{ axis: 'x', value: delta.trans, label: 'this drive', role: 'prior' }]}
          caption="Same hinges, longer arm. σ(δrot1) itself grows with δtrans through α₂, and the lateral error is that angle times the distance — so the curve bends upward instead of rising in a straight line."
          ariaLabel="A line chart in which the lateral spread of the endpoint grows faster than linearly as the length of the drive increases."
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="α₁ — rotation error from rotation"
          role="prediction"
          value={a1}
          min={0}
          max={0.3}
          step={0.005}
          onChange={setA1}
          help="The headline slider: slop in both hinges."
        />
        <Slider
          label="α₂ — rotation error from translation"
          value={a2}
          min={0}
          max={0.12}
          step={0.002}
          onChange={setA2}
          help="Why the chart curves: a longer drive makes the hinges noisier too."
        />
        <Slider
          label="α₃ — translation error from translation"
          value={a3}
          min={0}
          max={0.3}
          step={0.005}
          onChange={setA3}
        />
        <Slider
          label="α₄ — translation error from rotation"
          value={a4}
          min={0}
          max={0.12}
          step={0.002}
          onChange={setA4}
        />
        <Slider
          label="Final heading"
          value={finalHeading}
          min={-Math.PI}
          max={Math.PI}
          step={0.05}
          unit="rad"
          onChange={setFinalHeading}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

function Readout({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
