'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { ellipse2, type Mat } from '@/lib/prob/linalg';
import { boxplus, normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import { diffDriveStep } from '@/lib/sim/world';
import {
  odomFromPoses,
  sampleMotionModelOdometry,
  sampleMotionModelVelocity,
  type MotionAlphas,
  type OdomAlphas,
  type VelocityCmd,
} from '@/lib/models/motion';
import { cartesianMoments, symEig, tangentContour, tangentMoments } from '@/lib/models/motion-se2';
import {
  clear,
  drawCovariance,
  drawGrid,
  drawParticles,
  drawPath,
  drawRobot,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w9.1 — the Banana Machine.
 *
 * One command, a thousand futures. The cloud is drawn by the real
 * `sampleMotionModelVelocity` / `sampleMotionModelOdometry` from lib/models,
 * and the two overlays are the two ways of pretending it is a Gaussian: the
 * moment-matched ellipse in (x, y), which leaks probability into places the
 * robot cannot reach, and the same-order Gaussian fitted in exponential
 * coordinates, which — pushed back through exp — bends with the banana.
 *
 * The statistic under the canvas is the one that settles the argument: the
 * skewness of the cross-track residual. It is exactly zero for *any* Gaussian
 * in (x, y). It is not zero here.
 */

const CLOUD_MAX = 2500;
const PER_TICK = 90;
/**
 * The readouts are third moments, whose standard error is √(6/n) — 0.05 at
 * 2500 draws. The reference cloud behind the "𝒩 in exp coords" tile is drawn
 * larger still, so that tile is a property of the *model* rather than of one
 * unlucky sample.
 */
const AUX = 4000;

type PresetId = 'straight' | 'gentle' | 'hard' | 'pivot';

const PRESETS: Record<PresetId, { label: string; cmd: VelocityCmd }> = {
  straight: { label: 'Straight', cmd: { v: 1, omega: 0, dt: 2 } },
  gentle: { label: 'Gentle arc', cmd: { v: 1, omega: 0.5, dt: 1 } },
  hard: { label: 'Hard arc', cmd: { v: 1, omega: 1.2, dt: 1.4 } },
  pivot: { label: 'Pivot', cmd: { v: 0.06, omega: 1.5, dt: 1 } },
};

const VELOCITY_BASE: MotionAlphas = [0.05, 0.05, 0.05, 0.05, 0.01, 0.01];
const ODOM_BASE: OdomAlphas = [0.05, 0.05, 0.05, 0.05];

/** What each α actually does to the picture — the slider's own caption. */
const VELOCITY_MEANING = [
  'α₁ — speed error from speed. Smears the cloud *along* the arc.',
  'α₂ — speed error from turning. Fast turns cost distance accuracy.',
  'α₃ — turn error from speed. Driving fast bends the fan open.',
  'α₄ — turn error from turning. Fans the arc: the banana lives here.',
  'α₅ — final-rotation slack from speed. Heading blur, positions unmoved.',
  'α₆ — final-rotation slack from turning. Heading blur, positions unmoved.',
];

const ODOM_MEANING = [
  'α₁ — rotation error from rotation. Both hinges of the linkage wobble.',
  'α₂ — rotation error from translation. Long drives corrupt the hinges.',
  'α₃ — translation error from translation. Pure along-track stretch.',
  'α₄ — translation error from rotation. Turning costs distance accuracy.',
];

interface Params {
  preset: PresetId;
  model: 'velocity' | 'odometry';
  channel: number;
  alpha: number;
  isolate: boolean;
  showEllipse: boolean;
  showTangent: boolean;
}

interface State {
  stamp: string;
  rng: Rng;
  samples: Pose2[];
}

const X0: Pose2 = { x: 0, y: 0, theta: 0 };

function buildAlphas(p: Params): { velocity: MotionAlphas; odom: OdomAlphas } {
  const vBase = (p.isolate ? [0, 0, 0, 0, 0, 0] : [...VELOCITY_BASE]) as MotionAlphas;
  const oBase = (p.isolate ? [0, 0, 0, 0] : [...ODOM_BASE]) as OdomAlphas;
  if (p.model === 'velocity') vBase[p.channel] = p.alpha;
  else oBase[Math.min(p.channel, 3)] = p.alpha;
  return { velocity: vBase, odom: oBase };
}

/**
 * Skewness of the residual perpendicular to `heading`. Exactly zero for any
 * Gaussian in (x, y), whatever the axis — which is what makes it a test rather
 * than a description.
 *
 * The axis is the *commanded* final heading, not each cloud's own mean, so the
 * two clouds compared under the canvas are measured against the same ruler.
 */
function crossTrackSkew(poses: Pose2[], heading: number): number {
  const n = poses.length;
  if (n < 30) return 0;
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  let mean = 0;
  const proj = poses.map((p) => -p.x * s + p.y * c);
  for (const q of proj) mean += q / n;
  let m2 = 0;
  let m3 = 0;
  for (const q of proj) {
    const d = q - mean;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= n;
  m3 /= n;
  return m2 < 1e-12 ? 0 : m3 / Math.pow(m2, 1.5);
}

/** Correlation between the cross-track residual and the heading residual. */
function crossHeadingCorrelation(poses: Pose2[], heading: number): number {
  const n = poses.length;
  if (n < 30) return 0;
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const d = poses.map((p) => [-p.x * s + p.y * c, normalizeAngle(p.theta - heading)]);
  const m = [0, 0];
  for (const row of d) {
    m[0] += row[0] / n;
    m[1] += row[1] / n;
  }
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (const row of d) {
    const a = row[0] - m[0];
    const b = row[1] - m[1];
    saa += a * a;
    sbb += b * b;
    sab += a * b;
  }
  const denom = Math.sqrt(saa * sbb);
  return denom < 1e-12 ? 0 : sab / denom;
}

/** Draw from a Gaussian fitted in the exponential chart at `origin`, then push through exp. */
function tangentModelCloud(
  origin: Pose2,
  mean: number[],
  cov: Mat,
  count: number,
  rng: Rng,
): Pose2[] {
  const { values, vectors } = symEig(cov);
  const sd = values.map((l) => Math.sqrt(Math.max(l, 0)));
  const out: Pose2[] = [];
  for (let i = 0; i < count; i++) {
    const z = [rng.normal(), rng.normal(), rng.normal()];
    const e = [mean[0], mean[1], mean[2]];
    for (let k = 0; k < 3; k++) {
      const a = sd[k] * z[k];
      e[0] += a * vectors[k][0];
      e[1] += a * vectors[k][1];
      e[2] += a * vectors[k][2];
    }
    out.push(boxplus(origin, [e[0], e[1], e[2]]));
  }
  return out;
}

export function BananaMachine() {
  const [params, setParams] = useState<Params>({
    preset: 'gentle',
    model: 'velocity',
    channel: 3,
    alpha: 0.09,
    isolate: false,
    showEllipse: true,
    showTangent: false,
  });

  const stamp = `${params.preset}|${params.model}|${params.channel}|${params.alpha}|${params.isolate}`;
  const cmd = PRESETS[params.preset].cmd;
  const nominal = useMemo(() => diffDriveStep(X0, cmd.v, cmd.omega, cmd.dt), [cmd]);
  const alphas = useMemo(() => buildAlphas(params), [params]);

  /** One draw from whichever model is selected. Both come straight from lib. */
  const drawOne = useCallback(
    (rng: Rng): Pose2 => {
      if (params.model === 'velocity') {
        return sampleMotionModelVelocity(cmd, X0, alphas.velocity, rng);
      }
      return sampleMotionModelOdometry(odomFromPoses(X0, nominal), X0, alphas.odom, rng);
    },
    [params.model, cmd, alphas, nominal],
  );

  const init = useCallback(
    (seed: number): State => ({ stamp, rng: new Rng(seed), samples: [] }),
    [stamp],
  );

  const step = useCallback(
    (s: State): State => {
      // A parameter moved: throw the cloud away rather than mixing two models.
      const base = s.stamp === stamp ? s : { stamp, rng: s.rng, samples: [] };
      if (base.samples.length >= CLOUD_MAX) return base;
      const samples = base.samples.slice();
      for (let i = 0; i < PER_TICK && samples.length < CLOUD_MAX; i++) {
        samples.push(drawOne(base.rng));
      }
      return { stamp, rng: base.rng, samples };
    },
    [stamp, drawOne],
  );

  const sim = useSimulation<State>({ init, step, fps: 12, initialSeed: 17 });

  const stats = useMemo(() => {
    const poses = sim.state.samples;
    if (poses.length < 30) {
      return null;
    }
    const cart = cartesianMoments(poses);
    const tan = tangentMoments(poses, X0);
    const modelCloud = tangentModelCloud(X0, tan.mean, tan.cov, AUX, new Rng(101));
    return {
      n: poses.length,
      cart,
      tan,
      skewTruth: crossTrackSkew(poses, nominal.theta),
      skewTangentModel: crossTrackSkew(modelCloud, nominal.theta),
      rhoCrossHeading: crossHeadingCorrelation(poses, nominal.theta),
    };
  }, [sim.state.samples, nominal.theta]);

  /** How curved does the command have to be before the ellipse becomes a lie? */
  const sweep = useMemo(() => {
    if (params.model !== 'velocity') return null;
    const pts: { x: number; y: number }[] = [];
    for (let k = 0; k <= 8; k++) {
      const omega = (2 * k) / 8;
      const u: VelocityCmd = { v: cmd.v, omega, dt: cmd.dt };
      const rng = new Rng(4242);
      const poses: Pose2[] = [];
      for (let i = 0; i < 2500; i++) poses.push(sampleMotionModelVelocity(u, X0, alphas.velocity, rng));
      pts.push({ x: omega, y: crossTrackSkew(poses, diffDriveStep(X0, u.v, u.omega, u.dt).theta) });
    }
    return pts;
  }, [params.model, cmd.v, cmd.dt, alphas.velocity]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 0.5);
      const poses = sim.state.samples;

      // The commanded arc — what the robot was told to do, and never does.
      const arc: Pose2[] = [];
      for (let k = 0; k <= 40; k++) {
        arc.push(diffDriveStep(X0, cmd.v, cmd.omega, (cmd.dt * k) / 40));
      }
      drawPath(ctx, v, arc, p.truth, { dashed: true, lineWidth: 1.75 });
      drawRobot(ctx, v, nominal, p.truth, 0.16, { filled: false });

      // The cloud: every dot is one execution of the same command.
      drawParticles(
        ctx,
        v,
        poses.map((state) => ({ state, weight: 1 })),
        p.prediction,
        { showHeading: true, maxRadius: 1.4 },
      );

      if (stats) {
        if (params.showEllipse) {
          const cov2: Mat = [
            [stats.cart.cov[0][0], stats.cart.cov[0][1]],
            [stats.cart.cov[1][0], stats.cart.cov[1][1]],
          ];
          ctx.setLineDash([5, 4]);
          drawCovariance(
            ctx,
            v,
            [stats.cart.mean[0], stats.cart.mean[1]],
            ellipse2(cov2, 2),
            p.prediction,
            { fill: false, lineWidth: 1.6 },
          );
          ctx.setLineDash([]);
          label(ctx, '2σ Gaussian in (x, y)', sx(v, -0.55), sy(v, -0.92), p.prediction, {
            size: 10,
            weight: 600,
          });
          label(ctx, 'dashed — the caricature', sx(v, -0.55), sy(v, -1.02), p.truth, { size: 9 });
        }

        if (params.showTangent) {
          const contour = tangentContour(X0, stats.tan.mean as [number, number, number], stats.tan.cov, 2, 120);
          drawPath(ctx, v, contour, p.prediction, { lineWidth: 2 });
          label(ctx, '2σ Gaussian in exp. coordinates', sx(v, 1.05), sy(v, -0.92), p.prediction, {
            size: 10,
            weight: 600,
          });
          label(ctx, 'solid — an ellipse, seen through exp', sx(v, 1.05), sy(v, -1.02), p.truth, {
            size: 9,
          });
        }
      }

      // Where the robot started. Blue: this is the prior the motion acts on.
      drawRobot(ctx, v, X0, p.prior, 0.18);
      label(ctx, `x₍ₜ₋₁₎`, sx(v, -0.05), sy(v, -0.3), p.prior, { size: 10, weight: 600 });
      label(
        ctx,
        params.model === 'velocity'
          ? `u = (v ${cmd.v.toFixed(2)} m/s, ω ${cmd.omega.toFixed(2)} rad/s, Δt ${cmd.dt.toFixed(1)} s)`
          : `u = (δrot1 ${odomFromPoses(X0, nominal).rot1.toFixed(2)}, δtrans ${odomFromPoses(X0, nominal).trans.toFixed(2)}, δrot2 ${odomFromPoses(X0, nominal).rot2.toFixed(2)})`,
        sx(v, -0.55),
        sy(v, 1.28),
        p.truth,
        { size: 10 },
      );
    },
    [sim.state.samples, stats, params.showEllipse, params.showTangent, params.model, cmd, nominal],
  );

  const meanings = params.model === 'velocity' ? VELOCITY_MEANING : ODOM_MEANING;
  const channel = Math.min(params.channel, meanings.length - 1);

  return (
    <WidgetFrame
      id="w9.1"
      title="The Banana Machine"
      teaches="Motion noise is not an ellipse around the predicted pose. It is a banana — and the banana is a perfectly ordinary Gaussian, drawn in the wrong coordinates."
      colorKey={['prior', 'prediction', 'truth']}
      wide
      caption={
        <>
          The gray dashed arc is the command. Every orange dot is one execution of{' '}
          <em>that same command</em>, sampled from the real{' '}
          <code>sample_motion_model_velocity</code>. Notice first that the cloud is bent and
          lopsided, not elliptical: rotational error early in the motion gets multiplied by the
          translation that follows. Then turn on the dashed{' '}
          <strong>Gaussian in (x, y)</strong> and watch it spill off the outside of the bend into
          poses no execution ever produced — that is the ellipse an EKF carries. Now turn on the
          solid <strong>Gaussian in exponential coordinates</strong>: same number of parameters, and
          it hugs the cloud. The readout that settles it is the cross-track skew, which is exactly
          zero for any Gaussian in (x, y) and is not zero here. Things to try: select α₄ and push it
          to the top (the fan), then α₆ (heading blur with the positions untouched); switch the
          preset to <strong>Pivot</strong> and see the banana vanish, because a banana needs
          translation to bend.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.7, maxX: 2.7, minY: -1.15, maxY: 1.45 }}
        draw={draw}
        deps={[sim.tick, sim.state, stats, params]}
        aspect={2.1}
        padding={0}
        ariaLabel="A cloud of sampled robot poses after a single commanded arc, forming a curved banana shape, with an elliptical Gaussian overlay that does not match it."
      />

      <div className="border-t border-fd-border px-3 py-3">
        <Dashboard columns={4}>
          <StatTile label="samples" value={stats?.n ?? 0} precision={0} />
          <StatTile
            label="cross-track skew · samples"
            value={stats ? stats.skewTruth : 0}
            role="prediction"
            precision={3}
          />
          <StatTile
            label="cross-track skew · 𝒩 in exp coords"
            value={stats ? stats.skewTangentModel : 0}
            role="prediction"
            precision={3}
          />
          <StatTile
            label="ρ(cross-track, heading)"
            value={stats ? stats.rhoCrossHeading : 0}
            precision={3}
          />
        </Dashboard>
        <p className="mt-2 font-ui text-[0.72rem] leading-snug text-fd-muted-foreground">
          A moment-matched Gaussian in (x, y) has a cross-track skew of exactly 0.000 — it cannot
          represent the two numbers on the left even in principle. The Gaussian fitted in
          exponential coordinates has the same three means and six covariance entries, and
          reproduces them.
        </p>
      </div>

      {sweep ? (
        <div className="border-t border-fd-border px-3 py-3">
          <LineChart
            series={[{ id: 'cross-track skew', role: 'prediction', data: sweep }]}
            xLabel="commanded turn rate ω (rad/s)"
            yLabel="skewness of the cross-track residual"
            height={190}
            markers={[
              { axis: 'y', value: 0, label: 'any Gaussian', role: 'truth' },
              { axis: 'x', value: cmd.omega, label: 'this command', role: 'prior' },
            ]}
            caption="Straight-line motion is very nearly Gaussian in (x, y). Curvature is what breaks it, and the break arrives well before the noise looks large."
            ariaLabel="A line chart showing the skewness of the cross-track residual growing in magnitude as the commanded turn rate increases from zero."
          />
        </div>
      ) : null}

      <ControlPanel columns={1} title="the command">
        <ButtonRow>
          {(Object.keys(PRESETS) as PresetId[]).map((id) => (
            <ActionButton
              key={id}
              emphasis={params.preset === id}
              onClick={() => setParams((p) => ({ ...p, preset: id }))}
            >
              {PRESETS[id].label}
            </ActionButton>
          ))}
          <span className="mx-1 h-4 w-px bg-fd-border" />
          <ActionButton
            emphasis={params.model === 'velocity'}
            onClick={() => setParams((p) => ({ ...p, model: 'velocity', channel: 3 }))}
          >
            velocity model
          </ActionButton>
          <ActionButton
            emphasis={params.model === 'odometry'}
            onClick={() => setParams((p) => ({ ...p, model: 'odometry', channel: 0 }))}
          >
            odometry model
          </ActionButton>
        </ButtonRow>
      </ControlPanel>

      <ControlPanel columns={1} title="noise">
        <ButtonRow>
          {meanings.map((_, i) => (
            <ActionButton
              key={i}
              emphasis={channel === i}
              onClick={() => setParams((p) => ({ ...p, channel: i }))}
            >
              {`α${'₁₂₃₄₅₆'[i]}`}
            </ActionButton>
          ))}
        </ButtonRow>
        <Slider
          label={meanings[channel]}
          role="prediction"
          value={params.alpha}
          min={0}
          max={0.35}
          step={0.005}
          onChange={(v) => setParams((p) => ({ ...p, alpha: v }))}
          help="The α's are variances, not standard deviations — doubling one widens the cloud by only √2."
        />
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <Toggle
            label="Isolate (all other α = 0)"
            checked={params.isolate}
            onChange={(v) => setParams((p) => ({ ...p, isolate: v }))}
          />
          <Toggle
            label="Gaussian in (x, y)"
            role="prediction"
            checked={params.showEllipse}
            onChange={(v) => setParams((p) => ({ ...p, showEllipse: v }))}
          />
          <Toggle
            label="Gaussian in exp. coordinates"
            role="prediction"
            checked={params.showTangent}
            onChange={(v) => setParams((p) => ({ ...p, showTangent: v }))}
          />
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
      />
    </WidgetFrame>
  );
}
