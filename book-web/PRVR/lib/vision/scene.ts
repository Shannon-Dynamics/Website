/**
 * Synthetic visual-inertial scenes — the data the Chapter 18 widgets run on.
 *
 * Everything here is seeded and analytic: the geometry has a closed-form truth
 * so a residual of zero means the estimator is right, not lucky. Two scenes:
 *
 *  • `makeTwoViewScene` — a stereo pair looking at a wall of points, used by the
 *    Reprojection Playground. The initial estimate comes from DLT triangulation
 *    of the *noisy* pixels, so the reader starts where a real front end would.
 *  • `makeImuArc` — a constant-radius, constant-speed turn. On this trajectory
 *    the true IMU output is *constant*: ω̃ = (0,0,ω) and ã = (0, rω², 9.81),
 *    which makes every preintegration number in the chapter checkable by hand.
 */

import { transpose } from '../prob/linalg';
import { Rng } from '../prob/rng';
import { pinhole, projectPoint, triangulateDlt, type Pinhole } from './pinhole';
import { GRAVITY, type ImuBias, type ImuNoise, type ImuSample, type NavState } from './preint';
import { I3, apply, rotZ, type Pose3, type Vec3 } from './se3';
import type { ReprojObs } from './tiny-ba';

/* -------------------------------------------------------------------------- */
/* Two-view scene                                                              */
/* -------------------------------------------------------------------------- */

export interface TwoViewOptions {
  /** Stereo baseline in metres — the parallax knob. */
  baseline?: number;
  /** Pixel noise standard deviation of the feature detector. */
  sigmaPx?: number;
  seed?: number;
  nPoints?: number;
}

export interface TwoViewScene {
  /** Identity of this scene; widgets rebuild when it changes. */
  sig: string;
  cam: Pinhole;
  /** World-to-camera poses of the left and right cameras. */
  poses: Pose3[];
  /** Ground-truth point positions. */
  truth: Vec3[];
  /** Noisy pixel observations, two per point. */
  obs: ReprojObs[];
  /** The front end's initial guess: DLT triangulation of the noisy pixels. */
  initial: Vec3[];
  baseline: number;
  sigmaPx: number;
}

/** A 640×480 camera with a 60°-ish horizontal field of view. */
export const DEFAULT_CAM = pinhole(400, 400, 320, 240, 640, 480);

export function makeTwoViewScene(opts: TwoViewOptions = {}): TwoViewScene {
  const baseline = opts.baseline ?? 0.5;
  const sigmaPx = opts.sigmaPx ?? 1;
  const seed = opts.seed ?? 18;
  const nPoints = opts.nPoints ?? 9;
  const cam = DEFAULT_CAM;
  const rng = new Rng(seed);

  // Both cameras look along +Z; the right one is displaced by the baseline.
  const poses: Pose3[] = [
    { R: I3(), t: [0, 0, 0] },
    { R: I3(), t: [-baseline, 0, 0] },
  ];

  const truth: Vec3[] = [];
  for (let i = 0; i < nPoints; i++) {
    const z = 3 + 4 * rng.next();
    const x = (rng.next() - 0.5) * 1.7 * (z / 5);
    const y = (rng.next() - 0.5) * 1.2 * (z / 5);
    truth.push([x, y, z]);
  }
  // One point deliberately deep and centred: the depth-blind poster child.
  truth[0] = [0.2, -0.1, 5];

  const obs: ReprojObs[] = [];
  for (let j = 0; j < truth.length; j++) {
    for (let k = 0; k < poses.length; k++) {
      const z = projectPoint(cam, poses[k], truth[j]);
      if (!z) continue;
      obs.push({
        cam: k,
        pt: j,
        z: [z[0] + rng.normal(0, sigmaPx), z[1] + rng.normal(0, sigmaPx)],
      });
    }
  }

  const initial = truth.map((_, j) => {
    const views = obs.filter((o) => o.pt === j).map((o) => ({ tcw: poses[o.cam], z: o.z }));
    return views.length >= 2 ? triangulateDlt(cam, views) : [...truth[j]] as Vec3;
  });

  return {
    sig: `${baseline.toFixed(3)}|${sigmaPx.toFixed(3)}|${seed}|${nPoints}`,
    cam,
    poses,
    truth,
    obs,
    initial,
    baseline,
    sigmaPx,
  };
}

/* -------------------------------------------------------------------------- */
/* IMU trajectory                                                              */
/* -------------------------------------------------------------------------- */

export interface ImuArcOptions {
  /** Interval length in seconds — the keyframe spacing. */
  seconds?: number;
  /** Sample rate in Hz. */
  rate?: number;
  /** Turn rate about the world z-axis, rad/s. */
  omega?: number;
  /** Turn radius, metres. */
  radius?: number;
  bias?: ImuBias;
  noise?: ImuNoise;
  seed?: number;
}

export interface ImuArc {
  samples: ImuSample[];
  dt: number;
  /** The exact navigation state at the start and end of the interval. */
  start: NavState;
  end: NavState;
  trueBias: ImuBias;
  noise: ImuNoise;
  omega: number;
  radius: number;
  seconds: number;
}

/** Analytic state on the constant-turn trajectory at time `t`. */
export function arcState(t: number, omega: number, radius: number): NavState {
  const psi = omega * t;
  return {
    R: rotZ(psi),
    p: [radius * Math.sin(psi), radius * (1 - Math.cos(psi)), 0],
    v: [radius * omega * Math.cos(psi), radius * omega * Math.sin(psi), 0],
  };
}

/**
 * Sample a constant-turn arc.
 *
 * The clean part: because the turn is steady, the *true* body-frame
 * measurements do not depend on time at all — a constant angular rate and a
 * constant specific force whose z component is exactly −g. Everything the
 * widgets show as motion in the deltas is therefore integration, not a
 * changing input.
 */
export function makeImuArc(opts: ImuArcOptions = {}): ImuArc {
  const seconds = opts.seconds ?? 1;
  const rate = opts.rate ?? 200;
  const omega = opts.omega ?? 0.5;
  const radius = opts.radius ?? 2;
  const bias = opts.bias ?? { gyro: [0.004, -0.002, 0.006], acc: [0.02, 0.01, -0.03] };
  const noise = opts.noise ?? { gyro: 0.0035, acc: 0.02 };
  const rng = new Rng(opts.seed ?? 0x5ee3);
  const dt = 1 / rate;
  const n = Math.max(1, Math.round(seconds * rate));

  const trueGyro: Vec3 = [0, 0, omega];
  const trueAcc: Vec3 = [0, radius * omega * omega, -GRAVITY[2]];

  // Discrete noise: the datasheet density divided by √Δt.
  const sg = noise.gyro / Math.sqrt(dt);
  const sa = noise.acc / Math.sqrt(dt);

  const samples: ImuSample[] = [];
  for (let k = 0; k < n; k++) {
    samples.push({
      gyro: [
        trueGyro[0] + bias.gyro[0] + rng.normal(0, sg),
        trueGyro[1] + bias.gyro[1] + rng.normal(0, sg),
        trueGyro[2] + bias.gyro[2] + rng.normal(0, sg),
      ],
      acc: [
        trueAcc[0] + bias.acc[0] + rng.normal(0, sa),
        trueAcc[1] + bias.acc[1] + rng.normal(0, sa),
        trueAcc[2] + bias.acc[2] + rng.normal(0, sa),
      ],
    });
  }

  return {
    samples,
    dt,
    start: arcState(0, omega, radius),
    end: arcState(n * dt, omega, radius),
    trueBias: bias,
    noise,
    omega,
    radius,
    seconds: n * dt,
  };
}

/* -------------------------------------------------------------------------- */
/* A ring of cameras — the bundle-adjustment worked example                     */
/* -------------------------------------------------------------------------- */

export interface RingScene {
  cam: Pinhole;
  truthPoses: Pose3[];
  truthPoints: Vec3[];
  obs: ReprojObs[];
}

/**
 * `nCams` cameras on a ring at radius `r`, all looking at the origin, viewing a
 * cube of points. This is the scene the chapter's Rust example bundle-adjusts.
 */
export function makeRingScene(
  opts: { nCams?: number; nPoints?: number; radius?: number; sigmaPx?: number; seed?: number } = {},
): RingScene {
  const nCams = opts.nCams ?? 6;
  const nPoints = opts.nPoints ?? 24;
  const radius = opts.radius ?? 4;
  const sigmaPx = opts.sigmaPx ?? 1;
  const rng = new Rng(opts.seed ?? 0x5ee3d);
  const cam = DEFAULT_CAM;

  const truthPoints: Vec3[] = [];
  for (let i = 0; i < nPoints; i++) {
    truthPoints.push([
      (rng.next() - 0.5) * 1.6,
      (rng.next() - 0.5) * 1.6,
      (rng.next() - 0.5) * 1.6,
    ]);
  }

  const truthPoses: Pose3[] = [];
  for (let k = 0; k < nCams; k++) {
    const a = (2 * Math.PI * k) / nCams;
    const eye: Vec3 = [radius * Math.sin(a), 0.6 * Math.sin(2 * a), -radius * Math.cos(a)];
    // Camera looks at the origin: build R_wc from the forward axis, then invert.
    const f = norm3([-eye[0], -eye[1], -eye[2]]);
    const right = norm3(cross3([0, -1, 0], f));
    const down = cross3(f, right);
    const rwc = [
      [right[0], down[0], f[0]],
      [right[1], down[1], f[1]],
      [right[2], down[2], f[2]],
    ];
    const rcw = transpose(rwc);
    truthPoses.push({ R: rcw, t: neg3(apply(rcw, eye)) });
  }

  const obs: ReprojObs[] = [];
  for (let k = 0; k < nCams; k++) {
    for (let j = 0; j < nPoints; j++) {
      const z = projectPoint(cam, truthPoses[k], truthPoints[j]);
      if (!z || z[0] < 0 || z[0] > cam.width || z[1] < 0 || z[1] > cam.height) continue;
      obs.push({
        cam: k,
        pt: j,
        z: [z[0] + rng.normal(0, sigmaPx), z[1] + rng.normal(0, sigmaPx)],
      });
    }
  }

  return { cam, truthPoses, truthPoints, obs };
}

const cross3 = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm3 = (a: Vec3): Vec3 => {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
const neg3 = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];
