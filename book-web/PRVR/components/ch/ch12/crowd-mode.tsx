'use client';

import { useCallback, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { ParticleFilter } from '@/lib/filters/pf';
import { SurpriseDetector, augmentedResample, dominantCluster, perBeamLikelihood } from '@/lib/localize/augmented-mcl';
import { filterScan, subsamplePoses, type BeamVerdict } from '@/lib/localize/dynamic';
import {
  DEFAULT_BEAM_PARAMS,
  LikelihoodField,
  logLikelihoodFieldRangeFinderModel,
} from '@/lib/models/sensor';
import { odomFromPoses, sampleMotionModelOdometry, type OdomAlphas } from '@/lib/models/motion';
import { RUSTY, RUSTY_LIDAR, diffDriveSlipStep, pursuePoint, raycastScan } from '@/lib/sim/rusty';
import { APARTMENT, isFree, type Segment, type World } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';
import { Rng } from '@/lib/prob/rng';
import {
  clear,
  drawRobot,
  drawWorld,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w12.5 — Crowd Mode.
 *
 * The static-world assumption, violated on purpose. Four people walk the
 * corridor; the map does not know about them, and every beam that lands on a
 * body comes back shorter than the map predicts. Left alone, the filter treats
 * those readings as evidence that it is somewhere else, and drags itself toward
 * whatever pose explains a corridor full of unexpected walls.
 *
 * `test_range_measurement` (Table 8.4) asks, per beam, how much of the reading's
 * likelihood is owed to the model's *short* component, and drops the beams where
 * that share is large. The asymmetry is the point: a surprisingly **short**
 * reading is probably a person, while a surprisingly **long** one cannot be —
 * so long readings always survive, and they are exactly what tells a lost filter
 * that it is lost.
 */

const FIELD = new LikelihoodField(APARTMENT, 0.05);
const LIDAR = { ...RUSTY_LIDAR, nBeams: 32, sigmaR: 0.03, pDropout: 0.01 };
const BEAM = {
  ...DEFAULT_BEAM_PARAMS,
  sigmaHit: 0.18,
  zShort: 0.14,
  lambdaShort: 1.2,
  maxRange: LIDAR.maxRange,
};
const ALPHAS: OdomAlphas = [0.03, 0.02, 0.05, 0.02];
const DT = 0.4;
const M = 700;
/** Table 8.4's X̄ₜ: a representative subsample, not the whole set. */
const TEST_POSES = 48;

const ROUTE = [
  { x: 2.4, y: 4.35 },
  { x: 9.6, y: 4.35 },
  { x: 2.4, y: 4.35 },
];

interface Walker {
  x: number;
  y: number;
  vx: number;
  radius: number;
}

/** A person, as four short wall segments — cheap for the ray caster, and the
 *  right shape for "an obstacle the map has never heard of". */
function walkerSegments(w: Walker): Segment[] {
  const r = w.radius;
  return [
    { x1: w.x - r, y1: w.y - r, x2: w.x + r, y2: w.y - r },
    { x1: w.x + r, y1: w.y - r, x2: w.x + r, y2: w.y + r },
    { x1: w.x + r, y1: w.y + r, x2: w.x - r, y2: w.y + r },
    { x1: w.x - r, y1: w.y + r, x2: w.x - r, y2: w.y - r },
  ];
}

/** Closing room B's doorway: a genuine change to the map, not a person. */
const CLOSED_DOOR: Segment = { x1: 5.5, y1: 3.8, x2: 6.5, y2: 3.8 };

interface Params {
  filterOn: boolean;
  chi: number;
  door: boolean;
}

interface State {
  rng: Rng;
  world: Rng;
  pf: ParticleFilter;
  detector: SurpriseDetector;
  truth: Pose2;
  wheels: [number, number];
  waypoint: number;
  walkers: Walker[];
  scan: number[];
  angles: number[];
  verdicts: BeamVerdict[];
  rejected: number;
  estimate: Pose2;
  error: number;
  rmse: number;
  n: number;
}

function makeState(seed: number): State {
  const rng = new Rng(seed);
  const truth: Pose2 = { x: 2.4, y: 4.35, theta: 0 };
  return {
    rng,
    world: new Rng(seed * 977 + 3),
    // Position tracking, not global localization: this widget is about staying
    // localized while the world misbehaves.
    pf: ParticleFilter.gaussian(M, truth, { x: 0.15, y: 0.15, theta: 0.08 }, rng),
    detector: new SurpriseDetector({ alphaFast: 0.4, alphaSlow: 0.04 }),
    truth,
    wheels: [0, 0],
    waypoint: 1,
    walkers: [
      { x: 4.2, y: 4.2, vx: 0.32, radius: 0.22 },
      { x: 6.0, y: 4.6, vx: -0.28, radius: 0.2 },
      { x: 8.4, y: 4.3, vx: 0.22, radius: 0.24 },
      { x: 10.4, y: 4.6, vx: -0.34, radius: 0.2 },
    ],
    scan: [],
    angles: [],
    verdicts: [],
    rejected: 0,
    estimate: { ...truth },
    error: 0,
    rmse: 0,
    n: 0,
  };
}

export function CrowdMode() {
  const [params, setParams] = useState<Params>({ filterOn: true, chi: 0.15, door: false });
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const init = useCallback((seed: number) => makeState(seed), []);

  const step = useCallback((s: State, _tick: number): State => {
    const p = paramsRef.current;
    const { pf, rng, world } = s;
    const next: State = { ...s };

    // ---- people ----------------------------------------------------------
    next.walkers = s.walkers.map((w) => {
      let x = w.x + w.vx * DT;
      let vx = w.vx;
      if (x < 1.0 || x > 11.0) {
        vx = -vx;
        x = w.x + vx * DT;
      }
      return { ...w, x, vx };
    });

    // The world the *sensor* lives in: map plus people plus (optionally) a door
    // that has been closed since the map was made.
    const sensedWorld: World = {
      ...APARTMENT,
      walls: [
        ...APARTMENT.walls,
        ...next.walkers.flatMap(walkerSegments),
        ...(p.door ? [CLOSED_DOOR] : []),
      ],
    };

    // ---- the robot -------------------------------------------------------
    const wp = ROUTE[next.waypoint % ROUTE.length];
    const u = pursuePoint(s.truth, wp, { speed: 0.45, gain: 1.6, maxOmega: 1.0 });
    const out = diffDriveSlipStep(
      { pose: s.truth, wheelAngles: s.wheels },
      u,
      DT,
      APARTMENT,
      RUSTY,
      world,
    );
    next.truth = out.pose;
    next.wheels = out.wheelAngles;
    if (Math.hypot(out.pose.x - wp.x, out.pose.y - wp.y) < 0.5 || out.blocked) {
      next.waypoint = (s.waypoint + 1) % ROUTE.length;
    }
    const odom = odomFromPoses(s.truth, out.pose);
    pf.predict((x) => sampleMotionModelOdometry(odom, x, ALPHAS, rng));

    // ---- sensing, and the novelty test -----------------------------------
    const scan = raycastScan(sensedWorld, out.pose, LIDAR, world);
    next.scan = scan.ranges;
    next.angles = scan.angles;

    // The test is run against the *map*, using the predicted particles — the
    // filter never gets to see the people it is filtering out.
    const testPoses = subsamplePoses(pf.particles, TEST_POSES);
    const verdicts = p.filterOn
      ? filterScan(scan.ranges, scan.angles, testPoses, APARTMENT, BEAM, p.chi)
      : scan.ranges.map(() => ({ pShort: 0, pHit: 1, reject: false }));
    next.verdicts = verdicts;
    next.rejected = verdicts.filter((v) => v.reject).length;

    const z: number[] = [];
    const angles: number[] = [];
    for (let k = 0; k < scan.ranges.length; k += 2) {
      if (verdicts[k].reject) continue;
      z.push(scan.ranges[k]);
      angles.push(scan.angles[k]);
    }

    if (z.length > 0) {
      const raw: number[] = [];
      pf.correctLog((x) => {
        const l = logLikelihoodFieldRangeFinderModel(z, x, FIELD, BEAM, angles, LIDAR.offset);
        raw.push(l);
        return l;
      });
      let wAvg = 0;
      for (const l of raw) wAvg += perBeamLikelihood(l, z.length);
      wAvg /= raw.length;
      const pInject = s.detector.update(wAvg);
      const cluster = dominantCluster(pf.particles, 0.7);
      next.estimate = cluster.pose;
      next.error = Math.hypot(cluster.pose.x - out.pose.x, cluster.pose.y - out.pose.y);
      next.n = s.n + 1;
      next.rmse = Math.sqrt((s.rmse * s.rmse * s.n + next.error * next.error) / next.n);

      if (pInject > 0) {
        const res = augmentedResample(pf.particles, rng, pInject, (r) => {
          for (let g = 0; g < 200; g++) {
            const x = r.uniform(0, 12);
            const y = r.uniform(0, 9);
            if (isFree(APARTMENT, x, y, 0.2)) return { x, y, theta: r.uniform(-Math.PI, Math.PI) };
          }
          return { x: 6, y: 4.4, theta: 0 };
        });
        pf.particles = res.particles;
      } else {
        pf.resample(rng, 'lowVariance');
      }
    }
    return next;
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 5, initialSeed: 4 });
  const s = sim.state;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawWorld(ctx, v, APARTMENT, p);

      if (paramsRef.current.door) {
        // Truth gray, like the walkers: this is a real obstacle that the map
        // knows nothing about.
        ctx.save();
        ctx.strokeStyle = p.truth;
        ctx.lineWidth = 3.5;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, CLOSED_DOOR.x1), sy(v, CLOSED_DOOR.y1));
        ctx.lineTo(sx(v, CLOSED_DOOR.x2), sy(v, CLOSED_DOOR.y2));
        ctx.stroke();
        ctx.restore();
      }

      // The people: drawn as gray outlines, because to the map they do not exist.
      ctx.save();
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.5;
      for (const w of s.walkers) {
        ctx.beginPath();
        ctx.arc(sx(v, w.x), sy(v, w.y), Math.max(3, (w.radius / 12) * v.width * 0.9), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      // Beams: accepted in measurement green, rejected as dim stubs.
      ctx.save();
      for (let k = 0; k < s.scan.length; k++) {
        const rejected = s.verdicts[k]?.reject;
        const a = s.truth.theta + s.angles[k];
        const r = Math.min(s.scan[k], LIDAR.maxRange);
        ctx.strokeStyle = rejected ? p.truth : p.measurement;
        ctx.globalAlpha = rejected ? 0.55 : 0.25;
        ctx.lineWidth = rejected ? 1.4 : 1;
        ctx.setLineDash(rejected ? [2, 3] : []);
        ctx.beginPath();
        ctx.moveTo(sx(v, s.truth.x), sy(v, s.truth.y));
        ctx.lineTo(sx(v, s.truth.x + r * Math.cos(a)), sy(v, s.truth.y + r * Math.sin(a)));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.restore();

      ctx.save();
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.45;
      for (let i = 0; i < s.pf.particles.length; i += 2) {
        const q = s.pf.particles[i];
        ctx.beginPath();
        ctx.arc(sx(v, q.state.x), sy(v, q.state.y), 1.3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      drawRobot(ctx, v, s.estimate, p.posterior, 0.28);
      ctx.save();
      ctx.setLineDash([4, 3]);
      drawRobot(ctx, v, s.truth, p.truth, 0.32, { filled: false });
      ctx.restore();

      label(
        ctx,
        paramsRef.current.filterOn
          ? `test_range_measurement ON · ${s.rejected} / ${s.scan.length} beams rejected`
          : 'test_range_measurement OFF · every beam believed',
        8,
        12,
        paramsRef.current.filterOn ? p.measurement : p.prediction,
        { size: 10, weight: 600 },
      );
    },
    [s],
  );

  return (
    <WidgetFrame
      id="w12.5"
      title="Crowd Mode"
      teaches="A good sensor model does not handle people: the four-way mixture explains a short reading, it does not stop that reading from moving the belief. Rejecting the short-dominated beams does."
      colorKey={['measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Four people walk the corridor. The map has never heard of them, so every beam that lands
          on a body comes back short. Beams the filter accepts are drawn green; beams
          <code> test_range_measurement</code> rejects are drawn as dashed gray stubs — watch them
          track the walkers.
          <br />
          <strong>What to try.</strong> Turn the filter off and follow the RMSE readout: in this
          corridor the crowd drags the cloud clean off the truth within a dozen steps. Turn it back
          on and it holds. Now raise χ<sub>reject</sub>: at 0.6 barely a fifth of the beams are
          rejected and the estimate delocalizes almost as badly as with the filter off — the
          threshold is not cosmetic. Finally, tick <em>close room B&apos;s door</em>. That is a real
          change to the map, it also reads short, and the same test throws it away: the filter is
          now blind to the one thing it should have learned. Rejecting the unexpected is a trade,
          not a free lunch.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, s]}
        aspect={12 / 8}
        padding={0.3}
        ariaLabel="The apartment corridor with four moving people; laser beams that hit a person are drawn as dashed gray stubs and are excluded from the filter update."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="beams rejected" value={`${s.rejected} / ${s.scan.length}`} />
        <Stat label="|est − truth|" value={`${s.error.toFixed(2)} m`} />
        <Stat label="RMSE" value={`${s.rmse.toFixed(2)} m`} />
        <Stat label="M" value={String(s.pf.size)} />
      </div>

      <ControlPanel columns={3}>
        <Toggle
          label="test_range_measurement"
          role="measurement"
          checked={params.filterOn}
          onChange={(v) => setParams((q) => ({ ...q, filterOn: v }))}
        />
        <Slider
          label="χ_reject"
          role="measurement"
          value={params.chi}
          min={0.05}
          max={0.9}
          step={0.05}
          onChange={(v) => setParams((q) => ({ ...q, chi: v }))}
          help="Reject a beam when the posterior probability that it was caused by an unmodelled object exceeds this."
        />
        <Toggle
          label="Close room B's door"
          role="truth"
          checked={params.door}
          onChange={(v) => setParams((q) => ({ ...q, door: v }))}
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
      />
    </WidgetFrame>
  );
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
