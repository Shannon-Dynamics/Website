'use client';

import { useCallback, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import {
  ActionButton,
  ButtonRow,
  ControlPanel,
  Slider,
  Toggle,
  Transport,
} from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { ParticleFilter, type Particle } from '@/lib/filters/pf';
import {
  SurpriseDetector,
  augmentedResample,
  countClusters,
  dominantCluster,
  perBeamLikelihood,
} from '@/lib/localize/augmented-mcl';
import {
  DEFAULT_BEAM_PARAMS,
  LikelihoodField,
  logBeamRangeFinderModel,
  logLikelihoodFieldRangeFinderModel,
} from '@/lib/models/sensor';
import { odomFromPoses, sampleMotionModelOdometry, type OdomAlphas } from '@/lib/models/motion';
import {
  RUSTY,
  RUSTY_LIDAR,
  diffDriveSlipStep,
  encoderTicks,
  integrateOdometry,
  odometryDelta,
  pursuePoint,
  raycastScan,
  type EncoderTicks,
  type LidarParams,
} from '@/lib/sim/rusty';
import { APARTMENT, isFree } from '@/lib/sim/world';
import { poseDistance, type Pose2 } from '@/lib/geom/se2';
import { Rng } from '@/lib/prob/rng';
import {
  clear,
  drawRobot,
  drawWorld,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w12.1 — the MCL Theater.
 *
 * The book's centrepiece. Rusty wakes up with no idea where it is: particles
 * cover every free square metre of the Apartment. Driving and sensing collapse
 * that fog through a genuine two-room ambiguity into one hypothesis — and then
 * the reader presses Kidnap and watches plain MCL fail, confidently, forever,
 * while Augmented MCL notices its own surprise and rebuilds itself.
 *
 * Every number on screen comes from the library: `ParticleFilter` (Ch. 8),
 * `sample_motion_model_odometry` (Ch. 9), the likelihood field and beam model
 * (Ch. 10), and `lib/localize/augmented-mcl.ts` for the w_fast / w_slow
 * detector. The widget itself only choreographs and draws.
 */

/** Built once: the distance transform every particle's weight is looked up in. */
const FIELD = new LikelihoodField(APARTMENT, 0.05);

const LIDAR: LidarParams = { ...RUSTY_LIDAR, nBeams: 24, sigmaR: 0.03, pDropout: 0.02 };
/** Every second beam goes to the filter — beam subsampling, Chapter 10. */
const BEAM_STRIDE = 2;

const BEAM_PARAMS = { ...DEFAULT_BEAM_PARAMS, sigmaHit: 0.18, maxRange: LIDAR.maxRange };

const ALPHAS: Record<'good' | 'sloppy' | 'drunk', OdomAlphas> = {
  good: [0.008, 0.008, 0.015, 0.008],
  sloppy: [0.05, 0.03, 0.06, 0.03],
  drunk: [0.2, 0.1, 0.18, 0.1],
};

/** The preset route: out of room A, east down the corridor, north into room E. */
const ROUTE: { x: number; y: number }[] = [
  { x: 2.05, y: 2.4 },
  { x: 2.05, y: 4.35 },
  { x: 5.6, y: 4.35 },
  { x: 7.85, y: 4.35 },
  { x: 7.85, y: 6.4 },
  { x: 7.85, y: 4.35 },
  { x: 4.0, y: 4.35 },
  { x: 2.05, y: 4.35 },
  { x: 2.05, y: 2.4 },
];

/**
 * The mirrored route, for the symmetric-wing scenario: out of room C, west down
 * the corridor, north through the *study* door. The north doorways are not
 * mirror images of each other, so this is the drive that finally tells rooms A
 * and C apart — and until it does, the cloud is entitled to stay bimodal.
 */
const MIRRORED_ROUTE: { x: number; y: number }[] = [
  { x: 9.95, y: 0.9 },
  { x: 9.95, y: 4.35 },
  { x: 6.5, y: 4.35 },
  { x: 3.25, y: 4.35 },
  { x: 3.25, y: 6.4 },
  { x: 3.25, y: 4.35 },
  { x: 8.0, y: 4.35 },
  { x: 9.95, y: 4.35 },
  { x: 9.95, y: 2.4 },
];

const DT = 0.35;
type Scenario = 'global' | 'symmetric';
type Algorithm = 'mcl' | 'amcl';
type Phase = 'predict' | 'weight' | 'resample';

interface Params {
  logM: number;
  algorithm: Algorithm;
  field: boolean;
  kappa: number;
  odom: 'good' | 'sloppy' | 'drunk';
}

interface State {
  rng: Rng;
  /** Truth and sensor noise draw from their own stream, so changing M does not
   *  change the run the robot experiences. */
  world: Rng;
  pf: ParticleFilter;
  truth: Pose2;
  deadReckon: Pose2;
  ticks: EncoderTicks;
  wheelAngles: [number, number];
  waypoint: number;
  phase: Phase;
  scan: number[] | null;
  angles: number[];
  detector: SurpriseDetector;
  pInject: number;
  injected: number;
  ess: number;
  estimate: Pose2;
  mass: number;
  clusters: number;
  error: number;
  ms: number;
  kidnapAt: number | null;
  recoveredIn: number | null;
  step: number;
  route: { x: number; y: number }[];
}

const roomA = { minX: 0.3, maxX: 3.7, minY: 0.3, maxY: 3.5 };
const roomC = { minX: 8.3, maxX: 11.7, minY: 0.3, maxY: 3.5 };

function samplePose(rng: Rng, box = APARTMENT.bounds): Pose2 {
  for (let guard = 0; guard < 400; guard++) {
    const x = rng.uniform(box.minX, box.maxX);
    const y = rng.uniform(box.minY, box.maxY);
    if (isFree(APARTMENT, x, y, 0.2)) {
      return { x, y, theta: rng.uniform(-Math.PI, Math.PI) };
    }
  }
  return { x: 6, y: 4.4, theta: 0 };
}

/** Global initialization: X₀ ~ Uniform(free(m)) — line 1 of every MCL demo. */
function scatterParticles(m: number, rng: Rng, scenario: Scenario): Particle[] {
  if (scenario === 'symmetric') {
    // Exactly mirrored pairs, not two independent scatters. The map's symmetry
    // is σ(x, y, θ) = (12 − x, y, π − θ); seeding pairs makes the initial belief
    // *exactly* invariant under σ, so the two modes carry identical weight for
    // as long as the evidence is symmetric — which is the claim Exercise 2 asks
    // you to prove. Two independent scatters would break the tie immediately by
    // sampling luck, and the widget would look like it had a bug.
    const out: Particle[] = [];
    while (out.length < m) {
      const p = samplePose(rng, roomC);
      out.push({ state: p, weight: 1 / m });
      if (out.length < m) {
        out.push({
          state: { x: 12 - p.x, y: p.y, theta: Math.PI - p.theta },
          weight: 1 / m,
        });
      }
    }
    return out;
  }
  return ParticleFilter.uniformInWorld(m, APARTMENT, rng, 0.2).particles;
}

function makeState(seed: number, scenario: Scenario, m: number): State {
  const rng = new Rng(seed);
  const world = new Rng(seed * 7919 + 13);
  const route = scenario === 'symmetric' ? MIRRORED_ROUTE : ROUTE;
  const truth: Pose2 = { ...route[0], theta: Math.PI / 2 };
  const pf = new ParticleFilter(scatterParticles(m, rng, scenario));
  return {
    rng,
    world,
    pf,
    truth,
    deadReckon: { ...truth },
    ticks: { left: 0, right: 0 },
    wheelAngles: [0, 0],
    waypoint: 1,
    phase: 'resample',
    scan: null,
    angles: [],
    detector: new SurpriseDetector({ alphaFast: 0.5, alphaSlow: 0.05 }),
    pInject: 0,
    injected: 0,
    ess: m,
    estimate: { x: 6, y: 4.4, theta: 0 },
    mass: 0,
    clusters: 1,
    error: 0,
    ms: 0,
    kidnapAt: null,
    recoveredIn: null,
    step: 0,
    route,
  };
}

export function MclTheater() {
  const [params, setParams] = useState<Params>({
    logM: 11,
    algorithm: 'amcl',
    field: true,
    kappa: 1,
    odom: 'sloppy',
  });
  const [scenario, setScenario] = useState<Scenario>('global');
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const scenarioRef = useRef(scenario);
  scenarioRef.current = scenario;
  const kidnapRef = useRef(false);

  const init = useCallback(
    (seed: number) => makeState(seed, scenarioRef.current, 2 ** paramsRef.current.logM),
    [],
  );

  const step = useCallback((s: State, tick: number): State => {
    const p = paramsRef.current;
    const M = 2 ** p.logM;
    const phase: Phase = (['predict', 'weight', 'resample'] as const)[tick % 3];
    const { pf, rng, world } = s;
    const next: State = { ...s, phase, injected: phase === 'resample' ? s.injected : 0 };

    if (phase === 'predict') {
      // ---- the world moves -------------------------------------------------
      if (kidnapRef.current) {
        kidnapRef.current = false;
        // Teleport somewhere genuinely elsewhere: at least 4 m away.
        let target = samplePose(world);
        for (let i = 0; i < 40 && poseDistance(target, s.truth) < 4; i++) {
          target = samplePose(world);
        }
        // Only the pose changes. The wheels report nothing about a teleport —
        // which is exactly why the filter has no idea it happened.
        next.truth = target;
        next.kidnapAt = s.step;
        next.recoveredIn = null;
      }

      const wp = s.route[next.waypoint % s.route.length];
      const u = pursuePoint(next.truth, wp, { speed: 0.5, gain: 1.8, maxOmega: 1.1 });
      // Ground truth advances with wheel slip; the *encoders* never see the slip,
      // which is why the odometry the filter receives is already wrong.
      const outcome = diffDriveSlipStep(
        { pose: next.truth, wheelAngles: s.wheelAngles },
        u,
        DT,
        APARTMENT,
        RUSTY,
        world,
      );
      next.truth = outcome.pose;
      next.wheelAngles = outcome.wheelAngles;
      const ticks = encoderTicks(outcome.wheelAngles, RUSTY);
      const tau = odometryDelta(s.ticks, ticks, RUSTY);
      next.ticks = ticks;
      const dr = integrateOdometry(s.deadReckon, tau);
      // u_t is what the wheels reported — never the truth.
      const odom = odomFromPoses(s.deadReckon, dr);
      next.deadReckon = dr;
      if (Math.hypot(next.truth.x - wp.x, next.truth.y - wp.y) < 0.45 || outcome.blocked) {
        next.waypoint = (next.waypoint + 1) % s.route.length;
      }

      // ---- the filter predicts --------------------------------------------
      if (pf.size !== M) resize(pf, M, rng);
      pf.predict((x) => sampleMotionModelOdometry(odom, x, ALPHAS[p.odom], rng));
      next.step = s.step + 1;
      return next;
    }

    if (phase === 'weight') {
      const scan = raycastScan(APARTMENT, s.truth, LIDAR, world);
      const z: number[] = [];
      const angles: number[] = [];
      for (let k = 0; k < scan.ranges.length; k += BEAM_STRIDE) {
        z.push(scan.ranges[k]);
        angles.push(scan.angles[k]);
      }
      next.scan = z;
      next.angles = angles;

      const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
      const raw: number[] = [];
      // The importance weight *is* the measurement likelihood — everything else
      // cancelled when we chose the motion model as the proposal.
      pf.correctLog((x) => {
        const l = p.field
          ? logLikelihoodFieldRangeFinderModel(z, x, FIELD, BEAM_PARAMS, angles, LIDAR.offset)
          : logBeamRangeFinderModel(z, x, APARTMENT, BEAM_PARAMS, angles);
        const tempered = l / p.kappa;
        raw.push(tempered);
        return tempered;
      });
      const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

      next.ms = 0.7 * s.ms + 0.3 * (t1 - t0);
      next.ess = pf.effectiveSampleSize();

      // w_avg on a per-beam scale, so α_fast/α_slow do not depend on the map.
      let wAvg = 0;
      for (const l of raw) wAvg += perBeamLikelihood(l, z.length);
      wAvg /= Math.max(raw.length, 1);
      next.pInject = p.algorithm === 'amcl' ? s.detector.update(wAvg) : 0;

      const cluster = dominantCluster(pf.particles, 0.7);
      next.estimate = cluster.pose;
      next.mass = cluster.mass;
      next.clusters = countClusters(pf.particles, 0.6);
      next.error = Math.hypot(cluster.pose.x - s.truth.x, cluster.pose.y - s.truth.y);
      if (s.kidnapAt !== null && s.recoveredIn === null && next.error < 0.5 && cluster.mass > 0.5) {
        next.recoveredIn = s.step - s.kidnapAt;
      }
      return next;
    }

    // ---- resample -----------------------------------------------------------
    if (p.algorithm === 'amcl' && s.pInject > 0) {
      const out = augmentedResample(pf.particles, rng, s.pInject, (r) => samplePose(r));
      pf.particles = out.particles;
      next.injected = out.injected;
    } else {
      pf.resample(rng, 'lowVariance');
      next.injected = 0;
    }
    return next;
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 9, initialSeed: 7 });

  const restart = (nextScenario: Scenario) => {
    scenarioRef.current = nextScenario;
    setScenario(nextScenario);
    // The twin-rooms scenario is only legible with a slightly softened sensor:
    // at κ = 1 a 12-beam likelihood field is sharp enough to pick a room on the
    // first update, from an asymmetry of a few centimetres.
    const kappa = nextScenario === 'symmetric' ? 3 : 1;
    paramsRef.current = { ...paramsRef.current, kappa };
    setParams((q) => ({ ...q, kappa }));
    sim.setState(() => makeState(sim.seed, nextScenario, 2 ** paramsRef.current.logM));
  };

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      const s = sim.state;
      clear(ctx, v, p);
      drawWorld(ctx, v, APARTMENT, p);

      // --- the cloud, colored by where it is in the predict → weight → resample
      //     cycle. Freshly injected particles are drawn in prior blue, because
      //     that is literally what they are: fresh draws from the prior.
      const cloudColor =
        s.phase === 'predict' ? p.prediction : s.phase === 'weight' ? p.measurement : p.posterior;
      const particles = s.pf.particles;
      const stride = Math.max(1, Math.ceil(particles.length / 1400));
      const wMax = particles.reduce((m, q) => Math.max(m, q.weight), 0) || 1;
      const injectedFrom = particles.length - s.injected;

      ctx.save();
      for (let i = 0; i < particles.length; i += stride) {
        const q = particles[i];
        const w = q.weight / wMax;
        const isNew = s.phase === 'resample' && i >= injectedFrom;
        ctx.fillStyle = isNew ? p.prior : cloudColor;
        ctx.globalAlpha = isNew ? 0.9 : 0.2 + 0.6 * (s.phase === 'weight' ? w : 0.35);
        const px = sx(v, q.state.x);
        const py = sy(v, q.state.y);
        const r = isNew ? 2.2 : 1.1 + (s.phase === 'weight' ? 2.4 * Math.sqrt(w) : 0.6);
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
        if (stride <= 4) {
          ctx.globalAlpha *= 0.5;
          ctx.strokeStyle = ctx.fillStyle;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px + 6 * Math.cos(q.state.theta), py - 6 * Math.sin(q.state.theta));
          ctx.stroke();
        }
      }
      ctx.restore();

      // --- the scan, drawn from the *estimate*: a delocalized filter visibly
      //     hallucinates its own scan straight through walls.
      if (s.scan) {
        ctx.save();
        ctx.strokeStyle = p.measurement;
        ctx.globalAlpha = 0.28;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let k = 0; k < s.scan.length; k++) {
          const a = s.estimate.theta + s.angles[k];
          const r = Math.min(s.scan[k], LIDAR.maxRange);
          ctx.moveTo(sx(v, s.estimate.x), sy(v, s.estimate.y));
          ctx.lineTo(sx(v, s.estimate.x + r * Math.cos(a)), sy(v, s.estimate.y + r * Math.sin(a)));
        }
        ctx.stroke();
        ctx.restore();
      }

      drawRobot(ctx, v, s.estimate, p.posterior, 0.3);
      // Ground truth: the quantity the robot never gets to see.
      ctx.save();
      ctx.setLineDash([4, 3]);
      drawRobot(ctx, v, s.truth, p.truth, 0.34, { filled: false });
      ctx.restore();

      const phaseText =
        s.phase === 'predict'
          ? 'PREDICT  ·  sample_motion_model_odometry'
          : s.phase === 'weight'
            ? 'WEIGHT  ·  w = p(z | x, m)'
            : s.injected > 0
              ? `RESAMPLE  ·  ${s.injected} random poses injected`
              : 'RESAMPLE  ·  low-variance sampler';
      label(ctx, phaseText, 10, 14, s.phase === 'resample' && s.injected > 0 ? p.prior : cloudColor, {
        size: 11,
        weight: 600,
      });

      const banner =
        s.kidnapAt !== null && s.recoveredIn === null
          ? `KIDNAPPED ${s.step - s.kidnapAt} steps ago`
          : s.recoveredIn !== null
            ? `recovered in ${s.recoveredIn} steps`
            : s.clusters > 1
              ? `${s.clusters} hypotheses alive — ambiguity honestly represented`
              : `one hypothesis · ${(s.mass * 100).toFixed(0)}% of the belief`;
      label(ctx, banner, 10, v.height - 10, p.ink, { size: 11 });
      label(
        ctx,
        `${sl(v, 1).toFixed(0)} px = 1 m`,
        v.width - 10,
        v.height - 10,
        p.ink,
        { size: 10, align: 'right' },
      );
    },
    [sim.state],
  );

  const s = sim.state;
  const M = s.pf.size;
  const ratio = s.detector.wSlow > 0 ? s.detector.wFast / s.detector.wSlow : 1;

  return (
    <WidgetFrame
      id="w12.1"
      title="MCL Theater"
      teaches="MCL is not a new filter — it is the particle filter with a motion sampler and a sensor likelihood, and everything hard about it lives in what happens after the weights."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty wakes up knowing nothing: every free square metre of the Apartment holds particles.
          Watch one full cycle at a time — orange particles have just been pushed through the
          motion model, green ones have just been weighted by the scan (size and opacity{' '}
          <em>are</em> the weight), purple is the resampled posterior. The gray dashed outline is
          the truth; the solid purple triangle is the estimate, and the green rays are drawn from
          the <em>estimate</em>, so a lost filter visibly hallucinates its scan through walls.
          <br />
          <strong>What to try.</strong> Press <em>Kidnap</em> under <em>plain MCL</em> and wait: it
          almost never comes back, because resampling can only reuse poses the set already contains.
          Switch the algorithm to <em>Augmented MCL</em> — the particle set is kept — and kidnap
          again: within a few steps w<sub>fast</sub> dives below w<sub>slow</sub>, blue particles
          rain into the free space, and the filter rebuilds. Then press <em>Twin rooms</em>, which
          seeds exactly mirrored pairs in rooms A and C and softens the sensor to κ = 3. Watch the
          cluster counter: two hypotheses coexist for a dozen steps and then one wins. Now pull κ
          back to 1 and press <em>Twin rooms</em> again — the same evidence now decides in about
          three steps, because a sharp likelihood turns a few centimetres of asymmetry (and a bit of
          sampling luck) into a verdict.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state]}
        aspect={12 / 8}
        padding={0.35}
        ariaLabel="A floorplan of an apartment covered in particles representing the robot's belief about its pose. The cloud condenses onto the true robot as it drives, and scatters again when the robot is kidnapped."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-6">
        <Stat label="M" value={M.toLocaleString()} />
        <Stat label="ESS / M" value={`${((s.ess / Math.max(M, 1)) * 100).toFixed(0)}%`} />
        <Stat label="clusters" value={String(s.clusters)} />
        <Stat
          label="w_fast / w_slow"
          value={ratio.toFixed(2)}
          tint={s.pInject > 0 ? 'var(--pr-prior)' : undefined}
        />
        <Stat
          label="inject p"
          value={`${(s.pInject * 100).toFixed(0)}%`}
          tint={s.pInject > 0 ? 'var(--pr-prior)' : undefined}
        />
        <Stat
          label="|est − truth|"
          value={`${s.error.toFixed(2)} m`}
          tint={s.error > 1 ? 'var(--pr-prediction)' : undefined}
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Particles M = 2^k"
          value={params.logM}
          min={8}
          max={13}
          step={1}
          onChange={(v) => setParams((q) => ({ ...q, logM: v }))}
          format={(v) => (2 ** v).toLocaleString()}
          help="The headline knob. More particles means a finer approximation and a linearly larger bill — watch ms/update."
        />
        <Slider
          label="Tempering κ"
          role="measurement"
          value={params.kappa}
          min={1}
          max={6}
          step={0.5}
          onChange={(v) => setParams((q) => ({ ...q, kappa: v }))}
          help="Raises the likelihood to the power 1/κ. κ > 1 softens an overconfident sensor model — Chapter 10's cure for a filter that starves."
        />
        <Slider
          label="Odometry noise"
          role="prediction"
          value={params.odom === 'good' ? 0 : params.odom === 'sloppy' ? 1 : 2}
          min={0}
          max={2}
          step={1}
          onChange={(v) =>
            setParams((q) => ({ ...q, odom: (['good', 'sloppy', 'drunk'] as const)[v] }))
          }
          format={(v) => ['good', 'sloppy', 'drunk'][v]}
          help="The α₁…α₄ of sample_motion_model_odometry. Drunk wheels widen the proposal, which costs particles."
        />
        <Toggle
          label="Likelihood field (off = beam model)"
          role="measurement"
          checked={params.field}
          onChange={(v) => setParams((q) => ({ ...q, field: v }))}
        />
        <Toggle
          label="Augmented MCL (off = plain MCL)"
          role="prior"
          checked={params.algorithm === 'amcl'}
          onChange={(v) => setParams((q) => ({ ...q, algorithm: v ? 'amcl' : 'mcl' }))}
        />
        <div className="flex items-end">
          <span className="font-mono text-[0.7rem] text-fd-muted-foreground tabular-nums">
            {s.ms.toFixed(1)} ms / update
          </span>
        </div>
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2.5">
        <ButtonRow>
          <ActionButton onClick={() => (kidnapRef.current = true)} emphasis>
            Kidnap!
          </ActionButton>
          <ActionButton onClick={() => restart('global')}>Global wake-up</ActionButton>
          <ActionButton onClick={() => restart('symmetric')}>Twin rooms</ActionButton>
          <span className="font-ui text-[0.7rem] text-fd-muted-foreground">
            scenario:{' '}
            {scenario === 'global'
              ? 'uniform over free space'
              : 'mirrored pairs in rooms A and C, κ = 3'}
          </span>
        </ButtonRow>
      </div>

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

/** Honour the M slider mid-run by drawing a new population from the current one. */
function resize(pf: ParticleFilter, m: number, rng: Rng): void {
  const w = pf.particles.map((q) => q.weight);
  pf.particles = Array.from({ length: m }, () => {
    const src = pf.particles[rng.choice(w)];
    return { state: { ...src.state }, weight: 1 / m };
  });
}

function Stat({ label: l, value, tint }: { label: string; value: string; tint?: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums" style={tint ? { color: tint } : undefined}>
        {value}
      </div>
    </div>
  );
}
