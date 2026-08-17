'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { ParticleFilter } from '@/lib/filters/pf';
import { DEFAULT_BEAM_PARAMS, LikelihoodField, logLikelihoodFieldRangeFinderModel } from '@/lib/models/sensor';
import { odomFromPoses, sampleMotionModelOdometry, type OdomAlphas } from '@/lib/models/motion';
import { RUSTY_LIDAR, pursuePoint, raycastScan, type LidarParams } from '@/lib/sim/rusty';
import { APARTMENT, diffDriveStep, isFree } from '@/lib/sim/world';
import { poseDistance, type Pose2 } from '@/lib/geom/se2';
import { Rng } from '@/lib/prob/rng';
import { spread } from '@/lib/learn/calibration';
import {
  clear,
  drawParticles,
  drawRobot,
  drawWorld,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w25.3 — the Tempering Lab.
 *
 * Real MCL — the `ParticleFilter` of Chapter 8, the likelihood field of
 * Chapter 10, `sample_motion_model_odometry` of Chapter 9 — with exactly one
 * addition: the log-likelihood is multiplied by κ before it becomes a weight.
 *
 *   bel(x) = η · p(z | x, m)^κ · b̄el(x)
 *
 * κ = 1 is the Bayes filter. κ > 1 is what a beam model that pretends its 16
 * beams are independent is *effectively* doing, and what an overconfident
 * learned model does explicitly. The instrument is the ESS trace and the
 * kidnap-recovery tally underneath: neither of them is an opinion.
 */

const FIELD = new LikelihoodField(APARTMENT, 0.06);
const LIDAR: LidarParams = { ...RUSTY_LIDAR, nBeams: 16, sigmaR: 0.03, pDropout: 0.02 };
const BEAM_PARAMS = { ...DEFAULT_BEAM_PARAMS, sigmaHit: 0.16, maxRange: LIDAR.maxRange };
const ALPHAS: OdomAlphas = [0.02, 0.015, 0.03, 0.015];

const DT = 0.35;
const M = 400;
/** A kidnap every this many ticks; the tally underneath is the experiment. */
const KIDNAP_PERIOD = 90;
/** Recovered when the estimate comes back within this of the truth. */
const RECOVERY_RADIUS = 0.9;
const ESS_HISTORY = 150;

/** A patrol that visits both wings, so recovery is never trivially local. */
const ROUTE: { x: number; y: number }[] = [
  { x: 2.05, y: 2.4 },
  { x: 2.05, y: 4.4 },
  { x: 6.0, y: 4.4 },
  { x: 9.95, y: 4.4 },
  { x: 9.95, y: 2.0 },
  { x: 9.95, y: 4.4 },
  { x: 6.0, y: 4.4 },
  { x: 3.25, y: 4.4 },
  { x: 3.25, y: 6.4 },
  { x: 3.25, y: 4.4 },
];

interface Params {
  kappa: number;
  autoKidnap: boolean;
}

interface State {
  rng: Rng;
  world: Rng;
  pf: ParticleFilter;
  truth: Pose2;
  waypoint: number;
  scan: number[];
  angles: number[];
  ess: number;
  essHistory: number[];
  /** Sample spread of the per-particle log-likelihood — the `s` of F3. */
  logLikSpread: number;
  error: number;
  trials: number;
  recoveries: number;
  deprivations: number;
  sinceKidnap: number;
  recoveredThisTrial: boolean;
}

function freePose(rng: Rng): Pose2 {
  for (let guard = 0; guard < 4000; guard++) {
    const x = rng.uniform(0.4, 11.6);
    const y = rng.uniform(0.4, 8.6);
    if (isFree(APARTMENT, x, y, 0.4)) return { x, y, theta: rng.uniform(-Math.PI, Math.PI) };
  }
  return { x: 2.05, y: 2.4, theta: 0 };
}

export function TemperingLab() {
  const [params, setParams] = useState<Params>({ kappa: 1, autoKidnap: true });

  const init = useCallback((seed: number): State => {
    const rng = new Rng(seed);
    const world = new Rng(seed ^ 0x5bd1);
    const truth: Pose2 = { x: 2.05, y: 2.4, theta: Math.PI / 2 };
    // Position tracking, not global localization: the cloud starts around the
    // truth, so anything that goes wrong afterwards is the update's fault.
    const pf = ParticleFilter.gaussian(M, truth, { x: 0.3, y: 0.3, theta: 0.2 }, rng);
    return {
      rng,
      world,
      pf,
      truth,
      waypoint: 1,
      scan: [],
      angles: [],
      ess: M,
      essHistory: [],
      logLikSpread: 0,
      error: 0,
      trials: 0,
      recoveries: 0,
      deprivations: 0,
      sinceKidnap: 0,
      recoveredThisTrial: true,
    };
  }, []);

  const step = useCallback(
    (s: State): State => {
      const { rng, world, pf } = s;

      /* ---- the robot drives ------------------------------------------- */
      let waypoint = s.waypoint;
      const target = ROUTE[waypoint];
      if (Math.hypot(target.x - s.truth.x, target.y - s.truth.y) < 0.3) {
        waypoint = (waypoint + 1) % ROUTE.length;
      }
      const u = pursuePoint(s.truth, ROUTE[waypoint], { speed: 0.5 });
      let truth = diffDriveStep(s.truth, u.v, u.omega, DT);
      if (!isFree(APARTMENT, truth.x, truth.y, 0.18)) truth = s.truth;

      /* ---- kidnap ------------------------------------------------------ */
      let trials = s.trials;
      let recoveries = s.recoveries;
      let sinceKidnap = s.sinceKidnap + 1;
      let recoveredThisTrial = s.recoveredThisTrial;
      let prevTruth = s.truth;
      if (params.autoKidnap && sinceKidnap >= KIDNAP_PERIOD) {
        truth = freePose(world);
        // The filter is not told. Odometry reports the step it *commanded*.
        prevTruth = truth;
        sinceKidnap = 0;
        trials += 1;
        recoveredThisTrial = false;
        waypoint = Math.floor(world.uniform(0, ROUTE.length));
      }

      /* ---- predict ------------------------------------------------------ */
      const odom = odomFromPoses(prevTruth, truth);
      pf.predict((state) => sampleMotionModelOdometry(odom, state, ALPHAS, rng));

      /* ---- correct, tempered -------------------------------------------- */
      const scan = raycastScan(APARTMENT, truth, LIDAR, world);
      const logLik = pf.particles.map((p) =>
        logLikelihoodFieldRangeFinderModel(
          scan.ranges,
          p.state,
          FIELD,
          BEAM_PARAMS,
          scan.angles,
          LIDAR.offset,
        ),
      );
      const logLikSpread = spread(logLik);
      let i = 0;
      pf.correctLog(() => params.kappa * logLik[i++]);

      const ess = pf.effectiveSampleSize();
      let deprivations = s.deprivations;
      if (ess < 0.02 * M) deprivations += 1;
      // Resample only when the population has genuinely degenerated — the
      // standard M/2 rule, so that κ's effect on ESS is visible rather than
      // being erased by an unconditional resample every step.
      if (ess < M / 2) pf.resample(rng, 'lowVariance');

      const mean = pf.mean();
      const error = poseDistance(mean, truth);
      if (!recoveredThisTrial && error < RECOVERY_RADIUS && sinceKidnap > 3) {
        recoveredThisTrial = true;
        recoveries += 1;
      }

      return {
        ...s,
        truth,
        waypoint,
        scan: scan.ranges,
        angles: scan.angles,
        ess,
        essHistory: [...s.essHistory, ess / M].slice(-ESS_HISTORY),
        logLikSpread,
        error,
        trials,
        recoveries,
        deprivations,
        sinceKidnap,
        recoveredThisTrial,
      };
    },
    [params.kappa, params.autoKidnap],
  );

  const sim = useSimulation<State>({ init, step, fps: 14, initialSeed: 25 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;

      drawWorld(ctx, v, APARTMENT, p);
      drawParticles(ctx, v, s.pf.particles, p.posterior, { maxRadius: 2.4 });

      const mean = s.pf.mean();
      drawRobot(ctx, v, mean, p.posterior, 0.26, { filled: false });
      drawRobot(ctx, v, s.truth, p.truth, 0.24);

      // The ESS trace, painted along the bottom of the map.
      const h = s.essHistory;
      if (h.length > 2) {
        const y0 = sy(v, 0.15);
        const y1 = sy(v, 1.35);
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx(v, 0.2), y0);
        ctx.lineTo(sx(v, 11.8), y0);
        ctx.stroke();

        ctx.strokeStyle = p.measurement;
        ctx.lineWidth = 1.7;
        ctx.beginPath();
        h.forEach((e, i) => {
          const x = sx(v, 0.2 + (11.6 * i) / (ESS_HISTORY - 1));
          const y = y0 + (y1 - y0) * Math.min(1, e);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        label(ctx, 'ESS / M', sx(v, 0.3), y1 - 8, p.measurement, { size: 9 });
      }

      label(
        ctx,
        `κ = ${params.kappa.toFixed(2)}   ESS = ${(100 * (s.ess / M)).toFixed(1)}%   error = ${s.error.toFixed(2)} m`,
        sx(v, 0.25),
        sy(v, 8.65),
        s.ess < 0.02 * M ? p.prediction : p.ink,
        { size: 11, weight: 600 },
      );
      if (s.sinceKidnap < 12 && params.autoKidnap && s.trials > 0) {
        label(ctx, 'KIDNAPPED', sx(v, 11.75), sy(v, 8.65), p.prediction, {
          size: 11,
          weight: 700,
          align: 'right',
        });
      }
      // A scale bar, so the error readout means something metric.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(v, 10.8), sy(v, 0.35));
      ctx.lineTo(sx(v, 11.8), sy(v, 0.35));
      ctx.stroke();
      label(ctx, '1 m', sx(v, 11.3), sy(v, 0.35) - 9, p.truth, { size: 9, align: 'center' });
    },
    [sim.state, params.kappa, params.autoKidnap],
  );

  const stats = useMemo(() => {
    const s = sim.state;
    return {
      essPct: (100 * s.ess) / M,
      predicted: 100 * Math.exp(-params.kappa * params.kappa * s.logLikSpread * s.logLikSpread),
      error: s.error,
      recovery: s.trials > 0 ? (100 * s.recoveries) / s.trials : Number.NaN,
      deprivations: s.deprivations,
      essSpark: s.essHistory.slice(-40),
    };
  }, [sim.state, params.kappa]);

  return (
    <WidgetFrame
      id="w25.3"
      title="The Tempering Lab"
      teaches="Multiplying in more confidence is not free: the effective sample size falls off a cliff, and a filter with one live particle cannot recover from anything."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          Rusty patrols the Apartment under real MCL, and every {KIDNAP_PERIOD} steps it is picked
          up and put down somewhere random. The only thing κ changes is the exponent on the
          measurement likelihood. At κ&nbsp;=&nbsp;1 the filter recovers from most kidnaps; push κ
          past 2 and watch the green ESS trace flatten against the floor — the cloud is one
          particle wearing four hundred hats, so when the truth is not underneath it there is
          nothing left to re-weight. The <strong>predicted ESS</strong> tile is derivation
          F3&rsquo;s closed form, exp(−κ²s²), computed from the spread of this step&rsquo;s
          per-particle log-likelihoods; watch it track the measured value. κ&nbsp;&lt;&nbsp;1 is
          the honest repair for beams that are not independent — Thrun et al. propose exactly it,
          and call the exponent α.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state, params.kappa]}
        aspect={12 / 9}
        padding={0.2}
        ariaLabel="The apartment floorplan with a cloud of particle poses, the true robot, and the filter's estimate, above a trace of effective sample size over time."
      />

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 lg:grid-cols-4">
        <StatTile
          label="ESS / M"
          value={stats.essPct}
          unit="%"
          precision={1}
          role={stats.essPct < 5 ? 'prediction' : 'measurement'}
          sparkline={stats.essSpark}
        />
        <StatTile
          label="predicted ESS: exp(−κ²s²)"
          value={stats.predicted}
          unit="%"
          precision={1}
        />
        <StatTile
          label="kidnap recoveries"
          value={Number.isNaN(stats.recovery) ? '—' : `${stats.recovery.toFixed(0)}%`}
          role="posterior"
        />
        <StatTile
          label="deprivation events"
          value={stats.deprivations}
          role={stats.deprivations > 0 ? 'prediction' : undefined}
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="tempering exponent κ — the only knob"
          role="measurement"
          value={params.kappa}
          min={0.2}
          max={4}
          step={0.05}
          onChange={(v) => setParams((p) => ({ ...p, kappa: v }))}
          help="bel(x) = η · p(z | x, m)^κ · b̄el(x). κ > 1 sharpens; κ < 1 is the honest discount for correlated beams."
        />
        <Toggle
          label="auto-kidnap every 90 steps"
          checked={params.autoKidnap}
          onChange={(v) => setParams((p) => ({ ...p, autoKidnap: v }))}
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
