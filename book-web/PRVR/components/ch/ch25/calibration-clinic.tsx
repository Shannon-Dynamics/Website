'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { BarChart, LineChart, StatTile, type LineChartSeries } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT, beamAngles, isFree, rayCast } from '@/lib/sim/world';
import { beamLikelihood, type BeamParams } from '@/lib/models/sensor';
import type { Pose2 } from '@/lib/geom/se2';
import { randomizedPit, sampleBeam, scaleBeamWidth } from '@/lib/learn/beam-density';
import {
  calibrationReport,
  fitTemperature,
  logScore,
  temperedEss,
  type CalibrationReport,
} from '@/lib/learn/calibration';

/**
 * w25.1 — the Calibration Clinic.
 *
 * One sensor, three claims about it. The reliability diagram and the PIT
 * histogram are computed from the real beam mixture of Chapter 10 evaluated
 * against readings drawn from the *true* forward model, and the two numbers on
 * the right — effective sample size and posterior error — come from feeding
 * those same claims to a real importance-weighting step over particle clouds in
 * the Apartment.
 *
 * The punchline is arithmetic, not rhetoric: the width that minimizes the log
 * score and the width that minimizes the calibration error are **not the same
 * width**, and it is the calibrated one that the filter downstream cares about.
 */

/* -------------------------------------------------------------------------- */
/* The sensor, and the model of it                                             */
/* -------------------------------------------------------------------------- */

/** What Rusty's LiDAR really does. The reader never gets to see these numbers. */
const TRUE_PARAMS: BeamParams = {
  zHit: 0.8,
  zShort: 0.08,
  zMax: 0.05,
  zRand: 0.07,
  sigmaHit: 0.11,
  lambdaShort: 1.2,
  maxRange: 8,
};

/**
 * The hand-tuned model an engineer wrote down on the first afternoon: a sharp
 * Gaussian, a dropout spike, a thin uniform floor — and *no* unexpected-obstacle
 * component at all. Mis-specified on purpose, because a correctly specified
 * model fitted by maximum likelihood is calibrated for free, and then there is
 * nothing to learn from this widget.
 */
const HAND_TUNED: BeamParams = {
  zHit: 0.9,
  zShort: 0,
  zMax: 0.05,
  zRand: 0.05,
  sigmaHit: 0.05,
  lambdaShort: 1.2,
  maxRange: 8,
};

const N_EVAL = 1200;
const N_VAL = 800;
const BEAMS = 8;
const CLOUDS = 12;
const CLOUD_SIZE = 150;
const ANGLES = beamAngles({ nBeams: BEAMS, fov: 2 * Math.PI });

interface EvalPair {
  zStar: number;
  z: number;
}

function freePose(rng: Rng): Pose2 {
  for (let guard = 0; guard < 4000; guard++) {
    const x = rng.uniform(0.4, 11.6);
    const y = rng.uniform(0.4, 8.6);
    if (isFree(APARTMENT, x, y, 0.4)) return { x, y, theta: rng.uniform(-Math.PI, Math.PI) };
  }
  return { x: 2, y: 4.4, theta: 0 };
}

/**
 * Thrun et al. §9.3.2, verbatim in spirit: sample poses, sample measurements
 * from the forward model, and use the pairs as training data for the inverse
 * problem. Here the "training" is a single scalar — the width — but the recipe
 * is the one the 1999 draft already prescribed.
 */
function sampleEvalSet(n: number, seed: number): EvalPair[] {
  const rng = new Rng(seed);
  const out: EvalPair[] = [];
  for (let i = 0; i < n; i++) {
    const p = freePose(rng);
    const zStar = rayCast(APARTMENT, p.x, p.y, p.theta, TRUE_PARAMS.maxRange);
    out.push({ zStar, z: sampleBeam(zStar, TRUE_PARAMS, rng) });
  }
  return out;
}

/**
 * A tracking-grade particle cloud: the prediction has drifted about 30 cm off
 * the truth, and the measurement update has to pull it back. Ray casts depend
 * only on the geometry, so they are done once here and reused at every width
 * the slider visits — which is what makes the whole sweep interactive.
 */
interface Cloud {
  truth: Pose2;
  z: number[];
  poses: Pose2[];
  zStar: number[][];
  priorError: number;
}

function sampleCloud(seed: number): Cloud {
  const rng = new Rng(seed);
  const truth = freePose(rng);
  const z = ANGLES.map((a) =>
    sampleBeam(
      rayCast(APARTMENT, truth.x, truth.y, truth.theta + a, TRUE_PARAMS.maxRange),
      TRUE_PARAMS,
      rng,
    ),
  );
  const cx = truth.x + rng.normal(0, 0.3);
  const cy = truth.y + rng.normal(0, 0.3);
  const ct = truth.theta + rng.normal(0, 0.12);
  const poses: Pose2[] = Array.from({ length: CLOUD_SIZE }, () => ({
    x: cx + rng.normal(0, 0.35),
    y: cy + rng.normal(0, 0.35),
    theta: ct + rng.normal(0, 0.21),
  }));
  const zStar = poses.map((p) =>
    ANGLES.map((a) => rayCast(APARTMENT, p.x, p.y, p.theta + a, TRUE_PARAMS.maxRange)),
  );
  let ex = 0;
  let ey = 0;
  for (const p of poses) {
    ex += p.x / poses.length;
    ey += p.y / poses.length;
  }
  return { truth, z, poses, zStar, priorError: Math.hypot(ex - truth.x, ey - truth.y) };
}

const EVAL_SET = sampleEvalSet(N_EVAL, 25101);
const VAL_SET = sampleEvalSet(N_VAL, 25102);
const CLOUD_SET = Array.from({ length: CLOUDS }, (_, i) => sampleCloud(25200 + i));
const PRIOR_ERROR =
  CLOUD_SET.reduce((a, c) => a + c.priorError, 0) / CLOUD_SET.length;

/** Held-out NLL as a function of the width scale — the temperature objective. */
function validationNll(scale: number): number {
  const params = scaleBeamWidth(HAND_TUNED, scale);
  let nll = 0;
  for (const e of VAL_SET) nll += logScore(beamLikelihood(e.z, e.zStar, params));
  return nll / VAL_SET.length;
}

/** The fitted temperature: one scalar, one held-out set, no retraining. */
const FITTED = fitTemperature(validationNll, 0.5, 8);

function grade(params: BeamParams): CalibrationReport {
  // A fixed stream for the randomized PIT, so the diagram does not flicker as
  // the slider moves — the only randomness left is the evaluation set itself.
  const rng = new Rng(4242);
  const pit = EVAL_SET.map((e) => randomizedPit(e.z, e.zStar, params, rng));
  const scores = EVAL_SET.map((e) => Math.log(Math.max(beamLikelihood(e.z, e.zStar, params), 1e-300)));
  return calibrationReport(pit, scores);
}

interface Downstream {
  essFraction: number;
  error: number;
}

/** One importance-weighting step per cloud: exactly what MCL does, once. */
function downstream(params: BeamParams): Downstream {
  let ess = 0;
  let err = 0;
  for (const cloud of CLOUD_SET) {
    const ll = cloud.poses.map((_, i) => {
      let s = 0;
      for (let k = 0; k < BEAMS; k++) {
        s += Math.log(Math.max(beamLikelihood(cloud.z[k], cloud.zStar[i][k], params), 1e-300));
      }
      return s;
    });
    ess += temperedEss(ll, 1) / cloud.poses.length;

    let max = -Infinity;
    for (const l of ll) if (l > max) max = l;
    let total = 0;
    let ex = 0;
    let ey = 0;
    for (let i = 0; i < ll.length; i++) {
      const w = Math.exp(ll[i] - max);
      total += w;
      ex += w * cloud.poses[i].x;
      ey += w * cloud.poses[i].y;
    }
    err += Math.hypot(ex / total - cloud.truth.x, ey / total - cloud.truth.y);
  }
  return { essFraction: ess / CLOUD_SET.length, error: err / CLOUD_SET.length };
}

/* -------------------------------------------------------------------------- */
/* The autoplay sweep                                                          */
/* -------------------------------------------------------------------------- */

const MIN_SCALE = 0.6;
const MAX_SCALE = 6;
const SWEEP_STEPS = 56;

/** Ping-pong along a log-spaced ladder of widths. */
function sweepScale(tick: number): number {
  const phase = tick % (2 * SWEEP_STEPS);
  const u = phase < SWEEP_STEPS ? phase / SWEEP_STEPS : 2 - phase / SWEEP_STEPS;
  return Math.exp(Math.log(MIN_SCALE) + u * (Math.log(MAX_SCALE) - Math.log(MIN_SCALE)));
}

interface State {
  scale: number;
}

export function CalibrationClinic() {
  const [manual, setManual] = useState<number | null>(null);

  const init = useCallback((): State => ({ scale: sweepScale(0) }), []);
  const step = useCallback(
    (_s: State, tick: number): State => ({ scale: sweepScale(tick + 1) }),
    [],
  );
  const sim = useSimulation<State>({ init, step, fps: 8, initialSeed: 25 });

  const scale = manual ?? sim.state.scale;

  const setScale = (v: number) => {
    setManual(v);
    sim.pause();
  };

  /* ---- the three claims ------------------------------------------------- */

  const live = useMemo(() => {
    const params = scaleBeamWidth(HAND_TUNED, scale);
    return { params, report: grade(params), down: downstream(params) };
  }, [scale]);

  const references = useMemo(() => {
    const hand = scaleBeamWidth(HAND_TUNED, 1);
    const tempered = scaleBeamWidth(HAND_TUNED, FITTED.scale);
    return {
      hand: { params: hand, report: grade(hand), down: downstream(hand) },
      tempered: { params: tempered, report: grade(tempered), down: downstream(tempered) },
    };
  }, []);

  /** The offline sweep behind the caption's claim about the two optima. */
  const optima = useMemo(() => {
    let bestNll = { scale: 1, value: Infinity };
    let bestEce = { scale: 1, value: Infinity };
    for (let i = 0; i <= 40; i++) {
      const s = Math.exp(Math.log(MIN_SCALE) + (i / 40) * (Math.log(MAX_SCALE) - Math.log(MIN_SCALE)));
      const r = grade(scaleBeamWidth(HAND_TUNED, s));
      if (r.meanNll < bestNll.value) bestNll = { scale: s, value: r.meanNll };
      if (r.ece < bestEce.value) bestEce = { scale: s, value: r.ece };
    }
    return { nll: bestNll, ece: bestEce };
  }, []);

  /* ---- charts ----------------------------------------------------------- */

  const reliabilitySeries = useMemo<LineChartSeries[]>(() => {
    const curve = (report: CalibrationReport) => [
      { x: 0, y: 0 },
      ...report.bins.map((b) => ({ x: b.nominal, y: b.empirical })),
    ];
    return [
      {
        id: 'perfect calibration',
        role: 'truth',
        data: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
        ],
      },
      { id: 'hand-tuned (σ = 0.05 m)', role: 'prediction', data: curve(references.hand.report) },
      {
        id: `+ temperature (σ = ${(HAND_TUNED.sigmaHit * FITTED.scale).toFixed(3)} m)`,
        role: 'posterior',
        data: curve(references.tempered.report),
      },
      {
        id: `your width (σ = ${(HAND_TUNED.sigmaHit * scale).toFixed(3)} m)`,
        role: 'measurement',
        data: curve(live.report),
      },
    ];
  }, [live.report, references, scale]);

  const pitSeries = useMemo(
    () => [
      {
        id: 'PIT density',
        role: 'measurement' as const,
        data: live.report.pitHistogram.map((d, i) => ({
          x: ((i + 0.5) / live.report.pitHistogram.length).toFixed(2),
          y: d,
        })),
      },
    ],
    [live.report],
  );

  const sigma = HAND_TUNED.sigmaHit * scale;
  const hist = live.report.pitHistogram;
  const overconfident = hist[0] + hist[hist.length - 1] > 2.4;

  return (
    <WidgetFrame
      id="w25.1"
      title="The Calibration Clinic"
      teaches="A sharper sensor model is not a better one. Calibration is a measurable property, and it is the one the filter downstream actually consumes."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          One LiDAR, evaluated {N_EVAL} beams at a time in the Apartment. The engineer&rsquo;s
          hand-tuned model (orange) claims σ&nbsp;=&nbsp;5&nbsp;cm; the sensor really has
          σ&nbsp;=&nbsp;11&nbsp;cm plus unexpected obstacles the model does not represent at all.
          Its reliability curve sags far below the diagonal — every nominal interval catches fewer
          observations than it promises — and its PIT histogram is a <strong>U</strong>, the
          signature of overconfidence: readings keep landing in tails the model said were empty.
          Drag the width and watch the curve climb onto the diagonal and the U flatten. Then read
          the two right-hand tiles, which come from a real importance-weighting step over{' '}
          {CLOUDS} particle clouds: as the model sharpens, the effective sample size collapses —
          at σ&nbsp;=&nbsp;3&nbsp;cm a 150-particle cloud is doing the work of one particle — and
          the posterior error rises even though the model looks more confident.{' '}
          <strong>The two optima do not coincide:</strong> the log score is minimized at
          σ&nbsp;≈&nbsp;{(HAND_TUNED.sigmaHit * optima.nll.scale).toFixed(3)}&nbsp;m and the
          calibration error at σ&nbsp;≈&nbsp;{(HAND_TUNED.sigmaHit * optima.ece.scale).toFixed(3)}
          &nbsp;m, because a single mis-modelled dropout beam costs the log score hundreds of nats
          and it will happily wreck your central coverage to buy that back.
        </>
      }
    >
      <div className="grid gap-0 lg:grid-cols-[1.25fr_1fr] lg:divide-x lg:divide-fd-border">
        <div className="p-3">
          <p className="eyebrow mb-1">Reliability diagram — claimed vs. actual coverage</p>
          <LineChart
            series={reliabilitySeries}
            xLabel="nominal level p"
            yLabel="observed fraction ≤ p"
            height={280}
            yMin={0}
            yMax={1}
            ariaLabel="Reliability diagram. The gray diagonal is perfect calibration; curves sagging below it are overconfident and curves bowing above are underconfident."
          />
        </div>
        <div className="border-t border-fd-border p-3 lg:border-t-0">
          <p className="eyebrow mb-1">PIT histogram — flat is honest</p>
          <BarChart
            series={pitSeries}
            xLabel="PIT value"
            yLabel="density"
            height={280}
            maxValue={3}
            ariaLabel="Histogram of probability integral transform values. A flat histogram at density one means the forecast is calibrated; a U shape means it is too narrow."
            caption={
              overconfident
                ? 'Both end bars are tall: the model is too narrow, and the data keeps escaping into tails it declared empty.'
                : 'Flat, near density 1 — the model is telling the truth about its own uncertainty.'
            }
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 lg:grid-cols-4">
        <StatTile
          label="ECE (want 0)"
          value={live.report.ece}
          precision={4}
          role={live.report.ece > 0.04 ? 'prediction' : 'posterior'}
        />
        <StatTile label="mean NLL" value={live.report.meanNll} unit="nats/beam" precision={3} />
        <StatTile
          label="ESS after one update"
          value={live.down.essFraction * 100}
          unit="% of 150"
          precision={1}
          role={live.down.essFraction < 0.01 ? 'prediction' : 'measurement'}
        />
        <StatTile
          label="posterior error"
          value={live.down.error}
          unit="m"
          precision={3}
          role="posterior"
          trend={live.down.error - PRIOR_ERROR}
          trendLabel="vs. prior mean"
        />
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="claimed hit width σ_hit — the whole widget"
          role="measurement"
          value={scale}
          min={MIN_SCALE}
          max={MAX_SCALE}
          step={0.05}
          onChange={setScale}
          format={(v) => `${(HAND_TUNED.sigmaHit * v).toFixed(3)} m`}
          help="Multiplies the hand-tuned σ. Temperature scaling is exactly this one number, fitted on held-out data instead of guessed."
        />
      </ControlPanel>

      <div className="flex flex-wrap items-center gap-2 border-t border-fd-border px-3 py-2">
        <span className="eyebrow">jump to</span>
        <ButtonRow>
          <ActionButton onClick={() => setScale(1)}>hand-tuned</ActionButton>
          <ActionButton onClick={() => setScale(optima.nll.scale)}>best NLL</ActionButton>
          <ActionButton onClick={() => setScale(optima.ece.scale)}>best ECE</ActionButton>
          <ActionButton emphasis onClick={() => setScale(FITTED.scale)}>
            fitted temperature
          </ActionButton>
        </ButtonRow>
        <span className="ms-auto font-mono text-[0.7rem] text-fd-muted-foreground tabular-nums">
          σ = {sigma.toFixed(3)} m · true σ_hit = {TRUE_PARAMS.sigmaHit.toFixed(2)} m
        </span>
      </div>

      <Transport
        playing={sim.playing}
        onToggle={() => {
          setManual(null);
          sim.toggle();
        }}
        onStep={() => {
          setManual(null);
          sim.stepOnce();
        }}
        onReset={() => {
          setManual(null);
          sim.reset();
        }}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
