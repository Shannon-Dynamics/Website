'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import {
  DEFAULT_FASTSLAM_OPTIONS,
  FastSlam,
  initLandmark,
  landmarkJacobian,
  odometryPoseCovariance,
  poseJacobian,
  predictObservation,
  type ProposalKind,
} from '@/lib/filters/fastslam';
import { effectiveSampleSize } from '@/lib/filters/rbpf';
import { applyOdom, sampleMotionModelOdometry, type OdomAlphas, type OdomDelta } from '@/lib/models/motion';
import { landmarkObservation } from '@/lib/models/sensor';
import { ellipse2, inv, matMul, transpose, type Mat } from '@/lib/prob/linalg';
import { Rng } from '@/lib/prob/rng';
import { normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import {
  clear,
  drawCovariance,
  drawRobot,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w17.3 — Proposal Quality.
 *
 * One prediction step, frozen and repeated. The motion prior is orange, the
 * measurement likelihood green, their product — the posterior a perfect
 * proposal would sample from — purple. FastSLAM 1.0 draws from the orange and
 * pays with the weight; FastSLAM 2.0 draws from the purple directly.
 *
 * Both clouds come from the real `FastSlam` class stepped once, with
 * resampling disabled so the reader sees the raw proposal.
 */

const M = 160;
const PREV: Pose2 = { x: 0, y: 0, theta: 0 };
const U: OdomDelta = { rot1: 0, trans: 0.6, rot2: 0 };
/** Loose wheels: α₃ = 0.15 gives about 23 cm of translational spread. */
const ALPHAS: OdomAlphas = [0.03, 0.02, 0.15, 0.03];
const LANDMARK = { x: 2.6, y: 1.15, id: 0 };
/** A landmark seen fifty times already: 10 cm of residual uncertainty. */
const LM_SIGMA: Mat = [
  [0.01, 0],
  [0, 0.01],
];
/** χ²₂ at 90%. */
const CHI2_90 = 4.605;
const SCENE = { minX: -0.5, minY: -0.75, maxX: 3.1, maxY: 1.5 };

interface Trial {
  poses: Pose2[];
  weights: number[];
  neff: number;
  covered: number;
}

interface Geometry {
  xHat: Pose2;
  priorCov: Mat;
  likelihoodCov: Mat;
  postMean: [number, number];
  postCov: Mat;
}

interface State {
  rng: Rng;
  truth: Pose2;
  geom: Geometry;
  trials: Record<ProposalKind, Trial>;
  history: Record<ProposalKind, { neff: number; covered: number }[]>;
}

const addMat = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((v, j) => v + b[i][j]));

/** Everything analytic about this step: prior, likelihood, and their product. */
function geometry(z: { r: number; phi: number }, sigmaR: number, sigmaPhi: number): Geometry {
  const xHat = applyOdom(PREV, U);
  const r: Mat = odometryPoseCovariance(PREV, U, ALPHAS);
  const q: Mat = [
    [sigmaR * sigmaR, 0],
    [0, sigmaPhi * sigmaPhi],
  ];

  const mu: [number, number] = [LANDMARK.x, LANDMARK.y];
  const hm = landmarkJacobian(xHat, mu);
  const hx = poseJacobian(xHat, mu);
  // Q_j: the measurement covariance inflated by what the map does not know.
  const qj = addMat(matMul(matMul(hm, LM_SIGMA), transpose(hm)), q);
  const qjInv = inv(qj);

  // The set of poses consistent with z, as a covariance in (x, y): invert the
  // 2×2 position block of the pose Jacobian and push Q_j through it.
  const hxyInv = inv([
    [hx[0][0], hx[0][1]],
    [hx[1][0], hx[1][1]],
  ]);
  const likelihoodCov = matMul(matMul(hxyInv, qj), transpose(hxyInv));

  // FastSLAM 2.0's proposal: add precisions, let the measurement pull the mean.
  const info = addMat(matMul(matMul(transpose(hx), qjInv), hx), inv(r));
  const sigmaX = inv(info);
  const pred = predictObservation(xHat, mu);
  const nu = [z.r - pred.r, normalizeAngle(z.phi - pred.phi)];
  const gain = matMul(matMul(sigmaX, transpose(hx)), qjInv);

  return {
    xHat,
    priorCov: [
      [r[0][0], r[0][1]],
      [r[1][0], r[1][1]],
    ],
    likelihoodCov,
    postMean: [
      xHat.x + gain[0][0] * nu[0] + gain[0][1] * nu[1],
      xHat.y + gain[1][0] * nu[0] + gain[1][1] * nu[1],
    ],
    postCov: [
      [sigmaX[0][0], sigmaX[0][1]],
      [sigmaX[1][0], sigmaX[1][1]],
    ],
  };
}

/** One frozen step of the real filter under one proposal. */
function runProposal(
  proposal: ProposalKind,
  z: { r: number; phi: number; id: number },
  sigmaR: number,
  sigmaPhi: number,
  geom: Geometry,
  seed: number,
): Trial {
  const q: Mat = [
    [sigmaR * sigmaR, 0],
    [0, sigmaPhi * sigmaPhi],
  ];
  const fs = FastSlam.atPose(M, PREV, {
    ...DEFAULT_FASTSLAM_OPTIONS,
    proposal,
    alphas: ALPHAS,
    sigmaR,
    sigmaPhi,
    knownCorrespondence: true,
    // Resampling would hide the proposal behind its own cleanup.
    neffRatio: 0,
  });
  for (const p of fs.particles) {
    const lm = initLandmark(PREV, landmarkObservation(LANDMARK, PREV), q, 0, 0);
    lm.mu = [LANDMARK.x, LANDMARK.y];
    lm.sigma = LM_SIGMA.map((row) => row.slice());
    p.landmarks.push(lm);
  }

  fs.step(U, [z], new Rng(seed));

  const weights = fs.weights.slice();
  const poses = fs.particles.map((p) => ({ ...p.pose }));
  const pInv = inv(geom.postCov);
  let covered = 0;
  for (const pose of poses) {
    const dx = pose.x - geom.postMean[0];
    const dy = pose.y - geom.postMean[1];
    const d2 = dx * (pInv[0][0] * dx + pInv[0][1] * dy) + dy * (pInv[1][0] * dx + pInv[1][1] * dy);
    if (d2 <= CHI2_90) covered += 1;
  }
  return { poses, weights, neff: effectiveSampleSize(weights) / M, covered: covered / M };
}

export function ProposalQuality() {
  const [proposal, setProposal] = useState<ProposalKind>('motion-prior');
  const [sigmaR, setSigmaR] = useState(0.05);
  const paramsRef = useRef({ sigmaR });
  paramsRef.current = { sigmaR };

  const build = useCallback((rng: Rng, prevHistory?: State['history']): State => {
    const sr = paramsRef.current.sigmaR;
    const sp = Math.max(sr / 3, 0.006);
    // The truth is drawn from the motion model, so the measurement is honest:
    // the robot really is somewhere the prior allows.
    const truth = sampleMotionModelOdometry(U, PREV, ALPHAS, rng);
    const o = landmarkObservation(LANDMARK, truth);
    const z = { r: o.r + rng.normal(0, sr), phi: normalizeAngle(o.phi + rng.normal(0, sp)), id: 0 };
    const geom = geometry(z, sr, sp);
    const seed = Math.floor(rng.next() * 1e9);
    const trials: Record<ProposalKind, Trial> = {
      'motion-prior': runProposal('motion-prior', z, sr, sp, geom, seed),
      'measurement-aware': runProposal('measurement-aware', z, sr, sp, geom, seed),
    };
    const push = (k: ProposalKind) =>
      [...(prevHistory?.[k] ?? []), { neff: trials[k].neff, covered: trials[k].covered }].slice(-30);
    return {
      rng,
      truth,
      geom,
      trials,
      history: { 'motion-prior': push('motion-prior'), 'measurement-aware': push('measurement-aware') },
    };
  }, []);

  const init = useCallback((seed: number): State => build(new Rng(seed)), [build]);
  const step = useCallback((s: State): State => build(s.rng, s.history), [build]);

  const sim = useSimulation<State>({ init, step, fps: 1.5, initialSeed: 17 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { geom, truth, trials } = sim.state;
      const trial = trials[proposal];

      // The landmark whose sighting is doing all the work.
      ctx.fillStyle = p.accent;
      ctx.beginPath();
      ctx.arc(sx(v, LANDMARK.x), sy(v, LANDMARK.y), 5, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, 'mapped landmark  μⱼ, Σⱼ', sx(v, LANDMARK.x) + 9, sy(v, LANDMARK.y), p.accent, {
        size: 10,
      });

      // Where the robot was, and the odometry arrow to where it thinks it is.
      drawRobot(ctx, v, PREV, p.truth, 0.16, { filled: false });
      label(ctx, 'x_{t−1}', sx(v, PREV.x) - 6, sy(v, PREV.y) + 16, p.truth, { size: 10, align: 'center' });
      ctx.strokeStyle = p.prediction;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, PREV.x), sy(v, PREV.y));
      ctx.lineTo(sx(v, geom.xHat.x), sy(v, geom.xHat.y));
      ctx.stroke();
      ctx.setLineDash([]);

      // The three distributions, in the book's colors.
      drawCovariance(ctx, v, [geom.xHat.x, geom.xHat.y], ellipse2(geom.priorCov, 2), p.prediction, {
        alpha: 0.95,
      });
      drawCovariance(
        ctx,
        v,
        [geom.postMean[0], geom.postMean[1]],
        ellipse2(geom.likelihoodCov, 2),
        p.measurement,
        { alpha: 0.95 },
      );
      drawCovariance(
        ctx,
        v,
        [geom.postMean[0], geom.postMean[1]],
        ellipse2(geom.postCov, Math.sqrt(CHI2_90)),
        p.posterior,
        { alpha: 1, lineWidth: 2.2 },
      );

      // The samples this proposal actually produced.
      const wMax = Math.max(...trial.weights, 1e-12);
      ctx.save();
      for (let i = 0; i < trial.poses.length; i++) {
        const w = trial.weights[i] / wMax;
        ctx.globalAlpha = 0.15 + 0.8 * w;
        ctx.fillStyle = p.posterior;
        ctx.beginPath();
        ctx.arc(sx(v, trial.poses[i].x), sy(v, trial.poses[i].y), 1.2 + 3.4 * Math.sqrt(w), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      drawRobot(ctx, v, truth, p.truth, 0.18);

      // Legend, drawn in the corner in the same colors as the equations.
      const lx = sx(v, SCENE.minX) + 8;
      let ly = sy(v, SCENE.maxY) + 12;
      const legend: [string, string][] = [
        ['motion prior  p(x_t | x_{t−1}, u_t)', p.prediction],
        ['measurement likelihood  p(z_t | x_t)', p.measurement],
        ['posterior (90% mass)', p.posterior],
        ['true pose', p.truth],
      ];
      for (const [text, color] of legend) {
        label(ctx, `— ${text}`, lx, ly, color, { size: 10 });
        ly += 13;
      }

      label(
        ctx,
        proposal === 'motion-prior'
          ? 'FastSLAM 1.0 — sample orange, weight by green'
          : 'FastSLAM 2.0 — sample the product directly',
        sx(v, SCENE.maxX) - 8,
        sy(v, SCENE.maxY) + 12,
        p.ink,
        { size: 11, weight: 700, align: 'right' },
      );
      label(
        ctx,
        `${(trial.covered * 100).toFixed(0)}% of samples inside the purple 90% region`,
        sx(v, SCENE.maxX) - 8,
        sy(v, SCENE.maxY) + 27,
        p.posterior,
        { size: 10, align: 'right' },
      );
    },
    [sim.state, proposal],
  );

  const mean = useMemo(() => {
    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    const h = sim.state.history;
    return {
      one: {
        neff: avg(h['motion-prior'].map((s) => s.neff)),
        covered: avg(h['motion-prior'].map((s) => s.covered)),
      },
      two: {
        neff: avg(h['measurement-aware'].map((s) => s.neff)),
        covered: avg(h['measurement-aware'].map((s) => s.covered)),
      },
      trials: h['motion-prior'].length,
    };
  }, [sim.state.history]);

  return (
    <WidgetFrame
      id="w17.3"
      title="Proposal Quality"
      teaches="A better proposal beats more particles: sampling where the measurement already says the robot is turns wasted particles into useful ones."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          One step, repeated with a fresh measurement every tick. The
          <strong style={{ color: 'var(--pr-prediction)' }}> orange </strong> ellipse is where the
          wheels say the robot might be; the
          <strong style={{ color: 'var(--pr-measurement)' }}> green </strong> one is where the
          landmark sighting says it is; the
          <strong style={{ color: 'var(--pr-posterior)' }}> purple </strong> one is the product,
          which is what the filter is actually trying to represent. FastSLAM 1.0 must sample the
          orange and hope; as you sharpen the sensor with the slider, the green shrinks inside the
          orange and the fraction of useful samples collapses — that is particle depletion arriving
          one step at a time, before any resampling has happened. Switch to 2.0 and the same M
          particles all land inside the purple, with near-uniform weights. Note what 2.0 does
          <em>not</em> fix: if the map were wrong, it would sample confidently into the wrong place.
        </>
      }
    >
      <SimCanvas
        world={SCENE}
        draw={draw}
        deps={[sim.tick, sim.state, proposal]}
        aspect={2.1}
        padding={0.15}
        ariaLabel="A frozen prediction step: an orange motion-prior ellipse, a smaller green measurement-likelihood ellipse, a purple posterior ellipse, and the cloud of pose samples the selected proposal produced."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="1.0 · samples in 90%" value={`${(mean.one.covered * 100).toFixed(0)}%`} />
        <Stat label="1.0 · N_eff / M" value={mean.one.neff.toFixed(2)} alert={mean.one.neff < 0.5} />
        <Stat label="2.0 · samples in 90%" value={`${(mean.two.covered * 100).toFixed(0)}%`} />
        <Stat label="2.0 · N_eff / M" value={mean.two.neff.toFixed(2)} />
      </div>

      <div className="border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton
            onClick={() => setProposal('motion-prior')}
            emphasis={proposal === 'motion-prior'}
          >
            FastSLAM 1.0 proposal
          </ActionButton>
          <ActionButton
            onClick={() => setProposal('measurement-aware')}
            emphasis={proposal === 'measurement-aware'}
          >
            FastSLAM 2.0 proposal
          </ActionButton>
        </ButtonRow>
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="Range noise σ_r"
          role="measurement"
          value={sigmaR}
          min={0.02}
          max={0.4}
          step={0.01}
          unit="m"
          onChange={setSigmaR}
          help="Sharper sensor, smaller green ellipse — and a smaller fraction of 1.0's samples that matter."
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

function Stat({ label: l, value, alert }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={alert ? { color: 'var(--pr-prediction)' } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
