/**
 * Numerical self-checks for Chapter 25's learning module.
 *
 * Same contract as `lib/__checks__.ts`: these are *invariants*, not frozen
 * outputs — identities the mathematics guarantees, which would break if a
 * formula were mistranscribed. The two exceptions are the chapter's worked
 * numeric example (which the prose asks the reader to reproduce by hand) and
 * the closed-form ESS lemma, which is checked to a stated tolerance.
 *
 * Run them directly, or wire them in with:
 *
 *     import { runCh25Checks } from './learn/__checks_ch25__';
 *     out.push(...runCh25Checks());
 */

import { Rng } from '../prob/rng';
import { DEFAULT_BEAM_PARAMS, beamLikelihood } from '../models/sensor';
import { softResample, gradientTransmission } from '../filters/soft-resample';
import type { Particle } from '../filters/pf';
import {
  beamCdf,
  randomizedPit,
  sampleBeam,
  scaleBeamWidth,
} from './beam-density';
import {
  calibrationReport,
  fitTemperature,
  logScore,
  murphyDecomposition,
  predictedEssFraction,
  temperedEss,
} from './calibration';
import { DiffKf1d, finiteDiffGrad, simulateTraj1d, type TrajLog1d } from './diff-kf';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

/** Evaluation pairs (z*, z) drawn from the *true* sensor. */
function sampleEval(n: number, params = DEFAULT_BEAM_PARAMS, seed = 25) {
  const rng = new Rng(seed);
  const pairs: { zStar: number; z: number }[] = [];
  for (let i = 0; i < n; i++) {
    const zStar = rng.uniform(0.8, params.maxRange * 0.85);
    pairs.push({ zStar, z: sampleBeam(zStar, params, rng) });
  }
  return pairs;
}

function grade(
  pairs: { zStar: number; z: number }[],
  params = DEFAULT_BEAM_PARAMS,
  seed = 7,
) {
  const rng = new Rng(seed);
  const pit = pairs.map((p) => randomizedPit(p.z, p.zStar, params, rng));
  const scores = pairs.map((p) => -logScore(beamLikelihood(p.z, p.zStar, params)));
  return calibrationReport(pit, scores);
}

export function runCh25Checks(): CheckResult[] {
  const out: CheckResult[] = [];
  const ok = (name: string, pass: boolean, detail?: string) => out.push({ name, pass, detail });

  /* ---------------- calibration ---------------------------------------- */

  {
    // A CDF must be monotone and land on 1 at the sensor limit.
    const p = DEFAULT_BEAM_PARAMS;
    let monotone = true;
    let prev = -1;
    for (let z = 0; z <= p.maxRange; z += 0.05) {
      const c = beamCdf(z, 3.0, p);
      if (c < prev - 1e-12) monotone = false;
      prev = c;
    }
    const atMax = beamCdf(p.maxRange, 3.0, p);
    ok(
      'beamCdf is a CDF (monotone, F(z_max) = 1)',
      monotone && Math.abs(atMax - 1) < 1e-6,
      `F(z_max) = ${fmt(atMax)}`,
    );
  }

  {
    // D25.2: a model evaluated against its own samples is calibrated.
    const pairs = sampleEval(4000);
    const honest = grade(pairs);
    ok(
      'PIT of the true model is uniform (ECE < 0.02)',
      honest.ece < 0.02,
      `ECE = ${fmt(honest.ece)}, χ² = ${fmt(honest.pitChi2)} on 14 d.o.f.`,
    );

    // F3's premise: halving the claimed width is *accurate* but overconfident.
    const sharp = grade(pairs, scaleBeamWidth(DEFAULT_BEAM_PARAMS, 0.4));
    ok(
      'shrinking σ_hit raises ECE (accuracy ≠ calibration)',
      sharp.ece > honest.ece * 3,
      `ECE ${fmt(honest.ece)} → ${fmt(sharp.ece)} at scale 0.4`,
    );

    // Post-hoc temperature scaling recovers the honest width from a bad start.
    const fitted = fitTemperature((s) => {
      const p = scaleBeamWidth(scaleBeamWidth(DEFAULT_BEAM_PARAMS, 0.4), s);
      let nll = 0;
      for (const pair of pairs) nll += logScore(beamLikelihood(pair.z, pair.zStar, p));
      return nll / pairs.length;
    });
    // The mis-specified model starts at 0.4× the true width; the fitted scale
    // should multiply it back to ≈ 1.
    ok(
      'fitTemperature recovers the honest width (0.4 · s ≈ 1)',
      Math.abs(0.4 * fitted.scale - 1) < 0.15,
      `s = ${fmt(fitted.scale)}, so 0.4·s = ${fmt(0.4 * fitted.scale)}`,
    );
  }

  {
    // F2: Murphy's identity, BS = reliability − resolution + uncertainty.
    const rng = new Rng(4);
    const forecasts: number[] = [];
    const outcomes: (0 | 1)[] = [];
    for (let i = 0; i < 20000; i++) {
      const p = rng.uniform(0.02, 0.98);
      forecasts.push(p);
      outcomes.push(rng.next() < p ? 1 : 0);
    }
    const d = murphyDecomposition(forecasts, outcomes);
    const recomposed = d.reliability - d.resolution + d.uncertainty;
    ok(
      'Murphy decomposition recomposes the Brier score',
      Math.abs(recomposed - d.brier) < 5e-3,
      `BS = ${fmt(d.brier)}, rel − res + unc = ${fmt(recomposed)}`,
    );
  }

  /* ---------------- tempering and ESS ---------------------------------- */

  {
    // F3's lemma: ESS/M ≈ exp(−κ² s²) for log-likelihoods with spread s.
    const rng = new Rng(9);
    const s = 0.6;
    const M = 20000;
    const ll = Array.from({ length: M }, () => rng.normal(0, s));
    let worst = 0;
    for (const kappa of [0.5, 1, 1.5, 2]) {
      const measured = temperedEss(ll, kappa) / M;
      const predicted = predictedEssFraction(s, kappa);
      worst = Math.max(worst, Math.abs(measured - predicted));
    }
    ok('ESS/M ≈ exp(−κ² s²) across κ ∈ [0.5, 2]', worst < 0.05, `max |Δ| = ${fmt(worst)}`);
  }

  /* ---------------- soft resampling ------------------------------------ */

  {
    const weights = [0.4, 0.25, 0.2, 0.1, 0.04, 0.01];
    let worst = 0;
    for (const lambda of [0, 0.25, 0.5, 0.75, 1]) {
      worst = Math.max(worst, Math.abs(gradientTransmission(weights, lambda) - (1 - lambda)));
    }
    ok('expected gradient transmission equals 1 − λ', worst < 1e-12, `max |Δ| = ${fmt(worst)}`);
  }

  {
    // F6: soft resampling is unbiased. E[Σ w'_i f(x_i)] = Σ w_i f(x_i).
    const weights = [0.45, 0.28, 0.15, 0.08, 0.03, 0.01];
    const values = [1, 2, 3, 4, 5, 6];
    const particles: Particle[] = weights.map((w, i) => ({
      state: { x: values[i], y: 0, theta: 0 },
      weight: w,
    }));
    const target = weights.reduce((a, w, i) => a + w * values[i], 0);

    const rng = new Rng(31);
    let worst = 0;
    for (const lambda of [0.2, 0.5, 0.9]) {
      let acc = 0;
      const trials = 4000;
      for (let k = 0; k < trials; k++) {
        const res = softResample(particles, lambda, rng);
        let est = 0;
        for (const p of res.particles) est += p.weight * p.state.x;
        acc += est;
      }
      worst = Math.max(worst, Math.abs(acc / trials - target));
    }
    ok(
      'soft resampling is unbiased for every λ',
      worst < 0.05,
      `target = ${fmt(target)}, max |Δ| = ${fmt(worst)}`,
    );
  }

  /* ---------------- the differentiable filter --------------------------- */

  {
    // The chapter's worked example: one step, r = q = 1, σ₀ = 0, z₁ = 2.
    const log: TrajLog1d = {
      a: 1,
      b: 0,
      u: [0],
      z: [2],
      x: [0],
      mu0: 0,
      sigma0: 0,
      trueR: 1,
      trueQ: 1,
    };
    const kf = new DiffKf1d(0, 0);
    const g = kf.lossAndGrad(log, 'evidence');
    const expected = 0.5 * (Math.log(4 * Math.PI) + 2);
    ok(
      'worked example: L = ½(ln 4π + 2), ∂L/∂log r = ∂L/∂log q = −¼',
      Math.abs(g.loss - expected) < 1e-12 &&
        Math.abs(g.dLogR + 0.25) < 1e-12 &&
        Math.abs(g.dLogQ + 0.25) < 1e-12,
      `L = ${fmt(g.loss)}, ∇ = (${fmt(g.dLogR)}, ${fmt(g.dLogQ)})`,
    );
  }

  {
    // F5: the analytic sensitivity recursion agrees with finite differences.
    const log = simulateTraj1d(
      { steps: 120, a: 0.96, b: 0.1, r: 0.05, q: 0.4, x0: 0.2, sigma0: 0.6 },
      new Rng(2025),
    );
    let worst = 0;
    for (const loss of ['evidence', 'stateNll'] as const) {
      for (const theta of [
        [-1.5, 0.4],
        [0.3, -0.8],
        [-3, -3],
      ]) {
        const kf = new DiffKf1d(theta[0], theta[1]);
        const a = kf.lossAndGrad(log, loss);
        const fd = finiteDiffGrad(kf, log, loss);
        worst = Math.max(
          worst,
          Math.abs(a.dLogR - fd.dLogR) / (1 + Math.abs(fd.dLogR)),
          Math.abs(a.dLogQ - fd.dLogQ) / (1 + Math.abs(fd.dLogQ)),
        );
      }
    }
    ok('analytic KF gradient matches central differences', worst < 1e-5, `max rel Δ = ${fmt(worst)}`);
  }

  {
    // F4: maximizing the evidence recovers the noises that generated the log,
    // with no ground truth ever consulted.
    const log = simulateTraj1d(
      { steps: 1500, a: 1, b: 0.05, r: 0.08, q: 0.25, x0: 0, sigma0: 0.5 },
      new Rng(1234),
    );
    const kf = new DiffKf1d(Math.log(0.4), Math.log(0.02)); // deliberately wrong both ways
    for (let epoch = 0; epoch < 12000; epoch++) kf.sgdStep(log, 'evidence', 0.05 / log.z.length);
    const rErr = Math.abs(Math.log(kf.r / log.trueR));
    const qErr = Math.abs(Math.log(kf.q / log.trueQ));
    ok(
      'evidence training recovers (r, q) without ground truth',
      rErr < 0.2 && qErr < 0.1,
      `r: ${fmt(kf.r)} vs ${log.trueR}, q: ${fmt(kf.q)} vs ${log.trueQ}`,
    );
  }

  return out;
}
