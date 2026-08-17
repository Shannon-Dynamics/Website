/**
 * Numerical self-checks for the Chapter 14 EKF SLAM port.
 *
 * Same contract as `lib/__checks__.ts`: every entry asserts an identity the
 * mathematics guarantees, or reproduces a number printed in the chapter, so a
 * reader who does the algebra by hand gets what the widget shows. Call
 * `ekfSlamSelfChecks()` on its own, or splice it into `runSelfChecks()` —
 * the result shape is identical.
 */

import { angleDiff, normalizeAngle } from '../geom/se2';
import { EkfSlam, DEFAULT_SLAM_CONFIG, landmarkIndex, type SlamConfig } from './ekf-slam';
import { CourseSim, monteCarloNees } from './course';
import { chi2Envelope } from '../filters/consistency';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

/** The chapter's worked-example noise: σ_r = 0.1 m, σ_φ = 0.05 rad. */
function workedConfig(): SlamConfig {
  return {
    ...DEFAULT_SLAM_CONFIG,
    motion: { ...DEFAULT_SLAM_CONFIG.motion },
    sigmaR: 0.1,
    sigmaPhi: 0.05,
  };
}

export function ekfSlamSelfChecks(): CheckResult[] {
  const out: CheckResult[] = [];
  const check = (name: string, fn: () => { pass: boolean; detail?: string }) => {
    try {
      const r = fn();
      out.push({ name, ...r });
    } catch (e) {
      out.push({ name, pass: false, detail: String(e) });
    }
  };

  // ---------------------------------------------------------------------
  // D2 — landmark birth, the chapter's hand-checkable example
  // ---------------------------------------------------------------------

  check('ch14 D2: birth at r=2, φ=0 from an exact pose gives Σ_jj = diag(0.01, 0.01)', () => {
    const f = new EkfSlam(
      { x: 0, y: 0, theta: 0 },
      [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      workedConfig(),
    );
    const j = f.initLandmark({ r: 2, phi: 0, s: 7 }, 7);
    const [mx, my] = f.landmarkMean(j);
    const c = f.landmarkCov(j);
    // σ_r² = 0.01 along the beam, r²σ_φ² = 4·0.0025 = 0.01 across it.
    const pass =
      close(mx, 2, 1e-12) &&
      close(my, 0, 1e-12) &&
      close(c[0][0], 0.01, 1e-12) &&
      close(c[1][1], 0.01, 1e-12) &&
      close(c[0][1], 0, 1e-12);
    return { pass, detail: `μ=(${fmt(mx)}, ${fmt(my)}), Σ=diag(${fmt(c[0][0])}, ${fmt(c[1][1])})` };
  });

  check('ch14 D2: a second identical observation halves it to diag(0.005, 0.005)', () => {
    const f = new EkfSlam(
      { x: 0, y: 0, theta: 0 },
      [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      workedConfig(),
    );
    const j = f.initLandmark({ r: 2, phi: 0, s: 7 }, 7);
    f.updateLandmark(j, { r: 2, phi: 0, s: 7 });
    const c = f.landmarkCov(j);
    return {
      pass: close(c[0][0], 0.005, 1e-12) && close(c[1][1], 0.005, 1e-12),
      detail: `Σ=diag(${fmt(c[0][0])}, ${fmt(c[1][1])})`,
    };
  });

  // ---------------------------------------------------------------------
  // D4 — the floor set by the initial vehicle covariance
  // ---------------------------------------------------------------------

  check('ch14 D4(iii): Σ_jj converges to the initial vehicle variance, not to zero', () => {
    const s0 = 0.04;
    const f = new EkfSlam(
      { x: 0, y: 0, theta: 0 },
      [
        [s0, 0, 0],
        [0, s0, 0],
        [0, 0, 1e-10], // heading pinned, to isolate the translation argument
      ],
      workedConfig(),
    );
    const j = f.initLandmark({ r: 2, phi: 0, s: 1 }, 1);
    for (let i = 0; i < 400; i++) f.updateLandmark(j, { r: 2, phi: 0, s: 1 });
    const c = f.landmarkCov(j);
    const rho = f.correlation()[0][landmarkIndex(j)];
    return {
      pass: close(c[0][0], s0, 1e-3) && close(c[1][1], s0, 1e-3) && rho > 0.999,
      detail: `Σ_jj=diag(${fmt(c[0][0])}, ${fmt(c[1][1])}) vs σ₀²=${fmt(s0)}, ρ(x, m_x)=${fmt(rho)}`,
    };
  });

  // ---------------------------------------------------------------------
  // D1 — prediction is blockwise and O(N)
  // ---------------------------------------------------------------------

  check('ch14 D1: prediction leaves the map–map block of Σ exactly unchanged', () => {
    const f = new EkfSlam({ x: 0, y: 0, theta: 0 }, [
      [0.01, 0, 0],
      [0, 0.01, 0],
      [0, 0, 0.002],
    ]);
    for (let k = 0; k < 5; k++) {
      f.initLandmark({ r: 2 + 0.2 * k, phi: -1 + 0.4 * k, s: k }, k);
    }
    const before = f.sigma.map((r) => r.slice());
    f.predict({ v: 0.5, omega: 0.3, dt: 0.2 });
    let maxDiff = 0;
    for (let a = 3; a < f.dim; a++) {
      for (let b = 3; b < f.dim; b++) {
        maxDiff = Math.max(maxDiff, Math.abs(f.sigma[a][b] - before[a][b]));
      }
    }
    // The pose–map strip, by contrast, must have moved: motion rotates it.
    let stripMoved = 0;
    for (let b = 3; b < f.dim; b++) stripMoved = Math.max(stripMoved, Math.abs(f.sigma[0][b] - before[0][b]));
    return {
      pass: maxDiff < 1e-15 && stripMoved > 1e-9,
      detail: `max |ΔΣ_mm| = ${fmt(maxDiff)}, max |ΔΣ_xm| = ${fmt(stripMoved)}`,
    };
  });

  check('ch14 D1: the motion Jacobian matches a central difference', () => {
    const build = () => {
      const f = new EkfSlam({ x: 0.7, y: -0.3, theta: 0.4 }, [
        [1e-6, 0, 0],
        [0, 1e-6, 0],
        [0, 0, 1e-6],
      ]);
      return f;
    };
    const u = { v: 0.6, omega: 0.35, dt: 0.25 };
    const g = (pose: [number, number, number]) => {
      const f = build();
      f.mu[0] = pose[0];
      f.mu[1] = pose[1];
      f.mu[2] = pose[2];
      f.predict(u);
      return [f.mu[0], f.mu[1], f.mu[2]] as [number, number, number];
    };
    const base: [number, number, number] = [0.7, -0.3, 0.4];
    const eps = 1e-6;
    // Analytic Jacobian, read back out of a prediction with an identity Σ_xx.
    const probe = build();
    probe.sigma = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    probe.mu[0] = base[0];
    probe.mu[1] = base[1];
    probe.mu[2] = base[2];
    probe.cfg = { ...probe.cfg, motion: { alongTrack: 0, crossTrack: 0, headingPerMetre: 0, headingPerRadian: 0 } };
    probe.predict(u);
    // With Σ = I and R = 0, Σ⁺ = G Gᵀ, so the numeric G must reproduce it.
    const numeric = [0, 1, 2].map((col) => {
      const plus = base.slice() as [number, number, number];
      const minus = base.slice() as [number, number, number];
      plus[col] += eps;
      minus[col] -= eps;
      const a = g(plus);
      const b = g(minus);
      return [0, 1, 2].map((row) =>
        row === 2 ? angleDiff(a[2], b[2]) / (2 * eps) : (a[row] - b[row]) / (2 * eps),
      );
    });
    // numeric[col][row] → G[row][col]; form G Gᵀ and compare.
    let worst = 0;
    for (let i = 0; i < 3; i++) {
      for (let k = 0; k < 3; k++) {
        let s = 0;
        for (let c = 0; c < 3; c++) s += numeric[c][i] * numeric[c][k];
        worst = Math.max(worst, Math.abs(s - probe.sigma[i][k]));
      }
    }
    return { pass: worst < 1e-6, detail: `max |G Gᵀ − Σ⁺| = ${fmt(worst)}` };
  });

  // ---------------------------------------------------------------------
  // D3 — the update is Θ(N²) and its innovation is consistent
  // ---------------------------------------------------------------------

  check('ch14 D3: one observation update writes exactly (3 + 2N)² covariance entries', () => {
    const counts: string[] = [];
    let pass = true;
    for (const n of [1, 4, 16, 64]) {
      const f = new EkfSlam({ x: 0, y: 0, theta: 0 }, [
        [0.01, 0, 0],
        [0, 0.01, 0],
        [0, 0, 0.002],
      ]);
      for (let k = 0; k < n; k++) {
        f.initLandmark({ r: 2, phi: (k / n) * 2 * Math.PI - Math.PI, s: k }, k);
      }
      const before = f.entriesTouched;
      f.updateLandmark(0, { r: 2, phi: -Math.PI, s: 0 });
      const written = f.entriesTouched - before;
      const want = (3 + 2 * n) ** 2;
      if (written !== want) pass = false;
      counts.push(`N=${n}: ${written} (want ${want})`);
    }
    return { pass, detail: counts.join(', ') };
  });

  check('ch14 D3: the measurement Jacobian matches a central difference', () => {
    const f = new EkfSlam({ x: 0.2, y: -0.1, theta: 0.3 }, [
      [0.01, 0, 0],
      [0, 0.01, 0],
      [0, 0, 0.002],
    ]);
    const j = f.initLandmark({ r: 2.4, phi: 0.7, s: 0 }, 0);
    const { h } = f.expected(j);
    const idx = [0, 1, 2, landmarkIndex(j), landmarkIndex(j) + 1];
    const eps = 1e-6;
    let worst = 0;
    for (let c = 0; c < 5; c++) {
      const saved = f.mu[idx[c]];
      f.mu[idx[c]] = saved + eps;
      const zp = f.expected(j).z;
      f.mu[idx[c]] = saved - eps;
      const zm = f.expected(j).z;
      f.mu[idx[c]] = saved;
      worst = Math.max(worst, Math.abs((zp[0] - zm[0]) / (2 * eps) - h[0][c]));
      worst = Math.max(worst, Math.abs(angleDiff(zp[1], zm[1]) / (2 * eps) - h[1][c]));
    }
    return { pass: worst < 1e-6, detail: `max |h_analytic − h_numeric| = ${fmt(worst)}` };
  });

  // ---------------------------------------------------------------------
  // The lab: the Apartment landmark course
  // ---------------------------------------------------------------------

  check('ch14 lab: known correspondences map all 8 beacons and close the loop', () => {
    const sim = new CourseSim({ seed: 42, knownCorrespondence: true });
    let closures = 0;
    for (let i = 0; i < 320; i++) closures += sim.step().closures.length;
    const labels = [...sim.filter.labels].sort((a, b) => a - b);
    const expected = [0, 1, 2, 3, 4, 5, 6, 7];
    return {
      pass: labels.length === 8 && labels.every((l, i) => l === expected[i]) && closures > 0,
      detail: `N=${sim.filter.count}, labels=[${labels.join(',')}], loop closures=${closures}`,
    };
  });

  check('ch14 lab: the west cluster settles at the prior floor σ₀ = 0.05 m', () => {
    const sim = new CourseSim({ seed: 42, knownCorrespondence: true });
    for (let i = 0; i < 600; i++) sim.step();
    const slot = sim.filter.slotOf(0);
    const c = sim.filter.landmarkCov(slot);
    const sigma = Math.sqrt(0.5 * (c[0][0] + c[1][1]));
    // D4(iii): bounded below by the initial vehicle σ, and within measurement
    // noise of it after a few laps.
    return {
      pass: sigma > 0.045 && sigma < 0.09,
      detail: `σ(beacon 0) = ${fmt(sigma)} m vs σ₀ = 0.05 m`,
    };
  });

  check('ch14 w14.1: the diagonal-only ablation makes the filter certain and wrong', () => {
    const run = (ablate: boolean) => {
      const sim = new CourseSim({
        seed: 42,
        knownCorrespondence: true,
        config: { ablateCorrelations: ablate },
      });
      for (let i = 0; i < 400; i++) sim.step();
      let sq = 0;
      let k = 0;
      for (let j = 0; j < sim.filter.count; j++) {
        const b = sim.truthFor(j);
        if (!b) continue;
        const [mx, my] = sim.filter.landmarkMean(j);
        sq += (mx - b.x) ** 2 + (my - b.y) ** 2;
        k += 1;
      }
      return {
        claimed: Math.sqrt(sim.filter.mapUncertainty()),
        actual: k > 0 ? Math.sqrt(sq / k) : 0,
        rho: sim.filter.meanLandmarkCorrelation(),
      };
    };
    const honest = run(false);
    const ablated = run(true);
    return {
      // The ablated filter claims an order of magnitude less uncertainty than
      // the honest one while being at least as wrong — the chapter's point.
      pass:
        ablated.claimed < 0.1 * honest.claimed &&
        ablated.actual > 10 * ablated.claimed &&
        ablated.rho < 1e-12 &&
        honest.rho > 0.3,
      detail: `honest: claimed ${fmt(honest.claimed)} / actual ${fmt(honest.actual)}; ablated: claimed ${fmt(ablated.claimed)} / actual ${fmt(ablated.actual)}`,
    };
  });

  check('ch14 w14.2: mean NEES leaves the χ² band for long loops and not for short ones', () => {
    const short = monteCarloNees(20, 300, {
      seed: 11,
      knownCorrespondence: true,
      params: { eastX: 3.5 },
    });
    const long = monteCarloNees(20, 300, {
      seed: 11,
      knownCorrespondence: true,
      params: { eastX: 11.2 },
    });
    const band = chi2Envelope(3, 20);
    const tail = (a: number[]) => {
      const t = a.slice(200);
      return t.reduce((x, y) => x + y, 0) / t.length;
    };
    const s = tail(short.meanNees);
    const l = tail(long.meanNees);
    return {
      pass: s < band.hi && l > band.hi,
      detail: `short loop ${fmt(s)}, long loop ${fmt(l)}, band [${fmt(band.lo)}, ${fmt(band.hi)}]`,
    };
  });

  check('ch14 w14.4: the provisional list keeps clutter out of the state', () => {
    const count = (useProvisional: boolean) => {
      const sim = new CourseSim({
        seed: 21,
        knownCorrespondence: false,
        sensor: { clutterRate: 0.6 },
        config: { useProvisional },
      });
      for (let i = 0; i < 400; i++) sim.step();
      return sim.filter.count;
    };
    const without = count(false);
    const withList = count(true);
    return {
      pass: withList < without / 3,
      detail: `${without} landmarks without the list, ${withList} with it (8 beacons exist)`,
    };
  });

  check('ch14: Σ stays symmetric and positive-definite over a full run', () => {
    const sim = new CourseSim({ seed: 7, knownCorrespondence: true });
    for (let i = 0; i < 400; i++) sim.step();
    const S = sim.filter.sigma;
    let asym = 0;
    let minDiag = Infinity;
    for (let i = 0; i < S.length; i++) {
      minDiag = Math.min(minDiag, S[i][i]);
      for (let j = 0; j < S.length; j++) asym = Math.max(asym, Math.abs(S[i][j] - S[j][i]));
    }
    // Correlations must also stay in [-1, 1]; a violation means the downdate
    // has drifted out of the PSD cone.
    const rho = sim.filter.correlation();
    let worstRho = 0;
    for (let i = 0; i < rho.length; i++) {
      for (let j = 0; j < rho.length; j++) worstRho = Math.max(worstRho, Math.abs(rho[i][j]));
    }
    return {
      pass: asym < 1e-12 && minDiag > 0 && worstRho <= 1 + 1e-9,
      detail: `max asymmetry ${fmt(asym)}, min diagonal ${fmt(minDiag)}, max |ρ| ${fmt(worstRho)}`,
    };
  });

  check('ch14: bearing residuals wrap, so a heading near ±π is not a catastrophe', () => {
    const f = new EkfSlam({ x: 0, y: 0, theta: Math.PI - 0.01 }, [
      [0.04, 0, 0],
      [0, 0.04, 0],
      [0, 0, 0.01],
    ]);
    const j = f.initLandmark({ r: 2, phi: 0, s: 0 }, 0);
    // Nudge the estimate across the branch cut and update: the correction must
    // stay small, not swing by 2π.
    f.mu[2] = normalizeAngle(-Math.PI + 0.01);
    const before = f.mu[2];
    const info = f.updateLandmark(j, { r: 2, phi: 0, s: 0 });
    const moved = Math.abs(angleDiff(f.mu[2], before));
    return {
      pass: Math.abs(info.innovation[1]) < 0.1 && moved < 0.1,
      detail: `bearing innovation ${fmt(info.innovation[1])} rad, heading moved ${fmt(moved)} rad`,
    };
  });

  return out;
}
