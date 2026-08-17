/**
 * Numerical self-checks for the capstone stack.
 *
 * Same discipline as `lib/__checks__.ts`: every number the chapter prints is
 * reproduced here, so prose and code cannot drift apart silently. The mission
 * check is the expensive one and the important one — it is the book's
 * end-to-end regression, and it is deterministic under its seed.
 *
 * Run with:
 *   npx esbuild lib/capstone/checks.ts --bundle --platform=node --format=esm \
 *     --outfile=/tmp/ch26.mjs && node -e "import('/tmp/ch26.mjs').then(m=>m.runCapstoneChecks())"
 */

import { FitnessMonitor } from './detectors';
import { normalQuantile, safetyMargin } from './esdf';
import { runMission } from './stack';

export interface Check {
  name: string;
  ok: boolean;
  detail: string;
}

const near = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol;

export function capstoneChecks(): Check[] {
  const out: Check[] = [];
  const push = (name: string, ok: boolean, detail: string) => out.push({ name, ok, detail });

  // --- F2: the chance-constraint margin ------------------------------------
  const k = normalQuantile(0.99);
  push('Φ⁻¹(0.99) = 2.3263', near(k, 2.3263479, 1e-5), k.toFixed(7));

  const margin = safetyMargin(0.19, 0.05, 0.01);
  push('margin(r=0.19, σ=0.05, δ=0.01) = 0.3063 m', near(margin, 0.3063174, 1e-6), margin.toFixed(6));

  // A ten-fold tighter bound costs less than four centimetres: the Gaussian
  // tail is steep, which is why chance constraints are cheap — and why they are
  // a poor defence against error that is not actually Gaussian.
  const tighter = safetyMargin(0.19, 0.05, 0.001);
  push(
    'δ: 1% → 0.1% costs 3.8 cm',
    near(tighter - margin, 0.03819, 5e-5),
    (tighter - margin).toFixed(5),
  );

  // --- F4(a): the dual-EMA kidnap detector, the chapter's worked example ---
  const mon = new FitnessMonitor();
  mon.update(0.98); // seeds w_fast = w_slow = 0.98
  const alarm1 = mon.update(0.44);
  const rho1 = mon.rho;
  const alarm2 = mon.update(0.44);
  const rho2 = mon.rho;
  push('ρ₁ = 0.7450 after one bad scan', near(rho1, 0.745016, 1e-5), rho1.toFixed(6));
  push('ρ₂ = 0.6200 after two', near(rho2, 0.620047, 1e-5), rho2.toFixed(6));
  push('alarm on the second scan, not the first', !alarm1 && alarm2, `${alarm1} then ${alarm2}`);

  return out;
}

/**
 * The end-to-end mission regression. Separated because it takes a couple of
 * seconds: seed 42 must still map the apartment, and must still do it without
 * touching a wall.
 */
export function missionCheck(): Check[] {
  const r = runMission({ seed: 42 }, 3200);
  return [
    { name: 'seed 42 reaches Done', ok: r.mode === 'Done', detail: r.mode },
    {
      name: 'seed 42 coverage ≥ 99%',
      ok: r.coverage >= 0.99,
      detail: `${(r.coverage * 100).toFixed(1)}%`,
    },
    {
      name: 'seed 42 trajectory RMSE ≤ 0.20 m',
      ok: r.ate <= 0.2,
      detail: `${r.ate.toFixed(3)} m`,
    },
    {
      name: 'SLAM beats dead reckoning by 10× or more',
      ok: r.odomError / Math.max(r.ate, 1e-6) >= 10,
      detail: `${r.odomError.toFixed(2)} m odometry vs ${r.ate.toFixed(3)} m estimate`,
    },
    { name: 'seed 42 never touches a wall', ok: r.contacts === 0, detail: `${r.contacts} contacts` },
  ];
}

export function runCapstoneChecks(withMission = true): boolean {
  const checks = withMission ? [...capstoneChecks(), ...missionCheck()] : capstoneChecks();
  let ok = true;
  for (const c of checks) {
    if (!c.ok) ok = false;
    // eslint-disable-next-line no-console
    console.log(`${c.ok ? 'ok  ' : 'FAIL'}  ${c.name}  [${c.detail}]`);
  }
  return ok;
}
