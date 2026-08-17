/**
 * Numerical self-checks for Chapter 15's factor-graph module.
 *
 * Same contract as `lib/__checks__.ts`: these are *invariants*, not frozen
 * outputs — identities the mathematics guarantees, which would break if a
 * formula were mistranscribed. They are kept in a separate file so that the
 * chapter's algorithms carry their own proof of correctness next to the code.
 *
 * Wire them into `runSelfChecks()` (or run them directly) with:
 *
 *     import { runCh15Checks } from './optim/__checks_ch15__';
 *     out.push(...runCh15Checks());
 */

import { compose, se2Exp, se2Log, type Twist2 } from '../geom/se2';
import { solve } from '../prob/linalg';
import {
  buildIndex,
  gaussNewton,
  graphCost,
  levenbergMarquardt,
  linearizeGraph,
  poseRmse,
  retract,
  se2RightJacobian,
  se2RightJacobianInv,
} from './factor-graph';
import { kernelRho, kernelWeight, type Kernel } from './kernels';
import { buildPattern, orderVariables, schurMarginalize, symbolicElimination } from './ordering';
import { apartmentLoopData, buildSlamGraph, microChain1D, sparsityScene } from './scenes';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

export function runCh15Checks(): CheckResult[] {
  const out: CheckResult[] = [];
  const check = (name: string, fn: () => { pass: boolean; detail?: string }) => {
    try {
      const r = fn();
      out.push({ name, pass: r.pass, detail: r.detail });
    } catch (e) {
      out.push({ name, pass: false, detail: `threw: ${(e as Error).message}` });
    }
  };

  // 1 ---------------------------------------------------------------------
  check('optim: J_r(τ)⁻¹ is the first-order map of Log(Exp(τ)Exp(δ))', () => {
    const tau: Twist2 = [0.7, -0.4, 0.9];
    const d: Twist2 = [1e-6, -2e-6, 3e-6];
    const lhs = se2Log(compose(se2Exp(tau), se2Exp(d)));
    const Ji = se2RightJacobianInv(tau);
    let worst = 0;
    for (let i = 0; i < 3; i++) {
      const rhs = tau[i] + Ji[i][0] * d[0] + Ji[i][1] * d[1] + Ji[i][2] * d[2];
      worst = Math.max(worst, Math.abs(lhs[i] - rhs));
    }
    // …and it really is the inverse of J_r.
    const Jr = se2RightJacobian(tau);
    let inv = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += Jr[i][k] * Ji[k][j];
        inv = Math.max(inv, Math.abs(s - (i === j ? 1 : 0)));
      }
    }
    return {
      pass: worst < 1e-9 && inv < 1e-12,
      detail: `first-order error ${fmt(worst)}, ‖J_r J_r⁻¹ − I‖∞ = ${fmt(inv)}`,
    };
  });

  // 2 ---------------------------------------------------------------------
  check('optim: analytic factor Jacobians match ⊞ finite differences', () => {
    const data = apartmentLoopData({ seed: 7 });
    const graph = buildSlamGraph(data, { outlier: 1.0 });
    const v = data.init;
    const eps = 1e-6;
    let worst = 0;
    for (const f of graph.factors) {
      const lin = f.linearize(v);
      f.keys.forEach((k, a) => {
        const slot = data.index.slot.get(`${k.kind[0]}${k.id}`);
        if (slot === undefined) return;
        for (let c = 0; c < data.index.dims[slot]; c++) {
          const delta = new Array<number>(data.index.total).fill(0);
          delta[data.index.offsets[slot] + c] = eps;
          const rp = f.linearize(retract(v, delta, data.index)).r;
          for (let r = 0; r < f.dim; r++) {
            const num = (rp[r] - lin.r[r]) / eps;
            const ana = lin.J[a][r][c];
            worst = Math.max(worst, Math.abs(num - ana) / Math.max(1, Math.abs(ana)));
          }
        }
      });
    }
    return { pass: worst < 2e-4, detail: `worst relative error ${fmt(worst)} over ${graph.factors.length} factors` };
  });

  // 3 ---------------------------------------------------------------------
  check('optim: the 1-D micro chain reproduces the chapter’s worked example', () => {
    const { graph, index, init, expected } = microChain1D();
    const sys = linearizeGraph(graph, init, index);
    const okOmega =
      Math.abs(sys.Omega[0][0] - 2) < 1e-12 &&
      Math.abs(sys.Omega[0][1] + 1) < 1e-12 &&
      Math.abs(sys.Omega[0][2]) < 1e-12 &&
      Math.abs(sys.b[0] - 1) < 1e-12 &&
      Math.abs(sys.b[2] + 2.5) < 1e-12 &&
      Math.abs(sys.cost - 2.125) < 1e-12;
    const step = solve(
      sys.Omega,
      sys.b.map((x) => -x),
    );
    const { values } = gaussNewton(graph, init, index, { maxIterations: 5 });
    const okSolve = expected.every(
      (e, i) => Math.abs(step[i] - e) < 1e-9 && Math.abs(values.scalars[i] - e) < 1e-9,
    );
    // The tension is shared: both odometry intervals shrink by the same amount.
    const gapA = values.scalars[1] - values.scalars[0];
    const gapB = values.scalars[2] - values.scalars[1];
    const okShare = Math.abs(gapA - 0.875) < 1e-9 && Math.abs(gapB - 0.875) < 1e-9;
    const okCost = Math.abs(graphCost(graph, values) - 1 / 32) < 1e-12;
    return {
      pass: okOmega && okSolve && okShare && okCost,
      detail: `ŷ = (${values.scalars.map((x) => x.toFixed(3)).join(', ')}), J: 2.125 → ${fmt(graphCost(graph, values))}`,
    };
  });

  // 4 ---------------------------------------------------------------------
  check('optim: smoothing beats dead reckoning on the Apartment loop', () => {
    const data = apartmentLoopData({ seed: 42 });
    const graph = buildSlamGraph(data);
    const before = graphCost(graph, data.init);
    const { values, report } = levenbergMarquardt(graph, data.init, data.index, {
      maxIterations: 30,
    });
    const rmse0 = poseRmse(data.init.poses, data.truthPoses);
    const rmse1 = poseRmse(values.poses, data.truthPoses);
    return {
      pass: report.converged && report.finalCost < 0.1 * before && rmse1 < 0.5 * rmse0,
      detail: `J ${before.toFixed(0)} → ${report.finalCost.toFixed(1)} in ${report.iterations.length} steps; RMSE ${rmse0.toFixed(3)} → ${rmse1.toFixed(3)} m`,
    };
  });

  // 5 ---------------------------------------------------------------------
  check('optim: an outlier bends the L2 map, and a redescending kernel releases it', () => {
    const data = apartmentLoopData({ seed: 42 });
    const clean = levenbergMarquardt(buildSlamGraph(data), data.init, data.index, {
      maxIterations: 30,
    });
    const rmseClean = poseRmse(clean.values.poses, data.truthPoses);
    const fit = (kernel: Kernel) =>
      poseRmse(
        levenbergMarquardt(buildSlamGraph(data, { outlier: 1.6, kernel }), data.init, data.index, {
          maxIterations: 40,
        }).values.poses,
        data.truthPoses,
      );
    const l2 = fit({ type: 'l2' });
    const gm = fit({ type: 'geman', c: 1 });
    return {
      pass: l2 > 2 * rmseClean && Math.abs(gm - rmseClean) < 0.03,
      detail: `RMSE clean ${rmseClean.toFixed(3)}, L2 ${l2.toFixed(3)}, Geman–McClure ${gm.toFixed(3)} m`,
    };
  });

  // 6 ---------------------------------------------------------------------
  check('optim: w = ρ′(e)/e for every kernel', () => {
    const kernels: Kernel[] = [
      { type: 'l2' },
      { type: 'huber', k: 1.2 },
      { type: 'cauchy', c: 1 },
      { type: 'geman', c: 1.5 },
    ];
    const e = 2.3;
    const h = 1e-5;
    let worst = 0;
    for (const k of kernels) {
      const numeric = (kernelRho(k, e + h) - kernelRho(k, e - h)) / (2 * h) / e;
      worst = Math.max(worst, Math.abs(numeric - kernelWeight(k, e)));
    }
    return { pass: worst < 1e-6, detail: `max |w − ρ′/e| = ${fmt(worst)} at e = ${e}σ` };
  });

  // 7 ---------------------------------------------------------------------
  check('optim: Schur elimination changes nothing but the size of the system', () => {
    const scene = sparsityScene('loop');
    const order = orderVariables(scene.graph, 'chronological');
    const index = buildIndex(order);
    const sys = linearizeGraph(scene.graph, scene.values, index);
    const damp = (m: number[][]) => m.map((r, i) => r.map((x, j) => (i === j ? x + 1e-9 : x)));
    const full = solve(
      damp(sys.Omega),
      sys.b.map((x) => -x),
    );
    const reduced = schurMarginalize(
      sys,
      index,
      order.filter((k) => k.kind === 'landmark'),
    );
    const dx = solve(
      damp(reduced.Omega),
      reduced.b.map((x) => -x),
    );
    let worst = 0;
    reduced.index.order.forEach((k, s) => {
      const o = reduced.index.offsets[s];
      const fo = index.offsets[index.slot.get(`${k.kind[0]}${k.id}`) as number];
      for (let i = 0; i < reduced.index.dims[s]; i++) {
        worst = Math.max(worst, Math.abs(dx[o + i] - full[fo + i]));
      }
    });
    return { pass: worst < 1e-7, detail: `max |Δ_reduced − Δ_full| = ${fmt(worst)}` };
  });

  // 8 ---------------------------------------------------------------------
  check('optim: elimination only ever adds fill, and ordering changes the amount', () => {
    const scene = sparsityScene('loop');
    const counts = (['chronological', 'landmarks-first', 'poses-first', 'min-degree'] as const).map(
      (name) => symbolicElimination(buildPattern(scene.graph, orderVariables(scene.graph, name))),
    );
    const monotone = counts.every((r) => r.nnzL >= r.nnzLowerOmega);
    const spread = Math.max(...counts.map((c) => c.nnzL)) > Math.min(...counts.map((c) => c.nnzL));
    // The hub graph is the counter-example to "landmarks first is always best".
    const hub = sparsityScene('hub');
    const first = symbolicElimination(
      buildPattern(hub.graph, orderVariables(hub.graph, 'landmarks-first')),
    ).nnzL;
    const last = symbolicElimination(
      buildPattern(hub.graph, orderVariables(hub.graph, 'poses-first')),
    ).nnzL;
    return {
      pass: monotone && spread && first > last,
      detail: `loop nnz(L) ∈ [${Math.min(...counts.map((c) => c.nnzL))}, ${Math.max(...counts.map((c) => c.nnzL))}]; hub: landmarks-first ${first} vs poses-first ${last}`,
    };
  });

  return out;
}

/** Convenience for a console or a status widget. */
export function ch15CheckSummary(): { passed: number; total: number; failures: CheckResult[] } {
  const results = runCh15Checks();
  const failures = results.filter((r) => !r.pass);
  return { passed: results.length - failures.length, total: results.length, failures };
}
