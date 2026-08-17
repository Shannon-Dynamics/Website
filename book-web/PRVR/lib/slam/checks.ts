/**
 * Numerical self-checks for the Chapter 16 scan-matching and pose-graph port.
 *
 * Same contract as `lib/__checks__.ts`: each entry asserts an identity the
 * mathematics guarantees, or reproduces a worked example printed in the
 * chapter, so that a reader who does the algebra by hand gets the number the
 * widget shows. Splice `slamSelfChecks()` into `runSelfChecks()`' output — the
 * shape is identical — or run it on its own.
 */

import { between, se2Log, type Pose2 } from '../geom/se2';
import { Rng } from '../prob/rng';
import { APARTMENT, beamAngles, simulateScan } from '../sim/world';
import { estimateNormals, scanToCloud, transformCloud, voxelDownsample, VoxelMap, type Pt } from './cloud';
import { icp, svdAlign } from './icp';
import { buildNdt, ndtAlign } from './ndt';
import { informationFromSigmas, PoseGraph } from './posegraph';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

export function slamSelfChecks(): CheckResult[] {
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
  check('icp: closed-form alignment reproduces the chapter’s worked example', () => {
    const src: Pt[] = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ];
    // cos θ = 0.8, sin θ = 0.6 (θ = 36.8699°), t = (2, 1).
    const dst: Pt[] = [
      [2.8, 1.6],
      [1.4, 1.8],
      [1.2, 0.4],
      [2.6, 0.2],
    ];
    const t = svdAlign(src, dst);
    const err =
      Math.abs(Math.cos(t.theta) - 0.8) +
      Math.abs(Math.sin(t.theta) - 0.6) +
      Math.abs(t.x - 2) +
      Math.abs(t.y - 1);
    return {
      pass: err < 1e-12,
      detail: `θ = ${((t.theta * 180) / Math.PI).toFixed(4)}°, t = (${fmt(t.x)}, ${fmt(t.y)})`,
    };
  });

  // 2 ---------------------------------------------------------------------
  check('icp: alignment is exact under an arbitrary rigid motion of the same cloud', () => {
    const rng = new Rng(31);
    const src: Pt[] = Array.from({ length: 40 }, () => [rng.uniform(-3, 3), rng.uniform(-3, 3)]);
    const truth: Pose2 = { x: -1.4, y: 2.7, theta: -1.05 };
    const dst = transformCloud(truth, src);
    const t = svdAlign(src, dst);
    const e = se2Log(between(truth, t));
    const err = Math.hypot(e[0], e[1]) + Math.abs(e[2]);
    return { pass: err < 1e-12, detail: `‖T̂ ⊟ T‖ = ${fmt(err)} over 40 random points` };
  });

  // 3 ---------------------------------------------------------------------
  check('posegraph: a loop error spreads evenly around the cycle', () => {
    const g = new PoseGraph();
    const i = informationFromSigmas(1, 1, 1);
    g.addNode({ x: 0, y: 0, theta: 0 }, true);
    g.addNode({ x: 1, y: 0, theta: 0 });
    g.addNode({ x: 2, y: 0, theta: 0 });
    g.addEdge(0, 1, { x: 1, y: 0, theta: 0 }, i);
    g.addEdge(1, 2, { x: 1, y: 0, theta: 0 }, i);
    g.addEdge(0, 2, { x: 2.6, y: 0, theta: 0 }, i, 'loop');
    const before = g.chi2();
    g.optimize(30, 1e-14);
    const p = g.poses();
    const after = g.chi2();
    return {
      pass:
        Math.abs(before - 0.36) < 1e-12 &&
        Math.abs(p[1].x - 1.2) < 1e-9 &&
        Math.abs(p[2].x - 2.4) < 1e-9 &&
        Math.abs(after - 0.12) < 1e-9 &&
        Math.abs(p[0].x) < 1e-15,
      detail: `χ² ${fmt(before)} → ${fmt(after)}, poses ${p.map((q) => q.x.toFixed(3)).join(', ')}`,
    };
  });

  // 4 ---------------------------------------------------------------------
  check('posegraph: optimization never increases χ²', () => {
    const rng = new Rng(9);
    const g = new PoseGraph();
    let pose: Pose2 = { x: 0, y: 0, theta: 0 };
    g.addNode(pose, true);
    const omega = informationFromSigmas(0.05, 0.05, 0.02);
    for (let k = 1; k <= 12; k++) {
      const z: Pose2 = { x: 0.5, y: 0, theta: 0.4 };
      pose = { x: pose.x + rng.normal(0, 0.05), y: pose.y + rng.normal(0, 0.05), theta: pose.theta };
      g.addNode(pose);
      g.addEdge(k - 1, k, z, omega);
    }
    const report = g.optimize(25, 1e-10);
    let monotone = true;
    for (let k = 1; k < report.chi2.length; k++) {
      if (report.chi2[k] > report.chi2[0] + 1e-9) monotone = false;
    }
    return {
      pass: monotone && report.chi2[report.chi2.length - 1] < report.chi2[0],
      detail: `χ² ${fmt(report.chi2[0])} → ${fmt(report.chi2[report.chi2.length - 1])} in ${report.iterations} iterations`,
    };
  });

  // 5 ---------------------------------------------------------------------
  check('icp: both variants recover a known relative pose from two real sweeps', () => {
    const params = { nBeams: 180, fov: 2 * Math.PI, maxRange: 6, sigma: 0.012 };
    const angles = beamAngles(params);
    const rng = new Rng(0x1c9);
    const a: Pose2 = { x: 2.0, y: 1.9, theta: 0 };
    const b: Pose2 = { x: 2.5, y: 1.55, theta: 0.35 };
    const target = voxelDownsample(
      scanToCloud(simulateScan(APARTMENT, a, params, rng), angles, params.maxRange).points,
      0.06,
    );
    const map = new VoxelMap(0.25, 8);
    map.insert(target, estimateNormals(target, 0.3));
    const source = voxelDownsample(
      scanToCloud(simulateScan(APARTMENT, b, params, rng), angles, params.maxRange).points,
      0.09,
    );
    const truth = between(a, b);
    const errors = (['point-to-point', 'point-to-plane'] as const).map((variant) => {
      const r = icp(source, map, truth, { variant, tau: 0.7, maxIters: 30 });
      const e = se2Log(between(truth, r.pose));
      return Math.hypot(e[0], e[1]);
    });
    return {
      pass: errors.every((e) => e < 0.03),
      detail: `point-to-point ${fmt(errors[0])} m, point-to-plane ${fmt(errors[1])} m`,
    };
  });

  // 6 ---------------------------------------------------------------------
  check('ndt: the score is maximized at the true pose, and Newton finds it', () => {
    const params = { nBeams: 180, fov: 2 * Math.PI, maxRange: 6, sigma: 0.01 };
    const angles = beamAngles(params);
    const rng = new Rng(0x5c0);
    const pose: Pose2 = { x: 2.0, y: 1.9, theta: 0 };
    const cloud = voxelDownsample(
      scanToCloud(simulateScan(APARTMENT, pose, params, rng), angles, params.maxRange).points,
      0.08,
    );
    const nd = buildNdt(transformCloud(pose, cloud), 0.5, 4);
    const peak = nd.score(cloud, pose);
    const off = nd.score(cloud, { x: pose.x + 0.4, y: pose.y + 0.25, theta: pose.theta + 0.05 });
    const r = ndtAlign(cloud, nd, { x: pose.x + 0.4, y: pose.y + 0.25, theta: pose.theta + 0.05 });
    const err = Math.hypot(r.pose.x - pose.x, r.pose.y - pose.y);
    return {
      pass: peak > off && err < 0.05 && r.score > off,
      detail: `score ${fmt(off)} → ${fmt(r.score)} (peak ${fmt(peak)}), recovered to ${fmt(err)} m`,
    };
  });

  // 7 ---------------------------------------------------------------------
  check('cloud: the voxel hash agrees with brute-force nearest neighbour', () => {
    const rng = new Rng(4242);
    const pts: Pt[] = Array.from({ length: 500 }, () => [rng.uniform(-5, 5), rng.uniform(-5, 5)]);
    const map = new VoxelMap(0.3, 64);
    map.insert(pts);
    let worst = 0;
    for (let k = 0; k < 200; k++) {
      const q: Pt = [rng.uniform(-5, 5), rng.uniform(-5, 5)];
      const tau = 1.0;
      let bruteD = tau;
      for (const p of pts) {
        const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < bruteD) bruteD = d;
      }
      const idx = map.nearestIndex(q, tau);
      const hashD = idx < 0 ? tau : Math.hypot(map.pts[idx][0] - q[0], map.pts[idx][1] - q[1]);
      worst = Math.max(worst, Math.abs(hashD - bruteD));
    }
    return { pass: worst < 1e-12, detail: `max |hash − brute force| = ${fmt(worst)} over 200 queries` };
  });

  return out;
}
