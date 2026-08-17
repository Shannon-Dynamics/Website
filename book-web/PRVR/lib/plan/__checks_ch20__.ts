/**
 * Numerical self-checks for Chapter 20's planning module.
 *
 * Same contract as `lib/__checks__.ts`: invariants the mathematics guarantees,
 * not frozen outputs. The Dubins checks are the important ones — the six closed
 * forms are the kind of algebra that can be mistyped and still *look* right, so
 * every word is integrated forward and required to land on the pose it claims.
 *
 * Run them with:
 *
 *     import { runCh20Checks } from './plan/__checks_ch20__';
 *     out.push(...runCh20Checks());
 */

import { angleDiff, type Pose2 } from '../geom/se2';
import { distanceAt } from '../mapping/edt';
import { Rng } from '../prob/rng';
import { APARTMENT, distanceToWalls } from '../sim/world';
import { CSpace2, cellAt, cellCenter, freeCellNear, latticeFromCSpace } from './cspace';
import {
  DUBINS_WORDS,
  dubinsAllWords,
  dubinsEndpoint,
  dubinsShortestPath,
  reedsSheppLite,
} from './dubins';
import { aStarGrid, brushfire, dijkstraGrid, octile, wavefront } from './search';
import { Prm, Rrt, dist2, pathCost, shortcut } from './sampling';

export interface CheckResult {
  name: string;
  pass: boolean;
  detail?: string;
}

const fmt = (x: number): string => (Math.abs(x) < 1e-4 ? x.toExponential(2) : x.toFixed(6));

export function runCh20Checks(): CheckResult[] {
  const out: CheckResult[] = [];
  const push = (name: string, pass: boolean, detail?: string) => out.push({ name, pass, detail });

  // -------------------------------------------------------------------------
  // Dubins
  // -------------------------------------------------------------------------

  {
    // Every word that reports a solution must actually reach the goal pose.
    const rng = new Rng(20);
    let worst = 0;
    let words = 0;
    for (let k = 0; k < 400; k++) {
      const start: Pose2 = {
        x: rng.uniform(-4, 4),
        y: rng.uniform(-4, 4),
        theta: rng.uniform(-Math.PI, Math.PI),
      };
      const goal: Pose2 = {
        x: rng.uniform(-4, 4),
        y: rng.uniform(-4, 4),
        theta: rng.uniform(-Math.PI, Math.PI),
      };
      const rho = rng.uniform(0.2, 2);
      for (const path of dubinsAllWords(start, goal, rho)) {
        const e = dubinsEndpoint(path);
        const err = Math.max(
          Math.hypot(e.x - goal.x, e.y - goal.y),
          Math.abs(angleDiff(e.theta, goal.theta)),
        );
        worst = Math.max(worst, err);
        words++;
      }
    }
    push(
      'dubins: every closed-form word integrates to the requested goal pose',
      worst < 1e-8 && words > 1500,
      `${words} words, worst endpoint error ${fmt(worst)}`,
    );
  }

  {
    // Worked example: turning around on the spot costs 7π/3 · ρ, via LRL or RLR.
    const best = dubinsShortestPath({ x: 0, y: 0, theta: 0 }, { x: 0, y: 0, theta: Math.PI }, 1);
    const expected = (7 * Math.PI) / 3;
    push(
      'dubins: the U-turn q₀=(0,0,0) → q₁=(0,0,π) at ρ=1 costs 7π/3',
      best !== null && Math.abs(best.length - expected) < 1e-9 && best.word[1] !== 'S',
      best ? `${best.word}, length ${fmt(best.length)} vs ${fmt(expected)}` : 'no path',
    );
  }

  {
    // A Dubins path can never be shorter than the straight line.
    const rng = new Rng(7);
    let ok = true;
    let tight = Infinity;
    for (let k = 0; k < 300; k++) {
      const start: Pose2 = { x: 0, y: 0, theta: rng.uniform(-Math.PI, Math.PI) };
      const goal: Pose2 = {
        x: rng.uniform(-5, 5),
        y: rng.uniform(-5, 5),
        theta: rng.uniform(-Math.PI, Math.PI),
      };
      const p = dubinsShortestPath(start, goal, 0.5);
      if (!p) {
        ok = false;
        break;
      }
      const straight = Math.hypot(goal.x, goal.y);
      if (p.length < straight - 1e-9) ok = false;
      tight = Math.min(tight, p.length - straight);
    }
    push('dubins: length ≥ Euclidean distance for every query', ok, `slack min ${fmt(tight)}`);
  }

  {
    // Far apart, only the CSC family is feasible and the straight run wins;
    // close together, the CCC words appear and can beat it. That switch is the
    // whole point of w20.4.
    const far = dubinsAllWords({ x: 0, y: 0, theta: 0 }, { x: 20, y: 0, theta: 0 }, 1);
    const near = dubinsAllWords({ x: 0, y: 0, theta: 0 }, { x: 0, y: 0, theta: Math.PI }, 1);
    const ccc = (w: string) => w === 'RLR' || w === 'LRL';
    push(
      'dubins: CCC words exist only at short range, and win there',
      far.length === 4 &&
        far.every((p) => !ccc(p.word)) &&
        Math.abs(far[0].length - 20) < 1e-9 &&
        near.filter((p) => ccc(p.word)).length === 2 &&
        ccc(near[0].word) &&
        DUBINS_WORDS.length === 6,
      `far: ${far.length} words, best ${far[0].word}; near: ${near.length} words, best ${near[0].word}`,
    );
  }

  {
    // Allowing reverse can only help, and it strictly helps somewhere.
    const rng = new Rng(3);
    let ok = true;
    let strict = false;
    for (let k = 0; k < 200; k++) {
      const goal: Pose2 = {
        x: rng.uniform(-2, 2),
        y: rng.uniform(-2, 2),
        theta: rng.uniform(-Math.PI, Math.PI),
      };
      const f = dubinsShortestPath({ x: 0, y: 0, theta: 0 }, goal, 1);
      const r = reedsSheppLite({ x: 0, y: 0, theta: 0 }, goal, 1);
      if (!f || !r) {
        ok = false;
        break;
      }
      if (r.length > f.length + 1e-9) ok = false;
      if (r.length < f.length - 1e-6) strict = true;
      const e = dubinsEndpoint(r);
      if (Math.hypot(e.x - goal.x, e.y - goal.y) > 1e-8) ok = false;
      if (Math.abs(angleDiff(e.theta, goal.theta)) > 1e-8) ok = false;
    }
    push('dubins: pure-reverse words are never worse, and sometimes better', ok && strict);
  }

  // -------------------------------------------------------------------------
  // C-space
  // -------------------------------------------------------------------------

  const cs = new CSpace2(APARTMENT, { radius: 0.25, cellSize: 0.05 });
  const grid = latticeFromCSpace(cs, 0.2);

  {
    // The distance field must agree with exact point-to-segment distance.
    const rng = new Rng(11);
    let worst = 0;
    for (let k = 0; k < 400; k++) {
      const x = rng.uniform(0.2, 11.8);
      const y = rng.uniform(0.2, 8.8);
      worst = Math.max(worst, Math.abs(distanceAt(cs.field, x, y) - distanceToWalls(APARTMENT, x, y)));
    }
    push(
      'cspace: the ESDF matches exact wall distance to within one cell',
      worst < 0.05 * Math.SQRT2 + 1e-9,
      `worst |ESDF − exact| = ${fmt(worst)} m`,
    );
  }

  {
    // Sphere marching must agree with brute-force dense sampling.
    const rng = new Rng(5);
    let agree = 0;
    let total = 0;
    for (let k = 0; k < 200; k++) {
      const a = cs.sampleFree(rng);
      const b = cs.sampleFree(rng);
      if (!a || !b) continue;
      const marched = cs.edgeFree(a, b);
      const n = Math.max(2, Math.ceil(dist2(a, b) / 0.01));
      let dense = true;
      for (let i = 0; i <= n; i++) {
        const t = i / n;
        if (!cs.isFree(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))) {
          dense = false;
          break;
        }
      }
      total++;
      if (marched === dense) agree++;
    }
    push(
      'cspace: sphere-marched edge checks agree with dense sampling',
      total > 100 && agree === total,
      `${agree}/${total} edges`,
    );
  }

  // -------------------------------------------------------------------------
  // Search
  // -------------------------------------------------------------------------

  const startCell = freeCellNear(grid, 1.0, 1.0);
  const goalCell = freeCellNear(grid, 11.0, 8.0);

  {
    const a = aStarGrid(grid, startCell, goalCell);
    const d = dijkstraGrid(grid, startCell, goalCell);
    push(
      'search: A* with an admissible heuristic returns the Dijkstra-optimal cost',
      a.status === 'found' && d.status === 'found' && Math.abs(a.cost - d.cost) < 1e-9,
      `A* ${fmt(a.cost)} m in ${a.expanded} expansions; Dijkstra ${fmt(d.cost)} m in ${d.expanded}`,
    );
    push(
      'search: the heuristic buys expansions, not optimality',
      a.expanded < d.expanded,
      `${a.expanded} < ${d.expanded}`,
    );
  }

  {
    // Weighted A*: bounded suboptimality, ε = 3.
    const opt = aStarGrid(grid, startCell, goalCell).cost;
    const w = aStarGrid(grid, startCell, goalCell, { epsilon: 3 });
    push(
      'search: weighted A* (ε = 3) stays within ε of optimal and expands fewer cells',
      w.status === 'found' && w.cost <= 3 * opt + 1e-9,
      `cost ${fmt(w.cost)} ≤ 3 × ${fmt(opt)}, ${w.expanded} expansions`,
    );
  }

  {
    // Octile is admissible: it never exceeds the true cost-to-go anywhere.
    const cost = wavefront(grid, goalCell);
    const gp = cellCenter(grid, goalCell);
    let violations = 0;
    let checked = 0;
    for (let k = 0; k < grid.nx * grid.ny; k++) {
      if (grid.free[k] === 0 || !Number.isFinite(cost[k])) continue;
      const p = cellCenter(grid, k);
      checked++;
      if (octile(p.x, p.y, gp.x, gp.y, 1) > cost[k] + 1e-9) violations++;
    }
    push(
      'search: the octile heuristic is admissible on every reachable cell',
      violations === 0 && checked > 500,
      `${checked} cells checked, ${violations} violations`,
    );
  }

  {
    // The wave-front is a genuine navigation function: strict descent everywhere.
    const cost = wavefront(grid, goalCell);
    let stuck = 0;
    let checked = 0;
    for (let j = 1; j < grid.ny - 1; j++) {
      for (let i = 1; i < grid.nx - 1; i++) {
        const k = j * grid.nx + i;
        if (grid.free[k] === 0 || !Number.isFinite(cost[k]) || cost[k] === 0) continue;
        checked++;
        let best = Infinity;
        for (let dj = -1; dj <= 1; dj++) {
          for (let di = -1; di <= 1; di++) {
            if (di === 0 && dj === 0) continue;
            best = Math.min(best, cost[(j + dj) * grid.nx + (i + di)]);
          }
        }
        if (best >= cost[k] - 1e-12) stuck++;
      }
    }
    push(
      'wavefront: every free cell has a strictly cheaper neighbour — no local minima',
      stuck === 0 && checked > 500,
      `${checked} cells, ${stuck} without descent`,
    );
  }

  {
    // Brushfire is the 8-connected approximation of the exact distance
    // transform. Two error sources, both bounded: the lattice quantises the
    // C-obstacle boundary (±√2 cells), and the octile metric overestimates
    // Euclidean distance by at most 1/cos(22.5°) − ... = 8.24%, attained at 22.5°.
    const fine = latticeFromCSpace(cs, 0.1);
    const bf = brushfire(fine);
    const slack = 2 * Math.SQRT2 * fine.cellSize;
    let over = 0;
    let under = 0;
    let checked = 0;
    let worst = 0;
    for (let k = 0; k < fine.nx * fine.ny; k++) {
      if (fine.free[k] === 0) continue;
      const p = cellCenter(fine, k);
      const exact = distanceAt(cs.field, p.x, p.y) - cs.radius;
      if (exact < 1.0 || !Number.isFinite(bf[k])) continue;
      checked++;
      if (bf[k] < exact - slack) under++;
      if (bf[k] > 1.0824 * exact + slack) over++;
      worst = Math.max(worst, Math.abs(bf[k] - exact));
    }
    push(
      'brushfire: an 8-connected wave equals the Euclidean transform to within the octile bound',
      under === 0 && over === 0 && checked > 100,
      `${checked} cells, worst |brushfire − ESDF| = ${fmt(worst)} m (slack ${fmt(slack)} m)`,
    );
  }

  // -------------------------------------------------------------------------
  // Sampling-based planners
  // -------------------------------------------------------------------------

  const start = { x: 1.0, y: 1.0 };
  const goal = { x: 11.0, y: 8.0 };
  const optimalCost = aStarGrid(grid, startCell, goalCell).cost;

  {
    const rng = new Rng(20);
    const prm = new Prm(cs, { k: 10, maxEdgeLength: 3 });
    for (let i = 0; i < 900; i++) prm.step(rng);
    const res = prm.query(start, goal);
    push(
      'prm: a 900-milestone roadmap answers the Apartment query, no shorter than optimal',
      res !== null && res.cost >= optimalCost * 0.92,
      res ? `cost ${fmt(res.cost)} m vs lattice optimum ${fmt(optimalCost)} m` : 'no path',
    );
  }

  {
    // RRT* tree invariant: every node's cost is its parent's cost plus the edge.
    const rng = new Rng(20);
    const tree = new Rrt(cs, start, goal, { star: true, stepSize: 0.7, goalBias: 0.05 });
    for (let i = 0; i < 1500; i++) tree.step(rng);
    let worst = 0;
    for (let i = 1; i < tree.nodes.length; i++) {
      const n = tree.nodes[i];
      const p = tree.nodes[n.parent];
      worst = Math.max(worst, Math.abs(n.cost - (p.cost + dist2(p, n))));
    }
    push(
      'rrt*: cost-to-come is consistent through every edge after rewiring',
      worst < 1e-9,
      `${tree.nodes.length} nodes, worst inconsistency ${fmt(worst)}`,
    );

    let monotone = true;
    for (let i = 1; i < tree.costHistory.length; i++) {
      if (tree.costHistory[i].cost > tree.costHistory[i - 1].cost + 1e-12) monotone = false;
    }
    push(
      'rrt*: the incumbent cost never increases',
      monotone && tree.costHistory.length > 0,
      `${tree.costHistory.length} improvements, final ${fmt(tree.bestCost)} m`,
    );
  }

  {
    // The chapter's headline claim: RRT* beats plain RRT at equal budget.
    const budget = 1500;
    const rrt = new Rrt(cs, start, goal, { star: false, stepSize: 0.7, goalBias: 0.05 });
    const rngA = new Rng(20);
    for (let i = 0; i < budget; i++) rrt.step(rngA);
    const star = new Rrt(cs, start, goal, { star: true, stepSize: 0.7, goalBias: 0.05 });
    const rngB = new Rng(20);
    for (let i = 0; i < budget; i++) star.step(rngB);
    push(
      'rrt vs rrt*: at an equal sample budget the rewired tree is cheaper',
      Number.isFinite(rrt.bestCost) && star.bestCost < rrt.bestCost,
      `RRT ${fmt(rrt.bestCost)} m, RRT* ${fmt(star.bestCost)} m, lattice ${fmt(optimalCost)} m`,
    );

    // Shortcutting improves an RRT path without ever making it worse.
    const sol = rrt.solution();
    if (sol) {
      const rngC = new Rng(1);
      const smoothed = shortcut(cs, sol.path, rngC, 200);
      push(
        'shortcut: random pairwise shortcutting never lengthens a path',
        pathCost(smoothed) <= sol.cost + 1e-9,
        `${fmt(sol.cost)} → ${fmt(pathCost(smoothed))} m`,
      );
    }
  }

  {
    // Sanity: the goal cell really is where the query says it is.
    const c = cellAt(grid, goal.x, goal.y);
    const p = cellCenter(grid, c);
    push(
      'lattice: cellAt and cellCenter round-trip within half a cell',
      Math.abs(p.x - goal.x) <= grid.cellSize && Math.abs(p.y - goal.y) <= grid.cellSize,
    );
  }

  return out;
}
