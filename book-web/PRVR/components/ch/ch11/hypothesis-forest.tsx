'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { StatTile, Dashboard } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { angleDiff, normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import { ellipse2, type Mat } from '@/lib/prob/linalg';
import { sampleMotionModelVelocity, type MotionAlphas, type VelocityCmd } from '@/lib/models/motion';
import type { Landmark } from '@/lib/sim/world';
import {
  MhtLocalizer,
  gateThreshold,
  positionBlock,
  type Feature,
  type Hypothesis,
} from '@/lib/filters/ekf-localization';
import {
  clear,
  drawCovariance,
  drawPath,
  drawRobot,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w11.4 — the Hypothesis Forest.
 *
 * The surveyor recorded two landmarks where there is only one object: a
 * doorframe entered twice, 0.60 m apart. Rusty's gate contains both. An EKF
 * must pick one *now*, and if it picks wrong it will spend the rest of the run
 * being confidently wrong about it.
 *
 * MHT refuses to pick. It carries both, weighted by how well each explains the
 * data, and lets later evidence decide. The tree beside the map is the price:
 * every ambiguous feature multiplies the number of Gaussians, and only pruning
 * keeps that finite. Pruning is also the flaw — a branch thrown away is gone
 * for good.
 */

const DT = 0.2;
const V = 0.7;
const RUN = 80;
/** Low rotational noise on purpose: the ambiguity this widget is about lives
 *  along the corridor, and a wandering heading would only muddy it. */
const ALPHAS: MotionAlphas = [0.05, 0.002, 0.002, 0.01, 0.0005, 0.0005];
const SENSOR_RANGE = 3.0;
const FOV = (5 * Math.PI) / 6;
const SIGMA_R = 0.15;
const SIGMA_PHI = 0.04;
/** The feature detector runs at a quarter of the control rate, as they do. */
const DETECT_EVERY = 4;
/** How many steps one press of the beacon button keeps looking. */
const BEACON_BURST = 3;

const START: Pose2 = { x: 1.0, y: 2.0, theta: 0 };

const Q: Mat = [
  [SIGMA_R * SIGMA_R, 0],
  [0, SIGMA_PHI * SIGMA_PHI],
];

/** The two objects that actually exist and reflect. */
const TRUE_LANDMARKS: Landmark[] = [
  { x: 3.4, y: 3.3, id: 0 },
  { x: 7.4, y: 0.8, id: 1 },
];

/** A unique, unambiguous beacon — but the robot has to *choose* to look at it. */
const BEACON: Landmark = { x: 6.0, y: 3.6, id: 4 };

/** What the *map* says. Both objects were surveyed twice, 0.60 m apart. */
const MAP_LANDMARKS: Landmark[] = [
  { x: 3.4, y: 3.3, id: 0 },
  { x: 4.0, y: 3.3, id: 1 }, // phantom twin of object 0
  { x: 7.4, y: 0.8, id: 2 },
  { x: 8.0, y: 0.8, id: 3 }, // phantom twin of object 1
  BEACON,
];

const INITIAL_SIGMA: Mat = [
  [0.1225, 0, 0],
  [0, 0.0625, 0],
  [0, 0, 0.0036],
];

interface Params {
  pruneRatio: number;
  maxHyps: number;
}

interface State {
  rng: Rng;
  truth: Pose2;
  hyps: Hypothesis[];
  born: number;
  pruned: number[][];
  path: { x: number; y: number }[];
  /** Steps of beacon observation still owed, from the button. */
  pending: number;
}

/** Collapse runs of the same association: the *decisions*, not the frames. */
function compress(history: number[]): number[] {
  const out: number[] = [];
  for (const h of history) if (out[out.length - 1] !== h) out.push(h);
  return out;
}

export function HypothesisForest() {
  const [params, setParams] = useState<Params>({ pruneRatio: 0.005, maxHyps: 8 });

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      truth: { ...START },
      hyps: [
        {
          mu: { ...START },
          Sigma: INITIAL_SIGMA.map((r) => r.slice()),
          w: 1,
          history: [],
          id: 0,
          parent: -1,
        },
      ],
      born: 1,
      pruned: [],
      path: [{ x: START.x, y: START.y }],
      pending: 0,
    }),
    [],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      const { rng } = s;
      const u: VelocityCmd = { v: V, omega: 0, dt: DT };
      const truth = sampleMotionModelVelocity(u, s.truth, ALPHAS, rng);

      const mht = new MhtLocalizer(
        { mu: s.hyps[0].mu, Sigma: s.hyps[0].Sigma },
        {
          landmarks: MAP_LANDMARKS,
          Q,
          alphas: ALPHAS,
          gate2: gateThreshold(0.95, 2),
          pruneRatio: params.pruneRatio,
          maxHyps: params.maxHyps,
          mergeD2: 0.05,
          clutterDensity: 0.02,
          extraNoise: [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, (ALPHAS[4] * V * V) * DT * DT],
          ],
        },
      );
      mht.hyps = s.hyps;
      mht.born = s.born;
      mht.pruned = s.pruned;
      mht.predict(u);

      const features: Feature[] = [];
      const observe = (lm: Landmark, force = false) => {
        const dx = lm.x - truth.x;
        const dy = lm.y - truth.y;
        const r = Math.hypot(dx, dy);
        const phi = angleDiff(Math.atan2(dy, dx), truth.theta);
        if (!force && (r > SENSOR_RANGE || Math.abs(phi) > FOV / 2)) return;
        features.push({
          r: Math.max(0.05, r + rng.normal(0, SIGMA_R)),
          phi: normalizeAngle(phi + rng.normal(0, SIGMA_PHI)),
          truth: lm.id,
        });
      };

      if (tick % DETECT_EVERY === 0) {
        observe(TRUE_LANDMARKS[0]);
        observe(TRUE_LANDMARKS[1]);
      }
      // The beacon is never seen by accident: looking at it is an *action*.
      if (s.pending > 0) observe(BEACON, true);

      mht.correct(features);

      return {
        rng,
        truth,
        hyps: mht.hyps,
        born: mht.born,
        pruned: mht.pruned,
        path: [...s.path, { x: truth.x, y: truth.y }].slice(-RUN),
        pending: Math.max(0, s.pending - 1),
      };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 8, maxTicks: RUN, loop: true, initialSeed: 7 });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { truth, hyps } = sim.state;

      // The corridor.
      ctx.save();
      ctx.strokeStyle = p.wall;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, 0));
      ctx.lineTo(sx(v, 13), sy(v, 0));
      ctx.moveTo(sx(v, 0), sy(v, 4));
      ctx.lineTo(sx(v, 13), sy(v, 4));
      ctx.stroke();
      ctx.restore();

      // The map: five recorded landmarks. Filled circles are real objects;
      // hollow ones are the surveyor's phantoms, which reflect nothing. The
      // beacon is the diamond — real, unique, and only seen on request.
      const realIds = new Set(TRUE_LANDMARKS.map((l) => `${l.x},${l.y}`));
      for (const lm of MAP_LANDMARKS) {
        const X = sx(v, lm.x);
        const Y = sy(v, lm.y);
        ctx.save();
        ctx.strokeStyle = p.accent;
        ctx.fillStyle = p.accent;
        ctx.lineWidth = 1.5;
        if (lm.id === BEACON.id) {
          ctx.beginPath();
          ctx.moveTo(X, Y - 6);
          ctx.lineTo(X + 6, Y);
          ctx.lineTo(X, Y + 6);
          ctx.lineTo(X - 6, Y);
          ctx.closePath();
          if (sim.state.pending > 0) ctx.fill();
          else ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(X, Y, 4.5, 0, Math.PI * 2);
          if (realIds.has(`${lm.x},${lm.y}`)) ctx.fill();
          else ctx.stroke();
        }
        ctx.restore();
        label(ctx, lm.id === BEACON.id ? 'beacon' : `m${lm.id}`, X - 6, Y - 13, p.truth, { size: 9 });
      }

      drawPath(ctx, v, sim.state.path, p.truth, { dashed: true, lineWidth: 1.4, alpha: 0.8 });

      // Sensor reach, from the true pose.
      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.arc(sx(v, truth.x), sy(v, truth.y), sl(v, SENSOR_RANGE), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // The mixture. Opacity is weight, so the reader sees belief mass, not
      // just hypothesis count.
      const wMax = hyps.reduce((m, h) => Math.max(m, h.w), 1e-9);
      for (const h of hyps) {
        const a = 0.2 + 0.8 * (h.w / wMax);
        drawCovariance(ctx, v, [h.mu.x, h.mu.y], ellipse2(positionBlock(h.Sigma), 2), p.posterior, {
          alpha: a,
          lineWidth: 1.2 + 1.6 * (h.w / wMax),
        });
        drawRobot(ctx, v, h.mu, p.posterior, 0.17, { alpha: a });
      }

      drawRobot(ctx, v, truth, p.truth, 0.2, { filled: false });

      label(
        ctx,
        `${hyps.length} Gaussians alive   ·   best weight ${(wMax * 100).toFixed(0)}%`,
        sx(v, 0.2),
        sy(v, 3.72),
        p.posterior,
        { size: 11, weight: 600 },
      );
    },
    [sim.state],
  );

  /* ---------------- the association tree ---------------- */

  const tree = useMemo(() => buildTree(sim.state.hyps, sim.state.pruned), [sim.state]);

  const best = sim.state.hyps.reduce((m, h) => (h.w > m.w ? h : m), sim.state.hyps[0]);

  return (
    <WidgetFrame
      id="w11.4"
      title="The Hypothesis Forest"
      teaches="A filter does not have to decide immediately. Deferring the decision is a real option — and its price is measured in Gaussians."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          The surveyor entered two objects twice, 0.60 m apart. Filled circles are the objects that
          really reflect; hollow ones are the phantoms — and a phantom 0.60 m away sits comfortably
          inside a gate that is two metres long. Each time Rusty sees one of the doubled objects,
          MHT branches instead of guessing, and the tree on the right grows a level.{' '}
          <strong>What to notice:</strong> the two ellipses sit 0.6 m apart and neither dies,
          because both explain every subsequent reading perfectly well — each from its own pose.
          The weights settle near 80/20, not 100/0: that residual 20% is exactly the risk an EKF
          would have taken silently. <strong>What to try:</strong> press{' '}
          <em>look at the beacon</em> — a unique landmark the robot must choose to look at — and
          watch the mixture collapse. That is what an <em>active</em> localizer buys, and Chapter 22
          is about deciding when it is worth the detour. Then pull the prune ratio up to 0.5: the
          mixture becomes cheaper and decisive, and occasionally decisively wrong, because the
          branch it needed was thrown away three seconds ago.
        </>
      }
    >
      <div className="grid gap-0 lg:grid-cols-[1.5fr_1fr] lg:divide-x lg:divide-fd-border">
        <SimCanvas
          world={{ minX: 0, maxX: 13, minY: 0, maxY: 4 }}
          draw={draw}
          deps={[sim.tick, sim.state]}
          aspect={13 / 4.6}
          padding={0.25}
          ariaLabel="A corridor with five mapped landmarks, two of which are phantom duplicates. Several translucent purple uncertainty ellipses represent competing hypotheses about the robot's pose."
        />

        <div className="border-t border-fd-border p-3 lg:border-t-0">
          <p className="eyebrow mb-2">association histories</p>
          <svg
            viewBox={`0 0 ${tree.width} ${tree.height}`}
            className="w-full"
            role="img"
            aria-label="A tree of association histories. Live branches are drawn in purple with thickness proportional to weight; pruned branches are gray."
          >
            {tree.edges.map((e) => (
              <line
                key={e.key}
                x1={e.x1}
                y1={e.y1}
                x2={e.x2}
                y2={e.y2}
                stroke={e.alive ? 'var(--pr-posterior)' : 'var(--pr-grid)'}
                strokeWidth={e.alive ? 0.4 + 1.6 * e.w : 0.35}
                strokeDasharray={e.alive ? undefined : '1.5 1.5'}
              />
            ))}
            {tree.nodes.map((n) => (
              <g key={n.key}>
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={n.depth === 0 ? 2 : 1.7 + 2.2 * n.w}
                  fill={n.alive ? 'var(--pr-posterior)' : 'var(--pr-canvas-bg)'}
                  stroke={n.alive ? 'var(--pr-posterior)' : 'var(--pr-grid)'}
                  strokeWidth={0.35}
                  opacity={n.alive ? 0.35 + 0.65 * n.w : 0.8}
                />
                {n.depth > 0 ? (
                  <text
                    x={n.x + 3}
                    y={n.y + 1.2}
                    style={{ fontSize: 3 }}
                    fill={n.alive ? 'var(--pr-canvas-ink)' : 'var(--pr-truth)'}
                  >
                    m{n.label}
                  </text>
                ) : (
                  <text x={n.x + 3} y={n.y + 1.2} style={{ fontSize: 3 }} fill="var(--pr-truth)">
                    prior
                  </text>
                )}
              </g>
            ))}
          </svg>

          <Dashboard columns={2} className="mt-3">
            <StatTile label="Gaussians alive" value={sim.state.hyps.length} role="posterior" />
            <StatTile label="branches born" value={sim.state.born} role="prediction" />
            <StatTile
              label="best weight"
              value={best ? best.w : 0}
              precision={3}
              role="posterior"
              sparkline={sim.state.hyps.map((h) => h.w)}
            />
            <StatTile label="pruned away" value={sim.state.pruned.length} role="truth" />
          </Dashboard>
        </div>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Prune ratio"
          role="posterior"
          value={params.pruneRatio}
          min={0.001}
          max={0.6}
          step={0.001}
          format={(x) => x.toFixed(3)}
          onChange={(v) => setParams((s) => ({ ...s, pruneRatio: v }))}
          help="Drop any hypothesis whose weight falls below this fraction of the best. The mass you discard is the error you accept."
        />
        <Slider
          label="Max hypotheses H_max"
          value={params.maxHyps}
          min={1}
          max={16}
          step={1}
          format={(x) => x.toFixed(0)}
          onChange={(v) => setParams((s) => ({ ...s, maxHyps: Math.round(v) }))}
          help="H_max = 1 is exactly the EKF localizer of this chapter."
        />
        <ButtonRow>
          <ActionButton
            emphasis
            onClick={() => sim.setState((s) => ({ ...s, pending: BEACON_BURST }))}
          >
            look at the beacon
          </ActionButton>
        </ButtonRow>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

/* -------------------------------------------------------------------------- */
/* Tree layout                                                                 */
/* -------------------------------------------------------------------------- */

interface TreeNode {
  key: string;
  depth: number;
  label: number;
  x: number;
  y: number;
  w: number;
  alive: boolean;
}

interface TreeEdge {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  w: number;
  alive: boolean;
}

/**
 * A trie over compressed association histories. Live hypotheses contribute
 * their weight up every ancestor; pruned ones are drawn gray so the reader can
 * see what the filter can no longer get back.
 */
function buildTree(hyps: Hypothesis[], pruned: number[][]) {
  const width = 100;
  const byKey = new Map<string, { depth: number; label: number; w: number; alive: boolean }>();
  const leaves: string[] = [];

  const add = (history: number[], w: number, alive: boolean) => {
    const path = compress(history);
    let key = '';
    for (let d = 0; d <= path.length; d++) {
      if (d > 0) key = `${key}/${path[d - 1]}`;
      const node = byKey.get(key);
      if (node) {
        node.w += w;
        node.alive = node.alive || alive;
      } else {
        byKey.set(key, { depth: d, label: d === 0 ? -1 : path[d - 1], w, alive });
      }
      if (d === path.length && !leaves.includes(key)) leaves.push(key);
    }
  };

  for (const h of hyps) add(h.history, h.w, true);
  for (const p of pruned.slice(-10)) add(p, 0.02, false);

  const maxDepth = Math.max(1, ...Array.from(byKey.values()).map((n) => n.depth));
  const height = 12 + maxDepth * 16;
  const shown = leaves.slice(0, 14);
  const xOf = new Map<string, number>();
  shown.forEach((k, i) => xOf.set(k, 8 + ((width - 16) * (i + 0.5)) / shown.length));

  // Internal nodes sit above the mean of their descendants.
  const keys = Array.from(byKey.keys()).sort((a, b) => b.length - a.length);
  for (const k of keys) {
    if (xOf.has(k)) continue;
    const kids = keys.filter((c) => c.startsWith(`${k}/`) && xOf.has(c));
    if (kids.length === 0) continue;
    const xs = kids.map((c) => xOf.get(c) as number);
    xOf.set(k, xs.reduce((a, b) => a + b, 0) / xs.length);
  }

  const wMax = Math.max(...Array.from(byKey.values()).map((n) => n.w), 1e-9);
  const nodes: TreeNode[] = [];
  const edges: TreeEdge[] = [];

  for (const [key, n] of byKey) {
    const x = xOf.get(key);
    if (x === undefined) continue;
    const y = 8 + n.depth * 16;
    nodes.push({ key, depth: n.depth, label: n.label, x, y, w: n.w / wMax, alive: n.alive });
    if (n.depth > 0) {
      const parentKey = key.slice(0, key.lastIndexOf('/'));
      const px = xOf.get(parentKey);
      if (px !== undefined) {
        edges.push({
          key: `e${key}`,
          x1: px,
          y1: 8 + (n.depth - 1) * 16,
          x2: x,
          y2: y,
          w: n.w / wMax,
          alive: n.alive,
        });
      }
    }
  }

  return { nodes, edges, width, height };
}
