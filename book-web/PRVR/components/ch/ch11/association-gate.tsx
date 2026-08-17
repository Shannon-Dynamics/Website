'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { cholesky, ellipse2, type Mat } from '@/lib/prob/linalg';
import { normalizeAngle } from '@/lib/geom/se2';
import type { Landmark } from '@/lib/sim/world';
import {
  featureInnovation,
  gateThreshold,
  mahalanobis2,
  mlAssociate,
  predictAll,
  type LandmarkPrediction,
  type PoseBelief,
} from '@/lib/filters/ekf-localization';
import { clear, drawCovariance, drawRobot, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w11.1 — the Association Gate.
 *
 * Two linked panes. On the left, Rusty's predicted belief and five landmarks in
 * the plane. On the right, the same scene in *measurement* space: range across,
 * bearing up, every landmark's predicted measurement ẑʲ drawn with its
 * validation ellipse — the set of measurements the filter would accept as
 * coming from that landmark.
 *
 * The reader drags the green measurement dot. The winning landmark is decided
 * by the real `mlAssociate` from the library, and the metric toggle swaps the
 * only thing that changes: whether distance is measured in metres or in units
 * of the filter's own uncertainty. There is a large region where those two
 * answers disagree, and every point in it is a bug waiting to happen.
 */

/* -------------------------------------------------------------------------- */
/* The scene                                                                   */
/* -------------------------------------------------------------------------- */

/** Rusty's predicted pose. Uncertainty is anisotropic: 0.8 m along the heading,
 *  0.2 m across it — the shape a corridor always produces. */
const BELIEF: PoseBelief = {
  mu: { x: 2, y: 5, theta: 0 },
  Sigma: [
    [0.64, 0, 0],
    [0, 0.04, 0],
    [0, 0, 0.0025],
  ],
};

const LANDMARKS: Landmark[] = [
  // m0 sits exactly 4 m straight ahead, which makes its gate hand-checkable:
  // S = diag(0.64 + 0.04, 0.04/16 + 0.0025 + 0.0025) = diag(0.68, 0.0075).
  { x: 6.0, y: 5.0, id: 0 },
  // m1 is 0.86 m away from m0 — closer to the robot but off to one side. It is
  // the pair the whole widget is about: Euclidean and Mahalanobis disagree
  // about which of these two explains a reading near (3.55 m, 0.05 rad).
  { x: 5.4, y: 5.62, id: 1 },
  { x: 4.6, y: 7.2, id: 2 },
  { x: 8.6, y: 2.6, id: 3 },
  { x: 3.5, y: 3.4, id: 4 },
];

/** σ_r = 0.2 m, σ_φ = 0.05 rad — a decent laser-based feature detector. */
const Q: Mat = [
  [0.04, 0],
  [0, 0.0025],
];

/* Canvas layout, in canvas-world units. The left pane is the map at 1:1. */
const MAP = { x0: 0, x1: 11, y0: 0, y1: 10 };
const PANE = { x0: 12.6, x1: 23.0, y0: 1.0, y1: 9.2 };
const R_RANGE: [number, number] = [1.2, 7.6];
const PHI_RANGE: [number, number] = [-0.95, 0.95];

const toPaneX = (r: number) =>
  PANE.x0 + ((r - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0])) * (PANE.x1 - PANE.x0);
const toPaneY = (phi: number) =>
  (PANE.y0 + PANE.y1) / 2 +
  (phi / ((PHI_RANGE[1] - PHI_RANGE[0]) / 2)) * ((PANE.y1 - PANE.y0) / 2);
const fromPaneX = (px: number) =>
  R_RANGE[0] + ((px - PANE.x0) / (PANE.x1 - PANE.x0)) * (R_RANGE[1] - R_RANGE[0]);
const fromPaneY = (py: number) =>
  ((py - (PANE.y0 + PANE.y1) / 2) / ((PANE.y1 - PANE.y0) / 2)) *
  ((PHI_RANGE[1] - PHI_RANGE[0]) / 2);

interface Params {
  confidence: number;
  metric: boolean; // true = Mahalanobis, false = Euclidean
  useLogDet: boolean;
}

interface State {
  /** The measurement, in measurement space. */
  z: { r: number; phi: number };
  /** Orbit phase for the autoplay tour around the close pair. */
  phase: number;
  /** Set once the reader drags: the tour stops competing with them. */
  manual: boolean;
}

/** Contour of {ν : νᵀ S⁻¹ ν = γ}, as points in measurement space. */
function gateContour(zHat: [number, number], S: Mat, gamma: number, n = 64): [number, number][] {
  const L = cholesky(S);
  const k = Math.sqrt(gamma);
  const out: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const t = (i / n) * Math.PI * 2;
    const c = Math.cos(t);
    const s = Math.sin(t);
    out.push([zHat[0] + k * (L[0][0] * c), zHat[1] + k * (L[1][0] * c + L[1][1] * s)]);
  }
  return out;
}

export function AssociationGate() {
  const [params, setParams] = useState<Params>({
    confidence: 0.95,
    metric: true,
    useLogDet: true,
  });

  const preds = useMemo(() => predictAll(BELIEF, LANDMARKS, Q), []);

  // The autoplay tour circles the midpoint of the two predictions that overlap.
  const orbit = useMemo(() => {
    const a = preds[0].zHat;
    const b = preds[1].zHat;
    return {
      r: (a[0] + b[0]) / 2,
      phi: (a[1] + b[1]) / 2,
      dr: 0.55,
      dphi: 0.13,
    };
  }, [preds]);

  const init = useCallback(
    (): State => ({
      z: { r: orbit.r + orbit.dr, phi: orbit.phi },
      phase: 0,
      manual: false,
    }),
    [orbit],
  );

  const step = useCallback(
    (s: State): State => {
      if (s.manual) return s;
      const phase = s.phase + 0.06;
      return {
        ...s,
        phase,
        z: {
          r: orbit.r + orbit.dr * Math.cos(phase),
          phi: orbit.phi + orbit.dphi * Math.sin(phase),
        },
      };
    },
    [orbit],
  );

  const sim = useSimulation<State>({ init, step, fps: 24, initialSeed: 11 });

  const gate2 = useMemo(() => gateThreshold(params.confidence, 2), [params.confidence]);

  const scores = useMemo(() => {
    const z = sim.state.z;
    return preds.map((p) => {
      const nu = featureInnovation(z, p.zHat);
      const d2 = mahalanobis2(nu, p.S);
      const dx = z.r * Math.cos(z.phi) - p.zHat[0] * Math.cos(p.zHat[1]);
      const dy = z.r * Math.sin(z.phi) - p.zHat[0] * Math.sin(p.zHat[1]);
      return {
        p,
        d2,
        euclid: Math.hypot(dx, dy),
        score: params.useLogDet ? d2 + p.logDetS : d2,
      };
    });
  }, [preds, sim.state.z, params.useLogDet]);

  const association = useMemo(
    () =>
      mlAssociate(sim.state.z, preds, {
        gate2,
        metric: params.metric ? 'mahalanobis' : 'euclidean',
        useLogDet: params.useLogDet,
      }),
    [sim.state.z, preds, gate2, params.metric, params.useLogDet],
  );

  /** What the *other* metric would have said — the disagreement readout. */
  const rival = useMemo(
    () =>
      mlAssociate(sim.state.z, preds, {
        gate2,
        metric: params.metric ? 'euclidean' : 'mahalanobis',
        useLogDet: params.useLogDet,
      }),
    [sim.state.z, preds, gate2, params.metric, params.useLogDet],
  );

  const winner = association.kind === 'match' ? association.index : -1;
  const rivalWinner = rival.kind === 'match' ? rival.index : -1;
  const disagree = winner !== rivalWinner;

  /**
   * Drag in either pane. In the measurement pane the pointer *is* (r, φ); in the
   * map pane it is a point in metres, which we convert through μ̄ — the same
   * projection the Euclidean rule uses, and a good way to feel that the two
   * panes are one object seen twice.
   */
  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') return;
      let r: number;
      let phi: number;
      if (world[0] < (MAP.x1 + PANE.x0) / 2) {
        r = Math.hypot(world[0] - BELIEF.mu.x, world[1] - BELIEF.mu.y);
        phi = normalizeAngle(
          Math.atan2(world[1] - BELIEF.mu.y, world[0] - BELIEF.mu.x) - BELIEF.mu.theta,
        );
      } else {
        r = fromPaneX(world[0]);
        phi = fromPaneY(world[1]);
      }
      sim.setState((s) => ({
        ...s,
        manual: true,
        z: {
          r: Math.min(Math.max(r, R_RANGE[0]), R_RANGE[1]),
          phi: Math.min(Math.max(phi, PHI_RANGE[0]), PHI_RANGE[1]),
        },
      }));
    },
    [sim],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const z = sim.state.z;

      /* ---------------- left pane: the map ---------------- */
      ctx.save();
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(sx(v, MAP.x0), sy(v, MAP.y1), sl(v, MAP.x1 - MAP.x0), sl(v, MAP.y1 - MAP.y0));
      ctx.restore();
      label(ctx, 'map  (metres)', sx(v, MAP.x0 + 0.25), sy(v, MAP.y1 - 0.45), p.truth, { size: 10 });

      // Rusty's predicted belief: the 2σ position ellipse, elongated along the
      // heading. Everything in this widget follows from its shape.
      drawCovariance(
        ctx,
        v,
        [BELIEF.mu.x, BELIEF.mu.y],
        ellipse2(
          [
            [BELIEF.Sigma[0][0], BELIEF.Sigma[0][1]],
            [BELIEF.Sigma[1][0], BELIEF.Sigma[1][1]],
          ],
          2,
        ),
        p.prediction,
        { alpha: 0.9 },
      );
      drawRobot(ctx, v, BELIEF.mu, p.prediction, 0.42);

      // Landmarks, and the ray each predicted measurement corresponds to.
      for (const pred of preds) {
        const lm = pred.landmark;
        const isWinner = pred.index === winner;
        ctx.save();
        ctx.strokeStyle = isWinner ? p.posterior : p.accent;
        ctx.fillStyle = isWinner ? p.posterior : p.accent;
        ctx.lineWidth = isWinner ? 2.4 : 1.4;
        ctx.beginPath();
        ctx.arc(sx(v, lm.x), sy(v, lm.y), isWinner ? 6.5 : 4.5, 0, Math.PI * 2);
        ctx.fill();
        if (isWinner) {
          ctx.beginPath();
          ctx.arc(sx(v, lm.x), sy(v, lm.y), 11, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
        label(ctx, `m${pred.index}`, sx(v, lm.x) + 9, sy(v, lm.y) - 9, p.truth, { size: 9 });
      }

      // Where the measurement says the feature is, if the robot were at μ̄.
      const zx = BELIEF.mu.x + z.r * Math.cos(BELIEF.mu.theta + z.phi);
      const zy = BELIEF.mu.y + z.r * Math.sin(BELIEF.mu.theta + z.phi);
      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, BELIEF.mu.x), sy(v, BELIEF.mu.y));
      ctx.lineTo(sx(v, zx), sy(v, zy));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = p.measurement;
      ctx.beginPath();
      ctx.arc(sx(v, zx), sy(v, zy), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      label(ctx, 'z', sx(v, zx) + 8, sy(v, zy) + 9, p.measurement, { size: 10, weight: 600 });

      /* ---------------- right pane: measurement space ---------------- */
      ctx.save();
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.strokeRect(
        sx(v, PANE.x0),
        sy(v, PANE.y1),
        sl(v, PANE.x1 - PANE.x0),
        sl(v, PANE.y1 - PANE.y0),
      );
      // Zero-bearing line: straight ahead.
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(v, PANE.x0), sy(v, toPaneY(0)));
      ctx.lineTo(sx(v, PANE.x1), sy(v, toPaneY(0)));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      label(
        ctx,
        'measurement space   range →   bearing ↑',
        sx(v, PANE.x0 + 0.2),
        sy(v, PANE.y1 + 0.35),
        p.truth,
        { size: 10 },
      );

      const drawContour = (pred: LandmarkPrediction, color: string, lw: number, fill: boolean) => {
        const pts = gateContour(pred.zHat, pred.S, gate2);
        ctx.beginPath();
        pts.forEach(([r, phi], i) => {
          const X = sx(v, toPaneX(r));
          const Y = sy(v, toPaneY(phi));
          if (i === 0) ctx.moveTo(X, Y);
          else ctx.lineTo(X, Y);
        });
        ctx.closePath();
        if (fill) {
          ctx.save();
          ctx.globalAlpha = 0.14;
          ctx.fillStyle = color;
          ctx.fill();
          ctx.restore();
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.stroke();
      };

      for (const pred of preds) {
        const isWinner = pred.index === winner;
        ctx.save();
        ctx.setLineDash(isWinner ? [] : [5, 3]);
        drawContour(pred, isWinner ? p.posterior : p.prediction, isWinner ? 2.2 : 1.3, isWinner);
        ctx.setLineDash([]);
        ctx.fillStyle = isWinner ? p.posterior : p.prediction;
        ctx.beginPath();
        ctx.arc(sx(v, toPaneX(pred.zHat[0])), sy(v, toPaneY(pred.zHat[1])), 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        label(
          ctx,
          `ẑ${pred.index}`,
          sx(v, toPaneX(pred.zHat[0])) + 7,
          sy(v, toPaneY(pred.zHat[1])) - 8,
          p.truth,
          { size: 9 },
        );
      }

      // The measurement itself.
      const ZX = sx(v, toPaneX(z.r));
      const ZY = sy(v, toPaneY(z.phi));
      ctx.save();
      ctx.fillStyle = p.measurement;
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(ZX, ZY, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.arc(ZX, ZY, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // The innovation to the winner, drawn as the vector the filter will act on.
      if (winner >= 0) {
        const w = preds[winner];
        ctx.save();
        ctx.strokeStyle = p.posterior;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(sx(v, toPaneX(w.zHat[0])), sy(v, toPaneY(w.zHat[1])));
        ctx.lineTo(ZX, ZY);
        ctx.stroke();
        ctx.restore();
      }

      const verdict =
        association.kind === 'match'
          ? `accepted → m${association.index}   d² = ${association.d2.toFixed(2)}`
          : `OUTLIER   best d² = ${association.d2.toFixed(2)} > γ = ${gate2.toFixed(2)}`;
      label(
        ctx,
        verdict,
        sx(v, PANE.x0 + 0.2),
        sy(v, PANE.y0 - 0.45),
        association.kind === 'match' ? p.posterior : p.prediction,
        { size: 11, weight: 600 },
      );
      if (disagree) {
        label(
          ctx,
          params.metric
            ? `Euclidean would pick ${rivalWinner >= 0 ? `m${rivalWinner}` : 'nothing'}`
            : `Mahalanobis would pick ${rivalWinner >= 0 ? `m${rivalWinner}` : 'nothing'}`,
          sx(v, MAP.x0 + 0.25),
          sy(v, MAP.y0 + 0.45),
          p.measurement,
          { size: 10, weight: 600 },
        );
      }
    },
    [sim.state.z, preds, winner, gate2, association, disagree, rivalWinner, params.metric],
  );

  return (
    <WidgetFrame
      id="w11.1"
      title="The Association Gate"
      teaches="The nearest landmark in metres is not the nearest landmark in the metric that matters. Distance is measured in units of the filter's own uncertainty."
      colorKey={['prediction', 'measurement', 'posterior']}
      caption={
        <>
          Left: Rusty&rsquo;s predicted belief (orange, elongated along the heading, as a corridor
          always makes it) and five landmarks. Right: the same scene in measurement space, where
          each landmark&rsquo;s predicted measurement <em>ẑ</em> carries a validation ellipse — the
          set of readings the filter would accept as coming from it. The tour circles the green
          measurement around the deliberately close pair m0/m1; drag it anywhere to take over.
          <strong> What to notice:</strong> the gates are long in range and thin in bearing, because
          a 0.8 m position uncertainty <em>along</em> the line of sight is 4× cheaper than the same
          error across it. <strong>What to try:</strong> switch the metric to Euclidean and drag
          along the boundary between m0 and m1 — there is a whole band where the two rules pick
          different landmarks, and a filter that picks wrong there is about to be poisoned.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.3, maxX: 23.3, minY: -0.3, maxY: 10.3 }}
        draw={draw}
        deps={[sim.tick, sim.state, params, winner]}
        aspect={2.34}
        padding={0}
        onPointer={onPointer}
        cursor="crosshair"
        ariaLabel="Two linked panes. On the left, a robot with an elongated uncertainty ellipse and five landmarks. On the right, the same landmarks in range-bearing measurement space, each with an elliptical validation gate, and a draggable measurement whose winning association is highlighted."
      />

      <div className="overflow-x-auto border-t border-fd-border">
        <table className="w-full min-w-[26rem] border-collapse font-mono text-[0.7rem] tabular-nums">
          <thead>
            <tr className="border-b border-fd-border text-fd-muted-foreground">
              <th className="px-3 py-1 text-start font-medium">landmark</th>
              <th className="px-3 py-1 text-end font-medium">d²ₘ</th>
              <th className="px-3 py-1 text-end font-medium">log det S</th>
              <th className="px-3 py-1 text-end font-medium">ML score</th>
              <th className="px-3 py-1 text-end font-medium">metres</th>
              <th className="px-3 py-1 text-end font-medium">gate</th>
            </tr>
          </thead>
          <tbody>
            {scores.map((s) => {
              const isWinner = s.p.index === winner;
              return (
                <tr
                  key={s.p.index}
                  className="border-b border-fd-border/50 last:border-b-0"
                  style={isWinner ? { color: 'var(--pr-posterior)', fontWeight: 600 } : undefined}
                >
                  <td className="px-3 py-1">m{s.p.index}</td>
                  <td className="px-3 py-1 text-end">{s.d2.toFixed(2)}</td>
                  <td className="px-3 py-1 text-end">{s.p.logDetS.toFixed(2)}</td>
                  <td className="px-3 py-1 text-end">{s.score.toFixed(2)}</td>
                  <td className="px-3 py-1 text-end">{s.euclid.toFixed(2)}</td>
                  <td className="px-3 py-1 text-end">{s.d2 <= gate2 ? 'in' : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Gate confidence 1 − ε"
          role="prediction"
          value={params.confidence}
          min={0.5}
          max={0.995}
          step={0.005}
          format={(x) => `${(x * 100).toFixed(1)}%  γ=${gateThreshold(x, 2).toFixed(2)}`}
          onChange={(v) => setParams((s) => ({ ...s, confidence: v }))}
          help="χ² quantile with 2 degrees of freedom. Widening the gate admits more true matches — and more wrong ones."
        />
        <Toggle
          label={params.metric ? 'Metric: Mahalanobis' : 'Metric: Euclidean'}
          role="posterior"
          checked={params.metric}
          onChange={(v) => setParams((s) => ({ ...s, metric: v }))}
        />
        <Toggle
          label="Keep the log det S term"
          role="prediction"
          checked={params.useLogDet}
          onChange={(v) => setParams((s) => ({ ...s, useLogDet: v }))}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
