'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { Dashboard, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { se2Exp, type Pose2, type Twist2 } from '@/lib/geom/se2';
import { icrWorld, naiveInterp, screwParams, screwPath } from '@/lib/geom/screw';
import {
  clear,
  drawGrid,
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
 * w3.2 — the Exp/Log Lens.
 *
 * Left: the tangent space, a flat card where τ is an arrow and scaling it is
 * just multiplication. Right: the plane, where the same scaling traces an arc
 * around the instantaneous center of rotation. `exp` is the only thing between
 * them, and the orange ghost — the pose you get by interpolating (x, y, θ)
 * componentwise — shows what it costs to skip it.
 */

const ORIGIN: Pose2 = { x: 0, y: 0, theta: 0 };
const SAMPLES = 64;

interface State {
  /** Arc parameter, swept 0 → 1 and looped. */
  s: number;
}

export function ExpLogLens() {
  const [omega, setOmega] = useState(Math.PI / 2);
  const [v, setV] = useState<[number, number]>([1, 0]);
  const [showNaive, setShowNaive] = useState(true);

  const tau = useMemo<Twist2>(() => [v[0], v[1], omega], [v, omega]);
  const target = useMemo(() => se2Exp(tau), [tau]);
  const params = useMemo(() => screwParams(tau), [tau]);
  const center = useMemo(() => icrWorld(ORIGIN, tau), [tau]);

  const init = useCallback((): State => ({ s: 0 }), []);
  const step = useCallback((st: State): State => ({ s: (st.s + 1 / 60) % 1 }), []);
  const sim = useSimulation<State>({ init, step, fps: 30, initialSeed: 1 });
  const s = sim.state.s;

  const geodesicPath = useMemo(() => screwPath(ORIGIN, tau, SAMPLES), [tau]);
  const naivePath = useMemo(
    () => Array.from({ length: SAMPLES + 1 }, (_, i) => naiveInterp(ORIGIN, target, i / SAMPLES)),
    [target],
  );

  /** How far the componentwise lerp strays from the geodesic, over the whole sweep. */
  const errorProfile = useMemo(
    () =>
      Array.from({ length: 33 }, (_, i) => {
        const u = i / 32;
        const g = se2Exp([tau[0] * u, tau[1] * u, tau[2] * u]);
        const n = naiveInterp(ORIGIN, target, u);
        return Math.hypot(g.x - n.x, g.y - n.y);
      }),
    [tau, target],
  );

  const here = se2Exp([tau[0] * s, tau[1] * s, tau[2] * s]);
  const naiveHere = naiveInterp(ORIGIN, target, s);
  const gap = Math.hypot(here.x - naiveHere.x, here.y - naiveHere.y);

  /* ---------------------------------------------------------------- plane */
  const drawPlane = useCallback(
    (ctx: CanvasRenderingContext2D, vp: Viewport, p: Palette) => {
      clear(ctx, vp, p);
      drawGrid(ctx, vp, p, 0.5);

      if (center && Math.abs(params.radius ?? 0) < 8) {
        // The screw axis, seen end-on: every point of the body circles this one.
        const cx = sx(vp, center[0]);
        const cy = sy(vp, center[1]);
        ctx.strokeStyle = p.truth;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, sl(vp, Math.abs(params.radius ?? 0)), 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(sx(vp, here.x), sy(vp, here.y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = p.truth;
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, 'ICR', cx + 8, cy - 8, p.truth, { size: 10, weight: 600 });
      }

      drawPath(ctx, vp, geodesicPath, p.posterior, { lineWidth: 2.4 });
      if (showNaive) {
        drawPath(ctx, vp, naivePath, p.prediction, { dashed: true, lineWidth: 2 });
        drawRobot(ctx, vp, naiveHere, p.prediction, 0.18, { filled: false });
      }

      drawRobot(ctx, vp, ORIGIN, p.prior, 0.18, { filled: false, alpha: 0.7 });
      drawRobot(ctx, vp, target, p.truth, 0.18, { filled: false, alpha: 0.55 });
      drawRobot(ctx, vp, here, p.posterior, 0.2);

      label(ctx, 'x₀', sx(vp, 0) - 6, sy(vp, 0) + 18, p.prior, { size: 10, weight: 600 });
      label(ctx, 'x₀ ⊞ τ', sx(vp, target.x) + 10, sy(vp, target.y) - 10, p.truth, {
        size: 10,
        weight: 600,
      });
      label(ctx, `s = ${s.toFixed(2)}`, 10, 14, p.posterior, { size: 11, weight: 600 });
    },
    [geodesicPath, naivePath, here, naiveHere, target, center, params.radius, showNaive, s],
  );

  /* -------------------------------------------------------------- tangent */
  const drawTangent = useCallback(
    (ctx: CanvasRenderingContext2D, vp: Viewport, p: Palette) => {
      clear(ctx, vp, p);
      drawGrid(ctx, vp, p, 0.5);

      // Axes of the flat card.
      ctx.strokeStyle = p.truth;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(vp, -1.6), sy(vp, 0));
      ctx.lineTo(sx(vp, 1.6), sy(vp, 0));
      ctx.moveTo(sx(vp, 0), sy(vp, -1.35));
      ctx.lineTo(sx(vp, 0), sy(vp, 1.35));
      ctx.stroke();
      ctx.globalAlpha = 1;
      label(ctx, 'vₓ', sx(vp, 1.45), sy(vp, 0) + 13, p.truth, { size: 10 });
      label(ctx, 'v_y', sx(vp, 0) + 10, sy(vp, 1.28), p.truth, { size: 10 });

      // τ as an arrow; the swept point s·τ slides along it in a straight line.
      const ax = sx(vp, tau[0]);
      const ay = sy(vp, tau[1]);
      ctx.strokeStyle = p.prior;
      ctx.fillStyle = p.prior;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(sx(vp, 0), sy(vp, 0));
      ctx.lineTo(ax, ay);
      ctx.stroke();
      const ang = Math.atan2(tau[1], tau[0]);
      ctx.save();
      ctx.translate(ax, ay);
      ctx.rotate(-ang);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-8, 4);
      ctx.lineTo(-8, -4);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      label(ctx, 'τ', ax + 12, ay - 10, p.prior, { size: 12, weight: 700 });

      ctx.fillStyle = p.posterior;
      ctx.beginPath();
      ctx.arc(sx(vp, tau[0] * s), sy(vp, tau[1] * s), 5, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, 's·τ', sx(vp, tau[0] * s) + 9, sy(vp, tau[1] * s) + 11, p.posterior, { size: 10 });

      // ω is the third tangent coordinate; drawn as a bar, not an angle, to keep
      // the point that the card is flat in all three directions.
      const barY = sy(vp, -1.16);
      const half = sl(vp, 1.3);
      const mid = sx(vp, 0);
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(mid - half, barY);
      ctx.lineTo(mid + half, barY);
      ctx.stroke();
      ctx.strokeStyle = p.prior;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(mid, barY);
      ctx.lineTo(mid + (half * omega) / Math.PI, barY);
      ctx.stroke();
      label(ctx, `ω = ${omega.toFixed(2)}`, mid, barY - 12, p.prior, { size: 10, align: 'center' });
      label(ctx, 'tangent space  𝔰𝔢(2)', 10, 14, p.truth, { size: 10, weight: 600 });
    },
    [tau, s, omega],
  );

  const onTangentPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') return;
      const clamp = (x: number) => Math.max(-1.5, Math.min(1.5, x));
      setV([clamp(world[0]), clamp(world[1])]);
    },
    [],
  );

  return (
    <WidgetFrame
      id="w3.2"
      title="Exp/Log Lens"
      teaches="exp turns a straight line in the tangent space into a screw motion in the world — and componentwise interpolation of (x, y, θ) is not that motion."
      colorKey={['prior', 'prediction', 'posterior', 'truth']}
      wide
      caption={
        <>
          The dot slides along τ at constant speed on the left; on the right the robot follows{' '}
          <span className="font-mono">exp(s·τ)</span>, a perfect arc about the instantaneous center
          of rotation. Drag the arrowhead on the left card to change the body-frame velocity, and
          use the ω slider to change how hard Rusty turns. The orange ghost is the pose you get by
          interpolating the tuple (x, y, θ) componentwise between the same endpoints: it starts and
          ends in the right place and is wrong everywhere in between — it slides sideways, which no
          wheeled robot can do. Push ω toward ±π and watch the gap grow; drop it to zero and the
          arc straightens, the ICR runs off to infinity, and the two paths coincide.
        </>
      }
    >
      <div className="grid grid-cols-1 gap-0 md:grid-cols-[0.85fr_1.15fr] md:divide-x md:divide-fd-border">
        <SimCanvas
          world={{ minX: -1.6, maxX: 1.6, minY: -1.35, maxY: 1.35 }}
          draw={drawTangent}
          deps={[tau, s, omega]}
          aspect={1.15}
          padding={0.05}
          ariaLabel="A flat card representing the tangent space, with the twist tau drawn as an arrow from the origin and a dot sliding along it in a straight line."
          onPointer={onTangentPointer}
          cursor="crosshair"
        />
        <SimCanvas
          world={{ minX: -1.2, maxX: 2.0, minY: -0.9, maxY: 1.9 }}
          draw={drawPlane}
          deps={[here, naiveHere, target, showNaive, s]}
          aspect={1.35}
          padding={0.05}
          ariaLabel="The plane, showing a robot travelling along a circular arc from the origin to the endpoint of the exponential map, with the instantaneous center of rotation marked and a dashed straight-line path showing componentwise interpolation."
        />
      </div>

      <div className="border-t border-fd-border px-3 py-3">
        <Dashboard columns={3}>
          <StatTile
            label="turn radius |v|/ω"
            value={params.radius === null ? '∞' : Math.abs(params.radius)}
            unit="m"
            role="truth"
            precision={3}
          />
          <StatTile label="arc length |v|" value={params.arcLength} unit="m" role="posterior" precision={3} />
          <StatTile
            label="geodesic − lerp"
            value={gap}
            unit="m"
            role="prediction"
            precision={3}
            sparkline={errorProfile}
          />
        </Dashboard>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Angular velocity ω"
          role="prior"
          value={omega}
          min={-3.1}
          max={3.1}
          step={0.02}
          unit="rad"
          onChange={setOmega}
          help="The third tangent coordinate. Zero is a straight line; ±π is a half turn in one unit of time."
        />
        <Toggle
          label="Show componentwise lerp"
          role="prediction"
          checked={showNaive}
          onChange={setShowNaive}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
