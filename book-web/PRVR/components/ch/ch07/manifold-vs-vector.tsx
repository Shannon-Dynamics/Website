'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { angleDiff, normalizeAngle } from '@/lib/geom/se2';
import { normalPdf } from '@/lib/prob/gaussian';
import { Rng } from '@/lib/prob/rng';
import { clear, drawRobot, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w7.2 — Manifold vs. Vector.
 *
 * Rusty spins in place with a compass. Two filters run on the *identical*
 * stream of measurements and differ in exactly one character: one computes the
 * innovation as `z − μ`, the other as `z ⊟ μ`. Every time the true heading
 * crosses the ±π seam the first one yanks the estimate the long way round the
 * circle; the second does not notice the seam exists, because for it there
 * isn't one.
 *
 * The point is not that `z − μ` needs a `mod 2π` patch. The point is that the
 * heading was never a real number, and the filter that treats it as one is
 * computing a weighted average of two points in a space they do not live in.
 */

const DT = 0.25;
const HIST = 150;

interface Params {
  /** Compass noise σ, in radians. The headline control. */
  sigmaZ: number;
  /** Commanded turn rate, rad/s. */
  omega: number;
}

interface Frame {
  rng: Rng;
  /** True heading, always in (−π, π]. */
  truth: number;
  /** Vector-mode prior mean — deliberately *not* wrapped. */
  priorV: number;
  /** Manifold-mode prior mean, on the circle. */
  priorM: number;
  /** Shared prior variance (both filters run the same gain). */
  P: number;
  /** The compass reading, as a real compass reports it: wrapped to (−π, π]. */
  z: number;
  hist: { ev: number; em: number }[];
}

/** Where the two filters disagree, in one function. */
function fuse(f: Frame, z: number, R: number) {
  const K = f.P / (f.P + R);
  return {
    K,
    // Vector update: plain subtraction. 179° and −179° are 358° apart.
    postV: f.priorV + K * (z - f.priorV),
    // Manifold update: ⊟ then ⊞. 179° and −179° are 2° apart.
    postM: normalizeAngle(f.priorM + K * angleDiff(z, f.priorM)),
  };
}

export function ManifoldVsVector() {
  const [params, setParams] = useState<Params>({ sigmaZ: 0.14, omega: 0.9 });
  /** Set by dragging inside the circle; freezes the loop so the reader can aim. */
  const [manualZ, setManualZ] = useState<number | null>(null);

  const R = params.sigmaZ * params.sigmaZ;
  // Process noise grows with how hard the robot is turning: a fast spin is
  // exactly when you cross the seam and exactly when the gain is large.
  const q = (0.25 * Math.abs(params.omega) * DT + 0.02) ** 2;

  const init = useCallback((seed: number): Frame => {
    const rng = new Rng(seed);
    const truth = 2.6;
    return {
      rng,
      truth,
      priorV: truth - 0.2,
      priorM: truth - 0.2,
      P: 0.05,
      z: normalizeAngle(truth + rng.normal(0, 0.14)),
      hist: [],
    };
  }, []);

  const step = useCallback(
    (s: Frame): Frame => {
      const zEff = manualZ ?? s.z;
      const { K, postV, postM } = fuse(s, zEff, R);
      const Pp = (1 - K) * s.P;

      const truth = normalizeAngle(
        s.truth + params.omega * DT + s.rng.normal(0, 0.25 * Math.abs(params.omega) * DT),
      );

      return {
        rng: s.rng,
        truth,
        priorV: postV + params.omega * DT,
        priorM: normalizeAngle(postM + params.omega * DT),
        P: Pp + q,
        z: normalizeAngle(truth + s.rng.normal(0, params.sigmaZ)),
        hist: [
          ...s.hist,
          {
            ev: Math.abs(angleDiff(postV, s.truth)),
            em: Math.abs(angleDiff(postM, s.truth)),
          },
        ].slice(-HIST),
      };
    },
    [params, R, q, manualZ],
  );

  const sim = useSimulation<Frame>({ init, step, fps: 6, initialSeed: 12 });

  const z = manualZ ?? sim.state.z;
  const { K, postV, postM } = useMemo(() => fuse(sim.state, z, R), [sim.state, z, R]);

  // Geometry of the two panels, in world units.
  const CX = 2.15;
  const CY = 2.15;
  const RAD = 1.5;
  const LX0 = 4.5;
  const LX1 = 9.7;
  const LY = 3.05;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const s = sim.state;
      const sd = Math.sqrt(s.P);

      /* ---- left panel: the manifold itself ---------------------------- */
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx(v, CX), sy(v, CY), sl(v, RAD), 0, Math.PI * 2);
      ctx.stroke();

      // The seam: where a programmer's `mod 2π` puts a discontinuity, and where
      // the circle has nothing at all.
      ctx.strokeStyle = p.prediction;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(v, CX - RAD * 1.22), sy(v, CY));
      ctx.lineTo(sx(v, CX - RAD * 0.86), sy(v, CY));
      ctx.stroke();
      ctx.setLineDash([]);
      label(ctx, '±π seam', sx(v, CX - RAD * 1.24), sy(v, CY), p.prediction, {
        size: 9,
        align: 'right',
      });

      const arrow = (angle: number, color: string, len: number, width: number, head = 6) => {
        const ex = CX + len * Math.cos(angle);
        const ey = CY + len * Math.sin(angle);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(sx(v, CX), sy(v, CY));
        ctx.lineTo(sx(v, ex), sy(v, ey));
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx(v, ex), sy(v, ey));
        ctx.lineTo(
          sx(v, ex) - head * Math.cos(angle - 0.4),
          sy(v, ey) + head * Math.sin(angle - 0.4),
        );
        ctx.lineTo(
          sx(v, ex) - head * Math.cos(angle + 0.4),
          sy(v, ey) + head * Math.sin(angle + 0.4),
        );
        ctx.closePath();
        ctx.fill();
      };

      const wedge = (center: number, half: number, color: string, rIn: number, rOut: number) => {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.18;
        ctx.beginPath();
        ctx.arc(sx(v, CX), sy(v, CY), sl(v, rOut), -(center + half), -(center - half));
        ctx.arc(sx(v, CX), sy(v, CY), sl(v, rIn), -(center - half), -(center + half), true);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
      };

      wedge(normalizeAngle(s.priorM), 2 * sd, p.prior, RAD * 0.9, RAD * 1.05);
      wedge(z, 2 * params.sigmaZ, p.measurement, RAD * 1.06, RAD * 1.2);

      arrow(normalizeAngle(s.priorM), p.prior, RAD * 0.95, 2);
      arrow(z, p.measurement, RAD * 1.12, 2);
      arrow(normalizeAngle(postV), p.prediction, RAD * 0.72, 3, 8);
      arrow(normalizeAngle(postM), p.posterior, RAD * 0.55, 3, 8);

      drawRobot(ctx, v, { x: CX, y: CY, theta: s.truth }, p.truth, 0.3);

      label(ctx, 'prior', sx(v, CX) + 4, sy(v, CY + RAD) - 44, p.prior, { size: 9 });
      label(ctx, 'compass — drag me', sx(v, CX), sy(v, CY - RAD) + 34, p.measurement, {
        size: 9,
        align: 'center',
      });

      /* ---- right panel top: the same beliefs, on a line ---------------- */
      const toLine = (a: number) => LX0 + ((a + Math.PI) / (2 * Math.PI)) * (LX1 - LX0);
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, LX0), sy(v, LY));
      ctx.lineTo(sx(v, LX1), sy(v, LY));
      ctx.stroke();

      // The two ends of the line are the *same point* of the circle. Mark them.
      for (const end of [LX0, LX1]) {
        ctx.strokeStyle = p.prediction;
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, end), sy(v, LY - 0.1));
        ctx.lineTo(sx(v, end), sy(v, LY + 0.85));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      label(ctx, '−π', sx(v, LX0), sy(v, LY) + 12, p.prediction, { size: 9, align: 'center' });
      label(ctx, '+π', sx(v, LX1), sy(v, LY) + 12, p.prediction, { size: 9, align: 'center' });
      label(ctx, 'the same heading, drawn twice', sx(v, (LX0 + LX1) / 2), sy(v, LY) + 12, p.ink, {
        size: 9,
        align: 'center',
      });

      const bell = (mean: number, sigma: number, color: string) => {
        const peak = normalPdf(mean, mean, sigma);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i <= 220; i++) {
          const a = -Math.PI + (2 * Math.PI * i) / 220;
          const h = (normalPdf(a, mean, sigma) / peak) * 0.72;
          const px = sx(v, toLine(a));
          const py = sy(v, LY + h);
          if (!started) {
            ctx.moveTo(px, py);
            started = true;
          } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      };
      bell(normalizeAngle(s.priorM), sd, p.prior);
      bell(z, params.sigmaZ, p.measurement);

      const tick = (a: number, color: string, h: number, width = 2.5) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(sx(v, toLine(a)), sy(v, LY));
        ctx.lineTo(sx(v, toLine(a)), sy(v, LY + h));
        ctx.stroke();
      };
      tick(s.truth, p.truth, 0.85, 1.5);
      tick(normalizeAngle(postV), p.prediction, 0.62);
      tick(normalizeAngle(postM), p.posterior, 0.48);

      /* ---- right panel bottom: heading error over time ------------------ */
      const EY0 = 0.42;
      const EY1 = 2.12;
      const maxErr = Math.PI;
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, LX0), sy(v, EY0));
      ctx.lineTo(sx(v, LX1), sy(v, EY0));
      ctx.stroke();
      label(ctx, 'heading error', sx(v, LX0), sy(v, EY1) + 6, p.ink, { size: 9 });
      label(ctx, '180°', sx(v, LX0) - 4, sy(v, EY1), p.ink, { size: 8, align: 'right' });
      label(ctx, '0°', sx(v, LX0) - 4, sy(v, EY0), p.ink, { size: 8, align: 'right' });

      const series = (key: 'ev' | 'em', color: string, width: number) => {
        if (s.hist.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        s.hist.forEach((h, i) => {
          const px = sx(v, LX0 + ((LX1 - LX0) * i) / (HIST - 1));
          const py = sy(v, EY0 + (Math.min(h[key], maxErr) / maxErr) * (EY1 - EY0));
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      };
      series('ev', p.prediction, 2);
      series('em', p.posterior, 2);

      /* ---- the readout that gives the game away ------------------------ */
      const deg = (a: number) => `${((a * 180) / Math.PI).toFixed(0)}°`;
      label(ctx, `vector mean ${deg(postV)}`, sx(v, LX0), sy(v, EY1) + 20, p.prediction, {
        size: 10,
        weight: 600,
      });
      label(ctx, `manifold mean ${deg(postM)}`, sx(v, LX0 + 2.3), sy(v, EY1) + 20, p.posterior, {
        size: 10,
        weight: 600,
      });
      label(ctx, `truth ${deg(s.truth)}`, sx(v, LX0 + 4.5), sy(v, EY1) + 20, p.truth, {
        size: 10,
        weight: 600,
      });
    },
    [sim.state, z, postV, postM, params.sigmaZ],
  );

  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      if (phase === 'up') return;
      const dx = world[0] - CX;
      const dy = world[1] - CY;
      if (Math.hypot(dx, dy) > RAD * 1.45) return;
      sim.pause();
      setManualZ(Math.atan2(dy, dx));
    },
    [sim],
  );

  const rms = useMemo(() => {
    const h = sim.state.hist;
    if (h.length < 5) return { v: 0, m: 0 };
    const r = (key: 'ev' | 'em') =>
      Math.sqrt(h.reduce((a, x) => a + x[key] * x[key], 0) / h.length) * (180 / Math.PI);
    return { v: r('ev'), m: r('em') };
  }, [sim.state.hist]);

  return (
    <WidgetFrame
      id="w7.2"
      title="Manifold vs. Vector"
      teaches="Wrap-around is not an edge case to patch with mod 2π. It is a type error: the heading was never a real number."
      colorKey={['prior', 'measurement', 'prediction', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty spins, and a compass reports his heading wrapped to (−π,&nbsp;π] — as every real
          compass does. Two filters consume the identical readings. The{' '}
          <strong style={{ color: 'var(--pr-prediction)' }}>orange</strong> one computes its
          innovation as <code>z − μ</code>; the{' '}
          <strong style={{ color: 'var(--pr-posterior)' }}>purple</strong> one computes it as{' '}
          <code>z ⊟ μ</code>. Watch the moment the gray robot crosses the seam: the orange arrow
          swings the long way around the circle and ends up pointing at nothing in particular, while
          purple does not flinch. On the line at right you can see why — the two ends of that
          segment are the <em>same</em> heading, and the vector filter is averaging across the whole
          width of the picture. Drag the green compass arrow to aim a measurement by hand: the worst
          case is a gain near 0.5, where the vector update lands halfway around the world.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: 10, minY: 0, maxY: 4.3 }}
        draw={draw}
        deps={[sim.tick, sim.state, z, postV, postM]}
        aspect={2.32}
        padding={0}
        cursor="crosshair"
        onPointer={onPointer}
        ariaLabel="Left: a robot at the center of a circle with arrows for its prior heading, a compass measurement, and two fused estimates. Right: the same beliefs drawn on a line from minus pi to pi, and a chart of heading error over time in which the vector-mode filter spikes at every seam crossing."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="Kalman gain K" value={K.toFixed(3)} />
        <Stat label="RMS error — vector" value={`${rms.v.toFixed(1)}°`} role="prediction" />
        <Stat label="RMS error — manifold" value={`${rms.m.toFixed(1)}°`} role="posterior" />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Compass σ"
          role="measurement"
          value={params.sigmaZ}
          min={0.04}
          max={0.6}
          step={0.01}
          unit="rad"
          onChange={(v) => setParams((p) => ({ ...p, sigmaZ: v }))}
          help="The headline control. It sets the gain — and the damage a seam crossing does is largest when K is near ½."
        />
        <Slider
          label="Turn rate ω"
          role="prediction"
          value={params.omega}
          min={0.2}
          max={2}
          step={0.05}
          unit="rad/s"
          onChange={(v) => setParams((p) => ({ ...p, omega: v }))}
          help="How often Rusty crosses the seam."
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={() => {
          setManualZ(null);
          sim.toggle();
        }}
        onStep={sim.stepOnce}
        onReset={() => {
          setManualZ(null);
          sim.reset();
        }}
        onReseed={() => {
          setManualZ(null);
          sim.reseed();
        }}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function Stat({
  label: l,
  value,
  role,
}: {
  label: string;
  value: string;
  role?: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow" style={role ? { color: `var(--pr-${role})` } : undefined}>
        {l}
      </div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
