'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { mahalanobis, sampleMvn } from '@/lib/prob/gaussian';
import { diag, ellipse2, type Mat } from '@/lib/prob/linalg';
import { boxplus, se2Exp, type Pose2, type Twist2 } from '@/lib/geom/se2';
import { screwPath } from '@/lib/geom/screw';
import {
  clear,
  drawCovariance,
  drawGrid,
  drawPath,
  drawRobot,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w3.4 — the Banana Preview.
 *
 * A Gaussian in the tangent space is a perfectly ordinary ellipsoid. Push it
 * through `exp` and the cloud in (x, y) is a banana: bent, skewed, and with its
 * mean off the commanded endpoint. The overlay is the *best possible* ellipse —
 * the moment-matched Gaussian fitted to these very samples — and it still gets
 * the shape wrong, which is the honest statement of what a Gaussian filter is
 * betting when it tracks a pose.
 *
 * Chapter 9 derives this distribution; Chapter 7 teaches filters to respect it.
 */

const N = 700;
const START: Pose2 = { x: 0, y: 0, theta: 0 };
/** The commanded arc: drive 2 m of arc while turning about 69°. */
const TAU0: Twist2 = [2.0, 0, 1.2];
/** Reveal 40 samples per tick, so the banana grows rather than appearing. */
const PER_TICK = 40;
const TICKS = Math.ceil(N / PER_TICK);

interface State {
  revealed: number;
}

export function BananaPreview() {
  const [sigmaOmega, setSigmaOmega] = useState(0.32);
  const [sigmaV, setSigmaV] = useState(0.12);
  const [showEllipse, setShowEllipse] = useState(true);

  const cov = useMemo<Mat>(
    () => diag([sigmaV * sigmaV, sigmaV * sigmaV, sigmaOmega * sigmaOmega]),
    [sigmaV, sigmaOmega],
  );

  // The simulation clock only reveals samples; the samples themselves are a
  // pure function of (seed, Σ), so dragging a slider updates the cloud live
  // without disturbing the animation.
  const init = useCallback((): State => ({ revealed: 0 }), []);
  const step = useCallback(
    (s: State): State => ({ revealed: Math.min(N, s.revealed + PER_TICK) }),
    [],
  );

  const sim = useSimulation<State>({
    init,
    step,
    fps: 14,
    maxTicks: TICKS,
    loop: false,
    initialSeed: 20,
  });

  const poses = useMemo(() => {
    const rng = new Rng(sim.seed);
    const out: Pose2[] = [];
    for (let i = 0; i < N; i++) {
      // δ ~ N(0, Σ) in the tangent space, then x₀ ⊞ (τ₀ + δ).
      const d = sampleMvn([0, 0, 0], cov, rng);
      out.push(boxplus(START, [TAU0[0] + d[0], TAU0[1] + d[1], TAU0[2] + d[2]]));
    }
    return out;
  }, [sim.seed, cov]);

  const shown = useMemo(
    () => poses.slice(0, Math.max(sim.state.revealed, PER_TICK)),
    [poses, sim.state.revealed],
  );

  /** Moment-matched Gaussian in (x, y): the best ellipse these samples admit. */
  const fit = useMemo(() => {
    const n = shown.length;
    let mx = 0;
    let my = 0;
    for (const p of shown) {
      mx += p.x;
      my += p.y;
    }
    mx /= n;
    my /= n;
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (const p of shown) {
      const dx = p.x - mx;
      const dy = p.y - my;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
    }
    const k = Math.max(n - 1, 1);
    const c: Mat = [
      [sxx / k, sxy / k],
      [sxy / k, syy / k],
    ];
    return { mean: [mx, my] as [number, number], cov: c, ellipse: ellipse2(c, 2) };
  }, [shown]);

  const nominal = useMemo(() => se2Exp(TAU0), []);
  const arc = useMemo(() => screwPath(START, TAU0, 40), []);

  /** Fraction of samples inside the fitted 2σ ellipse. A genuine 2-D Gaussian
   *  puts 1 − e⁻² = 86.5% of its mass there. */
  const coverage = useMemo(() => {
    let inside = 0;
    for (const p of shown) {
      if (mahalanobis([p.x, p.y], fit.mean, fit.cov) <= 2) inside += 1;
    }
    return inside / shown.length;
  }, [shown, fit]);

  const bias = Math.hypot(fit.mean[0] - nominal.x, fit.mean[1] - nominal.y);

  /** The heading marginal, by contrast, is still exactly the Gaussian we put in. */
  const headingSigma = useMemo(() => {
    const n = shown.length;
    const mean = shown.reduce((a, q) => a + q.theta, 0) / n;
    const v = shown.reduce((a, q) => a + (q.theta - mean) ** 2, 0) / Math.max(n - 1, 1);
    return Math.sqrt(v);
  }, [shown]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 0.5);

      // The commanded motion: no noise, just exp(τ₀).
      drawPath(ctx, v, arc, p.truth, { dashed: true, lineWidth: 1.6, alpha: 0.85 });
      drawRobot(ctx, v, START, p.prior, 0.2, { filled: false });
      drawRobot(ctx, v, nominal, p.truth, 0.2, { filled: false });
      label(ctx, 'x₀', sx(v, 0) - 4, sy(v, 0) + 18, p.prior, { size: 10, weight: 600 });
      label(ctx, 'exp(τ₀)', sx(v, nominal.x) + 10, sy(v, nominal.y) + 4, p.truth, {
        size: 10,
        weight: 600,
      });

      if (showEllipse) {
        drawCovariance(ctx, v, fit.mean, fit.ellipse, p.prediction, { alpha: 0.95 });
        label(
          ctx,
          'moment-matched 2σ ellipse',
          sx(v, fit.mean[0]),
          sy(v, fit.mean[1]) - 10,
          p.prediction,
          { size: 10, align: 'center', weight: 600 },
        );
      }

      // The pushed-forward samples.
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.55;
      for (const q of shown) {
        ctx.beginPath();
        ctx.arc(sx(v, q.x), sy(v, q.y), 1.9, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // A handful of headings, so the reader remembers these are poses.
      for (let i = 0; i < shown.length; i += 60) {
        drawRobot(ctx, v, shown[i], p.posterior, 0.13, { filled: false, alpha: 0.7 });
      }

      // Sample mean vs. commanded endpoint: exp(E[τ]) ≠ E[exp(τ)].
      ctx.fillStyle = p.posterior;
      ctx.beginPath();
      ctx.arc(sx(v, fit.mean[0]), sy(v, fit.mean[1]), 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.3;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(v, fit.mean[0]), sy(v, fit.mean[1]));
      ctx.lineTo(sx(v, nominal.x), sy(v, nominal.y));
      ctx.stroke();
      ctx.setLineDash([]);

      label(ctx, `${shown.length} samples`, 10, 14, p.posterior, { size: 10, weight: 600 });
    },
    [shown, fit, nominal, arc, showEllipse],
  );

  return (
    <WidgetFrame
      id="w3.4"
      title="Banana Preview"
      teaches="Pose uncertainty is not an ellipse. A Gaussian in the tangent space becomes a bent, skewed cloud after exp — and its mean is not where you aimed."
      colorKey={['prior', 'prediction', 'posterior', 'truth']}
      caption={
        <>
          Seven hundred draws from a perfectly ordinary Gaussian{' '}
          <span className="font-mono">N(0, diag(σ_v², σ_v², σ_ω²))</span> in the tangent space, each
          pushed through <span className="font-mono">x₀ ⊞ (τ₀ + δ)</span>. Turn σ<sub>ω</sub> up: the
          cloud bends, because a heading error early in the arc becomes a position error that grows
          with the distance still to drive. The orange ellipse is not a strawman — it is the
          <em> best</em> Gaussian fit to these exact samples, and it still claims mass where there
          are no samples and misses the tips of the banana. Watch the coverage readout drift away
          from 86.5%, the value a real 2-D Gaussian would give, and watch the purple mean pull off
          the commanded endpoint: <span className="font-mono">E[exp(τ)] ≠ exp(E[τ])</span>. Drive
          σ<sub>ω</sub> toward zero and the banana straightens into an honest ellipse — that limit
          is exactly the bet the EKF makes. One control: the heading readout confirms that the
          <em> tangent-space</em> distribution is still precisely the Gaussian we specified. Nothing
          about the noise changed; only the space it was pushed into.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.5, maxX: 2.6, minY: -0.6, maxY: 2.2 }}
        draw={draw}
        deps={[shown, fit, showEllipse]}
        aspect={1.85}
        padding={0.08}
        ariaLabel="A scatter of several hundred robot poses forming a curved banana-shaped cloud, with an orange ellipse fitted to them that does not match the cloud's shape."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat
          label="2σ ellipse coverage"
          value={`${(coverage * 100).toFixed(1)}%`}
          hint="of 86.5%"
          alert={Math.abs(coverage - 0.8647) > 0.02}
        />
        <Stat label="‖sample mean − exp(τ₀)‖" value={`${bias.toFixed(3)} m`} alert={bias > 0.05} />
        <Stat
          label="measured heading σ"
          value={`${headingSigma.toFixed(3)} rad`}
          hint={`of ${sigmaOmega.toFixed(2)}`}
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Heading noise σ_ω"
          role="posterior"
          value={sigmaOmega}
          min={0.01}
          max={0.6}
          step={0.01}
          unit="rad"
          onChange={setSigmaOmega}
          help="The one knob that bends the cloud. Everything else only scales it."
        />
        <Slider
          label="Translation noise σ_v"
          role="posterior"
          value={sigmaV}
          min={0.01}
          max={0.4}
          step={0.01}
          unit="m"
          onChange={setSigmaV}
        />
        <Toggle
          label="Show fitted ellipse"
          role="prediction"
          checked={showEllipse}
          onChange={setShowEllipse}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
      />
    </WidgetFrame>
  );
}

function Stat({
  label: l,
  value,
  hint,
  alert,
}: {
  label: string;
  value: string;
  hint?: string;
  alert?: boolean;
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={alert ? { color: 'var(--pr-prediction)' } : undefined}
      >
        {value}
        {hint ? <span className="ml-1 text-[0.65rem] opacity-60">{hint}</span> : null}
      </div>
    </div>
  );
}
