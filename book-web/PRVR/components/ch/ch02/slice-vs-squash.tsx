'use client';

import Link from 'next/link';
import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { cholesky, ellipse2 } from '@/lib/prob/linalg';
import { normalPdf } from '@/lib/prob/gaussian';
import {
  chi2Quantile2,
  conditional2,
  covFromCorrelation,
  marginalize,
} from '@/lib/prob/joint-gaussian';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w2.4 — Slice vs. Squash.
 *
 * One joint Gaussian over (xₐ, x_b), two ways of reducing it to one dimension.
 * Squash the cloud flat and you get the marginal p(xₐ) — blue, unchanging.
 * Slice it at x_b = β and you get the conditional p(xₐ | x_b = β) — purple,
 * narrower by exactly √(1 − ρ²) and sliding with β.
 *
 * The slice line sweeps by itself so the reader sees the one thing that is
 * genuinely surprising: the purple curve moves but never changes width. The
 * conditional covariance does not depend on what you measured, only on the
 * fact that you measured. That is the Kalman update's geometry, four chapters
 * early, and the caption says so.
 */

const N_SAMPLES = 420;
const N_SIGMA_95 = Math.sqrt(chi2Quantile2(0.95));
const MU: [number, number] = [0, 0];

interface State {
  z: [number, number][];
  beta: number;
  dir: number;
}

export function SliceVsSquash() {
  const [rho, setRho] = useState(0.9);
  const [sa, setSa] = useState(2);
  const sb = 1;

  const init = useCallback((seed: number): State => {
    const r = new Rng(seed);
    return {
      z: Array.from({ length: N_SAMPLES }, () => [r.normal(), r.normal()] as [number, number]),
      beta: 0,
      dir: 1,
    };
  }, []);

  const step = useCallback((s: State): State => {
    let beta = s.beta + s.dir * 0.045 * sb;
    let dir = s.dir;
    if (beta > 2 * sb) {
      beta = 2 * sb;
      dir = -1;
    } else if (beta < -2 * sb) {
      beta = -2 * sb;
      dir = 1;
    }
    return { ...s, beta, dir };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 24, initialSeed: 19 });
  const { beta } = sim.state;

  const cov = useMemo(() => covFromCorrelation(sa, sb, rho), [sa, rho]);

  const cloud = useMemo(() => {
    const l = cholesky(cov);
    return sim.state.z.map(
      ([z0, z1]) =>
        [MU[0] + l[0][0] * z0, MU[1] + l[1][0] * z0 + l[1][1] * z1] as [number, number],
    );
  }, [cov, sim.state.z]);

  // Both reductions come from the library, so the curves on screen are the
  // Schur complement and the block copy, not a formula retyped for drawing.
  const marginal = useMemo(() => {
    const m = marginalize([MU[0], MU[1]], cov, [0]);
    return { mean: m.mean[0], variance: m.cov[0][0] };
  }, [cov]);
  const conditional = useMemo(() => conditional2([MU[0], MU[1]], cov, beta), [cov, beta]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);

      const band = Math.min(84, v.height * 0.26);
      const floorY = v.height - 6;

      // ---- joint cloud ----------------------------------------------------
      ctx.fillStyle = p.truth;
      for (const [x, y] of cloud) {
        // Points near the slice are the ones the conditional keeps.
        const near = Math.abs(y - beta) < 0.12 * sb;
        ctx.globalAlpha = near ? 0.95 : 0.28;
        ctx.fillStyle = near ? p.measurement : p.truth;
        ctx.beginPath();
        ctx.arc(sx(v, x), sy(v, y), near ? 2.1 : 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // ---- 95% ellipse ----------------------------------------------------
      const e = ellipse2(cov, N_SIGMA_95);
      ctx.save();
      ctx.translate(sx(v, MU[0]), sy(v, MU[1]));
      ctx.rotate(-e.angle);
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(sl(v, e.rx), 0.5), Math.max(sl(v, e.ry), 0.5), 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // ---- the slice ------------------------------------------------------
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(0, sy(v, beta));
      ctx.lineTo(v.width, sy(v, beta));
      ctx.stroke();
      label(ctx, `x_b = β = ${beta.toFixed(2)}`, 8, sy(v, beta) - 9, p.measurement, {
        size: 10,
        weight: 600,
      });

      // ---- the two 1-D answers, along the bottom --------------------------
      const peak = Math.max(
        normalPdf(marginal.mean, marginal.mean, Math.sqrt(marginal.variance)),
        normalPdf(conditional.mean, conditional.mean, Math.sqrt(conditional.variance)),
      );

      const paint = (m: number, variance: number, color: string, fill: boolean) => {
        const std = Math.sqrt(variance);
        const path = () => {
          ctx.beginPath();
          for (let px = 0; px <= v.width; px += 1) {
            const wx = v.minX + (px / v.width) * (v.maxX - v.minX);
            const h = (normalPdf(wx, m, std) / peak) * band;
            if (px === 0) ctx.moveTo(px, floorY - h);
            else ctx.lineTo(px, floorY - h);
          }
        };
        if (fill) {
          path();
          ctx.lineTo(v.width, floorY);
          ctx.lineTo(0, floorY);
          ctx.closePath();
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.14;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        path();
        ctx.strokeStyle = color;
        ctx.lineWidth = fill ? 2.2 : 1.6;
        ctx.stroke();
      };

      ctx.save();
      // A hairline floor so the two densities read as sharing one axis.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, floorY);
      ctx.lineTo(v.width, floorY);
      ctx.stroke();
      paint(marginal.mean, marginal.variance, p.prior, false);
      paint(conditional.mean, conditional.variance, p.posterior, true);
      ctx.restore();

      label(ctx, 'squash → marginal p(xₐ)', 8, floorY - band - 20, p.prior, { size: 10, weight: 600 });
      label(ctx, 'slice → conditional p(xₐ | x_b = β)', 8, floorY - band - 8, p.posterior, {
        size: 10,
        weight: 600,
      });
    },
    [cloud, cov, beta, marginal, conditional],
  );

  const shrink = Math.sqrt(1 - rho * rho);

  return (
    <WidgetFrame
      id="w2.4"
      title="Slice vs. Squash"
      teaches="Marginalizing and conditioning are different operations: one throws a variable away, the other spends it."
      colorKey={['prior', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          The gray cloud is a joint Gaussian over two correlated quantities — say Rusty&apos;s
          position along the corridor and the range its sensor should read. Flatten it and you get
          the blue <strong>marginal</strong>: what you know about xₐ knowing nothing else. Cut it
          at the green line and you get the purple <strong>conditional</strong>: what you know once
          x_b has been observed to be β. The purple curve slides as the slice moves, but{' '}
          <strong>its width never changes</strong> — σ<sub>a|b</sub> = σₐ√(1 − ρ²) contains no β at
          all. Learning <em>that</em> a correlated quantity was measured is what buys certainty;
          learning <em>what</em> it measured only relocates the estimate. Push ρ to zero and the
          two curves coincide, because an uncorrelated measurement teaches you nothing. This is the
          entire geometry of the Kalman update, and{' '}
          <Link href="/chapters/ch06-kalman-filters">Chapter 6</Link> does little more than give the
          factor Σ<sub>ab</sub>Σ<sub>bb</sub>⁻¹ a name.
        </>
      }
    >
      <SimCanvas
        // Extra room below the cloud: the bottom band of the canvas belongs to
        // the two 1-D densities, which are drawn in pixels rather than metres.
        world={{ minX: -6, maxX: 6, minY: -4.4, maxY: 3.2 }}
        draw={draw}
        deps={[sim.tick, cloud, marginal, conditional]}
        aspect={2.1}
        padding={0}
        ariaLabel="A tilted elliptical cloud of samples with a horizontal slice line sweeping up and down. Below it, a wide blue marginal density stays fixed while a narrow purple conditional density slides left and right with the slice, keeping the same width."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="σₐ (marginal)" value={Math.sqrt(marginal.variance).toFixed(3)} role="prior" />
        <Stat
          label="σ_{a|b} (conditional)"
          value={Math.sqrt(conditional.variance).toFixed(3)}
          role="posterior"
        />
        <Stat label="ratio √(1 − ρ²)" value={shrink.toFixed(3)} />
        <Stat label="μ_{a|b}" value={conditional.mean.toFixed(3)} role="posterior" />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Correlation ρ"
          role="posterior"
          value={rho}
          min={-0.98}
          max={0.98}
          step={0.01}
          onChange={setRho}
          help="How much x_b knows about xₐ. At ρ = 0 the slice teaches nothing."
        />
        <Slider
          label="σₐ"
          role="prior"
          value={sa}
          min={0.6}
          max={2.8}
          step={0.05}
          onChange={setSa}
        />
        <Slider
          label="Slice position β"
          role="measurement"
          value={beta}
          min={-2}
          max={2}
          step={0.02}
          onChange={(value) => {
            sim.pause();
            sim.setState((s) => ({ ...s, beta: value }));
          }}
          help="Sweeps automatically until you touch it."
        />
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

function Stat({
  label: l,
  value,
  role,
}: {
  label: string;
  value: string;
  role?: 'prior' | 'measurement' | 'posterior' | 'truth';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={role ? { color: `var(--pr-${role})` } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
