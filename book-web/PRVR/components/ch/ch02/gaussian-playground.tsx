'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { cholesky, ellipse2 } from '@/lib/prob/linalg';
import { normalPdf } from '@/lib/prob/gaussian';
import { covFromCorrelation, chi2Quantile2 } from '@/lib/prob/joint-gaussian';
import { clear, drawGrid, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w2.1 — the Gaussian Playground.
 *
 * One 2-D Gaussian drawn three ways at once: 500 samples, the iso-density
 * ellipses, and the eigenvector axes. The correlation ρ sweeps on its own, and
 * the crucial detail is that the *same* 500 standard-normal draws are pushed
 * through the changing Cholesky factor every frame — so the reader is watching
 * Derivation 4 (y = μ + Lz) execute, not a new random cloud each tick.
 *
 * The marginal densities are painted along the two axes. They never move while
 * the ellipse rotates, which is the whole argument against reading the
 * ellipse's axis lengths as standard deviations.
 */

const N_SAMPLES = 500;
/** The 95% ellipse for two degrees of freedom: √5.991 = 2.4477 σ, not 2σ. */
const N_SIGMA_95 = Math.sqrt(chi2Quantile2(0.95));

interface State {
  /** Fixed standard-normal draws. The transform changes; these never do. */
  z: [number, number][];
  rho: number;
  dir: number;
}

interface Params {
  sa: number;
  sb: number;
  mx: number;
  my: number;
}

export function GaussianPlayground() {
  const [params, setParams] = useState<Params>({ sa: 2, sb: 1, mx: 0, my: 0 });

  const init = useCallback((seed: number): State => {
    const r = new Rng(seed);
    return {
      z: Array.from({ length: N_SAMPLES }, () => [r.normal(), r.normal()] as [number, number]),
      rho: 0,
      dir: 1,
    };
  }, []);

  const step = useCallback((s: State): State => {
    let rho = s.rho + s.dir * 0.02;
    let dir = s.dir;
    if (rho > 0.95) {
      rho = 0.95;
      dir = -1;
    } else if (rho < -0.95) {
      rho = -0.95;
      dir = 1;
    }
    return { ...s, rho, dir };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 20, initialSeed: 7 });

  const { rho } = sim.state;
  const cov = useMemo(
    () => covFromCorrelation(params.sa, params.sb, rho),
    [params.sa, params.sb, rho],
  );

  /** Σ = L Lᵀ, then x = μ + L z. One factorization per frame, 500 reuses. */
  const cloud = useMemo(() => {
    const l = cholesky(cov);
    return sim.state.z.map(([z0, z1]) => [
      params.mx + l[0][0] * z0,
      params.my + l[1][0] * z0 + l[1][1] * z1,
    ] as [number, number]);
  }, [cov, sim.state.z, params.mx, params.my]);

  const geom = useMemo(() => {
    const tr = cov[0][0] + cov[1][1];
    const det = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    const disc = Math.sqrt(Math.max((tr * tr) / 4 - det, 0));
    return { l1: tr / 2 + disc, l2: Math.max(tr / 2 - disc, 0), det };
  }, [cov]);

  const setMean = useCallback((world: [number, number]) => {
    setParams((p) => ({
      ...p,
      mx: Math.max(-3, Math.min(3, world[0])),
      my: Math.max(-1.5, Math.min(1.5, world[1])),
    }));
  }, []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 1);

      // ---- the cloud: 500 draws, same z every frame ----------------------
      ctx.fillStyle = p.prior;
      ctx.globalAlpha = 0.35;
      for (const [x, y] of cloud) {
        ctx.beginPath();
        ctx.arc(sx(v, x), sy(v, y), 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // ---- iso-density ellipses -------------------------------------------
      const rings: { n: number; dash: number[] }[] = [
        { n: 1, dash: [] },
        { n: 2, dash: [] },
        { n: N_SIGMA_95, dash: [5, 4] },
      ];
      for (const { n, dash } of rings) {
        const e = ellipse2(cov, n);
        ctx.save();
        ctx.translate(sx(v, params.mx), sy(v, params.my));
        ctx.rotate(-e.angle);
        ctx.strokeStyle = p.prior;
        ctx.lineWidth = n === 1 ? 2 : 1.4;
        ctx.globalAlpha = n === 1 ? 1 : 0.65;
        ctx.setLineDash(dash);
        ctx.beginPath();
        ctx.ellipse(0, 0, Math.max(sl(v, e.rx), 0.5), Math.max(sl(v, e.ry), 0.5), 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // ---- eigenvector axes -----------------------------------------------
      const e1 = ellipse2(cov, 1);
      const ux = Math.cos(e1.angle);
      const uy = Math.sin(e1.angle);
      const axes: [number, number, number][] = [
        [ux, uy, Math.sqrt(geom.l1)],
        [-uy, ux, Math.sqrt(geom.l2)],
      ];
      ctx.strokeStyle = p.truth;
      ctx.lineWidth = 1.6;
      for (const [dx, dy, len] of axes) {
        const tipX = params.mx + dx * len;
        const tipY = params.my + dy * len;
        ctx.beginPath();
        ctx.moveTo(sx(v, params.mx), sy(v, params.my));
        ctx.lineTo(sx(v, tipX), sy(v, tipY));
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(sx(v, tipX), sy(v, tipY), 2.6, 0, Math.PI * 2);
        ctx.fillStyle = p.truth;
        ctx.fill();
      }
      label(ctx, '√λ₁', sx(v, params.mx + ux * Math.sqrt(geom.l1)) + 7, sy(v, params.my + uy * Math.sqrt(geom.l1)), p.truth, { size: 10 });

      // ---- the mean -------------------------------------------------------
      ctx.fillStyle = p.prior;
      ctx.beginPath();
      ctx.arc(sx(v, params.mx), sy(v, params.my), 3.4, 0, Math.PI * 2);
      ctx.fill();

      // ---- marginals, painted along the axes in pixel space ---------------
      // These are the answer to "how uncertain is x alone?" — and they are
      // stubbornly independent of ρ, which is the point of the widget.
      const band = Math.min(52, v.height * 0.22);
      const peakA = normalPdf(params.mx, params.mx, params.sa);
      ctx.strokeStyle = p.prior;
      ctx.fillStyle = p.prior;
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = 0.16;
      ctx.beginPath();
      ctx.moveTo(0, v.height);
      for (let px = 0; px <= v.width; px += 2) {
        const wx = v.minX + (px / v.width) * (v.maxX - v.minX);
        const h = (normalPdf(wx, params.mx, params.sa) / peakA) * band;
        ctx.lineTo(px, v.height - h);
      }
      ctx.lineTo(v.width, v.height);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let px = 0; px <= v.width; px += 2) {
        const wx = v.minX + (px / v.width) * (v.maxX - v.minX);
        const h = (normalPdf(wx, params.mx, params.sa) / peakA) * band;
        if (px === 0) ctx.moveTo(px, v.height - h);
        else ctx.lineTo(px, v.height - h);
      }
      ctx.stroke();

      const peakB = normalPdf(params.my, params.my, params.sb);
      ctx.globalAlpha = 0.16;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      for (let py = 0; py <= v.height; py += 2) {
        const wy = v.minY + ((v.height - py) / v.height) * (v.maxY - v.minY);
        const w = (normalPdf(wy, params.my, params.sb) / peakB) * band;
        ctx.lineTo(w, py);
      }
      ctx.lineTo(0, v.height);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let py = 0; py <= v.height; py += 2) {
        const wy = v.minY + ((v.height - py) / v.height) * (v.maxY - v.minY);
        const w = (normalPdf(wy, params.my, params.sb) / peakB) * band;
        if (py === 0) ctx.moveTo(w, py);
        else ctx.lineTo(w, py);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      label(ctx, 'marginal p(xₐ) — fixed', 8, v.height - band - 12, p.prior, { size: 10 });
      label(ctx, `ρ = ${rho >= 0 ? '+' : ''}${rho.toFixed(2)}`, v.width - 10, 16, p.prior, {
        size: 12,
        align: 'right',
        weight: 600,
      });
      // Near-|ρ| = 1 the small eigenvalue collapses: the "cloud" is really a
      // line, det Σ → 0, and the density does not exist. Say so out loud.
      if (geom.l2 < 5e-3 * geom.l1) {
        label(ctx, 'λ₂ → 0: degenerate, no density', v.width - 10, 32, p.prediction, {
          size: 10,
          align: 'right',
        });
      }
    },
    [cloud, cov, geom, params, rho],
  );

  return (
    <WidgetFrame
      id="w2.1"
      title="Gaussian Playground"
      teaches="The ellipse's axes are the eigenvectors of Σ, not the coordinate axes — so its half-widths are not the marginal standard deviations."
      colorKey={['prior', 'truth']}
      caption={
        <>
          The correlation ρ sweeps on its own; grab the slider to take over. Watch three things.
          The cloud <em>shears</em> — but it is the same 500 standard-normal draws every frame,
          pushed through a different Cholesky factor, which is all that <code>x = μ + Lz</code>{' '}
          means. The gray axes rotate with the eigenvectors of Σ while the blue marginal curves
          along the edges do not move at all: ρ changes how the two coordinates covary, not how
          uncertain either one is alone. And det Σ collapses toward zero as |ρ| → 1, because a
          perfectly correlated pair is really one number wearing two hats. Drag inside the panel to
          move μ; at ρ = 0.95 with σₐ = 2, σ_b = 1 you are looking at this chapter&apos;s worked
          matrix.
        </>
      }
    >
      <SimCanvas
        // The vertical span is what `fitViewport` honors at this aspect, so it
        // is sized to hold the 95% ellipse at the largest σ_b the slider allows.
        world={{ minX: -6, maxX: 6, minY: -3.8, maxY: 3.8 }}
        draw={draw}
        deps={[sim.tick, cloud, params]}
        aspect={2.1}
        padding={0}
        cursor="crosshair"
        ariaLabel="A cloud of 500 samples from a two-dimensional Gaussian with its one-sigma, two-sigma and 95 percent iso-density ellipses and eigenvector axes. As the correlation sweeps, the ellipse tilts while the marginal density curves drawn along the two axes stay fixed."
        onPointer={(world, phase, event) => {
          if (phase === 'down' || (phase === 'move' && event.buttons > 0)) setMean(world);
        }}
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="Σ" value={`[${cov[0][0].toFixed(2)} ${cov[0][1].toFixed(2)}; ${cov[1][0].toFixed(2)} ${cov[1][1].toFixed(2)}]`} />
        <Stat label="λ₁, λ₂" value={`${geom.l1.toFixed(3)}, ${geom.l2.toFixed(3)}`} />
        <Stat label="det Σ" value={geom.det.toFixed(4)} />
        <Stat label="95% ellipse" value={`${N_SIGMA_95.toFixed(3)} σ`} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Correlation ρ"
          role="prior"
          value={rho}
          min={-0.99}
          max={0.99}
          step={0.01}
          onChange={(value) => {
            sim.pause();
            sim.setState((s) => ({ ...s, rho: value }));
          }}
          help="Sweeps automatically until you touch it."
        />
        <Slider
          label="σₐ (horizontal)"
          value={params.sa}
          min={0.6}
          max={3}
          step={0.05}
          onChange={(v) => setParams((p) => ({ ...p, sa: v }))}
        />
        <Slider
          label="σ_b (vertical)"
          value={params.sb}
          min={0.4}
          max={1.5}
          step={0.05}
          onChange={(v) => setParams((p) => ({ ...p, sb: v }))}
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

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-[0.72rem] tabular-nums">{value}</div>
    </div>
  );
}
