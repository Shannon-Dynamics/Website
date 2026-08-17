'use client';

import { useMemo, useRef, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

/**
 * `ch17-dmp-sculptor` — dynamic movement primitives, and what they generalize.
 *
 * A DMP is a spring–damper pulled toward a goal, plus a learned forcing term
 * that shapes the path taken to get there. The spring guarantees convergence
 * no matter what the forcing term does; the forcing term carries the style of
 * the demonstrated motion. Move the goal and the shape survives — that is the
 * generalization property that made DMPs the standard motor representation.
 */
export function DmpSculptor() {
  const { mode } = useTheme();
  const [goal, setGoal] = useState(1.0);
  const [tau, setTau] = useState(1.0);
  const [nBasis, setNBasis] = useState(10);
  const [forcingScale, setForcingScale] = useState(1.0);
  /**
   * A demonstration the reader drew. Locally weighted regression fits the
   * forcing term to it — which is exactly how a DMP is taught from a
   * kinesthetic demonstration on a real arm, and it is a linear solve rather
   * than an RL problem.
   */
  const [demo, setDemo] = useState<number[] | null>(null);
  const drawing = useRef(false);
  const demoBuf = useRef<Array<{ t: number; y: number }>>([]);

  const { trajectory, basisCurves, forcing, converged } = useMemo(() => {
    // Canonical system: ẋ = −α_x x / τ, decaying from 1 to 0. It is the DMP's
    // clock, and because it decays the forcing term must vanish — which is why
    // convergence to the goal is guaranteed regardless of what was learned.
    const alphaX = 4.0;
    const alphaY = 25.0;
    const betaY = alphaY / 4; // critical damping
    const dt = 0.004;
    const steps = 900;
    const y0 = 0;

    // Gaussian basis functions spaced along the canonical variable.
    const centers = Array.from({ length: nBasis }, (_, i) =>
      Math.exp((-alphaX * i) / (nBasis - 1)),
    );
    const widths = centers.map((_, i) =>
      i < nBasis - 1 ? 1.2 / Math.pow(centers[i + 1] - centers[i], 2) : 1.2 / Math.pow(0.05, 2),
    );

    // Weights come either from the reader's drawing (locally weighted
    // regression on the demonstrated trajectory) or from a canned shape.
    let weights: number[];
    if (demo && demo.length > 8) {
      const n = demo.length;
      const y0d = demo[0];
      const gd = demo[n - 1];
      const dtD = 1 / n;
      const dy: number[] = demo.map((_, i) => (i === 0 ? 0 : (demo[i] - demo[i - 1]) / dtD));
      const ddy: number[] = dy.map((_, i) => (i === 0 ? 0 : (dy[i] - dy[i - 1]) / dtD));

      // Invert the transformation system for the forcing term it implies.
      const xs: number[] = [];
      const targets: number[] = [];
      let xx = 1;
      for (let i = 0; i < n; i++) {
        targets.push(ddy[i] - alphaY * (betaY * (gd - demo[i]) - dy[i]));
        xs.push(xx);
        xx -= alphaX * xx * dtD;
      }
      weights = centers.map((c, k) => {
        let num = 0;
        let den = 0;
        for (let i = 0; i < n; i++) {
          const psi = Math.exp(-widths[k] * Math.pow(xs[i] - c, 2));
          const sfac = xs[i] * (gd - y0d);
          num += psi * sfac * targets[i];
          den += psi * sfac * sfac;
        }
        return forcingScale * (Math.abs(den) > 1e-9 ? num / den : 0);
      });
    } else {
      weights = Array.from({ length: nBasis }, (_, i) => {
        const t = i / (nBasis - 1);
        return forcingScale * (140 * Math.sin(Math.PI * t) - 60 * Math.sin(2 * Math.PI * t));
      });
    }

    let x = 1;
    let y = y0;
    let dy = 0;
    const traj: Array<{ x: number; y: number }> = [];
    const forcingTrace: Array<{ x: number; y: number }> = [];

    for (let k = 0; k < steps; k++) {
      // Weighted sum of basis activations, normalized.
      let num = 0;
      let den = 0;
      for (let i = 0; i < nBasis; i++) {
        const psi = Math.exp(-widths[i] * Math.pow(x - centers[i], 2));
        num += psi * weights[i];
        den += psi;
      }
      // The (g − y₀)x factor makes the forcing scale with the movement extent
      // and vanish as x → 0.
      const f = den > 1e-9 ? (num / den) * x * (goal - y0) : 0;

      // Transformation system: a critically damped spring toward the goal,
      // perturbed by the learned forcing term.
      const ddy = (alphaY * (betaY * (goal - y) - dy) + f) / (tau * tau);
      dy += ddy * dt;
      y += dy * dt;
      x += (-alphaX * x * dt) / tau;

      if (k % 4 === 0) {
        traj.push({ x: k * dt, y });
        forcingTrace.push({ x: k * dt, y: f / 100 });
      }
    }

    // Basis activations over time, for the lower panel.
    const basis = Array.from({ length: Math.min(nBasis, 6) }, (_, i) => {
      const idx = Math.floor((i * (nBasis - 1)) / Math.max(1, Math.min(nBasis, 6) - 1));
      let xx = 1;
      const pts: Array<{ x: number; y: number }> = [];
      for (let k = 0; k < steps; k += 8) {
        const psi = Math.exp(-widths[idx] * Math.pow(xx - centers[idx], 2));
        pts.push({ x: k * dt, y: psi });
        xx += (-alphaX * xx * dt * 8) / tau;
      }
      return { id: `ψ${idx + 1}`, data: pts };
    });

    return {
      trajectory: traj,
      basisCurves: basis,
      forcing: forcingTrace,
      converged: Math.abs(y - goal),
    };
  }, [goal, tau, nBasis, forcingScale, demo]);

  const W = 400;
  const H = 175;
  const yVals = trajectory.map((p) => p.y);
  const lo = Math.min(...yVals, 0, goal) - 0.15;
  const hi = Math.max(...yVals, goal) + 0.15;
  const sx = (t: number) => 34 + (t / 3.6) * (W - 50);
  const sy = (v: number) => H - 24 - ((v - lo) / (hi - lo)) * (H - 44);

  return (
    <SimPanel
      title="A movement primitive you can reshape"
      id="ch17-dmp-sculptor"
      subtitle="τ²ÿ = α(β(g − y) − τẏ) + f(x) — a spring toward the goal, plus a learned forcing term carrying the demonstrated style."
      controls={
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Slider
            label="Goal g"
            value={goal}
            min={-0.6}
            max={2.2}
            step={0.05}
            onChange={setGoal}
            hint="move it — the shape survives"
          />
          <Slider
            label="Time scaling τ"
            value={tau}
            min={0.4}
            max={2.5}
            step={0.05}
            onChange={setTau}
            hint="slow down or speed up, same path"
          />
          <Slider
            label="Basis functions"
            value={nBasis}
            min={3}
            max={30}
            step={1}
            onChange={setNBasis}
            format={(v) => v.toFixed(0)}
            hint="capacity of the forcing term"
          />
          <button
            type="button"
            onClick={() => setDemo(null)}
            className="h-fit self-end rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
          >
            {demo ? 'Clear my demonstration' : 'No demonstration yet'}
          </button>
          <Slider
            label="Forcing amplitude"
            value={forcingScale}
            min={0}
            max={2}
            step={0.05}
            onChange={setForcingScale}
            hint="0 = pure spring, no style"
          />
        </div>
      }
      caption="Drag across the plot to draw a movement — an arc over an obstacle, a hook, anything — and locally weighted regression fits the forcing term to it in one linear solve, no reinforcement learning involved. Then drag the goal: your shape deforms smoothly and still terminates exactly at g, which is the generalization property that made DMPs the standard motor representation. Set the forcing amplitude to zero and the DMP becomes a plain critically-damped spring: it reaches the goal by the most boring path possible. Turn it up and the demonstrated shape reappears — the arc over the obstacle, the descent onto the target. Now drag the goal: the shape deforms smoothly rather than breaking, and the trajectory still ends exactly at g. That guarantee comes from the canonical system decaying to zero, which forces f to vanish and leaves the spring in charge at the end."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,185px]">
        <div>
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="max-w-full touch-none rounded-lg"
            style={{
              background: 'var(--surface-sunken)',
              cursor: 'crosshair',
            }}
            onPointerDown={(e) => {
              drawing.current = true;
              demoBuf.current = [];
              e.currentTarget.setPointerCapture?.(e.pointerId);
            }}
            onPointerMove={(e) => {
              if (!drawing.current) return;
              const r = e.currentTarget.getBoundingClientRect();
              const scx = W / r.width;
              const scy = H / r.height;
              const px = (e.clientX - r.left) * scx;
              const py = (e.clientY - r.top) * scy;
              // Invert the screen mapping back into (time, position).
              const t = ((px - 34) / (W - 50)) * 3.6;
              const yv = lo + ((H - 24 - py) / (H - 44)) * (hi - lo);
              if (t >= 0 && t <= 3.6) demoBuf.current.push({ t, y: yv });
            }}
            onPointerUp={() => {
              drawing.current = false;
              const pts = demoBuf.current;
              if (pts.length < 12) return;
              // Resample onto a uniform time grid — LWR wants even spacing.
              const N = 120;
              const t0 = pts[0].t;
              const t1 = pts[pts.length - 1].t;
              if (t1 - t0 < 0.4) return;
              const sampled: number[] = [];
              for (let i = 0; i < N; i++) {
                const tt = t0 + ((t1 - t0) * i) / (N - 1);
                let k = 0;
                while (k < pts.length - 1 && pts[k + 1].t < tt) k++;
                sampled.push(pts[k].y);
              }
              setDemo(sampled);
              setGoal(Number(sampled[sampled.length - 1].toFixed(2)));
            }}
            role="img"
            aria-label="DMP trajectory from start to goal; drag across to draw your own demonstration"
          >
            <line
              x1={30}
              y1={sy(goal)}
              x2={W - 12}
              y2={sy(goal)}
              stroke="var(--status-good)"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
            <text x={W - 46} y={sy(goal) - 5} fontSize={9.5} fill="var(--status-good)">
              goal g
            </text>

            {demo && (
              <path
                d={`M${demo
                  .map((v, i) => `${sx((i / (demo.length - 1)) * 3.6).toFixed(1)},${sy(v).toFixed(1)}`)
                  .join(' L')}`}
                fill="none"
                stroke={seriesColor(4, mode)}
                strokeWidth={4}
                opacity={0.32}
                strokeLinecap="round"
              />
            )}
            <path
              d={`M${trajectory.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' L')}`}
              fill="none"
              stroke={seriesColor(0, mode)}
              strokeWidth={2.5}
            />
            <circle cx={sx(0)} cy={sy(0)} r={5} fill={seriesColor(1, mode)} />
            <text x={sx(0) - 4} y={sy(0) + 18} fontSize={9.5} fill="var(--text-muted)">
              start
            </text>
            <text x={34} y={14} fontSize={9.5} fill="var(--text-muted)">
              {demo ? 'thick pale: your demonstration · thin: the fitted DMP' : 'position y(t) — drag across to draw a demonstration'}
            </text>
          </svg>

          <LineChart
            data={[...basisCurves, { id: 'forcing f(x)/100', data: forcing }]}
            height={155}
            xLegend="time (s)"
            yLegend="activation"
            caption="Basis activations sweep across the movement in order, gated by the canonical clock; their weighted sum is the forcing term."
          />
        </div>

        <div className="space-y-2">
          <StatTile
            label="Final error |y − g|"
            value={converged}
            status={converged < 0.02 ? 'good' : 'warning'}
            hint="the spring guarantees this → 0"
          />
          <StatTile
            label="Movement duration"
            value={3.6 * tau}
            unit="s"
            hint="τ rescales time, not shape"
          />
          <StatTile
            label="Learned parameters"
            value={nBasis}
            hint="one weight per basis function"
          />
          <p className="rounded-lg border border-hairline px-2.5 py-2 text-[11.5px] leading-snug text-ink-muted">
            A whole reaching motion in ten numbers — which is why policy search
            over DMP weights was tractable on real robots long before deep RL.
          </p>
        </div>
      </div>
    </SimPanel>
  );
}
