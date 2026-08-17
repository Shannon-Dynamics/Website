'use client';

import type { ReactNode } from 'react';
import type { Particle } from '@/lib/filters/pf';
import { ellipse2 } from '@/lib/prob/linalg';
import {
  drawCovariance,
  drawOccupancyGrid,
  drawParticles,
  drawPath,
  drawRobot,
  drawScan,
  drawSegments,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import { beamAngles } from '@/lib/sim/world';
import { AutonomyStack, walkerSegments, type Mode } from '@/lib/capstone/stack';

/**
 * Everything Chapter 26's widgets share: one renderer for the mission scene and
 * a handful of small readout components.
 *
 * The renderer takes an `AutonomyStack` and draws its *published products* —
 * map, belief, path, rollouts, frontiers — never its internals. That is the
 * same discipline the stack itself imposes on its tasks, and it is why the
 * Failure Theater can reuse the Grand Demo's picture unchanged.
 */

export interface SceneOptions {
  showTruth: boolean;
  showRollouts: boolean;
  showFrontiers: boolean;
  showScan: boolean;
  showPath: boolean;
  trailEst: { x: number; y: number }[];
  trailTruth: { x: number; y: number }[];
}

export function drawScene(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  p: Palette,
  s: AutonomyStack,
  o: SceneOptions,
): void {
  ctx.clearRect(0, 0, v.width, v.height);
  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, v.width, v.height);

  // --- the map Rusty built: grayscale log odds, Chapter 13's convention ----
  ctx.save();
  ctx.globalAlpha = 0.92;
  drawOccupancyGrid(ctx, v, s.grid, p);
  ctx.restore();

  // --- ground truth, only if the reader asks for it ------------------------
  if (o.showTruth) {
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.setLineDash([4, 4]);
    drawSegments(ctx, v, s.world.walls, p.truth, 1.4);
    ctx.restore();
    if (o.trailTruth.length > 1) drawPath(ctx, v, o.trailTruth, p.truth, { dashed: true, alpha: 0.75 });
  }

  // --- frontiers: the boundary of what is known ----------------------------
  if (o.showFrontiers) {
    const cs = s.grid.cellSize;
    const w = Math.max(2, sl(v, cs));
    ctx.save();
    for (const f of s.frontiers) {
      const chosen = s.mode.kind === 'Navigate' && Math.hypot(f.goal[0] - s.mode.goal[0], f.goal[1] - s.mode.goal[1]) < 0.3;
      ctx.fillStyle = chosen ? p.accent : p.prior;
      ctx.globalAlpha = chosen ? 0.85 : 0.4;
      for (const k of f.cells) {
        const i = k % s.grid.width;
        const j = (k - i) / s.grid.width;
        const [cx, cy] = s.grid.cellCenter(i, j);
        ctx.fillRect(sx(v, cx - cs / 2), sy(v, cy + cs / 2), w, w);
      }
    }
    ctx.restore();
  }

  // --- global plan ---------------------------------------------------------
  if (o.showPath && s.path.length > 1) {
    drawPath(ctx, v, s.path.map(([x, y]) => ({ x, y })), p.accent, { lineWidth: 2, alpha: 0.9 });
    const g = s.path[s.path.length - 1];
    ctx.save();
    ctx.fillStyle = p.accent;
    ctx.beginPath();
    ctx.arc(sx(v, g[0]), sy(v, g[1]), 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // --- MPPI rollout fan ----------------------------------------------------
  if (o.showRollouts && s.lastMppi) {
    const best = Math.max(...s.lastMppi.rollouts.map((r) => r.weight), 1e-9);
    ctx.save();
    for (const r of s.lastMppi.rollouts) {
      ctx.globalAlpha = 0.1 + 0.55 * (r.weight / best);
      ctx.strokeStyle = p.prediction;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, s.belief.mean.x), sy(v, s.belief.mean.y));
      for (const [x, y] of r.pts) ctx.lineTo(sx(v, x), sy(v, y));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = p.prediction;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(sx(v, s.belief.mean.x), sy(v, s.belief.mean.y));
    for (const [x, y] of s.lastMppi.nominal) ctx.lineTo(sx(v, x), sy(v, y));
    ctx.stroke();
    ctx.restore();
  }

  // --- live scan -----------------------------------------------------------
  if (o.showScan && s.scan) {
    drawScan(ctx, v, s.belief.mean, s.scan.v, beamAngles(s.cfg.scan), p.measurement, s.cfg.scan.maxRange);
  }

  // --- novelty: beams the map cannot explain -------------------------------
  if (s.novelty.length > 0) {
    ctx.save();
    ctx.fillStyle = p.prediction;
    for (const [x, y] of s.novelty) {
      ctx.beginPath();
      ctx.arc(sx(v, x), sy(v, y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // --- the walker is a physical object, so it is ground truth --------------
  if (s.walker) {
    ctx.save();
    ctx.globalAlpha = o.showTruth ? 0.9 : 0.35;
    drawSegments(ctx, v, walkerSegments(s.walker), p.truth, 2);
    ctx.restore();
  }

  // --- the belief ----------------------------------------------------------
  if (o.trailEst.length > 1) drawPath(ctx, v, o.trailEst, p.posterior, { lineWidth: 1.5, alpha: 0.7 });

  if (s.mode.kind === 'Relocalize') {
    // Six thousand particles is the right number for the filter and the wrong
    // number for a canvas, so the picture shows every eighth one.
    const shown: Particle[] = [];
    const all = s.relocalizer.particles;
    const stride = Math.max(1, Math.ceil(all.length / 800));
    for (let i = 0; i < all.length; i += stride) shown.push(all[i]);
    drawParticles(ctx, v, shown, p.posterior, { showHeading: false, maxRadius: 1.8 });
  } else {
    const cov = worldPositionCov(s.belief.cov, s.belief.mean.theta);
    drawCovariance(ctx, v, [s.belief.mean.x, s.belief.mean.y], ellipse2(cov, 3), p.posterior, { alpha: 0.95 });
  }

  if (o.showTruth) drawRobot(ctx, v, s.truth, p.truth, 0.26, { filled: false });
  drawRobot(ctx, v, s.belief.mean, p.posterior, 0.24);

  // --- the F2 margin, drawn as the ring the controller must respect --------
  ctx.save();
  ctx.strokeStyle = p.prediction;
  ctx.globalAlpha = 0.55;
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(sx(v, s.belief.mean.x), sy(v, s.belief.mean.y), Math.max(sl(v, s.margin), 2), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  label(ctx, modeLabel(s.mode), 8, 12, modeColor(s.mode, p), { size: 11, weight: 700 });
  label(ctx, `t = ${s.time.toFixed(1)} s`, v.width - 8, 12, p.ink, { size: 10, align: 'right' });
}

/** Rotate a body-frame 2×2 position covariance into the world frame. */
export function worldPositionCov(cov: number[][], theta: number): number[][] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const a = cov[0][0];
  const b = cov[0][1];
  const d = cov[1][1];
  return [
    [c * c * a - 2 * c * s * b + s * s * d, c * s * (a - d) + (c * c - s * s) * b],
    [c * s * (a - d) + (c * c - s * s) * b, s * s * a + 2 * c * s * b + c * c * d],
  ];
}

export function modeLabel(m: Mode): string {
  if (m.kind === 'Navigate') return `NAVIGATE → (${m.goal[0].toFixed(1)}, ${m.goal[1].toFixed(1)})`;
  if (m.kind === 'Recover') return `RECOVER · ${m.why}`;
  return m.kind.toUpperCase();
}

export function modeColor(m: Mode, p: Palette): string {
  switch (m.kind) {
    case 'Relocalize':
      return p.posterior;
    case 'Recover':
      return p.prediction;
    case 'Done':
      return p.measurement;
    default:
      return p.accent;
  }
}

/* -------------------------------------------------------------------------- */
/* Small readouts                                                              */
/* -------------------------------------------------------------------------- */

export function Readout({
  label: l,
  value,
  role,
  alarm,
}: {
  label: string;
  value: string;
  role?: 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';
  alarm?: boolean;
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-[0.78rem] tabular-nums"
        style={{ color: alarm ? 'var(--pr-prediction)' : role ? `var(--pr-${role})` : undefined }}
      >
        {value}
      </div>
    </div>
  );
}

export function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border-t border-fd-border px-3 py-2 first:border-t-0">
      <p className="eyebrow mb-1.5">{title}</p>
      {children}
    </div>
  );
}

/** A labelled horizontal bar, 0…1, used for staleness and utility. */
export function Bar({ frac, color, warn = false }: { frac: number; color: string; warn?: boolean }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-fd-muted">
      <div
        className="h-full rounded-full transition-[width] duration-150"
        style={{
          width: `${Math.max(2, Math.min(100, frac * 100))}%`,
          backgroundColor: warn ? 'var(--pr-prediction)' : color,
        }}
      />
    </div>
  );
}
