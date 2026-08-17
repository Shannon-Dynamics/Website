'use client';

import { useCallback, useMemo } from 'react';
import { Rng } from '@/lib/prob/rng';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { drawParticles, drawPath, drawRobot, drawSegments, clear, drawGrid } from '@/lib/sim/draw';
import { useSimulation } from '@/lib/sim/use-simulation';

/**
 * The book's opening image: a robot that does not know where it is, and a cloud
 * of hypotheses that collapses onto the truth as evidence arrives.
 *
 * Deliberately self-contained — a miniature Monte Carlo localization rather
 * than the full stack from Chapter 12 — so the front door of the book stays
 * fast and never breaks when the library beneath it changes.
 */

interface P {
  x: number;
  y: number;
  theta: number;
  w: number;
}

interface HeroState {
  rng: Rng;
  truth: { x: number; y: number; theta: number };
  particles: P[];
  trail: { x: number; y: number }[];
}

const ROOM = { minX: 0, minY: 0, maxX: 12, maxY: 7 };

const WALLS = [
  { x1: 0, y1: 0, x2: 12, y2: 0 },
  { x1: 12, y1: 0, x2: 12, y2: 7 },
  { x1: 12, y1: 7, x2: 0, y2: 7 },
  { x1: 0, y1: 7, x2: 0, y2: 0 },
  { x1: 4.5, y1: 0, x2: 4.5, y2: 2.6 },
  { x1: 4.5, y1: 4.4, x2: 4.5, y2: 7 },
  { x1: 8.5, y1: 7, x2: 8.5, y2: 4.2 },
  { x1: 8.5, y1: 2.4, x2: 8.5, y2: 0 },
];

/** Distance from a point to the nearest wall — a stand-in for a range reading. */
function nearestWall(x: number, y: number): number {
  let best = Infinity;
  for (const s of WALLS) {
    const dx = s.x2 - s.x1;
    const dy = s.y2 - s.y1;
    const len2 = dx * dx + dy * dy;
    const t = Math.max(0, Math.min(1, ((x - s.x1) * dx + (y - s.y1) * dy) / len2));
    const px = s.x1 + t * dx;
    const py = s.y1 + t * dy;
    best = Math.min(best, Math.hypot(x - px, y - py));
  }
  return best;
}

const TRUE_PATH = (t: number) => ({
  x: 6 + 4.2 * Math.cos(t * 0.035),
  y: 3.5 + 2.1 * Math.sin(t * 0.07),
});

export function HeroLocalization() {
  const init = useCallback((seed: number): HeroState => {
    const rng = new Rng(seed);
    const p0 = TRUE_PATH(0);
    return {
      rng,
      truth: { ...p0, theta: 0 },
      // Global localization: total ignorance is a cloud spread over the whole map.
      particles: Array.from({ length: 900 }, () => ({
        x: rng.uniform(ROOM.minX, ROOM.maxX),
        y: rng.uniform(ROOM.minY, ROOM.maxY),
        theta: rng.uniform(-Math.PI, Math.PI),
        w: 1 / 900,
      })),
      trail: [],
    };
  }, []);

  const step = useCallback((s: HeroState, tick: number): HeroState => {
    const next = TRUE_PATH(tick + 1);
    const prev = s.truth;
    const dx = next.x - prev.x;
    const dy = next.y - prev.y;
    const truth = { ...next, theta: Math.atan2(dy, dx) };

    // Predict: every hypothesis moves by the same command, plus its own noise.
    const particles = s.particles.map((p) => ({
      x: p.x + dx + s.rng.normal(0, 0.045),
      y: p.y + dy + s.rng.normal(0, 0.045),
      theta: p.theta + s.rng.normal(0, 0.03),
      w: p.w,
    }));

    // Correct: weight by agreement with a noisy range measurement.
    const z = nearestWall(truth.x, truth.y) + s.rng.normal(0, 0.12);
    let sum = 0;
    for (const p of particles) {
      const expected = nearestWall(p.x, p.y);
      const err = z - expected;
      const inside =
        p.x > ROOM.minX && p.x < ROOM.maxX && p.y > ROOM.minY && p.y < ROOM.maxY ? 1 : 0.02;
      p.w = inside * Math.exp(-(err * err) / (2 * 0.3 * 0.3)) + 1e-9;
      sum += p.w;
    }
    for (const p of particles) p.w /= sum;

    // Resample with the low-variance comb — one random number, evenly spaced.
    const M = particles.length;
    const r = s.rng.uniform(0, 1 / M);
    const resampled: P[] = [];
    let c = particles[0].w;
    let i = 0;
    for (let m = 0; m < M; m++) {
      const u = r + m / M;
      while (u > c && i < M - 1) {
        i += 1;
        c += particles[i].w;
      }
      const src = particles[i];
      resampled.push({
        x: src.x + s.rng.normal(0, 0.012),
        y: src.y + s.rng.normal(0, 0.012),
        theta: src.theta,
        w: 1 / M,
      });
    }

    const trail = [...s.trail, { x: truth.x, y: truth.y }].slice(-160);
    return { rng: s.rng, truth, particles: resampled, trail };
  }, []);

  const sim = useSimulation<HeroState>({
    init,
    step,
    fps: 26,
    maxTicks: 420,
    loop: true,
    initialSeed: 7,
  });

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Parameters<typeof clear>[1], p: Parameters<typeof clear>[2]) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 1);
      drawSegments(ctx, v, WALLS, p.wall, 2);
      drawPath(ctx, v, sim.state.trail, p.truth, { dashed: true, alpha: 0.55 });
      drawParticles(
        ctx,
        v,
        sim.state.particles.map((q) => ({ state: { x: q.x, y: q.y, theta: q.theta }, weight: q.w })),
        p.posterior,
        { showHeading: false, maxRadius: 2.1 },
      );
      drawRobot(ctx, v, sim.state.truth, p.truth, 0.3);
    },
    [sim.state],
  );

  const spread = useMemo(() => {
    const ps = sim.state.particles;
    const mx = ps.reduce((a, q) => a + q.x, 0) / ps.length;
    const my = ps.reduce((a, q) => a + q.y, 0) / ps.length;
    return Math.sqrt(ps.reduce((a, q) => a + (q.x - mx) ** 2 + (q.y - my) ** 2, 0) / ps.length);
  }, [sim.state.particles]);

  return (
    <div className="overflow-hidden rounded-lg border border-fd-border bg-fd-card">
      <SimCanvas
        world={ROOM}
        draw={draw}
        deps={[sim.tick]}
        aspect={12 / 7}
        ariaLabel="A robot moving through a floorplan while a cloud of position hypotheses collapses from covering the whole map onto the robot's true location."
      />
      <div className="flex items-center justify-between gap-4 border-t border-fd-border px-3 py-2 font-mono text-[0.7rem] text-fd-muted-foreground tabular-nums">
        <span>
          <span className="eyebrow mr-1.5">belief spread</span>
          {spread.toFixed(2)} m
        </span>
        <button
          type="button"
          onClick={sim.toggle}
          className="font-ui text-[0.72rem] font-medium text-fd-primary hover:underline"
        >
          {sim.playing ? 'Pause' : 'Play'}
        </button>
      </div>
    </div>
  );
}
