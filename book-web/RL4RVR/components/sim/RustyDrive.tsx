'use client';

import { useEffect, useRef, useState } from 'react';
import { Segmented, SimControls, SimPanel } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

interface Obstacle {
  x: number;
  y: number;
  r: number;
}

const W = 520;
const H = 300;
const GOAL = { x: 470, y: 150, r: 18 };

/** Three rooms of increasing clutter — the hand-coded controller's difficulty ladder. */
const LAYOUTS: Record<string, Obstacle[]> = {
  easy: [{ x: 240, y: 150, r: 34 }],
  cluttered: [
    { x: 190, y: 110, r: 28 },
    { x: 250, y: 205, r: 30 },
    { x: 330, y: 120, r: 26 },
  ],
  trap: [
    { x: 200, y: 150, r: 26 },
    { x: 285, y: 92, r: 26 },
    { x: 285, y: 208, r: 26 },
    { x: 350, y: 150, r: 30 },
    { x: 285, y: 150, r: 22 },
  ],
};

/**
 * `ch01-drive-rusty` — the hello-robot demo.
 *
 * Rusty runs a perfectly reasonable hand-written controller: steer toward the
 * goal, veer away from anything close. It works in an empty room. In the
 * cluttered room it stutters. In the trap room it gets stuck in a local
 * minimum forever — the wall the whole book is written against. The reader can
 * also drive manually to feel how much tacit judgment a human brings.
 */
export function RustyDrive() {
  const { mode } = useTheme();
  const [layout, setLayout] = useState<keyof typeof LAYOUTS>('easy');
  const [driver, setDriver] = useState<'scripted' | 'manual'>('scripted');
  const [playing, setPlaying] = useState(true);
  const [pose, setPose] = useState({ x: 45, y: 150, th: 0 });
  const [trail, setTrail] = useState<Array<{ x: number; y: number }>>([]);
  const [steps, setSteps] = useState(0);
  const [reached, setReached] = useState(false);
  const [stuck, setStuck] = useState(false);
  const keys = useRef<Set<string>>(new Set());
  const raf = useRef<number | null>(null);
  const progress = useRef<{ best: number; since: number }>({ best: Infinity, since: 0 });

  const obstacles = LAYOUTS[layout];

  const reset = () => {
    setPose({ x: 45, y: 150, th: 0 });
    setTrail([]);
    setSteps(0);
    setReached(false);
    setStuck(false);
    progress.current = { best: Infinity, since: 0 };
  };

  useEffect(() => {
    reset();
  }, [layout, driver]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        keys.current.add(e.key);
      }
    };
    const up = (e: KeyboardEvent) => keys.current.delete(e.key);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);

  useEffect(() => {
    if (!playing || reached) return;
    let mounted = true;

    const tick = () => {
      if (!mounted) return;
      setPose((p) => {
        let v = 0;
        let omega = 0;

        if (driver === 'manual') {
          if (keys.current.has('ArrowUp')) v = 2.2;
          if (keys.current.has('ArrowDown')) v = -1.4;
          if (keys.current.has('ArrowLeft')) omega = -0.07;
          if (keys.current.has('ArrowRight')) omega = 0.07;
        } else {
          // --- The hand-coded controller ------------------------------------
          // Attractive term: steer toward the goal.
          const dxg = GOAL.x - p.x;
          const dyg = GOAL.y - p.y;
          const goalAngle = Math.atan2(dyg, dxg);
          let err = goalAngle - p.th;
          while (err > Math.PI) err -= 2 * Math.PI;
          while (err < -Math.PI) err += 2 * Math.PI;

          // Repulsive term: veer away from anything within the safety radius.
          let repulse = 0;
          let minDist = Infinity;
          for (const o of obstacles) {
            const dx = p.x - o.x;
            const dy = p.y - o.y;
            const d = Math.hypot(dx, dy) - o.r;
            minDist = Math.min(minDist, d);
            if (d < 55) {
              const away = Math.atan2(dy, dx);
              let diff = away - p.th;
              while (diff > Math.PI) diff -= 2 * Math.PI;
              while (diff < -Math.PI) diff += 2 * Math.PI;
              repulse += (1.6 * diff) / Math.max(d, 6);
            }
          }

          omega = 0.06 * err + repulse * 0.5;
          omega = Math.max(-0.13, Math.min(0.13, omega));
          v = minDist < 16 ? 0.7 : 2.1;
        }

        const th = p.th + omega;
        let x = p.x + v * Math.cos(th);
        let y = p.y + v * Math.sin(th);

        // Walls and obstacles are hard: Rusty slides along them.
        x = Math.max(14, Math.min(W - 14, x));
        y = Math.max(14, Math.min(H - 14, y));
        for (const o of obstacles) {
          const dx = x - o.x;
          const dy = y - o.y;
          const d = Math.hypot(dx, dy);
          if (d < o.r + 11) {
            const push = (o.r + 11 - d) / Math.max(d, 0.01);
            x += dx * push;
            y += dy * push;
          }
        }

        const distToGoal = Math.hypot(GOAL.x - x, GOAL.y - y);
        if (distToGoal < GOAL.r + 8) setReached(true);

        // Local-minimum detector: no progress toward the goal for a long while.
        if (distToGoal < progress.current.best - 1) {
          progress.current.best = distToGoal;
          progress.current.since = 0;
        } else {
          progress.current.since += 1;
          if (progress.current.since > 240 && driver === 'scripted') setStuck(true);
        }

        setTrail((t) => {
          const next = [...t, { x, y }];
          return next.length > 900 ? next.slice(-900) : next;
        });
        setSteps((s) => s + 1);
        return { x, y, th };
      });
      raf.current = requestAnimationFrame(tick);
    };

    raf.current = requestAnimationFrame(tick);
    return () => {
      mounted = false;
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, driver, obstacles, reached]);

  const trailD = trail.length > 1 ? `M${trail.map((p) => `${p.x},${p.y}`).join(' L')}` : '';

  return (
    <SimPanel
      title="Drive Rusty — the case for learning, in thirty seconds"
      id="ch01-drive-rusty"
      subtitle="A hand-written potential-field controller: steer toward the goal, repel from obstacles."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Segmented
            label="Room"
            value={layout}
            onChange={(v) => setLayout(v)}
            options={[
              { value: 'easy', label: 'Empty' },
              { value: 'cluttered', label: 'Cluttered' },
              { value: 'trap', label: 'Concave trap' },
            ]}
          />
          <Segmented
            label="Driver"
            value={driver}
            onChange={setDriver}
            options={[
              { value: 'scripted', label: 'Hand-coded' },
              { value: 'manual', label: 'You (arrow keys)' },
            ]}
          />
          <SimControls
            playing={playing}
            onPlayPause={() => setPlaying((p) => !p)}
            onReset={() => {
              reset();
              setPlaying(true);
            }}
          />
        </div>
      }
      caption="The controller is not badly written — it is the textbook potential-field method, and in the empty room it is optimal. Add clutter and it wobbles; build a concave trap and it parks itself in a local minimum and stays there. You can drive out of the trap instantly, because you can see that backing up is progress. Encoding that judgment by hand, for every room, is the job reinforcement learning proposes to automate."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,200px]">
        <svg
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          className="max-w-full rounded-lg"
          style={{ background: 'var(--surface-sunken)' }}
          role="img"
          aria-label="Top-down view of Rusty navigating a room toward a goal"
        >
          <rect x={1} y={1} width={W - 2} height={H - 2} fill="none" stroke="var(--baseline)" rx={8} />

          {obstacles.map((o, i) => (
            <circle
              key={i}
              cx={o.x}
              cy={o.y}
              r={o.r}
              fill={mode === 'light' ? '#d8d7d0' : '#2c2c2a'}
              stroke="var(--baseline)"
            />
          ))}

          <circle
            cx={GOAL.x}
            cy={GOAL.y}
            r={GOAL.r}
            fill="color-mix(in srgb, var(--status-good) 22%, transparent)"
            stroke="var(--status-good)"
            strokeWidth={2}
          />
          <text
            x={GOAL.x}
            y={GOAL.y + 4}
            textAnchor="middle"
            fontSize={11}
            fontWeight={600}
            fill="var(--status-good)"
          >
            goal
          </text>

          {trailD && (
            <path d={trailD} fill="none" stroke={seriesColor(1, mode)} strokeWidth={2} opacity={0.75} />
          )}

          <g transform={`translate(${pose.x},${pose.y}) rotate(${(pose.th * 180) / Math.PI})`}>
            <circle r={11} fill={seriesColor(0, mode)} stroke="var(--surface-1)" strokeWidth={2} />
            <line x1={0} y1={0} x2={16} y2={0} stroke={seriesColor(0, mode)} strokeWidth={2.5} />
          </g>
        </svg>

        <div className="space-y-2">
          <StatTile label="Steps taken" value={steps} hint="control cycles elapsed" />
          <StatTile
            label="Distance to goal"
            value={Math.hypot(GOAL.x - pose.x, GOAL.y - pose.y)}
            unit="px"
          />
          <StatTile
            label="Outcome"
            value={reached ? 'Reached' : stuck ? 'Stuck' : 'Driving'}
            status={reached ? 'good' : stuck ? 'critical' : undefined}
            mono={false}
            hint={
              reached
                ? 'goal reached'
                : stuck
                  ? 'local minimum — no progress'
                  : 'in progress'
            }
          />
          {driver === 'manual' && (
            <p className="rounded-lg border border-hairline px-2.5 py-2 text-[11.5px] leading-snug text-ink-muted">
              Click the diagram, then steer with the arrow keys.
            </p>
          )}
        </div>
      </div>
    </SimPanel>
  );
}
