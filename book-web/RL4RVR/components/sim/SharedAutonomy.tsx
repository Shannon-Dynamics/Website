'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { BarChart } from '@/components/viz/BarChart';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

interface Goal {
  x: number;
  y: number;
  label: string;
}

const GOALS: Goal[] = [
  { x: 330, y: 55, label: 'A' },
  { x: 355, y: 150, label: 'B' },
  { x: 320, y: 240, label: 'C' },
];

/**
 * `ch21-shared-autonomy` — assistance as inference over a hidden goal.
 *
 * The robot cannot see which target the human wants. It maintains a belief from
 * the direction of their input, and blends its own goal-directed action with
 * theirs in proportion to confidence. Turn the blending up and task performance
 * improves while the sense of control drops — the trade that user studies keep
 * finding and that no reward function in this book captures.
 */
export function SharedAutonomy() {
  const { mode } = useTheme();
  const [blend, setBlend] = useState(0.5);
  const [inputNoise, setInputNoise] = useState(0.35);
  const [trueGoal, setTrueGoal] = useState(1);
  const [useQFilter, setUseQFilter] = useState(false);

  const sim = useMemo(() => {
    const start = { x: 60, y: 150 };
    let pos = { ...start };
    // Uniform prior over goals.
    let belief = GOALS.map(() => 1 / GOALS.length);
    const path: Array<{ x: number; y: number }> = [{ ...start }];
    const beliefTrace: number[][] = [belief.slice()];

    // Deterministic pseudo-noise so the widget is stable while sliders move.
    let seed = 12345;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff - 0.5;
    };

    for (let t = 0; t < 120; t++) {
      const target = GOALS[trueGoal];

      // The human's intended direction, corrupted by interface noise.
      const dxh = target.x - pos.x + rand() * inputNoise * 220;
      const dyh = target.y - pos.y + rand() * inputNoise * 220;
      const nh = Math.hypot(dxh, dyh) || 1;
      const human = { x: dxh / nh, y: dyh / nh };

      // Belief update: goals whose direction matches the input gain probability.
      belief = belief.map((b, i) => {
        const dx = GOALS[i].x - pos.x;
        const dy = GOALS[i].y - pos.y;
        const n = Math.hypot(dx, dy) || 1;
        const alignment = (human.x * dx + human.y * dy) / n;   // cosine similarity
        return b * Math.exp(2.2 * alignment);
      });
      const z = belief.reduce((a, b) => a + b, 0);
      belief = belief.map((b) => b / z);

      // The robot's action: expected direction under the belief.
      let rx = 0;
      let ry = 0;
      belief.forEach((b, i) => {
        const dx = GOALS[i].x - pos.x;
        const dy = GOALS[i].y - pos.y;
        const n = Math.hypot(dx, dy) || 1;
        rx += b * (dx / n);
        ry += b * (dy / n);
      });

      // A Q-filter intervenes only when the human's input is clearly poor —
      // measured here as disagreement with the robot's preferred direction.
      const agreement = human.x * rx + human.y * ry;
      const effective = useQFilter ? (agreement > 0.55 ? 0 : blend) : blend;

      const ax = (1 - effective) * human.x + effective * rx;
      const ay = (1 - effective) * human.y + effective * ry;
      const na = Math.hypot(ax, ay) || 1;

      pos = { x: pos.x + (ax / na) * 3.1, y: pos.y + (ay / na) * 3.1 };
      path.push({ ...pos });
      beliefTrace.push(belief.slice());

      if (Math.hypot(target.x - pos.x, target.y - pos.y) < 14) break;
    }

    const target = GOALS[trueGoal];
    const finalError = Math.hypot(target.x - pos.x, target.y - pos.y);
    const reached = finalError < 16;
    const ideal = Math.hypot(target.x - start.x, target.y - start.y);
    const travelled = path.reduce(
      (acc, p, i) => (i === 0 ? 0 : acc + Math.hypot(p.x - path[i - 1].x, p.y - path[i - 1].y)),
      0,
    );

    return {
      path,
      belief,
      reached,
      finalError,
      efficiency: reached ? Math.min(1, ideal / Math.max(travelled, 1)) : 0,
      steps: path.length,
    };
  }, [blend, inputNoise, trueGoal, useQFilter]);

  // Agency is what the user feels: how much of the motion was theirs.
  const agency = useQFilter ? 1 - blend * 0.45 : 1 - blend;
  const taskScore = sim.reached ? 0.45 + 0.55 * sim.efficiency : 0.12;

  const comparison = [
    { id: 'task success', value: taskScore * 100 },
    { id: 'user agency', value: agency * 100 },
  ];

  const W = 420;
  const H = 300;
  const pathD = `M${sim.path.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L')}`;

  return (
    <SimPanel
      title="Shared autonomy: assistance that helps, and assistance that annoys"
      id="ch21-shared-autonomy"
      subtitle="The robot infers which target you want from noisy input, then blends its goal-directed action with yours."
      controls={
        <div className="space-y-2.5">
          <div className="grid gap-3 sm:grid-cols-2">
            <Slider
              label="Blending α (how much the robot takes over)"
              value={blend}
              min={0}
              max={1}
              step={0.02}
              onChange={setBlend}
              hint={blend < 0.2 ? 'you are driving' : blend > 0.8 ? 'the robot is driving' : 'shared'}
            />
            <Slider
              label="Input noise"
              value={inputNoise}
              min={0}
              max={0.8}
              step={0.02}
              onChange={setInputNoise}
              hint="tremor, poor interface, limited bandwidth"
            />
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-1.5">
              <span className="text-[11.5px] text-ink-secondary">Intended target:</span>
              {GOALS.map((g, i) => (
                <button
                  key={g.label}
                  type="button"
                  onClick={() => setTrueGoal(i)}
                  aria-pressed={trueGoal === i}
                  className="rounded-md border px-2 py-0.5 text-[11.5px] font-semibold transition-colors"
                  style={{
                    borderColor: trueGoal === i ? seriesColor(1, mode) : 'var(--border-hairline)',
                    background:
                      trueGoal === i
                        ? `color-mix(in srgb, ${seriesColor(1, mode)} 14%, transparent)`
                        : 'transparent',
                    color: trueGoal === i ? 'var(--text-primary)' : 'var(--text-secondary)',
                  }}
                >
                  {g.label}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
              <input
                type="checkbox"
                checked={useQFilter}
                onChange={(e) => setUseQFilter(e.target.checked)}
                className="accent-[var(--series-1)]"
              />
              Q-filter (assist only when the input is clearly poor)
            </label>
          </div>
        </div>
      }
      caption="At α = 0 the noise carries you off course; at α = 1 the robot drives and you are a passenger — and if its belief settles on the wrong target, you cannot correct it. The Q-filter is the practical compromise: it stays out of the way while you are doing fine and intervenes only when the input is clearly poor, which recovers most of the task benefit at a fraction of the agency cost."
    >
      <div className="grid gap-4 lg:grid-cols-[1fr,205px]">
        <div>
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="max-w-full rounded-lg"
            style={{ background: 'var(--surface-sunken)' }}
            role="img"
            aria-label="Cursor path from start toward one of three targets under shared control"
          >
            {GOALS.map((g, i) => (
              <g key={g.label}>
                <circle
                  cx={g.x}
                  cy={g.y}
                  r={16}
                  fill={
                    i === trueGoal
                      ? `color-mix(in srgb, ${seriesColor(1, mode)} 24%, transparent)`
                      : 'transparent'
                  }
                  stroke={i === trueGoal ? seriesColor(1, mode) : 'var(--baseline)'}
                  strokeWidth={i === trueGoal ? 2.5 : 1.5}
                />
                <text
                  x={g.x}
                  y={g.y + 4}
                  textAnchor="middle"
                  fontSize={12}
                  fontWeight={600}
                  fill={i === trueGoal ? seriesColor(1, mode) : 'var(--text-muted)'}
                >
                  {g.label}
                </text>
                {/* Belief bar beside each goal */}
                <rect
                  x={g.x + 24}
                  y={g.y - 7}
                  width={46 * sim.belief[i]}
                  height={13}
                  rx={3}
                  fill={seriesColor(0, mode)}
                  opacity={0.75}
                />
                <text x={g.x + 24} y={g.y + 20} fontSize={9} fill="var(--text-muted)">
                  {(sim.belief[i] * 100).toFixed(0)}%
                </text>
              </g>
            ))}

            <path d={pathD} fill="none" stroke={seriesColor(0, mode)} strokeWidth={2.5} />
            <circle cx={60} cy={150} r={5} fill="var(--text-muted)" />
            <circle
              cx={sim.path[sim.path.length - 1].x}
              cy={sim.path[sim.path.length - 1].y}
              r={7}
              fill={seriesColor(0, mode)}
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
            <text x={10} y={290} fontSize={9.5} fill="var(--text-muted)">
              bars show the robot&apos;s belief over which target you want
            </text>
          </svg>

          <BarChart
            data={comparison}
            layout="horizontal"
            height={140}
            xLegend="score (%)"
            colorByIndex
            title="The trade nobody has resolved"
          />
        </div>

        <div className="space-y-2">
          <StatTile
            label="Reached target"
            value={sim.reached ? 'Yes' : 'No'}
            mono={false}
            status={sim.reached ? 'good' : 'critical'}
            hint={`${sim.steps} control steps`}
          />
          <StatTile
            label="Path efficiency"
            value={sim.efficiency}
            hint="straight-line distance / travelled"
          />
          <StatTile
            label="Belief confidence"
            value={Math.max(...sim.belief)}
            hint="probability on the leading goal"
          />
          <StatTile
            label="User agency"
            value={agency}
            hint="fraction of motion that was yours"
            status={agency > 0.6 ? 'good' : agency > 0.3 ? 'warning' : 'critical'}
          />
        </div>
      </div>
    </SimPanel>
  );
}
