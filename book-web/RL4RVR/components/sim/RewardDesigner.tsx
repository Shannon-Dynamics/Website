'use client';

import { useMemo } from 'react';
import {
  LAB_COLS,
  LAB_MAP,
  LAB_ROWS,
  NEUTRAL_TERMS,
  RewardLab,
  diagnose,
  type RewardTerms,
} from '@/lib/rl/rewardlab';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { useTheme } from '@/components/layout/ThemeProvider';
import { sequentialColor, seriesColor } from '@/lib/theme';
import { useWidgetState } from '@/lib/sim/useSimulation';
import { cn } from '@/lib/utils';

const ARROWS = ['↑', '→', '↓', '←'];

const PRESETS: Array<{ name: string; note: string; terms: Partial<RewardTerms> }> = {
  0: {
    name: 'Sparse',
    note: 'Goal bonus and step cost only. Honest, and the hardest to learn from.',
    terms: { proximity: 0, potential: 0, progress: 0 },
  },
  1: {
    name: 'Proximity bonus',
    note: 'Pay the agent for being near the goal. Reasonable-sounding.',
    terms: { proximity: 12, potential: 0, progress: 0 },
  },
  2: {
    name: 'Progress bonus',
    note: 'Pay for getting closer. Never charge for moving away.',
    terms: { proximity: 0, potential: 0, progress: 6 },
  },
  3: {
    name: 'Potential shaping',
    note: 'The same guidance, written the way Ng et al. prove is safe.',
    terms: { proximity: 0, potential: 4, progress: 0 },
  },
} as unknown as Array<{ name: string; note: string; terms: Partial<RewardTerms> }>;

const PRESET_LIST = [PRESETS[0], PRESETS[1], PRESETS[2], PRESETS[3]];

const DEFAULTS = {
  goal: 25,
  step: -1,
  hazard: -10,
  proximity: 0,
  potential: 0,
  progress: 0,
};

/**
 * `ch14-reward-designer` — the curse of goal specification, delivered personally.
 *
 * The reader composes a reward from named terms; value iteration then solves
 * that reward exactly and rolls out the optimal policy. Whatever happens is not
 * a training artifact — it is precisely what the stated objective asks for.
 *
 * The two exploits are reachable in one slider move each: a proximity bonus
 * makes the agent hover beside the goal rather than enter it (arriving ends the
 * payments), and an asymmetric progress bonus makes it oscillate forever.
 * Potential-based shaping supplies the same guidance and provably cannot do
 * either, which is the chapter's one piece of load-bearing theory.
 */
export function RewardDesigner() {
  const { mode } = useTheme();
  const [state, set, reset] = useWidgetState('ch14-reward-designer', DEFAULTS);

  const terms: RewardTerms = {
    goal: state.goal,
    step: state.step,
    hazard: state.hazard,
    proximity: state.proximity,
    potential: state.potential,
    progress: state.progress,
  };

  const { lab, solution, result, verdict } = useMemo(() => {
    const lab = new RewardLab(terms, 0.95);
    const solution = lab.solve();
    const result = lab.rollout(solution.policy);
    return { lab, solution, result, verdict: diagnose(terms, result) };
  }, [terms.goal, terms.step, terms.hazard, terms.proximity, terms.potential, terms.progress]);

  const vRange = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of lab.states) {
      if (lab.isTerminal(s)) continue;
      lo = Math.min(lo, solution.V[s]);
      hi = Math.max(hi, solution.V[s]);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi: Number.isFinite(hi) ? hi : 1 };
  }, [lab, solution]);

  const cell = 30;
  const W = LAB_COLS * cell;
  const H = LAB_ROWS * cell;

  const pathD =
    result.path.length > 1
      ? `M${result.path
          .map((s) => {
            const r = Math.floor(s / LAB_COLS);
            const c = s % LAB_COLS;
            return `${c * cell + cell / 2},${r * cell + cell / 2}`;
          })
          .join(' L')}`
      : '';

  return (
    <SimPanel
      title="Design a reward. Watch it get exploited."
      id="ch14-reward-designer"
      subtitle="Value iteration solves whatever reward you write, exactly. The behaviour below is not a training failure — it is the optimum of your objective."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11.5px] text-ink-secondary">Start from:</span>
            {PRESET_LIST.map((p, i) => (
              <button
                key={p.name}
                type="button"
                onClick={() => set({ ...DEFAULTS, ...p.terms })}
                title={p.note}
                className="rounded-md border border-hairline px-2 py-1 text-[11.5px] font-medium text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
              >
                {p.name}
              </button>
            ))}
            <button
              type="button"
              onClick={reset}
              className="rounded-md border border-hairline px-2 py-1 text-[11.5px] text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              Reset
            </button>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <Slider
              label="Goal bonus"
              value={state.goal}
              min={0}
              max={80}
              step={1}
              onChange={(v) => set({ goal: v })}
              format={(v) => `+${v.toFixed(0)}`}
              hint="paid once, on arrival"
            />
            <Slider
              label="Step cost"
              value={state.step}
              min={-5}
              max={0}
              step={0.25}
              onChange={(v) => set({ step: v })}
              hint="charged every step"
            />
            <Slider
              label="Hazard penalty"
              value={state.hazard}
              min={-40}
              max={0}
              step={1}
              onChange={(v) => set({ hazard: v })}
              hint="charged on entering a hazard"
            />
            <Slider
              label="Proximity bonus"
              value={state.proximity}
              min={0}
              max={30}
              step={0.5}
              onChange={(v) => set({ proximity: v })}
              hint="paid while near the goal"
            />
            <Slider
              label="Progress bonus"
              value={state.progress}
              min={0}
              max={15}
              step={0.25}
              onChange={(v) => set({ progress: v })}
              hint="paid for getting closer only"
            />
            <Slider
              label="Potential shaping"
              value={state.potential}
              min={0}
              max={15}
              step={0.25}
              onChange={(v) => set({ potential: v })}
              hint="γΦ(s′) − Φ(s), provably safe"
            />
          </div>
        </div>
      }
      caption="Push the proximity bonus past about 8 and the agent stops entering the goal: arriving ends the episode and ends the payments, so loitering beside it scores higher. Push the progress bonus past about 3 and it oscillates forever, farming the approach payment because nothing charges it for retreating. Now raise potential shaping as far as the slider goes — the route never changes, because Theorem 14.1 says it cannot. That is the difference between guidance and a new objective, and it is one line of algebra."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="max-w-full rounded-lg"
            role="img"
            aria-label="Gridworld with the optimal policy for the composed reward, and the resulting trajectory"
          >
            {lab.states.map((s) => {
              const r = Math.floor(s / LAB_COLS);
              const c = s % LAB_COLS;
              const ch = LAB_MAP[r][c];
              const t = (solution.V[s] - vRange.lo) / (vRange.hi - vRange.lo || 1);
              let fill = sequentialColor(t, mode);
              if (ch === 'H') fill = mode === 'light' ? '#f3c9c9' : '#4a2020';
              if (ch === 'G') fill = 'var(--status-good)';
              return (
                <g key={s}>
                  <rect
                    x={c * cell}
                    y={r * cell}
                    width={cell}
                    height={cell}
                    fill={fill}
                    stroke="var(--surface-1)"
                  />
                  {!lab.isTerminal(s) && solution.policy[s] >= 0 && (
                    <text
                      x={c * cell + cell / 2}
                      y={r * cell + cell / 2}
                      textAnchor="middle"
                      dominantBaseline="central"
                      fontSize={cell * 0.44}
                      fill={t > 0.55 ? '#ffffff' : 'var(--text-primary)'}
                      opacity={0.9}
                    >
                      {ARROWS[solution.policy[s]]}
                    </text>
                  )}
                  {ch === 'G' && (
                    <text
                      x={c * cell + cell / 2}
                      y={r * cell + cell / 2}
                      textAnchor="middle"
                      dominantBaseline="central"
                      fontSize={12}
                      fontWeight={700}
                      fill="#ffffff"
                    >
                      G
                    </text>
                  )}
                </g>
              );
            })}
            {pathD && (
              <path
                d={pathD}
                fill="none"
                stroke={seriesColor(1, mode)}
                strokeWidth={2.5}
                strokeLinecap="round"
                opacity={0.92}
              />
            )}
            <circle
              cx={(lab.start % LAB_COLS) * cell + cell / 2}
              cy={Math.floor(lab.start / LAB_COLS) * cell + cell / 2}
              r={6}
              fill={seriesColor(1, mode)}
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          </svg>
          <p className="mt-1.5 text-[11.5px] leading-snug text-ink-muted">
            Arrows are the optimal policy for <em>your</em> reward. The orange line is where the
            agent actually goes. Pink cells are the hazard.
          </p>
        </div>

        <div className="space-y-3">
          <div
            className="rounded-lg border px-3.5 py-3"
            style={{
              borderColor:
                verdict.status === 'good'
                  ? 'var(--status-good)'
                  : verdict.status === 'warning'
                    ? 'var(--status-warning)'
                    : 'var(--status-critical)',
              background:
                verdict.status === 'good'
                  ? 'color-mix(in srgb, var(--status-good) 8%, transparent)'
                  : verdict.status === 'warning'
                    ? 'color-mix(in srgb, var(--status-warning) 10%, transparent)'
                    : 'color-mix(in srgb, var(--status-critical) 8%, transparent)',
            }}
          >
            <p
              className="text-[13px] font-semibold"
              style={{
                color:
                  verdict.status === 'good'
                    ? 'var(--status-good)'
                    : verdict.status === 'warning'
                      ? 'var(--status-warning)'
                      : 'var(--status-critical)',
              }}
            >
              {verdict.verdict}
            </p>
            <p className="mt-1 text-[12.5px] leading-relaxed text-ink-secondary">{verdict.detail}</p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <StatTile
              label="Reached the goal"
              value={result.reached ? 'Yes' : 'No'}
              mono={false}
              status={result.reached ? 'good' : 'critical'}
              hint={result.reached ? `in ${result.steps} steps` : 'within 120 steps'}
            />
            <StatTile
              label="Your reward earned"
              value={result.designedReward}
              hint="what the agent optimized"
            />
            <StatTile
              label="Task reward earned"
              value={result.taskReward}
              status={result.taskReward > 0 ? 'good' : 'critical'}
              hint="what you actually wanted"
            />
            <StatTile
              label="Hazard cells entered"
              value={result.hazardHits}
              status={result.hazardHits > 0 ? 'warning' : 'good'}
            />
          </div>

          <div className="rounded-lg border border-hairline bg-surface-sunken px-3 py-2.5">
            <p className="mb-1 text-[11.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              The gap
            </p>
            <p className="text-[12.5px] leading-relaxed text-ink-secondary">
              When those two reward numbers disagree, you have found the gap between what you wrote
              and what you meant. The agent is not misbehaving; it is scoring well on the objective
              it was given and badly on the one in your head. Chapter 14 calls that the fourth
              curse, and it is the one no algorithm fixes.
            </p>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}
