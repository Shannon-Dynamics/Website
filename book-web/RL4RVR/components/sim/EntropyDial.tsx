'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor, sequentialColor } from '@/lib/theme';

/**
 * `ch11-entropy-dial` — maximum-entropy RL, made visible.
 *
 * SAC maximizes reward PLUS policy entropy, scaled by a temperature α. The
 * reader turns α and watches the optimal policy morph from a deterministic
 * spike (α → 0, standard RL) into a broad distribution that hedges across
 * every action worth considering. The Q-landscape underneath stays fixed —
 * only the temperature changes, which is what makes the effect legible.
 */
export function EntropyDial() {
  const { mode } = useTheme();
  const [alpha, setAlpha] = useState(0.4);
  const [bimodal, setBimodal] = useState(true);

  const { policy, qCurve, entropy, effectiveActions } = useMemo(() => {
    const N = 121;
    const actions = Array.from({ length: N }, (_, i) => -2 + (4 * i) / (N - 1));

    // A Q-landscape over a 1-D action: two good options (bimodal) or one.
    const q = actions.map((a) => {
      const peak1 = 1.0 * Math.exp(-((a + 0.85) ** 2) / 0.16);
      const peak2 = bimodal ? 0.92 * Math.exp(-((a - 0.95) ** 2) / 0.2) : 0;
      return peak1 + peak2;
    });

    // The max-entropy optimal policy is the Boltzmann distribution π ∝ exp(Q/α).
    const scaled = q.map((v) => v / Math.max(alpha, 1e-4));
    const maxS = Math.max(...scaled);
    const expv = scaled.map((v) => Math.exp(v - maxS));
    const Z = expv.reduce((a, b) => a + b, 0);
    const p = expv.map((v) => v / Z);

    // Differential entropy of the discretized policy, in nats.
    const da = actions[1] - actions[0];
    const H = -p.reduce((acc, pi) => acc + (pi > 1e-12 ? pi * Math.log(pi / da) : 0), 0);
    // Perplexity: roughly how many actions the policy is really spreading over.
    const perplexity = Math.exp(H) / da / actions.length * actions.length;

    return {
      policy: actions.map((a, i) => ({ x: a, y: p[i] / da })),
      qCurve: actions.map((a, i) => ({ x: a, y: q[i] })),
      entropy: H,
      effectiveActions: Math.min(actions.length, Math.max(1, perplexity)),
    };
  }, [alpha, bimodal]);

  const maxDensity = Math.max(...policy.map((p) => p.y));

  return (
    <SimPanel
      title="The entropy temperature"
      id="ch11-entropy-dial"
      subtitle="π*(a|s) ∝ exp(Q(s,a)/α) — the optimal maximum-entropy policy for a fixed Q-landscape."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            className="w-56"
            label="Temperature α"
            value={alpha}
            min={0.02}
            max={1.5}
            step={0.02}
            onChange={setAlpha}
            hint={
              alpha < 0.08
                ? 'α → 0 recovers standard, deterministic RL'
                : alpha > 1
                  ? 'exploration dominates — the reward barely matters'
                  : 'hedging across actions worth considering'
            }
          />
          <label className="flex items-center gap-2 text-[12px] text-ink-secondary">
            <input
              type="checkbox"
              checked={bimodal}
              onChange={(e) => setBimodal(e.target.checked)}
              className="accent-[var(--series-1)]"
            />
            Two good solutions (bimodal Q)
          </label>
        </div>
      }
      caption="With two nearly-equal options, a deterministic policy must commit to one and discards the other — and if the world shifts slightly, it has no fallback. The maximum-entropy policy keeps both alive in proportion to their value. This is why SAC explores well without an ε schedule, and why it degrades gracefully when the dynamics differ slightly from training."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,200px]">
        <div className="space-y-2">
          <LineChart
            data={[{ id: 'Q(s, a)', data: qCurve }]}
            height={140}
            xLegend="action a"
            yLegend="Q value"
            caption="The action-value landscape — fixed while you turn α."
          />
          <LineChart
            data={[{ id: 'π(a | s)', data: policy }]}
            height={165}
            xLegend="action a"
            yLegend="density"
            yMin={0}
          />
        </div>

        <div className="space-y-2">
          <StatTile
            label="Policy entropy H"
            value={entropy}
            unit="nats"
            hint={entropy < 0 ? 'near-deterministic' : 'spread over actions'}
          />
          <StatTile
            label="Peak density"
            value={maxDensity}
            hint="how sharply it commits"
          />
          <StatTile
            label="Effective actions"
            value={effectiveActions}
            hint="perplexity of the policy"
            status={effectiveActions > 2 ? 'good' : undefined}
          />
          <div className="rounded-lg border border-hairline p-2.5">
            <p className="mb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              Exploration breadth
            </p>
            <svg width="100%" height={26} viewBox="0 0 170 26" role="img" aria-label="Bar indicating how broadly the policy explores">
              {Array.from({ length: 20 }, (_, i) => (
                <rect
                  key={i}
                  x={i * 8.5}
                  y={4}
                  width={7}
                  height={18}
                  rx={2}
                  fill={
                    i / 19 < Math.min(1, entropy / 1.6 + 0.35)
                      ? sequentialColor(i / 19, mode)
                      : 'var(--gridline)'
                  }
                />
              ))}
            </svg>
            <p className="mt-1 text-[10.5px] leading-snug text-ink-muted">
              SAC tunes α automatically to hold entropy at a target — you rarely set it by hand.
            </p>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}
