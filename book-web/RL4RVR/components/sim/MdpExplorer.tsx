'use client';

import { useMemo, useState } from 'react';
import { GridWorld, ACTION_ARROWS, type Action } from '@/lib/rl/gridworld';
import { valueIteration, collect } from '@/lib/rl/dp';
import { GridWorldCanvas, ValueLegend } from './GridWorldCanvas';
import { SimPanel, Slider, Segmented } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';

/**
 * `ch04-mdp-editor` — the warehouse as a formal object you can poke.
 *
 * Click a cell to inspect the full distribution p(s′, r | s, a) for each
 * action: the fan of arrows *is* the four-argument dynamics function, and the
 * numbers beside them are the probabilities the Bellman equations sum over.
 * Changing γ or p_slip re-solves the MDP live, so the reader sees the optimal
 * policy respond to the model rather than to a hyperparameter.
 */
export function MdpExplorer() {
  const [gamma, setGamma] = useState(0.95);
  const [pSlip, setPSlip] = useState(0.2);
  const [action, setAction] = useState<Action>(1);
  const [selected, setSelected] = useState<number | null>(null);

  const env = useMemo(() => new GridWorld(undefined, { gamma, pSlip }), [gamma, pSlip]);
  const solution = useMemo(() => {
    const snaps = collect(valueIteration(env, 1e-5, 400));
    return snaps[snaps.length - 1];
  }, [env]);

  const state = selected ?? env.startState;
  const transitions = env.isTerminal(state) ? [] : env.transitions(state, action);
  const qValues = env.isTerminal(state)
    ? [0, 0, 0, 0]
    : ([0, 1, 2, 3] as Action[]).map((a) => env.actionValue(state, a, solution.V));

  const vRange = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, solution.V[s]);
      hi = Math.max(hi, solution.V[s]);
    }
    return { lo, hi };
  }, [solution, env]);

  return (
    <SimPanel
      title="The warehouse as an MDP"
      id="ch04-mdp-editor"
      subtitle="Click any floor cell to read its dynamics. The heatmap shows v*, the arrows show the greedy policy π*."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            className="w-44"
            label="Discount γ"
            value={gamma}
            min={0.5}
            max={0.995}
            step={0.005}
            onChange={setGamma}
            hint={`horizon ≈ ${(1 / (1 - gamma)).toFixed(0)} steps`}
          />
          <Slider
            className="w-44"
            label="Slip p_slip"
            value={pSlip}
            min={0}
            max={0.6}
            step={0.05}
            onChange={setPSlip}
          />
          <Segmented
            label="Inspect action"
            value={String(action) as '0' | '1' | '2' | '3'}
            onChange={(v) => setAction(Number(v) as Action)}
            options={[
              { value: '0', label: '↑ N' },
              { value: '1', label: '→ E' },
              { value: '2', label: '↓ S' },
              { value: '3', label: '← W' },
            ]}
          />
        </div>
      }
      caption="With p_slip = 0 the dynamics are deterministic and the arrows take the shortest path, hugging the shelves. Raise the slip and the optimal policy backs away from the shelving — a −10 bump that happens 10% of the time outweighs the step saved. Nothing about the algorithm changed; the model did."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          <GridWorldCanvas
            env={env}
            V={solution.V}
            policy={solution.policy}
            cellSize={34}
            onCellClick={setSelected}
            highlight={[state]}
          />
          <div className="mt-2">
            <ValueLegend min={vRange.lo} max={vRange.hi} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <StatTile label="Selected state" value={`s = ${state}`} mono />
            <StatTile
              label="v*(s)"
              value={solution.V[state]}
              hint={env.isTerminal(state) ? 'terminal' : 'optimal value'}
            />
          </div>

          <div className="rounded-lg border border-hairline bg-surface-sunken px-3 py-2.5">
            <p className="mb-2 text-[11.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              p(s′, r | s, a = {ACTION_ARROWS[action]})
            </p>
            {transitions.length === 0 ? (
              <p className="text-[12.5px] text-ink-muted">Terminal state — the episode ends here.</p>
            ) : (
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="text-ink-muted">
                    <th className="py-1 text-left font-medium">next s′</th>
                    <th className="py-1 text-right font-medium">reward r</th>
                    <th className="py-1 text-right font-medium">probability</th>
                  </tr>
                </thead>
                <tbody>
                  {transitions.map((t, i) => (
                    <tr key={i} className="border-t border-hairline">
                      <td className="tabular py-1 text-ink">
                        {t.next === state ? `${t.next} (blocked)` : t.next}
                      </td>
                      <td className="tabular py-1 text-right text-ink-secondary">{t.reward}</td>
                      <td className="tabular py-1 text-right font-semibold text-ink">
                        {t.prob.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                  <tr className="border-t border-baseline">
                    <td className="py-1 text-[11px] text-ink-muted" colSpan={2}>
                      sums to
                    </td>
                    <td className="tabular py-1 text-right text-[11px] text-ink-muted">
                      {transitions.reduce((a, t) => a + t.prob, 0).toFixed(2)}
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
          </div>

          <div className="rounded-lg border border-hairline bg-surface-sunken px-3 py-2.5">
            <p className="mb-2 text-[11.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              q*(s, a) — one-step lookahead
            </p>
            <div className="grid grid-cols-4 gap-2">
              {qValues.map((q, a) => {
                const best = q === Math.max(...qValues);
                return (
                  <div
                    key={a}
                    className="rounded-md border px-2 py-1.5 text-center"
                    style={{
                      borderColor: best ? 'var(--series-1)' : 'var(--border-hairline)',
                      background: best
                        ? 'color-mix(in srgb, var(--series-1) 10%, transparent)'
                        : 'transparent',
                    }}
                  >
                    <div className="text-[14px] text-ink">{ACTION_ARROWS[a]}</div>
                    <div className="tabular text-[11.5px] font-semibold text-ink-secondary">
                      {q.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-2 text-[11px] leading-snug text-ink-muted">
              The greedy action is the argmax — this is exactly the operation the Bellman optimality
              equation performs at every state.
            </p>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}
