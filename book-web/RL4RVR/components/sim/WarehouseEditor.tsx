'use client';

import { useCallback, useMemo, useState } from 'react';
import { GridWorld, WAREHOUSE_MAP, ACTION_ARROWS, type Action } from '@/lib/rl/gridworld';
import { collect, valueIteration } from '@/lib/rl/dp';
import { mulberry32 } from '@/lib/rl/random';
import { GridWorldCanvas, ValueLegend } from './GridWorldCanvas';
import { Segmented, SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { useWidgetState } from '@/lib/sim/useSimulation';
import { cn } from '@/lib/utils';

type Tool = 'shelf' | 'slip' | 'dock' | 'start';

const ROWS = WAREHOUSE_MAP.length;
const COLS = WAREHOUSE_MAP[0].length;

/** The map as a flat string, so a whole warehouse fits in a URL. */
function encodeMap(map: string[], slippery: Set<number>): string {
  const cells = map.join('').split('');
  slippery.forEach((s) => {
    if (cells[s] === '.') cells[s] = '~';
  });
  return cells.join('');
}

function decodeMap(encoded: string): { map: string[]; slippery: Set<number> } {
  const slippery = new Set<number>();
  const cells = encoded.split('').map((c, i) => {
    if (c === '~') {
      slippery.add(i);
      return '.';
    }
    return c;
  });
  const map: string[] = [];
  for (let r = 0; r < ROWS; r++) map.push(cells.slice(r * COLS, (r + 1) * COLS).join(''));
  return { map, slippery };
}

const DEFAULTS = {
  gamma: 0.95,
  pSlip: 0.2,
  slipPatch: 0.6,
  tool: 'shelf' as string,
  map: encodeMap(WAREHOUSE_MAP, new Set()),
};

/**
 * `ch04-warehouse-editor` — the MDP as something you author.
 *
 * Chapter 4 says an MDP is a modelling choice rather than a fact about the
 * robot. This is that claim made operable: wall off a corridor, move the dock,
 * paint a slippery patch, and value iteration re-solves immediately. The
 * optimal policy visibly reroutes — not because a parameter changed, but
 * because the *problem* did.
 *
 * Solving an 80-state MDP is microseconds, so this runs on the main thread and
 * responds within a frame of every click.
 */
export function WarehouseEditor() {
  const [state, set, reset] = useWidgetState('ch04-warehouse-editor', DEFAULTS);
  const [hoverPath, setHoverPath] = useState<number[] | null>(null);

  const { map, slippery } = useMemo(() => decodeMap(state.map), [state.map]);
  const tool = state.tool as Tool;

  const env = useMemo(() => {
    const slipFn = (s: number) => (slippery.has(s) ? state.slipPatch : undefined);
    return new GridWorld(map, { gamma: state.gamma, pSlip: state.pSlip }, slipFn);
  }, [map, slippery, state.gamma, state.pSlip, state.slipPatch]);

  const solution = useMemo(() => {
    const snaps = collect(valueIteration(env, 1e-5, 300));
    return snaps[snaps.length - 1];
  }, [env]);

  /** Is the dock reachable at all? A walled-off dock is a legal MDP too. */
  const reachable = useMemo(() => {
    const seen = new Set<number>([env.startState]);
    const queue = [env.startState];
    while (queue.length) {
      const s = queue.shift()!;
      if (s === env.goalState) return true;
      for (let a = 0 as Action; a < 4; a = (a + 1) as Action) {
        for (const t of env.transitions(s, a)) {
          if (!seen.has(t.next)) {
            seen.add(t.next);
            queue.push(t.next);
          }
        }
      }
    }
    return false;
  }, [env]);

  const rollout = useMemo(
    () => env.rollout(solution.policy, mulberry32(11), 300),
    [env, solution],
  );

  const vRange = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, solution.V[s]);
      hi = Math.max(hi, solution.V[s]);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi: Number.isFinite(hi) ? hi : 1 };
  }, [env, solution]);

  const onCell = useCallback(
    (s: number) => {
      // Computed from the previous state rather than a captured one, so
      // dragging across many cells records every edit instead of only the last.
      set((prev) => {
        const cells = prev.map.split('');
        const current = cells[s];

        if (tool === 'shelf') {
          // The dock and the start must stay on the floor.
          if (current === 'D' || current === 'S') return {};
          cells[s] = current === '#' ? '.' : '#';
        } else if (tool === 'slip') {
          if (current === '#') return {};
          cells[s] = current === '~' ? '.' : current === '.' ? '~' : current;
        } else if (tool === 'dock') {
          if (current === '#') return {};
          const old = cells.indexOf('D');
          if (old >= 0) cells[old] = '.';
          cells[s] = 'D';
        } else if (tool === 'start') {
          if (current === '#' || current === 'D') return {};
          const old = cells.indexOf('S');
          if (old >= 0) cells[old] = '.';
          cells[s] = 'S';
        }
        return { map: cells.join('') };
      });
    },
    [tool, set],
  );

  const slipCount = slippery.size;
  const shelfCount = state.map.split('').filter((c) => c === '#').length;

  return (
    <SimPanel
      title="Author the warehouse"
      id="ch04-warehouse-editor"
      subtitle="Wall off a corridor, move the dock, paint a slippery patch. Value iteration re-solves on every edit."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Tool"
              value={tool}
              onChange={(v) => set({ tool: v })}
              options={[
                { value: 'shelf', label: 'Shelf' },
                { value: 'slip', label: 'Slippery' },
                { value: 'dock', label: 'Dock' },
                { value: 'start', label: 'Start' },
              ]}
            />
            <button
              type="button"
              onClick={reset}
              className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              Restore the book&apos;s warehouse
            </button>
            <button
              type="button"
              onClick={() => {
                const cells = WAREHOUSE_MAP.join('').split('').map((c) => (c === '#' ? '.' : c));
                set({ map: cells.join('') });
              }}
              className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              Clear all shelves
            </button>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Slider
              label="Discount γ"
              value={state.gamma}
              min={0.5}
              max={0.995}
              step={0.005}
              onChange={(v) => set({ gamma: v })}
              hint={`horizon ≈ ${(1 / (1 - state.gamma)).toFixed(0)} steps`}
            />
            <Slider
              label="Floor slip p_slip"
              value={state.pSlip}
              min={0}
              max={0.6}
              step={0.05}
              onChange={(v) => set({ pSlip: v })}
              hint="everywhere unpainted"
            />
            <Slider
              label="Painted-patch slip"
              value={state.slipPatch}
              min={0}
              max={0.9}
              step={0.05}
              onChange={(v) => set({ slipPatch: v })}
              hint={`${slipCount} cell${slipCount === 1 ? '' : 's'} painted`}
            />
          </div>
        </div>
      }
      caption="Wall the corridor beside the dock and watch the whole value landscape re-route — the arrows change everywhere, not just at the wall, because value propagates. Paint a slippery patch across the shortest path and the optimal policy will detour around it exactly when the detour is cheaper than the expected cost of slipping. Both are the same lesson: the policy is a consequence of the model, and you just edited the model."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          <div className="relative">
            <GridWorldCanvas
              env={env}
              V={solution.V}
              policy={solution.policy}
              path={hoverPath ?? rollout.path}
              cellSize={34}
              onCellClick={onCell}
              highlight={[...slippery]}
              dragPaint
            />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <ValueLegend min={vRange.lo} max={vRange.hi} />
            <span className="flex items-center gap-1.5 text-[11px] text-ink-muted">
              <span
                className="inline-block h-3 w-3 rounded-sm border-2"
                style={{ borderColor: 'var(--series-4)' }}
                aria-hidden
              />
              slippery patch
            </span>
          </div>
          <p className="mt-1.5 text-[11.5px] leading-snug text-ink-muted">
            Click or <strong className="text-ink-secondary">drag</strong> across cells with the{' '}
            <strong className="text-ink-secondary">{tool}</strong> tool selected. The orange path is a rollout of the current optimal policy.
          </p>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <StatTile
              label="Dock reachable"
              value={reachable ? 'Yes' : 'No'}
              mono={false}
              status={reachable ? 'good' : 'critical'}
              hint={reachable ? 'a path exists' : 'you have walled it off'}
            />
            <StatTile
              label="Sweeps to converge"
              value={solution.sweep}
              hint="value iteration, θ = 1e−5"
            />
            <StatTile
              label="v* at the start"
              value={solution.V[env.startState]}
              hint="expected discounted return"
            />
            <StatTile
              label="Rollout return"
              value={rollout.totalReward}
              status={rollout.totalReward > 0 ? 'good' : 'warning'}
              hint={`${rollout.path.length} steps`}
            />
            <StatTile label="Free cells" value={env.states.length} hint={`${shelfCount} shelves`} />
            <StatTile
              label="Slippery cells"
              value={slipCount}
              hint={slipCount ? `p = ${state.slipPatch.toFixed(2)} there` : 'none painted'}
            />
          </div>

          <div className="rounded-lg border border-hairline bg-surface-sunken px-3 py-2.5">
            <p className="mb-1.5 text-[11.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              Optimal action, cell by cell
            </p>
            <p className="text-[12.5px] leading-relaxed text-ink-secondary">
              The arrows are <code className="font-mono text-[11.5px]">argmax_a q*(s,a)</code>,
              recomputed from scratch after every edit. Nothing is cached and nothing is
              interpolated — this is the Bellman optimality equation solved on the warehouse you
              just drew.
            </p>
            {!reachable && (
              <p className="mt-2 rounded border border-hairline bg-surface px-2 py-1.5 text-[12px] text-status-critical">
                The dock is unreachable. The MDP is still well defined — every policy simply has
                the same, bad value, so the arrows below are arbitrary among equals.
              </p>
            )}
          </div>

          <div className="rounded-lg border border-hairline bg-surface px-3 py-2.5">
            <p className="mb-1 text-[11.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              Try this
            </p>
            <ul className="space-y-1 text-[12.5px] leading-snug text-ink-secondary">
              <li>
                <strong className="text-ink">Trap the robot.</strong> Wall Rusty into a pocket and
                watch v* collapse to the step-cost floor everywhere inside it.
              </li>
              <li>
                <strong className="text-ink">Build a shortcut.</strong> Clear a shelf block and see
                how far the value change propagates from that one cell.
              </li>
              <li>
                <strong className="text-ink">Make a river.</strong> Paint a slippery line across the
                map, then raise the patch slip until the policy prefers the long way round.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}
