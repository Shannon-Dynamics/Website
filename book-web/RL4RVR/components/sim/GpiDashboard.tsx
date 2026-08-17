'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { GridWorld, type Action } from '@/lib/rl/gridworld';
import { collect, policyIteration, valueIteration, type DpSnapshot } from '@/lib/rl/dp';
import { policyEvaluation } from '@/lib/rl/dp';
import { GridWorldCanvas, ValueLegend } from './GridWorldCanvas';
import { Segmented, SimControls, SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { LineChart } from '@/components/viz/LineChart';

type Mode = 'policy-eval' | 'policy-iteration' | 'value-iteration';

/**
 * `ch05-gpi-dashboard` — the book's signature widget.
 *
 * Generalized policy iteration made watchable: the value heatmap and the greedy
 * policy arrows update sweep by sweep on Rusty's warehouse, with the
 * convergence measure Δ and the implied suboptimality bound streaming beside
 * them. Every quantity on screen is one the math named.
 */
export function GpiDashboard() {
  const [mode, setMode] = useState<Mode>('value-iteration');
  const [gamma, setGamma] = useState(0.95);
  const [pSlip, setPSlip] = useState(0.2);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [index, setIndex] = useState(0);
  const raf = useRef<number | null>(null);
  const acc = useRef(0);
  /**
   * Values the reader has corrupted by hand. GPI is self-correcting, and the
   * most direct way to show that is to let someone break it and watch the
   * subsequent sweeps repair the damage.
   */
  const [damage, setDamage] = useState<Map<number, number>>(new Map());
  const [damageAt, setDamageAt] = useState<number | null>(null);

  const env = useMemo(() => new GridWorld(undefined, { gamma, pSlip }), [gamma, pSlip]);

  const snapshots: DpSnapshot[] = useMemo(() => {
    if (mode === 'value-iteration') return collect(valueIteration(env, 1e-4, 200));
    if (mode === 'policy-iteration') return collect(policyIteration(env, 1e-4, 30));
    // Policy evaluation of a fixed "always east" policy — the widget's warm-up.
    const fixed = new Int8Array(env.nStates).fill(-1);
    for (const s of env.states) if (!env.isTerminal(s)) fixed[s] = 1;
    return collect(policyEvaluation(env, fixed, 1e-4, 200));
  }, [env, mode]);

  useEffect(() => {
    setIndex(0);
    setDamage(new Map());
    setDamageAt(null);
  }, [mode, gamma, pSlip]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      acc.current += (now - last) * (speed / 1000);
      last = now;
      if (acc.current >= 1) {
        const advance = Math.floor(acc.current);
        acc.current -= advance;
        setIndex((i) => {
          const next = i + advance;
          if (next >= snapshots.length - 1) {
            setPlaying(false);
            return snapshots.length - 1;
          }
          return next;
        });
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, speed, snapshots.length]);

  const rawSnap = snapshots[Math.min(index, snapshots.length - 1)];

  // Re-run the Bellman operator forward from the damaged values so the healing
  // shown is the algorithm's own, not an animation of one.
  const snap = useMemo(() => {
    if (!rawSnap || damage.size === 0 || damageAt === null) return rawSnap;
    const base = snapshots[Math.min(damageAt, snapshots.length - 1)];
    const V = Float64Array.from(base.V);
    damage.forEach((v, s) => {
      V[s] = v;
    });
    const sweepsSince = Math.max(0, index - damageAt);
    for (let k = 0; k < sweepsSince; k++) {
      for (const s of env.states) {
        if (env.isTerminal(s)) continue;
        let best = -Infinity;
        for (let a = 0 as Action; a < 4; a = (a + 1) as Action) {
          best = Math.max(best, env.actionValue(s, a, V));
        }
        V[s] = best;
      }
    }
    let delta = 0;
    for (const s of env.states) delta = Math.max(delta, Math.abs(V[s] - rawSnap.V[s]));
    return { ...rawSnap, V, policy: env.greedyPolicy(V), delta: sweepsSince === 0 ? rawSnap.delta : delta };
  }, [rawSnap, damage, damageAt, snapshots, index, env]);
  const vRange = useMemo(() => {
    if (!snap) return { lo: 0, hi: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of env.states) {
      if (env.isTerminal(s)) continue;
      lo = Math.min(lo, snap.V[s]);
      hi = Math.max(hi, snap.V[s]);
    }
    return { lo: Number.isFinite(lo) ? lo : 0, hi: Number.isFinite(hi) ? hi : 1 };
  }, [snap, env]);

  const deltaSeries = useMemo(
    () => [
      {
        id: 'Δ (max value change)',
        data: snapshots
          .slice(0, index + 1)
          .map((s, i) => ({ x: i + 1, y: Math.max(s.delta, 1e-6) })),
      },
    ],
    [snapshots, index],
  );

  if (!snap) return null;

  return (
    <SimPanel
      title="Generalized policy iteration, sweep by sweep"
      id="ch05-gpi-dashboard"
      subtitle="Rusty's warehouse: value heatmap + greedy policy arrows, with the convergence measure the theory names."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Algorithm"
              value={mode}
              onChange={(v) => setMode(v)}
              options={[
                { value: 'policy-eval', label: 'Policy evaluation' },
                { value: 'policy-iteration', label: 'Policy iteration' },
                { value: 'value-iteration', label: 'Value iteration' },
              ]}
            />
            <SimControls
              playing={playing}
              onPlayPause={() => setPlaying((p) => !p)}
              onStep={() => setIndex((i) => Math.min(i + 1, snapshots.length - 1))}
              onReset={() => {
                setPlaying(false);
                setIndex(0);
              }}
              speed={speed}
              onSpeedChange={setSpeed}
            />
            {damage.size > 0 && (
              <button
                type="button"
                onClick={() => {
                  setDamage(new Map());
                  setDamageAt(null);
                }}
                className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
              >
                Undo damage ({damage.size})
              </button>
            )}
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Slider
              label="Discount γ"
              value={gamma}
              min={0.5}
              max={0.995}
              step={0.005}
              onChange={setGamma}
              hint={`effective horizon ≈ ${(1 / (1 - gamma)).toFixed(0)} steps`}
            />
            <Slider
              label="Slip probability p_slip"
              value={pSlip}
              min={0}
              max={0.6}
              step={0.05}
              onChange={setPSlip}
              hint="lateral slip splits evenly"
            />
            <Slider
              label="Sweep"
              value={index}
              min={0}
              max={Math.max(snapshots.length - 1, 1)}
              step={1}
              onChange={(v) => {
                setPlaying(false);
                setIndex(v);
              }}
              format={(v) => `${v + 1} / ${snapshots.length}`}
            />
          </div>
        </div>
      }
      caption="Click any cell to smash its value to nonsense, then step forward: the next sweeps pull it back, and the damage bleeds outward one cell per sweep before disappearing. Nothing special handles the repair — it is the same Bellman backup running, which is exactly what &ldquo;self-correcting&rdquo; means. Drag γ toward 1 and watch value spread further from the dock before decaying — the effective horizon 1/(1−γ) made visible. Raise p_slip and the optimal policy stops hugging the shelves, because a slip into a shelf costs −10."
    >
      <div className="grid gap-4 lg:grid-cols-[auto,1fr]">
        <div>
          <GridWorldCanvas
            env={env}
            onCellClick={(cell) => {
              if (env.isTerminal(cell)) return;
              setPlaying(false);
              setDamage((prev) => {
                const next = new Map(prev);
                // Slam the value far from anything the operator would produce.
                next.set(cell, snap.V[cell] - 60);
                return next;
              });
              setDamageAt(index);
            }}
            highlight={[...damage.keys()]}
            V={snap.V}
            policy={snap.policy}
            showPolicy
            cellSize={34}
          />
          <div className="mt-2">
            <ValueLegend min={vRange.lo} max={vRange.hi} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <StatTile
              label="Sweep"
              value={`${snap.sweep}`}
              hint={snap.phase === 'improvement' ? 'policy improvement' : 'policy evaluation'}
            />
            <StatTile
              label="Δ = max|Vₖ₊₁ − Vₖ|"
              value={snap.delta}
              status={snap.delta < 1e-4 ? 'good' : undefined}
              hint={snap.delta < 1e-4 ? 'converged' : 'still changing'}
            />
            <StatTile
              label="Suboptimality bound"
              value={snap.suboptimalityBound}
              hint="2γΔ/(1−γ) ≥ ‖v_π − v*‖∞"
            />
            <StatTile
              label="States swept"
              value={env.states.length}
              hint={`${env.rows}×${env.cols} grid minus shelves`}
            />
          </div>

          <LineChart
            data={deltaSeries}
            height={190}
            xLegend="sweep"
            yLegend="Δ (log-ish)"
            caption="Δ contracts geometrically — the γ-contraction of the Bellman operator, measured."
          />
        </div>
      </div>
    </SimPanel>
  );
}
