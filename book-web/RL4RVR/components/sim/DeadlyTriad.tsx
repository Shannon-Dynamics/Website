'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { cn } from '@/lib/utils';

/**
 * `ch08-deadly-triad` — Baird's counterexample, diverging live.
 *
 * The three ingredients (function approximation, bootstrapping, off-policy
 * training) are individually harmless. Together they can make the parameters
 * grow without bound even though the true value function is representable and
 * the rewards are all zero. The reader switches each ingredient off and finds
 * that removing ANY one restores stability — which is exactly the content of
 * the claim.
 */
export function DeadlyTriad() {
  const [approximation, setApproximation] = useState(true);
  const [bootstrapping, setBootstrapping] = useState(true);
  const [offPolicy, setOffPolicy] = useState(true);
  const [alpha, setAlpha] = useState(0.01);

  const { series, finalNorm, diverged } = useMemo(() => {
    // Baird's 7-state star MDP: every action leads to state 6 (the "hub").
    // True v = 0 everywhere (all rewards zero) and IS representable —
    // divergence is entirely an artifact of the update, not the problem.
    const N_STATES = 7;
    const N_W = 8;
    const GAMMA = 0.99;
    const STEPS = 900;

    // Feature vectors: φ(s) for the six outer states and the hub.
    const phi: number[][] = [];
    for (let s = 0; s < 6; s++) {
      const f = new Array(N_W).fill(0);
      f[s] = 2;
      f[7] = 1;
      phi.push(f);
    }
    const hub = new Array(N_W).fill(0);
    hub[6] = 1;
    hub[7] = 2;
    phi.push(hub);

    // Tabular fallback when function approximation is switched off: one
    // parameter per state, so features cannot interfere with each other.
    const tabularPhi = Array.from({ length: N_STATES }, (_, s) => {
      const f = new Array(N_W).fill(0);
      f[s] = 1;
      return f;
    });

    const features = approximation ? phi : tabularPhi;
    let w = new Array(N_W).fill(1);
    w[6] = 10; // the classic pathological initialization

    const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);
    const trace: Array<{ x: number; y: number }> = [];
    const wTrace: Array<{ x: number; y: number }> = [];

    for (let t = 0; t < STEPS; t++) {
      // Off-policy: update states uniformly (the target policy always goes to
      // the hub, but the behaviour distribution is uniform over all states).
      // On-policy: update the hub, which is where the target policy actually is.
      const s = offPolicy ? t % N_STATES : 6;
      const next = 6;

      const vS = dot(w, features[s]);
      const vNext = dot(w, features[next]);

      // Bootstrapping uses the current estimate of the successor; without it we
      // regress toward the true return, which for this MDP is exactly zero.
      const target = bootstrapping ? 0 + GAMMA * vNext : 0;
      const delta = target - vS;

      // Semi-gradient update: no gradient taken through the target.
      for (let i = 0; i < N_W; i++) {
        w[i] += alpha * delta * features[s][i];
      }

      const norm = Math.sqrt(w.reduce((acc, x) => acc + x * x, 0));
      if (t % 6 === 0) {
        trace.push({ x: t, y: Math.min(norm, 1e12) });
        wTrace.push({ x: t, y: Math.max(-1e12, Math.min(1e12, w[6])) });
      }
      if (!Number.isFinite(norm) || norm > 1e12) break;
    }

    const norm = Math.sqrt(w.reduce((acc, x) => acc + x * x, 0));
    return {
      series: [
        { id: '‖w‖ (parameter norm)', data: trace },
        { id: 'w₇ (hub weight)', data: wTrace },
      ],
      finalNorm: norm,
      diverged: !Number.isFinite(norm) || norm > 1e6,
    };
  }, [approximation, bootstrapping, offPolicy, alpha]);

  const activeCount = [approximation, bootstrapping, offPolicy].filter(Boolean).length;

  const toggles = [
    {
      label: 'Function approximation',
      on: approximation,
      set: setApproximation,
      note: 'features shared between states',
    },
    {
      label: 'Bootstrapping',
      on: bootstrapping,
      set: setBootstrapping,
      note: 'target uses its own estimate',
    },
    {
      label: 'Off-policy',
      on: offPolicy,
      set: setOffPolicy,
      note: 'update distribution ≠ target policy',
    },
  ];

  return (
    <SimPanel
      title="The deadly triad, one ingredient at a time"
      id="ch08-deadly-triad"
      subtitle="Baird's counterexample: all rewards are zero, the true value function v = 0 is exactly representable, and the parameters still explode."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap gap-2">
            {toggles.map((t) => (
              <button
                key={t.label}
                type="button"
                onClick={() => t.set(!t.on)}
                aria-pressed={t.on}
                className={cn(
                  'rounded-md border px-2.5 py-1.5 text-left text-[11.5px] transition-colors',
                  t.on
                    ? 'border-status-critical bg-[color-mix(in_srgb,var(--status-critical)_10%,transparent)] text-ink'
                    : 'border-hairline text-ink-muted hover:bg-surface-sunken',
                )}
              >
                <span className="block font-semibold">
                  {t.on ? '● ' : '○ '}
                  {t.label}
                </span>
                <span className="block text-[10.5px] opacity-80">{t.note}</span>
              </button>
            ))}
          </div>
          <Slider
            label="Step size α"
            value={alpha}
            min={0.001}
            max={0.05}
            step={0.001}
            onChange={setAlpha}
            format={(v) => v.toFixed(3)}
            hint="smaller α delays divergence — it does not prevent it"
          />
        </div>
      }
      caption="With all three switched on, the parameter norm grows without bound: the algorithm is not slow to converge, it is actively diverging. Switch off any single ingredient and it stabilizes. That is the precise content of the deadly-triad claim — no two of the three are dangerous, all three together are."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,200px]">
        <LineChart
          data={series}
          height={260}
          xLegend="update"
          yLegend="magnitude"
          caption="Parameter norm and the hub weight over updates."
        />
        <div className="space-y-2">
          <StatTile
            label="Ingredients active"
            value={`${activeCount} of 3`}
            mono={false}
            status={activeCount === 3 ? 'critical' : 'good'}
          />
          <StatTile
            label="Final ‖w‖"
            value={Number.isFinite(finalNorm) ? finalNorm : NaN}
            status={diverged ? 'critical' : 'good'}
            hint={diverged ? 'diverged' : 'bounded'}
          />
          <p className="rounded-lg border border-hairline px-2.5 py-2 text-[11.5px] leading-snug text-ink-muted">
            {diverged
              ? 'All three present: the semi-gradient update is not a gradient of anything, and nothing forces it downhill.'
              : 'Stable — the missing ingredient breaks the divergence mechanism.'}
          </p>
        </div>
      </div>
    </SimPanel>
  );
}
