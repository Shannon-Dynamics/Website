'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { useTheme } from '@/components/layout/ThemeProvider';
import { sequentialColor } from '@/lib/theme';

/**
 * `ch07-lambda-dial` — the dial between TD(0) and Monte Carlo.
 *
 * Two views of the same number λ, side by side:
 *   left  — the weights (1−λ)λⁿ⁻¹ the λ-return places on each n-step return
 *   right — the eligibility trace decaying along a corridor Rusty just drove
 *
 * Drag λ to 0 and the weight collapses onto the one-step return: TD(0). Drag
 * it to 1 and the weight slides entirely onto the full return: Monte Carlo.
 * Everything in between is the bias–variance trade the chapter is about.
 */
export function LambdaDial() {
  const { mode } = useTheme();
  const [lambda, setLambda] = useState(0.7);
  const [gamma, setGamma] = useState(0.95);
  const horizon = 20;

  const weights = useMemo(
    () =>
      Array.from({ length: horizon }, (_, i) => {
        const n = i + 1;
        return { n, w: (1 - lambda) * Math.pow(lambda, n - 1) };
      }),
    [lambda],
  );

  const traceDecay = useMemo(
    () =>
      Array.from({ length: 14 }, (_, i) => ({
        step: i,
        e: Math.pow(gamma * lambda, i),
      })),
    [gamma, lambda],
  );

  const series = [
    {
      id: 'weight on n-step return',
      data: weights.map((w) => ({ x: w.n, y: w.w })),
    },
  ];

  const effectiveN =
    lambda >= 0.999 ? Infinity : weights.reduce((acc, w) => acc + w.n * w.w, 0) / (1 - Math.pow(lambda, horizon));

  return (
    <SimPanel
      title="The λ dial: from TD(0) to Monte Carlo"
      id="ch07-lambda-dial"
      subtitle="Gλ = (1−λ) Σₙ λⁿ⁻¹ Gₜ:ₜ₊ₙ — a geometric blend of every n-step return at once."
      controls={
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider
            label="Trace decay λ"
            value={lambda}
            min={0}
            max={1}
            step={0.01}
            onChange={setLambda}
            hint={
              lambda === 0
                ? 'λ = 0 → TD(0): one-step bootstrapping'
                : lambda >= 0.999
                  ? 'λ = 1 → Monte Carlo: the full return'
                  : `effective lookahead ≈ ${effectiveN.toFixed(1)} steps`
            }
          />
          <Slider
            label="Discount γ"
            value={gamma}
            min={0.5}
            max={0.999}
            step={0.005}
            onChange={setGamma}
            hint="traces decay at γλ per step"
          />
        </div>
      }
      caption="Drag λ to 1 and you have just rediscovered Monte Carlo: all the weight lands on the complete return, unbiased but high-variance. Drag it to 0 and only the one-step return survives — low variance, but every error in V(s′) is inherited. Robots usually want the middle, and the corridor on the right shows why: one update credits the whole approach path, not just the final cell."
    >
      <div className="grid gap-4 lg:grid-cols-[1fr,300px]">
        <LineChart
          data={series}
          height={250}
          xLegend="n (steps of lookahead)"
          yLegend="weight (1−λ)λⁿ⁻¹"
          showPoints
          table={{
            columns: ['n', 'weight'],
            rows: weights.map((w) => [w.n, w.w]),
          }}
        />

        <div className="space-y-3">
          <div className="rounded-lg border border-hairline bg-surface-sunken p-3">
            <p className="mb-2 text-[11.5px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              Eligibility along Rusty&apos;s corridor
            </p>
            <svg width="100%" height={64} viewBox="0 0 268 64" role="img" aria-label="Eligibility trace decaying backwards along a corridor of cells">
              {traceDecay.map((t, i) => (
                <g key={i}>
                  <rect
                    x={4 + i * 19}
                    y={16}
                    width={17}
                    height={26}
                    rx={3}
                    fill={sequentialColor(t.e, mode)}
                    stroke="var(--surface-1)"
                  />
                </g>
              ))}
              <text x={4} y={56} fontSize={9} fill="var(--text-muted)">
                current cell
              </text>
              <text x={196} y={56} fontSize={9} fill="var(--text-muted)">
                ← earlier
              </text>
            </svg>
            <p className="mt-1.5 text-[11px] leading-snug text-ink-muted">
              One TD error updates every shaded cell in proportion to its trace — the backward view.
            </p>
          </div>

          <StatTile
            label="Effective lookahead"
            value={lambda >= 0.999 ? '∞ (full return)' : `${effectiveN.toFixed(1)} steps`}
            mono={false}
            hint="mean n under the λ-weighting"
          />
          <StatTile
            label="Trace half-life"
            value={
              gamma * lambda <= 0 ? '0 steps' : `${(Math.log(0.5) / Math.log(gamma * lambda)).toFixed(1)} steps`
            }
            mono={false}
            hint="how far back credit reaches"
          />
        </div>
      </div>
    </SimPanel>
  );
}
