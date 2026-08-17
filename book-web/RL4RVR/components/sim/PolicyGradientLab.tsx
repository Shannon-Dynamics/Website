'use client';

import { useMemo, useState } from 'react';
import { BarChart } from '@/components/viz/BarChart';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';
import { gaussian, mulberry32 } from '@/lib/rl/random';

/**
 * `ch10-gradient-variance` — why every practical policy-gradient method
 * subtracts a baseline.
 *
 * All three estimators below are unbiased: their expectations agree. What
 * differs is spread, and spread is what you actually pay for in samples. The
 * histogram shows the same gradient estimated many times; the curve shows how
 * the variance falls as the batch grows.
 */
export function PolicyGradientLab() {
  const [batch, setBatch] = useState(16);
  const [rewardScale, setRewardScale] = useState(10);
  const [lambda, setLambda] = useState(0.95);

  const { histogram, variances, stats } = useMemo(() => {
    const rng = mulberry32(23);
    const TRIALS = 600;
    const trueGradient = 1.0;

    /** One gradient estimate from `batch` episodes under a given variance-reduction scheme. */
    const estimate = (scheme: 'reinforce' | 'baseline' | 'gae'): number => {
      let sum = 0;
      for (let i = 0; i < batch; i++) {
        // Score function ∇log π, mean-zero by construction.
        const score = gaussian(rng, 0, 1);
        // Return: a large common offset plus the part that actually depends on the action.
        const common = rewardScale;
        const actionDependent = trueGradient * score;
        const noise = gaussian(rng, 0, 1.5);

        if (scheme === 'reinforce') {
          sum += score * (common + actionDependent + noise);
        } else if (scheme === 'baseline') {
          // Subtracting a state-value baseline removes the common offset.
          sum += score * (actionDependent + noise);
        } else {
          // GAE additionally trades a little bias for a large variance cut.
          sum += score * (actionDependent + noise * (1 - lambda) * 2.2);
        }
      }
      return sum / batch;
    };

    const schemes = ['reinforce', 'baseline', 'gae'] as const;
    const samples: Record<string, number[]> = { reinforce: [], baseline: [], gae: [] };
    for (const s of schemes) {
      for (let t = 0; t < TRIALS; t++) samples[s].push(estimate(s));
    }

    const variance = (xs: number[]) => {
      const m = xs.reduce((a, b) => a + b, 0) / xs.length;
      return xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length;
    };
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

    // Histogram over a shared set of bins.
    const all = schemes.flatMap((s) => samples[s]);
    const lo = Math.max(-8, Math.min(...all));
    const hi = Math.min(8, Math.max(...all));
    const BINS = 26;
    const binWidth = (hi - lo) / BINS;
    const hist = Array.from({ length: BINS }, (_, i) => {
      const center = lo + binWidth * (i + 0.5);
      const row: Record<string, string | number> = { id: center.toFixed(1) };
      for (const s of schemes) {
        row[s] = samples[s].filter((v) => v >= lo + i * binWidth && v < lo + (i + 1) * binWidth).length;
      }
      return row;
    });

    const varCurve = [8, 16, 32, 64, 128, 256].map((b) => ({ b }));

    return {
      histogram: hist,
      variances: [
        {
          id: 'REINFORCE',
          data: varCurve.map((v) => ({ x: v.b, y: variance(samples.reinforce) * (batch / v.b) })),
        },
        {
          id: '+ baseline',
          data: varCurve.map((v) => ({ x: v.b, y: variance(samples.baseline) * (batch / v.b) })),
        },
        {
          id: `+ GAE (λ=${lambda})`,
          data: varCurve.map((v) => ({ x: v.b, y: variance(samples.gae) * (batch / v.b) })),
        },
      ],
      stats: schemes.map((s) => ({
        name: s,
        mean: mean(samples[s]),
        variance: variance(samples[s]),
      })),
    };
  }, [batch, rewardScale, lambda]);

  return (
    <SimPanel
      title="Three unbiased estimators, three very different variances"
      id="ch10-gradient-variance"
      subtitle="The same policy gradient, estimated 600 times under each scheme. All three centre on the same value."
      controls={
        <div className="grid gap-3 sm:grid-cols-3">
          <Slider
            label="Batch size (episodes)"
            value={batch}
            min={4}
            max={128}
            step={4}
            onChange={setBatch}
            format={(v) => v.toFixed(0)}
          />
          <Slider
            label="Reward offset"
            value={rewardScale}
            min={0}
            max={40}
            step={1}
            onChange={setRewardScale}
            format={(v) => v.toFixed(0)}
            hint="a constant added to every return"
          />
          <Slider
            label="GAE λ"
            value={lambda}
            min={0}
            max={0.99}
            step={0.01}
            onChange={setLambda}
            hint="bias–variance dial, exactly as in Ch 7"
          />
        </div>
      }
      caption="Crank the reward offset. REINFORCE's histogram spreads out badly even though the offset carries no information about which action was good — the score function multiplies it anyway. The baseline subtracts it right back out, which is why the expectation is unchanged but the spread collapses. This is the entire practical argument for advantage estimation."
    >
      <div className="space-y-3">
        <div className="grid gap-3 lg:grid-cols-3">
          {stats.map((s) => (
            <StatTile
              key={s.name}
              label={
                s.name === 'reinforce' ? 'REINFORCE' : s.name === 'baseline' ? '+ baseline' : '+ GAE'
              }
              value={s.variance}
              hint={`mean ≈ ${s.mean.toFixed(3)} — all unbiased`}
              status={s.variance < 1 ? 'good' : s.variance < 5 ? 'warning' : 'critical'}
            />
          ))}
        </div>

        <BarChart
          data={histogram}
          keys={['reinforce', 'baseline', 'gae']}
          indexBy="id"
          height={250}
          xLegend="gradient estimate"
          yLegend="count"
          title="Sampling distribution of the gradient estimate"
        />

        <LineChart
          data={variances}
          height={220}
          xLegend="batch size"
          yLegend="variance"
          caption="Variance falls as 1/batch for all three — but they start decades apart."
        />
      </div>
    </SimPanel>
  );
}
