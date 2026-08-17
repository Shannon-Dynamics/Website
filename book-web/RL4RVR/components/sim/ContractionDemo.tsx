'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';

/**
 * `ch02-contraction-map` — Banach's fixed-point theorem, made tactile.
 *
 * The reader sets the contraction modulus γ and the starting point, then
 * watches the iterates spiral into the unique fixed point. The a-priori bound
 * γ^k‖x₁ − x₀‖/(1−γ) is drawn beside the actual error, so the reader can see
 * the theorem is not merely true but *tight enough to be useful* — which is
 * exactly why Chapter 4 can promise value iteration converges, and Chapter 5
 * can tell you when to stop sweeping.
 */
export function ContractionDemo() {
  const [gamma, setGamma] = useState(0.8);
  const [x0, setX0] = useState(9);
  const [iterations, setIterations] = useState(24);

  // T(x) = γx + c is a γ-contraction with fixed point x* = c/(1−γ).
  const c = 2;
  const fixedPoint = c / (1 - gamma);

  const { series, table, finalErr, bound } = useMemo(() => {
    const xs: number[] = [x0];
    for (let k = 0; k < iterations; k++) {
      xs.push(gamma * xs[k] + c);
    }
    const firstStep = Math.abs(xs[1] - xs[0]);
    const errors = xs.map((x) => Math.abs(x - fixedPoint));
    const bounds = xs.map((_, k) => (Math.pow(gamma, k) * firstStep) / (1 - gamma));

    return {
      series: [
        { id: 'iterate xₖ', data: xs.map((y, x) => ({ x, y })) },
        {
          id: 'fixed point x*',
          data: xs.map((_, x) => ({ x, y: fixedPoint })),
        },
      ],
      table: {
        columns: ['k', 'xₖ', '|xₖ − x*|', 'a-priori bound'],
        rows: xs.map((x, k) => [k, x, errors[k], bounds[k]]),
      },
      finalErr: errors[errors.length - 1],
      bound: bounds[bounds.length - 1],
    };
  }, [gamma, x0, iterations, fixedPoint]);

  return (
    <SimPanel
      title="A contraction, iterated"
      id="ch02-contraction-map"
      subtitle="T(x) = γx + 2 — every application shrinks distances by a factor γ, so the iterates must converge to a single point."
      controls={
        <div className="grid gap-3 sm:grid-cols-3">
          <Slider
            label="Contraction modulus γ"
            value={gamma}
            min={0.05}
            max={0.98}
            step={0.01}
            onChange={setGamma}
            hint={`x* = ${fixedPoint.toFixed(2)}`}
          />
          <Slider
            label="Starting point x₀"
            value={x0}
            min={-10}
            max={30}
            step={0.5}
            onChange={setX0}
            format={(v) => v.toFixed(1)}
            hint="anywhere at all — that is the point"
          />
          <Slider
            label="Iterations"
            value={iterations}
            min={4}
            max={60}
            step={1}
            onChange={setIterations}
            format={(v) => v.toFixed(0)}
          />
        </div>
      }
      caption="Move x₀ anywhere: the iterates always land on the same x*. That is uniqueness. Push γ toward 1 and convergence slows to a crawl — the same slowdown you will feel in Chapter 5 when a far-sighted discount makes value iteration take hundreds of sweeps, and the reason γ is a modeling decision rather than a free parameter."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,190px]">
        <LineChart
          data={series}
          height={250}
          xLegend="iteration k"
          yLegend="value"
          dashed={['fixed point x*']}
          table={table}
          showPoints
        />
        <div className="space-y-2">
          <StatTile label="Fixed point x*" value={fixedPoint} hint="c / (1 − γ)" />
          <StatTile label="Actual error" value={finalErr} hint={`|x${iterations} − x*|`} />
          <StatTile
            label="A-priori bound"
            value={bound}
            hint="γᵏ‖x₁−x₀‖/(1−γ)"
            status={bound >= finalErr ? 'good' : 'critical'}
          />
          <p className="rounded-lg border border-hairline px-2.5 py-2 text-[11.5px] leading-snug text-ink-muted">
            The bound never dips below the error — as the theorem guarantees.
          </p>
        </div>
      </div>
    </SimPanel>
  );
}
