'use client';

import { useMemo, useState } from 'react';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { SimPanel, Slider } from './SimControls';

/**
 * `ch14-dimensionality-wall` — Kober's first curse, in arithmetic.
 *
 * A 7-DoF arm discretized coarsely already dwarfs the number of atoms you can
 * afford to visit. The widget lets the reader set the DoF and the bins per
 * dimension, then reports how long an exhaustive tabular sweep would take at a
 * realistic robot control rate — the number that killed tabular RL for
 * robotics and forced everything in Part II.
 */
export function CurseOfDimensionality() {
  const [dof, setDof] = useState(7);
  const [bins, setBins] = useState(10);
  const [rate, setRate] = useState(1000);

  // Position AND velocity per joint — the state is (q, q̇).
  const stateDims = dof * 2;
  const states = Math.pow(bins, stateDims);
  const secondsPerSweep = states / rate;

  const series = useMemo(() => {
    const pts = [];
    for (let d = 1; d <= 14; d++) {
      pts.push({ x: d, y: Math.log10(Math.pow(bins, d * 2)) });
    }
    return [{ id: 'log₁₀(number of states)', data: pts }];
  }, [bins]);

  const humanTime = (s: number): string => {
    if (!Number.isFinite(s)) return 'beyond counting';
    if (s < 60) return `${s.toFixed(1)} seconds`;
    if (s < 3600) return `${(s / 60).toFixed(1)} minutes`;
    if (s < 86400) return `${(s / 3600).toFixed(1)} hours`;
    if (s < 3.15e7) return `${(s / 86400).toFixed(1)} days`;
    const years = s / 3.15e7;
    if (years < 1e3) return `${years.toFixed(1)} years`;
    if (years < 1e9) return `${(years / 1e6).toFixed(2)} million years`;
    return `${(years / 1.38e10).toExponential(2)} × the age of the universe`;
  };

  return (
    <SimPanel
      title="The exponential wall"
      id="ch14-dimensionality-wall"
      subtitle="Discretize a robot's joint space and count the cells. One sweep of tabular DP must touch every one of them."
      controls={
        <div className="grid gap-3 sm:grid-cols-3">
          <Slider
            label="Degrees of freedom"
            value={dof}
            min={1}
            max={14}
            step={1}
            onChange={setDof}
            format={(v) => v.toFixed(0)}
            hint="7 = a typical research arm; 12 = a quadruped"
          />
          <Slider
            label="Bins per dimension"
            value={bins}
            min={3}
            max={20}
            step={1}
            onChange={setBins}
            format={(v) => v.toFixed(0)}
            hint="10 bins is a coarse discretization"
          />
          <Slider
            label="States visited per second"
            value={rate}
            min={100}
            max={1e6}
            step={100}
            onChange={setRate}
            format={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0))}
            hint="generous for real hardware"
          />
        </div>
      }
      caption="Set 7 DoF and 10 bins — a deliberately crude discretization of a modest arm — and a single sweep already outlives the solar system. No faster computer rescues this: the curve is exponential in the exponent. Part II's answer is to stop enumerating states and start generalizing across them."
    >
      <div className="grid gap-3 lg:grid-cols-[1fr,220px]">
        <LineChart
          data={series}
          height={250}
          xLegend="degrees of freedom"
          yLegend="log₁₀(states)"
          showPoints
          caption="Each additional joint multiplies the state count by bins² — position and velocity."
        />
        <div className="space-y-2">
          <StatTile
            label="State dimensions"
            value={stateDims}
            hint={`${dof} positions + ${dof} velocities`}
          />
          <StatTile
            label="Discrete states"
            value={states > 1e6 ? states.toExponential(2) : states.toFixed(0)}
            mono
          />
          <StatTile
            label="Time for one sweep"
            value={humanTime(secondsPerSweep)}
            mono={false}
            status={secondsPerSweep > 3.15e7 ? 'critical' : secondsPerSweep > 3600 ? 'warning' : 'good'}
            hint={
              secondsPerSweep > 3.15e7
                ? 'tabular methods are not an option'
                : secondsPerSweep > 3600
                  ? 'impractical on hardware'
                  : 'feasible'
            }
          />
        </div>
      </div>
    </SimPanel>
  );
}
