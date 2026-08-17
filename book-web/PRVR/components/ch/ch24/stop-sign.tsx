'use client';

import { useMemo, useState } from 'react';

import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider } from '@/components/sim/controls';
import { Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';
import { DEFAULT_EXPLORE_CONFIG, runExploration } from '@/lib/explore/explorer';
import { APARTMENT } from '@/lib/sim/world';

/**
 * w24.4 — Stop Sign.
 *
 * The curves are not drawn from a model: they are the log of one real run of
 * `runExploration` over the apartment, the same explorer w24.1 animates. The
 * reader drags the stopping threshold and reads off what the mission would
 * have cost and what it would have got.
 *
 * There is no theorem underneath the slider, and the chapter says so. A
 * gain-per-metre floor is a statement about the mission, not about information
 * theory — but at least its units are ones the mission can argue about.
 */

const START = { x: 6.0, y: 4.4, theta: 0 };
/** Ignore the first metres: every run looks brilliant while the first room fills in. */
const WARMUP_M = 3;

/** Bits-per-metre is a ratio of two noisy differences; smooth it before reading it. */
function smooth(xs: number[], half: number): number[] {
  return xs.map((_, i) => {
    let s = 0;
    let n = 0;
    for (let k = Math.max(0, i - half); k <= Math.min(xs.length - 1, i + half); k++) {
      s += xs[k];
      n++;
    }
    return s / n;
  });
}

export function StopSign() {
  const [threshold, setThreshold] = useState(4);

  const run = useMemo(
    () =>
      runExploration(
        {
          ...DEFAULT_EXPLORE_CONFIG,
          world: APARTMENT,
          start: START,
          policy: 'utility',
          seed: 24,
          // Run it to exhaustion: the point of this widget is the tail nobody
          // should have driven.
          stop: { gainRate: 0.02, minGain: 0.05 },
        },
        900,
      ),
    [],
  );

  const curves = useMemo(() => {
    const rows = run.samples;
    const h0 = rows[0].entropy;
    const rate = smooth(
      rows.map((r) => r.gainRate),
      6,
    );
    const every = Math.max(1, Math.floor(rows.length / 180));
    const gathered: { x: number; y: number }[] = [];
    const perMetre: { x: number; y: number }[] = [];
    for (let i = 0; i < rows.length; i += every) {
      gathered.push({ x: rows[i].distance, y: h0 - rows[i].entropy });
      perMetre.push({ x: rows[i].distance, y: rate[i] });
    }
    return { rows, h0, rate, gathered, perMetre, total: h0 - rows[rows.length - 1].entropy };
  }, [run]);

  const stop = useMemo(() => {
    const { rows, rate, h0, total } = curves;
    let idx = rows.length - 1;
    for (let i = 0; i < rows.length; i++) {
      if (rows[i].distance < WARMUP_M) continue;
      if (rate[i] < threshold) {
        idx = i;
        break;
      }
    }
    const row = rows[idx];
    const gathered = h0 - row.entropy;
    return {
      distance: row.distance,
      fraction: total > 0 ? gathered / total : 1,
      coverage: row.coverage * 100,
      wasted: rows[rows.length - 1].distance - row.distance,
      totalDistance: rows[rows.length - 1].distance,
    };
  }, [curves, threshold]);

  return (
    <WidgetFrame
      id="w24.4"
      title="Stop Sign"
      teaches="“Explore until the map is finished” is a budget, not a plan: the last few percent of the entropy costs a third of the run."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          One real run of the utility explorer over the apartment, logged to exhaustion. The top
          panel is information gathered — bits of map entropy removed — against metres driven; the
          bottom panel is the derivative of that curve, which is the quantity a stopping rule can
          actually act on. Drag the threshold. At ε = 15 b/m Rusty quits after 15 m holding barely
          half the information; at ε = 4 it stops at 42 m with 93% of it; and the remaining 57 m —
          more than half the whole run — buys the last 7%. Nothing on this chart
          tells you which is right — that is the mission&rsquo;s job, and it is why principled
          stopping is still an open problem.
        </>
      }
    >
      <div className="p-3">
        <Dashboard columns={4}>
          <StatTile
            label="stop at"
            value={stop.distance}
            unit="m"
            precision={1}
            role="posterior"
            trend={stop.distance - stop.totalDistance}
            trendLabel="vs. running out"
          />
          <StatTile
            label="information captured"
            value={stop.fraction * 100}
            unit="%"
            precision={1}
            role="measurement"
          />
          <StatTile label="cells resolved" value={stop.coverage} unit="%" precision={1} />
          <StatTile label="distance saved" value={stop.wasted} unit="m" precision={1} role="truth" />
        </Dashboard>

        <div className="mt-3">
          <Dashboard columns={2}>
            <DashboardPanel title="information gathered  H(m₀) − H(m)" span={2}>
              <LineChart
                series={[{ id: 'bits removed', role: 'posterior', data: curves.gathered }]}
                xLabel="distance travelled (m)"
                yLabel="bits"
                height={210}
                enableArea
                markers={[
                  { axis: 'x', value: stop.distance, label: 'stop', role: 'measurement' },
                ]}
                ariaLabel="Cumulative map entropy removed against distance travelled. The curve rises steeply for the first fifteen metres and then flattens, with a vertical marker showing where the chosen stopping threshold would end the run."
              />
            </DashboardPanel>

            <DashboardPanel title="gain per metre  dH/ds" span={2}>
              <LineChart
                series={[{ id: 'bits per metre', role: 'measurement', data: curves.perMetre }]}
                xLabel="distance travelled (m)"
                yLabel="bits / m"
                height={190}
                yMin={0}
                markers={[
                  { axis: 'y', value: threshold, label: 'ε', role: 'posterior' },
                  { axis: 'x', value: stop.distance, label: 'stop', role: 'posterior' },
                ]}
                ariaLabel="Bits of map entropy removed per metre travelled, smoothed. It spikes whenever the robot enters a new room and decays toward zero, with a horizontal line marking the stopping threshold."
              />
            </DashboardPanel>
          </Dashboard>
        </div>
      </div>

      <ControlPanel columns={1}>
        <Slider
          label="Stopping threshold ε  (bits per metre)"
          role="posterior"
          value={threshold}
          min={0.5}
          max={25}
          step={0.25}
          onChange={setThreshold}
          unit="b/m"
          help="Stop as soon as the smoothed gain rate falls below this. Absolute-entropy thresholds are worse: they depend on the size of the map."
        />
      </ControlPanel>
    </WidgetFrame>
  );
}
