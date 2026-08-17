'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ActionButton, ButtonRow, Transport } from '@/components/sim/controls';
import { Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { GridRbpf, type RbpfReport } from '@/lib/filters/rbpf';
import { OccupancyGrid } from '@/lib/mapping/occgrid';
import { applyOdom, odomFromPoses } from '@/lib/models/motion';
import { APARTMENT, simulateScan } from '@/lib/sim/world';
import { Rng } from '@/lib/prob/rng';
import type { Pose2 } from '@/lib/geom/se2';
import {
  BEAM_ANGLES,
  INVERSE_MODEL,
  LAP,
  MAP_OPTS,
  RBPF_BASE,
  SCAN_PARAMS,
  START_POSE,
  driveStep,
  odometryReading,
} from './apartment-lap';

/**
 * w17.2 — the Depletion Meter.
 *
 * Same lap, same filter, three resampling policies. The point is that the two
 * failure modes are on opposite ends of one knob and only *one* of them is
 * visible on the meter everybody watches:
 *
 *   never  → weight degeneracy. N_eff collapses toward 1: M universes are
 *            being simulated, one of them carries all the probability.
 *   always → ancestral collapse. N_eff looks perfect (it is reset to M every
 *            step by construction) while every surviving universe turns out to
 *            be a copy of the same great-grandparent.
 *   N_eff < M/2 → the standard compromise, and still not free.
 */

const M = 16;
/** Lineage labels are refreshed this often, so the gauge reads "of the last W". */
const WINDOW = 24;

type Policy = 'never' | 'selective' | 'always';

const POLICY_LABEL: Record<Policy, string> = {
  never: 'Never resample',
  selective: 'Selective (N_eff < M/2)',
  always: 'Resample every step',
};

interface Sample {
  t: number;
  neff: number;
  lineages: number;
  error: number;
  dead: number;
}

interface State {
  filter: GridRbpf;
  rng: Rng;
  world: Rng;
  truth: Pose2;
  dead: Pose2;
  report: RbpfReport;
  history: Sample[];
  resamples: number;
}

const EMPTY_REPORT: RbpfReport = {
  neff: M,
  stepNeff: M,
  resampled: false,
  distinctAncestors: M,
  weights: [],
  bestIndex: 0,
  scanMatched: 0,
};

export function DepletionMeter() {
  const [policy, setPolicy] = useState<Policy>('selective');
  const policyRef = useRef(policy);
  policyRef.current = policy;

  const init = useCallback((seed: number): State => {
    const p = policyRef.current;
    const filter = GridRbpf.atPose(M, START_POSE, () => new OccupancyGrid(MAP_OPTS), {
      ...RBPF_BASE,
      angles: BEAM_ANGLES,
      inverse: INVERSE_MODEL,
      selectiveResampling: p !== 'always',
      // "Never" is the same policy with a threshold of zero: N_eff can never
      // fall below it, so the branch never fires.
      neffRatio: p === 'never' ? 0 : RBPF_BASE.neffRatio,
    });
    return {
      filter,
      rng: new Rng(seed),
      world: new Rng(seed ^ 0x5eed),
      truth: { ...START_POSE },
      dead: { ...START_POSE },
      report: { ...EMPTY_REPORT, weights: filter.weights },
      history: [],
      resamples: 0,
    };
  }, []);

  const step = useCallback((s: State, tick: number): State => {
    // Fresh lineage labels every W steps: the gauge then answers "how many of
    // the hypotheses alive W steps ago still have descendants?"
    if (tick % WINDOW === 0) s.filter.particles.forEach((p, i) => (p.ancestor = i));

    const prev = s.truth;
    const truth = driveStep(prev, tick);
    const u = odometryReading(odomFromPoses(prev, truth));
    const dead = applyOdom(s.dead, u);
    const ranges = simulateScan(APARTMENT, truth, SCAN_PARAMS, s.world);
    const report = s.filter.step(u, ranges, s.rng);
    const best = s.filter.best().pose;

    const sample: Sample = {
      t: tick,
      neff: report.neff / M,
      lineages: report.distinctAncestors,
      error: Math.hypot(best.x - truth.x, best.y - truth.y),
      dead: Math.hypot(dead.x - truth.x, dead.y - truth.y),
    };

    return {
      ...s,
      truth,
      dead,
      report,
      history: [...s.history, sample],
      resamples: s.resamples + (report.resampled ? 1 : 0),
    };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 8, initialSeed: 42, maxTicks: LAP, loop: true });
  const { reset } = sim;

  useEffect(() => {
    reset();
  }, [policy, reset]);

  const { neffSeries, errorSeries, latest } = useMemo(() => {
    const h = sim.state.history;
    return {
      neffSeries: [
        { id: 'N_eff / M', role: 'posterior' as const, data: h.map((s) => ({ x: s.t, y: s.neff })) },
      ],
      errorSeries: [
        { id: 'best particle', role: 'posterior' as const, data: h.map((s) => ({ x: s.t, y: s.error })) },
        { id: 'raw odometry', role: 'prediction' as const, data: h.map((s) => ({ x: s.t, y: s.dead })) },
      ],
      latest: h[h.length - 1],
    };
  }, [sim.state.history]);

  const sparkNeff = useMemo(
    () => sim.state.history.slice(-40).map((s) => s.neff),
    [sim.state.history],
  );
  const sparkLineage = useMemo(
    () => sim.state.history.slice(-40).map((s) => s.lineages),
    [sim.state.history],
  );

  return (
    <WidgetFrame
      id="w17.2"
      title="Depletion Meter"
      teaches="Resampling is not free: it fixes weight degeneracy by spending diversity, and the meter that shows the first is blind to the second."
      colorKey={['prediction', 'posterior']}
      caption={
        <>
          The same lap, the same seed, the same measurements — only the resampling policy changes.
          With <em>Never</em>, watch <span style={{ color: 'var(--pr-posterior)' }}>N_eff/M</span> slide
          to about 0.1: all sixteen universes are still being simulated and paid for, but one of
          them holds essentially all the probability, so the filter is really running with a single
          hypothesis — and the error curve shows it. With <em>Resample every step</em>, N_eff looks
          healthy for the opposite reason: the weights are wiped clean every step, so the meter only
          ever reports one step of evidence. Meanwhile the lineage gauge falls to 1, which says every
          surviving universe is a copy of one ancestor and the population can no longer disagree
          about anything. Selective resampling buys the middle, and note how little it buys: even at
          <em> N</em>_eff &lt; <em>M</em>/2 it still fires on roughly half the steps here. Nothing on
          this panel restores diversity; that needs the injections of{' '}
          <Link href="/chapters/ch08-nonparametric-filters">Chapter 8</Link>.
        </>
      }
    >
      <div className="border-b border-fd-border px-3 py-2">
        <ButtonRow>
          {(['never', 'selective', 'always'] as const).map((p) => (
            <ActionButton key={p} onClick={() => setPolicy(p)} emphasis={policy === p}>
              {POLICY_LABEL[p]}
            </ActionButton>
          ))}
        </ButtonRow>
      </div>

      <div className="p-3">
        <Dashboard columns={4}>
          <StatTile
            label="N_eff / M"
            value={latest?.neff ?? 1}
            precision={2}
            role="posterior"
            sparkline={sparkNeff}
          />
          <StatTile
            label={`lineages (of ${M})`}
            value={latest?.lineages ?? M}
            sparkline={sparkLineage}
            precision={0}
          />
          <StatTile label="resamples" value={sim.state.resamples} precision={0} />
          <StatTile
            label="best-particle error"
            value={latest?.error ?? 0}
            unit="m"
            precision={2}
            role="posterior"
          />

          <DashboardPanel title="Effective sample size" span={2}>
            <LineChart
              series={neffSeries}
              xLabel="step"
              yLabel="N_eff / M"
              height={190}
              yMin={0}
              yMax={1.05}
              markers={[{ axis: 'y', value: 0.5, label: 'resample threshold' }]}
              ariaLabel="Effective sample size divided by the particle count, over the lap."
            />
          </DashboardPanel>

          <DashboardPanel title="Position error vs. dead reckoning" span={2}>
            <LineChart
              series={errorSeries}
              xLabel="step"
              yLabel="error (m)"
              height={190}
              yMin={0}
              ariaLabel="Position error of the highest-weight particle compared with raw odometry, over the lap."
            />
          </DashboardPanel>
        </Dashboard>
      </div>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
