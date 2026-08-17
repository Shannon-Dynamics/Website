'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle } from '@/components/sim/controls';
import { Dashboard, DashboardPanel, LineChart, StatTile } from '@/components/viz';
import {
  TIGER,
  type AlphaVec,
  type Belief,
  backupCandidates,
  envelope2,
  expandBeliefSet,
  pbvi,
  prune,
  qmdpAlphas,
  valueAt,
} from '@/lib/pomdp/finite';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w22.3 — the Alpha Forge.
 *
 * One exact backup per click, on the tiger. Two counters run beside the
 * picture: how many candidate α-vectors the cross-sum produced, and how many
 * survived pruning. Turn pruning off and the first number is all you get —
 * 3, then 27, then 2187, then 14 million — which is the honest reason nobody
 * solves POMDPs this way.
 *
 * PBVI mode replaces the exact backup with the point-based one: pick a handful
 * of beliefs, keep one vector each, and watch the envelope degrade gracefully
 * as you take points away.
 */

/** The reachable belief ladder from b = ½ under listening, plus the corners. */
const REACHABLE: Belief[] = (() => {
  let B: Belief[] = [
    [0.5, 0.5],
    [1, 0],
    [0, 1],
  ];
  for (let i = 0; i < 4; i++) B = expandBeliefSet(TIGER, B);
  return B;
})();

const IMMEDIATE: AlphaVec[] = TIGER.actions.map((_, u) => ({
  v: TIGER.states.map((_s, x) => TIGER.r[x][u]),
  action: u,
}));

interface Stage {
  horizon: number;
  /** |U| · |Γ|^|Z| — the analytic candidate count, whether or not we built them. */
  raw: number;
  kept: number;
  gamma: AlphaVec[];
}

/** Run exact VI to `horizon`, optionally without pruning. Cheap enough to redo live. */
function exactStages(horizon: number, doPrune: boolean): Stage[] {
  const A = TIGER.actions.length;
  const Z = TIGER.observations.length;
  let set = doPrune ? prune(IMMEDIATE) : IMMEDIATE;
  const stages: Stage[] = [{ horizon: 1, raw: A, kept: set.length, gamma: set }];
  for (let h = 2; h <= horizon; h++) {
    const raw = A * Math.pow(stages[stages.length - 1].kept, Z);
    // Without pruning the candidate set squares every step; refuse to build
    // more than a few thousand vectors and report the count instead.
    if (raw > 20000) {
      stages.push({ horizon: h, raw, kept: raw, gamma: [] });
      // Keep reporting the analytic growth even though nothing is materialized.
      for (let k = h + 1; k <= horizon; k++) {
        const prev = stages[stages.length - 1].kept;
        stages.push({ horizon: k, raw: A * Math.pow(prev, Z), kept: A * Math.pow(prev, Z), gamma: [] });
      }
      break;
    }
    const cand = backupCandidates(TIGER, set);
    set = doPrune ? prune(cand) : cand;
    stages.push({ horizon: h, raw, kept: set.length, gamma: set });
  }
  return stages;
}

export function AlphaForge() {
  const [horizon, setHorizon] = useState(3);
  const [pruning, setPruning] = useState(true);
  const [mode, setMode] = useState<'exact' | 'pbvi'>('exact');
  const [nPoints, setNPoints] = useState(5);

  const stages = useMemo(() => exactStages(horizon, pruning), [horizon, pruning]);
  const current = stages[stages.length - 1];

  const beliefSet = useMemo(() => REACHABLE.slice(0, nPoints), [nPoints]);
  const pbviOut = useMemo(() => pbvi(TIGER, beliefSet, Math.max(horizon, 30)), [beliefSet, horizon]);

  const shown = mode === 'exact' ? current.gamma : pbviOut.gamma;
  const segments = useMemo(() => envelope2(shown.length > 0 ? shown : IMMEDIATE), [shown]);

  /**
   * The converged answer, for the "how far off are we" readout.
   *
   * 240 backups: the Bellman operator is only a 0.95-contraction, so V(½) needs
   * about two hundred of them to settle on 19.3714 — but |Γ| balloons to ~100
   * around backup 40 and collapses back to 9 by backup 100, so essentially all
   * of the half-second is spent in that hump. Deferred to an effect rather than
   * a mount-time useMemo, because half a second of blocked main thread is worse
   * than one frame with a dash in the readout.
   */
  const [reference, setReference] = useState<AlphaVec[] | null>(null);
  useEffect(() => {
    const id = window.setTimeout(() => {
      let set = prune(IMMEDIATE);
      for (let i = 0; i < 240; i++) set = prune(backupCandidates(TIGER, set));
      setReference(set);
    }, 0);
    return () => window.clearTimeout(id);
  }, []);

  const qmdpThreshold = useMemo(() => {
    const seg = envelope2(qmdpAlphas(TIGER)).filter((s) => s.action === 2);
    return seg.length > 0 ? seg[0].tStart : 1;
  }, []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const actionColor = [p.measurement, p.prediction, p.posterior];
      if (shown.length === 0) {
        label(
          ctx,
          `${current.raw.toLocaleString()} candidate vectors — too many to draw, which is the point`,
          v.width / 2,
          v.height / 2,
          p.prediction,
          { size: 13, align: 'center', weight: 600 },
        );
        return;
      }

      const probe = (t: number) => valueAt(shown, [t, 1 - t]).value;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i <= 200; i++) {
        const val = probe(i / 200);
        lo = Math.min(lo, val);
        hi = Math.max(hi, val);
      }
      const span = Math.max(hi - lo, 1e-6);
      const yLo = lo - 0.5 * span;
      const yHi = hi + 0.15 * span;
      const yOf = (val: number) => sy(v, 0.14 + ((val - yLo) / (yHi - yLo)) * 0.78);

      // Every candidate, faint. With pruning off this is the explosion, visible.
      ctx.strokeStyle = p.truth;
      ctx.globalAlpha = shown.length > 400 ? 0.045 : shown.length > 60 ? 0.12 : 0.32;
      ctx.lineWidth = 1;
      for (const a of shown) {
        ctx.beginPath();
        ctx.moveTo(sx(v, 0), yOf(a.v[1]));
        ctx.lineTo(sx(v, 1), yOf(a.v[0]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // The upper envelope and the policy strip beneath it.
      for (const seg of segments) {
        ctx.strokeStyle = actionColor[seg.action];
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(sx(v, seg.tStart), yOf(seg.intercept + seg.slope * seg.tStart));
        ctx.lineTo(sx(v, seg.tEnd), yOf(seg.intercept + seg.slope * seg.tEnd));
        ctx.stroke();
        ctx.fillStyle = actionColor[seg.action];
        ctx.globalAlpha = 0.5;
        ctx.fillRect(sx(v, seg.tStart), sy(v, 0.07), Math.max(sl(v, seg.tEnd - seg.tStart), 1), sl(v, 0.04));
        ctx.globalAlpha = 1;
      }

      // In PBVI mode the belief points are the algorithm, so draw them.
      if (mode === 'pbvi') {
        for (const b of beliefSet) {
          const px = sx(v, b[0]);
          const py = yOf(valueAt(shown, b).value);
          ctx.fillStyle = p.prior;
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = p.prior;
          ctx.globalAlpha = 0.45;
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px, sy(v, 0.07));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
      }

      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, 0.07));
      ctx.lineTo(sx(v, 1), sy(v, 0.07));
      ctx.stroke();
      label(ctx, 'b(tiger-left) = 0', sx(v, 0), sy(v, 0.02), p.truth, { size: 9 });
      label(ctx, '1', sx(v, 1), sy(v, 0.02), p.truth, { size: 9, align: 'right' });
      label(
        ctx,
        mode === 'exact' ? `Γ${current.horizon}  ·  ${shown.length} vectors` : `PBVI  ·  |B| = ${beliefSet.length}`,
        sx(v, 0.01),
        sy(v, 0.95),
        p.posterior,
        { size: 11, weight: 600 },
      );
    },
    [shown, segments, mode, beliefSet, current],
  );

  const growth = useMemo(
    () => [
      {
        id: 'candidates before pruning',
        role: 'prediction' as const,
        data: stages.map((s) => ({ x: s.horizon, y: Math.log10(Math.max(s.raw, 1)) })),
      },
      {
        id: 'kept after pruning',
        role: 'posterior' as const,
        data: stages.map((s) => ({ x: s.horizon, y: Math.log10(Math.max(s.kept, 1)) })),
      },
    ],
    [stages],
  );

  const vHalf = shown.length > 0 ? valueAt(shown, [0.5, 0.5]).value : Number.NaN;
  const vRef = reference ? valueAt(reference, [0.5, 0.5]).value : Number.NaN;
  const threshold = useMemo(() => {
    const seg = segments.filter((s) => s.action === 2);
    return seg.length > 0 ? seg[0].tStart : Number.NaN;
  }, [segments]);

  return (
    <WidgetFrame
      id="w22.3"
      title="The Alpha Forge"
      teaches="Exact POMDP value iteration does not scale badly — it explodes combinatorially. Approximation is structural, not laziness."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Each backup produces one α-vector per (action, assignment of one surviving vector to each
          observation) — that is <code>|U|·|Γ|^|Z|</code> candidates, and only a handful of them
          ever touch the upper envelope. <strong>What to notice:</strong> turn pruning off and the
          counter reads 3, 27, 2187, 14,348,907 while the picture stops being drawable at horizon 4.
          Turn it back on and the same value function is carried by five to forty vectors.{' '}
          <strong>Then try PBVI:</strong> drag the belief-point count down to 3 and the envelope
          collapses onto the QMDP answer — threshold exactly 0.900 — because <code>{'{'}½, 1, 0{'}'}</code>{' '}
          cannot see that listening is worth anything at 0.9. Add the reachable point{' '}
          <code>0.85</code> and the threshold jumps to 0.960; add its mirror <code>0.15</code> and it
          settles at 0.958, within 0.003 of the exact 0.9603. Five belief points, chosen where the
          robot can actually get to, buy most of what the exact solver spends millions of vectors on.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, maxX: 1, minY: 0, maxY: 1 }}
        draw={draw}
        deps={[shown, segments, mode, beliefSet]}
        aspect={2.4}
        padding={0.03}
        ariaLabel="A plot of alpha-vectors as straight lines over the belief segment, with their upper envelope highlighted and a strip beneath showing the optimal action at each belief."
      />

      <ControlPanel columns={2}>
        <Slider
          label={mode === 'exact' ? 'Backups (horizon T)' : 'Backups used by PBVI'}
          role="posterior"
          value={horizon}
          min={1}
          max={12}
          step={1}
          format={(x) => x.toFixed(0)}
          onChange={(x) => setHorizon(Math.round(x))}
          help="One click of value iteration per unit. Γ_1 is the immediate-reward set."
        />
        <Slider
          label="PBVI belief points |B|"
          role="prior"
          value={nPoints}
          min={3}
          max={REACHABLE.length}
          step={1}
          format={(x) => x.toFixed(0)}
          onChange={(x) => setNPoints(Math.round(x))}
          help="Taken from the reachable ladder 0.5, 1, 0, 0.85, 0.15, 0.9698, 0.0302, …"
        />
        <Toggle label="Pruning on" role="posterior" checked={pruning} onChange={setPruning} />
        <ButtonRow>
          <ActionButton onClick={() => setMode('exact')} emphasis={mode === 'exact'}>
            Exact VI
          </ActionButton>
          <ActionButton onClick={() => setMode('pbvi')} emphasis={mode === 'pbvi'}>
            PBVI
          </ActionButton>
          <ActionButton onClick={() => setHorizon((h) => Math.min(h + 1, 12))}>
            One more backup
          </ActionButton>
        </ButtonRow>
      </ControlPanel>

      <div className="border-t border-fd-border p-3">
        <Dashboard columns={4}>
          <StatTile
            label="candidates this backup"
            value={current.raw.toLocaleString()}
            role="prediction"
          />
          <StatTile
            label={mode === 'exact' ? 'vectors kept' : 'PBVI vectors'}
            value={shown.length}
            role="posterior"
          />
          <StatTile
            label="open threshold"
            value={Number.isNaN(threshold) ? '—' : threshold.toFixed(4)}
            role="prior"
            trend={Number.isNaN(threshold) ? undefined : threshold - qmdpThreshold}
            trendLabel="vs QMDP's 0.900"
          />
          <StatTile
            label="V(½) against converged"
            value={
              Number.isNaN(vHalf) || Number.isNaN(vRef)
                ? '—'
                : `${vHalf.toFixed(2)} / ${vRef.toFixed(2)}`
            }
            role="truth"
          />
          <DashboardPanel title="Vector count per backup (log₁₀)" span="full">
            <LineChart
              series={growth}
              xLabel="horizon T"
              yLabel="log₁₀ (number of α-vectors)"
              height={210}
              curve="linear"
              yMin={0}
              caption="Orange is the cross-sum count the backup would have to enumerate; purple is what survives pruning. The gap is the entire reason point-based methods exist."
            />
          </DashboardPanel>
        </Dashboard>
      </div>
    </WidgetFrame>
  );
}
