'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { APARTMENT, beamAngles, rayCast, simulateScan, type World } from '@/lib/sim/world';
import { DEFAULT_BEAM_PARAMS, beamLikelihood } from '@/lib/models/sensor';

/**
 * w10.4 — the Overconfidence Meter.
 *
 * The scan likelihood is a product over beams, and a product is a lie whenever
 * the beams share an error. Here the shared error is a map that is uniformly
 * 1% too wide — the most ordinary defect a hand-measured or SLAM-built map can
 * have. Ramp K from 4 beams to 180 and watch the posterior over one axis needle
 * down to a fraction of a centimetre around a position that is 4 cm wrong: the
 * filter is not merely mistaken, it is *certain* and mistaken, and no amount of
 * further sensing will rescue it.
 *
 * Two mitigations are on the panel and they are the same mitigation: tempering
 * the log-likelihood by 1/κ, and throwing away every κ-th beam. The stat tiles
 * let the reader check that they land in the same place.
 */

const MAX_RANGE = 8;
const N_FULL = 180;
const CANDIDATES = 201;
const HALF_WIN = 0.5;
const TRUTH = { x: 2.6, y: 4.4, theta: 0 };
const K_LADDER = [4, 6, 9, 12, 18, 30, 45, 60, 90, 180];

/** A map that is wrong the way real maps are wrong: a uniform scale error. */
function stretched(world: World, s: number): World {
  const cx = 6;
  return {
    ...world,
    walls: world.walls.map((w) => ({
      x1: cx + (w.x1 - cx) * (1 + s),
      y1: w.y1,
      x2: cx + (w.x2 - cx) * (1 + s),
      y2: w.y2,
    })),
  };
}

const PARAMS = { ...DEFAULT_BEAM_PARAMS, maxRange: MAX_RANGE, sigmaHit: 0.12 };
const ANGLES = beamAngles({ nBeams: N_FULL, fov: 2 * Math.PI });
const XS = Array.from(
  { length: CANDIDATES },
  (_, i) => TRUTH.x - HALF_WIN + (2 * HALF_WIN * i) / (CANDIDATES - 1),
);
const TRUTH_INDEX = Math.round(((TRUTH.x - (TRUTH.x - HALF_WIN)) / (2 * HALF_WIN)) * (CANDIDATES - 1));

interface Cache {
  /** Simulated scan at the true pose, from the true world. */
  z: number[];
  /** zStar[c * N_FULL + k]: the range beam k should return from candidate c. */
  zStar: Float64Array;
  /** Residual z − z* at the true pose, for the correlation readout. */
  residual: number[];
}

/** All the ray casting happens once; every slider after that is a table lookup. */
function buildCache(mapErr: number, seed: number): Cache {
  const model = mapErr === 0 ? APARTMENT : stretched(APARTMENT, mapErr);
  const z = simulateScan(
    APARTMENT,
    TRUTH,
    { nBeams: N_FULL, fov: 2 * Math.PI, maxRange: MAX_RANGE, sigma: 0.05, zRand: 0.01, zMax: 0.02 },
    new Rng(seed),
  );
  const zStar = new Float64Array(CANDIDATES * N_FULL);
  for (let c = 0; c < CANDIDATES; c++) {
    for (let k = 0; k < N_FULL; k++) {
      zStar[c * N_FULL + k] = rayCast(model, XS[c], TRUTH.y, TRUTH.theta + ANGLES[k], MAX_RANGE);
    }
  }
  const residual: number[] = [];
  for (let k = 0; k < N_FULL; k++) {
    const star = zStar[TRUTH_INDEX * N_FULL + k];
    if (z[k] >= MAX_RANGE - 1e-9 || star >= MAX_RANGE - 1e-9) continue;
    residual.push(z[k] - star);
  }
  return { z, zStar, residual };
}

interface Posterior {
  p: number[];
  sd: number;
  bias: number;
  covered: boolean;
}

function posteriorOver(cache: Cache, K: number, kappa: number): Posterior {
  const stride = N_FULL / K;
  const ll = new Array<number>(CANDIDATES).fill(0);
  for (let c = 0; c < CANDIDATES; c++) {
    let acc = 0;
    for (let i = 0; i < K; i++) {
      const k = Math.round(i * stride) % N_FULL;
      acc += Math.log(
        Math.max(beamLikelihood(cache.z[k], cache.zStar[c * N_FULL + k], PARAMS), 1e-300),
      );
    }
    ll[c] = acc / kappa;
  }
  const m = Math.max(...ll);
  const w = ll.map((v) => Math.exp(v - m));
  const total = w.reduce((a, b) => a + b, 0);
  const p = w.map((v) => v / total);

  const mean = XS.reduce((a, x, i) => a + x * p[i], 0);
  const sd = Math.sqrt(XS.reduce((a, x, i) => a + p[i] * (x - mean) ** 2, 0));
  const mode = XS[ll.indexOf(m)];

  // Highest-density 95% set, and whether the truth is in it.
  const order = [...p.keys()].sort((a, b) => p[b] - p[a]);
  let acc = 0;
  let covered = false;
  for (const i of order) {
    acc += p[i];
    if (i === TRUTH_INDEX) covered = true;
    if (acc >= 0.95) break;
  }
  return { p, sd, bias: mode - TRUTH.x, covered };
}

interface State {
  step: number;
}

export function OverconfidenceMeter() {
  const [sweep, setSweep] = useState(true);
  const [K, setK] = useState(180);
  const [kappa, setKappa] = useState(1);
  const [mapError, setMapError] = useState(true);

  const cache = useMemo(() => buildCache(mapError ? 0.01 : 0, 21), [mapError]);

  const init = useCallback((): State => ({ step: 0 }), []);
  const stepFn = useCallback((s: State): State => ({ step: s.step + 1 }), []);
  const sim = useSimulation<State>({ init, step: stepFn, fps: 1.6, initialSeed: 1 });

  const activeK = sweep ? K_LADDER[sim.state.step % K_LADDER.length] : K;
  const post = useMemo(() => posteriorOver(cache, activeK, kappa), [cache, activeK, kappa]);
  /** The same scan, tempered hard enough to stay honest — the reference curve. */
  const honest = useMemo(() => posteriorOver(cache, activeK, 30), [cache, activeK]);

  const corr = useMemo(() => {
    const r = cache.residual;
    const n = r.length;
    if (n < 8) return 0;
    const mu = r.reduce((a, b) => a + b, 0) / n;
    const v = r.reduce((a, b) => a + (b - mu) ** 2, 0) / n;
    const c1 = r.slice(1).reduce((a, x, i) => a + (x - mu) * (r[i] - mu), 0) / (n - 1);
    return v > 0 ? c1 / v : 0;
  }, [cache]);

  const series = useMemo(() => {
    const peak = Math.max(...post.p);
    const peakH = Math.max(...honest.p);
    return [
      {
        id: 'p(x | z) as computed',
        role: 'measurement' as const,
        data: XS.map((x, i) => ({ x: (x - TRUTH.x) * 100, y: post.p[i] / peak })),
      },
      {
        id: 'tempered κ = 30',
        role: 'posterior' as const,
        data: XS.map((x, i) => ({ x: (x - TRUTH.x) * 100, y: honest.p[i] / peakH })),
      },
    ];
  }, [post, honest]);

  return (
    <WidgetFrame
      id="w10.4"
      title="The Overconfidence Meter"
      teaches="More beams do not mean better localization. Multiplying correlated evidence buys certainty without buying information."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          The posterior over one axis of the corridor, computed from a scan of <em>K</em> beams under
          a map that is 1% too wide — about 3 cm of error at the far wall. Zero on the axis is the
          true position. Watch <em>K</em> climb the ladder: the green curve narrows relentlessly while
          its peak sits several centimetres off the truth, and the coverage tile flips to{' '}
          <strong>no</strong> — the truth has left the filter&rsquo;s own 95% interval. Nothing about
          the sensor got worse; the beams simply share an error the product rule cannot see. Now push
          the tempering slider: dividing the log-likelihood by κ restores an honest width, and
          κ ≈ 30 is what it takes here. Turn the map error off and the same K is perfectly safe.
        </>
      }
    >
      <div className="grid grid-cols-2 gap-3 border-b border-fd-border p-3 sm:grid-cols-4">
        <StatTile label="beams used" value={activeK} role="measurement" />
        <StatTile label="posterior σ" value={post.sd * 100} unit="cm" precision={2} role="posterior" />
        <StatTile label="peak offset" value={post.bias * 100} unit="cm" precision={1} role="truth" />
        <StatTile label="truth in 95% set" value={post.covered ? 'yes' : 'no'} />
      </div>

      <div className="px-3 pt-3">
        <LineChart
          series={series}
          xLabel="position error (cm)"
          yLabel="posterior (peak = 1)"
          height={230}
          yMin={0}
          yMax={1.05}
          markers={[{ axis: 'x', value: 0, label: 'true pose', role: 'truth' }]}
          ariaLabel="Posterior over robot position along the corridor. As the beam count rises the curve becomes a narrow spike offset from the true position."
        />
      </div>

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="adjacent-beam residual correlation" value={corr.toFixed(3)} />
        <Stat
          label="σ at κ = 1 vs κ = 30"
          value={`${(post.sd * 100).toFixed(2)} → ${(honest.sd * 100).toFixed(2)} cm`}
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Beams K"
          role="measurement"
          value={activeK}
          min={4}
          max={180}
          step={1}
          onChange={(v) => {
            setSweep(false);
            setK(Math.round(v));
          }}
          format={(v) => v.toFixed(0)}
          help="How many of the 180 beams the model consumes. Moving this stops the automatic sweep."
        />
        <Slider
          label="Tempering κ"
          role="posterior"
          value={kappa}
          min={1}
          max={60}
          step={1}
          onChange={setKappa}
          format={(v) => v.toFixed(0)}
          help="Raise p to the power 1/κ. κ = 1 is the raw product; larger κ discounts evidence the beams do not independently carry."
        />
        <Toggle label="Map is 1% too wide" checked={mapError} onChange={setMapError} />
        <Toggle label="Sweep K automatically" checked={sweep} onChange={setSweep} />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
