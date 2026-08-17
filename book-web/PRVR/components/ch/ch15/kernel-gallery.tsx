'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile } from '@/components/viz';
import type { BookRole } from '@/lib/chart-theme';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { kernelInfluence, kernelRho, kernelWeight, type Kernel } from '@/lib/optim/kernels';

/**
 * w15.4 — the Kernel Gallery.
 *
 * Left: the influence function ψ(e) = ρ′(e), which is literally "how hard may
 * one residual pull". Right: the objective a robust fit is actually minimizing,
 * as the outlier is dragged away. L2's minimum follows the outlier; Huber's
 * resists; the redescending kernels ignore it — and grow a *second* basin
 * around it, which is the bill for that immunity.
 */

const SIGMA = 0.15;
const TRUE_MU = 2.0;
const N_INLIERS = 12;

interface Entry {
  id: string;
  kernel: Kernel;
  role: BookRole;
}

const GALLERY: Entry[] = [
  { id: 'L2', kernel: { type: 'l2' }, role: 'prediction' },
  { id: 'Huber k=1.5', kernel: { type: 'huber', k: 1.5 }, role: 'measurement' },
  { id: 'Cauchy c=1.5', kernel: { type: 'cauchy', c: 1.5 }, role: 'prior' },
  { id: 'Geman–McClure c=1.5', kernel: { type: 'geman', c: 1.5 }, role: 'posterior' },
];

/** IRLS for the simplest possible model: one constant, weighted by the kernel. */
function irlsMean(data: number[], kernel: Kernel, mu0: number, iterations = 60): number {
  let mu = mu0;
  for (let it = 0; it < iterations; it++) {
    let num = 0;
    let den = 0;
    for (const y of data) {
      const w = kernelWeight(kernel, Math.abs(y - mu) / SIGMA);
      num += w * y;
      den += w;
    }
    if (den < 1e-12) break;
    const next = num / den;
    if (Math.abs(next - mu) < 1e-12) return next;
    mu = next;
  }
  return mu;
}

export function KernelGallery() {
  const [manual, setManual] = useState<number | null>(null);

  const inliers = useMemo(() => {
    const rng = new Rng(15);
    return Array.from({ length: N_INLIERS }, () => TRUE_MU + rng.normal(0, SIGMA));
  }, []);

  // Autoplay drags the outlier out and back, so the reader sees the divergence
  // between the kernels without touching anything.
  const sim = useSimulation<{ i: number }>({
    init: () => ({ i: 0 }),
    step: (s) => ({ i: (s.i + 1) % 48 }),
    fps: 4,
  });
  const sweep = 0.1 + 2.3 * (1 - Math.cos((2 * Math.PI * sim.state.i) / 48)) * 0.5;
  const offset = manual ?? sweep;

  const data = useMemo(() => [...inliers, TRUE_MU + offset], [inliers, offset]);

  const influence = useMemo(
    () =>
      GALLERY.map((g) => ({
        id: g.id,
        role: g.role,
        data: Array.from({ length: 121 }, (_, i) => {
          const e = -6 + (12 * i) / 120;
          return { x: e, y: kernelInfluence(g.kernel, e) };
        }),
      })),
    [],
  );

  const landscape = useMemo(
    () =>
      GALLERY.map((g) => ({
        id: g.id,
        role: g.role,
        data: Array.from({ length: 161 }, (_, i) => {
          const mu = 1.4 + (1.4 + offset) * (i / 160);
          let j = 0;
          for (const y of data) j += kernelRho(g.kernel, Math.abs(y - mu) / SIGMA);
          return { x: mu, y: Math.log10(1 + j) };
        }),
      })),
    [data, offset],
  );

  const fits = useMemo(
    () =>
      GALLERY.map((g) => {
        const median = [...data].sort((a, b) => a - b)[Math.floor(data.length / 2)];
        return {
          id: g.id,
          role: g.role,
          good: irlsMean(data, g.kernel, median),
          bad: irlsMean(data, g.kernel, TRUE_MU + offset),
        };
      }),
    [data, offset],
  );

  const onSlider = useCallback(
    (v: number) => {
      sim.pause();
      setManual(v);
    },
    [sim],
  );

  return (
    <WidgetFrame
      id="w15.4"
      title="Kernel Gallery"
      teaches="Robust kernels are not a free lunch: the ones that ignore an outlier completely are exactly the ones that make the objective non-convex."
      colorKey={['prior', 'prediction', 'measurement', 'posterior']}
      wide
      caption={
        <>
          Left: the influence ψ(e) = ρ′(e) — how hard a residual of e sigmas is allowed to pull.
          L2&apos;s (orange) is the identity: it grows without bound, which is the whole problem.
          Huber (green) saturates at k, so a 30σ outlier still pulls, just no harder than a 1.5σ one.
          Cauchy (blue) and Geman–McClure (purple) <em>redescend</em> to zero. Right: the actual
          objective for fitting one constant to twelve good measurements and one bad one, as the bad
          one is dragged away. The L2 minimum slides after it. The redescending curves keep their
          minimum on the truth — and grow a second basin around the outlier. Read the two estimate
          rows below: from a sensible start every robust kernel is fine; started <em>at</em> the
          outlier, Geman–McClure stays there forever. That is the bill, and graduated non-convexity
          — annealing the kernel from convex to redescending — is how modern systems pay it.
        </>
      }
    >
      <div className="grid gap-0 lg:grid-cols-2 lg:divide-x lg:divide-fd-border">
        <div className="p-3">
          <LineChart
            series={influence}
            xLabel="whitened residual e (σ)"
            yLabel="influence ψ(e) = ρ′(e)"
            height={250}
            legend
            ariaLabel="Influence functions: L2 grows linearly without bound, Huber saturates, Cauchy and Geman-McClure return to zero."
          />
        </div>
        <div className="border-t border-fd-border p-3 lg:border-t-0">
          <LineChart
            series={landscape}
            xLabel="estimate μ"
            yLabel="log₁₀(1 + J)"
            height={250}
            legend={false}
            markers={[
              { axis: 'x', value: TRUE_MU, label: 'truth', role: 'truth' },
              { axis: 'x', value: TRUE_MU + offset, label: 'outlier' },
            ]}
            ariaLabel="The robust objective as a function of the estimate; redescending kernels show a second local minimum at the outlier."
          />
        </div>
      </div>

      <div className="px-3 pt-3">
        <Dashboard columns={4}>
          {fits.map((f) => (
            <StatTile
              key={f.id}
              label={f.id}
              value={f.good}
              unit="m"
              role={f.role}
              precision={3}
              trend={f.bad - f.good}
              trendLabel={`from a bad init: ${f.bad.toFixed(3)}`}
            />
          ))}
        </Dashboard>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Outlier distance from the truth"
          role="prediction"
          value={offset}
          min={0}
          max={2.4}
          step={0.05}
          unit="m"
          onChange={onSlider}
          help="One measurement out of thirteen is wrong by this much. Everything else is honest."
        />
        <div className="flex items-end">
          <p className="font-ui text-[0.78rem] leading-snug text-fd-muted-foreground">
            {offset < 0.35
              ? 'At this distance nothing is an outlier yet, and every kernel agrees — which is the point: a robust kernel must not change the answer when nothing is wrong.'
              : offset < 1.2
                ? 'The L2 estimate has already left the truth. Huber lags behind it; the redescending kernels have not moved.'
                : 'L2 is now visibly wrong, Huber is biased but bounded, and the redescending kernels have a second minimum sitting on the outlier.'}
          </p>
        </div>
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={() => {
          setManual(null);
          sim.toggle();
        }}
        onStep={sim.stepOnce}
        onReset={() => {
          setManual(null);
          sim.reset();
        }}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}
