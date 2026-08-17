'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { normalPdf } from '@/lib/prob/gaussian';
import { canonicalProduct1, toCanonical1, toMoments1 } from '@/lib/prob/canonical';
import { clear, label, sx, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w2.2 — the Blob Multiplier.
 *
 * Bayes rule with two Gaussians, drawn as what it actually is: a pointwise
 * product. The prior (blue) is Rusty's odometry; the likelihood (green) is the
 * wall-range sensor; the posterior (purple) is their normalized product,
 * computed by the library's `canonicalProduct1` — ξ and ω literally added,
 * exactly as the ledger underneath shows.
 *
 * The likelihood's variance sweeps from "sharper than the prior" to "useless",
 * so the posterior's mean slides all the way from the sensor to the prior
 * without ever leaving the interval between them, and its width stays below
 * both inputs at every moment. Nothing about that is an average.
 */

const X_MIN = 0;
const X_MAX = 12;

/** The chapter's worked example: N(5, 4) × N(6.5, 1) = N(6.2, 0.8). */
const BOOK = { m1: 5, v1: 4, m2: 6.5, v2: 1 };

interface State {
  /** Likelihood variance, swept on a log scale. */
  logV2: number;
  dir: number;
}

interface Params {
  m1: number;
  v1: number;
  m2: number;
}

export function BlobMultiplier() {
  const [params, setParams] = useState<Params>({ m1: BOOK.m1, v1: BOOK.v1, m2: BOOK.m2 });

  const init = useCallback((): State => ({ logV2: Math.log(BOOK.v2), dir: 1 }), []);

  const step = useCallback((s: State): State => {
    let logV2 = s.logV2 + s.dir * 0.06;
    let dir = s.dir;
    if (logV2 > Math.log(24)) {
      logV2 = Math.log(24);
      dir = -1;
    } else if (logV2 < Math.log(0.12)) {
      logV2 = Math.log(0.12);
      dir = 1;
    }
    return { logV2, dir };
  }, []);

  const sim = useSimulation<State>({ init, step, fps: 18, initialSeed: 2 });
  const v2 = Math.exp(sim.state.logV2);

  // The posterior comes from the real library routine, in canonical form —
  // the same three lines the Rust `Canonical::product` runs.
  const fused = useMemo(() => {
    const c1 = toCanonical1(params.m1, params.v1);
    const c2 = toCanonical1(params.m2, v2);
    const c = canonicalProduct1(c1, c2);
    return { c1, c2, c, ...toMoments1(c) };
  }, [params.m1, params.v1, params.m2, v2]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);

      const curves: {
        mean: number;
        variance: number;
        color: string;
        name: string;
        fill: boolean;
      }[] = [
        { mean: params.m1, variance: params.v1, color: p.prior, name: 'prior  p(x)', fill: false },
        {
          mean: params.m2,
          variance: v2,
          color: p.measurement,
          name: 'likelihood  p(z|x)',
          fill: false,
        },
        {
          mean: fused.mean,
          variance: fused.variance,
          color: p.posterior,
          name: 'posterior  ηp(z|x)p(x)',
          fill: true,
        },
      ];

      // A shared vertical scale so "narrower" reads as "taller", honestly.
      // Density has no natural world height, so the vertical axis is pixels:
      // the tallest curve fills the panel and the rest are drawn to match.
      const peak = Math.max(...curves.map((c) => normalPdf(c.mean, c.mean, Math.sqrt(c.variance))));
      const baseY = v.height - 22;
      const topY = 20;
      const yOf = (d: number) => baseY - (d / peak) * (baseY - topY);

      // Baseline.
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, baseY);
      ctx.lineTo(v.width, baseY);
      ctx.stroke();

      for (const c of curves) {
        const std = Math.sqrt(c.variance);
        const trace = () => {
          ctx.beginPath();
          for (let px = 0; px <= v.width; px += 1) {
            const wx = v.minX + (px / v.width) * (v.maxX - v.minX);
            const y = yOf(normalPdf(wx, c.mean, std));
            if (px === 0) ctx.moveTo(px, y);
            else ctx.lineTo(px, y);
          }
        };

        if (c.fill) {
          // Fill only the posterior: it is the answer, the other two are inputs.
          trace();
          ctx.lineTo(v.width, baseY);
          ctx.lineTo(0, baseY);
          ctx.closePath();
          ctx.fillStyle = c.color;
          ctx.globalAlpha = 0.14;
          ctx.fill();
          ctx.globalAlpha = 1;
        }

        trace();
        ctx.strokeStyle = c.color;
        ctx.lineWidth = c.fill ? 2.4 : 1.8;
        ctx.stroke();

        // Mean ticks: the visual claim is that purple sits between blue and green.
        ctx.strokeStyle = c.color;
        ctx.globalAlpha = 0.55;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx(v, c.mean), baseY);
        ctx.lineTo(sx(v, c.mean), yOf(normalPdf(c.mean, c.mean, std)));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      curves.forEach((c, i) => {
        label(ctx, c.name, 10, 14 + i * 14, c.color, { size: 10, weight: 600 });
      });

      for (let x = X_MIN + 2; x <= X_MAX - 2; x += 2) {
        label(ctx, x.toFixed(0), sx(v, x), baseY + 11, p.truth, { size: 9, align: 'center' });
      }
      label(ctx, 'x  (metres along the corridor)', v.width - 8, baseY + 11, p.truth, {
        size: 9,
        align: 'right',
      });
    },
    [params, v2, fused],
  );

  const atBookValues =
    Math.abs(params.m1 - BOOK.m1) < 1e-9 &&
    Math.abs(params.v1 - BOOK.v1) < 1e-9 &&
    Math.abs(params.m2 - BOOK.m2) < 1e-9 &&
    Math.abs(v2 - BOOK.v2) < 5e-3;

  return (
    <WidgetFrame
      id="w2.2"
      title="Blob Multiplier"
      teaches="Bayes fusion does not average two estimates — it precision-weights them, and the answer is always more certain than either input."
      colorKey={['prior', 'measurement', 'posterior']}
      caption={
        <>
          Rusty&apos;s odometry says <em>about 5 m, give or take 2</em>; the wall-range sensor says{' '}
          <em>6.5 m</em> with a confidence that sweeps from sharp to useless. The purple curve is
          the pointwise product of the other two, normalized — that is the whole of Bayes rule.
          Three things to watch. The posterior is <strong>always taller and narrower than both</strong>{' '}
          inputs, at every setting, because precisions add and never subtract. Its mean never
          leaves the segment between the two, sliding toward whichever curve is sharper. And as the
          likelihood widens past σ² ≈ 20 the posterior quietly becomes the prior: a sensor that
          could say anything has said nothing. Press reset to return to the book&apos;s numbers,
          N(5, 4) × N(6.5, 1) = N(6.2, 0.8), and check the ledger by hand.
        </>
      }
    >
      <SimCanvas
        // World aspect matches the canvas aspect exactly, so the horizontal
        // mapping is 1:1 and nothing is letterboxed. Density lives in pixels.
        world={{ minX: X_MIN, maxX: X_MAX, minY: 0, maxY: (X_MAX - X_MIN) / 2.5 }}
        draw={draw}
        deps={[sim.tick, params, fused]}
        aspect={2.5}
        padding={0}
        ariaLabel="Three bell curves on a shared axis: a wide blue prior, a green measurement likelihood whose width oscillates, and a purple posterior that is always narrower and taller than both and lies between their means."
      />

      <CanonicalLedger
        xi1={fused.c1.xi}
        om1={fused.c1.omega}
        xi2={fused.c2.xi}
        om2={fused.c2.omega}
        xi={fused.c.xi}
        om={fused.c.omega}
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="posterior μ" value={fused.mean.toFixed(3)} role="posterior" />
        <Stat label="posterior σ²" value={fused.variance.toFixed(3)} role="posterior" />
        <Stat
          label="worked example"
          value={atBookValues ? 'μ=6.2, σ²=0.8 ✓' : '—'}
        />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Likelihood σ²"
          role="measurement"
          value={v2}
          min={0.12}
          max={24}
          step={0.02}
          onChange={(value) => {
            sim.pause();
            sim.setState(() => ({ logV2: Math.log(value), dir: 1 }));
          }}
          help="Sweeps automatically until you touch it. How much the sensor is trusted."
        />
        <Slider
          label="Prior μ"
          role="prior"
          value={params.m1}
          min={1}
          max={11}
          step={0.1}
          onChange={(value) => setParams((p) => ({ ...p, m1: value }))}
        />
        <Slider
          label="Prior σ²"
          role="prior"
          value={params.v1}
          min={0.2}
          max={12}
          step={0.1}
          onChange={(value) => setParams((p) => ({ ...p, v1: value }))}
        />
        <Slider
          label="Likelihood μ"
          role="measurement"
          value={params.m2}
          min={1}
          max={11}
          step={0.1}
          onChange={(value) => setParams((p) => ({ ...p, m2: value }))}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={() => {
          setParams({ m1: BOOK.m1, v1: BOOK.v1, m2: BOOK.m2 });
          sim.pause();
          sim.setState(() => ({ logV2: Math.log(BOOK.v2), dir: 1 }));
        }}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

/**
 * The same fusion in canonical form, where the arithmetic is addition and
 * nothing else. Bars are proportional so the reader can see the third column
 * being built out of the first two.
 */
function CanonicalLedger({
  xi1,
  om1,
  xi2,
  om2,
  xi,
  om,
}: {
  xi1: number;
  om1: number;
  xi2: number;
  om2: number;
  xi: number;
  om: number;
}) {
  const omScale = Math.max(om, 1e-9);
  const xiScale = Math.max(Math.abs(xi), 1e-9);
  return (
    <div className="border-t border-fd-border px-3 py-2.5">
      <p className="eyebrow mb-1.5">canonical form: multiplication is addition</p>
      <div className="space-y-1.5 font-mono text-[0.7rem] tabular-nums">
        <LedgerRow
          symbol="Ω"
          parts={[
            { value: om1, width: om1 / omScale, role: 'prior' },
            { value: om2, width: om2 / omScale, role: 'measurement' },
          ]}
          total={om}
        />
        <LedgerRow
          symbol="ξ"
          parts={[
            { value: xi1, width: Math.abs(xi1) / xiScale, role: 'prior' },
            { value: xi2, width: Math.abs(xi2) / xiScale, role: 'measurement' },
          ]}
          total={xi}
        />
      </div>
    </div>
  );
}

function LedgerRow({
  symbol,
  parts,
  total,
}: {
  symbol: string;
  parts: { value: number; width: number; role: 'prior' | 'measurement' }[];
  total: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-4 shrink-0 text-fd-muted-foreground">{symbol}</span>
      <div className="flex h-3.5 min-w-0 flex-1 overflow-hidden rounded-[2px] bg-fd-muted">
        {parts.map((part, i) => (
          <div
            key={i}
            style={{
              width: `${Math.max(0, Math.min(100, part.width * 100))}%`,
              backgroundColor: `var(--pr-${part.role})`,
            }}
            title={`${part.value.toFixed(3)}`}
          />
        ))}
      </div>
      <span className="w-40 shrink-0 text-end">
        {parts.map((p) => p.value.toFixed(2)).join('  +  ')} = {total.toFixed(2)}
      </span>
    </div>
  );
}

function Stat({
  label: l,
  value,
  role,
}: {
  label: string;
  value: string;
  role?: 'prior' | 'measurement' | 'posterior';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={role ? { color: `var(--pr-${role})` } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
