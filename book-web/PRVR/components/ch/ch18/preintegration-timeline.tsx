'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';
import { arcState, makeImuArc } from '@/lib/vision/scene';
import {
  GRAVITY,
  biasCorrect,
  deltaSigmas,
  imuResidual,
  preintegrate,
  zeroBias,
  type ImuBias,
  type ImuNoise,
} from '@/lib/vision/preint';
import { logSO3 } from '@/lib/vision/se3';

/**
 * w18.2 — the Preintegration Timeline.
 *
 * Two things every reader gets wrong about an IMU in a factor graph, killed
 * here in one figure:
 *
 *  1. "the optimizer re-integrates the IMU every iteration." It does not. The
 *     ticks between two keyframes compress, once, into a single nine-vector
 *     factor with a 9×9 covariance, and the bias slider re-shapes that factor
 *     through a stored Jacobian in O(1) — the full re-integration (drawn as the
 *     ghost readout) agrees with it to micrometres.
 *  2. "an IMU measures position." It measures angular rate and specific force.
 *     Position is what is left after integrating twice with gravity in the
 *     loop, and the covariance panel shows exactly how fast that costs you.
 *
 * The trajectory is `makeImuArc` — a constant-radius, constant-speed turn whose
 * true IMU output is *constant*, so anything moving on screen is integration,
 * not a changing input.
 */

const RATE = 200;
const OMEGA = 0.5;
const RADIUS = 2;
const NOISE: ImuNoise = { gyro: 0.0035, acc: 0.02 };
/** The IMU's real gyro bias. The reader's job is to find it. */
const TRUE_BG = 0.05;

interface Params {
  /** The optimizer's current guess at b_g,z — the foreground parameter. */
  bg: number;
  /** Keyframe spacing in seconds. */
  seconds: number;
}

const ASPECT = 2.4;
const RAIL = { x0: 0.16, x1: 2.24, y: 0.79 };
const RES = { x0: 0.05, y0: 0.06, x1: 1.34, y1: 0.52 };
const COV = { x0: 1.5, y0: 0.06, x1: 1.92, y1: 0.48 };

const BLOCK_NAMES = ['δφ', 'δv', 'δp'];

export function PreintegrationTimeline() {
  const [params, setParams] = useState<Params>({ bg: 0, seconds: 1 });
  const [seed, setSeed] = useState(18);

  const arc = useMemo(
    () =>
      makeImuArc({
        seconds: params.seconds,
        rate: RATE,
        omega: OMEGA,
        radius: RADIUS,
        bias: { gyro: [0, 0, TRUE_BG], acc: [0, 0, 0] },
        noise: NOISE,
        seed,
      }),
    [params.seconds, seed],
  );

  const bias: ImuBias = useMemo(() => ({ gyro: [0, 0, params.bg], acc: [0, 0, 0] }), [params.bg]);

  const init = useCallback((): { k: number } => ({ k: 0 }), []);
  const step = useCallback(
    (s: { k: number }): { k: number } => {
      const n = arc.samples.length;
      // Fold a fixed fraction of the interval per frame so the sweep takes the
      // same time whatever the keyframe spacing.
      const chunk = Math.max(1, Math.round(n / 60));
      return { k: s.k >= n ? 0 : Math.min(n, s.k + chunk) };
    },
    [arc],
  );
  const sim = useSimulation<{ k: number }>({ init, step, fps: 20, initialSeed: 18 });

  /** The partial factor the sweep has accumulated so far. */
  const live = useMemo(() => {
    const k = Math.min(sim.state.k, arc.samples.length);
    const pre = preintegrate(arc.samples.slice(0, k), arc.dt, zeroBias(), NOISE);
    const corrected = biasCorrect(pre, bias);
    const sj = arcState(k * arc.dt, OMEGA, RADIUS);
    const r = k > 0 ? imuResidual(pre, arc.start, sj, GRAVITY, bias) : new Array(9).fill(0);
    const sigmas = deltaSigmas(pre);
    // Whiten the residual by the factor's own 1σ so the bars are in units the
    // optimizer actually uses: "how surprising is this, given the noise model".
    const whitened = r.map((v, i) => v / Math.sqrt(Math.max(pre.cov[i][i], 1e-12)));
    return { k, pre, corrected, sigmas, whitened, dPhiZ: logSO3(corrected.dR)[2] };
  }, [sim.state.k, arc, bias]);

  /**
   * The ghost: what a full re-integration at the reader's bias would produce.
   * It is O(n) and the corrected factor is O(1), and they agree — which is the
   * entire argument for preintegration.
   */
  const ghost = useMemo(() => {
    const pre = preintegrate(arc.samples, arc.dt, zeroBias(), NOISE);
    const corrected = biasCorrect(pre, bias);
    const exact = preintegrate(arc.samples, arc.dt, bias, NOISE);
    const dp = Math.hypot(
      corrected.dp[0] - exact.dp[0],
      corrected.dp[1] - exact.dp[1],
      corrected.dp[2] - exact.dp[2],
    );
    const dphi = Math.abs(logSO3(corrected.dR)[2] - logSO3(exact.dR)[2]);
    const truePhi = OMEGA * arc.seconds;
    return {
      n: arc.samples.length,
      dpErrorUm: dp * 1e6,
      dphiErrorUrad: dphi * 1e6,
      phiRaw: logSO3(pre.dR)[2],
      phiCorrected: logSO3(corrected.dR)[2],
      truePhi,
    };
  }, [arc, bias]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const W = (wx: number, wy: number): [number, number] => [sx(v, wx), sy(v, wy)];
      const n = arc.samples.length;
      const frac = n > 0 ? live.k / n : 0;

      /* ---- the rail: two keyframes and the raw samples between them ------- */
      const railX = (t: number) => RAIL.x0 + t * (RAIL.x1 - RAIL.x0);
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...W(RAIL.x0, RAIL.y));
      ctx.lineTo(...W(RAIL.x1, RAIL.y));
      ctx.stroke();

      // Raw ticks. At 200 Hz there are hundreds; draw a representative comb and
      // say the true count in words.
      const shown = Math.min(n, 130);
      for (let i = 0; i < shown; i++) {
        const t = shown === 1 ? 0 : i / (shown - 1);
        const folded = t <= frac;
        const h = 0.055 + 0.02 * Math.abs(Math.sin(i * 1.7));
        ctx.strokeStyle = folded ? p.truth : p.measurement;
        ctx.globalAlpha = folded ? 0.3 : 0.9;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(...W(railX(t), RAIL.y));
        ctx.lineTo(...W(railX(t), RAIL.y + h));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      label(
        ctx,
        `${n} raw samples at ${RATE} Hz — ω̃ and ã, six numbers each`,
        sx(v, RAIL.x1),
        sy(v, RAIL.y + 0.115),
        p.measurement,
        { size: 10, align: 'right' },
      );

      // The keyframe posts.
      for (const [x, name, sub] of [
        [RAIL.x0, 'keyframe i', 'R_i, v_i, p_i, b'],
        [RAIL.x1, 'keyframe j', 'R_j, v_j, p_j'],
      ] as const) {
        ctx.strokeStyle = p.prior;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(...W(x, RAIL.y - 0.1));
        ctx.lineTo(...W(x, RAIL.y + 0.14));
        ctx.stroke();
        label(ctx, name, sx(v, x), sy(v, RAIL.y + 0.185), p.prior, { size: 10, align: 'center' });
        label(ctx, sub, sx(v, x), sy(v, RAIL.y - 0.135), p.truth, { size: 9, align: 'center' });
      }

      // The factor blob: one edge of the graph, growing as samples fold in.
      const cx = (RAIL.x0 + RAIL.x1) / 2;
      const halfW = 0.06 + 0.4 * frac;
      const halfH = 0.028 + 0.026 * frac;
      ctx.fillStyle = p.posterior;
      ctx.globalAlpha = 0.16;
      ctx.beginPath();
      ctx.ellipse(
        ...W(cx, RAIL.y - 0.045),
        sl(v, halfW),
        sl(v, halfH),
        0,
        0,
        Math.PI * 2,
      );
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = p.posterior;
      ctx.lineWidth = 1.6;
      ctx.stroke();
      label(
        ctx,
        frac < 1 ? `folding… ${live.k}/${n}` : 'one factor: (ΔR, Δv, Δp), Σ₉ₓ₉, J_b',
        sx(v, cx),
        sy(v, RAIL.y - 0.045),
        p.posterior,
        { size: 10, align: 'center' },
      );

      /* ---- residual panel -------------------------------------------------- */
      panelFrame(ctx, v, p, RES, 'whitened IMU residual  r / σ');
      const bars = live.whitened;
      const bw = (RES.x1 - RES.x0 - 0.1) / 9;
      const mid = (RES.y0 + RES.y1) / 2 - 0.02;
      const maxAbs = Math.max(3, ...bars.map((b) => Math.abs(b)));
      const scaleY = (RES.y1 - mid - 0.06) / maxAbs;

      // The ±1σ band: inside it the factor is content, outside it is shouting.
      ctx.fillStyle = p.truth;
      ctx.globalAlpha = 0.12;
      ctx.fillRect(
        sx(v, RES.x0 + 0.05),
        sy(v, mid + scaleY),
        sl(v, RES.x1 - RES.x0 - 0.1),
        sl(v, 2 * scaleY),
      );
      ctx.globalAlpha = 1;
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(...W(RES.x0 + 0.05, mid));
      ctx.lineTo(...W(RES.x1 - 0.05, mid));
      ctx.stroke();

      bars.forEach((b, i) => {
        const x = RES.x0 + 0.05 + i * bw;
        const h = Math.max(Math.min(b, maxAbs), -maxAbs) * scaleY;
        ctx.fillStyle = i < 3 ? p.posterior : i < 6 ? p.prediction : p.prior;
        ctx.globalAlpha = Math.abs(b) > 1 ? 0.95 : 0.55;
        ctx.fillRect(
          sx(v, x + bw * 0.18),
          sy(v, mid + Math.max(h, 0)),
          sl(v, bw * 0.64),
          Math.max(sl(v, Math.abs(h)), 1),
        );
        ctx.globalAlpha = 1;
      });
      for (let g = 0; g < 3; g++) {
        label(
          ctx,
          BLOCK_NAMES[g],
          sx(v, RES.x0 + 0.05 + (3 * g + 1.5) * bw),
          sy(v, RES.y0 + 0.05),
          g === 0 ? p.posterior : g === 1 ? p.prediction : p.prior,
          { size: 10, align: 'center' },
        );
      }
      label(ctx, '±1σ', sx(v, RES.x1 - 0.06), sy(v, mid + scaleY + 0.02), p.truth, {
        size: 9,
        align: 'right',
      });

      /* ---- covariance panel ------------------------------------------------ */
      panelFrame(
        ctx,
        v,
        p,
        { x0: COV.x0 - 0.03, y0: RES.y0, x1: 2.35, y1: RES.y1 },
        'Σ₉ₓ₉ of (δφ, δv, δp)',
      );
      const cell = (COV.x1 - COV.x0) / 9;
      const dmax = Math.max(...live.pre.cov.map((row, i) => Math.abs(row[i])), 1e-12);
      for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
          // Correlation-normalized magnitude: the *pattern* is the lesson, and
          // the raw entries span six orders of magnitude.
          const dii = Math.sqrt(Math.max(live.pre.cov[i][i], 1e-18));
          const djj = Math.sqrt(Math.max(live.pre.cov[j][j], 1e-18));
          const c = Math.min(1, Math.abs(live.pre.cov[i][j]) / (dii * djj));
          ctx.fillStyle = p.prior;
          ctx.globalAlpha = 0.08 + 0.92 * c;
          ctx.fillRect(
            sx(v, COV.x0 + j * cell),
            sy(v, COV.y1 - i * cell),
            Math.ceil(sl(v, cell)) - 1,
            Math.ceil(sl(v, cell)) - 1,
          );
        }
      }
      ctx.globalAlpha = 1;
      for (let g = 1; g < 3; g++) {
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(...W(COV.x0 + 3 * g * cell, COV.y1));
        ctx.lineTo(...W(COV.x0 + 3 * g * cell, COV.y1 - 9 * cell));
        ctx.moveTo(...W(COV.x0, COV.y1 - 3 * g * cell));
        ctx.lineTo(...W(COV.x0 + 9 * cell, COV.y1 - 3 * g * cell));
        ctx.stroke();
      }
      const rows: [string, string][] = [
        ['σ_φ', `${(live.sigmas.phi * 1e3).toFixed(2)} mrad`],
        ['σ_v', `${(live.sigmas.v * 100).toFixed(2)} cm/s`],
        ['σ_p', `${(live.sigmas.p * 100).toFixed(2)} cm`],
        ['max |ρ|', dmax > 0 ? offDiagonalPeak(live.pre.cov).toFixed(2) : '—'],
      ];
      rows.forEach(([k, val], i) => {
        label(ctx, k, sx(v, 2.0), sy(v, COV.y1 - 0.02 - i * 0.1), p.truth, { size: 9.5 });
        label(ctx, val, sx(v, 2.34), sy(v, COV.y1 - 0.02 - i * 0.1), p.prior, {
          size: 9.5,
          align: 'right',
        });
      });
    },
    [arc, live],
  );

  const residualPhiZ = live.whitened[2];

  return (
    <WidgetFrame
      id="w18.2"
      title="Preintegration Timeline"
      teaches="Hundreds of IMU samples compress into one factor between two keyframes — and a change of bias re-shapes that factor in O(1), without touching a single raw sample."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          The green comb is raw IMU data; the purple blob is the single factor it becomes. Watch the
          9×9 covariance fill in as the sweep folds samples — and note the off-diagonal blocks: an
          attitude error <em>leaks</em> into velocity and position, which is why σ<sub>p</sub> grows
          like Δt<sup>3/2</sup> from accelerometer noise alone but faster still, toward
          Δt<sup>5/2</sup>, once gyro error tilts the integrated acceleration. The IMU here has a
          real gyro bias of{' '}
          <strong>+0.050 rad/s</strong> about z, and the factor was integrated at zero. Slide{' '}
          <strong>b<sub>g,z</sub></strong> and watch the first residual group collapse into the ±1σ
          band as you pass 0.050 — nothing is re-integrated to do that, only a stored Jacobian is
          applied. The ghost readout below reports what the expensive re-integration would have
          given: the two agree to micrometres.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0, minY: 0, maxX: ASPECT, maxY: 1 }}
        draw={draw}
        deps={[sim.tick, live, arc]}
        aspect={ASPECT}
        padding={0}
        ariaLabel="A timeline between two keyframes with hundreds of IMU sample ticks folding into a single factor, beside a bar chart of the whitened nine-vector residual and a 9 by 9 covariance block."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Stat label="samples in this factor" value={`${ghost.n}`} tone="measurement" />
        <Stat
          label="Δφ_z: raw / corrected / true"
          value={`${ghost.phiRaw.toFixed(3)} / ${ghost.phiCorrected.toFixed(3)} / ${ghost.truePhi.toFixed(3)}`}
          tone="posterior"
        />
        <Stat
          label="O(1) correction vs re-integration"
          value={`${ghost.dpErrorUm.toFixed(1)} µm`}
          tone="prediction"
        />
        <Stat
          label="rotation residual r_φz"
          value={`${residualPhiZ.toFixed(1)} σ`}
          tone={Math.abs(residualPhiZ) > 1 ? 'posterior' : 'measurement'}
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Estimated gyro bias b_g,z"
          role="prediction"
          value={params.bg}
          min={-0.1}
          max={0.1}
          step={0.002}
          unit="rad/s"
          format={(x) => x.toFixed(3)}
          onChange={(bg) => setParams((q) => ({ ...q, bg }))}
          help="The foreground parameter. The true bias is +0.050 rad/s; the factor was preintegrated at zero."
        />
        <Slider
          label="Keyframe spacing Δt_ij"
          role="truth"
          value={params.seconds}
          min={0.25}
          max={2}
          step={0.05}
          unit="s"
          onChange={(seconds) => setParams((q) => ({ ...q, seconds }))}
          help="Twice the interval is four times the samples — and far more than twice the uncertainty."
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => setSeed(Math.floor(Math.random() * 100000))}
        seed={seed}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

/* -------------------------------------------------------------------------- */

function panelFrame(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  p: Palette,
  box: { x0: number; y0: number; x1: number; y1: number },
  title: string,
) {
  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(
    sx(v, box.x0),
    sy(v, box.y1),
    sl(v, box.x1 - box.x0),
    sl(v, box.y1 - box.y0),
  );
  label(ctx, title, sx(v, box.x0 + 0.02), sy(v, box.y1 - 0.035), p.truth, { size: 10 });
}

/** The largest off-diagonal correlation — how coupled the nine errors are. */
function offDiagonalPeak(cov: number[][]): number {
  let peak = 0;
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 9; j++) {
      if (i === j) continue;
      const d = Math.sqrt(Math.max(cov[i][i], 1e-18) * Math.max(cov[j][j], 1e-18));
      peak = Math.max(peak, Math.abs(cov[i][j]) / d);
    }
  }
  return peak;
}

function Stat({
  label: l,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: 'measurement' | 'posterior' | 'prior' | 'prediction';
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={tone ? { color: `var(--pr-${tone})` } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
