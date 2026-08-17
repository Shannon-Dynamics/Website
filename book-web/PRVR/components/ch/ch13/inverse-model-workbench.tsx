'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile } from '@/components/viz';
import {
  DEFAULT_INVERSE_MODEL,
  inverseRangeSensorModel,
  logOddsToProb,
  probToLogOdds,
  type InverseModelParams,
} from '@/lib/mapping/occgrid';
import {
  learnedInverseLogOdds,
  trainInverseSensorModel,
  DEFAULT_LEARN_OPTIONS,
} from '@/lib/mapping/inverse-learned';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w13.3 — the Inverse-Model Workbench.
 *
 * One frozen beam, and the two ways of answering "what does this reading say
 * about that cell?". On the left of the tabs: Table 9.2, three hard regions and
 * two constants somebody chose. On the right: a logistic regression fitted to
 * triplets sampled from the *forward* model (§9.3), which was never told that α
 * or β exist and disagrees with the hand-crafted model exactly where intuition
 * is weakest — at grazing bearings, at long range, and at z_max.
 */

const MAX_RANGE = DEFAULT_LEARN_OPTIONS.maxRange;
const PAINT_CELL = 0.075;
const VIEW = { minX: -0.25, minY: -1.95, maxX: 5.25, maxY: 1.95 };
/** Beyond this bearing the model is extrapolating; the map keeps its prior. */
const PAINT_FOV = (58 * Math.PI) / 180;

const SENSOR = { x: 0, y: 0, theta: 0 };

export function InverseModelWorkbench() {
  const [alpha, setAlpha] = useState(0.35);
  const [beta, setBeta] = useState((22 * Math.PI) / 180);
  const [z, setZ] = useState(2.6);
  const [learned, setLearned] = useState(false);

  // Trained once, deterministically, from seed 0xC0FFEE. Every reader sees the
  // same weights; the "re-roll" of this widget is the training seed itself.
  const model = useMemo(() => trainInverseSensorModel(0xc0ffee), []);

  const params = useMemo<InverseModelParams>(
    () => ({
      ...DEFAULT_INVERSE_MODEL,
      alpha,
      beta,
      maxRange: MAX_RANGE,
      lOcc: probToLogOdds(0.75),
      lFree: probToLogOdds(0.3),
      l0: 0,
      clamp: undefined,
    }),
    [alpha, beta],
  );

  /** ℓ for one cell under whichever model is selected. */
  const evidence = useCallback(
    (cx: number, cy: number, useLearned: boolean): number => {
      if (useLearned) {
        const r = Math.hypot(cx, cy);
        const psi = Math.atan2(cy, cx);
        return learnedInverseLogOdds(model, r, psi, z);
      }
      return inverseRangeSensorModel([cx, cy], SENSOR, [z], [0], params);
    },
    [model, params, z],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const w = Math.ceil(sl(v, PAINT_CELL)) + 1;
      const nx = Math.ceil((VIEW.maxX - VIEW.minX) / PAINT_CELL);
      const ny = Math.ceil((VIEW.maxY - VIEW.minY) / PAINT_CELL);

      for (let j = 0; j < ny; j++) {
        const cy = VIEW.minY + (j + 0.5) * PAINT_CELL;
        for (let i = 0; i < nx; i++) {
          const cx = VIEW.minX + (i + 0.5) * PAINT_CELL;
          const psi = Math.atan2(cy, cx);
          const inField = Math.abs(psi) <= PAINT_FOV && Math.hypot(cx, cy) <= MAX_RANGE;
          const l = inField ? evidence(cx, cy, learned) : 0;
          const pr = logOddsToProb(l);
          // Grayscale, the field's universal convention: the darkness of a cell
          // is the probability it is occupied (Thrun et al., Figure 9.2).
          const shade = Math.round(255 * (1 - pr));
          ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
          ctx.fillRect(
            sx(v, VIEW.minX + i * PAINT_CELL),
            sy(v, VIEW.minY + (j + 1) * PAINT_CELL),
            w,
            w,
          );
        }
      }

      // The beam itself: axis out to the reading, the reading as a dot, and —
      // for the hand-crafted model only — the ±β/2 cone it was told to use.
      ctx.save();
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(sx(v, 0), sy(v, 0));
      ctx.lineTo(sx(v, z), sy(v, 0));
      ctx.stroke();
      if (!learned) {
        ctx.setLineDash([4, 4]);
        ctx.globalAlpha = 0.7;
        ctx.lineWidth = 1.2;
        for (const s of [-1, 1]) {
          ctx.beginPath();
          ctx.moveTo(sx(v, 0), sy(v, 0));
          ctx.lineTo(
            sx(v, MAX_RANGE * Math.cos((s * beta) / 2)),
            sy(v, MAX_RANGE * Math.sin((s * beta) / 2)),
          );
          ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
      ctx.fillStyle = p.measurement;
      ctx.beginPath();
      ctx.arc(sx(v, z), sy(v, 0), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      label(ctx, learned ? 'learned (§9.3)' : 'hand-crafted (Table 9.2)', sx(v, -0.1), sy(v, 1.72), p.ink, {
        size: 11,
        weight: 700,
      });
      label(
        ctx,
        z >= MAX_RANGE - 1e-6 ? 'z = z_max (no return)' : `z = ${z.toFixed(2)} m`,
        sx(v, z),
        sy(v, -0.28),
        p.measurement,
        { size: 10, align: 'center' },
      );
    },
    [beta, evidence, learned, z],
  );

  // ---- the profile along the beam axis ----------------------------------

  const profile = useMemo(() => {
    const hand: { x: number; y: number }[] = [];
    const learn: { x: number; y: number }[] = [];
    for (let r = 0.05; r <= MAX_RANGE; r += 0.025) {
      hand.push({ x: r, y: evidence(r, 0, false) });
      learn.push({ x: r, y: evidence(r, 0, true) });
    }
    return [
      { id: 'hand-crafted (Table 9.2)', role: 'measurement' as const, data: hand },
      { id: 'learned (§9.3)', role: 'posterior' as const, data: learn },
    ];
  }, [evidence]);

  const atReading = evidence(z, 0, learned);
  const behind = evidence(Math.min(z + 0.8, MAX_RANGE - 0.05), 0, learned);
  const grazing = evidence(z * Math.cos(beta / 2 + 0.06), z * Math.sin(beta / 2 + 0.06), learned);

  return (
    <WidgetFrame
      id="w13.3"
      title="Inverse-Model Workbench"
      teaches="The inverse sensor model is a design choice, not a law — and the model you get by fitting the forward one disagrees with the hand-crafted version exactly where you would have guessed wrong."
      colorKey={['prior', 'measurement', 'posterior']}
      caption={
        <>
          One beam, frozen, pointing right; the shading is p(m<sub>i</sub> | z, x) — black occupied,
          white free, mid-gray &ldquo;this reading says nothing&rdquo;. The hand-crafted model paints
          three flat regions with two hard edges, at ranges α apart and bearings β apart, both of
          which <em>you</em> chose. Switch to the learned model — a logistic regression fitted to
          24 000 triplets sampled from the Chapter 10 <em>forward</em> model, with no α and no β
          anywhere in it — and four things change that nobody wrote down. The edges go soft. The
          occupied band sits a little <em>past</em> the reading, because a cone returns the nearest
          point of a surface that keeps going. The band is narrower than β, because a return is most
          likely to have come from straight ahead. And far beyond the reading the curve flattens onto
          its own ℓ₀ of about −0.3 rather than 0 — the model inherited the prior of the maps it was
          trained on, and Table 9.1 must subtract <em>that</em> number, not zero. Finally press{' '}
          <strong>Max-range reading</strong>: the hand-crafted model carves free space with
          undiminished confidence all the way to z<sub>max</sub>, and the learned one carves it at
          well under half strength, because in training one reading in twenty was a dropout from a
          wall that was really there.
        </>
      }
    >
      <SimCanvas
        world={VIEW}
        draw={draw}
        deps={[alpha, beta, z, learned, model]}
        aspect={5.5 / 3.9}
        padding={0.05}
        ariaLabel="The inverse sensor model painted around one beam: free space along the ray, an occupied band at the reading, and the prior beyond it."
      />

      <div className="px-3 pt-3">
        <LineChart
          series={profile}
          xLabel="range r along the beam axis (m)"
          yLabel="evidence ℓ (nats)"
          height={210}
          markers={[
            { axis: 'y', value: 0, label: 'ℓ₀ = 0 (hand-crafted)', role: 'prior' },
            { axis: 'y', value: model.l0, label: 'ℓ₀ learned', role: 'posterior' },
            { axis: 'x', value: z, label: 'reading z', role: 'truth' },
          ]}
          ariaLabel="Evidence along the beam axis for both models: the hand-crafted model is a step function, the learned model a smooth curve that peaks near the reading and decays to zero beyond it."
        />
      </div>

      <div className="px-3 pb-3">
        <Dashboard columns={4}>
          <StatTile label="ℓ at the reading" value={atReading} role="measurement" precision={2} />
          <StatTile label="ℓ 0.8 m beyond" value={behind} precision={2} />
          <StatTile label="ℓ just outside the cone" value={grazing} precision={2} />
          <StatTile
            label="learned ℓ₀ · cross-entropy"
            value={`${model.l0.toFixed(2)} · ${model.loss.toFixed(3)}`}
            role="posterior"
          />
        </Dashboard>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Obstacle thickness α"
          role="measurement"
          value={alpha}
          min={0.1}
          max={1}
          step={0.05}
          unit="m"
          onChange={setAlpha}
          help="Hand-crafted model only. Thicker walls, blurrier maps, fewer holes."
        />
        <Slider
          label="Beam width β"
          value={(beta * 180) / Math.PI}
          min={2}
          max={40}
          step={1}
          unit="°"
          format={(v) => v.toFixed(0)}
          onChange={(v) => setBeta((v * Math.PI) / 180)}
          help="Hand-crafted model only. A LiDAR is a fraction of a degree; a sonar is 15–30°."
        />
        <Slider
          label="Reading z"
          role="measurement"
          value={z}
          min={0.4}
          max={MAX_RANGE}
          step={0.05}
          unit="m"
          onChange={setZ}
        />
        <Toggle
          label="Learned inverse model"
          role="posterior"
          checked={learned}
          onChange={setLearned}
        />
      </ControlPanel>

      <div className="border-t border-fd-border px-3 py-2">
        <ButtonRow>
          <ActionButton onClick={() => setZ(MAX_RANGE)}>Max-range reading</ActionButton>
          <ActionButton onClick={() => setZ(1.0)}>Close obstacle</ActionButton>
          <ActionButton onClick={() => setZ(4.2)}>Far obstacle</ActionButton>
        </ButtonRow>
      </div>
    </WidgetFrame>
  );
}
