'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Transport } from '@/components/sim/controls';
import { LineChart } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { angleDiff, normalizeAngle } from '@/lib/geom/se2';
import { clear, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w3.3 — the Wrap-Around Trap, and the chapter's hook.
 *
 * Two headings, one question: what is the average? The arithmetic mean is a
 * perfectly good answer on the real line and a catastrophe on the circle: for
 * two headings it is either exactly right or exactly backwards, and nothing in
 * between. The repair is the whole chapter in miniature — subtract on the
 * manifold (⊟), scale in the tangent space, add back on the manifold (⊞).
 *
 * Everything here runs the library's `angleDiff` / `normalizeAngle`, which are
 * the SO(2) specializations of `se2Log` and `se2Exp`.
 */

const DEG = 180 / Math.PI;
const RAD = Math.PI / 180;

interface State {
  /** The swept heading, in radians. */
  thetaB: number;
}

/** θ_A ⊞ ½(θ_B ⊟ θ_A) — the retraction-based mean, correct on the circle. */
function manifoldMean(a: number, b: number): number {
  return normalizeAngle(a + 0.5 * angleDiff(b, a));
}

export function WrapAroundTrap() {
  // The one foregrounded parameter: where needle A points. The default puts it
  // one degree short of the ±180° cut, which is where the trap lives.
  const [thetaADeg, setThetaADeg] = useState(179);
  const thetaA = thetaADeg * RAD;

  const init = useCallback((): State => ({ thetaB: -179 * RAD }), []);
  const step = useCallback(
    (s: State): State => ({ thetaB: normalizeAngle(s.thetaB + 2.5 * RAD) }),
    [],
  );

  const sim = useSimulation<State>({ init, step, fps: 20, initialSeed: 3 });
  const { thetaB } = sim.state;

  const naive = (thetaA + thetaB) / 2; // no wrapping: the bug, written out
  const correct = manifoldMean(thetaA, thetaB);
  const errorDeg = Math.abs(angleDiff(naive, correct)) * DEG;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const cx = sx(v, 0);
      const cy = sy(v, 0);
      const R = sl(v, 1);

      // ---- compass rose -------------------------------------------------
      ctx.strokeStyle = p.grid;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.stroke();

      for (let d = 0; d < 360; d += 15) {
        const a = d * RAD;
        const long = d % 90 === 0;
        const r0 = R * (long ? 0.9 : 0.955);
        ctx.strokeStyle = long ? p.truth : p.grid;
        ctx.lineWidth = long ? 1.5 : 1;
        ctx.beginPath();
        ctx.moveTo(cx + r0 * Math.cos(a), cy - r0 * Math.sin(a));
        ctx.lineTo(cx + R * Math.cos(a), cy - R * Math.sin(a));
        ctx.stroke();
      }
      for (const [d, text] of [
        [0, '0°'],
        [90, '90°'],
        [180, '±180°'],
        [270, '−90°'],
      ] as [number, string][]) {
        const a = d * RAD;
        label(ctx, text, cx + R * 1.12 * Math.cos(a), cy - R * 1.12 * Math.sin(a), p.truth, {
          size: 10,
          align: 'center',
        });
      }

      // The branch cut: the seam where (−π, π] wraps. Every bug in this widget
      // is a bug about crossing this line.
      ctx.strokeStyle = p.truth;
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx - R * 1.02, cy);
      ctx.stroke();
      ctx.setLineDash([]);

      // ---- the short rotation from A to B, i.e. θ_B ⊟ θ_A ----------------
      const delta = angleDiff(thetaB, thetaA);
      ctx.strokeStyle = p.posterior;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.arc(cx, cy, R * 0.55, -thetaA, -(thetaA + delta), delta > 0);
      ctx.stroke();
      ctx.globalAlpha = 1;

      const needle = (angle: number, len: number, color: string, text: string, dashed = false) => {
        const ex = cx + R * len * Math.cos(angle);
        const ey = cy - R * len * Math.sin(angle);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2.5;
        if (dashed) ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(ex, ey, 4, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, text, ex + 10 * Math.cos(angle), ey - 10 * Math.sin(angle), color, {
          size: 11,
          weight: 700,
          align: 'center',
        });
        ctx.restore();
      };

      needle(naive, 0.74, p.prediction, 'mean', true);
      needle(correct, 0.9, p.posterior, '⊞⊟');
      needle(thetaA, 1, p.prior, 'A');
      needle(thetaB, 1, p.measurement, 'B');

      ctx.fillStyle = p.ink;
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();

      // ---- readout column ------------------------------------------------
      const tx = sx(v, 1.45);
      let ty = sy(v, 0.92);
      const line = (text: string, color: string, weight = 500) => {
        label(ctx, text, tx, ty, color, { size: 11.5, weight });
        ty += 22;
      };
      line(`A          ${fmt(thetaA)}`, p.prior, 700);
      line(`B          ${fmt(thetaB)}`, p.measurement, 700);
      ty += 6;
      line(`(A + B)/2  ${fmt(naive)}`, p.prediction, 700);
      line(`A ⊞ ½(B ⊟ A)  ${fmt(correct)}`, p.posterior, 700);
      ty += 6;
      line(`B ⊟ A      ${fmt(delta)}`, p.ink);
      line(
        `disagreement  ${errorDeg.toFixed(1)}°`,
        errorDeg > 1 ? p.prediction : p.truth,
        700,
      );
    },
    [thetaA, thetaB, naive, correct, errorDeg],
  );

  const onPointer = useCallback(
    (world: [number, number]) => {
      const [wx, wy] = world;
      if (Math.hypot(wx, wy) > 1.35) return;
      sim.pause();
      sim.setState(() => ({ thetaB: Math.atan2(wy, wx) }));
    },
    [sim],
  );

  // The chart is a function of θ_A only, so dragging the slider reshapes it
  // while the animation of θ_B stays cheap.
  const series = useMemo(() => {
    const naiveData: { x: number; y: number }[] = [];
    const trueData: { x: number; y: number }[] = [];
    for (let d = -180; d <= 180; d += 2) {
      const b = d * RAD;
      naiveData.push({ x: d, y: ((b - thetaA) / 2) * DEG });
      trueData.push({ x: d, y: (angleDiff(b, thetaA) / 2) * DEG });
    }
    return [
      { id: 'arithmetic mean', role: 'prediction' as const, data: naiveData },
      { id: '⊟ then ⊞', role: 'posterior' as const, data: trueData },
    ];
  }, [thetaA]);

  return (
    <WidgetFrame
      id="w3.3"
      title="The Wrap-Around Trap"
      teaches="Angles are not numbers. Averaging two headings by adding them and halving is either exactly right or exactly backwards."
      colorKey={['prior', 'measurement', 'prediction', 'posterior']}
      caption={
        <>
          Needle <strong>B</strong> sweeps the circle while <strong>A</strong> stays put — drag
          inside the rose to place B yourself, or move A with the slider. Watch the orange needle:
          the moment B crosses the dashed seam at ±180°, the arithmetic mean flips to point
          <em> the other way</em>, through a region neither heading is anywhere near. The purple
          needle never does that, because it never adds angles: it measures the short rotation from
          A to B with <span className="font-mono">⊟</span>, halves that <em>rotation</em>, and
          applies it with <span className="font-mono">⊞</span>. The chart says the same thing
          globally — the correct mean is always within 90° of A, and the arithmetic one is not.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -1.3, maxX: 3.6, minY: -1.3, maxY: 1.3 }}
        draw={draw}
        deps={[thetaA, thetaB]}
        aspect={2.3}
        padding={0.06}
        ariaLabel="A compass rose with two heading needles, A and B. An orange needle shows their arithmetic mean and a purple needle shows the mean computed with the boxminus and boxplus operators. When B crosses the plus or minus 180 degree seam, the orange needle points in the opposite direction to the purple one."
        onPointer={onPointer}
        cursor="crosshair"
      />

      <div className="border-t border-fd-border px-3 py-3">
        <LineChart
          series={series}
          xLabel="heading B (degrees)"
          yLabel="mean, degrees from A"
          height={220}
          markers={[
            { axis: 'x', value: normalizeAngle(thetaA + Math.PI) * DEG, label: 'antipode of A', role: 'truth' },
            { axis: 'y', value: 90, label: '+90°', role: 'truth' },
            { axis: 'y', value: -90, role: 'truth' },
          ]}
          caption={
            <>
              Both means, measured as a rotation away from A, as B goes once around. They agree
              until B passes the antipode of A; after that the arithmetic mean keeps walking away
              while the correct mean folds back. Everything outside the ±90° band is a heading no
              honest average could return.
            </>
          }
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Heading A"
          role="prior"
          value={thetaADeg}
          min={-180}
          max={180}
          step={1}
          unit="°"
          format={(v) => v.toFixed(0)}
          onChange={setThetaADeg}
          help="Park A away from ±180° and the trap disappears — which is exactly why it survives code review."
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        tick={sim.tick}
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function fmt(rad: number): string {
  const d = normalizeAngle(rad) * DEG;
  return `${d >= 0 ? ' ' : ''}${d.toFixed(1)}°`;
}
