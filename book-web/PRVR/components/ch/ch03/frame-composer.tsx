'use client';

import { useCallback, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import {
  compose,
  inverseTransformPoint,
  normalizeAngle,
  pose2,
  transformPoint,
  type Pose2,
} from '@/lib/geom/se2';
import { clear, drawGrid, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w3.1 — the Frame Composer.
 *
 * Craig's leading-superscript notation is a type system written by hand: the
 * expression ᵂ_BT · ᴮ_LT type-checks because the adjacent B's cancel, and
 * ᵂ_BT · ᵂ_LT does not. This widget makes that concrete — the lidar frame is
 * carried by the body, a point has three sets of coordinates and only one
 * position, and the same increment applied about the body frame and about the
 * world frame lands in two different places.
 *
 * All transforms come from `lib/geom/se2`, the same code the chapter prints.
 */

/** The lidar's mount on Rusty: ᴮ_LT, fixed extrinsics. */
const MOUNT: Pose2 = pose2(0.62, 0.2, 0.4);

interface State {
  body: Pose2;
  point: [number, number];
}

type Handle = 'body' | 'point' | null;

export function FrameComposer() {
  const [deltaTheta, setDeltaTheta] = useState(0.9);
  const [deltaX, setDeltaX] = useState(0.8);
  const [showFixed, setShowFixed] = useState(true);
  const [drag, setDrag] = useState<Handle>(null);

  const init = useCallback(
    (): State => ({ body: pose2(2.2, 1.5, 0.5), point: [4.1, 2.4] }),
    [],
  );
  // Autoplay: the body turns. Nothing else is touched, and yet every number in
  // the panel below moves — because every one of them is expressed through ᵂ_BT.
  const step = useCallback(
    (s: State): State => ({ ...s, body: { ...s.body, theta: normalizeAngle(s.body.theta + 0.03) } }),
    [],
  );

  // 12 Hz, not 60: the numbers in the panel are meant to be read, not to blur.
  const sim = useSimulation<State>({ init, step, fps: 12, initialSeed: 7 });
  const { body, point } = sim.state;

  const worldFromLidar = compose(body, MOUNT); // ᵂ_LT = ᵂ_BT · ᴮ_LT
  const delta = pose2(deltaX, 0, deltaTheta);
  const current = compose(body, delta); // ᵂ_BT · Δ — increment in the body frame
  const fixed = compose(delta, body); // Δ · ᵂ_BT — increment in the world frame
  const gap = Math.hypot(current.x - fixed.x, current.y - fixed.y);

  const pBody = inverseTransformPoint(body, point);
  const pLidar = inverseTransformPoint(worldFromLidar, point);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 1);

      const frame = (
        pose: Pose2,
        color: string,
        name: string,
        opts: { len?: number; ghost?: boolean } = {},
      ) => {
        const { len = 0.62, ghost = false } = opts;
        const ox = sx(v, pose.x);
        const oy = sy(v, pose.y);
        const L = sl(v, len);
        ctx.save();
        ctx.globalAlpha = ghost ? 0.62 : 1;
        ctx.lineWidth = ghost ? 1.6 : 2.4;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        if (ghost) ctx.setLineDash([5, 4]);

        // x̂ solid with an arrowhead, ŷ thinner — the two columns of R.
        for (const [ang, tip] of [
          [pose.theta, true],
          [pose.theta + Math.PI / 2, false],
        ] as [number, boolean][]) {
          const ex = ox + L * Math.cos(ang);
          const ey = oy - L * Math.sin(ang);
          ctx.beginPath();
          ctx.moveTo(ox, oy);
          ctx.lineTo(ex, ey);
          ctx.stroke();
          if (tip && !ghost) {
            ctx.save();
            ctx.setLineDash([]);
            ctx.translate(ex, ey);
            ctx.rotate(-ang);
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(-7, 3.5);
            ctx.lineTo(-7, -3.5);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
          }
        }
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(ox, oy, ghost ? 2.5 : 4, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, name, ox - 12, oy + 13, color, { size: 11, weight: 700 });
        ctx.restore();
      };

      // The rigid link from body to lidar: the constant ᴮ_LT, drawn so the
      // reader sees it swing around with the body rather than sit still.
      ctx.strokeStyle = p.measurement;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(v, body.x), sy(v, body.y));
      ctx.lineTo(sx(v, worldFromLidar.x), sy(v, worldFromLidar.y));
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (showFixed) {
        // Two ghosts of the same increment: Δ about the body, Δ about the world.
        frame(current, p.posterior, 'B·Δ', { ghost: true, len: 0.5 });
        frame(fixed, p.prediction, 'Δ·B', { ghost: true, len: 0.5 });
        ctx.strokeStyle = p.prediction;
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(sx(v, current.x), sy(v, current.y));
        ctx.lineTo(sx(v, fixed.x), sy(v, fixed.y));
        ctx.stroke();
        ctx.setLineDash([]);
        label(
          ctx,
          `‖Δ·B − B·Δ‖ = ${gap.toFixed(2)} m`,
          (sx(v, current.x) + sx(v, fixed.x)) / 2,
          (sy(v, current.y) + sy(v, fixed.y)) / 2 - 12,
          p.prediction,
          { size: 10, align: 'center', weight: 600 },
        );
      }

      frame(pose2(0, 0, 0), p.truth, '{W}', { len: 0.75 });
      frame(body, p.prior, '{B}');
      frame(worldFromLidar, p.measurement, '{L}', { len: 0.5 });

      // The point exists once; only its coordinates are frame-dependent.
      const px = sx(v, point[0]);
      const py = sy(v, point[1]);
      ctx.strokeStyle = p.posterior;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      for (const f of [pose2(0, 0, 0), body, worldFromLidar]) {
        ctx.beginPath();
        ctx.moveTo(sx(v, f.x), sy(v, f.y));
        ctx.lineTo(px, py);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = p.posterior;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
      label(ctx, 'p', px + 9, py - 9, p.posterior, { size: 12, weight: 700 });
    },
    [body, point, worldFromLidar, current, fixed, gap, showFixed],
  );

  const onPointer = useCallback(
    (world: [number, number], phase: 'down' | 'move' | 'up') => {
      const [wx, wy] = world;
      if (phase === 'up') {
        setDrag(null);
        return;
      }
      if (phase === 'down') {
        sim.pause();
        const dBody = Math.hypot(wx - body.x, wy - body.y);
        const dPoint = Math.hypot(wx - point[0], wy - point[1]);
        const picked: Handle = dPoint < dBody && dPoint < 0.6 ? 'point' : dBody < 0.6 ? 'body' : null;
        setDrag(picked);
        if (!picked) return;
        sim.setState((s) =>
          picked === 'point' ? { ...s, point: [wx, wy] } : { ...s, body: { ...s.body, x: wx, y: wy } },
        );
        return;
      }
      if (!drag) return;
      sim.setState((s) =>
        drag === 'point' ? { ...s, point: [wx, wy] } : { ...s, body: { ...s.body, x: wx, y: wy } },
      );
    },
    [sim, drag, body.x, body.y, point],
  );

  return (
    <WidgetFrame
      id="w3.1"
      title="Frame Composer"
      teaches="The leading super/subscripts are a type system, not decoration — and transform composition does not commute."
      colorKey={['truth', 'prior', 'measurement', 'prediction', 'posterior']}
      caption={
        <>
          Drag <strong>{'{B}'}</strong> or the point <strong>p</strong>; the body also turns on its
          own. Three things to notice. (1) The lidar frame <strong>{'{L}'}</strong> is never
          positioned directly — it is <em>computed</em>, as ᵂ<sub>B</sub>T · ᴮ<sub>L</sub>T, which
          is why it swings when the body turns. (2) The point never moves, but all three coordinate
          triples below change, because coordinates belong to a frame and points do not. (3) The
          two dashed ghosts apply the <em>same</em> increment Δ — one about the body frame
          (post-multiply, the book&apos;s <span className="font-mono">⊞</span>) and one about the
          world frame (pre-multiply). They do not land in the same place, and the gap grows with
          Δθ. That gap is non-commutativity, in metres.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -0.9, maxX: 6.3, minY: -0.7, maxY: 3.7 }}
        draw={draw}
        deps={[body, point, showFixed, deltaTheta, deltaX]}
        aspect={1.95}
        padding={0.05}
        ariaLabel="Three coordinate frames in a plane: a fixed world frame at the origin, a body frame that rotates, and a lidar frame rigidly attached to the body. A draggable point is connected to all three frame origins by dashed lines. Two ghost frames show the same increment applied about the body frame and about the world frame, landing in different places."
        onPointer={onPointer}
        cursor="grab"
      />

      <div className="border-t border-fd-border px-3 py-3">
        <p className="eyebrow mb-2">the transform chain — adjacent indices cancel</p>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-[0.9rem]">
          <Craig from="W" to="L" color="var(--pr-measurement)" />
          <span className="text-fd-muted-foreground">=</span>
          <Craig from="W" to="B" color="var(--pr-prior)" cancel="to" />
          <Craig from="B" to="L" color="var(--pr-measurement)" cancel="from" />
        </div>

        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MatrixCard title="ᵂ_B T   (world ← body)" pose={body} color="var(--pr-prior)" />
          <MatrixCard title="ᴮ_L T   (body ← lidar)" pose={MOUNT} color="var(--pr-measurement)" />
          <MatrixCard
            title="ᵂ_L T   (world ← lidar)"
            pose={worldFromLidar}
            color="var(--pr-measurement)"
          />
        </div>

        <p className="eyebrow mt-4 mb-2">one point, three coordinate triples</p>
        <div className="grid grid-cols-1 gap-2 font-mono text-[0.78rem] tabular-nums sm:grid-cols-3">
          <Coord label="ᵂp" value={point} color="var(--pr-truth)" />
          <Coord label="ᴮp" value={pBody} color="var(--pr-prior)" />
          <Coord label="ᴸp" value={pLidar} color="var(--pr-measurement)" />
        </div>
        <p className="mt-2 font-ui text-[0.72rem] text-fd-muted-foreground">
          Round trip check: ᵂ<sub>B</sub>T · ᴮp ={' '}
          {fmtPair(transformPoint(body, pBody))} — the same ᵂp, recovered.
        </p>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Increment Δθ"
          role="prediction"
          value={deltaTheta}
          min={0}
          max={2.4}
          step={0.05}
          unit="rad"
          onChange={setDeltaTheta}
          help="Set this to zero and the two ghosts merge: pure translations do commute with each other, rotations do not."
        />
        <Slider
          label="Increment Δx"
          role="prediction"
          value={deltaX}
          min={0}
          max={1.6}
          step={0.05}
          unit="m"
          onChange={setDeltaX}
        />
        <Toggle
          label="Show Δ·B and B·Δ"
          role="prediction"
          checked={showFixed}
          onChange={setShowFixed}
        />
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

/** Craig's leading super/subscript, e.g. ᵂ_BT. `cancel` tints the index that
 *  annihilates against its neighbour in the chain. */
function Craig({
  from,
  to,
  color,
  cancel,
}: {
  from: string;
  to: string;
  color: string;
  cancel?: 'from' | 'to';
}) {
  const mark = (side: 'from' | 'to') =>
    cancel === side
      ? 'rounded-[2px] bg-fd-primary/15 px-[2px] text-fd-primary'
      : 'text-fd-muted-foreground';
  return (
    <span className="inline-flex items-center">
      <span className="inline-flex flex-col items-end text-[0.62em] leading-[1.05]">
        <span className={mark('from')}>{from}</span>
        <span className={mark('to')}>{to}</span>
      </span>
      <span className="ml-[1px] font-semibold" style={{ color }}>
        T
      </span>
    </span>
  );
}

function MatrixCard({ title, pose, color }: { title: string; pose: Pose2; color: string }) {
  const c = Math.cos(pose.theta);
  const s = Math.sin(pose.theta);
  const rows = [
    [c, -s, pose.x],
    [s, c, pose.y],
    [0, 0, 1],
  ];
  return (
    <div className="rounded-sm border border-fd-border border-l-2 px-2.5 py-2" style={{ borderLeftColor: color }}>
      <div className="eyebrow mb-1.5">{title}</div>
      <div className="grid grid-cols-3 gap-x-2 font-mono text-[0.72rem] tabular-nums">
        {rows.flat().map((x, i) => (
          <span key={i} className="text-end">
            {x.toFixed(3)}
          </span>
        ))}
      </div>
      <div className="mt-1.5 font-mono text-[0.68rem] text-fd-muted-foreground tabular-nums">
        (x, y, θ) = ({pose.x.toFixed(2)}, {pose.y.toFixed(2)}, {pose.theta.toFixed(3)})
      </div>
    </div>
  );
}

function Coord({ label: l, value, color }: { label: string; value: [number, number]; color: string }) {
  return (
    <div className="flex items-baseline gap-2">
      <span style={{ color }} className="font-semibold">
        {l}
      </span>
      <span className="text-fd-muted-foreground">{fmtPair(value)}</span>
    </div>
  );
}

const fmtPair = (v: [number, number]) => `(${v[0].toFixed(3)}, ${v[1].toFixed(3)})`;
