'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { boxminus, boxplus, normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import { diffDriveStep } from '@/lib/sim/world';
import { motionModelVelocity, type MotionAlphas, type VelocityCmd } from '@/lib/models/motion';
import { invertVelocityMotion } from '@/lib/models/motion-se2';
import {
  clear,
  drawGrid,
  drawPath,
  drawRobot,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w9.2 — Arc Anatomy.
 *
 * Two halves of the same idea. Forward: a constant (v, ω) is a rotation about
 * the instantaneous centre of curvature, and the exact update falls out of that
 * one picture. Backward: given a hypothesised pose, the *same* picture run in
 * reverse recovers the (v̂, ω̂) that would have been needed, which is all the
 * closed-form density does before it evaluates three noise terms.
 *
 * Drag the hypothesis anywhere and watch the construction chase it. The sampler
 * and the density are not two models; they are one geometric fact, read in two
 * directions.
 */

const ALPHAS: MotionAlphas = [0.06, 0.06, 0.06, 0.06, 0.02, 0.02];
const DT = 1;
const STEPS = 72;
const X0: Pose2 = { x: 0, y: 0, theta: 0 };

interface State {
  phase: number;
}

export function ArcAnatomy() {
  const [omega, setOmega] = useState(0.9);
  const [headingOffset, setHeadingOffset] = useState(0.25);
  const [showLie, setShowLie] = useState(false);
  const [target, setTarget] = useState<{ x: number; y: number }>({ x: 1.45, y: 1.25 });

  const cmd: VelocityCmd = useMemo(() => ({ v: 1, omega, dt: DT }), [omega]);
  const nominal = useMemo(() => diffDriveStep(X0, cmd.v, cmd.omega, cmd.dt), [cmd]);

  const init = useCallback((): State => ({ phase: 0 }), []);
  const step = useCallback((_: State, tick: number): State => ({ phase: (tick % STEPS) / STEPS }), []);
  const sim = useSimulation<State>({ init, step, fps: 24, maxTicks: STEPS, loop: true });

  /** The inversion is run on the dragged position plus a heading the reader controls. */
  const inv = useMemo(() => {
    const provisional = invertVelocityMotion(X0, { ...target, theta: 0 }, DT);
    const arcHeading = normalizeAngle(X0.theta + provisional.sweep);
    const hypothesis: Pose2 = { ...target, theta: normalizeAngle(arcHeading + headingOffset) };
    return {
      hypothesis,
      geom: invertVelocityMotion(X0, hypothesis, DT),
      density: motionModelVelocity(hypothesis, cmd, X0, ALPHAS),
      peak: motionModelVelocity(nominal, cmd, X0, ALPHAS),
      xi: boxminus(hypothesis, X0),
    };
  }, [target, headingOffset, cmd, nominal]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawGrid(ctx, v, p, 0.5);

      // ---- the commanded arc, built from its centre of rotation --------
      const commanded: Pose2[] = [];
      for (let k = 0; k <= 60; k++) commanded.push(diffDriveStep(X0, cmd.v, cmd.omega, (cmd.dt * k) / 60));
      drawPath(ctx, v, commanded, p.truth, { dashed: true, lineWidth: 1.6 });

      const swept = diffDriveStep(X0, cmd.v, cmd.omega, cmd.dt * sim.state.phase);

      if (Math.abs(cmd.omega) > 1e-6) {
        // ICC sits a signed radius v/ω to the left of the heading.
        const r = cmd.v / cmd.omega;
        const icc: [number, number] = [X0.x - r * Math.sin(X0.theta), X0.y + r * Math.cos(X0.theta)];

        ctx.strokeStyle = p.truth;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, icc[0]), sy(v, icc[1]));
        ctx.lineTo(sx(v, X0.x), sy(v, X0.y));
        ctx.moveTo(sx(v, icc[0]), sy(v, icc[1]));
        ctx.lineTo(sx(v, swept.x), sy(v, swept.y));
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = p.truth;
        ctx.beginPath();
        ctx.arc(sx(v, icc[0]), sy(v, icc[1]), 3.5, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, `ICC   r = v/ω = ${r.toFixed(2)} m`, sx(v, icc[0]) + 8, sy(v, icc[1]), p.truth, {
          size: 10,
        });
      } else {
        // ω = 0: the centre has gone to infinity and the arc is a line. Nothing
        // to draw, and nothing degenerate about the motion — which is the point.
        label(ctx, 'ω = 0 — the ICC is at infinity', sx(v, -1.5), sy(v, 1.9), p.truth, { size: 10 });
      }

      drawRobot(ctx, v, swept, p.truth, 0.16);
      label(
        ctx,
        `ωΔt = ${(cmd.omega * cmd.dt * sim.state.phase).toFixed(2)} rad`,
        sx(v, swept.x) + 10,
        sy(v, swept.y) - 10,
        p.truth,
        { size: 10 },
      );

      // ---- the inverted arc through the hypothesis ---------------------
      const g = inv.geom;
      if (!g.straight && Number.isFinite(g.radius)) {
        const a0 = Math.atan2(X0.y - g.center[1], X0.x - g.center[0]);
        const pts: { x: number; y: number }[] = [];
        for (let k = 0; k <= 60; k++) {
          const a = a0 + (g.sweep * k) / 60;
          pts.push({ x: g.center[0] + g.radius * Math.cos(a), y: g.center[1] + g.radius * Math.sin(a) });
        }
        drawPath(ctx, v, pts, p.prediction, { lineWidth: 2 });

        // The construction: perpendicular bisector meets the heading normal.
        ctx.strokeStyle = p.prediction;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, g.center[0]), sy(v, g.center[1]));
        ctx.lineTo(sx(v, X0.x), sy(v, X0.y));
        ctx.moveTo(sx(v, g.center[0]), sy(v, g.center[1]));
        ctx.lineTo(sx(v, inv.hypothesis.x), sy(v, inv.hypothesis.y));
        ctx.moveTo(sx(v, X0.x), sy(v, X0.y));
        ctx.lineTo(sx(v, inv.hypothesis.x), sy(v, inv.hypothesis.y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;

        ctx.fillStyle = p.prediction;
        ctx.beginPath();
        ctx.arc(sx(v, g.center[0]), sy(v, g.center[1]), 3, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, 'ICC*', sx(v, g.center[0]) + 7, sy(v, g.center[1]) + 9, p.prediction, { size: 10 });
      }

      // The γ̂ wedge: how far the hypothesis heading is from the arc's.
      const arcHeading = normalizeAngle(X0.theta + g.sweep);
      if (Math.abs(headingOffset) > 0.02) {
        const rad = sl(v, 0.34);
        ctx.strokeStyle = p.prediction;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(
          sx(v, inv.hypothesis.x),
          sy(v, inv.hypothesis.y),
          rad,
          -Math.max(arcHeading, inv.hypothesis.theta),
          -Math.min(arcHeading, inv.hypothesis.theta),
        );
        ctx.stroke();
        label(
          ctx,
          `γ̂Δt = ${headingOffset.toFixed(2)} rad`,
          sx(v, inv.hypothesis.x) + 12,
          sy(v, inv.hypothesis.y) + 16,
          p.prediction,
          { size: 10 },
        );
      }

      if (showLie) {
        // The other inversion: one twist that matches position *and* heading.
        const pts: Pose2[] = [];
        for (let k = 0; k <= 60; k++) {
          const s = k / 60;
          pts.push(boxplus(X0, [inv.xi[0] * s, inv.xi[1] * s, inv.xi[2] * s]));
        }
        drawPath(ctx, v, pts, p.posterior, { lineWidth: 1.6, dashed: true });
        label(ctx, 'log(x⁻¹x′) — the Lie inversion', sx(v, -1.5), sy(v, -1.35), p.posterior, {
          size: 10,
          weight: 600,
        });
      }

      drawRobot(ctx, v, inv.hypothesis, p.prediction, 0.2);
      drawRobot(ctx, v, X0, p.prior, 0.2);
      label(ctx, 'drag me', sx(v, inv.hypothesis.x) + 12, sy(v, inv.hypothesis.y) - 14, p.prediction, {
        size: 10,
        weight: 600,
      });
    },
    [cmd, sim.state.phase, inv, headingOffset, showLie],
  );

  const ratio = inv.peak > 0 ? inv.density / inv.peak : 0;

  return (
    <WidgetFrame
      id="w9.2"
      title="Arc Anatomy"
      teaches="The sampler and the closed-form density are the same geometry read in opposite directions — forward through the ICC, backward through it."
      colorKey={['prior', 'prediction', 'posterior', 'truth']}
      caption={
        <>
          <strong>Forward.</strong> Hold (v, ω) constant and the robot rotates about a single point,
          the instantaneous centre of curvature, a distance <em>v/ω</em> to its left. The gray robot
          sweeping the dashed arc is that rotation; the exact update in the text is nothing but
          &ldquo;rotate the pose about the ICC by ωΔt&rdquo;.{' '}
          <strong>Backward.</strong> Drag the orange robot anywhere. There is exactly one arc that
          leaves the blue pose tangentially and reaches that position, and the widget finds its
          centre <em>ICC*</em> the way Table 5.1 does — where the perpendicular bisector of the
          displacement meets the line normal to the start heading. From it come v̂ and ω̂; whatever
          heading is left over is γ̂. The density is then just three noise evaluations at
          (v − v̂, ω − ω̂, γ̂). Try dragging until v̂ ≈ v but ω̂ is far from ω, and watch the
          plausibility collapse on the turn alone. Then switch on the Lie inversion: a single twist
          that nails position <em>and</em> heading, at the cost of a sideways slide no differential
          drive can perform.
        </>
      }
    >
      <SimCanvas
        world={{ minX: -1.7, maxX: 2.9, minY: -1.5, maxY: 2.1 }}
        draw={draw}
        deps={[sim.tick, inv, showLie, cmd]}
        aspect={1.9}
        padding={0}
        cursor="grab"
        ariaLabel="A geometric construction: a robot at the origin, the centre of rotation of its commanded arc, and a second draggable robot whose reaching arc is reconstructed from its position."
        onPointer={(world, phase) => {
          if (phase === 'down' || phase === 'move') setTarget({ x: world[0], y: world[1] });
        }}
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Readout label="v̂ (inferred)" value={`${inv.geom.vHat.toFixed(3)} m/s`} note={`v = ${cmd.v.toFixed(2)}`} />
        <Readout
          label="ω̂ (inferred)"
          value={`${inv.geom.omegaHat.toFixed(3)} rad/s`}
          note={`ω = ${cmd.omega.toFixed(2)}`}
        />
        <Readout label="γ̂ (leftover)" value={`${inv.geom.gammaHat.toFixed(3)} rad/s`} note="0 if on the arc" />
        <Readout
          label="p(xₜ | u, xₜ₋₁)"
          value={inv.density.toExponential(2)}
          note={`${(ratio * 100).toFixed(1)}% of peak`}
        />
      </div>

      {showLie ? (
        <div className="border-t border-fd-border px-3 py-2 font-mono text-[0.72rem] text-fd-muted-foreground">
          ξ = log(x⁻¹x′) = ({inv.xi[0].toFixed(3)}, {inv.xi[1].toFixed(3)}, {inv.xi[2].toFixed(3)}) —
          the middle entry is the sideways slide the arc inversion refuses to allow.
        </div>
      ) : null}

      <ControlPanel columns={3}>
        <Slider
          label="Commanded ω"
          role="prediction"
          value={omega}
          min={-1.6}
          max={1.6}
          step={0.05}
          unit="rad/s"
          onChange={setOmega}
          help="The one parameter to play with: it moves the ICC, and at ω = 0 pushes it to infinity."
        />
        <Slider
          label="Hypothesis heading offset"
          value={headingOffset}
          min={-1.2}
          max={1.2}
          step={0.05}
          unit="rad"
          onChange={setHeadingOffset}
          help="Turns directly into γ̂: the heading the arc cannot account for."
        />
        <Toggle label="Show the Lie inversion" role="posterior" checked={showLie} onChange={setShowLie} />
      </ControlPanel>

      <Transport playing={sim.playing} onToggle={sim.toggle} onReset={sim.reset} tick={sim.tick} />
    </WidgetFrame>
  );
}

function Readout({ label: l, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
      <div className="font-mono text-[0.62rem] text-fd-muted-foreground">{note}</div>
    </div>
  );
}
