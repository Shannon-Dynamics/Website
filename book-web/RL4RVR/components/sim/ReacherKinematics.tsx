'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider, Segmented } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

/**
 * `ch13-kinematics-sandbox` — Reacher as a mechanism rather than a task.
 *
 * Drag the end-effector and watch inverse kinematics solve for joint angles;
 * drag the joints and watch forward kinematics place the tip. The manipulability
 * ellipsoid — the image of the unit joint-velocity ball under the Jacobian —
 * rounds out where the arm is dexterous and collapses to a line at a
 * singularity, which is the geometric content of det J = ℓ₁ℓ₂ sin q₂.
 */
export function ReacherKinematics() {
  const { mode } = useTheme();
  const [mode_, setMode_] = useState<'joint' | 'task'>('task');
  const [q1, setQ1] = useState(0.6);
  const [q2, setQ2] = useState(1.1);
  const [elbowUp, setElbowUp] = useState(true);
  const [dragging, setDragging] = useState(false);

  const L1 = 74;
  const L2 = 62;
  const CX = 150;
  const CY = 150;

  const fk = useMemo(() => {
    const x1 = L1 * Math.cos(q1);
    const y1 = L1 * Math.sin(q1);
    const x = x1 + L2 * Math.cos(q1 + q2);
    const y = y1 + L2 * Math.sin(q1 + q2);
    return { elbow: { x: x1, y: y1 }, tip: { x, y } };
  }, [q1, q2]);

  const jacobian = useMemo(() => {
    const s1 = Math.sin(q1);
    const c1 = Math.cos(q1);
    const s12 = Math.sin(q1 + q2);
    const c12 = Math.cos(q1 + q2);
    const J = [
      [-L1 * s1 - L2 * s12, -L2 * s12],
      [L1 * c1 + L2 * c12, L2 * c12],
    ];
    const det = L1 * L2 * Math.sin(q2);

    // Manipulability ellipsoid: eigen-decomposition of J Jᵀ (2×2, closed form).
    const a = J[0][0] ** 2 + J[0][1] ** 2;
    const b = J[0][0] * J[1][0] + J[0][1] * J[1][1];
    const c = J[1][0] ** 2 + J[1][1] ** 2;
    const tr = a + c;
    const disc = Math.sqrt(Math.max(0, (a - c) ** 2 + 4 * b * b));
    const l1 = (tr + disc) / 2;
    const l2 = (tr - disc) / 2;
    const theta = 0.5 * Math.atan2(2 * b, a - c);

    return {
      J,
      det,
      manipulability: Math.abs(det),
      major: Math.sqrt(Math.max(l1, 0)),
      minor: Math.sqrt(Math.max(l2, 0)),
      angle: theta,
      // Condition number: 1 is perfectly isotropic, ∞ at a singularity.
      condition: l2 > 1e-9 ? Math.sqrt(l1 / l2) : Infinity,
    };
  }, [q1, q2]);

  /** Inverse kinematics — both elbow branches, with reachability check. */
  const solveIk = (tx: number, ty: number, up: boolean) => {
    const r2 = tx * tx + ty * ty;
    const r = Math.sqrt(r2);
    if (r > L1 + L2 || r < Math.abs(L1 - L2)) return null;   // outside the annulus

    const cosQ2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    const q2n = (up ? 1 : -1) * Math.acos(Math.max(-1, Math.min(1, cosQ2)));
    const q1n =
      Math.atan2(ty, tx) - Math.atan2(L2 * Math.sin(q2n), L1 + L2 * Math.cos(q2n));
    return { q1: q1n, q2: q2n };
  };

  const handlePointer = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging || mode_ !== 'task') return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = 300 / rect.width;
    const tx = (e.clientX - rect.left) * scale - CX;
    const ty = (e.clientY - rect.top) * scale - CY;
    const sol = solveIk(tx, ty, elbowUp);
    if (sol) {
      setQ1(sol.q1);
      setQ2(sol.q2);
    }
  };

  const reach = Math.hypot(fk.tip.x, fk.tip.y);
  const nearSingular = Math.abs(Math.sin(q2)) < 0.12;

  return (
    <SimPanel
      title="Reacher: joint space and task space"
      id="ch13-kinematics-sandbox"
      subtitle="Drag the end-effector to solve inverse kinematics, or move the joints directly. The ellipse is the manipulability ellipsoid."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Segmented
            label="Control"
            value={mode_}
            onChange={setMode_}
            options={[
              { value: 'task', label: 'Task space (drag the tip)' },
              { value: 'joint', label: 'Joint space (sliders)' },
            ]}
          />
          {mode_ === 'joint' ? (
            <>
              <Slider
                className="w-40"
                label="q₁ (shoulder)"
                value={q1}
                min={-Math.PI}
                max={Math.PI}
                step={0.02}
                onChange={setQ1}
                format={(v) => `${((v * 180) / Math.PI).toFixed(0)}°`}
              />
              <Slider
                className="w-40"
                label="q₂ (elbow)"
                value={q2}
                min={-Math.PI}
                max={Math.PI}
                step={0.02}
                onChange={setQ2}
                format={(v) => `${((v * 180) / Math.PI).toFixed(0)}°`}
              />
            </>
          ) : (
            <button
              type="button"
              onClick={() => {
                const sol = solveIk(fk.tip.x, fk.tip.y, !elbowUp);
                if (sol) {
                  setQ1(sol.q1);
                  setQ2(sol.q2);
                  setElbowUp(!elbowUp);
                }
              }}
              className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              Switch to elbow-{elbowUp ? 'down' : 'up'}
            </button>
          )}
        </div>
      }
      caption="Switching elbow solutions holds the end-effector perfectly still while the arm folds through an entirely different configuration — that is redundancy, and Chapter 19 spends it on obstacle avoidance. Now straighten the arm until q₂ approaches zero: the ellipse flattens to a line, meaning there is a direction the tip simply cannot move, and any controller inverting J would demand infinite joint velocity to try."
    >
      <div className="grid gap-4 lg:grid-cols-[320px,1fr]">
        <svg
          width={300}
          height={300}
          viewBox="0 0 300 300"
          className="max-w-full touch-none rounded-lg"
          style={{ background: 'var(--surface-sunken)', cursor: dragging ? 'grabbing' : 'default' }}
          onPointerMove={handlePointer}
          onPointerUp={() => setDragging(false)}
          onPointerLeave={() => setDragging(false)}
          role="img"
          aria-label="Two-link arm with manipulability ellipsoid at the end-effector"
        >
          {/* Reachable workspace: the annulus */}
          <circle cx={CX} cy={CY} r={L1 + L2} fill="none" stroke="var(--gridline)" strokeDasharray="3 4" />
          <circle cx={CX} cy={CY} r={Math.abs(L1 - L2)} fill="none" stroke="var(--gridline)" strokeDasharray="3 4" />

          {/* Links */}
          <line
            x1={CX}
            y1={CY}
            x2={CX + fk.elbow.x}
            y2={CY + fk.elbow.y}
            stroke={seriesColor(0, mode)}
            strokeWidth={7}
            strokeLinecap="round"
          />
          <line
            x1={CX + fk.elbow.x}
            y1={CY + fk.elbow.y}
            x2={CX + fk.tip.x}
            y2={CY + fk.tip.y}
            stroke={seriesColor(0, mode)}
            strokeWidth={6}
            strokeLinecap="round"
            opacity={0.82}
          />

          {/* Manipulability ellipsoid at the tip */}
          <ellipse
            cx={CX + fk.tip.x}
            cy={CY + fk.tip.y}
            rx={Math.min(70, jacobian.major * 0.42)}
            ry={Math.min(70, jacobian.minor * 0.42)}
            transform={`rotate(${(jacobian.angle * 180) / Math.PI} ${CX + fk.tip.x} ${CY + fk.tip.y})`}
            fill={seriesColor(2, mode)}
            opacity={0.22}
            stroke={seriesColor(2, mode)}
            strokeWidth={1.5}
          />

          {/* Joints */}
          <circle cx={CX} cy={CY} r={7} fill="var(--text-muted)" />
          <circle
            cx={CX + fk.elbow.x}
            cy={CY + fk.elbow.y}
            r={6}
            fill="var(--surface-1)"
            stroke={seriesColor(0, mode)}
            strokeWidth={2.5}
          />
          <circle
            cx={CX + fk.tip.x}
            cy={CY + fk.tip.y}
            r={9}
            fill={seriesColor(1, mode)}
            stroke="var(--surface-1)"
            strokeWidth={2.5}
            style={{ cursor: mode_ === 'task' ? 'grab' : 'default' }}
            onPointerDown={() => mode_ === 'task' && setDragging(true)}
          />

          {nearSingular && (
            <text x={10} y={290} fontSize={11} fontWeight={600} fill="var(--status-critical)">
              near singularity — det J → 0
            </text>
          )}
          {mode_ === 'task' && !nearSingular && (
            <text x={10} y={290} fontSize={10} fill="var(--text-muted)">
              drag the orange tip
            </text>
          )}
        </svg>

        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <StatTile label="q₁" value={(q1 * 180) / Math.PI} unit="°" />
            <StatTile label="q₂" value={(q2 * 180) / Math.PI} unit="°" />
            <StatTile label="tip x" value={fk.tip.x / 74} hint="in units of ℓ₁" />
            <StatTile label="tip y" value={-fk.tip.y / 74} hint="in units of ℓ₁" />
            <StatTile
              label="det J = ℓ₁ℓ₂ sin q₂"
              value={jacobian.det / (74 * 62)}
              status={nearSingular ? 'critical' : 'good'}
              hint={nearSingular ? 'singular configuration' : 'invertible'}
            />
            <StatTile
              label="Condition number"
              value={jacobian.condition}
              status={jacobian.condition > 8 ? 'critical' : jacobian.condition > 3 ? 'warning' : 'good'}
              hint="1 = isotropic, ∞ = singular"
            />
          </div>

          <div className="rounded-lg border border-hairline p-3 text-[12.5px] leading-relaxed text-ink-secondary">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              Reading the ellipse
            </p>
            <p>
              The ellipse is the set of end-effector velocities reachable from a
              unit ball of joint velocities. A round ellipse means the arm moves
              equally well in every direction; an elongated one means it is fast
              along the major axis and sluggish across it.
            </p>
            <p className="mt-2">
              Current reach: <strong className="text-ink">{(reach / (L1 + L2)).toFixed(2)}</strong> of
              maximum. The dashed circles bound the reachable annulus — inverse
              kinematics has no solution outside them, which is a constraint any
              task-space action must respect.
            </p>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}
