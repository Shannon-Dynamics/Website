'use client';

import { useMemo, useState } from 'react';
import { SimPanel, Slider } from './SimControls';
import { StatTile } from '@/components/viz/StatTile';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

interface Contact {
  angle: number; // position on the object's rim, radians
}

/**
 * `ch20-grasp-wrench` — force closure, made draggable.
 *
 * A grasp resists arbitrary disturbances only if the wrenches its contacts can
 * generate positively span the whole wrench space. Move the contacts and watch
 * the friction cones rotate; when their span collapses to one side of the
 * object, force closure fails and the object squirts out under a load the
 * grasp cannot oppose. This is the analytic criterion that learned grasp
 * scorers approximate from data.
 */
export function GraspWrench() {
  const { mode } = useTheme();
  const [contacts, setContacts] = useState<Contact[]>([
    { angle: 0.4 },
    { angle: Math.PI - 0.4 },
  ]);
  const [friction, setFriction] = useState(0.5);
  const [dragging, setDragging] = useState<number | null>(null);

  const R = 58;
  const CX = 132;
  const CY = 118;

  const analysis = useMemo(() => {
    const halfAngle = Math.atan(friction);

    // Each contact contributes a cone of directions it can push along, centred
    // on the inward normal. Sample the cone edges as generating wrenches.
    const generators: Array<{ fx: number; fy: number; torque: number }> = [];
    for (const c of contacts) {
      const nx = -Math.cos(c.angle); // inward normal
      const ny = -Math.sin(c.angle);
      const px = Math.cos(c.angle) * R;
      const py = Math.sin(c.angle) * R;
      for (const edge of [-halfAngle, 0, halfAngle]) {
        const fx = nx * Math.cos(edge) - ny * Math.sin(edge);
        const fy = nx * Math.sin(edge) + ny * Math.cos(edge);
        generators.push({ fx, fy, torque: (px * fy - py * fx) / R });
      }
    }

    // Force closure in 2-D requires the force generators to positively span the
    // plane: no half-plane may contain all of them. Test by checking the
    // largest angular gap between sorted generator directions.
    const angles = generators.map((g) => Math.atan2(g.fy, g.fx)).sort((a, b) => a - b);
    let maxGap = 0;
    for (let i = 0; i < angles.length; i++) {
      const next = i === angles.length - 1 ? angles[0] + 2 * Math.PI : angles[i + 1];
      maxGap = Math.max(maxGap, next - angles[i]);
    }
    const forceClosure = maxGap < Math.PI - 1e-6;

    // Ferrari–Canny style quality: the radius of the largest wrench ball the
    // grasp can resist, approximated here by how far the gap is from a
    // half-plane, scaled by cone width.
    const quality = forceClosure
      ? Math.max(0, (Math.PI - maxGap) / Math.PI) * Math.min(1, friction / 0.6)
      : 0;

    return { halfAngle, generators, forceClosure, maxGap, quality };
  }, [contacts, friction]);

  const handlePointer = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragging === null) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = 264 / rect.width;
    const x = (e.clientX - rect.left) * scale - CX;
    const y = (e.clientY - rect.top) * scale - CY;
    const angle = Math.atan2(y, x);
    setContacts((cs) => cs.map((c, i) => (i === dragging ? { angle } : c)));
  };

  return (
    <SimPanel
      title="Force closure: when a grasp actually holds"
      id="ch20-grasp-wrench"
      subtitle="Drag the contact points around the object. Each cone shows the directions that contact can push, given the friction coefficient."
      controls={
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            className="w-56"
            label="Friction coefficient μ"
            value={friction}
            min={0.05}
            max={1.2}
            step={0.05}
            onChange={setFriction}
            hint={`cone half-angle ${((Math.atan(friction) * 180) / Math.PI).toFixed(0)}°`}
          />
          <button
            type="button"
            onClick={() =>
              setContacts((cs) =>
                cs.length < 3
                  ? [...cs, { angle: -Math.PI / 2 }]
                  : [{ angle: 0.4 }, { angle: Math.PI - 0.4 }],
              )
            }
            className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
          >
            {contacts.length < 3 ? 'Add third contact' : 'Reset to two contacts'}
          </button>
        </div>
      }
      caption="Two opposed contacts with generous friction give force closure — the cones point at each other and together span every direction a disturbance could come from. Slide them onto the same side and the span collapses into a half-plane: there is now a direction of push the grasp cannot resist at all, and the object escapes. Lower μ and the cones narrow until even opposed contacts fail. Everything Chapter 20's learned grasp scorer does is approximate this test from a depth image, without knowing the object's geometry."
    >
      <div className="grid gap-4 lg:grid-cols-[280px,1fr]">
        <div>
          <svg
            width={264}
            height={236}
            viewBox="0 0 264 236"
            className="max-w-full touch-none rounded-lg"
            style={{ background: 'var(--surface-sunken)', cursor: dragging !== null ? 'grabbing' : 'default' }}
            onPointerMove={handlePointer}
            onPointerUp={() => setDragging(null)}
            onPointerLeave={() => setDragging(null)}
            role="img"
            aria-label="Object with draggable contact points and their friction cones"
          >
            {/* Object */}
            <circle
              cx={CX}
              cy={CY}
              r={R}
              fill={mode === 'light' ? '#e6e5de' : '#2c2c2a'}
              stroke="var(--baseline)"
              strokeWidth={1.5}
            />

            {/* Friction cones */}
            {contacts.map((c, i) => {
              const px = CX + Math.cos(c.angle) * R;
              const py = CY + Math.sin(c.angle) * R;
              const nx = -Math.cos(c.angle);
              const ny = -Math.sin(c.angle);
              const L = 46;
              const ha = analysis.halfAngle;
              const e1x = px + (nx * Math.cos(ha) - ny * Math.sin(ha)) * L;
              const e1y = py + (nx * Math.sin(ha) + ny * Math.cos(ha)) * L;
              const e2x = px + (nx * Math.cos(-ha) - ny * Math.sin(-ha)) * L;
              const e2y = py + (nx * Math.sin(-ha) + ny * Math.cos(-ha)) * L;

              return (
                <g key={i}>
                  <path
                    d={`M${px},${py} L${e1x},${e1y} L${e2x},${e2y} Z`}
                    fill={seriesColor(i, mode)}
                    opacity={0.25}
                    stroke={seriesColor(i, mode)}
                    strokeWidth={1}
                  />
                  <circle
                    cx={px}
                    cy={py}
                    r={8}
                    fill={seriesColor(i, mode)}
                    stroke="var(--surface-1)"
                    strokeWidth={2}
                    style={{ cursor: 'grab' }}
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      setDragging(i);
                    }}
                  />
                </g>
              );
            })}

            <text x={10} y={222} fontSize={10} fill="var(--text-muted)">
              drag the contacts · shaded wedge = friction cone
            </text>
          </svg>
        </div>

        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <StatTile
              label="Force closure"
              value={analysis.forceClosure ? 'Yes' : 'No'}
              mono={false}
              status={analysis.forceClosure ? 'good' : 'critical'}
              hint={
                analysis.forceClosure
                  ? 'resists any disturbance wrench'
                  : 'a direction exists that defeats it'
              }
            />
            <StatTile
              label="Grasp quality"
              value={analysis.quality}
              hint="largest resistible wrench (normalized)"
              status={analysis.quality > 0.35 ? 'good' : analysis.quality > 0.1 ? 'warning' : 'critical'}
            />
            <StatTile
              label="Largest angular gap"
              value={(analysis.maxGap * 180) / Math.PI}
              unit="°"
              hint="must stay below 180°"
            />
            <StatTile
              label="Contacts"
              value={contacts.length}
              hint={`${analysis.generators.length} wrench generators`}
            />
          </div>

          <div className="rounded-lg border border-hairline p-3 text-[12.5px] leading-relaxed text-ink-secondary">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-muted">
              The criterion
            </p>
            <p>
              A grasp has force closure when its contact wrenches{' '}
              <em>positively span</em> the wrench space — every disturbance can be
              opposed by some non-negative combination of contact forces. In the
              plane this reduces to a readable test: no half-plane may contain all
              the generators, so the largest angular gap must be under 180°.
            </p>
            <p className="mt-2">
              The quality number is the radius of the largest wrench ball the grasp
              resists — the Ferrari–Canny metric. It is what separates a grasp that
              technically holds from one that survives being carried across a room.
            </p>
          </div>
        </div>
      </div>
    </SimPanel>
  );
}
