'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import type { Pose2 } from '@/lib/geom/se2';
import { APARTMENT, diffDriveStep } from '@/lib/sim/world';
import { sampleMotionModelVelocity, type MotionAlphas, type VelocityCmd } from '@/lib/models/motion';
import { classifyAgainstMap, DEFAULT_MAP_CONDITIONING, type MapVerdict } from '@/lib/models/motion-map';
import {
  clear,
  drawPath,
  drawRobot,
  drawWorld,
  label,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w9.4 — Map Squeeze.
 *
 * `sample_motion_model_with_map` (Table 5.7) multiplies the motion model by the
 * free-space indicator and rejection-samples. It is a genuinely good idea — the
 * banana stops leaking into the walls — and it is also an approximation with a
 * specific, findable failure: p(x_t | m) is a function of the *pose*, so the
 * test can only look at where the sample ended up.
 *
 * Park Rusty in a doorway and a fat slice of the accepted mass consists of
 * samples that drove straight through a wall and stopped somewhere legal. The
 * widget rings those, because a localizer that trusts them will happily place
 * the robot on the far side of a wall it never went through.
 */

const CLOUD = 420;
const CLEARANCE = 0.13;

type SpotId = 'doorway' | 'wall' | 'corridor';

const SPOTS: Record<SpotId, { label: string; pose: Pose2 }> = {
  doorway: { label: 'Facing a doorway', pose: { x: 2.05, y: 2.6, theta: Math.PI / 2 } },
  wall: { label: 'Facing a wall', pose: { x: 3.3, y: 2.6, theta: Math.PI / 2 } },
  corridor: { label: 'Open corridor', pose: { x: 1.4, y: 4.4, theta: 0 } },
};

interface Params {
  spot: SpotId;
  conditioned: boolean;
  noise: number;
  omega: number;
}

interface State {
  stamp: string;
  rng: Rng;
  verdicts: MapVerdict[];
}

export function MapSqueeze() {
  const [params, setParams] = useState<Params>({
    spot: 'doorway',
    conditioned: true,
    noise: 0.12,
    omega: 0,
  });

  const start = SPOTS[params.spot].pose;
  const cmd: VelocityCmd = useMemo(
    () => ({ v: 1, omega: params.omega, dt: 2 }),
    [params.omega],
  );
  const alphas: MotionAlphas = useMemo(() => {
    const n = params.noise;
    return [n, n, n, n, n / 4, n / 4];
  }, [params.noise]);

  const stamp = `${params.spot}|${params.noise}|${params.omega}`;

  const init = useCallback(
    (seed: number): State => ({ stamp, rng: new Rng(seed), verdicts: [] }),
    [stamp],
  );

  /** Samples arrive in batches so the reader watches the banana meet the wall. */
  const step = useCallback(
    (s: State): State => {
      const base = s.stamp === stamp ? s : { stamp, rng: s.rng, verdicts: [] };
      if (base.verdicts.length >= CLOUD) return base;
      const verdicts = base.verdicts.slice();
      const opts = { ...DEFAULT_MAP_CONDITIONING, clearance: CLEARANCE };
      for (let i = 0; i < 40 && verdicts.length < CLOUD; i++) {
        const pose = sampleMotionModelVelocity(cmd, start, alphas, base.rng);
        verdicts.push(classifyAgainstMap(APARTMENT, start, pose, opts));
      }
      return { stamp, rng: base.rng, verdicts };
    },
    [stamp, cmd, start, alphas],
  );

  const sim = useSimulation<State>({ init, step, fps: 10, initialSeed: 5 });

  const stats = useMemo(() => {
    const vs = sim.state.verdicts;
    const accepted = vs.filter((v) => v.endpointFree);
    const ghosts = accepted.filter((v) => !v.pathClear);
    return {
      total: vs.length,
      accepted: accepted.length,
      ghosts: ghosts.length,
      acceptRate: vs.length ? accepted.length / vs.length : 0,
      ghostRate: accepted.length ? ghosts.length / accepted.length : 0,
    };
  }, [sim.state.verdicts]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      drawWorld(ctx, v, APARTMENT, p);

      // The commanded arc, as always in truth-gray.
      const arc: Pose2[] = [];
      for (let k = 0; k <= 30; k++) arc.push(diffDriveStep(start, cmd.v, cmd.omega, (cmd.dt * k) / 30));
      drawPath(ctx, v, arc, p.truth, { dashed: true, lineWidth: 1.5 });

      for (const verdict of sim.state.verdicts) {
        const rejected = params.conditioned && !verdict.endpointFree;
        ctx.globalAlpha = rejected ? 0.09 : 0.55;
        ctx.fillStyle = p.prediction;
        ctx.beginPath();
        ctx.arc(sx(v, verdict.pose.x), sy(v, verdict.pose.y), 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // The interesting cell of the truth table: accepted, but it drove through
      // a wall to get there. Ringed in the chrome accent — deliberately not a
      // data color, because this is an annotation about the model, not data.
      if (params.conditioned) {
        ctx.strokeStyle = p.accent;
        ctx.lineWidth = 1.4;
        for (const verdict of sim.state.verdicts) {
          if (!verdict.endpointFree || verdict.pathClear) continue;
          drawPath(ctx, v, verdict.path, p.accent, { lineWidth: 0.9, alpha: 0.35 });
          ctx.beginPath();
          ctx.arc(sx(v, verdict.pose.x), sy(v, verdict.pose.y), 4.6, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      drawRobot(ctx, v, start, p.prior, 0.22);
      label(
        ctx,
        params.conditioned
          ? 'p(xₜ | uₜ, xₜ₋₁, m)  —  rejection-sampled against free space'
          : 'p(xₜ | uₜ, xₜ₋₁)  —  the map is not consulted',
        sx(v, 0.5),
        sy(v, 5.72),
        params.conditioned ? p.prediction : p.truth,
        { size: 11, weight: 600 },
      );
      if (params.conditioned && stats.ghosts > 0) {
        label(
          ctx,
          `${stats.ghosts} accepted samples reached free space through a wall`,
          sx(v, 0.5),
          sy(v, 5.46),
          p.accent,
          { size: 10, weight: 600 },
        );
      }
    },
    [sim.state.verdicts, params.conditioned, start, cmd, stats.ghosts],
  );

  return (
    <WidgetFrame
      id="w9.4"
      title="Map Squeeze"
      teaches="Conditioning motion on the map is not clipping. It renormalizes the banana into free space — and it checks only where the sample stopped, never how it got there."
      colorKey={['prior', 'prediction', 'truth']}
      caption={
        <>
          Rusty is pointed at a doorway. With conditioning <strong>off</strong>, a healthy fraction
          of the banana lies inside the wall: poses the robot could not possibly occupy still carry
          probability mass. Turn conditioning <strong>on</strong> and those samples are rejected,
          the survivors renormalized — the distribution is squeezed through the gap and the
          remaining mass is genuinely bimodal about the door frame. Now look at the ringed samples.
          Every one of them ended in legal free space and got accepted, and every one of them
          crossed a wall on the way. That is not a bug in the implementation; it is what
          p(x<sub>t</sub> | m) <em>means</em>. Push the noise up, or curve the command with ω, and
          the ring count climbs. Move Rusty to the open corridor and it collapses to a handful,
          which is why the approximation survived twenty years of deployment. Then try{' '}
          <strong>Facing a wall</strong>, where the robot physically cannot reach the corridor:
          four out of five accepted samples are on the other side of a solid wall, and the model
          has no way to know.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 0.3, maxX: 4.9, minY: 1.7, maxY: 5.9 }}
        draw={draw}
        deps={[sim.state.verdicts, params, stats.ghosts]}
        aspect={1.55}
        padding={0.1}
        ariaLabel="A robot in an apartment facing a doorway, with a cloud of sampled poses truncated by the walls; some accepted samples on the far side of the wall are ringed."
      />

      <div className="grid grid-cols-2 divide-x divide-fd-border border-t border-fd-border text-center sm:grid-cols-4">
        <Readout label="drawn" value={String(stats.total)} />
        <Readout label="accepted" value={String(stats.accepted)} />
        <Readout label="acceptance rate" value={`${(stats.acceptRate * 100).toFixed(0)}%`} />
        <Readout
          label="accepted through a wall"
          value={`${stats.ghosts} (${(stats.ghostRate * 100).toFixed(0)}%)`}
          alert={stats.ghostRate > 0.05}
        />
      </div>

      <ControlPanel columns={1} title="where Rusty stands">
        <ButtonRow>
          {(Object.keys(SPOTS) as SpotId[]).map((id) => (
            <ActionButton
              key={id}
              emphasis={params.spot === id}
              onClick={() => setParams((p) => ({ ...p, spot: id }))}
            >
              {SPOTS[id].label}
            </ActionButton>
          ))}
        </ButtonRow>
      </ControlPanel>

      <ControlPanel columns={3}>
        <Slider
          label="Motion noise α"
          role="prediction"
          value={params.noise}
          min={0.01}
          max={0.3}
          step={0.005}
          onChange={(v) => setParams((p) => ({ ...p, noise: v }))}
          help="One knob driving all six α's — the headline control here."
        />
        <Slider
          label="Commanded ω"
          value={params.omega}
          min={-1}
          max={1}
          step={0.05}
          unit="rad/s"
          onChange={(v) => setParams((p) => ({ ...p, omega: v }))}
          help="Curving the command bends paths through wall corners while the endpoints stay legal."
        />
        <Toggle
          label="Condition on the map"
          role="prediction"
          checked={params.conditioned}
          onChange={(v) => setParams((p) => ({ ...p, conditioned: v }))}
        />
      </ControlPanel>

      <Transport
        playing={sim.playing}
        onToggle={sim.toggle}
        onStep={sim.stepOnce}
        onReset={sim.reset}
        onReseed={() => sim.reseed()}
        seed={sim.seed}
        tick={sim.tick}
      />
    </WidgetFrame>
  );
}

function Readout({ label: l, value, alert }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={alert ? { color: 'var(--color-fd-primary)' } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
