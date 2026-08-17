'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  DEFAULT_PENDLE,
  energy,
  INTEGRATORS,
  simulate,
  swingUpController,
  wrapAngle,
  type Integrator,
  type PendleState,
} from '@/lib/rl/pendulum';
import { LineChart } from '@/components/viz/LineChart';
import { StatTile } from '@/components/viz/StatTile';
import { Segmented, SimControls, SimPanel, Slider } from './SimControls';
import { useTheme } from '@/components/layout/ThemeProvider';
import { seriesColor } from '@/lib/theme';

/**
 * `ch02-integrator-playground` — Pendle's continuous dynamics, discretized.
 *
 * The reader cranks Δt and watches explicit Euler pump energy into a system
 * that has none to give, while RK4 holds the line. This is the chapter's
 * bridge from "robots live in continuous time" to "our MDP has time steps",
 * and the same fact returns in Chapter 15 as a sim-to-real failure mode.
 */
export function PendleSim() {
  const { mode } = useTheme();
  const [integrator, setIntegrator] = useState<Integrator>('rk4');
  const [dt, setDt] = useState(0.02);
  const [control, setControl] = useState<'none' | 'swingup'>('none');
  const [playing, setPlaying] = useState(true);
  const [state, setState] = useState<PendleState>([Math.PI - 0.15, 0]);
  const [trace, setTrace] = useState<Array<{ t: number; e: number }>>([]);
  const raf = useRef<number | null>(null);
  const tRef = useRef(0);
  // Grabbing the bob overrides the dynamics: the reader becomes an external
  // hand on the system, and letting go hands it back mid-motion.
  const [grabbed, setGrabbed] = useState(false);
  const grabRef = useRef<{ theta: number; lastTheta: number; lastT: number } | null>(null);
  const [pushCount, setPushCount] = useState(0);

  const reset = () => {
    setState([Math.PI - 0.15, 0]);
    setTrace([]);
    tRef.current = 0;
  };

  useEffect(() => {
    if (!playing) return;
    let mounted = true;
    const tick = () => {
      if (!mounted) return;
      setState((s) => {
        if (grabRef.current) {
          // Under the reader's hand: position is imposed, velocity is inferred
          // from how fast they are moving it, so releasing transfers momentum.
          const now = performance.now();
          const dtGrab = Math.max(1e-3, (now - grabRef.current.lastT) / 1000);
          const omega = wrapAngle(grabRef.current.theta - grabRef.current.lastTheta) / dtGrab;
          grabRef.current.lastTheta = grabRef.current.theta;
          grabRef.current.lastT = now;
          return [grabRef.current.theta, Math.max(-14, Math.min(14, omega))];
        }
        const tau = control === 'swingup' ? swingUpController(s) : 0;
        const next = INTEGRATORS[integrator](s, tau, dt);
        tRef.current += dt;
        if (!Number.isFinite(next[0]) || Math.abs(next[1]) > 500) {
          setPlaying(false);
          return s;
        }
        setTrace((tr) => {
          const out = [...tr, { t: tRef.current, e: energy(next) }];
          return out.length > 400 ? out.slice(-400) : out;
        });
        return next;
      });
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      mounted = false;
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, integrator, dt, control]);

  // Offline comparison: the same initial condition under all three integrators.
  const comparison = useMemo(() => {
    const initial: PendleState = [Math.PI - 0.15, 0];
    const steps = Math.min(1200, Math.floor(12 / dt));
    return (['euler', 'semi-implicit', 'rk4'] as Integrator[]).map((ig) => {
      const { energies } = simulate(initial, steps, dt, ig);
      const stride = Math.max(1, Math.floor(energies.length / 220));
      return {
        id: ig === 'semi-implicit' ? 'semi-implicit Euler' : ig === 'rk4' ? 'RK4' : 'explicit Euler',
        data: energies
          .filter((_, i) => i % stride === 0)
          .map((e, i) => ({ x: i * stride * dt, y: Number.isFinite(e) ? e : null }))
          .filter((p) => p.y !== null) as Array<{ x: number; y: number }>,
      };
    });
  }, [dt]);

  const theta = state[0];
  const L = 90;
  const bobX = 130 + L * Math.sin(theta);
  const bobY = 110 - L * Math.cos(theta);
  const diverged = !Number.isFinite(state[0]) || Math.abs(state[1]) > 100;

  return (
    <SimPanel
      title="Pendle: continuous dynamics, discrete steps"
      id="ch02-integrator-playground"
      subtitle="m ℓ² θ̈ = m g ℓ sin θ − b θ̇ + τ, integrated three ways. θ = 0 is upright."
      controls={
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-end gap-4">
            <Segmented
              label="Integrator"
              value={integrator}
              onChange={(v) => {
                setIntegrator(v);
                reset();
              }}
              options={[
                { value: 'euler', label: 'Explicit Euler' },
                { value: 'semi-implicit', label: 'Semi-implicit' },
                { value: 'rk4', label: 'RK4' },
              ]}
            />
            <Segmented
              label="Torque"
              value={control}
              onChange={setControl}
              options={[
                { value: 'none', label: 'τ = 0 (free)' },
                { value: 'swingup', label: 'Energy swing-up' },
              ]}
            />
            <SimControls
              playing={playing}
              onPlayPause={() => setPlaying((p) => !p)}
              onReset={() => {
                setPlaying(false);
                setPushCount(0);
                reset();
              }}
            />
            <button
              type="button"
              onClick={() => {
                // An impulse, as if someone knocked the pole.
                setState(([th, om]) => [th, om + (Math.random() > 0.5 ? 3.2 : -3.2)]);
                setPushCount((n) => n + 1);
                setPlaying(true);
              }}
              className="rounded-md border border-hairline px-2.5 py-1.5 text-[12px] font-medium text-ink-secondary transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              Knock it ({pushCount})
            </button>
          </div>
          <Slider
            label="Time step Δt"
            value={dt}
            min={0.002}
            max={0.14}
            step={0.002}
            onChange={(v) => {
              setDt(v);
              reset();
            }}
            format={(v) => `${(v * 1000).toFixed(0)} ms`}
            hint="explicit Euler is stable only for small Δt — crank it and watch"
          />
        </div>
      }
      caption="Grab the bob and fling it, or knock it with the button — then switch the torque to energy swing-up and watch the controller fight its way back to upright. With τ = 0 the true system conserves energy exactly. Explicit Euler's energy climbs without bound (the pendulum spins up out of nothing), semi-implicit Euler oscillates around the truth, and RK4 tracks it to fourth order. Chapter 15 shows the same arithmetic deciding whether a policy trained in simulation survives contact with a real robot."
    >
      <div className="grid gap-4 lg:grid-cols-[260px,1fr]">
        <div>
          <svg
            width={260}
            height={220}
            viewBox="0 0 260 220"
            className="max-w-full touch-none rounded-lg"
            style={{ background: 'var(--surface-sunken)', cursor: grabbed ? 'grabbing' : 'default' }}
            onPointerMove={(e) => {
              if (!grabRef.current) return;
              const r = e.currentTarget.getBoundingClientRect();
              const sc = 260 / r.width;
              const x = (e.clientX - r.left) * sc - 130;
              const y = (e.clientY - r.top) * sc - 110;
              // Screen y grows downward; θ = 0 is straight up.
              grabRef.current.theta = Math.atan2(x, -y);
            }}
            onPointerUp={() => {
              grabRef.current = null;
              setGrabbed(false);
            }}
            onPointerLeave={() => {
              grabRef.current = null;
              setGrabbed(false);
            }}
            role="img"
            aria-label={`Pendulum at angle ${(wrapAngle(theta) * 180 / Math.PI).toFixed(0)} degrees from upright`}
          >
            {/* Upright reference */}
            <line
              x1={130}
              y1={110}
              x2={130}
              y2={20}
              stroke="var(--baseline)"
              strokeWidth={1}
              strokeDasharray="3 4"
            />
            <circle cx={130} cy={110} r={4} fill="var(--text-muted)" />
            {!diverged && (
              <>
                <line
                  x1={130}
                  y1={110}
                  x2={bobX}
                  y2={bobY}
                  stroke={seriesColor(0, mode)}
                  strokeWidth={3}
                  strokeLinecap="round"
                />
                <circle
                  cx={bobX}
                  cy={bobY}
                  r={13}
                  fill={seriesColor(0, mode)}
                  stroke={grabbed ? 'var(--series-4)' : 'var(--surface-1)'}
                  strokeWidth={grabbed ? 3 : 2}
                  style={{ cursor: 'grab' }}
                  onPointerDown={(e) => {
                    e.currentTarget.setPointerCapture?.(e.pointerId);
                    grabRef.current = {
                      theta: state[0],
                      lastTheta: state[0],
                      lastT: performance.now(),
                    };
                    setGrabbed(true);
                    setPlaying(true);
                  }}
                />
                {!grabbed && (
                  <text
                    x={bobX}
                    y={bobY - 20}
                    textAnchor="middle"
                    fontSize={9}
                    fill="var(--text-muted)"
                  >
                    drag me
                  </text>
                )}
              </>
            )}
            {diverged && (
              <text
                x={130}
                y={110}
                textAnchor="middle"
                fontSize={12}
                fill="var(--status-critical)"
                fontWeight={600}
              >
                diverged
              </text>
            )}
            <text x={10} y={205} fontSize={10} fill="var(--text-muted)">
              θ = {(wrapAngle(theta) * (180 / Math.PI)).toFixed(1)}° · θ̇ ={' '}
              {Number.isFinite(state[1]) ? state[1].toFixed(2) : '—'} rad/s
            </text>
          </svg>

          <div className="mt-2 grid grid-cols-2 gap-2">
            <StatTile
              label="Energy"
              value={Number.isFinite(state[0]) ? energy(state) : NaN}
              unit="J"
              status={diverged ? 'critical' : undefined}
              hint={diverged ? 'integrator unstable' : 'should stay flat at τ=0'}
            />
            <StatTile
              label="Δt"
              value={dt * 1000}
              unit="ms"
              hint={`${(1 / dt).toFixed(0)} Hz control rate`}
            />
          </div>
        </div>

        <LineChart
          data={comparison}
          height={280}
          xLegend="simulated time (s)"
          yLegend="total energy (J)"
          caption="Total mechanical energy under each integrator from the same initial state, τ = 0. A flat line is correct physics."
        />
      </div>
    </SimPanel>
  );
}
