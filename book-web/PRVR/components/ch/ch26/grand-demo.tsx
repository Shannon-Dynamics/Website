'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { APARTMENT } from '@/lib/sim/world';
import type { Palette, Viewport } from '@/lib/sim/draw';
import { AutonomyStack, MODE_ORDER, TASKS, type Mode } from '@/lib/capstone/stack';
import { positionSigma } from '@/lib/capstone/esdf';
import { Bar, Panel, Readout, drawScene } from './shared';

/**
 * w26.1 — The Grand Demo.
 *
 * The book's finale, and the only widget in it that runs *everything*: sensing,
 * SLAM, mapping, the distance field, frontier selection, A*, MPPI, and a
 * supervisor watching three detectors. Nothing is scripted. Press a chaos
 * button and the recovery you see is a threshold being crossed followed by a
 * mode with a plan, exactly as Derivation F4 describes.
 */

const TABS = ['Belief', 'Frontiers', 'MPPI', 'Timing'] as const;
type Tab = (typeof TABS)[number];

interface State {
  stack: AutonomyStack;
  trailEst: { x: number; y: number }[];
  trailTruth: { x: number; y: number }[];
}

export function GrandDemo() {
  const [showTruth, setShowTruth] = useState(false);
  const [calibrated, setCalibrated] = useState(true);
  const [delta, setDelta] = useState(0.01);
  const [walkerSpeed, setWalkerSpeed] = useState(0.8);
  const [tab, setTab] = useState<Tab>('Belief');

  const init = useCallback(
    (seed: number): State => ({
      stack: new AutonomyStack({ seed }),
      trailEst: [],
      trailTruth: [],
    }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      s.stack.cfg.calibrated = calibrated;
      s.stack.cfg.delta = delta;
      s.stack.step();
      if (s.stack.tick % 4 === 0) {
        s.trailEst.push({ x: s.stack.belief.mean.x, y: s.stack.belief.mean.y });
        s.trailTruth.push({ x: s.stack.truth.x, y: s.stack.truth.y });
        if (s.trailEst.length > 700) {
          s.trailEst.shift();
          s.trailTruth.shift();
        }
      }
      return { ...s };
    },
    [calibrated, delta],
  );

  const sim = useSimulation<State>({ init, step, fps: 45, initialSeed: 42, loop: false });
  const st = sim.state.stack;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      drawScene(ctx, v, p, sim.state.stack, {
        showTruth,
        showRollouts: tab === 'MPPI' || tab === 'Belief',
        showFrontiers: tab === 'Frontiers' || tab === 'Belief',
        showScan: true,
        showPath: true,
        trailEst: sim.state.trailEst,
        trailTruth: sim.state.trailTruth,
      });
    },
    [sim.state, showTruth, tab],
  );

  const drawTimeline = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      ctx.clearRect(0, 0, v.width, v.height);
      ctx.fillStyle = p.bg;
      ctx.fillRect(0, 0, v.width, v.height);
      const h = sim.state.stack.history;
      if (h.length < 2) return;

      const t1 = Math.max(h[h.length - 1].t, 20);
      const hMax = Math.max(...h.map((s) => s.entropy), 1);
      const X = (t: number) => (t / t1) * v.width;
      const Y = (e: number) => v.height - 6 - (e / hMax) * (v.height - 16);

      // Map entropy H(m_t): the mission's objective, falling.
      ctx.strokeStyle = p.prior;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      h.forEach((s, i) => (i === 0 ? ctx.moveTo(X(s.t), Y(s.entropy)) : ctx.lineTo(X(s.t), Y(s.entropy))));
      ctx.stroke();

      // Coverage, on its own 0–1 scale.
      ctx.strokeStyle = p.measurement;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      h.forEach((s, i) => {
        const y = v.height - 6 - s.coverage * (v.height - 16);
        return i === 0 ? ctx.moveTo(X(s.t), y) : ctx.lineTo(X(s.t), y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // Event pins.
      for (const e of sim.state.stack.events) {
        if (e.kind === 'ModeSwitch' || e.kind === 'GoalSelected' || e.kind === 'GoalReached') continue;
        ctx.strokeStyle = e.kind === 'ChaosInjected' ? p.prediction : p.posterior;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(X(e.t), 2);
        ctx.lineTo(X(e.t), v.height - 2);
        ctx.stroke();
      }

      ctx.fillStyle = p.prior;
      ctx.font = '600 9px ui-monospace, monospace';
      ctx.fillText(`H(m) = ${h[h.length - 1].entropy.toFixed(0)} bits`, 4, 10);
      ctx.fillStyle = p.measurement;
      ctx.fillText(`coverage ${(h[h.length - 1].coverage * 100).toFixed(1)}%`, 130, 10);
    },
    [sim.state],
  );

  const sigma = positionSigma(st.belief.cov);
  const recentEvents = useMemo(() => st.events.slice(-7).reverse(), [st.events, sim.tick]);

  return (
    <WidgetFrame
      id="w26.1"
      title="The Grand Demo"
      teaches="An autonomy stack is not a magic monolith: every internal is inspectable, and each recovery is a named statistic crossing a threshold."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      wide
      caption={
        <>
          Rusty starts in the middle of an apartment it has never seen, knowing only that it is at
          the origin of its own map frame. Watch the gray fog resolve into walls, the teal plan
          reach for the nearest <strong>blue frontier</strong>, and the orange MPPI fan bend around
          corners. The purple ellipse is the pose belief and the dashed orange ring around Rusty is
          the safety margin <span className="font-mono">r + k<sub>σ</sub>σ</span> of Derivation F2 —
          it visibly breathes as the filter gains and loses confidence.
          <br />
          <br />
          Then break it. <strong>Kidnap</strong> teleports Rusty into a room it has already mapped:
          the fitness ratio ρ dives, the supervisor switches to <em>Relocalize</em>, six thousand
          particles scatter across the known-free space and condense. <strong>Walker</strong> sends
          a person through the LiDAR — watch the novelty beams flag orange and the map refuse to
          learn them. <strong>Dropout</strong> cuts the sensor: the watchdog fires within 0.3 s, the
          margin ring swells while the estimator runs open loop, and Rusty creeps to a stop — at
          which point the ring stops growing, because a robot that is not moving accumulates no
          process noise. Finally, turn off <em>calibrated sensor model</em>: the scan matcher then
          treats each beam as five times more independent evidence than it is, the belief tightens
          by about a quarter, the margin ring shrinks — and the NIS bar climbs, because a filter
          that claims more certainty than it has finds its own updates surprising.
        </>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-[186px_minmax(0,1fr)_268px]">
        {/* ---- left rail: mode machine + chaos ---- */}
        <div className="order-2 border-t border-fd-border lg:order-1 lg:border-t-0 lg:border-r">
          <Panel title="mode (D26.4)">
            <ul className="space-y-1">
              {MODE_ORDER.map((k) => {
                const active = st.mode.kind === k;
                return (
                  <li
                    key={k}
                    className="flex items-center gap-2 font-mono text-[0.72rem]"
                    style={{ opacity: active ? 1 : 0.42 }}
                  >
                    <span
                      className="inline-block size-2 rounded-full"
                      style={{ backgroundColor: active ? modeColorVar(st.mode) : 'var(--pr-grid)' }}
                    />
                    {k}
                  </li>
                );
              })}
            </ul>
          </Panel>

          <Panel title="chaos">
            <ButtonRow>
              <ActionButton onClick={() => sim.setState((s) => (s.stack.kidnap(), { ...s }))} emphasis>
                Kidnap
              </ActionButton>
              <ActionButton onClick={() => sim.setState((s) => (s.stack.spawnWalker(walkerSpeed, 16), { ...s }))}>
                Walker
              </ActionButton>
              <ActionButton onClick={() => sim.setState((s) => (s.stack.dropSensor(2.5), { ...s }))}>
                Dropout
              </ActionButton>
            </ButtonRow>
          </Panel>

          <Panel title="detectors (F4)">
            <div className="space-y-1.5">
              <DetectorRow
                name="ρ  fitness"
                value={st.fitness.rho.toFixed(2)}
                frac={Math.min(1, st.fitness.rho)}
                warn={st.fitness.rho < st.fitness.rhoMin}
                note={`< ${st.fitness.rhoMin} ⇒ Relocalize`}
              />
              <DetectorRow
                name="ε  NIS"
                value={st.nis.toFixed(1)}
                frac={Math.min(1, st.nis / 16.3)}
                warn={st.nis > 7.815}
                note="χ²₃ gate 7.82 / 16.27"
              />
              <DetectorRow
                name="scan age"
                value={`${st.watchdog.age(st.time).toFixed(2)} s`}
                frac={Math.min(1, st.watchdog.age(st.time) / (3 * st.watchdog.period))}
                warn={st.watchdog.expired(st.time)}
                note="> 3τ ⇒ Recover"
              />
            </div>
          </Panel>
        </div>

        {/* ---- centre: the world ---- */}
        <div className="order-1 min-w-0 lg:order-2">
          <SimCanvas
            world={APARTMENT.bounds}
            draw={draw}
            deps={[sim.tick, sim.state, showTruth, tab]}
            aspect={12 / 9}
            padding={0.25}
            ariaLabel="A floorplan being mapped by a robot: gray unknown space resolving into white corridors and dark walls, with a purple pose-belief ellipse, a green laser scan, an orange fan of sampled control rollouts, and a teal planned path to a blue frontier."
          />
          <div className="border-t border-fd-border">
            <SimCanvas
              world={{ minX: 0, minY: 0, maxX: 1, maxY: 1 }}
              draw={drawTimeline}
              deps={[sim.tick, sim.state]}
              aspect={8}
              padding={0}
              ariaLabel="Mission timeline: map entropy falling over time, coverage rising, with vertical pins marking detector events."
            />
          </div>
        </div>

        {/* ---- right rail: inspectors ---- */}
        <div className="order-3 border-t border-fd-border lg:border-t-0 lg:border-l">
          <div className="flex border-b border-fd-border">
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                aria-pressed={tab === t}
                className={`flex-1 px-1 py-1.5 font-ui text-[0.68rem] font-medium transition-colors ${
                  tab === t ? 'bg-fd-accent text-fd-foreground' : 'text-fd-muted-foreground hover:bg-fd-accent/50'
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          {tab === 'Belief' ? (
            <>
              <div className="grid grid-cols-2 divide-x divide-fd-border border-b border-fd-border">
                <Readout label="σ_pose" value={`${sigma.toFixed(3)} m`} role="posterior" />
                <Readout label="F2 margin" value={`${st.margin.toFixed(3)} m`} role="prediction" />
              </div>
              <div className="grid grid-cols-2 divide-x divide-fd-border border-b border-fd-border">
                <Readout label="|est − truth|" value={`${st.error().toFixed(3)} m`} role="truth" />
                <Readout label="odometry only" value={`${odomError(st).toFixed(2)} m`} role="truth" />
              </div>
              <Panel title={st.mode.kind === 'Relocalize' ? 'relocalisation' : 'scan matching'}>
                {st.mode.kind === 'Relocalize' && st.reloc ? (
                  <dl className="space-y-1 font-mono text-[0.72rem]">
                    <Row k="particles" v={String(st.relocalizer.particles.length)} />
                    <Row k="spread" v={`${st.reloc.spread.toFixed(2)} m`} />
                    <Row k="modes" v={String(st.reloc.clusters)} />
                    <Row k="ESS" v={`${(st.reloc.ess * 100).toFixed(0)}%`} />
                    <Row k="scans" v={String(st.relocalizer.steps)} />
                  </dl>
                ) : (
                  <dl className="space-y-1 font-mono text-[0.72rem]">
                    <Row k="fitness" v={st.icpFitness.toFixed(3)} />
                    <Row k="ICP rmse" v={`${st.icpRmse.toFixed(3)} m`} />
                    <Row k="novel beams" v={String(st.novelty.length)} />
                    <Row k="map entropy" v={`${st.grid.entropy().toFixed(0)} bits`} />
                  </dl>
                )}
              </Panel>
            </>
          ) : null}

          {tab === 'Frontiers' ? (
            <Panel title={`${st.frontiers.length} frontiers · utility = gain · e^(−λ·cost)`}>
              {st.frontiers.length === 0 ? (
                <p className="font-ui text-[0.72rem] text-fd-muted-foreground">
                  None. Either the map is finished or the explorer is switched off.
                </p>
              ) : (
                <ul className="space-y-1.5">
                  {st.frontiers.slice(0, 7).map((f, i) => (
                    <li key={f.id} className="space-y-0.5">
                      <div className="flex justify-between font-mono text-[0.68rem] tabular-nums">
                        <span>
                          {i === 0 ? '★ ' : '  '}gain {f.gain}
                        </span>
                        <span className="opacity-70">
                          {Number.isFinite(f.cost) ? `${f.cost.toFixed(1)} m` : 'unreachable'}
                        </span>
                      </div>
                      <Bar
                        frac={f.utility / Math.max(st.frontiers[0].utility, 1e-9)}
                        color={i === 0 ? 'var(--color-fd-primary)' : 'var(--pr-prior)'}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          ) : null}

          {tab === 'MPPI' ? (
            <>
              <div className="grid grid-cols-2 divide-x divide-fd-border border-b border-fd-border">
                <Readout label="v" value={`${st.cmd.v.toFixed(2)} m/s`} role="prediction" />
                <Readout label="ω" value={`${st.cmd.omega.toFixed(2)} rad/s`} role="prediction" />
              </div>
              <Panel title="rollouts">
                <dl className="space-y-1 font-mono text-[0.72rem]">
                  <Row k="K × H" v={`56 × 16`} />
                  <Row k="best cost" v={st.lastMppi ? st.lastMppi.bestCost.toFixed(1) : '—'} />
                  <Row
                    k="inside margin"
                    v={st.lastMppi ? `${(st.lastMppi.infeasible * 100).toFixed(0)}%` : '—'}
                  />
                </dl>
                {st.lastMppi ? (
                  <div className="mt-2">
                    <Bar
                      frac={st.lastMppi.infeasible}
                      color="var(--pr-prediction)"
                      warn={st.lastMppi.infeasible > 0.5}
                    />
                    <p className="mt-1 font-ui text-[0.65rem] leading-snug text-fd-muted-foreground">
                      The fraction of imagined futures that would violate the F2 margin. When it
                      approaches 1 there is no safe control left and the robot stops.
                    </p>
                  </div>
                ) : null}
              </Panel>
            </>
          ) : null}

          {tab === 'Timing' ? (
            <Panel title="staleness vs. period (F3)">
              <ul className="space-y-1.5">
                {TASKS.map((t) => {
                  const s = st.stats[t.name];
                  const period = t.period * 0.05;
                  const off = st.cfg.disabled[t.name];
                  return (
                    <li key={t.name}>
                      <div className="flex justify-between font-mono text-[0.66rem] tabular-nums">
                        <span style={{ opacity: off ? 0.4 : 1 }}>{t.label}</span>
                        <span className="opacity-70">
                          {off ? 'off' : `${s.hz.toFixed(1)} Hz`}
                        </span>
                      </div>
                      <Bar
                        frac={Math.min(1, s.staleness / (3 * period))}
                        color="var(--pr-measurement)"
                        warn={s.staleness > 2.5 * period}
                      />
                    </li>
                  );
                })}
              </ul>
            </Panel>
          ) : null}

          <Panel title="event log">
            <ul className="space-y-0.5 font-mono text-[0.64rem] leading-tight">
              {recentEvents.length === 0 ? (
                <li className="text-fd-muted-foreground">—</li>
              ) : (
                recentEvents.map((e, i) => (
                  <li key={`${e.t}-${i}`} className="flex gap-1.5">
                    <span className="shrink-0 opacity-60">{e.t.toFixed(1)}</span>
                    <span
                      style={{
                        color:
                          e.kind === 'ChaosInjected'
                            ? 'var(--pr-prediction)'
                            : e.kind === 'RelocalizeConverged' || e.kind === 'MissionComplete'
                              ? 'var(--pr-measurement)'
                              : undefined,
                      }}
                    >
                      {e.kind}
                    </span>
                  </li>
                ))
              )}
            </ul>
          </Panel>
        </div>
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="Collision bound δ"
          role="prediction"
          value={delta}
          min={0.0005}
          max={0.2}
          step={0.0005}
          onChange={setDelta}
          format={(v) => v.toFixed(4)}
          help="Derivation F2. Smaller δ ⇒ larger k_σ ⇒ a wider margin ring and a more timid robot."
        />
        <Slider
          label="Walker speed"
          role="truth"
          value={walkerSpeed}
          min={0.3}
          max={2.6}
          step={0.1}
          unit="m/s"
          onChange={setWalkerSpeed}
          help="Derivation F3. Predict the speed at which the stack stops clearing the walker before you go looking for it."
        />
        <Toggle label="Ground-truth overlay" role="truth" checked={showTruth} onChange={setShowTruth} />
        <Toggle
          label="Calibrated sensor model"
          role="measurement"
          checked={calibrated}
          onChange={setCalibrated}
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
        speed={sim.speed}
        onSpeed={sim.setSpeed}
      />
    </WidgetFrame>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between">
      <dt className="opacity-65">{k}</dt>
      <dd className="tabular-nums">{v}</dd>
    </div>
  );
}

function DetectorRow({
  name,
  value,
  frac,
  warn,
  note,
}: {
  name: string;
  value: string;
  frac: number;
  warn: boolean;
  note: string;
}) {
  return (
    <div>
      <div className="flex justify-between font-mono text-[0.66rem] tabular-nums">
        <span>{name}</span>
        <span style={{ color: warn ? 'var(--pr-prediction)' : undefined }}>{value}</span>
      </div>
      <Bar frac={frac} color="var(--pr-posterior)" warn={warn} />
      <p className="mt-0.5 font-mono text-[0.58rem] opacity-55">{note}</p>
    </div>
  );
}

function modeColorVar(m: Mode): string {
  switch (m.kind) {
    case 'Relocalize':
      return 'var(--pr-posterior)';
    case 'Recover':
      return 'var(--pr-prediction)';
    case 'Done':
      return 'var(--pr-measurement)';
    default:
      return 'var(--color-fd-primary)';
  }
}

const odomError = (s: AutonomyStack) => Math.hypot(s.deadReckon.x - s.truth.x, s.deadReckon.y - s.truth.y);
