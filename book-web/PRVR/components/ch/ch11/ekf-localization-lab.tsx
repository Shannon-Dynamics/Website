'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ActionButton, ButtonRow, ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { Dashboard, LineChart, StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import { angleDiff, normalizeAngle, type Pose2 } from '@/lib/geom/se2';
import { ellipse2, type Mat } from '@/lib/prob/linalg';
import { chi2Envelope, rmse } from '@/lib/filters/consistency';
import { sampleMotionModelVelocity, type MotionAlphas, type VelocityCmd } from '@/lib/models/motion';
import { APARTMENT, collides, type Landmark } from '@/lib/sim/world';
import {
  EkfLocalizer,
  gateThreshold,
  poseNees,
  positionBlock,
  type Association,
  type Feature,
  type PoseBelief,
} from '@/lib/filters/ekf-localization';
import {
  clear,
  drawCovariance,
  drawPath,
  drawRobot,
  drawWorld,
  label,
  sl,
  sx,
  sy,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';

/**
 * w11.2 — the EKF Localization Lab.
 *
 * Rusty drives a closed loop down the Apartment's corridor while the real
 * `EkfLocalizer` from the library tracks him. The purple ellipse breathes:
 * swelling wherever no landmark is in the cone, snapping tight the instant one
 * is.
 *
 * The NEES chart underneath is the point of the widget. RMSE cannot tell a
 * good filter from a lucky one, and a small ellipse is not evidence of
 * anything. NEES compares the error the filter *makes* to the error it
 * *claims*, and a run that leaves the shaded band is lying — usually because a
 * feature was matched to the wrong landmark and the filter has no way to know.
 */

/* -------------------------------------------------------------------------- */
/* Trajectory: a racetrack down the corridor                                   */
/* -------------------------------------------------------------------------- */

const DT = 0.15;
const V = 0.9;
const STRAIGHT_STEPS = 66;
const TURN_STEPS = 8;
const LAP = 2 * (STRAIGHT_STEPS + TURN_STEPS);
const TURN_OMEGA = Math.PI / (TURN_STEPS * DT);
const START: Pose2 = { x: 1.5, y: 4.15, theta: 0 };

/** Commanded (v, ω) at tick t: straight, hard left, straight, hard left. */
function command(t: number): VelocityCmd {
  const k = t % LAP;
  const turning = k >= STRAIGHT_STEPS && k < STRAIGHT_STEPS + TURN_STEPS;
  const turning2 = k >= 2 * STRAIGHT_STEPS + TURN_STEPS;
  return { v: V, omega: turning || turning2 ? TURN_OMEGA : 0, dt: DT };
}

/** Motion noise, shared by the simulator and the filter — so any inconsistency
 *  the NEES chart shows comes from *association*, never from a rigged model. */
const ALPHAS: MotionAlphas = [0.025, 0.005, 0.005, 0.025, 0.001, 0.001];

const FOV = (2 * Math.PI) / 3; // ±60°, a forward-looking feature detector

/* -------------------------------------------------------------------------- */
/* Landmark presets                                                            */
/* -------------------------------------------------------------------------- */

type PresetId = 'sparse' | 'dense' | 'pair';

const SOUTH = 3.92;
const NORTH = 4.9;

const PRESETS: Record<PresetId, { label: string; landmarks: Landmark[] }> = {
  sparse: {
    label: 'sparse',
    landmarks: [
      { x: 1.2, y: NORTH, id: 0 },
      { x: 4.6, y: SOUTH, id: 1 },
      { x: 8.2, y: NORTH, id: 2 },
      { x: 11.0, y: SOUTH, id: 3 },
    ],
  },
  dense: {
    label: 'dense',
    landmarks: [
      { x: 1.2, y: NORTH, id: 0 },
      { x: 2.6, y: SOUTH, id: 1 },
      { x: 3.9, y: NORTH, id: 2 },
      { x: 5.2, y: SOUTH, id: 3 },
      { x: 6.5, y: NORTH, id: 4 },
      { x: 7.8, y: SOUTH, id: 5 },
      { x: 9.1, y: NORTH, id: 6 },
      { x: 10.4, y: SOUTH, id: 7 },
      { x: 11.4, y: NORTH, id: 8 },
      { x: 0.7, y: SOUTH, id: 9 },
    ],
  },
  pair: {
    label: 'close pair',
    landmarks: [
      { x: 1.2, y: NORTH, id: 0 },
      { x: 2.6, y: SOUTH, id: 1 },
      { x: 3.9, y: NORTH, id: 2 },
      { x: 5.2, y: SOUTH, id: 3 },
      { x: 6.5, y: NORTH, id: 4 },
      // The pathological pair: the two edges of a doorframe, 0.30 m apart, both
      // surveyed as landmarks — and both comfortably inside a gate that is a
      // metre and a half long in the range direction.
      { x: 7.8, y: SOUTH, id: 5 },
      { x: 8.1, y: SOUTH, id: 6 },
      { x: 9.1, y: NORTH, id: 7 },
      { x: 10.4, y: SOUTH, id: 8 },
      { x: 11.4, y: NORTH, id: 9 },
    ],
  },
};

/* -------------------------------------------------------------------------- */
/* Simulation                                                                  */
/* -------------------------------------------------------------------------- */

const HISTORY = 260;
const NEES_WINDOW = 25;
/** NEES is unbounded; a poisoned run reaches the thousands. Clip for display. */
const NEES_CLIP = 15;

interface Params {
  preset: PresetId;
  sensorRange: number;
  sigmaR: number;
  sigmaPhi: number;
  knownCorrespondence: boolean;
  confidence: number;
}

interface StepRecord {
  t: number;
  nees: number;
  err: number;
}

interface State {
  rng: Rng;
  truth: Pose2;
  bel: PoseBelief;
  history: StepRecord[];
  truthPath: { x: number; y: number }[];
  estPath: { x: number; y: number }[];
  /** Features observed this step, with what the associator decided. */
  shots: { feature: Feature; assoc: Association }[];
  matched: number;
  wrong: number;
  rejected: number;
}

const INITIAL_SIGMA: Mat = [
  [0.04, 0, 0],
  [0, 0.04, 0],
  [0, 0, 0.0025],
];

export function EkfLocalizationLab() {
  const [params, setParams] = useState<Params>({
    preset: 'dense',
    sensorRange: 3.2,
    sigmaR: 0.12,
    sigmaPhi: 0.035,
    knownCorrespondence: false,
    confidence: 0.95,
  });

  const init = useCallback(
    (seed: number): State => ({
      rng: new Rng(seed),
      truth: { ...START },
      bel: { mu: { ...START }, Sigma: INITIAL_SIGMA.map((r) => r.slice()) },
      history: [],
      truthPath: [{ x: START.x, y: START.y }],
      estPath: [{ x: START.x, y: START.y }],
      shots: [],
      matched: 0,
      wrong: 0,
      rejected: 0,
    }),
    [],
  );

  const step = useCallback(
    (s: State, tick: number): State => {
      const { rng } = s;
      const landmarks = PRESETS[params.preset].landmarks;
      const u = command(tick);

      // 1. The world moves. The wheels do not do exactly what they were told.
      const truth = sampleMotionModelVelocity(u, s.truth, ALPHAS, rng);

      // 2. The filter predicts with the *commanded* control and the Ch. 9
      //    Jacobians. The extra heading floor is the γ̂ term of Table 5.3: a
      //    rotation the (v, ω) parameterisation cannot express, so V M Vᵀ
      //    cannot contain it and it has to be added by hand.
      const gammaVar = (ALPHAS[4] * u.v * u.v + ALPHAS[5] * u.omega * u.omega) * DT * DT;
      const Q: Mat = [
        [params.sigmaR * params.sigmaR, 0],
        [0, params.sigmaPhi * params.sigmaPhi],
      ];
      const loc = new EkfLocalizer(s.bel, {
        landmarks,
        Q,
        alphas: ALPHAS,
        gate2: gateThreshold(params.confidence, 2),
        extraNoise: [
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, gammaVar],
        ],
      });
      loc.predict(u);

      // 3. Detect features: every landmark inside the cone, in range, and not
      //    behind a wall.
      const features: Feature[] = [];
      landmarks.forEach((lm, j) => {
        const dx = lm.x - truth.x;
        const dy = lm.y - truth.y;
        const r = Math.hypot(dx, dy);
        if (r > params.sensorRange) return;
        const phi = angleDiff(Math.atan2(dy, dx), truth.theta);
        if (Math.abs(phi) > FOV / 2) return;
        if (collides(APARTMENT, { x: truth.x, y: truth.y }, { x: lm.x, y: lm.y })) return;
        features.push({
          r: Math.max(0.05, r + rng.normal(0, params.sigmaR)),
          phi: normalizeAngle(phi + rng.normal(0, params.sigmaPhi)),
          truth: j,
        });
      });

      // 4. Correct — with the correspondence handed over, or inferred.
      const shots: { feature: Feature; assoc: Association }[] = [];
      let matched = s.matched;
      let wrong = s.wrong;
      let rejected = s.rejected;

      if (params.knownCorrespondence) {
        for (const f of features) {
          loc.correctKnown(f, f.truth as number);
          shots.push({ feature: f, assoc: { kind: 'match', index: f.truth as number, d2: 0, score: 0 } });
          matched += 1;
        }
      } else {
        const { associations } = loc.correct(features);
        associations.forEach((a, i) => {
          shots.push({ feature: features[i], assoc: a });
          if (a.kind === 'match') {
            matched += 1;
            if (a.index !== features[i].truth) wrong += 1;
          } else {
            rejected += 1;
          }
        });
      }

      const bel = loc.belief();
      const err = Math.hypot(truth.x - bel.mu.x, truth.y - bel.mu.y);
      const history = [...s.history, { t: tick, nees: poseNees(truth, bel), err }].slice(-HISTORY);

      return {
        rng,
        truth,
        bel,
        history,
        truthPath: [...s.truthPath, { x: truth.x, y: truth.y }].slice(-HISTORY),
        estPath: [...s.estPath, { x: bel.mu.x, y: bel.mu.y }].slice(-HISTORY),
        shots,
        matched,
        wrong,
        rejected,
      };
    },
    [params],
  );

  const sim = useSimulation<State>({ init, step, fps: 15, initialSeed: 11 });

  const landmarks = PRESETS[params.preset].landmarks;

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => {
      clear(ctx, v, p);
      const { truth, bel, shots } = sim.state;

      drawWorld(ctx, v, { ...APARTMENT, landmarks: undefined }, p);

      // Sensor cone, drawn from the *true* pose because that is where the
      // photons come from — a distinction that matters once the filter is lost.
      ctx.save();
      ctx.fillStyle = p.measurement;
      ctx.globalAlpha = 0.08;
      ctx.beginPath();
      ctx.moveTo(sx(v, truth.x), sy(v, truth.y));
      ctx.arc(
        sx(v, truth.x),
        sy(v, truth.y),
        sl(v, params.sensorRange),
        -(truth.theta + FOV / 2),
        -(truth.theta - FOV / 2),
      );
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Map landmarks. A landmark currently generating a feature is filled.
      const seen = new Set(shots.map((s) => s.feature.truth));
      landmarks.forEach((lm, j) => {
        ctx.save();
        ctx.strokeStyle = p.accent;
        ctx.fillStyle = p.accent;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx(v, lm.x), sy(v, lm.y), 4, 0, Math.PI * 2);
        if (seen.has(j)) ctx.fill();
        else ctx.stroke();
        ctx.restore();
      });

      drawPath(ctx, v, sim.state.truthPath, p.truth, { dashed: true, lineWidth: 1.4, alpha: 0.8 });
      drawPath(ctx, v, sim.state.estPath, p.posterior, { lineWidth: 1.6, alpha: 0.85 });

      // The correction: each accepted feature drawn from the *predicted*
      // landmark position to where the measurement put it.
      for (const shot of shots) {
        const a = shot.assoc;
        const zx = bel.mu.x + shot.feature.r * Math.cos(bel.mu.theta + shot.feature.phi);
        const zy = bel.mu.y + shot.feature.r * Math.sin(bel.mu.theta + shot.feature.phi);
        const isWrong = a.kind === 'match' && a.index !== shot.feature.truth;
        ctx.save();
        ctx.strokeStyle = a.kind === 'match' ? (isWrong ? p.prediction : p.measurement) : p.truth;
        ctx.globalAlpha = a.kind === 'match' ? 0.85 : 0.4;
        ctx.lineWidth = isWrong ? 2.4 : 1.2;
        if (a.kind !== 'match') ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(v, bel.mu.x), sy(v, bel.mu.y));
        ctx.lineTo(sx(v, zx), sy(v, zy));
        ctx.stroke();
        ctx.restore();
        if (isWrong) {
          const lm = landmarks[a.index];
          label(ctx, 'wrong match', sx(v, lm.x) + 7, sy(v, lm.y) - 10, p.prediction, {
            size: 10,
            weight: 700,
          });
        }
      }

      // Ground truth, then the posterior on top of it.
      drawRobot(ctx, v, truth, p.truth, 0.24, { filled: false });
      drawCovariance(ctx, v, [bel.mu.x, bel.mu.y], ellipse2(positionBlock(bel.Sigma), 2), p.posterior);
      drawRobot(ctx, v, bel.mu, p.posterior, 0.24);

      const sigTheta = Math.sqrt(Math.max(bel.Sigma[2][2], 0));
      label(
        ctx,
        `σθ = ${((sigTheta * 180) / Math.PI).toFixed(1)}°   features in view: ${shots.length}`,
        sx(v, 0.15),
        sy(v, 8.7),
        p.truth,
        { size: 10 },
      );
    },
    [sim.state, landmarks, params.sensorRange],
  );

  /* ---------------- readouts ---------------- */

  const stats = useMemo(() => {
    const h = sim.state.history;
    // Instantaneous NEES, clipped for display. Each value is marginally χ²₃
    // under a consistent filter, so the single-step 95% interval below is
    // exactly right — unlike a band drawn around a windowed *average*, whose
    // samples are anything but independent.
    const neesSeries = h.map((r) => ({ x: r.t, y: Math.min(r.nees, NEES_CLIP) }));
    const meanSeries: { x: number; y: number }[] = [];
    for (let i = NEES_WINDOW - 1; i < h.length; i++) {
      let acc = 0;
      for (let k = i - NEES_WINDOW + 1; k <= i; k++) acc += h[k].nees;
      meanSeries.push({ x: h[i].t, y: Math.min(acc / NEES_WINDOW, NEES_CLIP) });
    }
    const env = chi2Envelope(3, 1);
    const recent = h.slice(-NEES_WINDOW);
    const meanNees = recent.length ? recent.reduce((a, b) => a + b.nees, 0) / recent.length : 0;
    const inside = h.filter((r) => r.nees >= env.lo && r.nees <= env.hi).length;
    return {
      neesSeries,
      meanSeries,
      env,
      meanNees,
      coverage: h.length >= 30 ? inside / h.length : null,
      rmsePos: rmse(h.map((r) => r.err)),
      lastErr: h.length ? h[h.length - 1].err : 0,
      inconsistent: recent.length >= NEES_WINDOW && meanNees > 6,
    };
  }, [sim.state.history]);

  const setPreset = (preset: PresetId) => setParams((s) => ({ ...s, preset }));

  return (
    <WidgetFrame
      id="w11.2"
      title="The EKF Localization Lab"
      teaches="A small ellipse is not evidence of a good estimate. Consistency — not confidence — is what tells you whether to believe a filter."
      colorKey={['prediction', 'measurement', 'posterior', 'truth']}
      caption={
        <>
          Rusty laps the Apartment&rsquo;s corridor. Green rays are accepted features; gray dashed
          rays were rejected by the gate; an orange ray is a feature matched to the{' '}
          <em>wrong</em> landmark — the filter has no way to know, and you do only because this is a
          simulation. <strong>What to notice:</strong> the purple ellipse breathes. Shrink the
          sensor range until the corridor has landmark-free stretches and watch it swell along the
          direction of travel and snap tight the moment a landmark re-enters the cone. The NEES
          trace underneath should scatter around 3, with roughly one step in twenty poking above
          the χ² line — that is a filter telling the truth about itself. (The trace is clipped at
          15; a diverging run pins itself to the ceiling.) <strong>What to try:</strong> switch to
          the <em>close pair</em> preset, where two landmarks 0.30 m apart both fall inside the same
          gate. RMSE barely moves at first — and NEES goes through the roof, which is the whole
          argument for measuring consistency rather than error. Then re-roll the seed a few times
          on <em>dense</em>: roughly one run in four still catches a wrong match. Association is a
          probabilistic failure even in a map designed to avoid it.
        </>
      }
    >
      <SimCanvas
        world={APARTMENT.bounds}
        draw={draw}
        deps={[sim.tick, sim.state, params]}
        aspect={12 / 9}
        padding={0.25}
        ariaLabel="An apartment floorplan with a robot lapping the corridor. A purple uncertainty ellipse grows where no landmarks are visible and shrinks when features are matched; rays show accepted, rejected, and wrongly matched features."
      />

      <div className="border-t border-fd-border p-3">
        {stats.coverage !== null ? (
          <p className="mb-3 font-mono text-[0.7rem] text-fd-muted-foreground">
            NEES inside its 95% interval on {(stats.coverage * 100).toFixed(0)}% of steps
            <span className="opacity-60"> · 95% is what a truthful filter reports</span>
          </p>
        ) : null}
        <Dashboard columns={4}>
          <StatTile
            label="position error"
            value={stats.lastErr}
            unit="m"
            role="posterior"
            precision={3}
            sparkline={sim.state.history.slice(-60).map((r) => r.err)}
          />
          <StatTile label="RMSE (window)" value={stats.rmsePos} unit="m" precision={3} role="truth" />
          <StatTile
            label="mean NEES (25)"
            value={stats.meanNees}
            precision={2}
            role={stats.inconsistent ? 'prediction' : 'posterior'}
            trend={stats.meanNees - 3}
            trendLabel="from 3.00"
          />
          <StatTile
            label="wrong / rejected"
            value={`${sim.state.wrong} / ${sim.state.rejected}`}
            role="prediction"
          />
        </Dashboard>

        <div className="mt-3">
          {stats.neesSeries.length < 2 ? (
            <p className="flex h-[200px] items-center justify-center font-ui text-xs text-fd-muted-foreground">
              collecting the first NEES samples…
            </p>
          ) : (
          <LineChart
            series={[
              { id: 'NEES per step', role: 'posterior', data: stats.neesSeries },
              { id: `${NEES_WINDOW}-step mean`, role: 'truth', data: stats.meanSeries },
            ]}
            xLabel="step"
            yLabel={`NEES (clipped at ${NEES_CLIP})`}
            height={200}
            yMin={0}
            yMax={NEES_CLIP}
            curve="linear"
            markers={[
              { axis: 'y', value: 3, label: 'E[NEES] = 3', role: 'truth' },
              { axis: 'y', value: stats.env.hi, label: 'χ²₃ 97.5%', role: 'prediction' },
              { axis: 'y', value: stats.env.lo, role: 'prediction' },
            ]}
            ariaLabel="Normalized estimation error squared per step, with a 25-step running mean, against the 95 percent interval of the chi-squared distribution with three degrees of freedom."
          />
          )}
        </div>
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="Sensor range"
          role="measurement"
          value={params.sensorRange}
          min={1.2}
          max={6}
          step={0.1}
          unit="m"
          onChange={(v) => setParams((s) => ({ ...s, sensorRange: v }))}
          help="The one slider that matters here: it decides how long the corridor's landmark-free stretches are."
        />
        <Slider
          label="Range noise σ_r"
          role="measurement"
          value={params.sigmaR}
          min={0.03}
          max={0.45}
          step={0.01}
          unit="m"
          onChange={(v) => setParams((s) => ({ ...s, sigmaR: v }))}
        />
        <Slider
          label="Bearing noise σ_φ"
          role="measurement"
          value={params.sigmaPhi}
          min={0.01}
          max={0.15}
          step={0.005}
          unit="rad"
          onChange={(v) => setParams((s) => ({ ...s, sigmaPhi: v }))}
        />
        <Slider
          label="Gate confidence"
          role="prediction"
          value={params.confidence}
          min={0.5}
          max={0.995}
          step={0.005}
          format={(x) => `${(x * 100).toFixed(1)}%`}
          onChange={(v) => setParams((s) => ({ ...s, confidence: v }))}
          help="Tighter gates reject more outliers — and more true matches, which is how a filter starves."
        />
        <Toggle
          label="Known correspondence (safety rails)"
          role="posterior"
          checked={params.knownCorrespondence}
          onChange={(v) => setParams((s) => ({ ...s, knownCorrespondence: v }))}
        />
        <ButtonRow>
          {(Object.keys(PRESETS) as PresetId[]).map((id) => (
            <ActionButton key={id} onClick={() => setPreset(id)} emphasis={params.preset === id}>
              {PRESETS[id].label}
            </ActionButton>
          ))}
        </ButtonRow>
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
