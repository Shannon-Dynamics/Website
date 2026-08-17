'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { StatTile } from '@/components/viz';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import type { Particle } from '@/lib/filters/pf';
import { gradientTransmission, softResample } from '@/lib/filters/soft-resample';

/**
 * w25.4 — the Resampling Gradient Microscope.
 *
 * A particle filter unrolled into the computation graph it really is, with the
 * backward pass drawn as pulses travelling right to left. Classical resampling
 * is piecewise constant in the weights, so every pulse dies at the first
 * resample node it meets and the measurement model never learns anything. Soft
 * resampling opens the gate by exactly 1 − λ.
 *
 * Every number is computed, not drawn: the transmission coefficients come from
 * `gradientTransmission`, and the ESS meter from an actual `softResample` of a
 * genuinely degenerate twelve-particle set.
 */

const M = 12;
/** Nine nodes: predict/weight/resample ×3, with the loss replacing the last resample. */
const NODES = [
  { key: 'p1', kind: 'predict', label: 'predict', sub: 't=1' },
  { key: 'w1', kind: 'weight', label: 'weight', sub: 'p_θ(z₁|x)' },
  { key: 'r1', kind: 'resample', label: 'resample', sub: 'λ' },
  { key: 'p2', kind: 'predict', label: 'predict', sub: 't=2' },
  { key: 'w2', kind: 'weight', label: 'weight', sub: 'p_θ(z₂|x)' },
  { key: 'r2', kind: 'resample', label: 'resample', sub: 'λ' },
  { key: 'p3', kind: 'predict', label: 'predict', sub: 't=3' },
  { key: 'w3', kind: 'weight', label: 'weight', sub: 'p_θ(z₃|x)' },
  { key: 'loss', kind: 'loss', label: 'loss', sub: 'L(θ)' },
] as const;

const NODE_X = NODES.map((_, i) => 6 + i * 11);
const NODE_Y = 15;
const THETA_Y = 37;
const THETA_X = 50;
const HALF_W = 5;
const HALF_H = 4.2;

/** A deliberately degenerate weight vector: one particle holds most of the mass. */
function degenerateWeights(seed: number): number[] {
  const rng = new Rng(seed);
  const raw = Array.from({ length: M }, () => Math.exp(rng.normal(0, 1.5)));
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((w) => w / total);
}

const WEIGHTS = degenerateWeights(2504);

function ess(weights: readonly number[]): number {
  let s1 = 0;
  let s2 = 0;
  for (const w of weights) {
    s1 += w;
    s2 += w * w;
  }
  return s2 > 0 ? (s1 * s1) / s2 : 0;
}

interface State {
  phase: number;
}

export function ResamplingGradientMicroscope() {
  const [lambda, setLambda] = useState(0.4);
  const [hard, setHard] = useState(false);

  const init = useCallback((): State => ({ phase: 0 }), []);
  const step = useCallback(
    (s: State): State => ({ phase: (s.phase + 0.045) % 1 }),
    [],
  );
  const sim = useSimulation<State>({ init, step, fps: 30, initialSeed: 4 });

  /** λ = 1 *is* the classical resampler, so "hard" is not a different algorithm. */
  const effLambda = hard ? 1 : lambda;
  const tau = 1 - effLambda;

  /** Cumulative gradient transmission arriving at each node, walking back from L. */
  const arriving = useMemo(() => {
    const out: number[] = new Array(NODES.length).fill(1);
    let acc = 1;
    for (let i = NODES.length - 1; i >= 0; i--) {
      out[i] = acc;
      if (NODES[i].kind === 'resample') acc *= tau;
    }
    return out;
  }, [tau]);

  const resampled = useMemo(() => {
    const particles: Particle[] = WEIGHTS.map((w, i) => ({
      state: { x: i, y: 0, theta: 0 },
      weight: w,
    }));
    const res = softResample(particles, effLambda, new Rng(77));
    return {
      essAfter: ess(res.particles.map((p) => p.weight)),
      weights: res.particles.map((p) => p.weight),
      transmission: gradientTransmission(WEIGHTS, effLambda),
    };
  }, [effLambda]);

  const essBefore = ess(WEIGHTS);
  const reachingTheta = arriving[1]; // the earliest weight node

  const phase = sim.state.phase;

  return (
    <WidgetFrame
      id="w25.4"
      title="The Resampling Gradient Microscope"
      teaches="You cannot differentiate through a categorical draw — but you can move the weight dependence into the importance weight, and λ buys exactly 1 − λ of the gradient back."
      colorKey={['measurement', 'posterior', 'truth']}
      caption={
        <>
          The filter, unrolled. Pulses run right to left: that is the backward pass, carrying
          ∂L/∂θ from the loss towards the measurement model at the bottom. With{' '}
          <strong>classical resampling</strong> the pulses stop dead at every purple resample node,
          because multinomial resampling is piecewise constant in the weights — nudge a weight, and
          almost surely the same particles survive carrying the same 1/M. Only the last timestep
          ever reaches θ, so training a three-step filter trains one step of it. Open λ and the
          gates open with it: each resample node passes a fraction 1&nbsp;−&nbsp;λ, so the earliest
          weight node receives (1&nbsp;−&nbsp;λ)². The meters on the right are the bill: at
          λ&nbsp;=&nbsp;0 the gradient is perfect and the resampler does nothing at all — ESS comes
          out exactly as degenerate as it went in — while at λ&nbsp;=&nbsp;1 the population is
          restored and the gradient is gone.
        </>
      }
    >
      <div className="p-3">
        <svg
          viewBox="0 0 100 46"
          className="w-full"
          role="img"
          aria-label="A particle filter unrolled as a computation graph: predict, weight and resample nodes in sequence, ending at a loss node, with the measurement model parameters below. Animated pulses show the backward gradient, which stops at resample nodes when lambda is one."
        >
          {/* ---- the forward chain ------------------------------------- */}
          {NODES.slice(0, -1).map((n, i) => {
            const live = arriving[i] > 1e-3;
            return (
              <g key={`edge-${n.key}`}>
                <line
                  x1={NODE_X[i] + HALF_W}
                  y1={NODE_Y}
                  x2={NODE_X[i + 1] - HALF_W}
                  y2={NODE_Y}
                  stroke={live ? 'var(--pr-posterior)' : 'var(--pr-grid)'}
                  strokeWidth={live ? 0.55 : 0.4}
                  strokeDasharray={live ? undefined : '1 1'}
                  opacity={live ? 0.35 + 0.65 * arriving[i] : 1}
                />
                {live ? (
                  <circle
                    // Pulses travel backwards: from node i+1 towards node i.
                    cx={
                      NODE_X[i + 1] -
                      HALF_W -
                      phase * (NODE_X[i + 1] - NODE_X[i] - 2 * HALF_W)
                    }
                    cy={NODE_Y}
                    r={0.75 + 0.5 * arriving[i]}
                    fill="var(--pr-posterior)"
                    opacity={0.25 + 0.75 * arriving[i]}
                  />
                ) : null}
              </g>
            );
          })}

          {/* ---- θ, and the edges up into each weight node --------------- */}
          {[1, 4, 7].map((i) => {
            const live = arriving[i] > 1e-3;
            const x = NODE_X[i];
            const path = `M ${THETA_X} ${THETA_Y - HALF_H} C ${THETA_X} ${THETA_Y - 10}, ${x} ${NODE_Y + 12}, ${x} ${NODE_Y + HALF_H}`;
            return (
              <g key={`theta-${i}`}>
                <path
                  d={path}
                  fill="none"
                  stroke={live ? 'var(--pr-measurement)' : 'var(--pr-grid)'}
                  strokeWidth={live ? 0.5 : 0.35}
                  strokeDasharray={live ? undefined : '1 1'}
                  opacity={live ? 0.3 + 0.7 * arriving[i] : 1}
                />
                {live ? (
                  <circle r={0.7 + 0.5 * arriving[i]} fill="var(--pr-measurement)" opacity={0.3 + 0.7 * arriving[i]}>
                    <animateMotion dur="1.6s" repeatCount="indefinite" path={path} keyPoints="1;0" keyTimes="0;1" calcMode="linear" />
                  </circle>
                ) : null}
              </g>
            );
          })}

          <g>
            <rect
              x={THETA_X - 11}
              y={THETA_Y - HALF_H}
              width={22}
              height={2 * HALF_H}
              rx={1.2}
              fill="var(--pr-canvas-bg)"
              stroke="var(--pr-measurement)"
              strokeWidth={0.6}
            />
            <text
              x={THETA_X}
              y={THETA_Y - 0.6}
              textAnchor="middle"
              style={{ fontSize: 2.9, fontWeight: 600 }}
              fill="var(--pr-measurement)"
            >
              θ — measurement model
            </text>
            <text
              x={THETA_X}
              y={THETA_Y + 2.7}
              textAnchor="middle"
              style={{ fontSize: 2.4 }}
              fill="var(--pr-truth)"
            >
              gradient arriving: {(100 * reachingTheta).toFixed(0)}% from t=1
            </text>
          </g>

          {/* ---- the nodes ---------------------------------------------- */}
          {NODES.map((n, i) => {
            const color =
              n.kind === 'resample'
                ? 'var(--pr-posterior)'
                : n.kind === 'weight'
                  ? 'var(--pr-measurement)'
                  : n.kind === 'loss'
                    ? 'var(--color-fd-primary)'
                    : 'var(--pr-prediction)';
            const dead = n.kind === 'resample' && tau < 1e-3;
            return (
              <g key={n.key}>
                <rect
                  x={NODE_X[i] - HALF_W}
                  y={NODE_Y - HALF_H}
                  width={2 * HALF_W}
                  height={2 * HALF_H}
                  rx={1.2}
                  fill={dead ? 'var(--pr-posterior)' : 'var(--pr-canvas-bg)'}
                  stroke={color}
                  strokeWidth={0.6}
                  opacity={dead ? 0.9 : 1}
                />
                <text
                  x={NODE_X[i]}
                  y={NODE_Y - 0.7}
                  textAnchor="middle"
                  style={{ fontSize: 2.5, fontWeight: 600 }}
                  fill={dead ? 'var(--pr-canvas-bg)' : color}
                >
                  {n.label}
                </text>
                <text
                  x={NODE_X[i]}
                  y={NODE_Y + 2.6}
                  textAnchor="middle"
                  style={{ fontSize: 2.1 }}
                  fill={dead ? 'var(--pr-canvas-bg)' : 'var(--pr-truth)'}
                >
                  {n.sub}
                </text>
                {n.kind === 'resample' ? (
                  <text
                    x={NODE_X[i]}
                    y={NODE_Y - HALF_H - 1.4}
                    textAnchor="middle"
                    style={{ fontSize: 2.2, fontWeight: 600 }}
                    fill={dead ? 'var(--pr-prediction)' : 'var(--pr-posterior)'}
                  >
                    {dead ? '✕ gradient dies' : `× ${tau.toFixed(2)}`}
                  </text>
                ) : null}
              </g>
            );
          })}

          {/* ---- the surviving weights, as a strip under the graph ------- */}
          <text x={2} y={45.4} style={{ fontSize: 2.2 }} fill="var(--pr-truth)">
            w′ after resampling
          </text>
          {resampled.weights.map((w, i) => {
            const wMax = Math.max(...resampled.weights, 1e-9);
            const h = 3.4 * (w / wMax);
            return (
              <rect
                key={`w-${i}`}
                x={26 + i * 5.6}
                y={45.6 - h}
                width={4.4}
                height={h}
                fill="var(--pr-posterior)"
                opacity={0.75}
              />
            );
          })}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-fd-border p-3 lg:grid-cols-3">
        <StatTile
          label="gradient through one resample"
          value={resampled.transmission}
          precision={3}
          role="measurement"
        />
        <StatTile
          label="reaching θ from t = 1"
          value={reachingTheta}
          precision={3}
          role={reachingTheta < 0.02 ? 'prediction' : 'measurement'}
        />
        <StatTile
          label="ESS after resampling"
          value={resampled.essAfter}
          unit={`of ${M} (in: ${essBefore.toFixed(1)})`}
          precision={1}
          role="posterior"
        />
      </div>

      <ControlPanel columns={2}>
        <Slider
          label="soft-resampling mixture λ"
          role="posterior"
          value={lambda}
          min={0}
          max={1}
          step={0.02}
          onChange={(v) => {
            setLambda(v);
            setHard(false);
          }}
          help="q(i) = λ w_i + (1 − λ)/M. λ = 1 is the classical resampler; λ = 0 is plain importance sampling."
        />
        <Toggle
          label="classical (hard) resampling"
          checked={hard}
          onChange={setHard}
        />
      </ControlPanel>

      <Transport playing={sim.playing} onToggle={sim.toggle} onReset={sim.reset} tick={sim.tick} />
    </WidgetFrame>
  );
}
