'use client';

import { useCallback, useMemo, useState } from 'react';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { ControlPanel, Slider, Toggle, Transport } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';
import { Rng } from '@/lib/prob/rng';
import type { Pose2 } from '@/lib/geom/se2';
import type { Landmark } from '@/lib/sim/world';
import {
  landmarkModelKnownCorrespondence,
  landmarkObservation,
  type LandmarkSigmas,
} from '@/lib/models/sensor';
import { sampleLandmarkPoses } from '@/lib/models/landmark-sampling';
import { lowVarianceResample, type Particle } from '@/lib/filters/pf';
import { clear, drawRobot, label, sl, sx, sy, type Palette, type Viewport } from '@/lib/sim/draw';

/**
 * w10.5 — the Landmark Donut.
 *
 * Table 6.5 inverted: one range–bearing reading of a known landmark, five
 * hundred poses drawn from it. Two constraints in a three-dimensional pose
 * space cannot pin a robot down, and the picture says so — the samples form a
 * ring, with the heading spiralling around it so that every sample is looking
 * straight at the landmark.
 *
 * Add a second landmark and the ring becomes two blobs: the reading from
 * landmark 1 proposes, the reading from landmark 2 disposes, which is exactly
 * the importance-sampling move that Chapter 12 turns into a mixture proposal
 * for a kidnapped robot.
 */

const N = 500;
/**
 * Two landmarks placed symmetrically about the true pose, so the two-landmark
 * mode produces the textbook picture: two rings, two intersections, one of
 * which is the robot and one of which is its mirror image. Ambiguity that
 * geometry alone cannot resolve is the point.
 */
const A: Landmark = { x: 4.0, y: 2.4, id: 0 };
const B: Landmark = { x: 8.0, y: 2.4, id: 1 };
const TRUTH: Pose2 = { x: 6.0, y: 1.0, theta: 2.1 };

interface State {
  rng: Rng;
  poses: Pose2[];
  weights: number[];
}

export function LandmarkDonut() {
  const [sigmaR, setSigmaR] = useState(0.25);
  const [sigmaPhi, setSigmaPhi] = useState(0.12);
  const [twoLandmarks, setTwoLandmarks] = useState(false);

  const sigmas: LandmarkSigmas = useMemo(() => ({ r: sigmaR, phi: sigmaPhi }), [sigmaR, sigmaPhi]);
  // The two readings the robot actually got, noise-free for reproducibility:
  // the widget is about the *inverse*, not about detection noise.
  const fA = useMemo(() => landmarkObservation(A, TRUTH), []);
  const fB = useMemo(() => landmarkObservation(B, TRUTH), []);

  const init = useCallback(
    (seed: number): State => ({ rng: new Rng(seed), poses: [], weights: [] }),
    [],
  );

  const step = useCallback(
    (s: State): State => {
      // Fresh draws every tick: the donut shimmers, which makes it obvious that
      // it is a *distribution* and not a drawn shape.
      let poses = sampleLandmarkPoses(fA, A, sigmas, s.rng, N);
      let weights = new Array<number>(N).fill(1 / N);

      if (twoLandmarks) {
        // Importance sampling: propose from landmark A's inverse model, weight
        // by how well landmark B's reading is explained, resample.
        const w = poses.map((x) => landmarkModelKnownCorrespondence(fB, B, x, sigmas));
        const particles: Particle[] = poses.map((state, i) => ({ state, weight: w[i] }));
        const total = w.reduce((a, b) => a + b, 0);
        if (total > 0) {
          const resampled = lowVarianceResample(particles, s.rng);
          poses = resampled.map((p) => p.state);
          weights = resampled.map((p) => p.weight);
        }
      }
      return { rng: s.rng, poses, weights };
    },
    [fA, fB, sigmas, twoLandmarks],
  );

  const sim = useSimulation<State>({ init, step, fps: 3, initialSeed: 6 });

  const spread = useMemo(() => {
    const poses = sim.state.poses;
    if (poses.length === 0) return { x: 0, y: 0 };
    const mx = poses.reduce((a, p) => a + p.x, 0) / poses.length;
    const my = poses.reduce((a, p) => a + p.y, 0) / poses.length;
    return {
      x: Math.sqrt(poses.reduce((a, p) => a + (p.x - mx) ** 2, 0) / poses.length),
      y: Math.sqrt(poses.reduce((a, p) => a + (p.y - my) ** 2, 0) / poses.length),
    };
  }, [sim.state.poses]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, v: Viewport, pal: Palette) => {
      clear(ctx, v, pal);

      // Range rings: the geometry the sampler is inverting.
      const ring = (lm: Landmark, r: number) => {
        ctx.strokeStyle = pal.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(sx(v, lm.x), sy(v, lm.y), sl(v, r), 0, Math.PI * 2);
        ctx.stroke();
      };
      ring(A, fA.r);
      if (twoLandmarks) ring(B, fB.r);

      // The sample cloud, with a heading tick each: the third coordinate the
      // ring picture hides.
      for (const p of sim.state.poses) {
        const px = sx(v, p.x);
        const py = sy(v, p.y);
        ctx.strokeStyle = pal.measurement;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + 7 * Math.cos(p.theta), py - 7 * Math.sin(p.theta));
        ctx.stroke();
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = pal.measurement;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Landmarks.
      const mark = (lm: Landmark, name: string, on: boolean) => {
        ctx.fillStyle = on ? pal.accent : pal.unknown;
        ctx.beginPath();
        ctx.arc(sx(v, lm.x), sy(v, lm.y), 5, 0, Math.PI * 2);
        ctx.fill();
        label(ctx, name, sx(v, lm.x) + 8, sy(v, lm.y) - 2, on ? pal.accent : pal.unknown, {
          size: 11,
          weight: 600,
        });
      };
      mark(A, 'm₁', true);
      mark(B, 'm₂', twoLandmarks);

      // The pose the robot is actually in — which no sample is allowed to know.
      drawRobot(ctx, v, TRUTH, pal.truth, 0.28);
      label(ctx, 'true pose', sx(v, TRUTH.x) + 12, sy(v, TRUTH.y) + 12, pal.truth, { size: 10 });
    },
    [sim.state.poses, fA.r, fB.r, twoLandmarks],
  );

  return (
    <WidgetFrame
      id="w10.5"
      title="The Landmark Donut"
      teaches="One landmark observation never localizes a robot — two constraints cannot pin down three degrees of freedom, no matter how good the sensor is."
      colorKey={['measurement', 'truth']}
      caption={
        <>
          Five hundred poses drawn by <code>sample_landmark_model_known_correspondence</code> (Table
          6.5) from a single range–bearing reading of m₁. Each tick is a heading: every sample faces
          the landmark, because that is what the bearing φ pins down. Widening <strong>σ_r</strong>{' '}
          fattens the ring radially; widening σ_φ smears the headings and leaves the ring where it
          is — a different degree of freedom each time. Switch on <strong>m₂</strong> and the ring
          collapses to a blob about 20 cm across: four constraints on three unknowns, so the pose is
          now over-determined and what is left is noise, not ambiguity. The two-intersections
          picture you may have been expecting is what you get from two <em>ranges</em> without
          bearings — crank σ_φ to its maximum and a faint second cluster starts to reappear at the
          mirror position.
        </>
      }
    >
      <SimCanvas
        world={{ minX: 1.2, maxX: 10.8, minY: -0.5, maxY: 5.3 }}
        draw={draw}
        deps={[sim.tick, sim.state, twoLandmarks]}
        aspect={1.8}
        padding={0.1}
        ariaLabel="Five hundred sampled robot poses forming a ring around a landmark; with a second landmark enabled the ring collapses into two clusters where the two rings intersect."
      />

      <div className="grid grid-cols-3 divide-x divide-fd-border border-t border-fd-border text-center">
        <Stat label="samples" value={String(sim.state.poses.length)} />
        <Stat label="spread in x" value={`${spread.x.toFixed(2)} m`} />
        <Stat label="spread in y" value={`${spread.y.toFixed(2)} m`} />
      </div>

      <ControlPanel columns={3}>
        <Slider
          label="σ_r"
          role="measurement"
          value={sigmaR}
          min={0.02}
          max={0.8}
          step={0.02}
          unit="m"
          onChange={setSigmaR}
          help="Range noise. Thickens the ring radially and nothing else."
        />
        <Slider
          label="σ_φ"
          value={sigmaPhi}
          min={0.01}
          max={0.6}
          step={0.01}
          unit="rad"
          onChange={setSigmaPhi}
          help="Bearing noise. Smears the headings; the ring's position is untouched."
        />
        <Toggle label="Observe m₂ as well" checked={twoLandmarks} onChange={setTwoLandmarks} />
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

function Stat({ label: l, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{l}</div>
      <div className="font-mono text-sm tabular-nums">{value}</div>
    </div>
  );
}
