'use client';

import type { Pose2 } from '@/lib/geom/se2';
import type { Mat } from '@/lib/prob/linalg';
import { CourseSim, type CourseOptions } from '@/lib/slam/course';

/**
 * Shared scaffolding for the Chapter 14 widgets: the panel transform every
 * canvas uses, the small stat readout, and the *covariance movie* — a recorded
 * run, stored frame by frame, so a scrubber can rewind to the moment a filter
 * started lying. It is the browser twin of `CovMovie` in the Rust crate.
 */

export interface Panel {
  x0: number;
  y0: number;
  w: number;
  h: number;
}

/** Uniform fit of a world rectangle into a sub-rectangle of canvas world space. */
export function fitPanel(
  panel: Panel,
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
) {
  const bw = bounds.maxX - bounds.minX;
  const bh = bounds.maxY - bounds.minY;
  const scale = Math.min(panel.w / bw, panel.h / bh);
  const ox = panel.x0 + (panel.w - scale * bw) / 2;
  const oy = panel.y0 + (panel.h - scale * bh) / 2;
  return {
    scale,
    toX: (x: number) => ox + (x - bounds.minX) * scale,
    toY: (y: number) => oy + (y - bounds.minY) * scale,
    fromX: (u: number) => (u - ox) / scale + bounds.minX,
    fromY: (u: number) => (u - oy) / scale + bounds.minY,
  };
}

export function Stat({
  label,
  value,
  alarm,
}: {
  label: string;
  value: string;
  alarm?: boolean;
}) {
  return (
    <div className="px-2 py-1.5">
      <div className="eyebrow">{label}</div>
      <div
        className="font-mono text-sm tabular-nums"
        style={alarm ? { color: 'var(--pr-prediction)' } : undefined}
      >
        {value}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* The covariance movie                                                        */
/* -------------------------------------------------------------------------- */

export interface MovieFrame {
  truth: Pose2;
  est: Pose2;
  poseCov: Mat;
  landmarks: { x: number; y: number; cov: Mat; label: number }[];
  truthPath: { x: number; y: number }[];
  estPath: { x: number; y: number }[];
  nees: number;
  error: number;
  headingError: number;
  headingSigma: number;
  closure: boolean;
}

export interface Movie {
  frames: MovieFrame[];
  /** Ground-truth beacon positions, for the map RMSE readout. */
  mapRmse: number[];
}

/** Run the course once and keep every frame. Cheap: ~50 numbers per step. */
export function captureRun(steps: number, opts: CourseOptions): Movie {
  const sim = new CourseSim(opts);
  const frames: MovieFrame[] = [];
  const mapRmse: number[] = [];

  for (let i = 0; i < steps; i++) {
    const r = sim.step();
    const f = sim.filter;
    const landmarks = Array.from({ length: f.count }, (_, j) => {
      const [x, y] = f.landmarkMean(j);
      return { x, y, cov: f.landmarkCov(j), label: f.labels[j] };
    });
    let sq = 0;
    let k = 0;
    for (let j = 0; j < f.count; j++) {
      const b = sim.truthFor(j);
      if (!b) continue;
      sq += (landmarks[j].x - b.x) ** 2 + (landmarks[j].y - b.y) ** 2;
      k += 1;
    }
    mapRmse.push(k > 0 ? Math.sqrt(sq / k) : 0);
    // Sub-sample the paths: at 460 frames the full history would dominate the
    // recording, and a polyline every fourth step draws identically.
    frames.push({
      truth: { ...sim.truth },
      est: f.pose(),
      poseCov: f.poseCov(),
      landmarks,
      truthPath: sim.truthPath.filter((_, n) => n % 3 === 0),
      estPath: sim.estimatePath.filter((_, n) => n % 3 === 0),
      nees: r.nees,
      error: r.positionError,
      headingError: r.headingError,
      headingSigma: Math.sqrt(Math.max(f.poseCov()[2][2], 0)),
      closure: r.closures.length > 0,
    });
  }
  return { frames, mapRmse };
}
