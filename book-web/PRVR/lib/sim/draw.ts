/**
 * Canvas drawing primitives, in world coordinates.
 *
 * Every simulation shares these so a robot, a covariance ellipse, or a particle
 * cloud looks identical in every chapter. Colors always come from the resolved
 * CSS custom properties, so figures follow the reader's theme and the book's
 * color code stays consistent between prose, equations, and pixels.
 */

import type { Pose2 } from '../geom/se2';
import type { Segment, World } from './world';

export interface Viewport {
  /** World-space rectangle mapped onto the canvas. */
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
}

export interface Palette {
  prior: string;
  prediction: string;
  measurement: string;
  posterior: string;
  truth: string;
  grid: string;
  ink: string;
  wall: string;
  bg: string;
  free: string;
  occupied: string;
  unknown: string;
  accent: string;
}

const FALLBACK: Palette = {
  prior: '#3b82f6',
  prediction: '#ea580c',
  measurement: '#16a34a',
  posterior: '#9333ea',
  truth: '#6b7280',
  grid: 'rgba(15,23,42,0.07)',
  ink: '#0f172a',
  wall: '#334155',
  bg: '#fcfcfd',
  free: '#ffffff',
  occupied: '#0f172a',
  unknown: '#cbd5e1',
  accent: '#0d9488',
};

/** Read the book palette from CSS custom properties (theme-aware). */
export function readPalette(el?: Element | null): Palette {
  if (typeof window === 'undefined') return FALLBACK;
  const cs = getComputedStyle(el ?? document.documentElement);
  const get = (name: string, fallback: string) => {
    const v = cs.getPropertyValue(name).trim();
    return v.length > 0 ? v : fallback;
  };
  return {
    prior: get('--pr-prior', FALLBACK.prior),
    prediction: get('--pr-prediction', FALLBACK.prediction),
    measurement: get('--pr-measurement', FALLBACK.measurement),
    posterior: get('--pr-posterior', FALLBACK.posterior),
    truth: get('--pr-truth', FALLBACK.truth),
    grid: get('--pr-grid', FALLBACK.grid),
    ink: get('--pr-canvas-ink', FALLBACK.ink),
    wall: get('--pr-wall', FALLBACK.wall),
    bg: get('--pr-canvas-bg', FALLBACK.bg),
    free: get('--pr-free', FALLBACK.free),
    occupied: get('--pr-occupied', FALLBACK.occupied),
    unknown: get('--pr-unknown', FALLBACK.unknown),
    accent: get('--color-fd-primary', FALLBACK.accent),
  };
}

/** Parse `#rgb`, `#rrggbb`, `rgb()` or `rgba()` into components. */
function parseColor(c: string): [number, number, number] | null {
  const s = c.trim();
  if (s.startsWith('#')) {
    const hex = s.slice(1);
    if (hex.length === 3) {
      return [
        parseInt(hex[0] + hex[0], 16),
        parseInt(hex[1] + hex[1], 16),
        parseInt(hex[2] + hex[2], 16),
      ];
    }
    if (hex.length >= 6) {
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16),
      ];
    }
    return null;
  }
  const m = s.match(/rgba?\(([^)]+)\)/);
  if (!m) return null;
  const parts = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
  if (parts.length < 3 || parts.some(Number.isNaN)) return null;
  return [parts[0], parts[1], parts[2]];
}

const fade = (c: string, alpha: number) => {
  const rgb = parseColor(c);
  return rgb ? `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})` : c;
};

/**
 * Mute every estimation role except the one the reader is pointing at.
 *
 * This is what makes the book's colour convention operable: because every
 * widget already draws its prior in `p.prior` and its measurement in
 * `p.measurement`, fading the palette itself makes *all* of them respond to a
 * hovered equation term without a single widget knowing that the feature
 * exists. Chrome colours are left alone — the map and the grid should not
 * flicker when the reader points at an equation.
 */
export function mutePalette(p: Palette, hovered: RoleName | null): Palette {
  if (!hovered) return p;
  const roles: RoleName[] = ['prior', 'prediction', 'measurement', 'posterior', 'truth'];
  const out = { ...p };
  for (const role of roles) {
    if (role !== hovered) out[role] = fade(p[role], 0.1);
  }
  return out;
}

export type RoleName = 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';

/** Build a viewport that fits a world rectangle into a canvas, preserving aspect. */
export function fitViewport(
  world: { minX: number; minY: number; maxX: number; maxY: number },
  width: number,
  height: number,
  padding = 0.4,
): Viewport {
  const wx = world.maxX - world.minX + 2 * padding;
  const wy = world.maxY - world.minY + 2 * padding;
  const scale = Math.min(width / wx, height / wy);
  const cx = (world.minX + world.maxX) / 2;
  const cy = (world.minY + world.maxY) / 2;
  const halfW = width / (2 * scale);
  const halfH = height / (2 * scale);
  return {
    minX: cx - halfW,
    maxX: cx + halfW,
    minY: cy - halfH,
    maxY: cy + halfH,
    width,
    height,
  };
}

/** World → canvas pixel. Y is flipped so +y points up, as in the equations. */
export const sx = (v: Viewport, x: number) =>
  ((x - v.minX) / (v.maxX - v.minX)) * v.width;
export const sy = (v: Viewport, y: number) =>
  v.height - ((y - v.minY) / (v.maxY - v.minY)) * v.height;
/** Scalar length in world units → pixels. */
export const sl = (v: Viewport, d: number) => (d / (v.maxX - v.minX)) * v.width;

/** Canvas pixel → world, for pointer interaction. */
export function toWorld(v: Viewport, px: number, py: number): [number, number] {
  return [
    v.minX + (px / v.width) * (v.maxX - v.minX),
    v.minY + ((v.height - py) / v.height) * (v.maxY - v.minY),
  ];
}

export function clear(ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) {
  ctx.clearRect(0, 0, v.width, v.height);
  ctx.fillStyle = p.bg;
  ctx.fillRect(0, 0, v.width, v.height);
}

export function drawGrid(ctx: CanvasRenderingContext2D, v: Viewport, p: Palette, spacing = 1) {
  ctx.save();
  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const x0 = Math.ceil(v.minX / spacing) * spacing;
  for (let x = x0; x <= v.maxX; x += spacing) {
    ctx.moveTo(sx(v, x), 0);
    ctx.lineTo(sx(v, x), v.height);
  }
  const y0 = Math.ceil(v.minY / spacing) * spacing;
  for (let y = y0; y <= v.maxY; y += spacing) {
    ctx.moveTo(0, sy(v, y));
    ctx.lineTo(v.width, sy(v, y));
  }
  ctx.stroke();
  ctx.restore();
}

export function drawSegments(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  segs: Segment[],
  color: string,
  lineWidth = 2.5,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (const s of segs) {
    ctx.moveTo(sx(v, s.x1), sy(v, s.y1));
    ctx.lineTo(sx(v, s.x2), sy(v, s.y2));
  }
  ctx.stroke();
  ctx.restore();
}

export function drawWorld(ctx: CanvasRenderingContext2D, v: Viewport, world: World, p: Palette) {
  drawSegments(ctx, v, world.walls, p.wall, 2.5);
  if (world.landmarks) {
    ctx.save();
    for (const lm of world.landmarks) {
      ctx.fillStyle = p.accent;
      ctx.beginPath();
      ctx.arc(sx(v, lm.x), sy(v, lm.y), 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}

/** The robot: a triangle pointing along its heading, with an optional halo. */
export function drawRobot(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  pose: Pose2,
  color: string,
  size = 0.22,
  opts: { filled?: boolean; alpha?: number } = {},
) {
  const { filled = true, alpha = 1 } = opts;
  const r = sl(v, size);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(sx(v, pose.x), sy(v, pose.y));
  ctx.rotate(-pose.theta);
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.lineTo(-r * 0.72, r * 0.66);
  ctx.lineTo(-r * 0.38, 0);
  ctx.lineTo(-r * 0.72, -r * 0.66);
  ctx.closePath();
  if (filled) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }
  ctx.restore();
}

/** A LiDAR scan drawn as rays from the pose, with endpoint dots. */
export function drawScan(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  pose: Pose2,
  ranges: number[],
  angles: number[],
  color: string,
  maxRange: number,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.22;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < ranges.length; i++) {
    const a = pose.theta + angles[i];
    const r = Math.min(ranges[i], maxRange);
    ctx.moveTo(sx(v, pose.x), sy(v, pose.y));
    ctx.lineTo(sx(v, pose.x + r * Math.cos(a)), sy(v, pose.y + r * Math.sin(a)));
  }
  ctx.stroke();

  ctx.globalAlpha = 0.9;
  ctx.fillStyle = color;
  for (let i = 0; i < ranges.length; i++) {
    if (ranges[i] >= maxRange) continue;
    const a = pose.theta + angles[i];
    ctx.beginPath();
    ctx.arc(
      sx(v, pose.x + ranges[i] * Math.cos(a)),
      sy(v, pose.y + ranges[i] * Math.sin(a)),
      1.6,
      0,
      Math.PI * 2,
    );
    ctx.fill();
  }
  ctx.restore();
}

/** A particle cloud. Weight controls both alpha and radius. */
export function drawParticles(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  particles: { state: Pose2; weight: number }[],
  color: string,
  opts: { showHeading?: boolean; maxRadius?: number } = {},
) {
  const { showHeading = true, maxRadius = 2.6 } = opts;
  if (particles.length === 0) return;
  const wMax = particles.reduce((m, p) => Math.max(m, p.weight), 0) || 1;
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  for (const p of particles) {
    const w = p.weight / wMax;
    const px = sx(v, p.state.x);
    const py = sy(v, p.state.y);
    ctx.globalAlpha = 0.18 + 0.72 * w;
    ctx.beginPath();
    ctx.arc(px, py, 1 + maxRadius * Math.sqrt(w), 0, Math.PI * 2);
    ctx.fill();
    if (showHeading) {
      const len = 6 + 6 * w;
      ctx.globalAlpha = 0.12 + 0.4 * w;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + len * Math.cos(p.state.theta), py - len * Math.sin(p.state.theta));
      ctx.stroke();
    }
  }
  ctx.restore();
}

/**
 * A covariance ellipse: the iso-contour of a 2-D Gaussian at `nSigma`.
 * `cov` is the 2×2 position block; angle comes from its eigenvectors.
 */
export function drawCovariance(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  mean: [number, number],
  ellipse: { rx: number; ry: number; angle: number },
  color: string,
  opts: { fill?: boolean; lineWidth?: number; alpha?: number } = {},
) {
  const { fill = true, lineWidth = 1.75, alpha = 1 } = opts;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(sx(v, mean[0]), sy(v, mean[1]));
  ctx.rotate(-ellipse.angle);
  ctx.beginPath();
  ctx.ellipse(0, 0, Math.max(sl(v, ellipse.rx), 0.5), Math.max(sl(v, ellipse.ry), 0.5), 0, 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha * 0.14;
    ctx.fill();
    ctx.globalAlpha = alpha;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.restore();
}

/** A trajectory polyline. */
export function drawPath(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  path: { x: number; y: number }[],
  color: string,
  opts: { dashed?: boolean; lineWidth?: number; alpha?: number } = {},
) {
  if (path.length < 2) return;
  const { dashed = false, lineWidth = 1.75, alpha = 1 } = opts;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  if (dashed) ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(sx(v, path[0].x), sy(v, path[0].y));
  for (let i = 1; i < path.length; i++) ctx.lineTo(sx(v, path[i].x), sy(v, path[i].y));
  ctx.stroke();
  ctx.restore();
}

/** An occupancy grid rendered as grayscale: white free, dark occupied, gray unknown. */
export function drawOccupancyGrid(
  ctx: CanvasRenderingContext2D,
  v: Viewport,
  grid: {
    width: number;
    height: number;
    cellSize: number;
    origin: { x: number; y: number };
    getProbabilityArray: () => Float64Array;
  },
  p: Palette,
) {
  const probs = grid.getProbabilityArray();
  const cs = grid.cellSize;
  const w = Math.ceil(sl(v, cs)) + 1;
  ctx.save();
  for (let j = 0; j < grid.height; j++) {
    for (let i = 0; i < grid.width; i++) {
      const pr = probs[j * grid.width + i];
      // 0.5 means "no evidence yet" — draw it as the unknown gray, not mid-tone.
      if (Math.abs(pr - 0.5) < 0.02) continue;
      const shade = Math.round(255 * (1 - pr));
      ctx.fillStyle = `rgb(${shade},${shade},${shade})`;
      const x = grid.origin.x + i * cs;
      const y = grid.origin.y + j * cs;
      ctx.fillRect(sx(v, x), sy(v, y + cs), w, w);
    }
  }
  ctx.restore();
}

/** A small text label in canvas space (already-scaled pixels). */
export function label(
  ctx: CanvasRenderingContext2D,
  text: string,
  px: number,
  py: number,
  color: string,
  opts: { size?: number; align?: CanvasTextAlign; weight?: number } = {},
) {
  const { size = 11, align = 'left', weight = 500 } = opts;
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ui-monospace, monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, px, py);
  ctx.restore();
}
