/**
 * Visual language for the whole book.
 *
 * The palette is the validated reference instance of the dataviz method: eight
 * categorical slots in a FIXED order (assigned in sequence, never cycled), a
 * one-hue sequential ramp for magnitude, and a blue↔red diverging pair with a
 * neutral gray midpoint for polarity. Both modes pass the lightness band,
 * chroma floor, CVD separation, normal-vision floor and contrast checks.
 *
 * Semantic mapping used consistently across every chapter (CLAUDE.md §5):
 *   value function   → sequential ramp
 *   policy           → directional arrows, slot-1 hue
 *   reward / δ / A   → diverging (negative = blue, positive = red)
 *   uncertainty      → fan/band, series hue at low alpha
 */

export type Mode = 'light' | 'dark';

export const CATEGORICAL: Record<Mode, string[]> = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};

/**
 * Series cap for all-pairs chart forms (scatter, bubble, small multiples):
 * only the first three slots validate under `--pairs all`. Past three, fold to
 * "Other" or facet — never extend the palette.
 */
export const ALL_PAIRS_SERIES_CAP = 3;

export const SEQUENTIAL: Record<Mode, string[]> = {
  light: ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
  // Dark flips the anchor: the step nearest the surface is the dark one.
  dark: ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
};

export const DIVERGING: Record<Mode, { neg: string; mid: string; pos: string }> = {
  light: { neg: '#2a78d6', mid: '#f0efec', pos: '#e34948' },
  dark: { neg: '#3987e5', mid: '#383835', pos: '#e66767' },
};

/** Status is fixed — never themed, always shipped with an icon + label. */
export const STATUS = {
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  critical: '#d03b3b',
} as const;

export const CHROME: Record<
  Mode,
  {
    surface: string;
    page: string;
    textPrimary: string;
    textSecondary: string;
    textMuted: string;
    gridline: string;
    baseline: string;
  }
> = {
  light: {
    surface: '#fcfcfb',
    page: '#f9f9f7',
    textPrimary: '#0b0b0b',
    textSecondary: '#52514e',
    textMuted: '#898781',
    gridline: '#e1e0d9',
    baseline: '#c3c2b7',
  },
  dark: {
    surface: '#1a1a19',
    page: '#0d0d0d',
    textPrimary: '#ffffff',
    textSecondary: '#c3c2b7',
    textMuted: '#898781',
    gridline: '#2c2c2a',
    baseline: '#383835',
  },
};

/** Linear interpolation between two hex colors in sRGB. */
function lerpHex(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const p = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `#${p.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Sequential colormap for magnitude — `t` in [0,1] maps light→dark (light mode).
 * Use for value functions, visit counts, Q-heatmaps.
 */
export function sequentialColor(t: number, mode: Mode = 'light'): string {
  const ramp = SEQUENTIAL[mode];
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * (ramp.length - 1);
  const i = Math.floor(x);
  return i >= ramp.length - 1 ? ramp[ramp.length - 1] : lerpHex(ramp[i], ramp[i + 1], x - i);
}

/**
 * Diverging colormap for polarity — `t` in [-1,1]; 0 is the neutral midpoint.
 * Use for TD error δ, advantage A, signed reward.
 */
export function divergingColor(t: number, mode: Mode = 'light'): string {
  const { neg, mid, pos } = DIVERGING[mode];
  const x = Math.max(-1, Math.min(1, Number.isFinite(t) ? t : 0));
  return x < 0 ? lerpHex(mid, neg, -x) : lerpHex(mid, pos, x);
}

/** Categorical slot by index — assigned in sequence, never cycled past 8. */
export function seriesColor(index: number, mode: Mode = 'light'): string {
  const slots = CATEGORICAL[mode];
  return slots[Math.min(index, slots.length - 1)];
}

/** Nivo theme object, driven by the same tokens as the CSS. */
export function nivoTheme(mode: Mode) {
  const c = CHROME[mode];
  return {
    background: 'transparent',
    text: { fontSize: 11, fill: c.textSecondary, fontFamily: 'var(--font-sans)' },
    axis: {
      domain: { line: { stroke: c.baseline, strokeWidth: 1 } },
      legend: { text: { fontSize: 11, fill: c.textSecondary, fontWeight: 500 } },
      ticks: {
        line: { stroke: c.baseline, strokeWidth: 1 },
        text: { fontSize: 10, fill: c.textMuted },
      },
    },
    grid: { line: { stroke: c.gridline, strokeWidth: 1 } },
    legends: {
      text: { fontSize: 11, fill: c.textSecondary },
      ticks: { text: { fontSize: 10, fill: c.textMuted } },
    },
    tooltip: {
      container: {
        background: mode === 'light' ? '#ffffff' : '#232322',
        color: c.textPrimary,
        fontSize: 12,
        borderRadius: 8,
        boxShadow:
          mode === 'light' ? '0 4px 14px rgba(11,11,11,0.12)' : '0 4px 14px rgba(0,0,0,0.5)',
        padding: '8px 10px',
        border: `1px solid ${mode === 'light' ? 'rgba(11,11,11,0.10)' : 'rgba(255,255,255,0.10)'}`,
      },
    },
    annotations: {
      text: { fill: c.textPrimary, fontSize: 11 },
      link: { stroke: c.baseline, strokeWidth: 1 },
      outline: { stroke: c.baseline, strokeWidth: 1 },
addOutline: { stroke: c.baseline },
    },
    crosshair: { line: { stroke: c.textMuted, strokeWidth: 1, strokeDasharray: '4 4' } },
  };
}

/** Palette metadata surfaced in the UI so readers can see the encoding rules. */
export const VISUAL_LANGUAGE = [
  { role: 'Value function', encoding: 'Sequential blue ramp', token: '--seq-*' },
  { role: 'Policy', encoding: 'Directional arrows, slot-1 hue', token: '--series-1' },
  { role: 'Reward / TD error / advantage', encoding: 'Diverging blue↔red, gray midpoint', token: '--div-*' },
  { role: 'Uncertainty', encoding: 'Fan or band, series hue at low alpha', token: '--series-N' },
] as const;
