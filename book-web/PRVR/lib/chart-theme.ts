/**
 * The bridge between the book's CSS design tokens and Nivo.
 *
 * Nivo needs plain color strings, not CSS variables: it hands most of them to
 * d3-color, and anything d3-color cannot parse (notably `oklch()`, which is what
 * Tailwind v4's default palette compiles to) silently becomes black. So every
 * color a chart uses is *resolved* here — read out of the live stylesheet with
 * `getComputedStyle` — and re-read whenever the theme class on <html> flips.
 *
 * Static export has no DOM, so the module ships a light-theme snapshot of the
 * same tokens. Charts render real SVG at build time with those values and swap
 * to the resolved ones after hydration.
 */

import { useEffect, useMemo, useState, useSyncExternalStore } from 'react';
import type { PartialTheme } from '@nivo/theming';

/* -------------------------------------------------------------------------- */
/* The book color code                                                        */
/* -------------------------------------------------------------------------- */

/** The five meanings the book's data colors are reserved for. */
export type BookRole = 'prior' | 'prediction' | 'measurement' | 'posterior' | 'truth';

/**
 * Fixed assignment order for series that do not declare a role. Never cycled
 * past its length in a way that invents meaning — a figure needing more than
 * five distinct series wants small multiples, not a sixth hue.
 */
export const BOOK_ROLE_ORDER: readonly BookRole[] = [
  'prior',
  'prediction',
  'measurement',
  'posterior',
  'truth',
];

export type BookColors = Record<BookRole, string>;

/** CSS variables usable directly in HTML/SVG style props (no JS resolution). */
export const ROLE_VAR: Record<BookRole, string> = {
  prior: 'var(--pr-prior)',
  prediction: 'var(--pr-prediction)',
  measurement: 'var(--pr-measurement)',
  posterior: 'var(--pr-posterior)',
  truth: 'var(--pr-truth)',
};

/* -------------------------------------------------------------------------- */
/* Token store                                                                */
/* -------------------------------------------------------------------------- */

const TOKEN_PROPERTY = {
  prior: '--pr-prior',
  prediction: '--pr-prediction',
  measurement: '--pr-measurement',
  posterior: '--pr-posterior',
  truth: '--pr-truth',
  grid: '--pr-grid',
  canvasBg: '--pr-canvas-bg',
  canvasInk: '--pr-canvas-ink',
  background: '--color-fd-background',
  surface: '--color-fd-card',
  popover: '--color-fd-popover',
  foreground: '--color-fd-foreground',
  mutedForeground: '--color-fd-muted-foreground',
  border: '--color-fd-border',
  accent: '--color-fd-primary',
  fontUi: '--font-ui',
  fontMono: '--font-mono',
} as const;

export type ChartTokens = Record<keyof typeof TOKEN_PROPERTY, string>;

/**
 * Light-theme values, mirroring app/global.css. Used for static export and for
 * the hydration pass, so server and client agree on the first paint.
 */
const FALLBACK_TOKENS: ChartTokens = {
  prior: '#3b82f6',
  prediction: '#ea580c',
  measurement: '#16a34a',
  posterior: '#9333ea',
  truth: '#6b7280',
  grid: 'rgb(15 23 42 / 0.07)',
  canvasBg: '#fcfcfd',
  canvasInk: '#0f172a',
  background: 'hsl(0, 0%, 96%)',
  surface: 'hsl(0, 0%, 94.7%)',
  popover: 'hsl(0, 0%, 98%)',
  foreground: 'hsl(0, 0%, 3.9%)',
  mutedForeground: 'hsl(0, 0%, 45.1%)',
  border: 'hsla(0, 0%, 80%, 50%)',
  accent: '#0d9488',
  fontUi: "'IBM Plex Sans', system-ui, sans-serif",
  fontMono: "'IBM Plex Mono', ui-monospace, monospace",
};

const TOKEN_NAMES = Object.keys(TOKEN_PROPERTY) as (keyof ChartTokens)[];

function readTokens(): ChartTokens {
  if (typeof document === 'undefined') return FALLBACK_TOKENS;
  const computed = getComputedStyle(document.documentElement);
  const next = { ...FALLBACK_TOKENS };
  for (const name of TOKEN_NAMES) {
    const value = computed.getPropertyValue(TOKEN_PROPERTY[name]).trim();
    if (value) next[name] = value;
  }
  return next;
}

function sameTokens(a: ChartTokens, b: ChartTokens): boolean {
  return TOKEN_NAMES.every((name) => a[name] === b[name]);
}

/*
 * One module-level store rather than one observer per chart: a dashboard can
 * hold a dozen charts, and they all read the same four dozen bytes of theme.
 */
let snapshot: ChartTokens = FALLBACK_TOKENS;
let observer: MutationObserver | null = null;
const listeners = new Set<() => void>();

function refresh(): void {
  const next = readTokens();
  if (sameTokens(snapshot, next)) return;
  snapshot = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (!observer) {
    // fumadocs toggles `.dark` on <html>; `style` catches inline overrides.
    observer = new MutationObserver(refresh);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'style'],
    });
    refresh();
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      observer?.disconnect();
      observer = null;
    }
  };
}

const getSnapshot = (): ChartTokens => snapshot;
const getServerSnapshot = (): ChartTokens => FALLBACK_TOKENS;

/** Every design token the charting layer needs, resolved to real color strings. */
export function useChartTokens(): ChartTokens {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

/** The five reserved data colors, resolved for the theme currently on screen. */
export function useBookColors(): BookColors {
  const tokens = useChartTokens();
  return useMemo(
    () => ({
      prior: tokens.prior,
      prediction: tokens.prediction,
      measurement: tokens.measurement,
      posterior: tokens.posterior,
      truth: tokens.truth,
    }),
    [tokens],
  );
}

/* -------------------------------------------------------------------------- */
/* Nivo theme                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * A Nivo theme built from the live tokens: recessive hairline grid, no tick
 * marks, tabular numerals on every axis, eyebrow-styled axis legends, and a
 * transparent background so a chart takes the color of whatever panel holds it.
 */
export function useChartTheme(): PartialTheme {
  const tokens = useChartTokens();

  return useMemo<PartialTheme>(
    () => ({
      background: 'transparent',
      text: {
        fontFamily: tokens.fontUi,
        fontSize: 11,
        fill: tokens.mutedForeground,
      },
      axis: {
        domain: {
          line: { stroke: tokens.border, strokeWidth: 1 },
        },
        ticks: {
          // Gridlines carry the reading; tick marks would just add ink.
          line: { stroke: 'transparent', strokeWidth: 0 },
          text: {
            fontFamily: tokens.fontMono,
            fontSize: 10,
            fill: tokens.mutedForeground,
            fontVariantNumeric: 'tabular-nums',
          },
        },
        legend: {
          text: {
            fontFamily: tokens.fontUi,
            fontSize: 10,
            fontWeight: 600,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            fill: tokens.mutedForeground,
          },
        },
      },
      grid: {
        line: { stroke: tokens.grid, strokeWidth: 1 },
      },
      crosshair: {
        line: {
          stroke: tokens.foreground,
          strokeWidth: 1,
          strokeOpacity: 0.3,
          strokeDasharray: 'none',
        },
      },
      labels: {
        text: {
          fontFamily: tokens.fontMono,
          fontSize: 10,
          fill: tokens.foreground,
          fontVariantNumeric: 'tabular-nums',
        },
      },
      markers: {
        lineColor: tokens.mutedForeground,
        lineStrokeWidth: 1,
      },
      // Tooltip chrome lives in TooltipCard (real Tailwind tokens, real theme
      // switching); the Nivo container is flattened out of the way.
      tooltip: {
        container: {
          background: 'transparent',
          boxShadow: 'none',
          border: 'none',
          borderRadius: 0,
          padding: 0,
          color: 'inherit',
          fontFamily: tokens.fontUi,
        },
      },
    }),
    [tokens],
  );
}

/** True when the reader asked the OS for less motion. Charts then skip springs. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribeToMotionPreference, getMotionPreference, () => false);
}

function subscribeToMotionPreference(listener: () => void): () => void {
  const query = window.matchMedia('(prefers-reduced-motion: reduce)');
  query.addEventListener('change', listener);
  return () => query.removeEventListener('change', listener);
}

function getMotionPreference(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** False on the server and through hydration, true from the first effect on. */
export function useAfterMount(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

/**
 * Whether a chart should animate.
 *
 * Springs are off for the very first frame on purpose. react-spring renders a
 * spring at its `from` value, and for bars that is height zero — an animated
 * first paint exports a plot with no data in it. Charts therefore draw
 * themselves finished, and animate only once the reader can change something.
 */
export function useChartAnimation(): boolean {
  const mounted = useAfterMount();
  const reducedMotion = usePrefersReducedMotion();
  return mounted && !reducedMotion;
}

/* -------------------------------------------------------------------------- */
/* Color math                                                                 */
/* -------------------------------------------------------------------------- */

export interface Rgb {
  r: number;
  g: number;
  b: number;
  a: number;
}

const HEX_PATTERN = /^#([0-9a-f]{3,8})$/i;
const FUNCTIONAL_PATTERN = /^(rgba?|hsla?)\(([^)]+)\)$/i;

function scaledChannel(token: string, scale: number): number {
  return token.endsWith('%') ? (parseFloat(token) / 100) * scale : parseFloat(token);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const hue = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;
  const sector = Math.floor(hue / 60) % 6;
  const rgb: [number, number, number] =
    sector === 0
      ? [c, x, 0]
      : sector === 1
        ? [x, c, 0]
        : sector === 2
          ? [0, c, x]
          : sector === 3
            ? [0, x, c]
            : sector === 4
              ? [x, 0, c]
              : [c, 0, x];
  return [(rgb[0] + m) * 255, (rgb[1] + m) * 255, (rgb[2] + m) * 255];
}

/** Parses the color notations the book's stylesheet actually emits. */
export function parseCssColor(input: string): Rgb | null {
  const value = input.trim().toLowerCase();

  const hex = HEX_PATTERN.exec(value);
  if (hex) {
    const digits = hex[1];
    if (digits.length === 3 || digits.length === 4) {
      const at = (i: number) => parseInt(digits[i] + digits[i], 16);
      return {
        r: at(0),
        g: at(1),
        b: at(2),
        a: digits.length === 4 ? at(3) / 255 : 1,
      };
    }
    if (digits.length === 6 || digits.length === 8) {
      const at = (i: number) => parseInt(digits.slice(i, i + 2), 16);
      return {
        r: at(0),
        g: at(2),
        b: at(4),
        a: digits.length === 8 ? at(6) / 255 : 1,
      };
    }
    return null;
  }

  const functional = FUNCTIONAL_PATTERN.exec(value);
  if (!functional) return null;
  const parts = functional[2].split(/[\s,/]+/).filter(Boolean);
  if (parts.length < 3) return null;
  const alpha = parts.length > 3 ? scaledChannel(parts[3], 1) : 1;

  if (functional[1].startsWith('rgb')) {
    return {
      r: scaledChannel(parts[0], 255),
      g: scaledChannel(parts[1], 255),
      b: scaledChannel(parts[2], 255),
      a: alpha,
    };
  }

  const [r, g, b] = hslToRgb(
    parseFloat(parts[0]),
    parseFloat(parts[1]) / 100,
    parseFloat(parts[2]) / 100,
  );
  return { r, g, b, a: alpha };
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const clamp255 = (v: number) => Math.min(255, Math.max(0, Math.round(v)));

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);

interface Oklab {
  L: number;
  a: number;
  b: number;
}

/*
 * Oklab (Ottosson 2020). Ramps are mixed here rather than in sRGB because an
 * sRGB lerp between two hues dips through a muddy, darker middle — which on a
 * covariance heatmap reads as a value that is not in the data.
 */
function rgbToOklab({ r, g, b }: Rgb): Oklab {
  const lr = srgbToLinear(r / 255);
  const lg = srgbToLinear(g / 255);
  const lb = srgbToLinear(b / 255);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return {
    L: 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    a: 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    b: 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  };
}

function oklabToRgb({ L, a, b }: Oklab, alpha: number): Rgb {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return {
    r: clamp255(linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s) * 255),
    g: clamp255(linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s) * 255),
    b: clamp255(linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s) * 255),
    a: alpha,
  };
}

function toCss({ r, g, b, a }: Rgb): string {
  if (a >= 1) {
    const hex = (v: number) => clamp255(v).toString(16).padStart(2, '0');
    return `#${hex(r)}${hex(g)}${hex(b)}`;
  }
  return `rgba(${clamp255(r)}, ${clamp255(g)}, ${clamp255(b)}, ${Number(a.toFixed(3))})`;
}

/** Perceptual mix of two CSS colors; `t = 0` returns `from`, `t = 1` returns `to`. */
export function mixColors(from: string, to: string, t: number): string {
  const a = parseCssColor(from);
  const b = parseCssColor(to);
  if (!a || !b) return t < 0.5 ? from : to;
  const amount = clamp01(t);
  const labA = rgbToOklab(a);
  const labB = rgbToOklab(b);
  return toCss(
    oklabToRgb(
      {
        L: labA.L + (labB.L - labA.L) * amount,
        a: labA.a + (labB.a - labA.a) * amount,
        b: labA.b + (labB.b - labA.b) * amount,
      },
      a.a + (b.a - a.a) * amount,
    ),
  );
}

/** Shifts perceptual lightness; positive lightens, negative darkens. */
export function adjustLightness(color: string, delta: number): string {
  const rgb = parseCssColor(color);
  if (!rgb) return color;
  const lab = rgbToOklab(rgb);
  return toCss(oklabToRgb({ ...lab, L: clamp01(lab.L + delta) }, rgb.a));
}

/** Same hue, new alpha — used for particle clouds and area washes. */
export function withAlpha(color: string, alpha: number): string {
  const rgb = parseCssColor(color);
  if (!rgb) return color;
  return `rgba(${clamp255(rgb.r)}, ${clamp255(rgb.g)}, ${clamp255(rgb.b)}, ${Number(
    clamp01(alpha).toFixed(3),
  )})`;
}

/** WCAG relative luminance, 0 (black) to 1 (white). */
export function relativeLuminance(color: string): number {
  const rgb = parseCssColor(color);
  if (!rgb) return 0;
  return (
    0.2126 * srgbToLinear(rgb.r / 255) +
    0.7152 * srgbToLinear(rgb.g / 255) +
    0.0722 * srgbToLinear(rgb.b / 255)
  );
}

/** Ink that stays legible on `background` — for labels set inside a filled mark. */
export function readableInk(background: string, dark = '#0f172a', light = '#f8fafc'): string {
  return relativeLuminance(background) > 0.4 ? dark : light;
}

/* -------------------------------------------------------------------------- */
/* Formatting                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Compact fixed-point formatting for readouts and tooltips: enough digits to
 * distinguish a probability from zero, few enough to stay scannable.
 */
export function formatNumber(value: number, precision?: number): string {
  if (!Number.isFinite(value)) return '—';
  const magnitude = Math.abs(value);
  if (precision === undefined && magnitude !== 0 && (magnitude >= 1e6 || magnitude < 1e-4)) {
    return value.toExponential(2);
  }
  const digits = precision ?? (magnitude >= 100 ? 1 : magnitude >= 1 ? 2 : 3);
  const fixed = value.toFixed(digits);
  if (!fixed.includes('.')) return fixed;
  return fixed.replace(/0+$/, '').replace(/\.$/, '');
}
