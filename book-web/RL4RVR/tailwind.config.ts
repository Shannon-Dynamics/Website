import type { Config } from 'tailwindcss';

/**
 * Design tokens map onto the CSS custom properties declared in app/globals.css,
 * so light/dark values swap in exactly one place (see globals.css).
 * Palette provenance: the validated reference instance from the dataviz method —
 * every categorical slot passes the lightness band, chroma floor, CVD separation,
 * normal-vision floor and contrast checks in BOTH modes.
 */
const config: Config = {
  darkMode: ['class', '[data-theme="dark"]'],
  content: [
    './app/**/*.{ts,tsx,mdx}',
    './components/**/*.{ts,tsx}',
    './content/**/*.mdx',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: 'var(--surface-1)',
          raised: 'var(--surface-2)',
          sunken: 'var(--surface-0)',
          page: 'var(--page-plane)',
        },
        ink: {
          DEFAULT: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          muted: 'var(--text-muted)',
        },
        hairline: 'var(--border-hairline)',
        grid: 'var(--gridline)',
        baseline: 'var(--baseline)',
        series: {
          1: 'var(--series-1)',
          2: 'var(--series-2)',
          3: 'var(--series-3)',
          4: 'var(--series-4)',
          5: 'var(--series-5)',
          6: 'var(--series-6)',
          7: 'var(--series-7)',
          8: 'var(--series-8)',
        },
        status: {
          good: 'var(--status-good)',
          warning: 'var(--status-warning)',
          serious: 'var(--status-serious)',
          critical: 'var(--status-critical)',
        },
        accent: 'var(--accent)',
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      maxWidth: {
        prose: '68ch',
        reading: '46rem',
      },
      typography: () => ({
        book: {
          css: {
            '--tw-prose-body': 'var(--text-primary)',
            '--tw-prose-headings': 'var(--text-primary)',
            '--tw-prose-links': 'var(--accent)',
            '--tw-prose-bold': 'var(--text-primary)',
            '--tw-prose-counters': 'var(--text-muted)',
            '--tw-prose-bullets': 'var(--baseline)',
            '--tw-prose-hr': 'var(--border-hairline)',
            '--tw-prose-quotes': 'var(--text-secondary)',
            '--tw-prose-quote-borders': 'var(--series-1)',
            '--tw-prose-captions': 'var(--text-muted)',
            '--tw-prose-th-borders': 'var(--baseline)',
            '--tw-prose-td-borders': 'var(--border-hairline)',
            maxWidth: 'none',
          },
        },
      }),
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(0.9)', opacity: '0.7' },
          '70%': { transform: 'scale(1.6)', opacity: '0' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.4s ease-out both',
        'pulse-ring': 'pulse-ring 1.6s ease-out infinite',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};

export default config;
