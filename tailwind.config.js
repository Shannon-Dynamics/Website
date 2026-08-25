/**
 * The configuration the pages used to declare inline, next to the CDN script
 * that compiled Tailwind in the visitor's browser on every page load.
 *
 * @type {import('tailwindcss').Config}
 */
module.exports = {
  // Every hand-written page. The book landing pages and the Library pages use
  // the same utilities as the homepage, so one stylesheet serves all six.
  content: ['./*.html'],
  theme: {
    extend: {
      colors: {
        slatebg: '#1b2529',
        graphite: '#10181b',
        accent: '#4C7EFF',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  // What `?plugins=forms,container-queries` on the CDN URL used to load.
  plugins: [require('@tailwindcss/forms'), require('@tailwindcss/container-queries')],
};
