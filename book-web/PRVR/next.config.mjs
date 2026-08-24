import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/**
 * The book is its own deployment.
 *
 * It used to be exported to static HTML and copied into the marketing site
 * under `/books/<slug>/`, which is why this file once carried a `basePath` and
 * `output: 'export'`. It now ships as a Next.js app on its own origin
 * (prvr.shannon.id), so it is mounted at `/` and has a server behind it: search
 * is served from a route handler, images go through the optimizer, and pages can
 * revalidate instead of being frozen at build time.
 *
 * @type {import('next').NextConfig}
 */
const config = {
  reactStrictMode: true,

  // The URLs the book was indexed under end in a slash — /chapters/ch01/ — and
  // the old paths redirect here path-for-path from the marketing site. Keeping
  // the slash keeps both of those from turning into a redirect hop.
  trailingSlash: true,

  typescript: { ignoreBuildErrors: false },

  experimental: {
    /**
     * Bound the prerender workers.
     *
     * Vercel's basic build machine has 8 GB, and this book's first build there
     * was SIGKILLed by the OOM killer. Left to itself Next sizes the worker
     * pool from the host's core count — seven of them on the machine this was
     * developed on — and each worker holds its own copy of the chapter bundles,
     * which are large: 26 chapters of MDX, KaTeX and simulation components.
     * Two is enough to overlap I/O without the pool being what fills the box.
     */
    cpus: 2,
  },
};

export default withMDX(config);
