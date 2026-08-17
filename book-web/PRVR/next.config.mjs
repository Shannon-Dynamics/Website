import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/**
 * Where the book is mounted.
 *
 * On GitHub Pages a *project* site lives under `/<repo>/`, not at the root, so
 * every asset URL the export bakes in has to carry that prefix or the reader
 * gets a page of unstyled HTML and no simulations. CI passes the prefix in via
 * `NEXT_PUBLIC_BASE_PATH` (from `actions/configure-pages`, which also returns
 * an empty prefix for a user site or a custom domain). Locally it is unset, so
 * `npm run dev` and `npm run preview` keep serving from `/`.
 *
 * `configure-pages` reports "/" for a root-mounted site; Next rejects that as a
 * basePath, so normalize it — and any trailing slash — away.
 */
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/+$/, '');

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  // The book is a static site: every chapter renders at build time, so readers
  // download HTML + the simulation bundles and nothing else.
  output: 'export',
  basePath,
  // Rewrites `/_next/...` in the emitted HTML. Next derives this from basePath
  // anyway; stating it keeps the two from drifting if one is ever overridden.
  assetPrefix: basePath || undefined,

  // Emit `chapters/<slug>/index.html` rather than `chapters/<slug>.html`.
  // The book is mounted inside the marketing site on GitHub Pages, which has no
  // rewrite rules: a directory with an index resolves there unambiguously,
  // where an extensionless path relies on the host guessing an `.html`
  // extension. The sibling book (RL4RVR) already exports this way.
  trailingSlash: true,

  images: { unoptimized: true },
  typescript: { ignoreBuildErrors: false },
};

export default withMDX(config);
