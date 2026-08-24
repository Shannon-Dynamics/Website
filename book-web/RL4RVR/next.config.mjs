/**
 * The book is its own deployment.
 *
 * It used to be exported to static HTML and copied into the marketing site
 * under `/books/<slug>/`, which is why this file once carried a `basePath` and
 * `output: 'export'`. It now ships as a Next.js app on its own origin
 * (rl4rvr.shannon.id): mounted at `/`, with a server behind it, so the chapter
 * pages keep prerendering while images go through the optimizer and route
 * handlers are available if the book ever needs one.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  // The URLs the book was indexed under end in a slash — /chapters/ch01/ — and
  // the old paths redirect here path-for-path from the marketing site. Keeping
  // the slash keeps both of those from turning into a redirect hop. BookHeader
  // normalises it away before matching routes.
  trailingSlash: true,

  reactStrictMode: true,

  // Nivo ships ESM packages that benefit from transpilation in the server bundle.
  transpilePackages: [
    '@nivo/core',
    '@nivo/bar',
    '@nivo/line',
    '@nivo/heatmap',
    '@nivo/radar',
    '@nivo/scatterplot',
  ],

  experimental: {
    optimizePackageImports: ['lucide-react'],
  },
};

export default nextConfig;
