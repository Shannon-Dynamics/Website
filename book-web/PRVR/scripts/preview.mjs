#!/usr/bin/env node
/**
 * Serve the built book the way a real host would: with compression.
 *
 * A chapter of this book is about 3 MB of HTML and gzips to roughly 230 kB, so
 * a plain static file server (python -m http.server, `npx serve` without a
 * flag, or opening the files over file://) makes the site feel an order of
 * magnitude heavier than it is in production. GitHub Pages, Vercel and any
 * normal CDN compress automatically; this script exists so local reading
 * matches that.
 */

import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';
import { createGzip } from 'node:zlib';
import { pipeline } from 'node:stream';
import { fileURLToPath } from 'node:url';

// fileURLToPath, not .pathname: this project's path contains spaces, which a
// URL percent-encodes.
const ROOT = fileURLToPath(new URL('../out/', import.meta.url));
const PORT = Number(process.env.PORT ?? 3000);

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};
const COMPRESSIBLE = new Set(['.html', '.js', '.css', '.json', '.svg', '.txt']);

if (!existsSync(ROOT)) {
  console.error('No build found. Run `npm run build` first.');
  process.exit(1);
}

// A build made for GitHub Pages is mounted under `/<repo>/`, so its asset URLs
// carry that prefix. Set the same variable here as at build time and the
// preview serves it from the same place the deployment will. Unset — the usual
// case — the book is served from the root, exactly as `npm run dev` does.
const BASE = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/+$/, '');

// The `.html` fallback below is not a convenience: it is what GitHub Pages
// itself does, and it is why the export needs no trailing slashes.
function resolve(urlPath) {
  let clean = normalize(decodeURIComponent(urlPath.split('?')[0])).replace(/^(\.\.[/\\])+/, '');
  if (BASE && (clean === BASE || clean.startsWith(`${BASE}/`))) {
    clean = clean.slice(BASE.length) || '/';
  }
  const candidates = [
    join(ROOT, clean),
    join(ROOT, clean, 'index.html'),
    join(ROOT, `${clean}.html`),
  ];
  return candidates.find((p) => existsSync(p) && statSync(p).isFile());
}

createServer((req, res) => {
  const file = resolve(req.url ?? '/') ?? join(ROOT, '404.html');
  if (!existsSync(file)) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('Not found');
    return;
  }

  const ext = extname(file);
  const headers = {
    'content-type': TYPES[ext] ?? 'application/octet-stream',
    // Hashed assets are immutable; pages should revalidate.
    'cache-control': file.includes('/_next/static/')
      ? 'public, max-age=31536000, immutable'
      : 'no-cache',
  };

  const wantsGzip = /\bgzip\b/.test(req.headers['accept-encoding'] ?? '');
  if (wantsGzip && COMPRESSIBLE.has(ext)) {
    headers['content-encoding'] = 'gzip';
    headers.vary = 'Accept-Encoding';
    res.writeHead(200, headers);
    pipeline(createReadStream(file), createGzip({ level: 6 }), res, () => {});
    return;
  }

  headers['content-length'] = statSync(file).size;
  res.writeHead(200, headers);
  createReadStream(file).pipe(res);
}).listen(PORT, () => {
  console.log(`Book served with compression at http://localhost:${PORT}${BASE}`);
});
