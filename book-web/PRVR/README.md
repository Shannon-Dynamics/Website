# Probabilistic Robotics via Rust — the web book

The interactive book itself: a statically-exported Next.js site where every chapter carries live
simulations, and the algorithms in those simulations are the ones the chapter teaches.

## Running it

Requires **Node ≥ 20.9** (Next.js 16). If your system Node is older, a local install works fine:

```sh
curl -sSL https://nodejs.org/dist/v24.19.0/node-v24.19.0-linux-x64.tar.xz \
  | tar -xJ -C ~/.local/node --strip-components=1
export PATH="$HOME/.local/node/bin:$PATH"
```

Then:

```sh
npm install
npm run dev        # development server — slow by design, see below
npm run build      # typecheck + static export to out/
npm run preview    # serve the built book with compression (this is what to read)
npm run check      # numerical self-checks of the algorithm library
npm run check:book # chapters, widget ids, cross-links, citations, internal links
npm run typecheck
```

**Read the book through `npm run preview`, not `npm run dev`.** The chapters are
long and dense with pre-rendered mathematics — a big one is about 3 MB of HTML,
which compresses to roughly 230 kB. `preview` serves the built output with gzip,
the way GitHub Pages or any CDN would. A development server ships unminified
React, recompiles on navigation, and sends everything uncompressed; on a modest
machine that is the difference between a book and a slideshow.

If you serve `out/` some other way, make sure compression is on. `python3 -m
http.server` does not compress, and a chapter will arrive 13× larger than it
needs to.

After adding or renaming a chapter file, regenerate the content index with `npx fumadocs-mdx`
(the `postinstall` hook also does this).

## Deployment

Pushing to `main` builds the book and publishes it to GitHub Pages
([`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)); a pull request runs the same
checks and stops short of deploying. The workflow enables Pages with the Actions build type on its
first run, so **Settings → Pages → Source** needs no manual step — but if the repository was ever
set to *Deploy from a branch*, switch it back to *GitHub Actions* or `deploy-pages` will refuse.

A project site is served from `https://<owner>.github.io/<repo>/`, not the domain root, so the
export is built with `basePath` set to that prefix. CI reads it from `actions/configure-pages` and
passes it as `NEXT_PUBLIC_BASE_PATH`; locally the variable is unset and the book mounts at `/`, so
`dev` and `preview` are unaffected. Two consequences worth remembering:

- **Link internally with `next/link`, never a raw `<a href="/…">`.** `basePath` only rewrites what
  Next itself emits, so a raw anchor keeps pointing at the domain root — it works perfectly on
  localhost and 404s in production, which is the worst way for a bug to behave. `npm run check:book`
  fails the build on one. Markdown links inside MDX are safe: fumadocs renders them through
  `next/link`. Fragment links (`#w16.1`) are unaffected either way.
- To reproduce the deployed build locally, set the variable for *both* steps:

  ```sh
  NEXT_PUBLIC_BASE_PATH=/ProbabilisticRoboticRust npm run build
  NEXT_PUBLIC_BASE_PATH=/ProbabilisticRoboticRust npm run preview
  ```

Attaching a custom domain later needs no code change: `configure-pages` reports a root mount and
`next.config.mjs` normalizes it away.

## How it fits together

```
app/                    routes: landing page, /chapters/[[...slug]], /notation
content/chapters/       one MDX file per chapter + meta.json for sidebar order
components/
  book/                 prose components (Overview, Derivation, Algorithm, Exercises, References…)
  sim/                  WidgetFrame, SimCanvas, transport + parameter controls
  viz/                  Nivo charts, dashboard shell, stat tiles
  ch/chNN/              each chapter's interactive widgets
lib/
  prob/ geom/ models/   the algorithm library the simulations run
  filters/ mapping/ sim/
  __checks__.ts         numerical invariants for all of the above
```

**The simulations are not mock-ups.** `lib/` is a faithful TypeScript port of the Rust the book
prints, and `lib/__checks__.ts` pins it with invariants the mathematics guarantees — exp/log
round-trips on SE(2), a hand-computed Kalman update, beam-model normalization, the entropy
behaviour of prediction versus correction. `npm run check` runs them; CI fails if any breaks.

## Design constraints worth knowing before editing

- **The four estimation colors are reserved for data.** Blue is prior, orange is prediction, green
  is measurement, purple is posterior, gray is ground truth — in prose, in equations
  (`\htmlClass{term-prior}{…}`), in figures, and in widget UI. Chrome uses teal, deliberately
  outside those hues. Always reference them as `var(--pr-*)`, never a literal hex, so both themes
  work.
- **Everything stochastic is seeded.** `lib/prob/rng.ts`, never `Math.random()`, and the seed is
  shown in the widget's control bar so a reader can reproduce or re-roll a run.
- **Widgets autoplay.** Interaction is an invitation, not a requirement: a reader who touches
  nothing should still learn the point.
- **KaTeX is pre-rendered** at build time with a frozen global macro table (`lib/katex-macros.ts`),
  and `katex` is pinned via `overrides` — a version skew between the renderer and the stylesheet
  silently breaks every equation on the site.

Writing a chapter: see [AUTHORING.md](./AUTHORING.md). The design specs live in `../Chapter-NN.md`,
and `../TOC.md` is the book's contract.
